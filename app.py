"""Local web app: natural-language photo search over the CLIP index."""
import argparse
import random
import re
import secrets
import socket
import sqlite3
import subprocess
from datetime import datetime

UUID_RE = re.compile(r"^[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$")


def is_valid_uuid(value):
    """Photos asset uuids are always this exact format; reject anything else before
    it ever reaches a shell command or AppleScript string (defense against injection)."""
    return isinstance(value, str) and bool(UUID_RE.match(value))

import numpy as np
import open_clip
import torch
from flask import Flask, request, send_file, jsonify, redirect

DB_PATH = "photos_index.db"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

ACCESS_TOKEN = None  # set at startup only when --lan is used


def delete_photos(uuids):
    """Move photos to Recently Deleted via PhotoKit (Photos.app's own AppleScript
    `delete` command is broken/errors on this macOS version)."""
    if not uuids:
        return
    result = subprocess.run(
        ["swift", "delete_photos.swift"] + list(uuids),
        capture_output=True, text=True,
    )
    freed_bytes = 0
    for line in result.stdout.splitlines():
        if line.startswith("FREED_BYTES:"):
            try:
                freed_bytes = int(line.split(":", 1)[1])
            except ValueError:
                pass

    conn = get_db()
    conn.execute(
        "INSERT INTO stats (key, value) VALUES ('deleted_count', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = value + excluded.value",
        (len(uuids),),
    )
    conn.execute(
        "INSERT INTO stats (key, value) VALUES ('freed_bytes', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = value + excluded.value",
        (freed_bytes,),
    )
    conn.commit()
    conn.close()

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)


@app.before_request
def require_token_on_lan():
    if ACCESS_TOKEN is None:
        return  # localhost-only mode: no auth needed
    supplied = request.args.get("token", "")
    if secrets.compare_digest(supplied, ACCESS_TOKEN):
        from flask import session
        session["authed"] = True
        # strip the token from the visible URL after establishing the session cookie
        if request.method == "GET" and request.path != "/thumb":
            clean_url = request.path
            other_args = {k: v for k, v in request.args.items() if k != "token"}
            if other_args:
                from urllib.parse import urlencode
                clean_url += "?" + urlencode(other_args)
            return redirect(clean_url)
        return
    from flask import session
    if session.get("authed"):
        return
    return "Access token required. Use the full URL printed in the terminal (includes ?token=...).", 403


device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Loading CLIP model...")
model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
model = model.to(device).eval()
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

_cache = None  # (uuids, paths, dates, lats, lons, embeddings ndarray, persons_list)
known_persons = set()


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS reviews (uuid TEXT PRIMARY KEY, decision TEXT, reviewed_at TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS stats (key TEXT PRIMARY KEY, value INTEGER)")
    return conn


def load_index():
    global _cache, known_persons
    conn = get_db()
    cols = {row[1] for row in conn.execute("PRAGMA table_info(photos)")}
    has_persons = "persons" in cols
    select_cols = "uuid, path, date, latitude, longitude, embedding" + (", persons" if has_persons else "")
    rows = conn.execute(f"SELECT {select_cols} FROM photos").fetchall()
    conn.close()
    uuids, paths, dates, lats, lons, embs, persons_list = [], [], [], [], [], [], []
    known = set()
    for row in rows:
        uuid, path, date, lat, lon, emb_blob = row[:6]
        persons = row[6] if has_persons and len(row) > 6 else ""
        uuids.append(uuid)
        paths.append(path)
        dates.append(date)
        lats.append(lat)
        lons.append(lon)
        embs.append(np.frombuffer(emb_blob, dtype=np.float32))
        names = [n for n in (persons or "").split(",") if n]
        persons_list.append(names)
        known.update(n.lower() for n in names)
    embeddings = np.vstack(embs) if embs else np.zeros((0, 512), dtype=np.float32)
    known_persons = known
    _cache = (uuids, paths, dates, lats, lons, embeddings, persons_list)
    return _cache


@app.route("/")
def index():
    return HTML_PAGE


@app.route("/stats")
def stats():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    rows = dict(conn.execute("SELECT key, value FROM stats"))
    conn.close()
    return jsonify({
        "total": total,
        "deleted_count": rows.get("deleted_count", 0),
        "freed_bytes": rows.get("freed_bytes", 0),
    })


@app.route("/manifest.json")
def manifest():
    return jsonify({
        "name": "Photo Search",
        "short_name": "Photos",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#0c0d10",
        "theme_color": "#0c0d10",
        "icons": [
            {"src": "/static/icon-192.png", "sizes": "192x192", "type": "image/png"},
            {"src": "/static/icon-512.png", "sizes": "512x512", "type": "image/png"},
        ],
    })


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    date_from = request.args.get("from") or None
    date_to = request.args.get("to") or None
    top_k = int(request.args.get("k", 60))

    if _cache is None:
        load_index()
    uuids, paths, dates, lats, lons, embeddings, persons_list = _cache

    if embeddings.shape[0] == 0:
        return jsonify([])

    mask = np.ones(len(uuids), dtype=bool)
    if date_from:
        mask &= np.array([d is not None and d >= date_from for d in dates])
    if date_to:
        mask &= np.array([d is not None and d <= date_to for d in dates])

    # Check the query for a known person's name (case-insensitive substring, whole
    # word/phrase); if found, filter to that person and search the remaining text.
    remaining_query = query
    matched_name = None
    lower_query = query.lower()
    for name in sorted(known_persons, key=len, reverse=True):
        if name and f" {name} " in f" {lower_query} ":
            matched_name = name
            break
    if matched_name:
        person_mask = np.array([matched_name in [p.lower() for p in persons] for persons in persons_list])
        mask &= person_mask
        # strip the matched name (case-insensitively) out of the remaining query text
        idx_pos = lower_query.find(matched_name)
        remaining_query = (query[:idx_pos] + query[idx_pos + len(matched_name):]).strip()

    if remaining_query:
        with torch.no_grad():
            text = tokenizer([remaining_query]).to(device)
            text_emb = model.encode_text(text)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            text_emb = text_emb.cpu().numpy()[0]
        sims = embeddings @ text_emb
    else:
        sims = np.array([datetime.fromisoformat(d).timestamp() if d else 0 for d in dates])

    sims = np.where(mask, sims, -1e9)
    order = np.argsort(-sims)[:top_k]

    results = [
        {"uuid": uuids[i], "score": float(sims[i]), "date": dates[i], "lat": lats[i], "lon": lons[i]}
        for i in order if sims[i] > -1e8
    ]
    return jsonify(results)


@app.route("/thumb/<uuid>")
def thumb(uuid):
    if not is_valid_uuid(uuid):
        return "invalid uuid", 400
    if _cache is None:
        load_index()
    uuids, paths, *_ = _cache
    try:
        path = paths[uuids.index(uuid)]
    except ValueError:
        return "not found", 404
    return send_file(path)


@app.route("/reveal/<uuid>", methods=["POST"])
def reveal(uuid):
    if not is_valid_uuid(uuid):
        return jsonify({"error": "invalid uuid"}), 400
    script = f'''
    tell application "Photos"
        set targetItems to (media items whose id contains "{uuid}")
        if (count of targetItems) > 0 then spotlight item 1 of targetItems
    end tell
    '''
    subprocess.run(["osascript", "-e", script])
    subprocess.run(["open", "-a", "Photos"])
    return jsonify({"ok": True})


def reviewed_uuids(conn):
    return {row[0] for row in conn.execute("SELECT uuid FROM reviews")}


@app.route("/queue/duplicates")
def queue_duplicates():
    conn = get_db()
    reviewed = reviewed_uuids(conn)
    clusters = {}
    for uuid, cluster_id in conn.execute("SELECT uuid, cluster_id FROM clusters"):
        if uuid in reviewed:
            continue
        clusters.setdefault(cluster_id, []).append(uuid)
    conn.close()
    clusters = {cid: members for cid, members in clusters.items() if len(members) >= 2}
    if not clusters:
        return jsonify({"cluster_id": None, "uuids": []})
    cid = max(clusters, key=lambda c: len(clusters[c]))
    return jsonify({"cluster_id": cid, "uuids": clusters[cid]})


@app.route("/queue/shuffle")
def queue_shuffle():
    n = int(request.args.get("n", 20))
    if _cache is None:
        load_index()
    uuids = _cache[0]
    conn = get_db()
    reviewed = reviewed_uuids(conn)
    in_cluster = {row[0] for row in conn.execute("SELECT uuid FROM clusters")}
    conn.close()
    candidates = [u for u in uuids if u not in reviewed and u not in in_cluster]
    random.shuffle(candidates)
    return jsonify({"uuids": candidates[:n]})


def queue_by_tag(tag, n=20):
    conn = get_db()
    conn.execute("CREATE TABLE IF NOT EXISTS tags (uuid TEXT, tag TEXT, PRIMARY KEY (uuid, tag))")
    reviewed = reviewed_uuids(conn)
    candidates = [
        row[0] for row in conn.execute("SELECT uuid FROM tags WHERE tag = ?", (tag,))
        if row[0] not in reviewed
    ]
    conn.close()
    random.shuffle(candidates)
    return jsonify({"uuids": candidates[:n]})


@app.route("/queue/screenshots")
def queue_screenshots():
    return queue_by_tag("screenshot", int(request.args.get("n", 20)))


@app.route("/queue/blurry")
def queue_blurry():
    return queue_by_tag("blurry", int(request.args.get("n", 20)))


@app.route("/review", methods=["POST"])
def review():
    data = request.get_json()
    uuid = data["uuid"]
    decision = data["decision"]
    if not is_valid_uuid(uuid):
        return jsonify({"error": "invalid uuid"}), 400
    if decision not in ("keep", "delete"):
        return jsonify({"error": "invalid decision"}), 400

    conn = get_db()
    conn.execute(
        "INSERT OR REPLACE INTO reviews (uuid, decision, reviewed_at) VALUES (?, ?, ?)",
        (uuid, decision, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()

    if decision == "delete":
        delete_photos([uuid])

    return jsonify({"ok": True})


@app.route("/review-batch", methods=["POST"])
def review_batch():
    data = request.get_json()
    deletes = [u for u in data.get("deletes", []) if is_valid_uuid(u)]
    keeps = [u for u in data.get("keeps", []) if is_valid_uuid(u)]

    conn = get_db()
    now = datetime.now().isoformat()
    conn.executemany(
        "INSERT OR REPLACE INTO reviews (uuid, decision, reviewed_at) VALUES (?, 'delete', ?)",
        [(u, now) for u in deletes],
    )
    conn.executemany(
        "INSERT OR REPLACE INTO reviews (uuid, decision, reviewed_at) VALUES (?, 'keep', ?)",
        [(u, now) for u in keeps],
    )
    conn.commit()
    conn.close()

    delete_photos(deletes)

    return jsonify({"ok": True})


@app.route("/review-ui")
def review_ui():
    return REVIEW_HTML_PAGE


SHARED_STYLE = """
  :root {
    --bg: #0c0d10; --panel: #17181c; --panel-2: #1f2126; --border: #2a2c33;
    --text: #f0f1f3; --muted: #9297a3; --accent: #5b8cff; --accent-2: #7aa2ff;
    --keep: #34c77b; --keep-dim: #1f7a4d; --delete: #ff5c6c; --delete-dim: #99333d;
    --radius: 14px; --radius-sm: 9px;
    --shadow: 0 8px 24px rgba(0,0,0,0.35);
  }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif;
    margin: 0; background: var(--bg); color: var(--text); -webkit-font-smoothing: antialiased;
  }
  a { text-decoration: none; }
  button {
    font-family: inherit; cursor: pointer; border: none; border-radius: var(--radius-sm);
    transition: transform 0.08s ease, opacity 0.15s ease, background 0.15s ease;
  }
  button:active { transform: scale(0.97); }
  button:disabled { opacity: 0.5; cursor: default; transform: none; }
  .btn-primary { background: var(--accent); color: white; }
  .btn-primary:hover:not(:disabled) { background: var(--accent-2); }
  .btn-ghost { background: var(--panel-2); color: var(--text); border: 1px solid var(--border); }
  .btn-ghost:hover { background: var(--panel); }

  .spinner {
    width: 18px; height: 18px; border-radius: 50%;
    border: 2.5px solid rgba(255,255,255,0.25); border-top-color: currentColor;
    animation: spin 0.7s linear infinite; display: inline-block; vertical-align: middle;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .fade-in { animation: fadeIn 0.25s ease; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
"""

PWA_HEAD = """
<link rel="manifest" href="/manifest.json">
<link rel="apple-touch-icon" href="/static/icon-192.png">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Photos">
<meta name="theme-color" content="#0c0d10">
"""

HTML_PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Photo Search</title>
""" + PWA_HEAD + """
<style>
""" + SHARED_STYLE + """
  body { padding: 28px 20px 40px; }
  h1 { font-size: 22px; font-weight: 650; margin: 0 0 6px; letter-spacing: -0.01em; }
  h1 .count { font-size: 13px; font-weight: 500; color: var(--muted); margin-left: 8px; }
  #controls {
    display: flex; gap: 10px; margin-bottom: 24px; flex-wrap: wrap; align-items: center;
    background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 14px; box-shadow: var(--shadow);
  }
  input[type=text] {
    flex: 1; min-width: 240px; padding: 12px 14px; font-size: 15px; border-radius: var(--radius-sm);
    border: 1px solid var(--border); background: var(--panel-2); color: var(--text);
  }
  input[type=text]:focus { outline: none; border-color: var(--accent); }
  input[type=date] {
    padding: 10px 12px; border-radius: var(--radius-sm); border: 1px solid var(--border);
    background: var(--panel-2); color: var(--text); color-scheme: dark;
  }
  #controls button { padding: 11px 18px; font-size: 14px; font-weight: 550; }
  #status { color: var(--muted); font-size: 13px; margin: 0 0 14px; min-height: 18px; }
  #grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 10px; }
  .item {
    position: relative; cursor: pointer; border-radius: var(--radius-sm); overflow: hidden;
    border: 1px solid var(--border); background: var(--panel);
  }
  .item img { width: 100%; height: 160px; object-fit: cover; display: block; transition: transform 0.2s ease; }
  .item:hover img { transform: scale(1.04); }
  .item .meta {
    font-size: 11px; color: #eee; padding: 5px 7px; background: rgba(0,0,0,0.55);
    position: absolute; bottom: 0; left: 0; right: 0;
  }
</style>
</head>
<body>
  <h1>📷 Photo Search<span class="count" id="photoCount"></span></h1>
  <div id="controls">
    <input type="text" id="q" placeholder="e.g. sunset over the ocean, birthday cake, dog on the beach">
    <input type="date" id="from">
    <input type="date" id="to">
    <button class="btn-primary" onclick="doSearch()">Search</button>
    <a href="/review-ui?queue=duplicates"><button type="button" class="btn-ghost">🧹 Review Duplicates</button></a>
    <a href="/review-ui?queue=shuffle"><button type="button" class="btn-ghost">🎲 Quick Shuffle (20)</button></a>
    <a href="/review-ui?queue=screenshots"><button type="button" class="btn-ghost">🖼️ Screenshots</button></a>
    <a href="/review-ui?queue=blurry"><button type="button" class="btn-ghost">🌫️ Blurry</button></a>
  </div>
  <div id="status"></div>
  <div id="grid"></div>

<script>
async function doSearch() {
  const status = document.getElementById('status');
  const grid = document.getElementById('grid');
  status.innerHTML = '<span class="spinner"></span> Searching...';
  const q = document.getElementById('q').value;
  const from = document.getElementById('from').value;
  const to = document.getElementById('to').value;
  const params = new URLSearchParams({q, from, to});
  const res = await fetch('/search?' + params.toString());
  const data = await res.json();
  grid.innerHTML = '';
  status.textContent = data.length ? `${data.length} photos` : 'No matches found';
  for (const item of data) {
    const div = document.createElement('div');
    div.className = 'item fade-in';
    const dateStr = item.date ? item.date.split('T')[0] : '';
    div.innerHTML = `<img src="/thumb/${item.uuid}" loading="lazy"><div class="meta">${dateStr}</div>`;
    div.onclick = () => fetch('/reveal/' + item.uuid, {method: 'POST'});
    grid.appendChild(div);
  }
}
document.getElementById('q').addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(); });
async function loadStats() {
  const res = await fetch('/stats');
  const data = await res.json();
  let text = `· ${data.total.toLocaleString()} photos indexed`;
  if (data.deleted_count > 0) {
    const gb = (data.freed_bytes / 1e9).toFixed(1);
    text += ` · 🗑️ ${data.deleted_count.toLocaleString()} deleted · ~${gb} GB freed`;
  }
  document.getElementById('photoCount').textContent = text;
}
loadStats();
doSearch();
</script>
</body>
</html>
"""

REVIEW_HTML_PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review Photos</title>
""" + PWA_HEAD + """
<style>
""" + SHARED_STYLE + """
  body { height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
  #topbar {
    padding: 14px 16px; text-align: center; font-size: 14px; color: var(--muted);
    border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: center; gap: 8px;
  }
  #topbar a { color: var(--accent-2); }
  #stage { flex: 1; display: flex; align-items: center; justify-content: center; position: relative; overflow: hidden; }
  #stage img { max-width: 92%; max-height: 92%; border-radius: var(--radius); object-fit: contain;
               box-shadow: var(--shadow); touch-action: pan-y; will-change: transform; user-select: none; }
  #buttons { display: flex; gap: 12px; padding: 18px; justify-content: center; align-items: center; }
  button.action {
    flex: 1; max-width: 150px; padding: 16px; font-size: 16px; font-weight: 600; color: white;
    display: flex; align-items: center; justify-content: center; gap: 8px; box-shadow: var(--shadow);
  }
  button.action-sm {
    padding: 14px; font-size: 20px; width: 52px; height: 52px; flex: none; border-radius: 50%;
    background: var(--panel-2); border: 1px solid var(--border); display: flex; align-items: center; justify-content: center;
  }
  button.action-sm:hover:not(:disabled) { background: var(--panel); }
  #keep { background: var(--keep); }
  #keep:hover:not(:disabled) { background: #3fdb8a; }
  #del { background: var(--delete); }
  #del:hover:not(:disabled) { background: #ff7080; }
  #empty { text-align: center; padding: 60px 20px; color: var(--muted); font-size: 15px; }

  #summary { text-align: center; padding: 40px 24px; }
  #summary .big { font-size: 40px; margin-bottom: 8px; }
  #summary .counts { font-size: 16px; color: var(--muted); margin-bottom: 28px; }
  #summary .counts b { color: var(--text); }
  #summary button { padding: 15px 32px; font-size: 15px; font-weight: 600; margin: 6px; }
  .drop-hint {
    position: absolute; top: 20px; font-size: 22px; font-weight: 700; padding: 8px 18px; border-radius: 20px;
    opacity: 0; transition: opacity 0.1s ease; pointer-events: none;
  }
  #hintDelete { left: 20px; background: var(--delete); color: white; }
  #hintKeep { right: 20px; background: var(--keep); color: white; }

  #clusterGrid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px;
    padding: 16px; overflow-y: auto; width: 100%; box-sizing: border-box; flex: 1;
  }
  .cshot {
    position: relative; border-radius: var(--radius); overflow: hidden; border: 3px solid transparent;
    cursor: pointer; background: var(--panel); transition: border-color 0.15s ease, transform 0.1s ease;
    box-shadow: var(--shadow);
  }
  .cshot:active { transform: scale(0.98); }
  .cshot img { width: 100%; height: 340px; object-fit: contain; background: #000; display: block; }
  .cshot .tag {
    position: absolute; top: 8px; left: 8px; padding: 5px 10px; border-radius: 20px; font-size: 12px;
    font-weight: 600; color: white; display: flex; align-items: center; gap: 4px;
  }
  .cshot.mark-delete { border-color: var(--delete); }
  .cshot.mark-delete .tag { background: var(--delete); }
  .cshot.mark-keep { border-color: var(--keep); }
  .cshot.mark-keep .tag { background: var(--keep); }
  .cshot .zoom {
    position: absolute; bottom: 8px; right: 8px; background: rgba(0,0,0,0.65); color: white;
    border-radius: 20px; padding: 5px 10px; font-size: 12px;
  }

  #confirmBar { padding: 16px; text-align: center; border-top: 1px solid var(--border); }
  #confirmBar button {
    padding: 15px 32px; font-size: 15px; font-weight: 600; display: inline-flex; align-items: center; gap: 8px;
  }

  #zoomOverlay {
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.94); z-index: 10;
    align-items: center; justify-content: center;
  }
  #zoomOverlay img { max-width: 96%; max-height: 96%; object-fit: contain; }
  #zoomOverlay.open { display: flex; }

  #loadingOverlay {
    display: none; position: fixed; inset: 0; background: rgba(12,13,16,0.75); z-index: 20;
    align-items: center; justify-content: center; flex-direction: column; gap: 12px; color: var(--muted);
    font-size: 14px; backdrop-filter: blur(2px);
  }
  #loadingOverlay.open { display: flex; }
  #loadingOverlay .spinner { width: 28px; height: 28px; border-width: 3px; }
</style>
</head>
<body>
  <div id="topbar"><a href="/">&larr; Search</a></div>
  <div id="stage">
    <span class="drop-hint" id="hintDelete">🗑️ Delete</span>
    <span class="drop-hint" id="hintKeep">✅ Keep</span>
    <img id="photo" src="">
  </div>
  <div id="buttons">
    <button class="action-sm" id="undo" onclick="undo()" title="Undo" disabled>↩️</button>
    <button class="action" id="del" onclick="decide('delete')">🗑️ Delete</button>
    <button class="action" id="keep" onclick="decide('keep')">✅ Keep</button>
    <button class="action-sm" id="share" onclick="sharePhoto()" title="Share">📤</button>
  </div>
  <div id="loadingOverlay"><span class="spinner"></span><span id="loadingText">Loading...</span></div>

<script>
const params = new URLSearchParams(location.search);
const queueType = params.get('queue') || 'shuffle';
const QUEUE_ENDPOINTS = {
  shuffle: '/queue/shuffle?n=20',
  screenshots: '/queue/screenshots?n=20',
  blurry: '/queue/blurry?n=20',
};
let items = [];
let idx = 0;
let marks = {};
let history = [];  // stack of {uuid, decision} for undo

function showLoading(text) {
  document.getElementById('loadingText').textContent = text || 'Loading...';
  document.getElementById('loadingOverlay').classList.add('open');
}
function hideLoading() {
  document.getElementById('loadingOverlay').classList.remove('open');
}

async function loadQueue() {
  showLoading('Loading photos...');
  if (queueType === 'duplicates') {
    await loadClusterView();
    hideLoading();
    return;
  }
  const endpoint = QUEUE_ENDPOINTS[queueType] || QUEUE_ENDPOINTS.shuffle;
  const res = await fetch(endpoint);
  const data = await res.json();
  items = data.uuids || [];
  idx = 0;
  marks = {};
  history = [];
  render();
  hideLoading();
}

const QUEUE_LABELS = { shuffle: 'Quick Shuffle', screenshots: 'Screenshots', blurry: 'Blurry' };

function render() {
  const topbar = document.getElementById('topbar');
  const img = document.getElementById('photo');
  const buttons = document.getElementById('buttons');
  document.getElementById('undo').disabled = history.length === 0;
  if (idx >= items.length) {
    showSummary();
    return;
  }
  document.getElementById('stage').style.display = 'flex';
  buttons.style.display = 'flex';
  const label = QUEUE_LABELS[queueType] || queueType;
  topbar.innerHTML = `<a href="/">&larr; Search</a>&nbsp;&nbsp;·&nbsp;&nbsp;${idx + 1} of ${items.length} (${label})`;
  img.className = 'fade-in';
  img.style.transform = '';
  img.style.transition = '';
  img.src = '/thumb/' + items[idx];
  if (idx + 1 < items.length) {
    new Image().src = '/thumb/' + items[idx + 1];
  }
}

function decide(decision) {
  if (idx >= items.length) return;
  const uuid = items[idx];
  marks[uuid] = decision;
  history.push(uuid);
  idx++;
  render();
}

function undo() {
  if (history.length === 0) return;
  const uuid = history.pop();
  delete marks[uuid];
  idx--;
  render();
}

async function sharePhoto() {
  if (idx >= items.length) return;
  const uuid = items[idx];
  try {
    const res = await fetch('/thumb/' + uuid);
    const blob = await res.blob();
    const file = new File([blob], 'photo.jpg', { type: blob.type || 'image/jpeg' });
    if (navigator.canShare && navigator.canShare({ files: [file] })) {
      await navigator.share({ files: [file] });
    } else {
      alert('Sharing photos isn\\'t supported in this browser — try opening this page in Safari on your iPhone.');
    }
  } catch (e) {
    if (e.name !== 'AbortError') {
      alert('Could not share this photo.');
    }
  }
}

function showSummary() {
  document.getElementById('stage').style.display = 'none';
  document.getElementById('buttons').style.display = 'none';
  const topbar = document.getElementById('topbar');
  const label = QUEUE_LABELS[queueType] || queueType;
  topbar.innerHTML = `<a href="/">&larr; Search</a>&nbsp;&nbsp;·&nbsp;&nbsp;${label} complete`;
  const deleteCount = Object.values(marks).filter(m => m === 'delete').length;
  const keepCount = Object.values(marks).filter(m => m === 'keep').length;
  let summaryEl = document.getElementById('summary');
  if (!summaryEl) {
    summaryEl = document.createElement('div');
    summaryEl.id = 'summary';
    document.body.insertBefore(summaryEl, document.getElementById('loadingOverlay'));
  }
  if (items.length === 0) {
    summaryEl.innerHTML = `
      <div class="big">🎉</div>
      <div class="counts">No more photos in this queue right now.</div>
      <button class="btn-ghost" onclick="location.href='/'">Back to Search</button>
    `;
    return;
  }
  summaryEl.innerHTML = `
    <div class="big">✅</div>
    <div class="counts">${keepCount} to keep, <b>${deleteCount} to delete</b></div>
    <button class="btn-primary" id="confirmBtn" onclick="confirmShuffle()">Confirm</button>
    <button class="btn-ghost" onclick="location.href='/'">Cancel & go back</button>
  `;
}

async function confirmShuffle() {
  const btn = document.getElementById('confirmBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Saving...';
  showLoading('Saving your choices...');

  const deletes = items.filter(u => marks[u] === 'delete');
  const keeps = items.filter(u => marks[u] === 'keep');
  await fetch('/review-batch', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({deletes, keeps})
  });
  document.getElementById('summary').remove();
  await loadQueue();
  hideLoading();
}

// Touch swipe gestures (shuffle/screenshots/blurry only)
let dragStartX = null;
const photoEl = document.getElementById('photo');
photoEl.addEventListener('touchstart', e => {
  if (queueType === 'duplicates' || idx >= items.length) return;
  dragStartX = e.touches[0].clientX;
  photoEl.style.transition = 'none';
});
photoEl.addEventListener('touchmove', e => {
  if (dragStartX === null) return;
  const dx = e.touches[0].clientX - dragStartX;
  photoEl.style.transform = `translateX(${dx}px) rotate(${dx / 20}deg)`;
  document.getElementById('hintDelete').style.opacity = dx < -30 ? Math.min(1, -dx / 120) : 0;
  document.getElementById('hintKeep').style.opacity = dx > 30 ? Math.min(1, dx / 120) : 0;
});
photoEl.addEventListener('touchend', e => {
  if (dragStartX === null) return;
  const dx = e.changedTouches[0].clientX - dragStartX;
  dragStartX = null;
  document.getElementById('hintDelete').style.opacity = 0;
  document.getElementById('hintKeep').style.opacity = 0;
  photoEl.style.transition = 'transform 0.25s ease';
  const threshold = 100;
  if (dx < -threshold) {
    photoEl.style.transform = `translateX(-600px) rotate(-20deg)`;
    setTimeout(() => decide('delete'), 150);
  } else if (dx > threshold) {
    photoEl.style.transform = `translateX(600px) rotate(20deg)`;
    setTimeout(() => decide('keep'), 150);
  } else {
    photoEl.style.transform = '';
  }
});

async function loadClusterView() {
  document.body.innerHTML = `
    <div id="topbar"><a href="/">&larr; Search</a></div>
    <div id="clusterGrid"></div>
    <div id="confirmBar"><button class="btn-primary" id="confirmBtn" onclick="confirmCluster()">Confirm &amp; Next Cluster</button></div>
    <div id="zoomOverlay" onclick="this.classList.remove('open')"><img id="zoomImg" src=""></div>
    <div id="loadingOverlay"><span class="spinner"></span><span id="loadingText">Loading...</span></div>
  `;
  const res = await fetch('/queue/duplicates');
  const data = await res.json();
  items = data.uuids || [];
  marks = {};
  const topbar = document.getElementById('topbar');
  const grid = document.getElementById('clusterGrid');
  if (items.length === 0) {
    topbar.innerHTML = '<a href="/">&larr; Search</a>&nbsp;&nbsp;·&nbsp;&nbsp;No more duplicate clusters';
    grid.innerHTML = '<div id="empty">🎉 You have reviewed all detected duplicates.</div>';
    document.getElementById('confirmBar').style.display = 'none';
    return;
  }
  topbar.innerHTML = `<a href="/">&larr; Search</a>&nbsp;&nbsp;·&nbsp;&nbsp;${items.length} similar photos — tap a photo to zoom, tap the tag to mark Delete`;
  grid.innerHTML = '';
  for (const uuid of items) {
    marks[uuid] = 'keep';
    const div = document.createElement('div');
    div.className = 'cshot mark-keep fade-in';
    div.id = 'c-' + uuid;
    div.innerHTML = `<img src="/thumb/${uuid}" loading="lazy"><span class="tag">✅ Keep</span><span class="zoom">🔍 Zoom</span>`;
    div.querySelector('img').onclick = (e) => { e.stopPropagation(); openZoom(uuid); };
    div.querySelector('.zoom').onclick = (e) => { e.stopPropagation(); openZoom(uuid); };
    div.querySelector('.tag').onclick = (e) => { e.stopPropagation(); toggleMark(uuid); };
    div.onclick = () => toggleMark(uuid);
    grid.appendChild(div);
  }
}

function openZoom(uuid) {
  document.getElementById('zoomImg').src = '/thumb/' + uuid;
  document.getElementById('zoomOverlay').classList.add('open');
}

function toggleMark(uuid) {
  const next = marks[uuid] === 'keep' ? 'delete' : 'keep';
  marks[uuid] = next;
  const el = document.getElementById('c-' + uuid);
  el.classList.remove('mark-delete', 'mark-keep');
  el.classList.add(next === 'delete' ? 'mark-delete' : 'mark-keep');
  el.querySelector('.tag').textContent = next === 'delete' ? '🗑️ Delete' : '✅ Keep';
}

async function confirmCluster() {
  const btn = document.getElementById('confirmBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Saving...';
  showLoading('Saving your choices...');

  const deletes = items.filter(u => marks[u] === 'delete');
  const keeps = items.filter(u => marks[u] !== 'delete');
  await fetch('/review-batch', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({deletes, keeps})
  });
  await loadClusterView();
  hideLoading();
}

document.addEventListener('keydown', e => {
  if (queueType !== 'duplicates') {
    if (e.key === 'ArrowLeft') decide('delete');
    if (e.key === 'ArrowRight') decide('keep');
  }
});

loadQueue();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local photo search & cleanup tool.")
    parser.add_argument(
        "--lan", action="store_true",
        help="Allow access from other devices on your WiFi (e.g. your phone). "
             "Off by default for safety — only your Mac can reach the app otherwise. "
             "Generates a one-time access token you must use to connect.",
    )
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()

    load_index()

    if args.lan:
        ACCESS_TOKEN = secrets.token_urlsafe(12)
        lan_ip = socket.gethostbyname(socket.gethostname())
        print("=" * 60)
        print("LAN access enabled. Anyone with this exact link can use the app:")
        print(f"  http://{lan_ip}:{args.port}/?token={ACCESS_TOKEN}")
        print("Do not share this link — it grants full access, including deleting photos.")
        print("=" * 60)
        app.run(host="0.0.0.0", debug=False, port=args.port, threaded=True)
    else:
        print(f"Open on this Mac only: http://localhost:{args.port}")
        print("(Run with --lan to allow access from your phone on the same WiFi.)")
        app.run(host="127.0.0.1", debug=False, port=args.port, threaded=True)
