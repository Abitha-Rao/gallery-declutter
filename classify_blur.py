"""One-off: flag likely blurry photos using a simple edge-variance sharpness proxy
(no OpenCV dependency needed). Computes a score per photo, stores it, then flags
the least-sharp tail as 'blurry' in the shared tags table."""
import sqlite3

import numpy as np
from PIL import Image, ImageFilter

DB_PATH = "photos_index.db"
BLUR_PERCENTILE = 10  # flag the bottom N% sharpest as "blurry"


def sharpness_score(path):
    img = Image.open(path).convert("L")
    img.thumbnail((400, 400))  # downscale for speed, doesn't need full-res
    edges = img.filter(ImageFilter.FIND_EDGES)
    arr = np.asarray(edges, dtype=np.float32)
    return float(arr.var())


def main():
    conn = sqlite3.connect(DB_PATH)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(photos)")}
    if "blur_score" not in cols:
        conn.execute("ALTER TABLE photos ADD COLUMN blur_score REAL")
    conn.execute("CREATE TABLE IF NOT EXISTS tags (uuid TEXT, tag TEXT, PRIMARY KEY (uuid, tag))")

    rows = conn.execute("SELECT uuid, path FROM photos").fetchall()
    print(f"Scoring sharpness for {len(rows)} photos...")

    scores = {}
    for i, (uuid, path) in enumerate(rows):
        try:
            scores[uuid] = sharpness_score(path)
        except Exception as e:
            print(f"skip {path}: {e}")
            continue
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(rows)}")

    for uuid, score in scores.items():
        conn.execute("UPDATE photos SET blur_score = ? WHERE uuid = ?", (score, uuid))
    conn.commit()

    threshold = np.percentile(list(scores.values()), BLUR_PERCENTILE)
    conn.execute("DELETE FROM tags WHERE tag = 'blurry'")
    flagged = 0
    for uuid, score in scores.items():
        if score <= threshold:
            conn.execute("INSERT OR REPLACE INTO tags (uuid, tag) VALUES (?, 'blurry')", (uuid,))
            flagged += 1

    conn.commit()
    conn.close()
    print(f"Flagged {flagged} likely-blurry photos (bottom {BLUR_PERCENTILE}% sharpness, threshold={threshold:.1f}).")


if __name__ == "__main__":
    main()
