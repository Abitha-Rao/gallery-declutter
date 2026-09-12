"""Index the macOS Photos library: CLIP embeddings + metadata into SQLite."""
import sqlite3
import sys
import time

import numpy as np
import open_clip
import osxphotos
import torch
from PIL import Image

DB_PATH = "photos_index.db"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


def get_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(device).eval()
    return model, preprocess, device


def init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS photos (
            uuid TEXT PRIMARY KEY,
            path TEXT,
            date TEXT,
            latitude REAL,
            longitude REAL,
            embedding BLOB,
            persons TEXT
        )
    """)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(photos)")}
    if "persons" not in cols:
        conn.execute("ALTER TABLE photos ADD COLUMN persons TEXT")
    conn.commit()


def already_indexed(conn):
    return {row[0] for row in conn.execute("SELECT uuid FROM photos")}


def main():
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    done = already_indexed(conn)

    print("Loading CLIP model...")
    model, preprocess, device = get_model()
    print(f"Using device: {device}")

    photosdb = osxphotos.PhotosDB()
    photos = [p for p in photosdb.photos(images=True, movies=False) if not p.shared]
    print(f"Found {len(photos)} photos in library, {len(done)} already indexed.")

    start = time.time()
    indexed = 0
    for i, p in enumerate(photos):
        if p.uuid in done:
            continue
        path = p.path
        if not path:
            derivatives = p.path_derivatives
            path = derivatives[0] if derivatives else None
        if not path:
            continue  # no local original or cached preview available
        try:
            img = Image.open(path).convert("RGB")
            tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            emb_bytes = emb.cpu().numpy().astype(np.float32).tobytes()
        except Exception as e:
            print(f"skip {path}: {e}", file=sys.stderr)
            continue

        lat, lon = (p.location if p.location else (None, None))
        persons = ",".join(p.persons) if p.persons else ""
        conn.execute(
            "INSERT OR REPLACE INTO photos (uuid, path, date, latitude, longitude, embedding, persons) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (p.uuid, path, p.date.isoformat() if p.date else None, lat, lon, emb_bytes, persons),
        )
        indexed += 1
        if indexed % 200 == 0:
            conn.commit()
            elapsed = time.time() - start
            print(f"Indexed {indexed} ({i+1}/{len(photos)}) in {elapsed:.0f}s")

    conn.commit()
    print(f"Done. Newly indexed: {indexed}")


if __name__ == "__main__":
    main()
