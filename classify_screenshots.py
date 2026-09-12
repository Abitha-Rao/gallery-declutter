"""One-off: flag likely screenshots/documents using CLIP zero-shot classification
against the embeddings already computed by build_index.py. No new images processed —
just text-prompt comparisons against existing embeddings."""
import sqlite3

import numpy as np
import open_clip
import torch

DB_PATH = "photos_index.db"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
MARGIN = 0.02  # how much higher the screenshot-prompt similarity must be to flag

SCREENSHOT_PROMPTS = [
    "a screenshot of a phone or computer screen",
    "a screenshot of text, an app, or a website",
]
PHOTO_PROMPTS = [
    "a photograph taken with a camera",
    "a photo of a real physical scene",
]


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    def embed_prompts(prompts):
        with torch.no_grad():
            text = tokenizer(prompts).to(device)
            emb = model.encode_text(text)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().mean(axis=0)

    screenshot_vec = embed_prompts(SCREENSHOT_PROMPTS)
    photo_vec = embed_prompts(PHOTO_PROMPTS)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS tags (uuid TEXT, tag TEXT, PRIMARY KEY (uuid, tag))")
    conn.execute("DELETE FROM tags WHERE tag = 'screenshot'")

    rows = conn.execute("SELECT uuid, embedding FROM photos").fetchall()
    print(f"Classifying {len(rows)} photos...")

    flagged = 0
    for uuid, emb_blob in rows:
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        screenshot_sim = float(emb @ screenshot_vec)
        photo_sim = float(emb @ photo_vec)
        if screenshot_sim - photo_sim > MARGIN:
            conn.execute("INSERT OR REPLACE INTO tags (uuid, tag) VALUES (?, 'screenshot')", (uuid,))
            flagged += 1

    conn.commit()
    conn.close()
    print(f"Flagged {flagged} likely screenshots/documents.")


if __name__ == "__main__":
    main()
