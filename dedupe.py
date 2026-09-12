"""One-time near-duplicate clustering over the photo index.

Compares each photo only to others taken within a short time window (bursts/
repeats are taken close together), avoiding an O(n^2) full similarity matrix.
"""
import sqlite3
from datetime import datetime, timedelta

import numpy as np

DB_PATH = "photos_index.db"
TIME_WINDOW = timedelta(minutes=5)
SIMILARITY_THRESHOLD = 0.93


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build_clusters():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS clusters (uuid TEXT PRIMARY KEY, cluster_id INTEGER)")
    rows = conn.execute("SELECT uuid, date, embedding FROM photos WHERE date IS NOT NULL").fetchall()

    items = []
    for uuid, date_str, emb_blob in rows:
        try:
            dt = datetime.fromisoformat(date_str)
        except ValueError:
            continue
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        items.append((dt, uuid, emb))
    items.sort(key=lambda x: x[0])

    n = len(items)
    uf = UnionFind(n)

    for i in range(n):
        dt_i, _, emb_i = items[i]
        for j in range(i + 1, n):
            dt_j = items[j][0]
            if dt_j - dt_i > TIME_WINDOW:
                break
            sim = float(emb_i @ items[j][2])
            if sim >= SIMILARITY_THRESHOLD:
                uf.union(i, j)

    groups = {}
    for i in range(n):
        root = uf.find(i)
        groups.setdefault(root, []).append(items[i][1])

    conn.execute("DELETE FROM clusters")
    cluster_id = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        for uuid in members:
            conn.execute("INSERT INTO clusters (uuid, cluster_id) VALUES (?, ?)", (uuid, cluster_id))
        cluster_id += 1
    conn.commit()
    print(f"Found {cluster_id} duplicate clusters covering "
          f"{sum(len(m) for m in groups.values() if len(m) >= 2)} photos.")
    conn.close()


if __name__ == "__main__":
    build_clusters()
