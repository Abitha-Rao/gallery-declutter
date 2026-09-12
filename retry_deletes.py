"""One-off: retry deleting photos marked 'delete' in reviews but never actually
removed from Photos, due to Photos.app's AppleScript `delete` command being broken
on this macOS version. Uses the PhotoKit-based delete_photos.swift helper instead."""
import sqlite3
import subprocess

conn = sqlite3.connect("photos_index.db")
uuids = [row[0] for row in conn.execute("SELECT uuid FROM reviews WHERE decision = 'delete'")]
conn.close()

print(f"Retrying delete for {len(uuids)} photos via PhotoKit...")

CHUNK = 20
ok_total = 0
for i in range(0, len(uuids), CHUNK):
    chunk = uuids[i:i + CHUNK]
    result = subprocess.run(
        ["swift", "delete_photos.swift"] + chunk,
        capture_output=True, text=True,
    )
    print(f"  [{i+1}-{i+len(chunk)}/{len(uuids)}]", result.stdout.strip() or result.stderr.strip())

print("\nDone. Check Photos' Recently Deleted album to confirm.")

print("Done. Check Photos' Recently Deleted album to confirm.")
