"""One-off: backfill the `persons` column for photos indexed before it existed.
Fast — no CLIP embedding, just re-reads Photos' own People data via osxphotos."""
import sqlite3

import osxphotos

DB_PATH = "photos_index.db"


def main():
    conn = sqlite3.connect(DB_PATH)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(photos)")}
    if "persons" not in cols:
        conn.execute("ALTER TABLE photos ADD COLUMN persons TEXT")
        conn.commit()

    photosdb = osxphotos.PhotosDB()
    photos = {p.uuid: p for p in photosdb.photos(images=True, movies=False) if not p.shared}

    indexed_uuids = [row[0] for row in conn.execute("SELECT uuid FROM photos")]
    print(f"Backfilling persons for {len(indexed_uuids)} indexed photos...")

    updated = 0
    named = 0
    for uuid in indexed_uuids:
        p = photos.get(uuid)
        if p is None:
            continue
        persons = ",".join(p.persons) if p.persons else ""
        if persons:
            named += 1
        conn.execute("UPDATE photos SET persons = ? WHERE uuid = ?", (persons, uuid))
        updated += 1

    conn.commit()
    print(f"Done. Updated {updated} rows, {named} have at least one named person.")

    distinct = set()
    for row in conn.execute("SELECT DISTINCT persons FROM photos WHERE persons != ''"):
        distinct.update(row[0].split(","))
    print(f"Distinct named people found: {sorted(distinct)}")
    conn.close()


if __name__ == "__main__":
    main()
