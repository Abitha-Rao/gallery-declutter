# Photo Search & Cleanup

A local, private tool for searching and decluttering a large Apple Photos library — natural-language search, duplicate/screenshot/blurry detection, and a swipe-to-declutter review flow (usable from your phone on the same WiFi). Everything runs on your own Mac; nothing is uploaded anywhere.

## What it does

- **Natural-language search** — "sunset over the ocean", "birthday cake", "screenshot of text" — powered by CLIP embeddings computed once for your whole library. Also supports date-range filters and (if you use Photos' People feature) searching by name, e.g. "Mom at the beach".
- **Duplicate detection** — finds burst/near-identical shots and lets you compare them side by side before deciding what to keep.
- **Screenshot / document detection** and **blurry-photo detection** — dedicated cleanup queues for the easiest, most obvious deletes.
- **Swipe-to-declutter** — a Tinder-style review flow (swipe or tap) for random batches of photos, with **nothing deleted until you explicitly confirm** at the end of a batch, plus an Undo.
- **Share button** — share the currently-reviewed photo directly from the browser (iOS Safari's native share sheet).
- **Installable as a home-screen app** on iPhone (Add to Home Screen in Safari).

## Privacy & safety

- **Nothing leaves your Mac.** No cloud services, no accounts, no telemetry. Your Photos library is read locally; CLIP runs locally.
- **Deletes are always recoverable.** "Delete" moves a photo to Photos' own **Recently Deleted** album (kept ~30 days, like deleting normally in Photos) — never a permanent, irreversible delete.
- **Nothing is deleted without your explicit confirmation.** The review flows mark decisions locally as you swipe; the actual delete only happens when you tap Confirm.
- **Not reachable from your network by default.** The app only listens on your Mac (`127.0.0.1`) unless you explicitly start it with `--lan`, which requires a randomly generated access token to connect — don't share that link.

## Requirements

- A Mac running macOS with your library in the **Photos** app.
- **Python 3.11 or newer.** macOS's built-in `python3` is often too old (3.9) — install a current version from [python.org](https://www.python.org/downloads/) if `setup.sh` can't find one.
- [Xcode Command Line Tools](https://developer.apple.com/xcode/resources/) (for `swift`, used to safely delete photos via PhotoKit) — usually already installed if you've used git or homebrew before; otherwise `xcode-select --install`.

## Setup

```bash
git clone <this-repo-url>
cd <this-repo>
./setup.sh
```

This creates a virtual environment and installs dependencies. Then, with the venv activated (`source venv/bin/activate`):

```bash
python build_index.py            # indexes your Photos library (one-time; can take a while for large libraries)
python dedupe.py                 # finds duplicate/burst clusters
python classify_screenshots.py   # flags likely screenshots/documents
python classify_blur.py          # flags likely blurry photos
python app.py                    # starts the app at http://localhost:5050
```

Run these from a **Terminal window you'll reuse** — macOS permissions below are granted per-app, and you'll want them to stick across the whole setup.

### macOS permissions you'll be asked for

The first time you run `build_index.py`, macOS will ask for a few permissions. If a step below doesn't prompt automatically, grant it manually in **System Settings → Privacy & Security**:

1. **Photos access** — lets the script read your library. Should prompt automatically on first run.
2. **Full Disk Access** — some Photos library internals require this. Add **Terminal** (or whichever app you're running commands from) under **Privacy & Security → Full Disk Access**, then **fully quit and reopen that app** (toggling the setting doesn't affect already-running processes).
3. **Automation → Photos** — required for the delete/reveal features to control the Photos app. Under **Privacy & Security → Automation**, make sure your terminal app has the **Photos** toggle enabled.

If `build_index.py` fails with a `PermissionError` copying `Photos.sqlite`, it's almost always #2 — grant Full Disk Access and **restart the terminal app**, not just re-run the command.

### If your library uses "Optimize Mac Storage" (iCloud)

Photos not fully downloaded to your Mac won't have a usable original file, but the indexer will automatically fall back to Photos' cached preview image, so nothing gets skipped — just note that shared/zoomed images will be preview-resolution, not full quality, unless you switch to "Download Originals to this Mac" in Photos' settings first.

## Using it

- **On your Mac:** open `http://localhost:5050`.
- **From your phone (optional):** run `python app.py --lan` instead. It prints a link like `http://<your-mac-ip>:5050/?token=...` — open that exact link in Safari on your phone (same WiFi). The token is required to connect; don't share this link with anyone else, since it grants full search + delete access.
- Add to your phone's home screen via Safari's Share → **Add to Home Screen** for a full-screen app icon.

## Known limitations

- macOS's own AppleScript `delete` command for Photos is broken on some recent macOS versions (fails silently or errors). This tool works around it using PhotoKit directly (`delete_photos.swift`) — if you hit delete issues, check that Xcode Command Line Tools are installed so `swift` is available.
- Search quality depends on CLIP's general-purpose understanding — it's good at scenes/objects, less precise for very specific things (exact document text, precise counts, etc.).
- The "screenshot" and "blurry" detectors are heuristics tuned on one library; you may want to adjust the thresholds (`MARGIN` in `classify_screenshots.py`, `BLUR_PERCENTILE` in `classify_blur.py`) for your own photos.

## License

MIT — see [LICENSE](LICENSE).
