#!/bin/bash
# One-time setup: creates a venv with the right Python version and installs dependencies.
set -e

cd "$(dirname "$0")"

PYTHON=""
for candidate in python3.12 python3.11 python3.13; do
    if command -v "$candidate" &> /dev/null; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Python 3.11+ not found."
    echo "This tool needs Python 3.11 or newer (macOS's built-in python3 is usually too old)."
    echo "Install it from https://www.python.org/downloads/ (the official .pkg installer), then re-run this script."
    exit 1
fi

echo "Using $PYTHON ($($PYTHON --version))"
$PYTHON -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt

echo ""
echo "Setup complete. Next steps:"
echo "  1. source venv/bin/activate"
echo "  2. python build_index.py     # indexes your Photos library (one-time, can take a while)"
echo "  3. python dedupe.py          # finds duplicate/burst photos"
echo "  4. python classify_screenshots.py"
echo "  5. python classify_blur.py"
echo "  6. python app.py             # starts the app at http://localhost:5050"
echo ""
echo "See README.md for the macOS permissions you'll need to grant along the way."
