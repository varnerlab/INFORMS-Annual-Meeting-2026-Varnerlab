#!/usr/bin/env bash
# Build the INFORMS 2026 deck and lint it. Usage (from the repo root): bash talk/check_deck.sh <expected pages>
# Fails on a build error, an overfull box, an undefined reference, a glyph the font does not have, a missing Helvetica
# Neue font, a page count other than the argument, or a forbidden string in the slide and figure sources. Writes page PNGs to talk/build/png/.
set -euo pipefail

expected="${1:?usage: bash talk/check_deck.sh <expected pages>}"
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"
fail=0

# build
if ! make -s >build.log 2>&1; then
    echo "FAIL: build error (see talk/build.log and talk/informs2026.log)"
    tail -20 informs2026.log || true
    exit 1
fi

# overfull boxes and undefined references
if grep -n "Overfull" informs2026.log; then
    echo "FAIL: overfull box"
    fail=1
fi
if grep -n "undefined" informs2026.log | grep -v "fontspec"; then
    echo "FAIL: undefined reference"
    fail=1
fi

# a character the font lacks prints as nothing
if grep -n "Missing character" informs2026.log; then
    echo "FAIL: missing glyph"
    fail=1
fi

# the slide font must be embedded, not a fallback
if ! pdffonts informs2026.pdf | grep -q "HelveticaNeue-Light"; then
    echo "FAIL: Helvetica Neue Light is not embedded"
    fail=1
fi

# page count
pages="$(pdfinfo informs2026.pdf | awk '/^Pages:/ {print $2}')"
if [ "$pages" != "$expected" ]; then
    echo "FAIL: $pages pages, expected $expected"
    fail=1
fi

# forbidden strings in the slide copy and figures: em dashes, internal labels, placeholders, and words the talk
# never uses
sources="$(ls slides/*.tex figs/*.tex 2>/dev/null || true)"
if [ -n "$sources" ]; then
    for pattern in "---" "global_gamma" "l8b" "fixed:" "SCS" "TODO" "TBD"; do
        if grep -n -F -- "$pattern" $sources; then
            echo "FAIL: forbidden string '$pattern'"
            fail=1
        fi
    done
    for word in "returns" "hindsight"; do
        if grep -n -i -w -- "$word" $sources; then
            echo "FAIL: forbidden word '$word'"
            fail=1
        fi
    done
fi

# page images for a visual check
rm -rf build/png
mkdir -p build/png
pdftoppm -r 60 -png informs2026.pdf build/png/page

if [ "$fail" -ne 0 ]; then
    exit 1
fi
echo "OK: $pages pages, fonts embedded, no overfull boxes, no forbidden strings"
