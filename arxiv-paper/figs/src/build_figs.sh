#!/bin/sh
# Build the paper's figures from the talk's figure code and data: compile each source here with XeLaTeX from the
# talk/ folder (the figure code reads data/*.csv relative to it), then copy the PDFs into figs/main and
# figs/supplement. Run from anywhere: sh arxiv-paper/figs/src/build_figs.sh
set -e
SRC=$(cd "$(dirname "$0")" && pwd)
PAPER=$(cd "$SRC/../.." && pwd)
TALK=$(cd "$PAPER/../talk" && pwd)
OUT=$(mktemp -d)
cd "$TALK"
for f in "$SRC"/fig*.tex; do
  name=$(basename "$f" .tex)
  TEXINPUTS="$SRC//:" xelatex -interaction=nonstopmode -halt-on-error -output-directory="$OUT" "$f" > "$OUT/$name.out" 2>&1 \
    || { echo "FAILED: $name (log: $OUT/$name.log)"; exit 1; }
  case "$name" in
    figS_*) cp "$OUT/$name.pdf" "$PAPER/figs/supplement/$name.pdf" ;;
    *)      cp "$OUT/$name.pdf" "$PAPER/figs/main/$name.pdf" ;;
  esac
  echo "built $name"
done
