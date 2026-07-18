#!/usr/bin/env bash
# Records the README demo session showcasing the toolkit's visual output.
#
# Record and render with:
#   asciinema rec --cols 110 --rows 36 -c scripts/record_readme_demo.sh demo.cast
#   agg --speed 3 --font-size 16 demo.cast assets/beautiful_output_demo.gif
set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

# Clean previous demo artifacts and enter a fresh workspace.
DEMO_DIR="$(mktemp -d /tmp/jspace_demo.XXXXXX)"
rm -rf "$DEMO_DIR"
mkdir -p "$DEMO_DIR"
cd "$DEMO_DIR"

# Use the repo's virtual environment.
source "$REPO_DIR/.venv/bin/activate"

export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-demo}}"

# Simulate typing a line at a human-like speed.
type_line() {
    local line="$1"
    printf '$ '
    for (( i=0; i<${#line}; i++ )); do
        printf '%s' "${line:$i:1}"
        sleep 0.015
    done
    printf '\n'
}

clear

type_line "# J-Space Toolkit — workspace geometry, beautifully rendered"
sleep 0.3

type_line "python scripts/prepare_corpus.py --n 8 --out corpus.json"
sleep 0.2
python "$REPO_DIR/scripts/prepare_corpus.py" --n 8 --out corpus.json --workspace .
sleep 0.4

type_line "python -m scripts.workspace_geometry \\"
type_line "    --model gpt2 --corpus corpus.json --n-probes 256"
sleep 0.2
PYTHONPATH="$REPO_DIR" python -m scripts.workspace_geometry \
  --model gpt2 \
  --corpus corpus.json \
  --output-dir workspace_out \
  --cache-dir lens_cache \
  --workspace . \
  --max-positions 16 \
  --n-probes 256 \
  --dtype float32 \
  2> >(grep -v -E "Warning: You are sending unauthenticated requests|Loading weights|LOAD REPORT|UNEXPECTED| transformer\.h\.|Notes:|can be ignored|^Key |^---|^$" >&2)
sleep 0.8

type_line "python scripts/inline_image.py workspace_out/cka_block.png --width 72"
sleep 0.2
python "$REPO_DIR/scripts/inline_image.py" workspace_out/cka_block.png --width 72
sleep 1.0

type_line "ls -lh workspace_out/"
sleep 0.2
ls -lh workspace_out/
sleep 0.4

type_line "# open workspace_out/report.html for the full interactive report"
sleep 1.5
