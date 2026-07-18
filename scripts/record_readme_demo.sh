#!/usr/bin/env bash
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

type_line "# J-Space Toolkit — workspace geometry demo"
sleep 0.3

line1="python scripts/prepare_corpus.py --n 128 --out corpus.json"
type_line "$line1"
sleep 0.2
python "$REPO_DIR/scripts/prepare_corpus.py" --n 128 --out corpus.json --workspace .
sleep 0.5

type_line "python -m scripts.workspace_geometry \\"
type_line "    --model sshleifer/tiny-gpt2 \\"
type_line "    --corpus corpus.json \\"
type_line "    --max-positions 16 \\"
type_line "    --n-probes 64 \\"
type_line "    --target-layer 1"
sleep 0.2
PYTHONPATH="$REPO_DIR" python -m scripts.workspace_geometry \
  --model sshleifer/tiny-gpt2 \
  --corpus corpus.json \
  --output-dir workspace_out \
  --cache-dir lens_cache \
  --workspace . \
  --max-positions 16 \
  --n-probes 64 \
  --dtype float32 \
  --target-layer 1 \
  2> >(grep -v -E "Warning: You are sending unauthenticated requests|\[transformers\].*torch_dtype is deprecated|GPT2LMHeadModel LOAD REPORT|UNEXPECTED| transformer\.h\.|Notes:|can be ignored" >&2)
sleep 0.5

type_line "cat workspace_out/metrics.json"
sleep 0.2
python - <<'PY'
import json
with open("workspace_out/metrics.json") as f:
    m = json.load(f)
print(f"model:            {m['model']}")
print(f"target_layer:     {m['target_layer']}")
print(f"n_layers:         {m['n_layers']}")
print(f"workspace band:   [{m['workspace_start']}, {m['workspace_end']}]")
print(f"mean CKA:         {m['mean_cka']:.4f}")
PY
sleep 0.5

type_line "python scripts/inline_image.py workspace_out/cka_block.png --width 70"
sleep 0.2
PYTHONPATH="$REPO_DIR" python "$REPO_DIR/scripts/inline_image.py" workspace_out/cka_block.png --width 70
sleep 0.5

type_line "ls -lh workspace_out/"
sleep 0.2
ls -lh workspace_out/
sleep 0.5
