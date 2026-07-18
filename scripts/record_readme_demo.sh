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

clear
echo "# J-Space Toolkit — workspace geometry demo"
sleep 0.3

echo ""
echo "# Step 1: prepare a tiny corpus"
sleep 0.3
python "$REPO_DIR/scripts/prepare_corpus.py" --n 128 --out corpus.json --workspace .
sleep 1.0

echo ""
echo "# Step 2: compute the CKA workspace-geometry plot"
sleep 0.3
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
sleep 1.0

echo ""
echo "# Step 3: inspect the generated metrics"
sleep 0.3
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
sleep 1.0

echo ""
echo "# Step 4: view the generated CKA heatmap"
sleep 0.3
PYTHONPATH="$REPO_DIR" python "$REPO_DIR/scripts/inline_image.py" workspace_out/cka_block.png --width 70
sleep 1.0

echo ""
echo "# Generated files:"
ls -lh workspace_out/
sleep 1.0
