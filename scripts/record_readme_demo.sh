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

echo "$ # J-Space Toolkit — workspace geometry demo"
sleep 0.5

echo ""
echo "$ python scripts/prepare_corpus.py --n 128 --out corpus.json"
sleep 0.5
python "$REPO_DIR/scripts/prepare_corpus.py" --n 128 --out corpus.json --workspace .
sleep 0.5

echo ""
echo "$ python -m scripts.workspace_geometry \\"
echo "    --model sshleifer/tiny-gpt2 \\"
echo "    --corpus corpus.json \\"
echo "    --max-positions 16 \\"
echo "    --n-probes 64 \\"
echo "    --target-layer 1"
sleep 0.5
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

echo ""
echo "$ cat workspace_out/metrics.json"
sleep 0.5
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

echo ""
echo "$ python scripts/inline_image.py workspace_out/cka_block.png --width 70"
sleep 0.5
PYTHONPATH="$REPO_DIR" python "$REPO_DIR/scripts/inline_image.py" workspace_out/cka_block.png --width 70
sleep 0.5

echo ""
echo "$ ls -lh workspace_out/"
sleep 0.5
ls -lh workspace_out/
sleep 0.5
