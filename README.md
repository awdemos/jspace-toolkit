# J-Space Toolkit

A white-box interpretability toolkit for decoder-only transformers that implements the **Jacobian Lens (J-Lens)** and **J-space** methods from ["Verbalizable Representations of Features and the Grand Convergence of Mechanistic Interpretability"](https://transformer-circuits.pub/2025/jan-update/index.html).

In plain English: it trains a cheap linear lens that lets you peek inside a model mid-sequence and ask, "If this layer's hidden state could talk, which tokens would it say next?" It then decomposes those hidden states into a sparse, human-readable vocabulary of "J-lens vectors" and lets you edit them causally.

> **Disclaimer:** This is an early research implementation. It has not been peer reviewed and could contain errors. Use it as a starting point for your own experiments. Bug reports and pull requests are very welcome.

---

## What you get

- **Train a J-Lens** for every layer of any `transformers` decoder-only model.
- **Read out intermediate layers** into token distributions without running the rest of the model.
- **Decompose activations** into sparse non-negative J-space coefficients.
- **Intervene causally** with coordinate swaps, top-k ablations, and steering vectors.
- **Auto-discover the workspace band** where the model's representations become verbalizable.
- **Memory-mapped caching** so you train once and reuse the lens matrices.
- **CLI + notebook demos** for quick exploration.

---

## Install

```bash
git clone <repo>
cd jspace-toolkit
pip install -e ".[dev]"
```

For a reproducible install, use the committed `uv.lock` (or
`requirements.txt` with hashes).

---

## Security note

This toolkit loads and executes model code from HuggingFace repositories. Only
load models you trust. The CLI does **not** accept `--hf-token`; use the
`HF_TOKEN` environment variable or `huggingface-cli login`. By default only a
small allowlist of model IDs may be loaded; use `--allow-unlisted-model` and
`--model-revision` only after reviewing the repository.

---

## 30-second CLI demo

```bash
# 1. Build a tiny corpus
python scripts/prepare_corpus.py --n 1024 --out corpus.json

# 2. Train the lens (uses sshleifer/tiny-gpt2 by default, finishes on CPU)
python -m scripts.train_lens \
  --model sshleifer/tiny-gpt2 \
  --corpus corpus.json \
  --workspace . \
  --max-positions 128 \
  --dtype float32

# Matrices land in lens_cache/<fingerprint>/
```

The entry point is also registered as `train-jspace-lens` after pip install.

### CKA workspace-geometry plot

```bash
# 1. Build a tiny corpus
python scripts/prepare_corpus.py --n 256 --out corpus.json

# 2. Compute the CKA workspace-geometry plot
python -m scripts.workspace_geometry \
  --model sshleifer/tiny-gpt2 \
  --corpus corpus.json \
  --max-positions 128 \
  --dtype float32 \
  --n-probes 1024

# Outputs land in workspace_out/:
#   cka_block.png     - heatmap of layer-by-layer CKA with inferred workspace overlay
#   layer_metrics.png - per-layer kurtosis / accuracy / autocorrelation figure
#   metrics.json      - workspace boundaries + CKA statistics
#   report.html       - self-contained visual report (pass --no-report to skip)
```

`--n-probes` controls how many vocabulary tokens are used as shared probes for the geometry (default 4096). For tiny models on CPU, use a smaller value such as 256 or 512 to keep memory low.

![workspace_geometry.py in action — rich terminal UI, styled plots, and a self-contained HTML report](assets/beautiful_output_demo.gif)

---

## What the code looks like

```python
import torch

from jspace.model_adapter import (
    cache_residuals,
    get_unembedding_matrix,
    layer_indices,
    load_model,
    normalize_fn,
)
from jspace.jacobian_lens import train_jacobian_lens
from jspace.readout import lens_readout
from jspace.decomposition import decompose_jspace
from jspace.interventions import coordinate_swap, apply_intervention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer = load_model("gpt2", device)

# Train J_l matrices for every layer (cached on disk)
texts = ["The capital of France is", "The capital of Japan is"]
enc = tokenizer(texts, return_tensors="pt", padding=True)
J = train_jacobian_lens(
    model,
    enc["input_ids"],
    attention_mask=enc["attention_mask"],
    target_layer=layer_indices(model)[-2],  # penultimate layer target
)

W_U = get_unembedding_matrix(model)
norm = normalize_fn(model)

# Read out layer 8 as if it were the final logits
residuals = cache_residuals(
    model, enc["input_ids"], layers=[8], attention_mask=enc["attention_mask"]
)
h_8 = residuals[8][0, -1]  # last position of the first sequence
top_ids, top_probs = lens_readout(h_8, J[8], W_U, norm)
print(tokenizer.decode(top_ids.tolist()))

# Decompose a hidden state into sparse J-space coefficients
# Pass the pre-built dictionary V = W_U @ J_l (rows are the J-lens vectors) ...
V = W_U @ J[8]
coeffs, h_J, h_perp = decompose_jspace(h_8, V=V, k=10)

# ...or pass J_l and W_U and let the function build V for you.
coeffs, h_J, h_perp = decompose_jspace(h_8, J_l=J[8], W_U=W_U, k=10)

# Swap the J-space coordinates of two tokens and run the model with the edit
paris_id = tokenizer.encode(" Paris", add_special_tokens=False)[0]
london_id = tokenizer.encode(" London", add_special_tokens=False)[0]

def swap(h: torch.Tensor, _pos: int) -> torch.Tensor:
    return coordinate_swap(h, paris_id, london_id, V)

edited_logits = apply_intervention(
    model,
    swap,
    layer_band=(6, 10),
    input_ids=enc["input_ids"],
    attention_mask=enc["attention_mask"],
)
```


```
jspace/
├── model_adapter.py    # HuggingFace loading, residual caching, final norm helpers
├── jacobian_lens.py  # Train J_l with batched VJP and causal future-position averaging
├── readout.py        # lens_readout, token_logit, token_similarity
├── decomposition.py  # sparse non-negative OMP + J-space occupancy
├── interventions.py  # swap, ablate, steer, and causal hook runner
├── discovery.py      # CKA / kurtosis / accuracy workspace discovery
└── utils.py          # memory-mapped cache helpers
scripts/
├── prepare_corpus.py     # tiny corpus generator for demos
├── train_lens.py         # CLI entry point for training
├── workspace_geometry.py # CKA workspace-geometry plot + metrics
└── __init__.py
tests/                # pytest suite using sshleifer/tiny-gpt2
demo.ipynb            # interactive walkthrough
```

---

## Run the tests

```bash
pytest tests/ -v
```

The suite uses `sshleifer/tiny-gpt2` so the Jacobian pass runs comfortably on CPU.

---

## Notes

- The J-Lens is trained in FP32 for gradient accuracy. You can use BF16 for larger models at the cost of some approximation.
- Layers with `d_model > 8192` are cached with memory-mapped files automatically.
- `target_layer` stops the forward pass at a fixed penultimate layer and filters source layers accordingly, matching the paper's definition of `J_{l → L-1}`.

---

## Disclaimer

This is an early research implementation. It has not been peer reviewed and could contain errors. Use it as a starting point for your own experiments. Bug reports and pull requests are welcome.

---

## License

MIT
