# J-Space Toolkit

A white-box interpretability toolkit that trains and applies the **Jacobian Lens (J-Lens)** and **J-space** methods to any open decoder-only transformer.

## Features

- **Jacobian Lens training** over a corpus of prompts, with per-layer memory-mapped caching.
- **Lens readout** that decodes intermediate residual activations into a token distribution.
- **Sparse J-space decomposition** with non-negative Orthogonal Matching Pursuit.
- **Causal interventions**: coordinate swaps, J-space ablations, and steering vectors.
- **Workspace layer auto-discovery** via CKA, kurtosis, and next-token accuracy proxies.
- CLI and notebook demos.

## Installation

```bash
git clone <repo>
cd jspace-toolkit
pip install -e .
```

## Quick start: CLI

Generate a small corpus and train a lens:

```bash
python scripts/prepare_corpus.py --n 1024 --out corpus.json
python train_lens.py --model gpt2 --corpus corpus.json --max-positions 128 --dtype float32
```

Cached `J_l` matrices are written to `lens_cache/<fingerprint>/`.

## Notebook demo

Open `demo.ipynb` for a worked example on a small open model, including:

- J-Lens training
- Verbal-report sanity check
- CKA workspace-structure plot
- Inferred workspace boundaries

## Project structure

```
jspace/
├── model_adapter.py    # HF loading, residual caching, normalization
├── jacobian_lens.py  # Training loop with FP32 batched VJP
├── readout.py        # lens_readout and per-token probes
├── decomposition.py  # sparse J-space decomposition
├── interventions.py  # swap, ablate, steer
├── discovery.py      # workspace auto-discovery
└── utils.py          # caching helpers
```

## Running tests

```bash
pytest tests/ -v
```

Tests use `sshleifer/tiny-gpt2` so the Jacobian pass finishes quickly on CPU.

## Notes

- The J-Lens is computed in FP32 for gradient accuracy. Large models can use BF16 at the cost of approximation.
- `d_model > 8192` layers are memory-mapped automatically.
