"""Regression tests for bug fixes (see commit history)."""

import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from jspace import JSpaceError
from jspace.decomposition import _resolve_V, jspace_occupancy
from jspace.interventions import apply_intervention
from jspace.jacobian_lens import _attach_frozen_qk_hooks, train_jacobian_lens
from jspace.model_adapter import layer_indices, load_model
from jspace.utils import model_fingerprint


def test_resolve_V_uses_j_lens_vector_geometry():
    """V must be W_U @ J_l (rows = J-lens vectors), not W_U @ J_l.T."""
    d = 8
    vocab = 5
    J_l = np.arange(d * d, dtype=np.float32).reshape(d, d)  # non-symmetric
    W_U = torch.randn(vocab, d)
    V = _resolve_V(None, J_l, W_U, torch.device("cpu"), torch.float32)
    expected = W_U @ torch.from_numpy(J_l)
    assert torch.allclose(V, expected)


def test_occupancy_accumulates_selected_vectors():
    """A 2-sparse signal must need at least 2 vectors (was 1 before the fix)."""
    d = 32
    V = torch.randn(50, d)
    V = V / (V.norm(dim=-1, keepdim=True) + 1e-9)
    h = (2.0 * V[3] + 1.5 * V[17]).to(torch.float32)
    k = jspace_occupancy(h, V=V, max_k=10, threshold=1e-3, random_seed=0)
    assert 2 <= k <= 10


def test_fingerprint_changes_with_corpus_and_revision():
    base = model_fingerprint("gpt2", 10, False)
    assert base != model_fingerprint("gpt2", 10, False, corpus_hash="abc")
    assert base != model_fingerprint("gpt2", 10, False, revision="v1")
    assert model_fingerprint("gpt2", 10, False) == model_fingerprint("gpt2", 10, False)


def test_train_jacobian_lens_rejects_empty_corpus():
    corpus = torch.zeros(0, 4, dtype=torch.long)
    with pytest.raises(JSpaceError, match="empty"):
        train_jacobian_lens(None, corpus)


def test_train_jacobian_lens_rejects_mismatched_mask():
    corpus = torch.ones(2, 4, dtype=torch.long)
    mask = torch.ones(2, 5, dtype=torch.long)
    with pytest.raises(JSpaceError, match="attention_mask"):
        train_jacobian_lens(None, corpus, attention_mask=mask)


def test_frozen_qk_hooks_match_gpt2_c_attn():
    """GPT-2's fused c_attn must be hooked (0 hooks before the fix)."""
    model, _ = load_model("sshleifer/tiny-gpt2", torch.device("cpu"), torch.float32)
    handles = _attach_frozen_qk_hooks(model)
    try:
        assert len(handles) == len(layer_indices(model))
    finally:
        for h in handles:
            h.remove()


def test_apply_intervention_accepts_batched_inputs():
    """apply_intervention must not crash for batch size > 1."""
    model, tokenizer = load_model("sshleifer/tiny-gpt2", torch.device("cpu"), torch.float32)
    enc = tokenizer(["hello world", "hi"], return_tensors="pt", padding=True)
    logits = apply_intervention(
        model,
        lambda h, layer_idx: h,
        (0, 1),
        enc["input_ids"],
        attention_mask=enc["attention_mask"],
    )
    assert logits.shape[0] == 2
    assert logits.shape[1] == enc["input_ids"].shape[1]


def test_inline_image_color_index_stays_in_palette():
    """Near-white greys used to overflow to index 256 pre-fix (b05e14e)."""
    from scripts.inline_image import _rgb_to_256

    assert _rgb_to_256(255, 255, 255) == 231
    assert _rgb_to_256(250, 250, 250) == 231
    assert _rgb_to_256(248, 248, 248) == 231  # was 256 before the fix
    assert _rgb_to_256(247, 247, 247) == 255
    assert _rgb_to_256(255, 0, 0) == 196
    for r in range(0, 256, 11):
        for g in range(0, 256, 11):
            for b in range(0, 256, 11):
                assert 0 <= _rgb_to_256(r, g, b) <= 255


def test_inline_image_render_escape_codes_bounded(tmp_path):
    from PIL import Image

    from scripts.inline_image import render

    Image.new("RGB", (3, 2), (250, 250, 250)).save(tmp_path / "near_white.png")
    out = render(str(tmp_path / "near_white.png"), width=3)
    for seq in out.split("\033[38;5;")[1:]:
        assert 0 <= int(seq.split("m")[0]) <= 255


def test_inline_image_closes_image_handle(tmp_path, monkeypatch):
    from PIL import Image

    from scripts import inline_image

    Image.new("RGB", (2, 2), (10, 20, 30)).save(tmp_path / "t.png")
    opened = []
    real_open = Image.open

    def spy(path):
        handle = real_open(path)
        opened.append(handle)
        return handle

    monkeypatch.setattr(inline_image.Image, "open", spy)
    inline_image.render(str(tmp_path / "t.png"), width=2)

    assert len(opened) == 1
    assert opened[0].fp is None  # closed by the ``with`` block


def _wg_argv(tmp_path, corpus, *extra):
    return [
        "prog",
        "--model",
        "sshleifer/tiny-gpt2",
        "--corpus",
        str(corpus),
        "--workspace",
        str(tmp_path),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--output-dir",
        str(tmp_path / "out"),
        *extra,
    ]


def test_workspace_geometry_probe_ids_error_precedes_model_load(tmp_path):
    """Malformed --probe-ids must fail before model loading or training."""
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps(["The cat sat."]))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.workspace_geometry",
            "--model",
            "totally-bogus-model-name",
            "--corpus",
            str(corpus),
            "--workspace",
            str(tmp_path),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output-dir",
            str(tmp_path / "out"),
            "--probe-ids",
            "1,2,x",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "--probe-ids" in result.stderr
    # The allowlist error would name the model; it must never be reached.
    assert "totally-bogus-model-name" not in result.stderr


def test_workspace_geometry_rejects_single_layer_model(tmp_path, monkeypatch, capsys):
    from transformers import GPT2Config, GPT2LMHeadModel

    import scripts.workspace_geometry as wg

    config = GPT2Config(n_layer=1, n_head=1, n_embd=8, n_positions=8, n_ctx=8, vocab_size=16)
    monkeypatch.setattr(wg, "load_model", lambda *a, **k: (GPT2LMHeadModel(config), object()))
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps(["The cat sat."]))
    monkeypatch.setattr(sys, "argv", _wg_argv(tmp_path, corpus))

    with pytest.raises(SystemExit) as exc:
        wg.main()
    assert exc.value.code == 1
    assert "1 layer" in capsys.readouterr().err


@pytest.mark.parametrize("bad_layer", ["-1", "99"])
def test_workspace_geometry_rejects_out_of_range_target_layer(
    tmp_path, monkeypatch, capsys, bad_layer
):
    import scripts.workspace_geometry as wg

    def boom(*a, **k):
        raise AssertionError("training must not run for an invalid --target-layer")

    monkeypatch.setattr(wg, "train_jacobian_lens", boom)
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps(["The cat sat on the mat."]))
    monkeypatch.setattr(sys, "argv", _wg_argv(tmp_path, corpus, "--target-layer", bad_layer))

    with pytest.raises(SystemExit) as exc:
        wg.main()
    assert exc.value.code == 1
    assert "out of range" in capsys.readouterr().err


def test_workspace_geometry_rejects_out_of_range_probe_ids_before_training(
    tmp_path, monkeypatch, capsys
):
    import scripts.workspace_geometry as wg

    def boom(*a, **k):
        raise AssertionError("training must not run for out-of-range --probe-ids")

    monkeypatch.setattr(wg, "train_jacobian_lens", boom)
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps(["The cat sat on the mat."]))
    monkeypatch.setattr(sys, "argv", _wg_argv(tmp_path, corpus, "--probe-ids", f"1,{10**9}"))

    with pytest.raises(SystemExit) as exc:
        wg.main()
    assert exc.value.code == 1
    assert "out of range" in capsys.readouterr().err
