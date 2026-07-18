"""Smoke tests for console-script entry points."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def tiny_corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(["The cat sat.", "In 1950, scientists discovered."]))
    return path


def test_prepare_corpus_smoke(tmp_path: Path) -> None:
    out = tmp_path / "corpus.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.prepare_corpus",
            "--n",
            "4",
            "--out",
            str(out),
            "--workspace",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    data = json.loads(out.read_text())
    assert len(data) == 4


def test_prepare_corpus_bad_parent(tmp_path: Path) -> None:
    out = tmp_path / "nonexistent" / "corpus.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.prepare_corpus",
            "--n",
            "1",
            "--out",
            str(out),
            "--workspace",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1


def test_prepare_corpus_path_escapes_workspace(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.prepare_corpus",
            "--n",
            "1",
            "--out",
            "../escaped.json",
            "--workspace",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "outside" in result.stderr.lower()


def test_train_lens_rejects_hf_token_argument(tmp_path: Path, tiny_corpus: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.train_lens",
            "--model",
            "sshleifer/tiny-gpt2",
            "--corpus",
            str(tiny_corpus),
            "--cache-dir",
            str(tmp_path / "lens_cache"),
            "--workspace",
            str(tmp_path),
            "--hf-token",
            "hf_secret",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--hf-token" in result.stderr


@pytest.mark.slow
@pytest.mark.parametrize("frozen_qk", [False, True])
def test_train_jspace_lens_smoke(tmp_path: Path, tiny_corpus: Path, frozen_qk: bool) -> None:
    cache_dir = tmp_path / "lens_cache"
    cmd = [
        sys.executable,
        "-m",
        "scripts.train_lens",
        "--model",
        "sshleifer/tiny-gpt2",
        "--corpus",
        str(tiny_corpus),
        "--cache-dir",
        str(cache_dir),
        "--workspace",
        str(tmp_path),
        "--max-positions",
        "16",
        "--batch-size",
        "1",
        "--dtype",
        "float32",
    ]
    if frozen_qk:
        cmd.append("--frozen-qk")

    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert any(cache_dir.rglob("J_*.safetensors"))
    assert any(cache_dir.rglob("J_*.sha256"))


@pytest.mark.slow
def test_workspace_geometry_cli_smoke(tmp_path: Path, tiny_corpus: Path) -> None:
    output_dir = tmp_path / "workspace_out"
    cmd = [
        sys.executable,
        "-m",
        "scripts.workspace_geometry",
        "--model",
        "sshleifer/tiny-gpt2",
        "--corpus",
        str(tiny_corpus),
        "--output-dir",
        str(output_dir),
        "--cache-dir",
        str(tmp_path / "lens_cache"),
        "--workspace",
        str(tmp_path),
        "--max-positions",
        "16",
        "--batch-size",
        "1",
        "--n-probes",
        "64",
        "--dtype",
        "float32",
    ]
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert (output_dir / "cka_block.png").exists()
    assert (output_dir / "metrics.json").exists()
