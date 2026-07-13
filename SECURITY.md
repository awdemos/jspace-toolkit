# Security Policy

## Threat model

`jspace-toolkit` loads transformer model weights, tokenizers, and configuration
files from the HuggingFace Hub (or from local paths). Treating model artifacts
as untrusted data is essential: a malicious repository can exploit the model-
loading pipeline to execute arbitrary code on the researcher's machine,
effectively giving the attacker the privileges of the invoking user.

Key risks:

- **Model-supply-chain RCE** — compromised or typo-squatted HuggingFace
  repositories can run code during `from_pretrained()`.
- **Dependency-supply-chain RCE** — unpinned or malicious PyPI packages can run
  code at install or import time.
- **Credential exposure** — passing HuggingFace tokens on the command line
  leaks them to shell history and process listings.
- **Path traversal** — user-controlled output/cache/input paths can escape the
  intended workspace.
- **Cache poisoning** — an attacker with write access to the lens cache can
  corrupt matrices, exhaust memory, or attempt symlink-based reads.

## Secure usage

1. **Only load models you trust.** The toolkit ships with a default allowlist of
   known public models. To load an unlisted model you must explicitly opt in
   (`--allow-unlisted-model`) and provide a pinned revision
   (`--model-revision`).
2. **Do not pass `--hf-token` on the command line.** The toolkit does not
   accept this argument. Set the `HF_TOKEN` environment variable, or run
   `huggingface-cli login` before using gated models.
3. **Run inside a sandbox.** For untrusted models or automated jobs, use a
   container or VM with no access to cloud credentials, `~/.ssh`, or other
   sensitive host paths.
4. **Keep dependencies pinned.** Install from the committed `uv.lock` or
   `requirements.txt` with hashes. Review the output of `pip-audit` before
   updating.
5. **Keep outputs inside the workspace.** Use `--workspace` to confine output,
   cache, and corpus paths. The toolkit rejects paths that escape the workspace
   or traverse symlinks outside it.

## Reporting vulnerabilities

If you discover a security issue, please email the maintainers privately at
**security@example.com** with a clear description and reproduction steps. Do
not open a public issue until a fix has been released.
