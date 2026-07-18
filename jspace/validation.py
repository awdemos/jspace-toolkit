"""Path validation helpers for CLI tools."""

from __future__ import annotations

from pathlib import Path

from jspace import JSpaceError


def validate_workspace(path: Path | str) -> Path:
    """Resolve and validate a workspace root directory."""
    workspace = Path(path)
    if not workspace.exists():
        raise JSpaceError(f"Workspace does not exist: {workspace}")
    if not workspace.is_dir():
        raise JSpaceError(f"Workspace is not a directory: {workspace}")
    return workspace.resolve()


def validate_path(
    path: Path | str,
    workspace: Path | str,
    *,
    must_exist: bool = False,
    must_be_file: bool = False,
    must_be_directory: bool = False,
) -> Path:
    """Resolve ``path`` and ensure it is contained within ``workspace``.

    Symlinks are followed by :meth:`Path.resolve`; if they escape the
    workspace the validation fails. Parent directories for output paths must
    already exist and also be inside the workspace.
    """
    workspace = validate_workspace(workspace)
    raw = Path(path)

    # If the path exists, resolve strictly to surface broken symlinks.
    if raw.exists():
        resolved = raw.resolve(strict=True)
    else:
        # For not-yet-created output paths, resolve the parent and append the
        # name so that ``..`` components still collapse to an absolute path.
        parent = raw.parent.resolve(strict=False)
        resolved = (parent / raw.name).resolve(strict=False)

    # Ensure containment inside the workspace.
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise JSpaceError(
            f"Path {path!r} resolves to {str(resolved)!r}, which is outside "
            f"the workspace {str(workspace)!r}."
        ) from exc

    # Validate the parent directory for not-yet-existing paths.
    if not resolved.exists():
        parent = resolved.parent
        try:
            parent.relative_to(workspace)
        except ValueError as exc:
            raise JSpaceError(
                f"Parent directory of {path!r} is outside the workspace."
            ) from exc
        if not parent.is_dir():
            raise JSpaceError(
                f"Parent directory does not exist or is not a directory: {parent}"
            )

    if must_exist and not resolved.exists():
        raise JSpaceError(f"Path does not exist: {resolved}")
    if must_be_file and not resolved.is_file():
        raise JSpaceError(f"Path is not a file: {resolved}")
    if must_be_directory and not resolved.is_dir():
        raise JSpaceError(f"Path is not a directory: {resolved}")

    return resolved
