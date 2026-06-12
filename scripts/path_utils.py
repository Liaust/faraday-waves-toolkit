from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def project_path(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def relpath(path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(p)


def add_script_imports(*subdirs: str) -> None:
    for directory in [SCRIPTS_DIR, *(SCRIPTS_DIR / subdir for subdir in subdirs)]:
        if directory.is_dir() and str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
