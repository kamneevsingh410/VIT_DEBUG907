from __future__ import annotations

import getpass
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", "out", "external", "debug907", "__pycache__",
             ".pytest_cache", "node_modules"}
TEXT = {".py", ".md", ".ps1", ".sh", ".cmd", ".toml", ".txt", ".json", ".cfg"}


def _this_user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return ""


_USER = _this_user()
MACHINE_PATH = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]|/home/\w|/Users/\w"
                          + (rf"|[\\/]{re.escape(_USER)}[\\/]" if len(_USER) >= 3 else ""))


def test_the_guard_itself_catches_windows_and_posix_paths():
    assert MACHINE_PATH.search(r'-File "C:\Users\someone\repo\tools\x.ps1"')
    assert MACHINE_PATH.search(r"cd C:\Users\someone\repo")
    assert MACHINE_PATH.search("/home/someone/repo") and MACHINE_PATH.search("C:/Users/x")
    assert not MACHINE_PATH.search("pause OneDrive sync before long runs")


def source_files():
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts) or not path.is_file():
            continue
        if path.suffix in TEXT or path.name in {"Dockerfile", ".dockerignore"}:
            yield path


def test_no_machine_specific_paths():
    offenders = []
    for path in source_files():
        if path.name == "test_portability.py":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for n, line in enumerate(text.splitlines(), 1):
            if MACHINE_PATH.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{n}: {line.strip()[:80]}")
    assert not offenders, "\n".join(offenders)


def test_every_windows_script_has_a_posix_twin():
    missing = []
    for script in list(ROOT.glob("*.ps1")) + list(ROOT.glob("tools/*.ps1")):
        if not script.with_suffix(".sh").exists():
            missing.append(str(script.relative_to(ROOT)))
    if (ROOT / "debug907.cmd").exists() and not (ROOT / "debug907.sh").exists():
        missing.append("debug907.cmd")
    assert not missing, missing


def test_shell_scripts_and_the_dockerfile_have_lf_line_endings_on_disk():
    root = Path(__file__).resolve().parent.parent
    skip = {"out", "external", ".venv", "venv", ".git", "node_modules", "debug907"}
    files = [p.relative_to(root).as_posix() for p in root.rglob("*.sh")
             if not skip & set(p.relative_to(root).parts)]
    files += ["Dockerfile"] if (root / "Dockerfile").exists() else []
    assert files, "no shell scripts found"
    bad = [f for f in files if b"\r" in (root / f).read_bytes()]
    assert not bad, f"CRLF line endings: {bad}"
