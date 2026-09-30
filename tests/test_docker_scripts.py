from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def fake_docker(tmp_path: Path, behaviour: str) -> Path:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    body = {"hang": "sleep 30\n", "down": "echo 'Cannot connect to the Docker daemon' >&2; exit 1\n"}
    script = bindir / "docker"
    script.write_text("#!/usr/bin/env sh\n" + body[behaviour], encoding="utf-8", newline="\n")
    script.chmod(0o755)
    return bindir


def posix_bash() -> str | None:
    if sys.platform == "win32":
        git = shutil.which("git")
        if git:
            for parent in Path(git).parents:
                candidate = parent / "bin" / "bash.exe"
                if candidate.exists() and "system32" not in str(candidate).lower():
                    return str(candidate)
        return None
    return shutil.which("bash")


@pytest.mark.skipif(posix_bash() is None, reason="needs a POSIX bash")
@pytest.mark.parametrize("behaviour", ["hang", "down"])
def test_docker_pipeline_sh_says_how_to_start_docker(tmp_path, behaviour):
    bindir = fake_docker(tmp_path, behaviour)
    env = dict(os.environ, PATH=f"{bindir}{os.pathsep}{os.environ['PATH']}",
               DOCKER_INFO_TIMEOUT="2", DOCKER_START_WAIT="0")
    started = time.monotonic()
    proc = subprocess.run([posix_bash(), (ROOT / "tools" / "docker_pipeline.sh").as_posix(), "--skip-build"],
                          capture_output=True, text=True, env=env, timeout=25)
    assert time.monotonic() - started < 20, "must not hang on docker info"
    assert proc.returncode != 0
    out = proc.stdout + proc.stderr
    assert "Docker is not running" in out or "not answering" in out
    assert "Start Docker Desktop" in out


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell script")
def test_docker_pipeline_ps1_detects_a_hanging_engine_quickly():
    text = (ROOT / "tools" / "docker_pipeline.ps1").read_text(encoding="utf-8")
    assert "Wait-Job" in text and "InfoTimeout" in text
    assert "Start Docker Desktop" in text
