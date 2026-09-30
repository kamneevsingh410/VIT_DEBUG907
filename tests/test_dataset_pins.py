from __future__ import annotations

import json
from pathlib import Path

from bench import dataset

ROOT = Path(__file__).resolve().parent.parent


def test_loader_pins_the_revision_mteb_uses():
    art = json.loads((ROOT / "appsretrieval_results.json").read_text())
    assert dataset.APPS_REVISION == art["dataset_revision"]


def test_dockerfile_bakes_the_same_revisions():
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert dataset.APPS_REVISION in dockerfile
    assert dataset.QRELS_REVISION in dockerfile


def test_loader_passes_revision_on_every_load():
    import inspect
    src = inspect.getsource(dataset.load_appsretrieval)
    calls = [ln for ln in src.splitlines() if "load_dataset(" in ln and "import" not in ln]
    joined = "\n".join(src.splitlines())
    assert len(calls) == 3
    assert joined.count("revision=") == 3


def test_docker_bakes_the_pinned_model_revision():
    from pathlib import Path
    from retrieval.embed import MODELS
    rev = MODELS["gte-modernbert"].revision
    assert rev and rev in (Path(__file__).resolve().parent.parent / "Dockerfile").read_text()
