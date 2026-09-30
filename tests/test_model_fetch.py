from __future__ import annotations

import ssl
from pathlib import Path

import pytest

from retrieval import embed
from retrieval.embed import MODEL_NAME, EmbedderUnavailable, ensure_model

NEEDED = ["modules.json", "config.json", "config_sentence_transformers.json",
          "model.safetensors", "tokenizer.json", "tokenizer_config.json",
          "special_tokens_map.json", "1_Pooling/config.json"]


class FakeHub:

    def __init__(self, root: Path, cached=False, partial=False, fail=None,
                 fail_once_with=None):
        self.root, self.fail, self.fail_once_with = root, fail, fail_once_with
        self.calls: list[dict] = []
        self.snapshot = root / "hub" / "snapshot"
        if cached or partial:
            self._write(self.snapshot, NEEDED if cached else ["model.safetensors", "README.md"])

    @staticmethod
    def _write(folder: Path, names):
        for name in names:
            (folder / name).parent.mkdir(parents=True, exist_ok=True)
            (folder / name).write_text("x", encoding="utf-8")

    def __call__(self, repo_id, revision=None, local_files_only=False, local_dir=None,
                 allow_patterns=None, ignore_patterns=None, **kw):
        from huggingface_hub.errors import LocalEntryNotFoundError
        self.calls.append({"revision": revision, "local_files_only": local_files_only,
                           "local_dir": local_dir, "allow": allow_patterns,
                           "ignore": ignore_patterns})
        target = Path(local_dir) if local_dir else self.snapshot
        if local_files_only:
            if target.exists():
                return str(target)
            raise LocalEntryNotFoundError("not cached")
        if self.fail_once_with is not None:
            exc, self.fail_once_with = self.fail_once_with, None
            self._write(target, ["model.safetensors"])
            raise exc
        if self.fail is not None:
            raise self.fail
        self._write(target, NEEDED)
        return str(target)


@pytest.fixture()
def hub(monkeypatch, tmp_path):
    def install(symlinks=True, **kw):
        fake = FakeHub(tmp_path, **kw)
        monkeypatch.setattr(embed, "_snapshot_download", fake)
        monkeypatch.setattr(embed, "_plain_files_dir", lambda name: tmp_path / "plain" / "m")
        monkeypatch.setattr(embed, "_symlinks_supported", lambda: symlinks)
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        return fake
    return install


def downloads(fake):
    return [c for c in fake.calls if not c["local_files_only"]]


def test_cached_model_touches_no_network(hub, capsys):
    fake = hub(cached=True)
    ensure_model(MODEL_NAME)
    assert downloads(fake) == []
    assert "downloading" not in capsys.readouterr().out


def test_missing_model_is_downloaded_once_pinned_and_without_onnx(hub, capsys):
    fake = hub()
    path = ensure_model(MODEL_NAME)
    out = capsys.readouterr().out
    assert "First run: downloading the model" in out and "one time only" in out
    (download,) = downloads(fake)
    assert download["revision"] == embed.MODELS["gte-modernbert"].revision
    assert "onnx/*" in download["ignore"] and "*.safetensors" in download["allow"]
    assert (Path(path) / "modules.json").exists()


def test_partial_snapshot_is_not_treated_as_cached(hub):
    fake = hub(partial=True)
    path = ensure_model(MODEL_NAME)
    assert len(downloads(fake)) == 1
    assert (Path(path) / "tokenizer.json").exists()


def test_windows_symlink_refusal_falls_back_and_is_found_next_time(hub):
    refusal = OSError(1314, "A required privilege is not held by the client", "x", 1314)
    fake = hub(fail_once_with=refusal)
    path = ensure_model(MODEL_NAME)
    assert fake.calls[-1]["local_dir"] is not None and str(path) == str(fake.calls[-1]["local_dir"])
    before = len(downloads(fake))
    assert str(ensure_model(MODEL_NAME)) == str(path)
    assert len(downloads(fake)) == before


def test_no_symlink_support_goes_straight_to_plain_files(hub):
    fake = hub(symlinks=False)
    ensure_model(MODEL_NAME)
    (download,) = downloads(fake)
    assert download["local_dir"] is not None


def test_offline_with_no_cache_is_one_friendly_line(hub, monkeypatch):
    hub()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(EmbedderUnavailable, match="not downloaded yet"):
        ensure_model(MODEL_NAME)


def test_no_internet_on_first_run(hub):
    import requests
    hub(fail=requests.exceptions.ConnectionError("no route"))
    with pytest.raises(EmbedderUnavailable, match="internet connection"):
        ensure_model(MODEL_NAME)


def test_ssl_error_suggests_another_network(hub):
    import requests
    hub(fail=requests.exceptions.SSLError(ssl.SSLError("CERTIFICATE_VERIFY_FAILED")))
    with pytest.raises(EmbedderUnavailable, match="hotspot"):
        ensure_model(MODEL_NAME)


def test_windows_environment_defaults():
    env: dict[str, str] = {}
    embed.configure_hub_environment(platform="win32", environ=env)
    assert env["HF_HUB_DISABLE_SYMLINKS_WARNING"] == "1"
    other: dict[str, str] = {}
    embed.configure_hub_environment(platform="linux", environ=other)
    assert other == {}


def test_every_shipped_model_is_pinned():
    assert embed.MODELS["gte-modernbert"].revision == "e7f32e3c00f91d699e8c43b53106206bcc72bb22"
