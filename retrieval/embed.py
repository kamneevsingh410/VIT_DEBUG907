from __future__ import annotations

import math
import struct
from dataclasses import dataclass

MODEL_NAME = "Alibaba-NLP/gte-modernbert-base"
MAX_SEQ_LENGTH = 1024


@dataclass(frozen=True)
class ModelSpec:

    name: str
    trust_remote_code: bool = False
    query_prefix: str = ""
    doc_prefix: str = ""
    note: str = ""
    revision: str | None = None


MODELS: dict[str, ModelSpec] = {
    "jina": ModelSpec(
        "jinaai/jina-embeddings-v2-base-code", trust_remote_code=True,
        note="current baseline; needs transformers<5"),
    "gte-modernbert": ModelSpec(
        "Alibaba-NLP/gte-modernbert-base", trust_remote_code=False,
        note="ModernBERT backbone, needs transformers>=4.48 (have 4.56.2)",
        revision="e7f32e3c00f91d699e8c43b53106206bcc72bb22"),
    "granite-r2": ModelSpec(
        "ibm-granite/granite-embedding-english-r2", trust_remote_code=False,
        note="IBM Granite English R2, 149M params"),
}

QUANTIZE = False
_MODEL = None
_LOADED_KEY: tuple[str, int, bool] | None = None


def set_threads(n: int | None = None) -> int:
    import os

    import torch
    count = n or os.cpu_count() or 1
    torch.set_num_threads(count)
    return torch.get_num_threads()


class EmbedderUnavailable(RuntimeError):
    pass


MODEL_DOWNLOAD_MB = 300

ALLOW_PATTERNS = ["*.json", "*.txt", "*.safetensors", "*.model", "1_Pooling/*",
                  "2_Normalize/*"]
IGNORE_PATTERNS = ["onnx/*", "openvino/*", "*.onnx", "*.bin", "*.h5", "*.msgpack",
                   "*.ot"]
REQUIRED_FILES = ["modules.json", "config.json"]
REQUIRED_ANY = [("model.safetensors", "pytorch_model.bin"),
                ("tokenizer.json", "tokenizer_config.json")]


def configure_hub_environment(platform: str | None = None,
                              environ: dict | None = None) -> None:
    import os
    import sys
    platform = platform or sys.platform
    environ = os.environ if environ is None else environ
    if platform == "win32":
        environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")


def _snapshot_download(repo_id, **kwargs):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id, **kwargs)


def _symlinks_supported() -> bool:
    try:
        from huggingface_hub.file_download import are_symlinks_supported
        return bool(are_symlinks_supported())
    except Exception:
        return True


def _plain_files_dir(name: str):
    from pathlib import Path
    from huggingface_hub.constants import HF_HOME
    return Path(HF_HOME) / "debug907-models" / name.replace("/", "--")


def _spec(name: str) -> "ModelSpec | None":
    return next((m for m in MODELS.values() if m.name == name), None)


def _is_complete(folder) -> bool:
    from pathlib import Path
    folder = Path(folder)
    return (all((folder / f).exists() for f in REQUIRED_FILES)
            and all(any((folder / f).exists() for f in group) for group in REQUIRED_ANY))


def _is_symlink_refusal(exc: BaseException) -> bool:
    return (getattr(exc, "winerror", None) == 1314 or getattr(exc, "errno", None) == 1314
            or "symlink" in str(exc).lower() or "privilege is not held" in str(exc).lower())


def _is_ssl_error(exc: BaseException) -> bool:
    import ssl
    seen = exc
    while seen is not None:
        if isinstance(seen, ssl.SSLError) or "SSLError" in type(seen).__name__ \
                or "CERTIFICATE_VERIFY_FAILED" in str(seen):
            return True
        seen = seen.__cause__ or seen.__context__
    return False


def _find_local(name: str, revision: str | None):
    plain = _plain_files_dir(name)
    if _is_complete(plain):
        return plain
    try:
        path = _snapshot_download(name, revision=revision, local_files_only=True,
                                  allow_patterns=ALLOW_PATTERNS, ignore_patterns=IGNORE_PATTERNS)
    except Exception:
        return None
    return path if _is_complete(path) else None


def ensure_model(name: str = MODEL_NAME, say=print):
    import os

    spec = _spec(name)
    revision = spec.revision if spec else None
    local = _find_local(name, revision)
    if local is not None:
        return local
    if os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}:
        raise EmbedderUnavailable(
            f"the model {name} is not downloaded yet and offline mode is on "
            "(HF_HUB_OFFLINE=1). Run once with internet access, or unset HF_HUB_OFFLINE.")
    say(f"First run: downloading the model {name} (~{MODEL_DOWNLOAD_MB} MB, one time only) ...")
    patterns = {"allow_patterns": ALLOW_PATTERNS, "ignore_patterns": IGNORE_PATTERNS}
    try:
        from huggingface_hub.utils import logging as hub_logging
        hub_logging.set_verbosity_error()
    except Exception:
        pass
    try:
        if not _symlinks_supported():
            return _snapshot_download(name, revision=revision, local_dir=_plain_files_dir(name),
                                      **patterns)
        try:
            return _snapshot_download(name, revision=revision, **patterns)
        except OSError as exc:
            if not _is_symlink_refusal(exc):
                raise
            say("  (Windows refused a symlink; downloading as plain files instead)")
            return _snapshot_download(name, revision=revision, local_dir=_plain_files_dir(name),
                                      **patterns)
    except EmbedderUnavailable:
        raise
    except Exception as exc:
        if _is_ssl_error(exc):
            raise EmbedderUnavailable(
                "could not verify the download site's SSL certificate (often a "
                "campus or corporate network intercepting HTTPS). Try a different "
                "network or a phone hotspot.") from exc
        raise EmbedderUnavailable(
            f"could not download {name}: no internet connection? The first run needs "
            f"internet once (~{MODEL_DOWNLOAD_MB} MB); after that it works offline. "
            f"({type(exc).__name__})") from exc


def model_is_cached(name: str = MODEL_NAME) -> bool:
    spec = _spec(name)
    return _find_local(name, spec.revision if spec else None) is not None


def check_environment() -> tuple[bool, str]:
    try:
        import transformers
    except ImportError:
        return False, "transformers is not installed (pip install -r requirements.txt)"
    try:
        import sentence_transformers
    except ImportError:
        return False, "sentence-transformers is not installed"

    version = getattr(transformers, "__version__", "0")
    major = int(version.split(".")[0]) if version[:1].isdigit() else 0
    minor = int(version.split(".")[1]) if "." in version else 0
    if major >= 5:
        return False, (
            f"transformers {version} is too new: jina-v2 (kept as an ablation "
            "baseline) has remote code importing find_pruneable_heads_and_indices, "
            "removed in 5.x, and sentence-transformers 6.x pins transformers>=5. "
            'Fix: pip install "transformers<5" "sentence-transformers<5"'
        )
    if major == 4 and minor < 48:
        return False, (
            f"transformers {version} is too old for {MODEL_NAME}: the ModernBERT "
            'backbone landed in 4.48. Fix: pip install "transformers>=4.48,<5"'
        )
    return True, (f"transformers {version}, "
                  f"sentence-transformers {getattr(sentence_transformers, '__version__', '?')}")


def load_model(name: str = MODEL_NAME, max_seq_length: int = MAX_SEQ_LENGTH,
               quantize: bool = QUANTIZE):
    global _MODEL, _LOADED_KEY
    key = (name, max_seq_length, quantize)
    if _MODEL is not None and _LOADED_KEY == key:
        return _MODEL

    ok, message = check_environment()
    if not ok:
        raise EmbedderUnavailable(message)

    from sentence_transformers import SentenceTransformer
    spec = next((m for m in MODELS.values() if m.name == name), None)
    trust = spec.trust_remote_code if spec else False
    path = ensure_model(name)
    try:
        model = SentenceTransformer(str(path), trust_remote_code=trust, device="cpu",
                                    local_files_only=True)
    except Exception as exc:
        raise EmbedderUnavailable(f"could not load {name}: "
                                  f"{type(exc).__name__}: {exc}") from exc
    model.max_seq_length = max_seq_length
    if quantize:
        import torch
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8)
        model.max_seq_length = max_seq_length
    _MODEL, _LOADED_KEY = model, key
    return model


def available() -> bool:
    try:
        load_model()
    except EmbedderUnavailable:
        return False
    return True


CACHE_PATH = "out/embed_cache.db"


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds} s"
    minutes, s = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} min {s:02d} s"
    hours, m = divmod(minutes, 60)
    return f"{hours} h {m:02d} min"


def progress_line(done: int, total: int, elapsed: float) -> str:
    rate = done / elapsed if elapsed > 0 else 0.0
    left = (total - done) / rate if rate > 0 else 0.0
    tail = "done" if done >= total else f"about {format_duration(left)} left"
    return (f"encoded {done:,} / {total:,} texts ({100 * done // max(total, 1)}%)  "
            f"{rate:.1f} texts/s  {tail}")


class TextCache:

    def __init__(self, path: str = CACHE_PATH) -> None:
        import pathlib
        import sqlite3
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "  sig TEXT NOT NULL, key TEXT NOT NULL, vector BLOB NOT NULL,"
            "  PRIMARY KEY (sig, key))")
        self.conn.commit()

    @staticmethod
    def key(text: str) -> str:
        import hashlib
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_many(self, sig: str, keys: list[str]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for i in range(0, len(keys), 500):
            chunk = keys[i:i + 500]
            marks = ",".join("?" * len(chunk))
            for k, blob in self.conn.execute(
                    f"SELECT key, vector FROM cache WHERE sig = ? AND key IN ({marks})",
                    [sig, *chunk]):
                out[k] = unpack(blob)
        return out

    def put_many(self, sig: str, items: list[tuple[str, list[float]]]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO cache (sig, key, vector) VALUES (?,?,?)",
            [(sig, k, pack(v)) for k, v in items])
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


_CACHE: "TextCache | None" = None


def get_cache(path: str = CACHE_PATH) -> "TextCache":
    global _CACHE
    if _CACHE is None:
        _CACHE = TextCache(path)
    return _CACHE


@dataclass
class Embedder:

    name: str = MODEL_NAME
    max_seq_length: int = MAX_SEQ_LENGTH
    batch_size: int = 16
    quantize: bool = QUANTIZE
    cache: bool | str = True
    prefix: str = ""

    @property
    def signature(self) -> str:
        base = f"{self.name}|seq{self.max_seq_length}|int8={self.quantize}"
        return f"{base}|pfx={self.prefix}" if self.prefix else base

    def encode(self, texts: list[str], *, show_progress: bool = False) -> list[list[float]]:
        if not texts:
            return []
        self.last_cache_hits, self.last_model_encoded = 0, len(texts)
        if not self.cache:
            return self._encode_raw(texts, show_progress)

        store = get_cache()
        keys = [TextCache.key(t) for t in texts]
        if self.cache == "write":
            fresh = self._encode_raw(texts, show_progress)
            store.put_many(self.signature, list(dict(zip(keys, fresh)).items()))
            return fresh
        hits = store.get_many(self.signature, list(dict.fromkeys(keys)))
        missing_idx = [i for i, k in enumerate(keys) if k not in hits]
        self.last_cache_hits = len(texts) - len(missing_idx)
        self.last_model_encoded = len(missing_idx)
        if show_progress and hits:
            print(f"  vector cache: {len(texts) - len(missing_idx)}/{len(texts)} hit")
        if missing_idx:
            hits.update(self._encode_resumable(texts, keys, missing_idx, store, show_progress))
        return [hits[k] for k in keys]

    RESUME_STEP_BATCHES = 16

    def _encode_resumable(self, texts, keys, missing_idx, store, show_progress):
        import time as _time
        unique: list[int] = []
        seen: set[str] = set()
        for i in sorted(missing_idx, key=lambda i: -len(texts[i])):
            if keys[i] not in seen:
                seen.add(keys[i])
                unique.append(i)
        step = max(1, self.batch_size) * self.RESUME_STEP_BATCHES
        done: dict[str, list[float]] = {}
        started = _time.perf_counter()
        several = len(unique) > step
        for lo in range(0, len(unique), step):
            part = unique[lo:lo + step]
            fresh = self._encode_raw([texts[i] for i in part], show_progress and not several)
            items = [(keys[i], v) for i, v in zip(part, fresh)]
            store.put_many(self.signature, items)
            done.update(items)
            if show_progress and several:
                print("  " + progress_line(len(done), len(unique), _time.perf_counter() - started),
                      flush=True)
        return done

    def _encode_raw(self, texts: list[str], show_progress: bool) -> list[list[float]]:
        model = load_model(self.name, self.max_seq_length, self.quantize)
        if self.prefix:
            texts = [self.prefix + t for t in texts]
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return [[float(x) for x in row] for row in vectors]

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]

    def warm(self) -> None:
        load_model(self.name, self.max_seq_length, self.quantize)

    @property
    def dimension(self) -> int:
        return int(load_model(self.name, self.max_seq_length, self.quantize)
                   .get_sentence_embedding_dimension())


def pack(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return float(sum(a * b for a, b in zip(left, right)))


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        return list(vector)
    return [value / norm for value in vector]


def mean_pool(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dims = len(vectors[0])
    total = [0.0] * dims
    for vector in vectors:
        for i, value in enumerate(vector):
            total[i] += value
    return normalize([value / len(vectors) for value in total])
