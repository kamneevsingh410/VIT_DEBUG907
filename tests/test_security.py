from __future__ import annotations

import inspect

from retrieval import embed


def test_unknown_model_never_trusts_remote_code():
    src = inspect.getsource(embed.load_model)
    assert "if spec else False" in src
    assert "if spec else True" not in src


def test_only_jina_is_allowed_remote_code():
    trusted = {k for k, v in embed.MODELS.items() if v.trust_remote_code}
    assert trusted == {"jina"}, trusted


def test_shipped_model_runs_without_remote_code():
    shipped = next(v for v in embed.MODELS.values() if v.name == embed.MODEL_NAME)
    assert shipped.trust_remote_code is False


def test_fts_query_quotes_and_filters_user_terms():
    from retrieval.sparse import build_match_query
    q = build_match_query(["ok", 'x" OR 1', "NEAR", "a*", "drop;table", "fine_2"])
    assert q == '"ok" OR "fine_2"'
