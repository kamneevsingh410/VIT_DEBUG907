from __future__ import annotations

import pytest

from retrieval.router import MAX_WORDS, Vocabulary, clean, route, variants

USAGE = ["where is otplease used", "Where is otplease used?", "who uses otplease",
         "what uses otplease", "which files use otplease", "find usages of otplease",
         "show me all references to otplease", "where do we reference otplease",
         "usages of otplease", "where is otplease referenced anywhere", "how is otplease used"]
CALLERS = ["who calls otplease", "what calls otplease?", "which functions call otplease",
           "where is otplease called", "where is otplease called from",
           "find the callers of otplease", "callers of otplease", "show call sites of otplease"]
DEFINITION = ["where is otplease defined", "where is otplease implemented",
              "show me the definition of otplease", "go to the implementation of otplease",
              "definition of otplease", "where does otplease come from"]
ORDER = [("which functions call validate before save", "before"),
         ("what code calls validate before calling save", "before"),
         ("does validate get called before save", "before"),
         ("is validate called after save", "after"),
         ("where is validate called after save", "after"),
         ("find the functions that call validate before save", "before")]
SEARCH = ["how is the registry login token checked", "where is the session token checked",
          "find the length of the longest increasing subsequence",
          "parse the config file and merge it with the environment",
          "how do I use this", "what does this function do", "reverse a linked list",
          "given n integers find where the sum is used as the maximum",
          "You are given an array a of n integers. Find where is the maximum used " * 3]


@pytest.mark.parametrize("q", USAGE)
def test_usage_questions(q):
    r = route(q)
    assert r is not None and r.kind == "usage" and r.phrases == ["otplease"]


@pytest.mark.parametrize("q", CALLERS)
def test_callers_questions(q):
    r = route(q)
    assert r is not None and r.kind == "callers" and r.phrases == ["otplease"]


@pytest.mark.parametrize("q", DEFINITION)
def test_definition_questions(q):
    r = route(q)
    assert r is not None and r.kind == "definition" and r.phrases == ["otplease"]


@pytest.mark.parametrize("q,order", ORDER)
def test_order_questions(q, order):
    r = route(q)
    assert r is not None and r.kind == "order" and r.phrases == ["validate", "save"]
    assert r.order == order


@pytest.mark.parametrize("q", SEARCH)
def test_everything_else_is_a_ranked_search(q):
    assert route(q) is None


def test_long_or_multiline_text_is_never_routed():
    assert route("where is " + " ".join(["x"] * MAX_WORDS) + " used") is None
    assert route("where is otplease used\nand more") is None


def test_filler_words_and_quotes():
    assert clean("the otplease function") == "otplease"
    assert clean('"npm-shrinkwrap.json"') == '"npm-shrinkwrap.json"'
    assert route('find usages of "npm-shrinkwrap.json"').phrases == ['"npm-shrinkwrap.json"']


def test_variants_cover_the_naming_styles():
    v = variants(["bluetooth", "settings", "deeplink"])
    for want in ("bluetoothSettingsDeeplink", "BluetoothSettingsDeeplink", "bluetooth_settings_deeplink",
                 "BLUETOOTH_SETTINGS_DEEPLINK", "bluetooth-settings-deeplink", "bluetoothsettingsdeeplink"):
        assert want in v


CODE = ['BLUETOOTH_SETTINGS_DEEPLINK = "app://settings/bluetooth"',
        "def open_settings(link):\n    return launch(BLUETOOTH_SETTINGS_DEEPLINK)",
        "const getCredentialsByURI = (uri) => creds[uri]",
        "function otplease(npm, opts, fn) { return fn(opts) }",
        "otplease(npm, opts, publish)", "const cfg = loadConfigFile('npmrc')"]


def test_names_in_words_resolve_to_identifiers_in_the_index():
    v = Vocabulary(CODE)
    r = v.resolve("bluetooth settings deeplink")
    assert r.best == "BLUETOOTH_SETTINGS_DEEPLINK" and r.how == "variant"
    assert v.resolve("otplease").how == "exact"
    assert v.resolve("credentials by uri").best == "getCredentialsByURI"
    assert v.resolve("loadConfigFil").best == "loadConfigFile"
    assert v.resolve("'app://settings/bluetooth'").how == "quoted"
    assert v.resolve("nothing like this at all").best is None


def test_the_tool_answers_english_questions_and_says_how_it_read_them(tmp_path, monkeypatch):
    from tests.test_tool import FakeEmbedder, run_tool, indexed_db, sandbox
    code = tmp_path / "project"
    code.mkdir()
    (code / "cfg.py").write_text(
        "def load_config(path):\n    return parse_yaml(open(path))\n\n\n"
        "def reload_all():\n    validate()\n    cfg = load_config('a.yml')\n    save(cfg)\n",
        encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.chdir(tmp_path)
    db = indexed_db(code, out)
    rc, text = run_tool(["where is load config defined", "who calls load_config",
                         "where is load_config used", "which functions call validate before save",
                         "more matches", "open 1", "where is quantum flux used", "help", "quit"],
                        db=db)
    assert rc == 0
    assert "Understood as: the definition of load_config" in text
    assert "Understood as: callers of load_config()" in text
    assert "Understood as: usages of load_config" in text
    assert "Understood as: functions that call validate() before save()" in text
    assert "no name like 'quantum flux' is in this index - searching instead" in text
    assert "Ask in plain English" in text
    assert "Traceback" not in text


def test_search_shortcut_never_routes(tmp_path, monkeypatch):
    from tests.test_tool import run_tool, indexed_db
    code = tmp_path / "project"
    code.mkdir()
    (code / "a.py").write_text("def load_config(path):\n    return open(path)\n", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.chdir(tmp_path)
    rc, text = run_tool([":search where is load_config used", "quit"], db=indexed_db(code, out))
    assert rc == 0 and "Understood as" not in text and "results" in text


def test_callable_questions_never_resolve_to_a_string_literal():
    v = Vocabulary(['log.silly("load actual")', "function loadActual() {}", "loadActual()"])
    assert v.resolve("load actual").best == "load actual"
    assert v.resolve("load actual", identifiers_only=True).best == "loadActual"
