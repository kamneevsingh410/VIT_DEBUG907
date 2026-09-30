from __future__ import annotations

from pipeline.repo import extract_repo

BODY = ("function parseArgs(argv) {\n  const out = {}\n"
        "  for (const a of argv) out[a] = true\n  return out\n}\n")


def test_generated_snapshot_folders_are_skipped(tmp_path):
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "cli.js").write_text(BODY, encoding="utf-8")
    for snap in ("tap-snapshots/test/lib", "src/__snapshots__"):
        (tmp_path / snap).mkdir(parents=True)
        (tmp_path / snap / "cli.js.test.cjs").write_text(BODY, encoding="utf-8")
    ids = list(extract_repo(tmp_path))
    assert ids and all(i.startswith("lib/") for i in ids), ids


def test_vendored_code_is_skipped_unless_asked(tmp_path):
    (tmp_path / "node_modules" / "dep").mkdir(parents=True)
    (tmp_path / "node_modules" / "dep" / "index.js").write_text(BODY, encoding="utf-8")
    assert list(extract_repo(tmp_path)) == []
    assert list(extract_repo(tmp_path, include_vendored=True))


def test_braces_inside_parameters_do_not_end_the_function(tmp_path):
    src = (
        "const explainEdge = ({ name, type }, depth, chalk) => {\n"
        "  const dep = chalk.bold(name)\n"
        "  return `${dep} ${type} at depth ${depth}`\n"
        "}\n"
        "\n"
        "const shouldPrint = (path, opts = {}) => {\n"
        "  const ext = path.split('.').pop()\n"
        "  return !opts.textOnly || ['js', 'md'].includes(ext)\n"
        "}\n"
        "\n"
        "function collect (items) {\n"
        "  return items.map((x) => {\n"
        "    return { id: x.id, name: x.name }\n"
        "  })\n"
        "}\n")
    (tmp_path / "a.js").write_text(src, encoding="utf-8")
    got = extract_repo(tmp_path)
    by_name = {k.split("::")[1].split("#")[0]: v for k, v in got.items()}
    assert "at depth" in by_name["explainEdge"]
    assert "includes(ext)" in by_name["shouldPrint"]
    assert by_name["collect"].rstrip().endswith("}") and "x.name" in by_name["collect"]
    assert "shouldPrint" not in by_name["explainEdge"]


def test_expression_arrows_without_semicolons_end_where_the_expression_ends(tmp_path):
    src = (
        "const isAsyncFn = (v) => typeof v === 'function' && /^\\[Async:/.test(inspect(v))\n"
        "\n"
        "const loadMock = (t, opts = {}) => _loadMock(t, {\n"
        "  ...opts,\n"
        "  config: { ...opts.config },\n"
        "})\n"
        "\n"
        "const pick = (list) => list\n"
        "  .filter(Boolean)\n"
        "  .map(String)\n"
        "\n"
        "function later (a) {\n"
        "  return a + 1 + unrelatedHelperCall(a)\n"
        "}\n")
    (tmp_path / "b.js").write_text(src, encoding="utf-8")
    by_name = {k.split("::")[1].split("#")[0]: v for k, v in extract_repo(tmp_path).items()}
    assert "loadMock" not in by_name.get("isAsyncFn", "")
    assert by_name["loadMock"].rstrip().endswith("})") and "later" not in by_name["loadMock"]
    assert ".map(String)" in by_name["pick"] and "later" not in by_name["pick"]


def test_python_definitions_come_from_the_syntax_tree_not_docstrings(tmp_path):
    src = (
        "import functools\n"
        "\n"
        "def all_snippets(index):\n"
        '    """Smallest first, so when a\n'
        "    class snippet and its method overlap, the method wins.\n"
        '    def fake(): this is prose, not code"""\n'
        "    return sorted(index)\n"
        "\n"
        "class Store:\n"
        "    @functools.lru_cache\n"
        "    def load(self, key):\n"
        "        return self.data.get(key, None)\n"
    )
    (tmp_path / "w.py").write_text(src, encoding="utf-8")
    got = extract_repo(tmp_path)
    names = sorted(k.split("::")[1] for k in got)
    assert names == ["Store#9", "all_snippets#3", "load#10"]
    assert got["w.py::load#10"].startswith("@functools.lru_cache")
    assert "return sorted(index)" in got["w.py::all_snippets#3"]
