from __future__ import annotations

from tools.memorisation_probe import changed_fraction, rename, rename_js

JS = """async function publish (registry, opts = {}) {
  const { token, otp: code } = opts
  // token must be valid before publishing
  let tries = 0
  for (const url of registry.urls) {
    const res = await fetch(url, { token, headers: { 'x-token': token } })
    tries += 1
    if (res.ok) return res.body.token
  }
  items.forEach(item => log(item, tries, "tries"))
  return code
}"""


def test_js_locals_are_renamed_consistently():
    out = rename_js(JS)
    assert out is not None
    for old in ("registry", "opts", "tries", "url", "res", "item"):
        assert f" {old} " not in out.replace("(", " ").replace(")", " ").replace(",", " ")
    assert out.count("async function publish (") == 1


def test_js_strings_comments_properties_and_keys_are_untouched():
    out = rename_js(JS)
    assert "// token must be valid before publishing" in out
    assert "'x-token'" in out and '"tries"' in out
    assert ".urls" in out and "res.body.token" not in out and ".body.token" in out
    assert "headers:" in out
    assert "{ token, headers" not in out or "token:" not in out
    assert "await fetch(" in out and "log(" in out


def test_js_without_locals_is_not_renamable():
    assert rename_js("function f () { return g(h) }") is None


def test_python_methods_are_dedented_before_renaming():
    method = ("    def score(self, items):\n"
              "        total = 0\n"
              "        for item in items:\n"
              "            total += item.weight\n"
              "        return total\n")
    out = rename("recommend/model.py::score#10", method)
    assert out is not None and "total" not in out and "item.weight" not in out
    assert ".weight" in out and "def score(self" in out


def test_changed_fraction_counts_identifier_tokens():
    assert changed_fraction("a = b + c", "x = b + y") == 2 / 3
    assert changed_fraction("a", "a b") == 0.0
