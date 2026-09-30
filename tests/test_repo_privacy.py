from __future__ import annotations

from pipeline.repo import extract_repo, scan_repo

FN = ("function handler (req) {\n  const user = lookup(req.id)\n"
      "  return render(user)\n}\n")


def make(tmp_path, files: dict[str, str]):
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path


def test_secret_files_are_never_indexed(tmp_path):
    root = make(tmp_path, {
        "app.js": FN,
        ".env.js": FN, ".env": "API_KEY=abc", "config/.env.local": "X=1",
        "certs/server.pem.js": FN, "keys/deploy.key": "x",
        "credentials.js": FN, "secrets.py": "def f():\n    return 'placeholder value here'\n",
        "service-account.json": "{}", "id_rsa": "x",
    })
    result = scan_repo(root)
    assert set(result.corpus) == {"app.js::handler#1"}
    assert result.skipped_files["secret file"] >= 3


def test_snippets_containing_secrets_are_dropped(tmp_path):
    root = make(tmp_path, {
        "a.py": "def connect():\n    key = 'AKIAIOSFODNN7EXAMPLE'\n    return client(key)\n",
        "b.js": "function auth () {\n  const token = 'ghp_" + "a" * 36 + "'\n  return token\n}\n",
        "c.py": ("def private():\n    pem = '''-----BEGIN RSA PRIVATE KEY-----\nMIIE\n"
                 "-----END RSA PRIVATE KEY-----'''\n    return pem\n"),
        "d.py": "def settings():\n    api_key = \"" + "x" * 32 + "\"\n    return api_key\n",
        "ok.py": "def add(a, b):\n    # the sum of two numbers, nothing secret\n    return a + b\n",
    })
    result = scan_repo(root)
    assert set(result.corpus) == {"ok.py::add#1"}
    assert result.skipped_snippets["contains a secret"] == 4


def test_virtualenvs_notebooks_lockfiles_minified_and_builds_are_skipped(tmp_path):
    root = make(tmp_path, {
        "src/app.js": FN,
        "myenv/pyvenv.cfg": "home = /usr/bin",
        "myenv/lib/site.py": "def f(x):\n    return x * 2 + 1 + x\n",
        "analysis.ipynb": "{}", "package-lock.json": "{}", "poetry.lock": "",
        "static/app.min.js": FN, "static/vendor.bundle.js": FN,
        "static/packed.js": "var a=1;" * 2000,
        ".next/server.js": FN, "dist/out.js": FN, ".turbo/x.js": FN,
    })
    result = scan_repo(root)
    assert set(result.corpus) == {"src/app.js::handler#1"}
    assert result.skipped_dirs["virtual environment"] == 1
    assert result.skipped_dirs["build output"] == 3
    assert result.skipped_files["minified or generated"] == 3
    assert result.skipped_files["not source code (docs, data, lockfiles, notebooks)"] >= 3
    assert result.languages == {"javascript": 1}


def test_extract_repo_still_returns_just_the_corpus(tmp_path):
    root = make(tmp_path, {"app.js": FN, ".env": "SECRET=1"})
    assert list(extract_repo(root)) == ["app.js::handler#1"]


def test_other_languages_are_indexed_as_windows_when_asked(tmp_path):
    java = "public class Auth {\n  boolean check(String pw) {\n    return hash(pw).equals(stored);\n  }\n}\n"
    root = make(tmp_path, {"Auth.java": java, "main.go": "package main\n\nfunc main() {\n\tprintln(\"hello world from go\")\n}\n",
                           "app.js": FN, ".env": "X=1"})
    assert set(scan_repo(root).corpus) == {"app.js::handler#1"}
    wide = scan_repo(root, all_languages=True)
    assert "app.js::handler#1" in wide.corpus
    assert any(k.startswith("Auth.java::window#") for k in wide.corpus)
    assert any(k.startswith("main.go::window#") for k in wide.corpus)
    assert wide.languages.get("java") == 1 and wide.languages.get("go") == 1
    assert not any(".env" in k for k in wide.corpus)
