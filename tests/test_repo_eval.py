from __future__ import annotations

import json
from pathlib import Path

import cli
from bench.repo_eval import evaluate_repo, leading_comment, pseudo_queries
from pipeline.repo import extract_repo
from tests.test_tool import FakeEmbedder

SRC = """\
// read the yaml configuration file from disk and parse it
const loadConfig = (path) => {
  const raw = fs.readFileSync(path, 'utf8')
  return yaml.parse(raw)
}

// compute the greatest common divisor of two integers
function gcd (a, b) {
  while (b) { [a, b] = [b, a % b] }
  return a
}

class Registry {
  // send the package tarball to the remote registry server
  publish (tarball) {
    return this.client.put(this.url, tarball)
  }
}
"""


def project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "lib.js").write_text(SRC, encoding="utf-8")
    return root


def test_leading_comment_forms():
    lines = ["/**", " * Parse the thing.", " * @param x the input", " */", "", "function f (x) {"]
    assert leading_comment(lines, 5) == "Parse the thing."
    assert leading_comment(["// a b", "// c d", "const g = () => {"], 2) == "a b c d"
    assert leading_comment(["x = 1", "function h () {"], 1) == ""


def test_pseudo_queries_drop_leaked_comments(tmp_path):
    root = project(tmp_path)
    queries, stats = pseudo_queries(root, extract_repo(root))
    answers = {q.answer.split("::")[1].split("#")[0] for q in queries}
    assert answers == {"loadConfig", "gcd"}
    assert stats["leaked"] == 1


def test_eval_repo_end_to_end(tmp_path):
    root = project(tmp_path)
    qfile = tmp_path / "q.json"
    qfile.write_text(json.dumps({"queries": [
        {"query": "upload a package to the registry", "answer": "lib.js::publish"}]}),
        encoding="utf-8")
    report = evaluate_repo(root, tmp_path / "e.db", qfile, FakeEmbedder(), results=tmp_path)
    assert report["handwritten"] == 1 and not report["handwritten_missing"]
    assert set(report["table"]["all"]) == {
        "shipped (code view, dense)", "nl view, dense", "multi-view (max of nl, code)",
        "code view + BM25 (RRF)"}
    assert (tmp_path / "repo_eval.md").read_text(encoding="utf-8").startswith("# Held-out retrieval on real code")


def test_eval_repo_is_a_subcommand():
    args = cli.build_parser().parse_args(["eval-repo", "some/dir", "--queries", "q.json"])
    assert args.command == "eval-repo" and args.queries == Path("q.json")
    assert args.threads == 8
