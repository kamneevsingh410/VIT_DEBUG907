from __future__ import annotations

import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import TextIO

from banner import render as render_banner
from retrieval import workflows

HELP = """\
  Ask in plain English. For example:
    how is the registry login token checked      a ranked search
    where is otplease used                       every line that uses it
    who calls getCredentialsByURI                the lines that call it
    where is otplease defined                    where it is defined
    which functions call validate before save    call order inside functions
  Names can be written in words ("the bluetooth settings deeplink" finds
  BLUETOOTH_SETTINGS_DEEPLINK); the tool says how it read the question.

  open 2 / show 2   show result 2 in full        more   the next page
  more matches      the other names a question could have meant
  help / quit       this help / exit (Ctrl+C or Ctrl+D also exit)

  Optional shortcuts:
  :search <text>    always a ranked search, even for "where is X used"
  :open N           show result N in full
  :more             next page of results
  :index <path> [label]  index a folder of code (as version label) and switch to it
  :version [label]  list versions, or search one version only
  :all-versions     search every indexed version together
  :collapse on|off|auto  one row per function across versions (auto: on
                    whenever several versions are searched); a row reads
                    "in v1 = v2 != v3": = same code as before, != changed
  :history N        the versions result N exists in, with a diff per change
  :newest on|off    in a collapsed row, show the newest of the identical
                    versions (today's line numbers); off by default
  :hubness on|off|auto  the hubness correction (auto: on for the
                    AppsRetrieval corpus, as in the submitted result)
  :tests on|off     include test files in results (your own code: hidden by default)
  :uses <term>      every line that mentions a name, URL or string (exact)
  :calls X before Y functions that call X before they call Y (source order)
  :stats            session statistics
  :help             this help
  :quit             exit   (Ctrl+C or Ctrl+D also exit)"""

OUT_DIR = Path("out")
DEFAULT_DB = OUT_DIR / "real.db"
LAST_INDEX_FILE = OUT_DIR / "last_index.txt"
FIXTURE_DBS = {Path("out/index.db"), Path("out/selftest.db")}
PREVIEW_LINES = 4


def is_fixture(db: Path) -> bool:
    target = Path(db).resolve()
    return any(target == fixture.resolve() for fixture in FIXTURE_DBS)


class Tool:

    def __init__(self, db: Path | None = None, *, stdin: TextIO | None = None,
                 stdout: TextIO | None = None, embedder=None, debug: bool = False,
                 page_size: int = 5, cwd: Path | None = None,
                 versions: str | None = None) -> None:
        from rich.console import Console

        self.stdin = stdin or sys.stdin
        self.console = Console(file=stdout or sys.stdout, highlight=False,
                               soft_wrap=True)
        self.requested_db = Path(db) if db else None
        self.embedder = embedder
        self.debug = debug
        self.page_size = page_size
        self.cwd = Path(cwd) if cwd else Path.cwd()
        self.index = None
        self.db: Path | None = None
        self.hits: list[tuple[str, float]] = []
        self.page = 0
        self.last_query = ""
        self.latencies: list[float] = []
        self.version_filter: str | None = versions
        self.result_versions: dict[str, list[str]] = {}
        self.collapse: bool | None = None
        self.show_tests = False
        self.prefer_newest = False
        self.hubness: bool | None = None

    def say(self, text: str = "", style: str | None = None) -> None:
        from retrieval.display import safe
        self.console.print(safe(text), style=style, markup=False, highlight=False)

    def fail(self, exc: BaseException, hint: str | None = None) -> None:
        message = str(exc) or type(exc).__name__
        self.say(f"  error: {message}", style="bold red")
        if hint:
            self.say(f"  try:   {hint}", style="yellow")
        if self.debug:
            self.say(traceback.format_exc())

    def read(self, prompt: str) -> str | None:
        if self.stdin is sys.stdin and sys.stdin.isatty():
            try:
                return input(prompt)
            except EOFError:
                return None
        self.console.print(prompt, end="", markup=False, highlight=False)
        self.console.file.flush()
        line = self.stdin.readline()
        if line == "":
            return None
        return line.rstrip("\r\n")

    def load_model(self) -> None:
        if self.embedder is None:
            from retrieval.embed import Embedder
            self.embedder = Embedder()
        started = time.perf_counter()
        warm = getattr(self.embedder, "warm", None)
        name = getattr(self.embedder, "name", "")
        from retrieval import embed
        if name in {m.name for m in embed.MODELS.values()} and not embed.model_is_cached(name):
            embed.configure_hub_environment()
            embed.ensure_model(name, say=lambda msg: self.say(f"  {msg}", style="yellow"))
        if self.console.is_terminal:
            with self.console.status("  loading model ...", spinner="line"):
                if warm:
                    warm()
        else:
            self.say("  loading model ...")
            if warm:
                warm()
        self.model_seconds = time.perf_counter() - started

    def choose_db(self) -> Path:
        if self.requested_db:
            return self.requested_db
        try:
            remembered = Path(LAST_INDEX_FILE.read_text(encoding="utf-8").strip())
            if workflows.index_is_usable(remembered) and not is_fixture(remembered):
                return remembered
        except OSError:
            pass
        return DEFAULT_DB

    def open_index(self, db: Path, versions: str | None = None) -> None:
        from retrieval.index import IndexUnavailable, load as load_index

        if is_fixture(db):
            raise ValueError(f"{db} is the 30-snippet FIXTURE used by selftest and "
                             "tests, not a real corpus")
        try:
            new = load_index(db, versions=versions)
        except IndexUnavailable as exc:
            raise IndexProblem(str(exc)) from None
        if self.index is not None:
            self.index.close()
        self.index = new
        self.db = db
        self.version_filter = versions
        self.hits = []
        try:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            LAST_INDEX_FILE.write_text(str(db), encoding="utf-8")
        except OSError:
            pass

    def ask_what_to_index(self, target: Path) -> Path | None:
        self.say(f"  no index yet at {target}. What should I index?")
        self.say(f"    [1] a folder of code   (default: {self.cwd})")
        self.say("    [2] the CoIR AppsRetrieval corpus  (8,765 snippets; "
                 "the first run encodes for ~50 min)")
        self.say("    [q] quit")
        choice = self.read("  choice [1]: ")
        if choice is None or choice.strip().lower() in {"q", "quit"}:
            return None
        choice = choice.strip() or "1"
        if choice == "2":
            db = target if self.requested_db else DEFAULT_DB
            self.say("  indexing the AppsRetrieval corpus ...")
            report = workflows.index_apps(db, embedder=self.embedder,
                                          show_progress=self.console.is_terminal)
            self.report_index(report)
            return db
        if choice != "1":
            self.say(f"  unknown choice {choice!r}")
            return self.ask_what_to_index(target)
        folder = self.read(f"  folder [{self.cwd}]: ")
        if folder is None:
            return None
        root = Path(folder.strip().strip('"')) if folder.strip() else self.cwd
        return self.index_folder(root)

    def index_folder(self, root: Path, version: str = "v1") -> Path:
        if not root.is_absolute():
            root = self.cwd / root
        root = root.resolve()
        name = "".join(c if c.isalnum() or c in "-_" else "_" for c in root.name) or "repo"
        db = OUT_DIR / f"{name}.db"
        self.say(f"  indexing {root} as version {version} ...")
        report = workflows.index_folder(root, db, version=version, embedder=self.embedder,
                                        show_progress=self.console.is_terminal)
        self.report_index(report)
        return db

    def report_index(self, report: workflows.IndexReport) -> None:
        st = report.stats
        self.say(f"  indexed {report.snippets:,} snippets into {report.db} "
                 f"(version {report.version}): {st.encoded_vectors:,} new vectors "
                 f"({st.model_encoded:,} texts through the model, {st.text_cache_hits:,} "
                 f"from the text cache), {st.reused_vectors:,} reused, {report.seconds:.0f}s")

    def shown_db(self) -> str:
        try:
            return str(Path(self.db).resolve().relative_to(Path.cwd().resolve()))
        except ValueError:
            return str(self.db)

    def status_line(self) -> str:
        model = getattr(self.embedder, "name", "?").split("/")[-1]
        seq = getattr(self.embedder, "max_seq_length", "?")
        versions = ""
        if len(self.index.available_versions) > 1:
            versions = f"  | version {', '.join(self.index.loaded_versions)}"
        return (f"  ready  | {model} (seq {seq})  | index {self.shown_db()}  | "
                f"{len(self.index.ids):,} snippets{versions}  | model loaded in "
                f"{getattr(self, 'model_seconds', 0):.1f}s")

    def show_versions(self) -> None:
        self.say(f"  versions   {', '.join(self.index.available_versions)}")
        self.say(f"  searching  {', '.join(self.index.loaded_versions)}")

    def switch_versions(self, versions: str | None) -> None:
        if versions and versions != "all" and "," in versions:
            versions = [v.strip() for v in versions.split(",") if v.strip()]
        self.open_index(self.db, versions=versions)
        self.say(f"  searching  {', '.join(self.index.loaded_versions)}  "
                 f"({len(self.index.ids):,} snippets)", style="green")

    def run_query(self, text: str, route_it: bool = True) -> None:
        if route_it:
            from retrieval.router import route
            r = route(text)
            if r is not None:
                self.answer(text, r)
                return
        started = time.perf_counter()
        repo = getattr(self.index, "corpus_kind", "") == "repo"
        result = workflows.shipped_search(self.index, text, self.embedder,
                                          top_k=self.page_size * 10, collapse=self.collapse,
                                          include_tests=self.show_tests or not repo,
                                          prefer_newest=self.prefer_newest,
                                          hubness=self.hubness)
        ms = (time.perf_counter() - started) * 1000
        self.latencies.append(ms)
        self.hits = result.hits
        self.result_versions = result.versions
        self.page = 0
        self.last_query = text
        label, margin = workflows.confidence(self.hits, getattr(self.index, "identity_of", None))
        self.confidence_label = label
        distinct = ""
        if len(self.index.loaded_versions) > 1:
            top10 = [sid for sid, _ in self.hits[:10]]
            names = {self.index.identity_of.get(sid, sid) for sid in top10}
            distinct = f"  |  distinct functions {len(names)} of {len(top10)}"
        self.say(f"\n  {len(self.hits)} results  |  {ms:.1f} ms  |  confidence {label} "
                 f"(margin {margin:.3f}){distinct}"
                 + ("  |  tests hidden (:tests on)" if repo and not self.show_tests else ""),
                 style="dim")
        self.show_page()

    def show_page(self) -> None:
        lo = self.page * self.page_size
        chunk = self.hits[lo:lo + self.page_size]
        if not chunk:
            self.say("  no more results")
            return
        for offset, (sid, score) in enumerate(chunk):
            rank = lo + offset + 1
            self.show_result(rank, sid, score, preview=True)
        remaining = len(self.hits) - (lo + len(chunk))
        if remaining > 0:
            self.say(f"  ... {remaining} more  (:more, or :open N)", style="dim")

    def show_result(self, rank: int, sid: str, score: float, preview: bool) -> None:
        from rich.syntax import Syntax

        info = workflows.describe_hit(self.index, sid)
        where = info.path
        if info.start is not None:
            where += f":{info.start}-{info.end}"
        tag = f"  [{self.confidence_label}]" if rank == 1 else ""
        if len(self.index.loaded_versions) > 1:
            from retrieval.versions import evolution, evolution_label
            found = evolution_label(evolution(self.index, sid)) or self.index.version_of.get(sid, "?")
            tag += f"  ({self.index.version_of.get(sid, '?')}; in {found})"
        self.say(f"\n  {rank:>2}.  {score:.4f}  {where}  {info.name}{tag}", style="bold")
        from retrieval.display import safe
        text = safe(self.index.content(sid))
        lines = text.splitlines()
        if preview:
            lines = lines[:PREVIEW_LINES]
        start = info.start or 1
        syntax = Syntax("\n".join(lines), info.language, line_numbers=True,
                        start_line=start, word_wrap=False,
                        background_color="default", theme="ansi_dark")
        self.console.print(syntax)

    def open_result(self, arg: str) -> None:
        if not self.hits:
            self.say("  run a query first")
            return
        if not arg.isdigit() or not 1 <= int(arg) <= len(self.hits):
            self.say(f"  :open takes a result number between 1 and {len(self.hits)}")
            return
        sid, score = self.hits[int(arg) - 1]
        self.show_result(int(arg), sid, score, preview=False)

    def show_history(self, arg: str) -> None:
        from retrieval.versions import evolution, history_diffs
        if not self.hits:
            self.say("  run a query first")
            return
        if not arg.isdigit() or not 1 <= int(arg) <= len(self.hits):
            self.say(f"  :history takes a result number between 1 and {len(self.hits)}")
            return
        sid, _ = self.hits[int(arg) - 1]
        appearances = evolution(self.index, sid)
        name = workflows.describe_hit(self.index, sid).name
        missing = [v for v in self.index.loaded_versions
                   if v not in {a.version for a in appearances}]
        self.say(f"\n  history of {name}: in {len(appearances)} of "
                 f"{len(self.index.loaded_versions)} searched versions", style="bold")
        for a in appearances:
            info = workflows.describe_hit(self.index, a.uid)
            where = info.path + (f":{info.start}-{info.end}" if info.start is not None else "")
            state = "changed" if a.changed else ("first seen" if a is appearances[0] else "same")
            self.say(f"    {a.version:<12} {state:<10} {a.hash[:10]}  {where}")
        if missing:
            self.say(f"    not in: {', '.join(missing)}", style="dim")
        if len(self.index.loaded_versions) < len(self.index.available_versions):
            self.say("    (only searched versions are shown; :all-versions for every one)",
                     style="dim")
        for old, new, diff in history_diffs(self.index, sid):
            self.say(f"\n  {old} -> {new}", style="bold")
            for line in diff[2:]:
                style = ("green" if line.startswith("+") else
                         "red" if line.startswith("-") else "dim" if line.startswith("@@") else None)
                self.say(f"    {line}", style=style)
        if not any(a.changed for a in appearances) and len(appearances) > 1:
            self.say("  identical in every version", style="dim")

    def show_uses(self, term: str, limit: int = 40) -> None:
        if not term:
            self.say("  usage: :uses <name, URL or string>")
            return
        started = time.perf_counter()
        hits = workflows.uses(self.index, term)
        ms = (time.perf_counter() - started) * 1000
        units = len({h.unit for h in hits})
        noun = "snippets" if getattr(self.index, "corpus_kind", "") == "apps" else "files"
        if units == 1:
            noun = noun[:-1]
        self.say(f"\n  {len(hits)} lines in {units} {noun} mention {term!r}  |  {ms:.0f} ms"
                 "  (exact match)", style="dim")
        for h in hits[:limit]:
            self.say(f"  {h.path}:{h.line}  {h.name}")
            self.say(f"      {h.text[:110]}", style="dim")
        if len(hits) > limit:
            self.say(f"  ... {len(hits) - limit} more", style="dim")

    def show_calls_before(self, first: str, second: str, limit: int = 40) -> None:
        started = time.perf_counter()
        hits = workflows.calls_before(self.index, first, second)
        ms = (time.perf_counter() - started) * 1000
        noun = "function" if len(hits) == 1 else "functions"
        self.say(f"\n  {len(hits)} {noun} call {first}() before {second}()  |  {ms:.0f} ms"
                 "  (source order within a function; aliases not followed)", style="dim")
        for h in hits[:limit]:
            self.say(f"  {h.path}:{h.first_line}  {h.name}   "
                     f"{first} at line {h.first_line}, {second} at line {h.second_line}")

    def show_lines(self, hits, what: str, limit: int = 40) -> None:
        noun = "line" if len(hits) == 1 else "lines"
        self.say(f"\n  {len(hits)} {noun}: {what}  (lexical: no call graph, aliases not "
                 "followed)", style="dim")
        for h in hits[:limit]:
            self.say(f"  {h.path}:{h.line}  {h.name}")
            self.say(f"      {h.text[:110]}", style="dim")
        if len(hits) > limit:
            self.say(f"  ... {len(hits) - limit} more", style="dim")

    def answer(self, question: str, r) -> None:
        vocab = workflows.vocabulary(self.index)
        res = [vocab.resolve(p, identifiers_only=r.kind != "usage") for p in r.phrases]
        missing = [x for x in res if not x.best]
        if missing:
            self.say(f"  Understood as: {r.kind} of {', '.join(repr(x.phrase) for x in res)}, but "
                     f"no name like {missing[0].phrase!r} is in this index - searching instead.",
                     style="yellow")
            self.run_query(question, route_it=False)
            return
        self.more_matches = {x.best: x.others for x in res if x.others}
        extra = sum(len(x.others) for x in res)
        tail = (f"  ({extra} other match{'es' if extra != 1 else ''} - type 'more matches' to see)"
                if extra else "")
        names = [x.best for x in res]
        if r.kind == "usage":
            self.say(f"  Understood as: usages of {names[0]}{tail}", style="green")
            self.show_uses(names[0])
        elif r.kind == "callers":
            self.say(f"  Understood as: callers of {names[0]}(){tail}", style="green")
            self.show_lines(workflows.callers(self.index, names[0]), f"calls to {names[0]}()")
        elif r.kind == "definition":
            self.say(f"  Understood as: the definition of {names[0]}{tail}", style="green")
            self.show_lines(workflows.definitions(self.index, names[0]), f"{names[0]} defined")
        else:
            first, second = names if r.order == "before" else names[::-1]
            self.say(f"  Understood as: functions that call {names[0]}() {r.order} "
                     f"{names[1]}(){tail}", style="green")
            self.show_calls_before(first, second)

    def show_more_matches(self) -> None:
        if not getattr(self, "more_matches", None):
            self.say("  no other matches for the last question")
            return
        for best, others in self.more_matches.items():
            self.say(f"  for {best}: did you mean {', '.join(others)}?")

    def show_stats(self) -> None:
        self.say(f"  index      {self.shown_db()}  ({len(self.index.ids):,} snippets)")
        self.say(f"  model      {getattr(self.embedder, 'name', '?')}")
        if not self.latencies:
            self.say("  queries    0")
            return
        ordered = sorted(self.latencies)
        p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
        self.say(f"  queries    {len(ordered)}   p50 {statistics.median(ordered):.1f} ms"
                 f"   p95 {p95:.1f} ms   max {ordered[-1]:.1f} ms")

    def handle(self, line: str) -> bool:
        line = line.strip()
        if not line:
            return True
        if not line.startswith(":"):
            import re
            low = " ".join(line.lower().rstrip(".!").split())
            if low in {"quit", "exit", "bye", "q"}:
                return False
            if low in {"help", "?"}:
                self.say(HELP)
                return True
            if low in {"more", "next", "next page"}:
                return self.handle(":more")
            if low in {"more matches", "other matches", "did you mean"}:
                self.show_more_matches()
                return True
            m = re.fullmatch(r"(?:open|show)\s+(?:result\s+)?(\d+)", low)
            if m:
                self.open_result(m.group(1))
                return True
            self.run_query(line)
            return True
        cmd, _, arg = line[1:].partition(" ")
        cmd, arg = cmd.lower(), arg.strip()
        if cmd in {"q", "quit", "exit"}:
            return False
        if cmd == "help":
            self.say(HELP)
        elif cmd == "open":
            self.open_result(arg)
        elif cmd == "more":
            if not self.hits:
                self.say("  run a query first")
            else:
                self.page += 1
                self.show_page()
        elif cmd == "index":
            if not arg:
                self.say("  usage: :index <path to a folder of code> [version label]")
            else:
                import shlex
                parts = shlex.split(arg, posix=False)
                label = parts[1] if len(parts) > 1 else "v1"
                db = self.index_folder(Path(parts[0].strip('"')), version=label)
                self.open_index(db, versions=label)
                self.say(self.status_line(), style="green")
        elif cmd == "version":
            if arg:
                self.switch_versions(arg)
            else:
                self.show_versions()
        elif cmd == "all-versions":
            self.switch_versions("all")
        elif cmd == "collapse":
            modes = {"on": True, "off": False, "auto": None}
            if arg.lower() not in modes:
                self.say("  usage: :collapse on | off | auto")
            else:
                self.collapse = modes[arg.lower()]
                self.say(f"  collapse   {arg.lower()}")
        elif cmd == "history":
            self.show_history(arg)
        elif cmd == "hubness":
            modes = {"on": True, "off": False, "auto": None}
            if arg.lower() not in modes:
                self.say("  usage: :hubness on | off | auto")
            else:
                self.hubness = modes[arg.lower()]
                self.say(f"  hubness    {arg.lower()}")
        elif cmd == "newest":
            if arg.lower() not in {"on", "off"}:
                self.say("  usage: :newest on | off")
            else:
                self.prefer_newest = arg.lower() == "on"
                self.say(f"  newest     {arg.lower()}")
        elif cmd == "tests":
            if arg.lower() not in {"on", "off"}:
                self.say("  usage: :tests on | off")
            else:
                self.show_tests = arg.lower() == "on"
                self.say(f"  tests      {'shown' if self.show_tests else 'hidden'}")
        elif cmd == "search":
            if not arg:
                self.say("  usage: :search <question>")
            else:
                self.run_query(arg, route_it=False)
        elif cmd == "uses":
            self.show_uses(arg.strip('"'))
        elif cmd == "calls":
            first, sep, second = arg.partition(" before ")
            if not sep or not first.strip() or not second.strip():
                self.say("  usage: :calls <X> before <Y>")
            else:
                self.show_calls_before(first.strip(), second.strip())
        elif cmd == "stats":
            self.show_stats()
        else:
            self.say(f"  unknown command :{cmd}")
            self.say(HELP)
        return True

    def run(self) -> int:
        self.say(render_banner())
        try:
            self.load_model()
            db = self.choose_db()
            if not workflows.index_is_usable(db):
                db = self.ask_what_to_index(db)
                if db is None:
                    self.say("  bye.")
                    return 0
            self.open_index(db, versions=self.version_filter)
            self.say(self.status_line(), style="green")
            self.say("  type a question, or :help\n", style="dim")
        except KeyboardInterrupt:
            self.say("\n  bye.")
            return 0
        except Exception as exc:
            self.fail(exc, hint=hint_for(exc))
            return 1

        while True:
            try:
                try:
                    line = self.read("debug907> ")
                except OSError as exc:
                    self.fail(exc, hint="run debug907 in an interactive terminal "
                                        "(with Docker, add -it)")
                    return 1
                if line is None:
                    self.say("\n  bye.")
                    return 0
                if not self.handle(line):
                    self.say("  bye.")
                    return 0
            except KeyboardInterrupt:
                self.say("\n  bye.")
                return 0
            except Exception as exc:
                self.fail(exc, hint=hint_for(exc))


def hint_for(exc: BaseException) -> str | None:
    import sqlite3

    from retrieval.embed import EmbedderUnavailable

    if isinstance(exc, EmbedderUnavailable):
        text = str(exc)
        if "hotspot" in text:
            return "switch to another network (a phone hotspot works), then relaunch"
        if "internet" in text or "offline mode" in text:
            return "connect to the internet once for the one-time model download, then relaunch"
        return "debug907 doctor   (checks the environment and the model)"
    if isinstance(exc, FileNotFoundError):
        return "check the path, or quote it if it contains spaces"
    if isinstance(exc, sqlite3.DatabaseError):
        return "the index file is damaged; rebuild it with :index <path>"
    if isinstance(exc, IndexProblem):
        if "is not in" in str(exc):
            return ":version   (lists the versions in this index)"
        return "rebuild it: debug907 index (AppsRetrieval) or :index <folder>"
    if isinstance(exc, ValueError) and "FIXTURE" in str(exc):
        return "pass --db with a real index, or run debug907 and index a folder"
    return "rerun with --debug for the full traceback"


class IndexProblem(RuntimeError):
    pass


def main(db: Path | None = None, debug: bool = False, versions: str | None = None) -> int:
    launched_from = Path.cwd()
    if db is not None and not Path(db).is_absolute():
        db = (launched_from / db).resolve()
    os.chdir(Path(__file__).resolve().parent)
    return Tool(db=db, debug=debug, cwd=launched_from, versions=versions).run()
