# debug907: technical report

**Samsung PRISM Generative AI Hackathon 2026–27 · Theme 01**
**Team:** Aadvik Chawla, Kamneev Singh

Every number below comes from a committed file under `bench/results/`, named
beside it. Timings are on a laptop CPU (16 logical CPUs, CPU only); timings
marked *under load* were taken while other jobs ran on the same machine.

## Contents

1. Summary · 2. The problem · 3. Architecture and the shipped configuration ·
4. Dataset and environment · 5. Statistical method · 6. Model selection ·
7. Ablation study · 8. Document text: raw versus the code view · 9. The hubness
correction · 10. What is fitted, and what is not · 11. Screening format: JSON and
CSV · 12. Reproducibility · 13. Failure analysis, confidence, tough queries ·
14. Generalisation: held-out benchmarks and the stock model · 15. The terminal
tool and plain-English questions · 16. Indexing your own code · 17. P1: versions,
incremental indexing, commit sync · 18. Bonus: evolutionary retrieval ·
19. Memorisation and reliance on names · 20. Measured and not adopted ·
21. Engineering quality · 22. Latency · 23. The guidelines' worked example ·
24. Limitations · 25. Security review

---

## 1. Summary

**CoIR AppsRetrieval test split (3,765 queries), MTEB 2.21.5:**

| NDCG@10 | MRR@10 | Recall@100 |
|---:|---:|---:|
| **0.60952** | **0.56320** | **0.92457** |

File: `appsretrieval_results.json`, 5.6× the first artifact (jina, 0.10982).
**Shipped configuration:** `Alibaba-NLP/gte-modernbert-base` (pinned revision
`e7f32e3c`), fp32, max sequence 1,024, dense retrieval, documents embedded as given
(raw text), with a **hubness correction** (§9): score = cos(q, d) − 0.75 · hub(d),
where hub(d) is d's mean cosine to its 20 nearest AppsRetrieval *training* queries.
The two earlier artifacts, plain cosine on a processed "code view" (0.50431 / 0.45583
/ 0.87862) and the hubness correction on the code view (0.53926 / 0.49249 / 0.89296),
are recorded in `appsretrieval_results.meta.json` and reproduce exactly with
`run_mteb.py --code-view [--no-hubness]`.

## 2. The problem

The Theme 01 guidelines define it this way. Given a library of code and a
natural-language query, rank the snippets by relevance. Generating or explaining
answers is out of scope. Screening is NDCG@10 and MRR on CoIR AppsRetrieval via
MTEB. The hands-on round runs the code, looks at P1 (versions) and the Bonus
(evolutionary retrieval), and cares about speed. Rubric: prototype 30, depth 25,
innovation 20, relevance 15, presentation 10.

## 3. Architecture and the shipped configuration

```
INDEX   source -> extract functions -> document text -> bi-encoder -> vectors (keyed by exact text)
                                    \-> enriched text -> SQLite FTS5 (BM25)
SEARCH  question -> router -> exact lookups (usage / callers / definition / order)
                           -> bi-encoder -> cosine over a numpy matrix (- beta*hub on AppsRetrieval) -> top k
        (built and switchable: BM25 + RRF fusion, interaction rerank, structural adjustment)
```

The document text is the raw source for AppsRetrieval and a "code view" (a header
of names and signatures, then the code) for repositories (§8). The shipped path is
the bi-encoder and cosine. BM25, RRF and both rerank stages are built, tested and
one switch away, and each one **measured worse** (§7, §16). The guideline
improvement axes map onto code as follows:
- query categorisation and preprocessing: `pipeline/query_proc.py`
- snippet categorisation and processing: `pipeline/doc_proc.py`, `pipeline/chunk.py`
- multiple passes: `retrieval/fuse.py`, `retrieval/rerank.py`

`retrieval/workflows.py` is the **single implementation** of every user-facing
path: the tool, `search`, `index`, `index-repo`, `rank`, `eval`, `reproduce` and
Docker all call it. The 30-snippet fixture is used only by `selftest` and the
tests, and it prints FIXTURE when used. The README has the same architecture as a
diagram.

## 4. Dataset and environment

AppsRetrieval: 8,765 snippets, 3,765 test queries, one relevant snippet each.
Python solutions to competitive-programming problems (bare scripts). Queries are
long problem statements: median 1,601 characters, p95 ~740 tokens. Dataset
revisions are pinned (`bench/dataset.py`).

The environment is `requirements.txt`, an exact lockfile of 62 `==` pins (torch
2.14.0 CPU, transformers 4.56.2, sentence-transformers 4.1.0, MTEB 2.21.5), with
environment markers so that Python 3.11 gets numpy 2.4.6 and scipy 1.17.1.
transformers must be ≥4.48 for ModernBERT and <5 for the jina baseline; one
environment runs everything. `python cli.py doctor` diagnoses it, and
`python cli.py --version` prints the commit, model revision and configuration.

## 5. Statistical method

- NDCG@10 uses the MTEB / pytrec_eval gain convention.
- The paired bootstrap resamples the queries (500–1,000 draws), so per-query
  difficulty cancels out (`bench/significance.py`). "Significant" means the 95%
  interval of the difference excludes zero.
- The 1,000-query stratified sample is allocated by query-length decile. It was
  validated to preserve the order of every configuration pair the full split can
  distinguish (`significance.json`). A 300-query random sample did not: it matched the
  overall score but put the configurations in the wrong order.
- The 2,765 test queries outside the sample were never used for any choice, so
  they are a held-out check for everything chosen on the sample.

## 6. Model selection

1,000-query stratified sample, fp32, seq 512 (`model_comparison_sample1000.json`):

| model | NDCG@10 | R@100 | vs jina |
|---|---:|---:|---|
| **gte-modernbert-base** | **0.4906** | **0.863** | **+0.347** [0.320, 0.373], p≈0.000 |
| jina-v2-base-code | 0.1434 | 0.432 | - |
| granite-embedding-english-r2 | 0.1275 | 0.370 | −0.016, n.s. |

The order and rough size match published CoIR numbers, so the harness is
calibrated. Artifacts: jina int8 0.10982 → gte seq 512 0.48908 → **gte seq 1,024
0.50431**. The sequence-length gain is query-side: 512 tokens truncated about a
quarter of the queries.

## 7. Ablation study

Stratified sample, gte, seq 512, paired bootstrap against the shipped row
(`ablation_real.json`). **Every change from the shipped configuration is significant at
p≈0.000 unless marked.** The whole table was re-run after the vector-cache key fix
(§21): all 15 rows identical to full float precision.

| group | variant | NDCG@10 | vs shipped |
|---|---|---:|---:|
| controls | BM25 only, raw code | 0.0053 | −0.485 |
| | BM25 + enrichment + preprocessing | 0.0313 | −0.459 |
| passes (axis 5) | + interaction rerank | 0.4782 | −0.012 |
| | + BM25 fusion, w=0.15 / 0.30 | 0.4634 / 0.4481 | −0.027 / −0.043 |
| query (axes 1–2) | strip `-----Input-----` sections | 0.2308 | −0.260 |
| | embed processed tokens, not raw | 0.0855 | −0.405 |
| document (axes 3–4) | nl view only | 0.0483 | −0.442 |
| | multi-view max | 0.4906 | −0.000 (n.s.) |
| | **seq 1,024** | **0.5092** | **+0.019 → adopted** |

The finding: the stronger the encoder, the more hand-engineering hurts. The
simplest configuration is also the fastest. The hybrid track (BM25 + fusion +
rerank) was also measured with gte on its own and rejected
(`hybrid_search_gte.json`).

## 8. Document text: raw versus the code view

The code view is a header of signatures, then the code cut to its first 512 words
and joined with single spaces, so every newline and indent is gone. Embedding each
AppsRetrieval document as given (only its ends stripped, exactly as MTEB passes it)
instead (`doc_variant_raw.json`):

| | plain cosine | with hubness |
|---|---:|---:|
| code view | 0.5043 | 0.5393 (the 0.53926 artifact) |
| **raw documents** | 0.5754 | **0.6096** |

Raw + hubness beats the code-view artifact by **+0.0702 [+0.0618, +0.0784]** on the
full split and by +0.0702 on the 2,765 held-out queries. On plain cosine, raw text at
1,024 tokens (0.5754) is level with the model card's published 57.54 for this model:
the processing had been costing about 7 points. The same finding holds out of sample
on **CosQA** (`second_benchmark.json`, no tuning, plain cosine): code view 37.62, raw
documents 43.06, model card 43.47.

On the team's repositories the raw view was not significantly better (pooled
+0.0122, p≈0.33, `raw_repos.json`), so repository indexes keep the code view.

## 9. The hubness correction

**Why.** In the failure analysis (§13), 19.3% of wrong #1 answers came from a few
"hub" snippets that are close to everything; one contest template was the wrong #1
for 22 unrelated queries (`failures.json`).

**Method** (`hubness.json`). score = cos(q, d) − β·hub(d). hub(d) is d's mean cosine to
its k nearest AppsRetrieval **training** queries, with a document's own training
queries excluded; test queries are never used. k ∈ {10, 20, 50} and β from 0.05 to
1.0 were swept on the validated sample; k = 20, β = 0.75 won. The table was built on
the raw-document vectors with those k and β copied, never re-tuned
(`tools/build_hub_table.py`).

**Checks on the shipped table** (`hub_table_checks.json`):

| check | plain → hub | change [95% CI] |
|---|---|---|
| full split (3,765) | 0.5753 → 0.6093 | +0.0340 [+0.0276, +0.0402] |
| the 2,765 held-out queries | 0.5722 → 0.6053 | +0.0331 [+0.0266, +0.0404] |
| every train-answer document removed from the corpus | 0.6177 → 0.6414 | +0.0237 [+0.0176, +0.0297] |
| the same, held-out queries only | 0.6154 → 0.6374 | +0.0220 [+0.0152, +0.0290] |

All significant. Train-answer documents are 5,000 of 8,765. The hub score barely
separates them from the rest (AUC 0.491), so the gain is not a split effect. On the
code-view vectors the same checks gave +0.0350 (full), +0.0328 (held-out) and +0.0324
(train answers removed), and the hub share of wrong #1s fell from 19.3% to 6.2%.
Document-only hubs (no queries) gave +0.0014, n.s.; on npm the best tool-only variant
was +0.0267, p≈0.068, n.s., so repository indexes are not corrected.

**Inside MTEB.** MTEB scores with the encoder's plain dot product (`similarity()`), so
one appended dimension (+1 for queries, −β·hub(d) for documents) makes MTEB compute
the corrected score exactly. The encoder applies the committed table
(`data/apps_hubness.json`) only when MTEB says the task is AppsRetrieval, and refuses
a table built for a different document text. The tool applies the same table on an
AppsRetrieval index, so `reproduce --full` matches the artifact on both paths.
`:hubness on|off|auto` in the tool and `search --no-hubness` switch it.

**Disclosed caveats** (§24): the table is computed for this corpus, and
it was measured on full problem statements, not on short hand-typed questions.

## 10. What is fitted, and what is not

- **Nothing is trained.** The encoder's weights (gte-modernbert-base, pinned revision
  `e7f32e3c`) are frozen; no fine-tuning of any kind.
- **The only data-fitted component is the hub table** (`data/apps_hubness.json`):
  each document's mean cosine to its 20 nearest AppsRetrieval **training** queries,
  own answers excluded. Two numbers, k = 20 and β = 0.75, were chosen on the
  validated sample. The gain was verified on the 2,765 test queries never used for
  any choice, and with every train-answer document removed. No test query or test
  label was used.
- **Everything else is configuration chosen by measurement**: the document text
  (raw), the sequence length (1,024), dense retrieval only. Each choice has its
  ablation or held-out check in `bench/results/`.

## 11. Screening format: JSON and CSV

MTEB exports per-query predictions natively (`prediction_folder`). The re-run
reproduces the artifact exactly: 0.60952 / 0.5632031661712937 / 0.92457.
`appsretrieval_rankings.csv` (`query_id,corpus_id,rank,score`, top 100) is scored by
`tools/score_csv.py`: plain Python, independent of MTEB, with pinned qrels. It gives
**0.60952 / 0.56320, +0.00000**. An exact match required pytrec_eval's tie order (doc
id descending): 3,508 queries contain exact score ties.

The artifact and its CSV are byte-exact in git (`.gitattributes -text`), so their
SHA-256 is the same on every platform and in the release: `ec84a471…` for the JSON,
recorded in `appsretrieval_results.meta.json` and checked by a test.

## 12. Reproducibility

- **`debug907 reproduce`**: the validated sample must hit 0.6206 ± 0.003.
  **`reproduce --full`**: both the official MTEB path and the tool's own search path
  must match the artifact within 0.001; measured +0.00000 (MTEB) and +0.00009 (tool).
- **The guidelines' own snippet.** `PrePostPipelineEncoder()` with no arguments is the
  submitted configuration, so the evaluation snippet in the Theme 01 guidelines
  reproduces 0.60952 / 0.56320 / 0.92457 exactly (`tools/guidelines_template.py`).
  On MTEB 2.21.5 the snippet's `json.dump` needs `default=str`, because the result
  contains a datetime.
- **Clean install**: a fresh venv, an empty model cache, a fresh
  export of the committed code and nothing but the README. It found three bugs, all
  fixed: the first download was 4.3 GB; a Windows symlink refusal left a partial
  snapshot; a docstring was parsed as a class. After the fixes: 288 MB download,
  index + query in 126 s, then fully offline with the network disabled. A no-cache
  install of the lockfile takes 622 s.
- **Python versions**: 3.11, 3.12 and 3.14 each installed from
  an empty venv with the README commands, then `doctor`, the test suite and
  `reproduce` (0.6206) pass. Vectors encoded from scratch are identical on 3.11
  (numpy 2.4.6) and 3.14 (numpy 2.5.1): maximum absolute difference 0.0. 3.13 was not
  available to test.
- **First-run model fetch.** The model downloads once (~290 MB), pinned, with
  progress; after that everything is offline, and a cached model touches no network.
  No-internet, offline-mode and SSL failures each get a one-line message with a fix.
- **Docker**. The model and the dataset are baked in,
  runs are offline, the image runs as a non-root user, and the test suite runs during
  the build. `tools/docker_pipeline.ps1` / `.sh` builds, indexes the corpus in the
  container and fails unless `reproduce` hits the shipped number. It passed on every
  configuration, most recently 0.6206 (+0.0000).
- **Optional pre-built index.** `export-index` writes a deterministic 35.7 MB file;
  `import-index` checks its SHA-256 against the hash pinned in
  `data/prebuilt_index.json` **before opening it**, then the decompressed database's
  own hash, then that the encoder signature and view are the shipped ones
  (§25). `verify-index` re-encodes a random sample from the stored source with
  no cache: 200 of 200 at min cosine 1.0000000. `reproduce` on the imported index gives
  0.6206. Building from scratch remains the default; the screening score does not use
  the file.
- **Resumable indexing.** Cache misses are encoded in steps of whole batches, each
  stored before the next, so an interrupted build resumes without re-encoding finished
  work, with a progress line giving the time remaining. The batches are the same as in
  one call: maximum absolute difference 0.0 on 300 real documents.

**Python versions**, each from an empty virtual environment with the README commands
(Windows 11):

| Python | README install | `doctor` | test suite | `reproduce` |
|---|---|---|---|---|
| 3.11.15 | passes with the numpy 2.4.6 / scipy 1.17.1 environment markers (without them: `No matching distribution found for numpy==2.5.1`) | READY | pass | 0.6206 (+0.0000) |
| 3.12.13 | pass (298 s) | READY | pass | 0.6206 (+0.0000) |
| 3.14.6 | pass (298 s) | READY | pass | 0.6206 (+0.0000) |

**Docker pipeline runs** (every run step `--network none`; the build runs the full test
suite, 354 passed and 10 git/PowerShell-only skips inside the image):

| configuration | volume | image | reproduce target | got |
|---|---|---|---|---|
| final submission (pre-built index, resumable encoding, environment markers) | reused | 1.16 GB | 0.6206 ± 0.003 | **0.6206 (+0.0000)** |
| raw documents, non-root image | fresh, cold encode | 1.16 GB | 0.6206 ± 0.003 | **0.6206 (+0.0000)**, 801 s |
| code view + hubness correction (the 0.53926 artifact) | fresh, cold encode (44.0 min) | 0.87 GB | 0.5503 ± 0.003 | **0.5503 (+0.0000)** |
| code view, plain cosine (the 0.50431 artifact) | reused | 0.87 GB | 0.5092 ± 0.003 | **0.5092 (+0.0000)** |

The in-image test run caught CRLF shell scripts from a Windows working copy, a test that
assumed git inside the image, and a symlink-test fixture below the extractor's minimum
function length; all fixed. The image grew from 0.87 to 1.16 GB with the non-root user,
because `chown -R` rewrites the baked model cache into a new layer.

## 13. Failure analysis, confidence, tough queries

**Failure analysis** (full split, `failures.json`): the relevant solution lands at rank
1 for **36.6%** of queries, ranks 2–10 for 29.3%, ranks 11–100 for 22.0%, and outside
the top 100 for 12.1%.
- **Flat by length.** NDCG@10 by query-length tercile is 0.52 / 0.50 / 0.50, and by
  solution-length tercile 0.50 / 0.51 / 0.51.
- **Two real weak spots.** Solutions over ~4,000 characters are truncated by the
  1,024-token window (17 queries, 20% in the top 10). Russian statements score 41% in
  the top 10 against 66%.
- **50 misses, each read by hand** (`data/failure_labels.json`): shared technique 25
  (the #1 solves a *different* problem the same way); closed-form insight 19;
  scaffolding 3; unreadable (base64 `exec`) 1; library idiom 1; non-English 1.
- **Hubs.** 20% of wrong #1s come from snippets that are the wrong #1 for five or more
  queries, which motivated §9.

**Confidence** (`confidence.json`). The label comes from the margin between the #1 and
#2 scores: high ≥ 0.03, low < 0.01, medium in between. Thresholds were set on the
sample and checked on the **2,765 held-out queries**, on the hubness-corrected scores:

| label | share (held-out) | top-1 correct |
|---|---:|---:|
| high | 28% | **82%** |
| medium | 34% | 33% |
| low | 38% | 15% |

The margin's AUC for a correct #1 is **0.818** held out (0.799 on plain cosine).

**Tough queries** (`showcase.json`). On the sample, **230** queries have the solution at
#1 while jina-v2-base-code leaves it outside its top 10. The three shown share
**zero** vocabulary with their solutions.

## 14. Generalisation: held-out benchmarks and the stock model

The submitted encoder, `PrePostPipelineEncoder()` with its defaults, through the
official `mteb.evaluate` on every natural-language → code retrieval task in the
installed MTEB with at most 25,000 documents. Nothing was tuned; the AppsRetrieval hub
table is off, and the script asserts it (`generalisation.json`, `tools/generalisation.py`).

| held-out task (MTEB, no tuning) | docs / queries | NDCG@10 | published gte-modernbert-base |
|---|---:|---:|---|
| CodeSearchNetRetrieval, **JavaScript** | 1,000 / 1,000 | **80.84** | none comparable* |
| CodeSearchNetRetrieval, Python · Go · Ruby · Java · PHP | 1,000 / 1,000 each | 90.26 · 95.89 · 85.19 · 91.59 · 88.87 | none comparable* |
| StackOverflowQA | 19,931 / 1,994 | **91.21** | 91.2 (model card) · 90.88 (MTEB results repository) |
| WikiSQLRetrieval | 2,048 / 2,048 | 94.70 | none |
| HumanEvalRetrieval | 158 / 158 | 84.48 | none |
| MBPPRetrieval | 974 / 974 | 80.73 | none |
| DS1000Retrieval | 1,998 / 1,998 | 51.78 (R@100 97.2) | none |
| FreshStackRetrieval | 3,804 / 672 | 31.84 (R@100 69.6) | none |
| CosQA | 20,604 / 500 | 43.06 | 43.47 (model card) · 42.18 (MTEB results repository) |

*The model card's CodeSearchNet numbers are for CoIR's million-document version, a
different corpus. Tasks over 25,000 documents were not run; they are listed with
their sizes in `generalisation.json`.

Where a published number exists, ours matches it. Every CodeSearchNet language is
80–96, and JavaScript, the hackathon deck's language, is the lowest of the six. The
two weak tasks fail in ways the design predicts: DS1000 finds the right snippet (R@100
97.2) but cannot rank near-identical library snippets; FreshStack asks about libraries
newer than the model's training data.

**The stock model.** Loaded correctly (the pinned snapshot, CLS pooling, its default
8,192 tokens, raw documents, no hub table) through the official MTEB AppsRetrieval run,
the stock model scores **NDCG@10 0.57673, MRR@10 0.52885** (model card 57.54; 70.6 min,
peak 4.8 GB; `naive_baseline_pinned.json`). The submission is +0.033 above it; the
hubness correction is the difference. Loaded through `mteb.get_model` as MTEB
registers it, the same model scores only **0.31908** (`naive_baseline.json`): the
registry pins an old revision without the sentence-transformers config, so the model
silently loads with mean pooling instead of its trained CLS pooling (shown by encoding
the same texts both ways).

## 15. The terminal tool and plain-English questions

`debug907` (or `python cli.py`) opens the tool. It shows a lowercase ASCII banner with
the credits, loads the model, shows a status line, and asks what to index if there is
no index yet. Each result shows rank, score, `path:lines`, name, a syntax-highlighted
preview, latency and a confidence label.

**Plain-English questions** (`english_questions.json`). Usage, callers, definition and
call-order questions in ordinary English are routed to exact lookups, and names
written in words are resolved to the index's identifiers ("the reify finish helper"
→ `reifyFinish`). The tool prints how it read each question ("Understood as: …").
12 of 12 hand-written questions are correct on npm; 0 of the 8,765 benchmark queries
are routed away from ranked search.

| commands | notes |
|---|---|
| `open N`, `more`, `more matches`, `help`, `quit` | plain controls, no colon needed |
| `:index <folder> [label]` | index a folder, optionally as a version |
| `:version [label]`, `:all-versions`, `:collapse on\|off\|auto`, `:history N` | versions (§17–18) |
| `:tests on\|off` | include test files in repository results (§16) |
| `:uses`, `:calls X before Y` | exact code queries |
| `:hubness on\|off\|auto`, `:stats` | the correction; index statistics |

Errors are one friendly line with a fix; `--debug` shows the full traceback. A real
session is in `tool_session.txt` and in the README.

**Structural queries** are exact lexical operations: `:uses <term>` lists every line
with file:line; `:calls X before Y` lists functions whose first call to X precedes
their first call to Y, in source order within one function, following no aliases and
no call graph. Overlapping snippets report each line once; on AppsRetrieval each
snippet is its own unit.

| index | `:uses` | `:calls` |
|---|---:|---:|
| npm, 2,672 functions | 153 ms | 302 ms |
| AppsRetrieval, 8,765 snippets | 440 ms | 322 ms |

**Batch ranking** (`batch_ranking.json`): `index-corpus` indexes any snippet file,
`rank --queries` ranks a question file into `query_id,corpus_id,rank,score`, and
`eval --qrels` scores it. The CSV scores identically to the shipped path. The deck's
optimisation suggestions were not built: that is generation, which is out of scope.

## 16. Indexing your own code

`index-repo` extracts JS/TS/Python functions (`pipeline/repo.py`); `--all-languages`
adds 60-line windows for other languages. Extraction fixes found on real code:
- **Signature-only snippets:** braces in parameters (`({ a })`, `opts = {}`) ended a
  function on its first line; 140 of 2,597 npm functions were indexed as their
  signature alone. Now 0.
- **Regex literals:** a `//` inside a regex literal was stripped as a comment.
- **No-semicolon arrows:** expression-bodied arrows without a `;` ran into the next
  function (51 cases, now 0).
- **Snapshots:** generated snapshot folders are skipped (1,269 of npm's 3,866
  "functions").
- **Python** uses the syntax tree; a docstring line beginning "class …" had been
  extracted as a class.

**Privacy defaults** (tested): secret files are never indexed (`.env*`, `*.pem`,
`*.key`, keystores, `.npmrc`, credentials and service-account files); a snippet that
*contains* a private key or a known token format is dropped; symbolic links are never
followed; virtual environments are detected by `pyvenv.cfg`; minified files are
detected by name and by line length; every skip is reported by reason.

**Test files are hidden by default** in repository indexes (`:tests on`,
`search --include-tests`). On the team's project they crowded implementation out of
the top 10; hiding them lifted the hand-written questions from top-1 38% to 63% and
top-10 88% to 100% (`local_repo_test.json`).

**Real code: npm** (`repo_eval.json`). `debug907 eval-repo` uses 107 functions whose
leading comment becomes the question (79 leaks dropped), plus 20 hand-written
questions (`data/npm_queries.json`), paired bootstrap against the shipped
configuration:

| configuration | NDCG@10 (127 q) | vs shipped |
|---|---:|---|
| **shipped (code view, dense)** | **0.5975** | - |
| multi-view | 0.5711 | −0.026, n.s. (p≈0.052) |
| code + BM25 (RRF) | 0.5441 | **−0.053** |
| nl view | 0.4068 | **−0.191** |

**Real code: the team's private project** (`local_repo_test.json`). Two repositories,
read-only, **aggregate numbers only**:

| | dsa-recommendation (Python) | dsa-website (TS/JS) |
|---|---:|---:|
| files / snippets | 116 / 1,273 | 141 / 469 |
| automatic questions (docstrings / comments) | 193 | 62 |
| right file in the top 10 (top-1) | 94% (55%) | 100% (76%) |
| 8 hand-written questions, expected file in the top 3 | 7 / 8 | 8 / 8 |
| two real commits as versions: `:version` / `:collapse` | work, 0 errors | work, 0 errors |

Secrets were skipped (1 and 2 files). The checkouts were byte-identical and git-clean
before and after the test, as the test enforces.

## 17. P1: versions, incremental indexing, commit sync

Vectors are keyed by the exact encoded text, so unchanged code is never re-encoded in
any version, and versions coexist in one index (snippets keyed by id and version).
npm v10.8.0 → v10.9.0, five releases in **one** index; the encoder never read the text
cache, so the cold cost is real. 8 threads, idle (`p1_versions.json`):

| release | functions | texts through the model | total | a full rebuild would cost |
|---|---:|---:|---:|---:|
| v10.8.0 (cold) | 2,660 | 2,506 | 1,271 s | 1,342 s |
| v10.8.1 | 2,655 | 170 | 104 s | 1,339 s |
| v10.8.2 | 2,656 | 28 | 28 s | 1,340 s |
| v10.8.3 | 2,658 | 23 | 34 s | 1,341 s |
| v10.9.0 | 2,663 | 22 | 33 s | 1,343 s |

Each later release re-encodes 1–6% of its functions, a 92–99% saving. Moved functions
get new ids but keep their vectors, because the key is the text. The index grows
~15 MB per release, and all five versions search together.

**Commit sync.** `debug907 sync <repo> [--watch]` indexes a commit as a version named
by its short hash. It is read-only by construction (`rev-parse`, `ls-tree` and
`cat-file`; no checkout and no hooks), and only the blobs the extractor could index
are read, which took export on npm from 45 s to 1.3 s. Idle timings, 8 threads: the
first sync 1,355 s; release to release 34–123 s; the last 20 npm commits a median of
12.8 s each (18 of them changed no indexed code). Tests on throwaway repositories
include a byte-for-byte snapshot of the repository before and after; they caught a
sub-folder of another repository resolving to the enclosing repository, and a
`cat-file` pipe deadlock on large commits. Both fixed.

## 18. Bonus: evolutionary retrieval

- **Collapse.** Searching all versions at once, the same function appears once per
  version and crowds everything else out. Collapse shows one row per function, and
  each row reads `in v1 = v2 != v3` (`=` identical code, `!=` changed). Through the
  tool on three npm releases and three queries, the top 10 holds only
  **4 distinct functions with collapse off, 10 with it on**, for every query.
- **History.** `:history N` lists the versions a result exists in and prints a unified
  diff for each change. `:newest on` shows the newest of the identical versions, so
  paths and line numbers are today's. Versions are linked by `path::name` when that
  name is unique in every release.
- **Benchmark** (`evolutionary.json`). Each of the sample's gold solutions got synthetic
  later versions in a copy of the index: variables renamed (996), renamed in the
  original style (996), and safely reordered (361). With all versions searched,
  collapse raises distinct solutions in the top 10 from 8.77 to 10.00, and NDCG@10
  from 0.5112 to 0.5175.

## 19. Memorisation and reliance on names

Replacing the gold solution by its renamed version costs **−0.36** (var1-style) and
**−0.37** (style-preserving) NDCG@10; a safe reorder costs +0.004
(`evolutionary.json`). The style-preserving control rules out a "repeated `var` token"
explanation: the encoder relies heavily on identifiers.

**On private code** (`memorisation.json`), the same style-preserving rename on held-out
questions (docstrings and comments):

| corpus | identifier tokens renamed | relative NDCG@10 drop |
|---|---:|---:|
| AppsRetrieval (public), dose-matched | 28% | 56% |
| npm (public) | 30% | 29% |
| dsa-recommendation (private) | 26% | 15% (−0.084, significant) |
| dsa-website (private) | 18% | 2% (n.s.) |

**On three encoders** (`rename_models.json`), over all 996 renamed solutions: gte 72%,
granite 55%, jina 72% relative drop. gte loses significantly more than granite (+0.18
[+0.11, +0.25]) and no more than jina (+0.00 [−0.06, +0.07]).

**Reading: inconclusive.** The drop on private code is much smaller, which favours
memorisation of public code contributing; jina losing as much as gte favours genuine
reliance on names (the statements share variable letters with their solutions). The
question types differ and the private samples are small (183 and 61). granite is not a
clean control: its model card lists its sources without naming CoIR, and the Granite R2
paper (arXiv 2508.21085) says it uses no CoIR training data but trains on Project
CodeNet problem–solution pairs, the same kind of data as APPS; and the other encoders'
low starting scores (~0.13) squeeze their possible drop. The proper fix,
rename-augmented fine-tuning, is out of scope for a frozen-model submission.

## 20. Measured and not adopted

Every idea shipped behind a switch, off, until a paired bootstrap said it wins; the
screening artifact could be replaced only by a change that wins significantly on the
full split and on the held-out queries, and runs inside the MTEB encoder.

| idea | result | file |
|---|---|---|
| BM25 fusion, interaction rerank, query stripping, nl view | all significantly worse (§7) | `ablation_real.json` |
| Query-free hubness (document-only hub scores) | full split +0.0007 over plain cosine (n.s.), −0.0343 vs training-query hubs; repositories pooled +0.0032 (n.s.) | `hubness_queryfree.json` |
| Long documents as mean-pooled windows | only 69 long documents; full +0.0007, held-out +0.0002 (n.s.) | `long_docs.json` |
| Canonical view (identifiers renamed, averaged in) | −0.0701 full, −0.0709 held-out; rename drop 72% → 61% | `doc_variant_canonical.json` |
| Adaptive second pass on LOW confidence (description and I/O sections embedded separately, merged) | mean merge −0.016, RRF −0.032, both significant; +~700 ms per LOW query | `adaptive.json` |

## 21. Engineering quality

**Fixes found by review and testing**, each with a test written failing first:
- **Vector-cache keys.** Stored vectors had been keyed by the raw snippet rather than
  the encoded text, so a preprocessing change could reuse stale vectors. They are keyed
  by the exact encoded text plus the full encoder signature. An audit checked 140,064
  stored vectors against the text cache, and 160 fresh re-encodes matched at cosine
  1.000000. **No reported number was affected**; MTEB was never affected, because its
  cache is keyed by exact text.
- **Incremental by default.** `index` no longer deletes the database; `--reset`
  rebuilds the text tables only, and stored vectors are dropped only with
  `--drop-vectors`. The build output separates new and reused vectors, and model work
  from text-cache hits.
- **Versions coexist** (§17), and logic tuned to AppsRetrieval (category affinities,
  `-----Input-----` stripping) never runs on your own code.
- **The empty-index investigation**: a write-ahead-log
  hypothesis was falsified (12 of 12 two-container rounds clean). The real bug was that
  opening a missing index path created an empty 64 KB file. Fixed.
- **Smaller fixes:** one-off search latency included the model load; confidence with
  several versions counted copies of #1 as the runner-up; an explicit `--db` equal to
  the default was ignored; the reranker now reads its candidates in one query (34.6 →
  26.6 ms, identical rankings); the hub table is parsed once instead of per query.

**Security** (§25): terminal escape sequences from indexed code, symlinks,
output paths that could overwrite the artifact, git option injection in `sync`, a root
Docker user, an assert-based SQL check and shell-script line endings were fixed, each
with a test; the pre-built index import was reviewed; the full git history was scanned
for secrets.

**Tests.** 364 tests in 39 files, all green, also run inside the Docker build (354
passed and 10 git/PowerShell-only skips in the image). Every bug in this report has a
regression test. Privacy, read-only and portability rules are enforced by tests. The
tests need no model and no network: a fake encoder drives the real indexing, search
and tool code (the one worked-example ranking test uses the real encoder when it is
available).

## 22. Latency

8 threads, idle (`latency.json`):
- cold start ~14 s, of which ~12 s is the model load (55 s on the first start after a
  reboot, with a cold disk cache);
- index load 0.8 s;
- warm query p50 **691 ms** on AppsRetrieval's ~1,600-character statements, and
  ~100–150 ms for short developer questions;
- 3.5 ms with a cached query vector;
- peak RAM 2.3 GB; index on disk 71 MB.

Search itself is one matrix-vector product; the encoder's forward pass dominates.
Dense scoring as a numpy matrix product took search from 827 to 5 ms per query (163×).

## 23. The guidelines' worked example

The Theme 01 guidelines give three JavaScript snippets and the question "How is the
input preprocessed before going to the main function?", and expect Code#1
(`normalize`) > Code#2 (`check`) > Code#3 (`perf`). Indexed with `index-corpus` at
shipped defaults, ranked with `rank` and asked in the tool (`guideline_example.json`):
the question is routed to ranked search (not the call-order lookup), but the order is
**perf > check > normalize** (0.552 / 0.548 / 0.534), the reverse of the expected
order, within 0.018 and at low confidence. The snippets are four unindented lines with
placeholder names. Nothing was tuned for it; a permanent test pins the measured order.
A realistic-name version of the same idea (`handleUtterance`, `hasLocalePrefix`,
`playAlarmSound`) ranks in the expected order (0.591 > 0.498 > 0.410).

## 24. Limitations

Ordered by how likely each is to matter to a judge. Every item names its
evidence in `bench/results/`.

### Accuracy

1. **The guidelines' worked example ranks in reverse** on its verbatim snippets:
   perf > check > normalize, 0.552 / 0.548 / 0.534 (`guideline_example.json`). The
   snippets are four unindented lines with placeholder names, and the tool marks the
   result low confidence. Nothing was tuned for it. A realistic-name version of the
   same idea ranks in the expected order.
2. **The hubness correction is specific to AppsRetrieval.** Each document's hub score
   comes from the 5,000 AppsRetrieval *training* queries; no test query or test label
   is used. It is a table computed for this corpus (`data/apps_hubness.json`), which a
   reviewer may read as dataset-specific tuning. Three checks on the shipped table
   (`hub_table_checks.json`): +0.0331 on the 2,765 held-out queries, +0.0237 with every
   train-answer document removed from the corpus (both significant), and k and β
   chosen on the validated sample only. The earlier artifacts' scores are recorded in
   `appsretrieval_results.meta.json` and reproduce exactly (0.53926 with
   `run_mteb.py --code-view`, 0.50431 with `--code-view --no-hubness`). A new corpus
   needs its own queries; on npm the query-free variant was not significant, so
   repository indexes are not corrected. The gain was measured on the benchmark's full
   problem statements, not on short hand-typed questions. `:hubness off` in the tool and
   `search --no-hubness` switch it off.
3. **The encoder leans hard on identifiers.** Renaming local variables costs −0.37
   NDCG@10 on AppsRetrieval with either naming style; reordering statements costs
   nothing (`evolutionary.json`). The same rename on the team's **private** code, which
   no model can have seen, costs far less: 15% and 2% relative drop, against 56% on
   AppsRetrieval at a matched rename intensity and 29% on public npm
   (`memorisation.json`); the question types differ and the private samples are small
   (183 and 61). On other encoders (`rename_models.json`) gte loses more than granite
   (significant) but no more than jina, and every encoder leans on names.
   **Memorisation is therefore inconclusive**: neither established nor ruled out.
   granite is not a clean control: its model card lists its sources without naming
   CoIR, and the Granite R2 paper (arXiv 2508.21085) says it uses no CoIR training data
   but trains on Project CodeNet problem–solution pairs, the same kind of data as APPS.
4. **Two weak held-out tasks** (`generalisation.json`). DS1000: NDCG@10 51.78 with R@100
   97.2, because its answers are near-identical pandas/numpy snippets that are found but
   not ranked first. FreshStack: 31.84, questions about libraries newer than the model's
   training data, answered by repository chunks.
5. **Long solutions and non-English statements.** Beyond ~4,000 characters the
   1,024-token window truncates (17 queries, 20% in the top 10 vs 65%). Russian
   statements are at 41% vs 66% (`failures.json`).
6. **Confidence is calibrated on AppsRetrieval only.** High / medium / low hold out of
   sample there (AUC 0.818 on the hubness-corrected margins, `confidence.json`). On other
   code the labels are indicative.
7. **Real code evidence is two codebases.** The AppsRetrieval choices held up on npm
   (`repo_eval.json`) and on the team's private project (`local_repo_test.json`). That is
   good evidence, not a benchmark suite.

### Using it on your own code

8. **JS/TS extraction is a careful regex, not a parser.** It handles destructured
   parameters, no-semicolon arrow functions and regex literals
   (`tests/test_repo_extract.py`). Exotic syntax (decorators on JS classes, computed
   method names) can still be mis-split. Python uses the syntax tree.
9. **The first index takes time on a CPU.** npm's 2,506 first-party texts take 1,271 s
   cold on an idle machine; the 8,765-snippet benchmark 45–60 min. A new version then
   costs seconds to a few minutes (`p1_versions.json`). Builds resume after an
   interruption and print the time remaining.
10. **Test files are hidden by default in repository indexes.** That was measured as a
    gain (`local_repo_test.json`), but a question *about* tests needs `:tests on` (or
    `search --include-tests`).
11. **Structural questions are lexical.** "Who calls X", "where is X used" and
    "X before Y" match names in source order within a function: no call graph, no
    aliases, no callbacks. English questions are recognised by fixed patterns of at most
    14 words; anything else is answered by ranked search, and the "Understood as" line
    shows which happened (`english_questions.json`).
12. **The secret filter is deliberately conservative.** A snippet with a token-like
    literal is dropped even when the token is fake (one npm test was). A real secret is
    never indexed.
13. **Raw document text is used for AppsRetrieval only.** On repositories the raw view
    was not significantly better than the code view (pooled +0.012, `raw_repos.json`), so
    repository indexes keep the code view.

### Running it

14. **Cold start ~14 s** (model load ~12 s; 55 s on the first start after a reboot with a
    cold disk cache; `latency.json`). The tool pays it once per session; a one-shot
    `search` pays it per call.
15. **Query latency depends on query length.** p50 is 691 ms for AppsRetrieval's
    ~1,600-character statements, and ~100–150 ms for short developer questions; the
    encoder's forward pass dominates, the search itself takes a few ms.
16. **Exhaustive kNN.** Fine up to tens of thousands of snippets; a million-snippet
    repository would need an approximate-nearest-neighbour index.
17. **Single process, single user.** One SQLite file per index.
18. **Docker Desktop must be running first.** If its engine is not answering, the
    pipeline stops at step 1 with instructions (§12).
19. **Python 3.13 is untested** (not available on the test machine); 3.11, 3.12 and
    3.14 are tested (§12). 3.11 uses numpy 2.4.6 and scipy 1.17.1
    through environment markers.
20. **The pre-built index is tied to one build.** Its SHA-256 is pinned in
    `data/prebuilt_index.json`, so only the exported file with that hash imports.
    It is optional; the screening score does not use it.
21. **jina needs `trust_remote_code`.** It is kept only as an ablation baseline. The
    shipped model does not use it, and unknown models are refused it.

## 25. Security review

The review covers the whole codebase, including the optional pre-built index, and the full git history. Every finding that was fixed has a test, and each test was written
and seen failing before its fix (`tests/test_security_review.py`, and
`tests/test_portability.py` for line endings).

### Fixed

| # | finding | risk | fix | test |
|---|---|---|---|---|
| 1 | **Terminal injection.** Indexed code was printed as-is, so ESC sequences, OSC 8 hyperlinks, bell and backspace characters, or C1 controls in a malicious repository reached the terminal. | A repository could clear the screen, recolour or spoof the tool's output, or plant a hidden hyperlink. | `retrieval/display.py` shows every control character except newline and tab as a visible escape (`\x1b`). Applied to every line the tool prints and to the full-snippet view. `debug907 search`, `describe` and the `reproduce` example do the same. Rich markup in code was already printed literally (`markup=False`). | `test_indexed_code_cannot_send_control_sequences_to_the_terminal`, `test_the_search_command_also_strips_control_sequences` |
| 2 | **Symlinked files were read.** `os.walk` does not descend into linked folders, but it does list linked files, and the extractor read them. | `index-repo` on a hostile folder could index files outside it, e.g. a link to a key file (the secret filter matches by name, so a renamed link could slip past). | Links to files and to folders are skipped, and counted as "symbolic link" in the skip report. No cycles are possible. | `test_symlinked_files_and_folders_are_skipped_by_the_walk` (portable); `test_a_symlinked_file_pointing_outside_the_repo_is_not_indexed` (real links; runs on Linux, including inside the Docker build) |
| 3 | **Output paths.** `rank --out appsretrieval_results.json` would overwrite the shipped artifact. Any output could replace an unrelated existing file. | Losing the submission file, or a user's file, to a typo. | `cli.output_refused`: no output may be a shipped file (the artifact, its CSV, meta.json, backups, `data/`, `requirements.txt`, `Dockerfile`, `README.md`). An existing file can be replaced only by the same kind of output (rankings CSV, rankings JSONL, SQLite index). Applied to `rank`, `index-corpus`, `index-repo` and `sync`. | `test_rank_refuses_to_overwrite_the_shipped_artifact`, `test_outputs_never_overwrite_an_unrelated_existing_file` |
| 4 | **Option injection in `sync --rev`.** A revision starting with `-` reached `git rev-parse` as an argument. | git could read it as an option, e.g. an output path. | A revision starting with `-` is refused. All git calls already used argument lists, never a shell. | `test_sync_rejects_a_revision_that_looks_like_an_option` |
| 5 | **The Docker image ran as root.** | A container escape, or a malicious project mounted with `-Project`, would run with root rights in the container. | A `debug907` user (uid 1000) owns only `/app/out` and the baked model/dataset cache. `USER debug907`. | `test_the_docker_image_runs_as_a_non_root_user` |
| 6 | **SQL table allowlist used `assert`.** | `python -O` strips asserts, which would leave the interpolated table name unchecked. It was not exploitable (the names are constants), but the check could silently disappear. | An explicit `_check_table()` that raises. | covered by the existing index tests |
| 7 | **CRLF shell scripts in the working copy** (found by the Docker build). | A Docker image built from a Windows checkout failed. | Scripts normalised to LF; `Dockerfile` added to `.gitattributes`. | `test_shell_scripts_and_the_dockerfile_have_lf_line_endings_on_disk` |

### The optional pre-built index

`debug907 import-index` accepts a file from the internet, so it was built to trust
nothing but the hash pinned in this repository (`data/prebuilt_index.json`).
Code: `retrieval/prebuilt.py`; tests: `tests/test_prebuilt_index.py`.

| concern | how it is handled | test |
|---|---|---|
| A tampered or substituted file | The SHA-256 of the raw bytes is computed and compared **before anything opens the file**: neither gzip nor SQLite ever parses a file that is not byte-for-byte ours, so a malicious archive or database cannot reach either parser. A mismatch is refused. | `test_a_hash_mismatch_is_refused_before_the_file_is_opened` (a non-gzip file is refused with "nothing was opened") |
| Decompression bomb | Decompression stops and the file is refused at the pinned uncompressed size; the database's own SHA-256 is then checked as well. | covered by the pinned sizes; `test_a_good_import_is_searchable` checks the db hash path |
| A correct file for the wrong configuration | Opened read-only (`mode=ro`), and only these tables are allowed; the encoder signature, the view (`raw`), the corpus kind and the snippet count must match the shipped configuration. | `test_a_wrong_encoder_signature_is_refused`, `test_a_wrong_view_is_refused_even_with_matching_hashes` |
| Path traversal through the URL | The URL never chooses a path: a download is always written to `out/downloads/appsretrieval_index.db.gz` (as `.part` until complete), and is deleted after the install. Only `https://` is accepted, and the download is capped at the pinned size. Integrity rests on the pinned hash, not on the transport, so even a redirect cannot change what is installed. | `test_only_https_urls_and_a_fixed_download_name` |
| Overwriting the artifact or other files | The destination goes through `cli.output_refused`, so it can never be `appsretrieval_results.json` or any shipped file or anything under `data/`, and it must be an earlier SQLite index. An existing index is never replaced without `--force`. The temporary `.importing` file is removed on every refusal. | `test_import_can_never_write_over_a_shipped_file`, `test_an_existing_index_is_not_replaced_without_force` |
| Trusting the vectors | `debug907 verify-index` re-encodes a random sample from the stored source with no cache and compares both the text key and the vector (min cosine ≥ 0.9999). It opens the index read-only and needs no network. Measured on the release file: 200 of 200, min cosine 1.0000000. | `test_verify_index_passes_clean_and_catches_a_tampered_vector`, `test_verify_index_catches_edited_source_text` |

`export-index` refuses to write into `data/` or over a shipped file. It copies only the
tables above, and only the vectors that the snippets reference. It refuses any index
that is not the shipped AppsRetrieval configuration. The file holds public corpus
snippets and vectors only: no queries, no relevance labels, no rankings.

### Checked, no change needed

- **Archive extraction.** `debug907 sync` does not extract archives: it lists the commit (`git ls-tree`) and streams only the wanted blobs (`git cat-file`). It refuses any path that resolves outside its scratch folder, and skips symlink blobs (mode 120000). The one tar extraction (`tools/local_repo_test.py`) uses `filter="data"`.
- **Model loading.** The shipped model loads with `trust_remote_code=False`, enforced by `tests/test_security.py`. Only jina, an ablation-only model, may run remote code, and an unknown model name never can. The pinned snapshot ships `model.safetensors` and no pickle weights. There is no `torch.load`, `pickle`, `joblib` or `yaml.load` anywhere in the code.
- **Subprocesses.** Every `git` and `docker` call uses an argument list; there is no `shell=True`, `os.system` or `os.popen`. Commit ids passed on to `ls-tree` and `cat-file` are full hashes resolved by `rev-parse`.
- **Regex denial of service.** Seven adversarial files at the 400 KB size cap took at most 0.34 s each: unbalanced parentheses, 600-parameter functions, escaped quotes, `a*` regex literals, comment runs, 9,000 nested `def`s and 150-deep arrow functions. Lines over 2,000 characters are skipped as generated. The English-question router only sees one line of at most 14 words.
- **SQL / FTS5.** Every value is bound as a parameter. The FTS5 query is built from quoted, filtered terms (`tests/test_security.py::test_fts_query_quotes_and_filters_user_terms`). Table names come from the fixed allowlist (finding 6).
- **Artifact integrity across platforms.** The artifact's recorded SHA-256 had been taken from the Windows working copy (CRLF line endings), so a Linux or macOS clone would not have matched it. The result files are now byte-exact in git (`.gitattributes -text`), and `meta.json` records the one hash that holds everywhere (`ec84a471…`). `tests/test_screening_format.py::test_the_artifact_bytes_match_the_recorded_hash_on_every_platform`.
- **Secrets in the git history.** Scanned every commit for AWS, GitHub, Slack, OpenAI, Stripe, Google and Hugging Face tokens, private-key blocks, password/secret/token assignments and email addresses. The only hits are the privacy tests' fake fixtures: AWS's documented example key `AKIAIOSFODNN7EXAMPLE` and a truncated dummy RSA header. The indexer never indexes `.env*`, `*.pem`, `*.key` or credential files, nor a function that contains a token-shaped string (`tests/test_repo_privacy.py`).

### Remaining

- **A local path in old history: reviewed, left deliberately.** Two past commits contain a local Windows user-profile path (the drive, the Users folder, the account name and an OneDrive Desktop folder): `59cfcc3` (the report) and `64d6743` (tools/docker_pipeline.ps1). Re-reviewed: it is a folder path only, with no secret, token, password or file content. The current tree is clean, and `tests/test_portability.py` keeps it clean. The history is **deliberately not rewritten**: that would mean force-pushing published commits and changing every hash after them, including the commit the release will point to. It was the team's decision.
- **Real symlinks on Windows.** The real-symlink test is skipped on Windows without Developer Mode, because links cannot be created there. The portable test and the Linux Docker build cover the rule.
- **Image size.** The non-root user costs ~0.3 GB: `chown -R` copies the baked model and dataset
  cache into a new layer (image 0.87 → 1.16 GB). Creating the user before the download steps
  would avoid it.
- **Lexical lookups are not a sandbox.** `index-repo` reads the files it is pointed at, by design, and only reads them: nothing in the indexed folder is ever executed.
