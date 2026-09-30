# debug907

> **Demo:** [https://youtu.be/vh2xW-vu988]

Finding something in an unfamiliar codebase can take longer than it should. You may know what the code is supposed to do, but not the function name or the file where it lives.

debug907 is a local code-search tool for that situation. Ask a question in normal English and it shows the most relevant code snippets, along with their file paths and line numbers. It was made for Samsung PRISM GenAI Hackathon 2026–27, Theme 01.

## What you can do with it

- Search code by describing what you need.
- Ask direct questions like `where is setrecursionlimit used` or `who calls getCredentialsByURI`.
- Index JavaScript, TypeScript, and Python projects.
- Search different versions of the same project.
- Rank a file of queries against a code corpus and save the result as CSV.
- Run the submitted CoIR AppsRetrieval benchmark setup.

It also avoids indexing obvious clutter such as `node_modules`, generated files, virtual environments, symlinks, `.env` files, keys, and content that looks like a secret.

## Result

For the CoIR AppsRetrieval test split, the submitted run got these results:

| Metric | Score |
|---|---:|
| NDCG@10 | **0.60952** |
| MRR@10 | **0.56320** |
| Recall@100 | **0.92457** |

The saved submission result is in `appsretrieval_results.json`. The detailed report, including the experiments behind these numbers, is in [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md).

## How it works

For normal questions, debug907 turns both the question and the code into vectors using `gte-modernbert-base`, then looks for the closest matches. For a few very specific question types—such as “where is X used?” or “who calls X?”—it does an exact source lookup instead. This is faster and usually gives a cleaner answer for those cases.

The AppsRetrieval benchmark has one extra hubness-correction step. That is only used for the benchmark because that is where it was tested; searches in your own repository use normal cosine similarity.

## Tech used

- Python
- PyTorch, Transformers, and Sentence Transformers
- `Alibaba-NLP/gte-modernbert-base` for embeddings
- SQLite for storing snippets, vectors, and index information
- MTEB and CoIR AppsRetrieval for evaluation
- Docker for a repeatable container setup

## Folder guide

```text
cli.py                  command-line commands
app.py                  interactive terminal app
encoder.py              encoder used for MTEB evaluation
retrieval/              indexing, search, embeddings, routing, versions
pipeline/               code extraction and text processing
bench/                  evaluation code and metrics
tests/                  test suite
tools/                  helper scripts, scoring, Docker workflow
data/                   sample data and index metadata
docs/TECHNICAL_REPORT.md full project report
```

## Setup

You need Python 3.11, 3.12, or 3.14. The first run downloads the model once (roughly 300 MB). After that, the project can run offline.

From the folder that contains `cli.py`:

```bash
python -m venv .venv
```

Activate it:

```powershell
# Windows PowerShell
.venv\Scripts\activate
```

```bash
# Linux/macOS
source .venv/bin/activate
```

On Linux, install the CPU version of PyTorch first:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
```

Then install everything else and check that the setup is okay:

```bash
python -m pip install -r requirements.txt
python cli.py doctor
```

The last command should end with `result  READY`.

## Using the tool

### Search the AppsRetrieval dataset

First, build the index. This is the slow part: the full benchmark normally takes about 45–60 minutes on a laptop CPU. If it stops halfway through, running the same command again continues from where it left off.

```bash
python cli.py index
python cli.py
```

Once the prompt opens, try questions like these:

```text
how does npm decide that an installed package is outdated
where is setrecursionlimit used
who calls getCredentialsByURI
which functions call input before sorted
```

For a one-time search, you do not need to open the interactive tool:

```bash
python cli.py search "reverse a linked list"
```

### Search a project on your computer

```bash
python cli.py index-repo path/to/your/project --out out/myproject.db
python cli.py --db out/myproject.db
```

The index is created in `out/`; the project you are indexing is only read, not changed. JavaScript, TypeScript, and Python are handled at function level. Test files stay out of the normal results unless you turn them on in the tool.

### Search old and new versions of a Git repository

```bash
python cli.py sync path/to/your/repository --rev HEAD --out out/project_versions.db
python cli.py interactive --db out/project_versions.db --all-versions
```

`sync` only reads Git data. It does not check out commits or modify the repository. Code that has not changed keeps the same saved embedding, so later versions are much quicker to add.

### Batch ranking

If you have a file of questions, use `rank` to get a CSV of results. Query files can be JSONL, JSON, CSV, or TSV and should have fields such as `id` and `text`.

```bash
python cli.py rank --queries data/example_questions.jsonl --out out/my_rankings.csv
```

The output has four columns: `query_id`, `corpus_id`, `rank`, and `score`.

To score the output against your relevance labels:

```bash
python cli.py eval --qrels path/to/qrels.jsonl --rankings out/my_rankings.csv
```

## Commands

There is no web API in this project. It is a command-line tool.

| Command | What it does |
|---|---|
| `index` | Builds the AppsRetrieval index |
| `index-repo <folder>` | Builds an index for your JS, TS, or Python project |
| `search "question"` | Searches an index once and prints the results |
| `interactive` | Opens the terminal interface |
| `sync <repo>` | Adds a Git commit as a searchable version |
| `rank --queries ... --out ...` | Writes query rankings to a CSV file |
| `reproduce` | Runs the checked benchmark reproduction |
| `doctor` | Checks the environment and model setup |

Use `python cli.py --help` for the full command list.

## Docker

Docker is optional. It is useful when you want a clean, repeatable setup without installing Python packages yourself. The image runs tests while it is built and runs as a non-root user.

```powershell
# Windows PowerShell
powershell -ExecutionPolicy Bypass -File .\tools\docker_pipeline.ps1
```

```bash
# Linux/macOS
bash tools/docker_pipeline.sh
```

Make sure Docker Desktop is open and its engine is running before you start. Add `-Interactive` in PowerShell or `--interactive` on Linux/macOS to open the terminal tool inside the container.

## Testing

The project has unit and integration tests for the extraction, indexing, search, routing, versioning, security checks, and benchmark workflow.

```bash
python -m pytest -q
```

There is also a quick end-to-end check:

```bash
python cli.py selftest
```

To reproduce the validated 1,000-query sample:

```bash
python cli.py reproduce
```

For the full AppsRetrieval evaluation:

```bash
python cli.py reproduce --full
```

## Why a few things were done this way

- **SQLite:** one portable file is easier to move around and inspect than setting up a separate database server.
- **Function-sized snippets:** a result should take you to a useful part of the code, not dump an entire file on you.
- **Incremental indexing:** embedding code is the expensive part, so unchanged snippets are reused between versions.
- **Simple exact lookups:** for questions about calls, definitions, and usage, direct matching is more predictable than semantic search.
- **Strict secret filtering:** it may skip the occasional fake token in a test, but it is safer than accidentally putting a real one into an index.
- **CPU support:** a GPU helps with speed, but it should not be required just to use the project.

## Current limitations

- The first large index takes time on CPU. AppsRetrieval takes roughly 45–60 minutes.
- JS and TS extraction is regex-based, so very unusual syntax can confuse it. Python uses the AST.
- “Who calls X?” and similar direct lookups are lexical. They are not a full call graph and do not resolve aliases or callbacks.
- Search checks every vector, which works well for normal-sized projects but will need an approximate index for very large codebases.
- The confidence labels and hubness correction were measured on AppsRetrieval, so they should not be treated as guarantees for every repository.

Possible next steps are parser-based JS/TS extraction, approximate nearest-neighbour search for much bigger indexes, and a small web interface.

## More information

The full report is here: [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md). It covers model selection, evaluation, failure cases, security review, and the reasoning behind the final setup.

Created by Kamneev Singh and Aadvik Chawla.
