#!/usr/bin/env bash
set -uo pipefail

IMAGE=debug907:submission
VOLUME=debug907-out
THREADS=8
PROJECT="" INTERACTIVE=0 FRESH=0 CLEAN=0 SKIP_BUILD=0 DRY_RUN=0 SELF_TEST=0

while [ $# -gt 0 ]; do
  case "$1" in
    --project) PROJECT="${2:?--project needs a path}"; shift 2 ;;
    --interactive) INTERACTIVE=1; shift ;;
    --threads) THREADS="${2:?--threads needs a number}"; shift 2 ;;
    --fresh) FRESH=1; shift ;;
    --clean) CLEAN=1; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --self-test) SELF_TEST=1; shift ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (try --help)"; exit 2 ;;
  esac
done

if [ -n "$PROJECT" ]; then
  [ -d "$PROJECT" ] || { echo "project folder not found: $PROJECT"; exit 1; }
  PROJECT="$(cd "$PROJECT" && (pwd -W 2>/dev/null || pwd))"
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if (cd "$ROOT" && pwd -W) >/dev/null 2>&1; then
  ROOT="$(cd "$ROOT" && pwd -W)"; export MSYS_NO_PATHCONV=1
fi
OUT="$ROOT/out"
STEP=0

command -v docker >/dev/null 2>&1 || { echo "docker not found on PATH. Install Docker first."; exit 1; }

step() { STEP=$((STEP + 1)); printf '\n== %s. %s ==\n' "$STEP" "$1"; }
fail() { printf '\nFAILED at step %s: %s\n' "$STEP" "$1"; exit 1; }
INFO_TIMEOUT="${DOCKER_INFO_TIMEOUT:-15}"
START_WAIT="${DOCKER_START_WAIT:-180}"
daemon_up() {
  if command -v timeout >/dev/null 2>&1; then
    timeout "$INFO_TIMEOUT" docker info >/dev/null 2>&1
  else
    docker info >/dev/null 2>&1
  fi
}
docker_help() {
  cat <<'EOF'

Docker is not running, or its engine is not answering.
  Windows / macOS: Start Docker Desktop (Start menu / Applications), wait until
                   it shows "Engine running", then run this command again.
                   If it stays on "Starting", quit Docker Desktop and start it again.
  Linux:           sudo systemctl start docker
EOF
}

run_docker() {
  echo "   > docker $*"
  [ "$DRY_RUN" = 1 ] && return 0
  local started=$SECONDS
  docker "$@"
  local code=$?
  echo "   ($((SECONDS - started))s)"
  [ "$code" -eq 0 ] || fail "docker $1 exited $code"
}

pipeline() { run_docker run --rm --network none -v "$VOLUME:/app/out" "$IMAGE" "$@"; }
pipeline_tty() { run_docker run --rm -it --network none -v "$VOLUME:/app/out" "$IMAGE" "$@"; }

cd "$ROOT" || exit 1

if [ "$SELF_TEST" = 1 ]; then
  echo "self test: $(command -v docker)"
  echo "   client : $(docker --version)" || exit 1
  if daemon_up; then echo "   daemon : RUNNING"; else echo "   daemon : not running"; fi
  echo "self test passed - the docker call path works."
  exit 0
fi

echo "debug907 docker pipeline   repo: $ROOT   threads: $THREADS"
[ "$DRY_RUN" = 1 ] && echo "DRY RUN - nothing will be executed"
[ -n "$PROJECT" ] && echo "Reminder: pause any cloud sync (OneDrive, Dropbox) - the project index is written to the repo's out/ folder."

step "Docker daemon running"
if [ "$DRY_RUN" = 0 ] && ! daemon_up; then
  if [ "$START_WAIT" -gt 0 ] && [ "$(uname -s)" = "Darwin" ] && open -a Docker 2>/dev/null; then
    echo "   starting Docker Desktop and waiting up to ${START_WAIT}s..."
    waited=0
    while [ "$waited" -lt "$START_WAIT" ]; do sleep 5; waited=$((waited + 5)); daemon_up && break; done
  fi
  if ! daemon_up; then
    docker_help
    fail "Docker is not running (see above)"
  fi
fi
echo "   docker is running"

if [ "$SKIP_BUILD" = 1 ]; then
  step "build (skipped, reusing existing image)"
else
  step "build image$([ "$CLEAN" = 1 ] && echo ' (--no-cache)') - bakes model + dataset, runs pytest + selftest"
  if [ "$CLEAN" = 1 ]; then run_docker build --no-cache -t "$IMAGE" .
  else run_docker build -t "$IMAGE" .; fi
  if [ "$DRY_RUN" = 0 ]; then
    docker image inspect "$IMAGE" --format '{{.Size}}' | awk '{printf "   image size: %.2f GB\n", $1/1e9}'
  fi
fi

step "doctor - environment and encoder (network disabled)"
run_docker run --rm --network none "$IMAGE" doctor

if [ "$FRESH" = 1 ] && [ "$DRY_RUN" = 0 ]; then
  docker volume rm -f "$VOLUME" >/dev/null 2>&1
  echo "   (--fresh) removed volume $VOLUME - the next step is a full cold encode"
fi
step "real index - full AppsRetrieval corpus, encoded inside the container"
echo "   first run encodes all 8,765 snippets (~45-60 min on $THREADS threads); later runs reuse the volume"
pipeline index --db out/real.db --threads "$THREADS"

step "reproduce - shipped config on the validated 1,000-query sample (must hit 0.6206)"
pipeline reproduce --db out/real.db --threads "$THREADS"

echo
if [ "$DRY_RUN" = 1 ]; then
  echo "DRY RUN complete - nothing was executed, nothing was verified."
else
  echo "PIPELINE PASSED - the image reproduces the shipped result on real data, offline."
fi

if [ "$INTERACTIVE" = 1 ]; then
  step "interactive search on the real corpus (type :quit to exit)"
  pipeline_tty interactive --db out/real.db
fi

if [ -n "$PROJECT" ]; then
  PROJECT_PATH="$PROJECT"
  NAME="$(basename "$PROJECT_PATH" | sed 's/[^A-Za-z0-9_-]/_/g')"
  DB="out/$NAME.db"
  mkdir -p "$OUT"

  step "index your project: $PROJECT_PATH"
  run_docker run --rm --network none -v "$PROJECT_PATH:/code:ro" -v "$OUT:/app/out" \
    "$IMAGE" index-repo /code --out "$DB" --threads "$THREADS"

  step "interactive search on your project (type :quit to exit)"
  run_docker run --rm -it --network none -v "$OUT:/app/out" "$IMAGE" interactive --db "$DB"
fi
