import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mteb

from encoder import PrePostPipelineEncoder

OUT = Path("out/guidelines_template_results.json")


def main() -> None:
    model = PrePostPipelineEncoder()
    task = mteb.get_task("AppsRetrieval")
    result = mteb.evaluate(
        model,
        [task],
        encode_kwargs={"batch_size": 64},
    )
    task_result = list(result.task_results)[0]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(task_result.to_dict(), f, indent=2, default=str)


def check() -> int:
    got = json.loads(OUT.read_text(encoding="utf-8"))["scores"]["test"][0]
    want = json.loads(Path("appsretrieval_results.json").read_text(encoding="utf-8"))["scores"]["test"][0]
    ok = all(abs(got[k] - want[k]) <= 1e-5 for k in ("ndcg_at_10", "mrr_at_10"))
    for k in ("ndcg_at_10", "mrr_at_10", "recall_at_100"):
        print(f"{k:<14} template {got[k]:.5f}   artifact {want[k]:.5f}")
    print("TEMPLATE REPRODUCES THE ARTIFACT" if ok else "TEMPLATE DOES NOT MATCH THE ARTIFACT")
    return 0 if ok else 1


if __name__ == "__main__":
    main()
    raise SystemExit(check())
