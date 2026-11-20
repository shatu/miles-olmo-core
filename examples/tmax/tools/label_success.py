"""Per-task success labels from a rollout-only labeling run's traces (run_tmax_olmo.py --rollout-only).

Reads ``traces-*.jsonl`` under a run's trace directory (trace_dir/exp_name) and writes one row per task:

- ``n``: episodes the task's tests scored. Skipped: aborted and fleet-lost episodes (reward None) and episodes whose
  sandbox never started (``start_failed``: scored 0 for training, but not an outcome of the policy);
- ``successes``: scored episodes with reward >= --success-threshold (TMax rewards are 0 or 1);
- ``success_rate`` and the unbiased ``pass@k`` estimate 1 - C(n - c, k) / C(n, k) for each requested k <= n;
- ``ends``: end-reason counts; ``mean_tool_calls`` per scored episode.

    python label_success.py RUN_TRACE_DIR --out labels.csv [--k 1 2 4 8 16] [--strata pilot.jsonl.strata.csv]

With ``--strata``, the summary also reports the success rate per stratum of the prior (sample_prompts_by_difficulty.py).
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace_dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--success-threshold", type=float, default=1.0)
    parser.add_argument("--strata", help="the sampler's <out>.strata.csv, to summarize by stratum")
    args = parser.parse_args()

    tasks = defaultdict(lambda: {"n": 0, "successes": 0, "ends": Counter(), "tool_calls": 0})
    unscored = Counter()
    for path in sorted(Path(args.trace_dir).glob("traces-*.jsonl")):
        with open(path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line cut off by preemption
                metrics = record.get("metrics") or {}
                if record.get("reward") is None or metrics.get("end_reason") == "start_failed":
                    unscored[metrics.get("end_reason", "unknown")] += 1
                    continue
                task = tasks[record["task_id"]]
                task["n"] += 1
                task["successes"] += record["reward"] >= args.success_threshold
                task["ends"][metrics.get("end_reason", "unknown")] += 1
                task["tool_calls"] += metrics.get("tool_calls", 0)

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["task_id", "n", "successes", "success_rate", *[f"pass@{k}" for k in args.k], "mean_tool_calls", "ends"]
        )
        for task_id, t in sorted(tasks.items()):
            n, c = t["n"], t["successes"]
            writer.writerow(
                [
                    task_id,
                    n,
                    c,
                    round(c / n, 4),
                    *[round(pass_at_k(n, c, k), 4) if k <= n else "" for k in args.k],
                    round(t["tool_calls"] / n, 1),
                    json.dumps(dict(t["ends"]), sort_keys=True),
                ]
            )

    episodes = sum(t["n"] for t in tasks.values())
    solved = sum(t["successes"] > 0 for t in tasks.values())
    print(f"tasks {len(tasks)}  scored episodes {episodes}  unscored {sum(unscored.values())} {dict(unscored)}")
    if tasks:
        print(
            f"success rate {sum(t['successes'] for t in tasks.values()) / episodes:.4f}  tasks solved at least once {solved}"
        )
        n_hist = Counter(t["n"] for t in tasks.values())
        print(f"episodes per task: {dict(sorted(n_hist.items()))}")
        rate_hist = Counter(t["successes"] for t in tasks.values())
        print(f"tasks by successes: {dict(sorted(rate_hist.items()))}")
    if args.strata and tasks:
        with open(args.strata) as f:
            stratum = {row["task_id"]: row["stratum"] for row in csv.DictReader(f)}
        by = defaultdict(lambda: [0, 0, 0, 0])
        for task_id, t in tasks.items():
            s = by[stratum.get(task_id, "?")]
            s[0] += 1
            s[1] += t["n"]
            s[2] += t["successes"]
            s[3] += t["successes"] > 0
        for name, (count, n, c, any_success) in sorted(by.items()):
            print(
                f"stratum {name}: tasks {count}  success rate {c / max(n, 1):.4f}  solved at least once {any_success}"
            )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
