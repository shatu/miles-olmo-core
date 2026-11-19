"""Sample a prompt subset stratified by a prior per-task success rate, e.g. for a success-labeling pilot.

The prior is a CSV with ``task_id``, ``n_rollouts`` and ``success_rate`` columns, such as the pooled Qwen TMax
statistics in ``rollout_partitions/terminal_v1_since_jun2_tmax15k/task_success_stats.csv``. Tasks the prior never
solved are left out unless a stratum starts at 0 and ``--keep-unsolved`` is given.

    python sample_prompts_by_difficulty.py --prompts miles_prompts.jsonl --stats task_success_stats.csv \\
        --strata 0.7:1:400 0.5:0.7:300 0.1:0.5:200 0:0.1:100 --min-rollouts 20 --seed 0 --out pilot.jsonl

Each stratum ``LO:HI:N`` takes N tasks with LO <= success_rate < HI (HI = 1 includes 1). The output keeps the prompt
rows unchanged, in random order; ``<out>.strata.csv`` lists each task's stratum and prior.
"""

import argparse
import csv
import json
import random


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", required=True, help="prompt JSONL with metadata.task_id per row")
    parser.add_argument("--stats", required=True, help="CSV with task_id, n_rollouts, success_rate")
    parser.add_argument("--strata", nargs="+", required=True, help="LO:HI:N per stratum")
    parser.add_argument("--min-rollouts", type=int, default=20, help="ignore tasks with fewer prior rollouts")
    parser.add_argument("--keep-unsolved", action="store_true", help="allow tasks the prior never solved")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    prompts = {}
    with open(args.prompts) as f:
        for line in f:
            prompts[json.loads(line)["metadata"]["task_id"]] = line
    prior = {}
    with open(args.stats) as f:
        for row in csv.DictReader(f):
            if row["task_id"] in prompts and int(row["n_rollouts"]) >= args.min_rollouts:
                prior[row["task_id"]] = float(row["success_rate"])

    rng = random.Random(args.seed)
    chosen = []
    for spec in args.strata:
        lo, hi, n = spec.split(":")
        lo, hi, n = float(lo), float(hi), int(n)
        pool = sorted(
            task
            for task, rate in prior.items()
            if (lo <= rate < hi or (hi >= 1 and rate == 1)) and (args.keep_unsolved or rate > 0)
        )
        if len(pool) < n:
            raise SystemExit(f"stratum {spec}: only {len(pool)} tasks")
        chosen += [(task, spec, prior[task]) for task in rng.sample(pool, n)]
        print(f"stratum {spec}: {n} of {len(pool)} tasks")

    rng.shuffle(chosen)
    with open(args.out, "w") as f:
        for task, _, _ in chosen:
            f.write(prompts[task])
    with open(f"{args.out}.strata.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "stratum", "prior_success_rate"])
        writer.writerows(chosen)
    print(f"wrote {len(chosen)} prompts to {args.out}")


if __name__ == "__main__":
    main()
