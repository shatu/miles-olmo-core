"""Build an ordered prompt file that starts RL on tasks the policy already solves, then mixes in new easy tasks.

Run it with run_tmax_olmo.py's ``rollout_shuffle: false`` so the order holds:

1. **Warm start:** the seed tasks (e.g. a labeling run's solved tasks) only, ``--warm-passes`` times, each pass in a
   new random order, topped up with seed tasks to a whole number of batches.
2. **Expansion:** every batch of ``--batch`` prompts holds ``--seed-per-batch`` seed tasks (cycled) and the rest new
   tasks, until the new tasks run out.

New tasks are the ``--n-new`` shortest task prompts among the candidates: tasks with at least ``--min-rollouts`` prior
rollouts and a prior success rate of at least ``--min-prior`` (e.g. the pooled Qwen TMax statistics), minus those in
``--exclude`` (e.g. a labeling run's labels, whose unsolved tasks gave no signal). On the Olmo 3.5 Tiny h250-b2e pilot,
prompt length predicted solvability better than the Qwen rate (AUROC 0.72 vs 0.57).

    python build_curriculum_prompts.py --prompts miles_prompts.jsonl --seed-tasks solvable_task_ids.txt \\
        --stats task_success_stats.csv --exclude labels.csv --n-new 800 --out curriculum.jsonl

``<out>.csv`` lists each position's task, its part (warm / seed / new), prompt length and prior.

``--extend CURRICULUM.jsonl`` continues an existing curriculum (and its ``.csv``) instead: the output starts with that
file's prompts unchanged, in order, then appends mixed batches of the next ``--n-new`` shortest candidates not already in
it. A run that trained on the existing file resumes from its checkpoint at the same prompt offset (``--exp-name``), so
it carries on into the appended tasks.
"""

import argparse
import csv
import json
import random


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", required=True, help="prompt JSONL with metadata.task_id per row")
    parser.add_argument("--seed-tasks", required=True, help="task ids to start from, one per line")
    parser.add_argument("--stats", required=True, help="prior CSV with task_id, n_rollouts, success_rate")
    parser.add_argument("--exclude", help="CSV with a task_id column (or a list of ids) to leave out of the new tasks")
    parser.add_argument("--n-new", type=int, default=800)
    parser.add_argument("--min-prior", type=float, default=0.3)
    parser.add_argument("--min-rollouts", type=int, default=20)
    parser.add_argument("--warm-passes", type=int, default=2)
    parser.add_argument("--batch", type=int, default=16, help="prompts per rollout step (rollout_batch_size)")
    parser.add_argument("--seed-per-batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--extend", help="existing curriculum JSONL (with its .csv) to keep as the prefix and append to"
    )
    args = parser.parse_args()

    prompts = {}
    with open(args.prompts) as f:
        for line in f:
            prompts[json.loads(line)["metadata"]["task_id"]] = line
    seeds = [t for t in open(args.seed_tasks).read().split() if t]
    missing = [t for t in seeds if t not in prompts]
    if missing:
        raise SystemExit(f"{len(missing)} seed tasks are not in {args.prompts}, e.g. {missing[:3]}")
    excluded = set(seeds)
    if args.exclude:
        with open(args.exclude) as f:
            text = f.read()
        excluded |= (
            {row["task_id"] for row in csv.DictReader(text.splitlines())}
            if "task_id" in text[:200]
            else set(text.split())
        )
    prior = {}
    with open(args.stats) as f:
        for row in csv.DictReader(f):
            if int(row["n_rollouts"]) >= args.min_rollouts:
                prior[row["task_id"]] = float(row["success_rate"])

    def length(task):
        return len(json.loads(prompts[task])["prompt"][-1]["content"])

    prefix_lines, prefix = [], []
    if args.extend:
        with open(args.extend) as f:
            prefix_lines = f.readlines()
        with open(f"{args.extend}.csv") as f:
            prefix = [(row["task_id"], row["part"]) for row in csv.DictReader(f)]
        if [json.loads(line)["metadata"]["task_id"] for line in prefix_lines] != [t for t, _ in prefix]:
            raise SystemExit(f"{args.extend} and its .csv list different tasks")
        excluded |= {t for t, _ in prefix}

    candidates = [t for t in prompts if t not in excluded and prior.get(t, -1) >= args.min_prior]
    new = sorted(candidates, key=lambda t: (length(t), t))[: args.n_new]
    print(f"new tasks: the {len(new)} shortest of {len(candidates)} candidates (prompt chars <= {length(new[-1])})")

    rng = random.Random(args.seed)
    order = []
    while not args.extend and (len(order) < args.warm_passes * len(seeds) or len(order) % args.batch):
        block = seeds[:]
        rng.shuffle(block)
        order += [(t, "warm") for t in block]
    # Fill whole batches only, so every mixed batch below lands on a batch boundary.
    if not args.extend:
        order = order[: max(args.warm_passes * len(seeds) + (-args.warm_passes * len(seeds)) % args.batch, args.batch)]
    rng.shuffle(new)
    per_batch_new = args.batch - args.seed_per_batch
    cycle = seeds[:]
    rng.shuffle(cycle)
    position = 0
    for start in range(0, len(new), per_batch_new):
        batch = [(t, "new") for t in new[start : start + per_batch_new]]
        for _ in range(args.seed_per_batch):
            batch.append((cycle[position % len(cycle)], "seed"))
            position += 1
        rng.shuffle(batch)
        order += batch

    with open(args.out, "w") as f:
        f.writelines(prefix_lines)
        for task, _ in order:
            f.write(prompts[task])
    with open(f"{args.out}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["position", "task_id", "part", "prompt_chars", "prior_success_rate"])
        for i, (task, part) in enumerate(prefix + order):
            writer.writerow([i, task, part, length(task), prior.get(task, "")])
    warm = sum(part == "warm" for _, part in order)
    print(
        f"wrote {len(prefix) + len(order)} prompts: {len(prefix)} from {args.extend or 'no existing curriculum'}, "
        f"{warm} warm-start ({warm / args.batch:.1f} batches of {args.batch}), then {len(order) - warm} mixed "
        f"({args.seed_per_batch} seed + {per_batch_new} new per batch)"
    )


if __name__ == "__main__":
    main()
