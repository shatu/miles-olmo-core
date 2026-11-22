"""build_curriculum_prompts.py: a fresh curriculum, and --extend keeping an existing one as an unchanged prefix."""

import csv
import json
import subprocess
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[4] / "examples" / "tmax" / "tools" / "build_curriculum_prompts.py"


def _inputs(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    with prompts.open("w") as f:
        for i in range(40):
            # Task i's prompt has 100 + i characters, so shorter tasks have lower ids.
            row = {"prompt": [{"role": "user", "content": "x" * (100 + i)}], "metadata": {"task_id": f"t{i:02d}"}}
            f.write(json.dumps(row) + "\n")
    seeds = tmp_path / "seeds.txt"
    seeds.write_text("t30\nt31\nt32\n")
    stats = tmp_path / "stats.csv"
    with stats.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "n_rollouts", "success_rate"])
        for i in range(40):
            writer.writerow([f"t{i:02d}", 50, 0.1 if i % 5 == 4 else 0.5])  # every fifth task is below --min-prior
    return prompts, seeds, stats


def _build(prompts, seeds, stats, out, *extra):
    subprocess.run(
        [sys.executable, str(TOOL), "--prompts", str(prompts), "--seed-tasks", str(seeds), "--stats", str(stats)]
        + ["--batch", "4", "--seed-per-batch", "1", "--out", str(out), *extra],
        check=True,
        capture_output=True,
    )
    lines = out.read_text().splitlines()
    rows = list(csv.DictReader(open(f"{out}.csv")))
    assert [json.loads(line)["metadata"]["task_id"] for line in lines] == [row["task_id"] for row in rows]
    return lines, rows


def test_extend_keeps_the_prefix_and_appends_the_next_shortest_tasks(tmp_path):
    prompts, seeds, stats = _inputs(tmp_path)
    base, base_rows = _build(prompts, seeds, stats, tmp_path / "base.jsonl", "--n-new", "6")
    base_new = {row["task_id"] for row in base_rows if row["part"] == "new"}
    assert base_new == {"t00", "t01", "t02", "t03", "t05", "t06"}  # the 6 shortest above --min-prior

    extended, rows = _build(
        prompts, seeds, stats, tmp_path / "extended.jsonl", "--n-new", "6", "--extend", str(tmp_path / "base.jsonl")
    )
    assert extended[: len(base)] == base
    appended = rows[len(base) :]
    assert {row["part"] for row in appended} == {"new", "seed"}
    assert {row["task_id"] for row in appended if row["part"] == "new"} == {"t07", "t08", "t10", "t11", "t12", "t13"}
    assert {row["task_id"] for row in appended if row["part"] == "seed"} <= {"t30", "t31", "t32"}
    assert [int(row["position"]) for row in rows] == list(range(len(rows)))
