"""Recount the runs behind the manuscript from the published records.

The article states that it reports 652 runs and that each one resolves to a file
in this release. This script checks that claim against the records as shipped,
so a reviewer can confirm the figure without rerunning anything:

    python scripts/count_runs.py

Exit status is 0 when the four published record sets add up to 652 and 1 when
they do not.

Why the four sets are separate directories rather than one. A run is identified
by its protocol, transfer direction, model family, seed, and the size of the
training set it saw. That last field matters: the cross-station runs were
repeated on the full 1.21M flows as a check on the 300k subsample, and in the
working tree those repeats were stored under two different names. Counting
files there would have double-counted them. Here each run appears once.

    results/grid/           200   five lightweight families, 300k subsample
    results/grid_deep/       40   two deep tabular families, 300k subsample
    results/grid_full_1m/    12   full 1.21M verification
    results/decomposition/  400   composition design and prior control
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "grid": 200,
    "grid_deep": 40,
    "grid_full_1m": 12,
    "decomposition": 400,
}
EXPECTED_TOTAL = 652


def main() -> int:
    counts = {}
    seeds = defaultdict(set)
    models = Counter()
    splits = Counter()
    problems = []

    for name in EXPECTED:
        directory = ROOT / "results" / name
        if not directory.is_dir():
            problems.append(f"missing directory results/{name}/")
            counts[name] = 0
            continue
        records = sorted(directory.glob("*.json"))
        counts[name] = len(records)
        for path in records:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                problems.append(f"unreadable: {path.name}: {exc}")
                continue
            if name != "decomposition":
                models[str(record.get("model"))] += 1
                seeds[str(record.get("model"))].add(str(record.get("seed")))
            splits[str(record.get("split"))] += 1

    width = max(len(n) for n in EXPECTED)
    print("published record sets")
    for name, expected in EXPECTED.items():
        got = counts[name]
        flag = "" if got == expected else f"   expected {expected}"
        print(f"  results/{name + '/':<{width + 1}}  {got:>4}{flag}")
    total = sum(counts.values())
    print(f"  {'total':<{width + 10}}  {total:>4}")

    print("\nmodel families in the accuracy grid")
    for model in sorted(models):
        print(f"  {model:<14}{models[model]:>4} runs over "
              f"{len(seeds[model]):>2} seeds")

    print("\nprotocols")
    for split in sorted(splits):
        print(f"  {split:<28}{splits[split]:>4}")

    for problem in problems:
        print(f"\nPROBLEM: {problem}")

    print()
    if total == EXPECTED_TOTAL and not problems:
        print(f"MATCH: {total} runs, as the article reports.")
        return 0
    print(f"MISMATCH: {total} runs published, article reports {EXPECTED_TOTAL}.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
