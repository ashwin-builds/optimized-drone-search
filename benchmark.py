"""
Benchmark the continuous algorithms on the UW scenario.

Every algorithm runs on the same seeds, so seed s gives every algorithm the
same people at the same positions, spotted at the same point of their cell's
search. That makes paired comparisons (algorithm minus baseline, seed by seed)
much tighter than comparing two independent averages.

    python benchmark.py --drones 10 20 50 --people 60 250 --runs 30
    python benchmark.py --drones 20 --people 250 --runs 100 --csv results.csv --check

Reports, per (drones, people): mean and 95% confidence interval of each
metric, and for each algorithm the paired difference from the baseline.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field

from continuous_model import CONTINUOUS_ALGORITHMS
from metrics import summarize
from scenario import DEPOT_DISTANCE_KM, Scenario

# two-sided 95% Student t critical values by degrees of freedom
_T95 = [(1, 12.706), (2, 4.303), (3, 3.182), (4, 2.776), (5, 2.571), (6, 2.447), (7, 2.365),
        (8, 2.306), (9, 2.262), (10, 2.228), (12, 2.179), (15, 2.131), (20, 2.086),
        (25, 2.060), (30, 2.042), (40, 2.021), (60, 2.000), (120, 1.980)]

SUMMARY_METRICS = ["completion", "mean_wait", "max_wait", "all_found", "unused_kits", "trips"]


def t95(df: int) -> float:
    crit = 1.960
    for d, value in reversed(_T95):
        if df >= d:
            return value if df < 1000 else crit
    return _T95[0][1]


def mean_ci(values: list[float]) -> tuple[float, float]:
    """Mean and half-width of its 95% confidence interval."""
    if not values:
        return math.nan, math.nan
    m = statistics.fmean(values)
    if len(values) < 2:
        return m, math.nan
    return m, t95(len(values) - 1) * statistics.stdev(values) / math.sqrt(len(values))


@dataclass
class BenchmarkConfig:
    drones: list[int] = field(default_factory=lambda: [20])
    people: list[int] = field(default_factory=lambda: [60])
    runs: int = 30
    seed: int = 0
    algos: list[str] = field(default_factory=lambda: list(CONTINUOUS_ALGORITHMS))
    baseline: str = "brute"
    depot: str = "redmond"
    cell_side: float = 25.0
    check: bool = False

    def jobs(self) -> list[tuple]:
        return [(algo, n, p, seed, self.depot, self.cell_side, self.check)
                for n in self.drones for p in self.people
                for seed in range(self.seed, self.seed + self.runs)
                for algo in self.algos]


def run_one(job: tuple) -> dict:
    algo, n, p, seed, depot, cell_side, check = job
    sc = Scenario(depot=depot, cell_side_target_m=cell_side)
    model = CONTINUOUS_ALGORITHMS[algo](n, n_people=p, seed=seed, check=check,
                                        **sc.model_kwargs())
    model.run()
    return {"algo": algo, "drones": n, "people": p, "seed": seed, **summarize(model)}


def run_benchmark(cfg: BenchmarkConfig, workers: int | None = None, progress=None) -> list[dict]:
    """Run every job; progress(done, total) is called as results come in."""
    jobs = cfg.jobs()
    rows = []
    workers = workers or os.cpu_count() or 1
    if workers == 1:
        for i, job in enumerate(jobs, 1):
            rows.append(run_one(job))
            if progress:
                progress(i, len(jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_one, job) for job in jobs]
            for i, fut in enumerate(as_completed(futures), 1):
                rows.append(fut.result())
                if progress:
                    progress(i, len(jobs))
    rows.sort(key=lambda r: (r["drones"], r["people"], r["algo"], r["seed"]))
    return rows


def summarize_rows(rows: list[dict], baseline: str = "brute",
                   metrics: list[str] = SUMMARY_METRICS) -> list[dict]:
    """One summary per (drones, people, algo): mean/CI per metric, paired diff vs baseline."""
    groups: dict[tuple, dict[str, dict[int, dict]]] = {}
    for r in rows:
        groups.setdefault((r["drones"], r["people"]), {}).setdefault(r["algo"], {})[r["seed"]] = r
    out = []
    for (n, p), by_algo in sorted(groups.items()):
        base = by_algo.get(baseline)
        for algo, by_seed in by_algo.items():
            s = {"drones": n, "people": p, "algo": algo, "runs": len(by_seed)}
            for m in metrics:
                s[m], s[f"{m}_ci"] = mean_ci([r[m] for r in by_seed.values()])
            if base is not None and algo != baseline:
                seeds = sorted(set(by_seed) & set(base))
                for m in ("completion", "mean_wait"):
                    diffs = [by_seed[k][m] - base[k][m] for k in seeds]
                    s[f"{m}_diff"], s[f"{m}_diff_ci"] = mean_ci(diffs)
                comp = [(by_seed[k]["completion"], base[k]["completion"]) for k in seeds]
                s["faster"] = sum(a < b - 1e-9 for a, b in comp)
                s["slower"] = sum(a > b + 1e-9 for a, b in comp)
            out.append(s)
    return out


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: list[dict], baseline: str) -> None:
    name = lambda key: CONTINUOUS_ALGORITHMS[key].name
    last = None
    for s in summary:
        if (s["drones"], s["people"]) != last:
            last = (s["drones"], s["people"])
            print(f"\n{s['drones']} drones, {s['people']} people, {s['runs']} runs")
            print(f"  {'algorithm':<16} {'completion (h)':>17} {'mean wait (min)':>17} "
                  f"{'max wait':>9} {'all found':>10} {'unused':>7}   vs {name(baseline)}")
        vs = ""
        if "completion_diff" in s:
            vs = (f"{s['completion_diff'] / 60:+.2f} ± {s['completion_diff_ci'] / 60:.2f} h, "
                  f"wait {s['mean_wait_diff']:+.1f} ± {s['mean_wait_diff_ci']:.1f} min, "
                  f"faster in {s['faster']}/{s['runs']}")
        print(f"  {name(s['algo']):<16} {s['completion'] / 60:>9.2f} ± {s['completion_ci'] / 60:<5.2f}"
              f" {s['mean_wait']:>9.1f} ± {s['mean_wait_ci']:<5.1f} {s['max_wait']:>9.1f}"
              f" {s['all_found'] / 60:>9.2f}h {s['unused_kits']:>7.1f}   {vs}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark the continuous algorithms.")
    parser.add_argument("--drones", type=int, nargs="+", default=[20])
    parser.add_argument("--people", type=int, nargs="+", default=[60])
    parser.add_argument("--runs", type=int, default=30, help="seeds per configuration")
    parser.add_argument("--seed", type=int, default=0, help="first seed")
    parser.add_argument("--algos", nargs="+", choices=sorted(CONTINUOUS_ALGORITHMS),
                        default=list(CONTINUOUS_ALGORITHMS))
    parser.add_argument("--baseline", choices=sorted(CONTINUOUS_ALGORITHMS), default="brute")
    parser.add_argument("--depot", choices=sorted(DEPOT_DISTANCE_KM), default="redmond")
    parser.add_argument("--cell-side", type=float, default=25.0)
    parser.add_argument("--check", action="store_true", help="run built-in invariant checks")
    parser.add_argument("--workers", type=int, default=None, help="processes (default: all cores)")
    parser.add_argument("--csv", help="write one row per run to this file")
    args = parser.parse_args()

    algos = args.algos if args.baseline in args.algos else args.algos + [args.baseline]
    cfg = BenchmarkConfig(drones=args.drones, people=args.people, runs=args.runs, seed=args.seed,
                          algos=algos, baseline=args.baseline, depot=args.depot,
                          cell_side=args.cell_side, check=args.check)
    print(Scenario(depot=cfg.depot, cell_side_target_m=cfg.cell_side).describe())
    total = len(cfg.jobs())

    def progress(done, total):
        if done == total or done % max(1, total // 20) == 0:
            print(f"  {done}/{total} runs", flush=True)

    rows = run_benchmark(cfg, workers=args.workers, progress=progress)
    print()
    print_summary(summarize_rows(rows, cfg.baseline), cfg.baseline)
    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nwrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
