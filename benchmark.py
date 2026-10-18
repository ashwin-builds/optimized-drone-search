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
import ast
import math
import os
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field

from continuous_model import CONTINUOUS_ALGORITHMS
from metrics import summarize
from scenario import CAMPUS_CHOICES, DEPOT_CHOICES, PEOPLE_CHOICES, Scenario, draw_people

# two-sided 95% Student t critical values by degrees of freedom
_T95 = [(1, 12.706), (2, 4.303), (3, 3.182), (4, 2.776), (5, 2.571), (6, 2.447), (7, 2.365),
        (8, 2.306), (9, 2.262), (10, 2.228), (12, 2.179), (15, 2.131), (20, 2.086),
        (25, 2.060), (30, 2.042), (40, 2.021), (60, 2.000), (120, 1.980)]

SUMMARY_METRICS = ["completion", "mean_wait", "max_wait", "all_found", "all_served", "n_people",
                   "unused_kits", "trips"]


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


def parse_spec(spec: str) -> tuple[str, dict]:
    """'expected:search_en_route=False,k=1' -> ('expected', {'search_en_route': False, 'k': 1})."""
    key, _, rest = spec.partition(":")
    if key not in CONTINUOUS_ALGORITHMS:
        raise ValueError(f"unknown algorithm {key!r}; choose from {sorted(CONTINUOUS_ALGORITHMS)}")
    overrides = {}
    for item in filter(None, rest.split(",")):
        name, _, value = item.partition("=")
        try:
            overrides[name.strip()] = ast.literal_eval(value.strip())
        except (ValueError, SyntaxError):
            overrides[name.strip()] = value.strip()
    return key, overrides


def spec_name(spec: str) -> str:
    key, overrides = parse_spec(spec)
    extra = ", ".join(f"{k}={v}" for k, v in overrides.items())
    return CONTINUOUS_ALGORITHMS[key].name + (f" [{extra}]" if extra else "")


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
    campus: str = "auto"
    placement: str = "floor"     # Scenario(people=...): "floor" or "density"
    check: bool = False

    def jobs(self) -> list[tuple]:
        return [(algo, n, p, seed, self.depot, self.cell_side, self.campus, self.placement,
                 self.check)
                for n in self.drones for p in self.people
                for seed in range(self.seed, self.seed + self.runs)
                for algo in self.algos]


def run_one(job: tuple) -> dict:
    algo, n, p, seed, depot, cell_side, campus, placement, check = job
    key, overrides = parse_spec(algo)
    sc = Scenario(depot=depot, cell_side_target_m=cell_side, campus=campus, people=placement)
    # p is the expected head count; each seed draws its own (the same for every algorithm)
    model = CONTINUOUS_ALGORITHMS[key](n, n_people=draw_people(p, seed), expected_people=p,
                                       seed=seed, check=check, **(sc.model_kwargs() | overrides))
    model.run()
    return {"algo": algo, "drones": n, "people": p, "seed": seed, "placement": placement,
            **summarize(model)}


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


PAIRED_METRICS = ("completion", "mean_wait", "max_wait", "all_found", "all_served")


def summarize_rows(rows: list[dict], baseline: str = "brute",
                   metrics: list[str] = SUMMARY_METRICS,
                   compare_to: list[str] = ()) -> list[dict]:
    """One summary per (drones, people, algo): mean/CI per metric, paired diff vs baseline.

    `compare_to` adds paired differences against more algorithms, under
    s["vs"][other][metric] = (mean, ci), plus s["vs"][other]["faster"].
    """
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
            s["vs"] = {}
            for other in compare_to:
                ref = by_algo.get(other)
                if ref is None or other == algo:
                    continue
                seeds = sorted(set(by_seed) & set(ref))
                vs = {m: mean_ci([by_seed[k][m] - ref[k][m] for k in seeds]) for m in PAIRED_METRICS}
                vs["faster"] = sum(by_seed[k]["completion"] < ref[k]["completion"] - 1e-9
                                   for k in seeds)
                vs["runs"] = len(seeds)
                s["vs"][other] = vs
            out.append(s)
    return out


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        fields = list(dict.fromkeys(k for r in rows for k in r))   # some models add columns
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: list[dict], baseline: str) -> None:
    name = spec_name
    last = None
    for s in summary:
        if (s["drones"], s["people"]) != last:
            last = (s["drones"], s["people"])
            print(f"\n{s['drones']} drones, {s['people']} expected people, {s['runs']} runs")
            print(f"  {'algorithm':<44} {'80% found (h)':>17} {'mean wait (min)':>17} "
                  f"{'max wait':>9} {'all found':>10} {'unused':>7}   vs {name(baseline)}")
        vs = ""
        if "completion_diff" in s:
            vs = (f"{s['completion_diff'] / 60:+.2f} ± {s['completion_diff_ci'] / 60:.2f} h, "
                  f"wait {s['mean_wait_diff']:+.1f} ± {s['mean_wait_diff_ci']:.1f} min, "
                  f"faster in {s['faster']}/{s['runs']}")
        print(f"  {name(s['algo']):<44} {s['completion'] / 60:>9.2f} ± {s['completion_ci'] / 60:<5.2f}"
              f" {s['mean_wait']:>9.1f} ± {s['mean_wait_ci']:<5.1f} {s['max_wait']:>9.1f}"
              f" {s['all_found'] / 60:>9.2f}h {s['unused_kits']:>7.1f}   {vs}")


def print_paired(summary: list[dict], other: str) -> None:
    """Paired differences (algorithm minus `other`) for the headline metrics."""
    print(f"\npaired differences vs {spec_name(other)} (negative = better; 95% CI)")
    last = None
    for s in summary:
        vs = s["vs"].get(other)
        if vs is None:
            continue
        if (s["drones"], s["people"]) != last:
            last = (s["drones"], s["people"])
            print(f"  {s['drones']} drones, {s['people']} people")
            print(f"    {'algorithm':<44} {'completion (min)':>17} {'mean wait':>14} "
                  f"{'max wait':>14} {'all found':>14}  faster")
        cell = lambda m: f"{vs[m][0]:+7.1f} ± {vs[m][1]:<5.1f}"
        print(f"    {spec_name(s['algo']):<44} {cell('completion'):>17} {cell('mean_wait'):>14} "
              f"{cell('max_wait'):>14} {cell('all_found'):>14}  {vs['faster']}/{vs['runs']}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark the continuous algorithms.")
    parser.add_argument("--drones", type=int, nargs="+", default=[20])
    parser.add_argument("--people", type=int, nargs="+", default=[60])
    parser.add_argument("--runs", type=int, default=30, help="seeds per configuration")
    parser.add_argument("--seed", type=int, default=0, help="first seed")
    parser.add_argument("--algos", nargs="+", default=list(CONTINUOUS_ALGORITHMS),
                        help="algorithm keys, optionally with overrides, e.g. "
                             "expected:search_en_route=False")
    parser.add_argument("--placement", choices=PEOPLE_CHOICES, default="floor",
                        help="how the true people are placed (Scenario.people)")
    parser.add_argument("--compare-to", nargs="*", default=[],
                        help="extra algorithms to report paired differences against")
    parser.add_argument("--baseline", default="brute")
    parser.add_argument("--depot", choices=DEPOT_CHOICES, default="redmond")
    parser.add_argument("--cell-side", type=float, default=25.0)
    parser.add_argument("--campus", choices=CAMPUS_CHOICES, default="auto",
                        help="real UW map, square approximation, or auto (map if downloaded)")
    parser.add_argument("--check", action="store_true", help="run built-in invariant checks")
    parser.add_argument("--workers", type=int, default=None, help="processes (default: all cores)")
    parser.add_argument("--csv", help="write one row per run to this file")
    args = parser.parse_args()

    algos = list(args.algos)
    for extra in [args.baseline, *args.compare_to]:
        if extra not in algos:
            algos.append(extra)
    for spec in algos:
        parse_spec(spec)                      # fail fast on a typo
    cfg = BenchmarkConfig(drones=args.drones, people=args.people, runs=args.runs, seed=args.seed,
                          algos=algos, baseline=args.baseline, depot=args.depot,
                          cell_side=args.cell_side, campus=args.campus,
                          placement=args.placement, check=args.check)
    print(Scenario(depot=cfg.depot, cell_side_target_m=cfg.cell_side,
                   campus=cfg.campus).describe())
    print(f"  true people placed by: {cfg.placement}")
    total = len(cfg.jobs())

    def progress(done, total):
        if done == total or done % max(1, total // 20) == 0:
            print(f"  {done}/{total} runs", flush=True)

    rows = run_benchmark(cfg, workers=args.workers, progress=progress)
    print()
    summary = summarize_rows(rows, cfg.baseline, compare_to=args.compare_to)
    print_summary(summary, cfg.baseline)
    for other in args.compare_to:
        print_paired(summary, other)
    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nwrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
