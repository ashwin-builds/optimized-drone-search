"""
Run the expected-value model and the find-then-fetch baseline side by side.

Both models get the same seed, so they see the same people layout.

    python compare.py -k 20 -p 15 -n 5 --seed 3                  # animate one scenario
    python compare.py -k 20 -p 15 -n 5 --seed 3 --save cmp.gif   # or .mp4
    python compare.py -k 20 -p 15 -n 5 --runs 200                # no animation, just stats
"""

from __future__ import annotations

import argparse
import statistics

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from brute_force import ALGORITHMS
from visualize import (ModelPanel, add_common_args, legend_handles, model_kwargs,
                       show_or_save, simulate)

ORDER = ["expected", "brute"]


def verdict(steps: dict[str, int]) -> str:
    a, b = (ALGORITHMS[key].name for key in ORDER)
    ta, tb = (steps[key] for key in ORDER)
    if ta == tb:
        return f"tie: both finished in {ta} steps"
    winner, loser, tw, tl = (a, b, ta, tb) if ta < tb else (b, a, tb, ta)
    return f"{winner} finished {tl - tw} step{'' if tl - tw == 1 else 's'} sooner ({tw} vs {tl}, {100 * (tl - tw) / tl:.0f}% faster)"


def animate(args) -> None:
    models = {key: ALGORITHMS[key](**model_kwargs(args)) for key in ORDER}
    frames = {key: simulate(m, args.max_steps) for key, m in models.items()}
    steps = {key: m.step_count for key, m in models.items()}
    for key in ORDER:
        print(f"{models[key].name:>16}: {steps[key]} steps"
              + ("" if models[key].done else " (did not finish)"))
    print(verdict(steps))

    fig = plt.figure(figsize=(18, 9.5))
    gs = fig.add_gridspec(3, 4, width_ratios=[2.2, 1, 2.2, 1], height_ratios=[8, 1, 0.9],
                          wspace=0.06, hspace=0.18, left=0.02, right=0.99, top=0.92, bottom=0.02)
    panels, titles = {}, {}
    for col, key in enumerate(ORDER):
        ax_grid = fig.add_subplot(gs[0, 2 * col])
        panels[key] = ModelPanel(fig, ax_grid, fig.add_subplot(gs[1, 2 * col]),
                                 fig.add_subplot(gs[0:2, 2 * col + 1]), models[key],
                                 frames[key], args.show_hidden, fontsize=8)
        titles[key] = ax_grid.set_title("", fontsize=12, loc="left")

    ax_bottom = fig.add_subplot(gs[2, :])
    ax_bottom.axis("off")
    ax_bottom.legend(handles=legend_handles(args.show_hidden), loc="center", ncol=5,
                     fontsize=8.5, frameon=False)
    m0 = models[ORDER[0]]
    header = (f"{len(m0.drones)} drones, {m0.k}x{m0.k} grid, {len(m0.people)} people"
              f"  (seed {args.seed})")
    suptitle = fig.suptitle(header, fontsize=13)
    n_frames = max(len(f) for f in frames.values())

    def draw(i):
        for key in ORDER:
            step = panels[key].draw(i)
            finished = i >= len(frames[key]) - 1 and models[key].done
            status = f"DONE in {step} steps" if finished else f"step {step}"
            titles[key].set_text(f"{models[key].name}  -  {status}")
            titles[key].set_color("#2ca02c" if finished else "black")
        suptitle.set_text(f"{header}\n{verdict(steps)}" if i >= n_frames - 1 else header)

    anim = FuncAnimation(fig, draw, frames=n_frames + 8, interval=args.interval,
                         blit=False, repeat=False)
    show_or_save(anim, args)


def batch(args) -> None:
    base = args.seed if args.seed is not None else 0
    results = {key: [] for key in ORDER}
    waits = {key: [] for key in ORDER}   # per run: mean steps from found to supplied
    for seed in range(base, base + args.runs):
        kwargs = model_kwargs(args) | {"seed": seed}
        for key in ORDER:
            model = ALGORITHMS[key](**kwargs)
            results[key].append(model.run(args.max_steps))
            if model.people:
                waits[key].append(statistics.mean(p.supplied_step - p.found_step
                                                  for p in model.people))

    a, b = (results[key] for key in ORDER)
    wins = sum(x < y for x, y in zip(a, b))
    losses = sum(x > y for x, y in zip(a, b))
    print(f"k={args.k}  p={args.p}  n={args.n}  runs={args.runs}  (seeds {base}..{base + args.runs - 1})")
    print(f"{'':>16} {'--- steps to completion ---':>34}   person wait")
    print(f"{'algorithm':>16} {'mean':>7} {'median':>7} {'stdev':>6} {'min':>4} {'max':>4}"
          f"   (found->supplied)")
    for key in ORDER:
        r = results[key]
        sd = statistics.stdev(r) if len(r) > 1 else 0.0
        w = f"{statistics.mean(waits[key]):.2f}" if waits[key] else "-"
        print(f"{ALGORITHMS[key].name:>16} {statistics.mean(r):>7.2f} {statistics.median(r):>7.1f} "
              f"{sd:>6.2f} {min(r):>4} {max(r):>4}   {w:>8}")
    diffs = [y - x for x, y in zip(a, b)]
    print(f"{ALGORITHMS['expected'].name} faster in {wins}, slower in {losses}, tied in "
          f"{args.runs - wins - losses} runs; mean saving {statistics.mean(diffs):.2f} steps")


def main():
    parser = argparse.ArgumentParser(
        description="Compare the expected-value model with the find-then-fetch baseline.")
    add_common_args(parser)
    parser.add_argument("--runs", type=int, default=0,
                        help="run this many seeds without animation and print statistics")
    args = parser.parse_args()
    if args.runs:
        batch(args)
    else:
        animate(args)


if __name__ == "__main__":
    main()
