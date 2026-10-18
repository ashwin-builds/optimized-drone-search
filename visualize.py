"""
Animated 2D view of the drone search model.

    python visualize.py -k 20 -p 15 -n 5
    python visualize.py -k 30 -p 40 -n 8 --seed 3 --interval 300 --show-hidden
    python visualize.py -k 20 -p 15 -n 5 --save run.gif      # or run.mp4

Each frame shows the state after step t: the drones' actions during that step
and the cells/people as they stand once the step's searches and deliveries
are done. The side panel shows the decision the controller made at the start
of the step, i.e. the terms of  H + r = S * lambda + U.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from drone_model import DroneSearchModel, StepRecord

LOADED = "#1f77b4"      # drone carrying supplies
EMPTY = "#ff7f0e"       # drone without supplies
WAITING = "#d62728"     # found, waiting for supplies
SUPPLIED = "#2ca02c"    # found and supplied
HIDDEN = "#7f7f7f"      # not yet found (only with --show-hidden)
CELL_COLORS = ListedColormap(["#d0d0d0", "#fbfbf6"])  # unsearched, searched


@dataclass
class Snapshot:
    step: int
    searched: np.ndarray        # [y, x] bool
    found: np.ndarray           # per person bool
    supplied: np.ndarray        # per person bool
    drones: list[tuple]         # (pos, action, carried, has_supplies)
    record: StepRecord | None   # None for the initial frame
    n_unfound: int
    n_waiting: int
    n_supplied: int
    unsearched: int
    lam_next: float             # lambda for the coming decision


def snapshot(model: DroneSearchModel, record: StepRecord | None) -> Snapshot:
    k = model.k
    searched = np.ones((k, k), dtype=bool)
    for x, y in model.unsearched:
        searched[y, x] = False
    return Snapshot(
        step=model.step_count,
        searched=searched,
        found=np.array([p.found_step is not None for p in model.people], dtype=bool),
        supplied=np.array([p.supplied_step is not None for p in model.people], dtype=bool),
        drones=[(d.pos, d.action, d.carried, d.has_supplies) for d in model.drones],
        record=record,
        n_unfound=model.n_unfound,
        n_waiting=len(model.waiting),
        n_supplied=model.n_supplied,
        unsearched=len(model.unsearched),
        lam_next=model.expected_finds_per_drone(),
    )


def simulate(model: DroneSearchModel, max_steps: int) -> list[Snapshot]:
    frames = [snapshot(model, None)]
    while not model.done and model.step_count < max_steps:
        record = model.step()
        frames.append(snapshot(model, record))
    return frames


def spread(n: int, radius: float) -> list[tuple[float, float]]:
    """Offsets that spread n markers sharing a cell around its centre."""
    if n == 1:
        return [(0.0, 0.0)]
    return [(radius * math.cos(2 * math.pi * i / n), radius * math.sin(2 * math.pi * i / n))
            for i in range(n)]


def person_offsets(model: DroneSearchModel) -> np.ndarray:
    xy = np.zeros((len(model.people), 2))
    for cell, people in model.people_in_cell.items():
        for person, (dx, dy) in zip(people, spread(len(people), 0.22)):
            xy[person.id] = (cell[0] + dx, cell[1] + dy)
    return xy


def info_text(snap: Snapshot, model: DroneSearchModel) -> str:
    p = len(model.people)
    N = len(model.drones)
    A = model.A
    lines = [
        f"step t = {snap.step}",
        "",
        f"drones N          {N}",
        f"cell area A       {A:g}",
        f"people found      {p - snap.n_unfound} / {p}",
        f"people supplied   {snap.n_supplied} / {p}",
        f"unsearched cells  {snap.unsearched} / {model.k ** 2}",
        "",
    ]
    rec = snap.record
    if rec is None:
        lines += [
            "initial state",
            "",
            f"lambda = A*n/R = {snap.lam_next:.3f}",
            "(expected finds per searching",
            " drone on the first step)",
        ]
        return "\n".join(lines)

    lhs = rec.carrying + rec.returning
    rhs = rec.searching * rec.lam + rec.waiting_before
    cells_before = rec.unsearched_after + rec.searching
    surplus = N > cells_before
    lines += [
        "decision at start of step",
        f"  n  unfound        {rec.unfound_after + rec.found}",
        f"  R  unsearched     {cells_before * A:g}",
        f"  lambda = A*n/R    {rec.lam:.3f}",
        f"  E[finds] = S*lam  {rec.searching * rec.lam:.2f}",
        "",
        "  H + r = S*lambda + U",
        f"  H  loaded drones  {rec.carrying}",
        f"  r  sent back      {rec.returning}",
        f"  S  searching      {rec.searching}",
        f"  U  waiting        {rec.waiting_before}",
        f"  {lhs:>5.2f} vs {rhs:<5.2f} (imbalance {lhs - rhs:+.2f})",
    ]
    if surplus:
        lines.append("  surplus: N > cells, balance skipped")
    lines += [
        "",
        "outcome of step",
        f"  found this step   {rec.found}",
        f"  supplied          {rec.supplied}",
        f"  still waiting     {rec.waiting_after}",
        f"  idle drones       {rec.idle}",
    ]
    if model.done and snap.step == model.step_count:
        lines += ["", f"COMPLETE in {snap.step} steps of t"]
    return "\n".join(lines)


def build_animation(model: DroneSearchModel, frames: list[Snapshot], interval: int,
                    show_hidden: bool, hold: int = 6):
    k = model.k
    N = len(model.drones)
    fig = plt.figure(figsize=(13, 8.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.6, 1], height_ratios=[8, 1],
                          wspace=0.08, hspace=0.12, left=0.03, right=0.98, top=0.93, bottom=0.04)
    ax = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[:, 1])
    ax_depot = fig.add_subplot(gs[1, 0])

    # grid
    img = ax.imshow(frames[0].searched.astype(int), cmap=CELL_COLORS, vmin=0, vmax=1,
                    origin="lower", extent=(-0.5, k - 0.5, -0.5, k - 0.5), interpolation="nearest")
    if k <= 40:
        ax.set_xticks(np.arange(-0.5, k, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, k, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, k - 0.5)
    ax.set_ylim(-0.5, k - 0.5)
    ax.set_aspect("equal")
    title = fig.suptitle("", fontsize=13)

    # marker sizes scale with cell size in points
    cell_pt = 8 * 72 * 0.85 / k
    person_s = (0.32 * cell_pt) ** 2
    drone_s = (0.55 * cell_pt) ** 2

    people_xy = person_offsets(model)
    hidden_sc = ax.scatter([], [], s=person_s, marker="o", facecolors="none",
                           edgecolors=HIDDEN, linewidths=0.8, alpha=0.6, zorder=6)
    waiting_sc = ax.scatter([], [], s=person_s * 1.2, marker="X", c=WAITING, edgecolors="white", linewidths=0.5, zorder=7)
    supplied_sc = ax.scatter([], [], s=person_s * 1.2, marker="P", c=SUPPLIED, edgecolors="white", linewidths=0.5, zorder=7)
    search_sc = ax.scatter([], [], s=drone_s, marker="^", edgecolors="black", linewidths=0.6, zorder=5)
    idle_sc = ax.scatter([], [], s=drone_s * 0.8, marker="o", facecolors="none", linewidths=2, zorder=5)

    # depot strip for drones making the round trip
    ax_depot.set_xlim(-0.5, max(N, 10) - 0.5)
    ax_depot.set_ylim(-1, 1)
    ax_depot.set_xticks([])
    ax_depot.set_yticks([])
    ax_depot.set_facecolor("#fff3e6")
    ax_depot.set_title("depot round trip: empty drones sent back this step (arrive loaded)",
                       fontsize=9, loc="left")
    depot_sc = ax_depot.scatter([], [], s=min(drone_s, 220), marker=">", c=EMPTY,
                                edgecolors="black", linewidths=0.6)

    ax_info.axis("off")
    info = ax_info.text(0.0, 0.99, "", va="top", ha="left", family="monospace", fontsize=9.5,
                        transform=ax_info.transAxes)

    legend = [
        Line2D([], [], marker="^", ls="", color=LOADED, mec="black", ms=9, label="searching, loaded"),
        Line2D([], [], marker="^", ls="", color=EMPTY, mec="black", ms=9, label="searching, empty"),
        Line2D([], [], marker="o", ls="", mfc="none", mec=LOADED, mew=2, ms=9, label="idle, loaded"),
        Line2D([], [], marker="o", ls="", mfc="none", mec=EMPTY, mew=2, ms=9, label="idle, empty"),
        Line2D([], [], marker="X", ls="", color=WAITING, ms=8, label="found, needs supplies"),
        Line2D([], [], marker="P", ls="", color=SUPPLIED, ms=8, label="found, supplied"),
    ]
    if show_hidden:
        legend.append(Line2D([], [], marker="o", ls="", mfc="none", mec=HIDDEN, ms=7,
                             label="not yet found"))
    legend += [
        plt.Rectangle((0, 0), 1, 1, fc=CELL_COLORS(0), ec="none", label="unsearched"),
        plt.Rectangle((0, 0), 1, 1, fc=CELL_COLORS(1), ec="#999", label="searched"),
    ]
    ax_info.legend(handles=legend, loc="lower left", fontsize=8.5, frameon=False)

    empty_xy = np.empty((0, 2))

    def offsets(mask):
        return people_xy[mask] if mask.any() else empty_xy

    def draw(i):
        snap = frames[min(i, len(frames) - 1)]
        img.set_data(snap.searched.astype(int))
        hidden_sc.set_offsets(offsets(~snap.found) if show_hidden else empty_xy)
        waiting_sc.set_offsets(offsets(snap.found & ~snap.supplied))
        supplied_sc.set_offsets(offsets(snap.supplied))

        # field drones; spread any that share a cell
        # before the first step there is no action yet: show current supplies
        field = [(pos, action, carried if snap.record else has)
                 for pos, action, carried, has in snap.drones if action != "return"]
        by_cell: dict[tuple, list] = {}
        for d in field:
            by_cell.setdefault(d[0], []).append(d)
        search_xy, search_c, idle_xy, idle_c = [], [], [], []
        for (x, y), ds in by_cell.items():
            for (_, action, carried), (dx, dy) in zip(ds, spread(len(ds), 0.25)):
                color = LOADED if carried else EMPTY
                if action == "search":
                    search_xy.append((x + dx, y + dy))
                    search_c.append(color)
                else:
                    idle_xy.append((x + dx, y + dy))
                    idle_c.append(color)
        search_sc.set_offsets(np.array(search_xy) if search_xy else empty_xy)
        search_sc.set_facecolors(search_c or "none")
        idle_sc.set_offsets(np.array(idle_xy) if idle_xy else empty_xy)
        idle_sc.set_edgecolors(idle_c or "none")

        n_ret = sum(action == "return" for _, action, _, _ in snap.drones)
        depot_sc.set_offsets(np.column_stack([np.arange(n_ret), np.zeros(n_ret)])
                             if n_ret else empty_xy)

        title.set_text(f"{N} drones searching a {k}x{k} grid for {len(model.people)} people"
                       f"  -  step {snap.step}")
        info.set_text(info_text(snap, model))
        return img, hidden_sc, waiting_sc, supplied_sc, search_sc, idle_sc, depot_sc, info, title

    anim = FuncAnimation(fig, draw, frames=len(frames) + hold, interval=interval,
                         blit=False, repeat=False)
    return fig, anim


def main():
    parser = argparse.ArgumentParser(description="Animate the drone search model.")
    parser.add_argument("-k", type=int, required=True, help="grid side length in cells (k x k)")
    parser.add_argument("-p", type=int, required=True, help="number of people")
    parser.add_argument("-n", type=int, required=True, help="number of drones")
    parser.add_argument("--area", type=float, default=1.0, help="area A of one cell (default 1)")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--interval", type=int, default=500, help="ms per frame (default 500)")
    parser.add_argument("--start-empty", action="store_true",
                        help="drones start without supplies")
    parser.add_argument("--show-hidden", action="store_true",
                        help="also draw people who have not been found yet")
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--save", metavar="FILE",
                        help="write the animation to FILE (.gif or .mp4) instead of showing it")
    args = parser.parse_args()

    model = DroneSearchModel(n_drones=args.n, grid_size=args.k, n_people=args.p,
                             cell_area=args.area, start_with_supplies=not args.start_empty,
                             seed=args.seed)
    frames = simulate(model, args.max_steps)
    print(f"completed in {model.step_count} steps of t" if model.done
          else f"stopped after {args.max_steps} steps without finishing")
    fig, anim = build_animation(model, frames, args.interval, args.show_hidden)

    if args.save:
        fps = max(1, round(1000 / args.interval))
        writer = "pillow" if args.save.lower().endswith(".gif") else "ffmpeg"
        anim.save(args.save, writer=writer, fps=fps)
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
