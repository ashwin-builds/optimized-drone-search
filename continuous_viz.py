"""
Animated view of the continuous model on the UW earthquake scenario.

    python continuous_viz.py -n 20                          # expected-value rule
    python continuous_viz.py -n 20 --algo both              # side by side with the baseline
    python continuous_viz.py -n 20 --algo expected keep brute  # any set, side by side
    python continuous_viz.py -n 20 -p 100 --depot bothell --until 90
    python continuous_viz.py -n 20 --algo both --save uw.gif   # or .mp4

Frames are sampled at evenly spaced times (minutes). Cells shade in as they
are searched; drones on a depot trip move along the strip under the grid
(out empty, back loaded). The side panel shows the most recent decision.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from continuous_model import CONTINUOUS_ALGORITHMS, ContinuousModel, Decision
from scenario import CAMPUS_CHOICES, DEPOT_DISTANCE_KM, Scenario
from visualize import EMPTY, HIDDEN, LOADED, SUPPLIED, WAITING, show_or_save, spread

ASSIGNED = "#9467bd"    # found, a drone is flying to them
# peak groups (categorical palette order; scavengers are grey)
GROUP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
                "#e34948"]
CELL_CMAP = LinearSegmentedColormap.from_list("search", ["#d0d0d0", "#fbfbf6"])
CELL_CMAP.set_bad("#ecebe7")  # off campus (real map)


@dataclass
class CSnapshot:
    time: float
    cells: np.ndarray            # [y, x] 0 unsearched .. 1 searched
    found: np.ndarray
    supplied: np.ndarray
    assigned: np.ndarray         # a drone is flying to them
    drones: list[tuple]          # (pos, status, has_supplies, progress, id)
    flights: list[tuple]         # (drone xy, person xy) for delivery flights
    decision: Decision | None
    n_unfound: int
    n_waiting: int
    n_supplied: int
    cells_searched: int
    done: bool


def snapshot(model: ContinuousModel, t: float) -> CSnapshot:
    k = model.k
    cells = np.zeros((k, k))
    if model.search_cells is not None:
        cells[:] = np.nan
        for x, y in model.search_cells:
            cells[y, x] = 0.0
    for x, y in model.searched:
        cells[y, x] = 1.0
    drones, flights = [], []
    assigned = np.zeros(len(model.people), dtype=bool)
    # model positions are in cell units with cell (x, y) spanning [x, x+1);
    # the plot puts cell centres on integers, hence the -0.5
    for d in model.drones:
        pos = (d.loc[0] - 0.5, d.loc[1] - 0.5)
        frac = 0.0
        if d.status in ("deliver", "transit", "exit"):
            end = {"deliver": lambda: d.target.xy,
                   "transit": lambda: (d.cell[0] + 0.5, d.cell[1] + 0.5),
                   "exit": lambda: model.depot_point}[d.status]()
            span = d.until - d.started
            frac = min(1.0, (t - d.started) / span) if span > 0 else 1.0
            (x0, y0), (x1, y1) = d.origin, end
            pos = (x0 + frac * (x1 - x0) - 0.5, y0 + frac * (y1 - y0) - 0.5)
            if d.status == "deliver":
                assigned[d.target.id] = True
                flights.append((pos, d.target.id))
        elif d.status == "search":
            frac = min(1.0, (t - d.started) / model.tau)
            cells[d.cell[1], d.cell[0]] = frac
        elif d.status == "return":
            frac = min(1.0, (t - d.started) / model.T)
        drones.append((pos, d.status, d.has_supplies, frac, d.id, model._group_label(d)))
    return CSnapshot(
        time=t,
        cells=cells,
        found=np.array([p.found_step is not None for p in model.people], dtype=bool),
        supplied=np.array([p.supplied_step is not None for p in model.people], dtype=bool),
        assigned=assigned,
        drones=drones,
        flights=flights,
        decision=model.decisions[-1] if model.decisions else None,
        n_unfound=model.n_unfound,
        n_waiting=len(model.waiting),
        n_supplied=model.n_supplied,
        cells_searched=len(model.searched),
        done=model.done,
    )


def simulate(model: ContinuousModel, times: np.ndarray) -> list[CSnapshot]:
    frames = []
    for t in times:
        model.run_until(t)
        frames.append(snapshot(model, min(t, model.completion_time or t)))
    return frames


def clock(minutes: float) -> str:
    h, m = divmod(minutes, 60)
    return f"{int(h)}h {m:04.1f}m"


def info_text(snap: CSnapshot, model: ContinuousModel, sc: Scenario) -> str:
    p = len(model.people)
    lines = [
        f"algorithm: {model.name}",
        "",
        f"time  {clock(snap.time)}",
        "",
        f"drones N          {len(model.drones)}",
        f"search / cell     {model.tau * 60:.1f} s",
        f"round trip T      {model.T:.1f} min",
        "",
        f"people found      {p - snap.n_unfound} / {p}",
        f"people supplied   {snap.n_supplied} / {p}",
        f"waiting           {snap.n_waiting}",
        f"drone en route    {int(snap.assigned.sum())}",
        f"cells searched    {snap.cells_searched} / {model.k ** 2}",
        "",
    ]
    dec = snap.decision
    if dec is not None:
        left = dec.R / model.A * model.tau / len(model.drones)
        lhs, rhs = dec.H + dec.F + dec.r, dec.U + dec.E
        lines += [
            f"last decision at {clock(dec.time)}",
            f"  mode            {dec.mode}",
            f"  n  unfound      {dec.n}",
            f"  R  unsearched   {dec.R:.3f} km2",
            f"  search left     {left:.1f} min (vs T)",
            f"  E[finds in T]   {dec.E:.2f}",
            "",
            "  H + F + r = U + E",
            f"  H  loaded       {dec.H}",
            f"  F  flying back  {dec.F}",
            f"  r  sent back    {dec.r}",
            f"  U  waiting      {dec.U}",
            f"  S  searching    {dec.S}",
            f"  {lhs:.2f} vs {rhs:.2f}  ({lhs - rhs:+.2f})",
        ]
    if snap.done:
        lines += ["", f"COMPLETE at {clock(model.completion_time)}"]
    return "\n".join(lines)


LEGEND = [
    Line2D([], [], marker="^", ls="", color=LOADED, mec="black", ms=9, label="searching, loaded"),
    Line2D([], [], marker="^", ls="", color=EMPTY, mec="black", ms=9, label="searching, empty"),
    Line2D([], [], marker="o", ls="", mfc="none", mec=LOADED, mew=2, ms=9, label="idle, loaded"),
    Line2D([], [], marker="o", ls="", mfc="none", mec=EMPTY, mew=2, ms=9, label="idle, empty"),
    Line2D([], [], marker=">", ls="", color=EMPTY, mec="black", ms=9, label="to depot (empty)"),
    Line2D([], [], marker="<", ls="", color=LOADED, mec="black", ms=9, label="from depot (loaded)"),
    Line2D([], [], marker="D", ls="--", color=LOADED, mec="black", ms=7, lw=1,
           label="flying to serve"),
    Line2D([], [], marker="X", ls="", color=WAITING, ms=8, label="found, needs supplies"),
    Line2D([], [], marker="X", ls="", color=ASSIGNED, ms=8, label="found, drone en route"),
    Line2D([], [], marker="P", ls="", color=SUPPLIED, ms=8, label="found, supplied"),
    Line2D([], [], marker="o", ls="", mfc="none", mec=HIDDEN, ms=7, label="not yet found"),
    plt.Rectangle((0, 0), 1, 1, fc=CELL_CMAP(0.0), ec="none", label="unsearched"),
    plt.Rectangle((0, 0), 1, 1, fc=CELL_CMAP(1.0), ec="#999", label="searched"),
]


def legend_handles(show_hidden: bool) -> list:
    return [h for h in LEGEND if show_hidden or h.get_label() != "not yet found"]


class ContinuousPanel:
    def __init__(self, fig, ax_grid, ax_depot, ax_info, model: ContinuousModel,
                 frames: list[CSnapshot], sc: Scenario, show_hidden: bool, fontsize: float = 9):
        self.model, self.frames, self.sc, self.show_hidden = model, frames, sc, show_hidden
        k = model.k
        self.img = ax_grid.imshow(frames[0].cells, cmap=CELL_CMAP, vmin=0, vmax=1, origin="lower",
                                  extent=(-0.5, k - 0.5, -0.5, k - 0.5), interpolation="nearest")
        if k <= 40:
            ax_grid.set_xticks(np.arange(-0.5, k, 1), minor=True)
            ax_grid.set_yticks(np.arange(-0.5, k, 1), minor=True)
            ax_grid.grid(which="minor", color="white", linewidth=0.6)
            ax_grid.tick_params(which="minor", length=0)
        ax_grid.set_xticks([])
        ax_grid.set_yticks([])
        ax_grid.set_aspect("equal")
        if sc.uses_map:
            from matplotlib.collections import PolyCollection
            outlines = sc.map_grid[0].outline_cells(sc.cell_side_m)
            shift = lambda rings: [[(x - 0.5, y - 0.5) for x, y in r] for r in rings]
            ax_grid.add_collection(PolyCollection(shift(outlines["buildings"]), facecolors="none",
                                                  edgecolors="#8f8e86", linewidths=0.4, zorder=2))
            ax_grid.add_collection(PolyCollection(shift(outlines["campus"]), facecolors="none",
                                                  edgecolors="#52514e", linewidths=1.0, zorder=2))
        label = ("UW Seattle campus (OpenStreetMap)" if sc.uses_map
                 else "UW Seattle campus (square approximation)")
        ax_grid.set_xlabel(f"{label}, {sc.cell_side_m:.1f} m cells", fontsize=8.5)

        pos = ax_grid.get_position()
        fig_w, fig_h = fig.get_size_inches()
        cell_pt = min(pos.width * fig_w, pos.height * fig_h) * 72 / k
        # fine grids make cells tiny; keep markers readable
        person_s = max((0.32 * cell_pt) ** 2, 14)
        drone_s = max((0.55 * cell_pt) ** 2, 45)

        self.people_xy = (np.array([(p.xy[0] - 0.5, p.xy[1] - 0.5) for p in model.people])
                          if model.people else np.empty((0, 2)))
        self.hidden_sc = ax_grid.scatter([], [], s=person_s, marker="o", facecolors="none",
                                         edgecolors=HIDDEN, linewidths=0.8, alpha=0.6, zorder=6)
        self.waiting_sc = ax_grid.scatter([], [], s=person_s * 1.2, marker="X", c=WAITING,
                                          edgecolors="white", linewidths=0.5, zorder=7)
        self.supplied_sc = ax_grid.scatter([], [], s=person_s * 1.2, marker="P", c=SUPPLIED,
                                           edgecolors="white", linewidths=0.5, zorder=7)
        self.assigned_sc = ax_grid.scatter([], [], s=person_s * 1.2, marker="X", c=ASSIGNED,
                                           edgecolors="white", linewidths=0.5, zorder=7)
        self.flight_lines = LineCollection([], colors=LOADED, linewidths=1, linestyles="--",
                                           alpha=0.8, zorder=4)
        ax_grid.add_collection(self.flight_lines)
        self.deliver_sc = ax_grid.scatter([], [], s=drone_s * 0.6, marker="D", c=LOADED,
                                          edgecolors="black", linewidths=0.6, zorder=8)
        self.search_sc = ax_grid.scatter([], [], s=drone_s, marker="^", edgecolors="black",
                                         linewidths=0.6, zorder=5)
        self.idle_sc = ax_grid.scatter([], [], s=drone_s * 0.8, marker="o", facecolors="none",
                                       linewidths=2, zorder=5)
        self.exit_sc = ax_grid.scatter([], [], s=drone_s * 0.7, marker=">", c=EMPTY,
                                       edgecolors="black", linewidths=0.5, zorder=8)

        ax_depot.set_xlim(-0.04, 1.04)
        ax_depot.set_ylim(-1.2, 1.2)
        ax_depot.set_yticks([])
        ax_depot.set_xticks([0, 1])
        ax_depot.set_xticklabels(["campus", f"{sc.depot.title()} depot ({sc.distance_km:g} km)"],
                                 fontsize=8)
        ax_depot.set_facecolor("#fff3e6")
        ax_depot.set_title("depot round trip", fontsize=8.5, loc="left")
        s = min(drone_s, 120)
        self.out_sc = ax_depot.scatter([], [], s=s, marker=">", c=EMPTY, edgecolors="black",
                                       linewidths=0.5)
        self.back_sc = ax_depot.scatter([], [], s=s, marker="<", c=LOADED, edgecolors="black",
                                        linewidths=0.5)
        n = len(model.drones)
        self.lane = {d.id: (np.linspace(-0.9, 0.9, min(n, 9))[d.id % min(n, 9)] if n > 1 else 0.0)
                     for d in model.drones}

        ax_info.axis("off")
        self.info = ax_info.text(0.0, 1.0, "", va="top", ha="left", family="monospace",
                                 fontsize=fontsize, transform=ax_info.transAxes)

    def draw(self, i: int) -> CSnapshot:
        snap = self.frames[min(i, len(self.frames) - 1)]
        empty_xy = np.empty((0, 2))

        def offsets(mask):
            return self.people_xy[mask] if mask.any() else empty_xy

        def pts(xy):
            return np.array(xy) if xy else empty_xy

        self.img.set_data(snap.cells)
        self.hidden_sc.set_offsets(offsets(~snap.found) if self.show_hidden else empty_xy)
        self.waiting_sc.set_offsets(offsets(snap.found & ~snap.supplied & ~snap.assigned))
        self.assigned_sc.set_offsets(offsets(snap.assigned))
        flight_xy = [pos for pos, _ in snap.flights]
        self.deliver_sc.set_offsets(pts(flight_xy))
        self.flight_lines.set_segments([[pos, tuple(self.people_xy[pid])]
                                        for pos, pid in snap.flights])
        self.supplied_sc.set_offsets(offsets(snap.supplied))

        by_cell: dict[tuple, list] = {}
        out_xy, back_xy, exit_xy = [], [], []
        for pos, status, loaded, frac, did, group in snap.drones:
            if status == "deliver":
                continue  # drawn above, between its start and the person
            if status == "exit":
                exit_xy.append(pos)
                continue
            if status == "transit":
                status = "search"  # flying to the cell it is about to search
            if status == "return":
                # outbound for the first half of the trip, inbound (loaded) for the second
                x = 2 * frac if frac < 0.5 else 2 * (1 - frac)
                (out_xy if frac < 0.5 else back_xy).append((x, self.lane[did]))
            else:
                by_cell.setdefault(pos, []).append((status, loaded, group))
        search_xy, search_c, idle_xy, idle_c = [], [], [], []
        for (x, y), ds in by_cell.items():
            for (status, loaded, group), (dx, dy) in zip(ds, spread(len(ds), max(0.25, 0.012 * self.model.k))):
                # peak-group models: colour by group (grey = scavenger); others: loaded / empty
                color = (LOADED if loaded else EMPTY) if group is None else \
                    ("#85847e" if group < 0 else GROUP_COLORS[group % len(GROUP_COLORS)])
                if status == "search":
                    search_xy.append((x + dx, y + dy))
                    search_c.append(color)
                else:
                    idle_xy.append((x + dx, y + dy))
                    idle_c.append(color)
        self.search_sc.set_offsets(pts(search_xy))
        self.search_sc.set_facecolors(search_c or "none")
        self.idle_sc.set_offsets(pts(idle_xy))
        self.idle_sc.set_edgecolors(idle_c or "none")
        self.exit_sc.set_offsets(pts(exit_xy))
        self.out_sc.set_offsets(pts(out_xy))
        self.back_sc.set_offsets(pts(back_xy))
        self.info.set_text(info_text(snap, self.model, self.sc))
        return snap


def main():
    parser = argparse.ArgumentParser(description="Animate the continuous model (UW scenario).")
    parser.add_argument("-n", type=int, default=20, help="number of drones (default 20)")
    parser.add_argument("-p", type=int, default=None,
                        help="people needing aid (default: population x aid fraction)")
    parser.add_argument("--depot", choices=sorted(DEPOT_DISTANCE_KM), default="redmond")
    parser.add_argument("--aid-fraction", type=float, default=0.005)
    parser.add_argument("--cell-side", type=float, default=25.0,
                        help="target cell side in metres (default 25)")
    parser.add_argument("--campus", choices=CAMPUS_CHOICES, default="auto",
                        help="real UW map, square approximation, or auto (map if downloaded)")
    parser.add_argument("--algo", nargs="+", choices=sorted(CONTINUOUS_ALGORITHMS) + ["both"],
                        default=["expected"],
                        help="one or more algorithms to show side by side; "
                             "'both' = expected + brute")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=240, help="frames to sample (default 240)")
    parser.add_argument("--until", type=float, default=None,
                        help="only animate the first UNTIL minutes")
    parser.add_argument("--interval", type=int, default=80, help="ms per frame (default 80)")
    parser.add_argument("--show-hidden", action="store_true",
                        help="also draw people who have not been found yet")
    parser.add_argument("--save", metavar="FILE", help="write .gif or .mp4 instead of showing")
    args = parser.parse_args()

    sc = Scenario(depot=args.depot, aid_fraction=args.aid_fraction,
                  cell_side_target_m=args.cell_side, campus=args.campus)
    p = sc.n_people if args.p is None else args.p
    keys = []
    for key in args.algo:
        for k in (["expected", "brute"] if key == "both" else [key]):
            if k not in keys:
                keys.append(k)

    # completion times first, so every panel shares one clock
    finish = {key: CONTINUOUS_ALGORITHMS[key](args.n, n_people=p, seed=args.seed,
                                              **sc.model_kwargs()).run() for key in keys}
    print(sc.describe(p))
    for key in keys:
        print(f"{CONTINUOUS_ALGORITHMS[key].name:>16}: complete at {clock(finish[key])}")
    end = max(finish.values())
    if args.until is not None:
        end = min(end, args.until)
    times = np.unique(np.concatenate([np.linspace(0, end, args.frames),
                                      [f for f in finish.values() if f <= end]]))

    models = {key: CONTINUOUS_ALGORITHMS[key](args.n, n_people=p, seed=args.seed,
                                              **sc.model_kwargs()) for key in keys}
    frames = {key: simulate(models[key], times) for key in keys}

    cols = len(keys)
    fig = plt.figure(figsize=(9 * cols, 9))
    ratios = [2.2, 1] * cols
    gs = fig.add_gridspec(3, 2 * cols, width_ratios=ratios, height_ratios=[8, 1.1, 0.9],
                          wspace=0.06, hspace=0.25, left=0.02, right=0.99, top=0.88, bottom=0.02)
    panels, titles = {}, {}
    for col, key in enumerate(keys):
        ax_grid = fig.add_subplot(gs[0, 2 * col])
        panels[key] = ContinuousPanel(fig, ax_grid, fig.add_subplot(gs[1, 2 * col]),
                                      fig.add_subplot(gs[0:2, 2 * col + 1]), models[key],
                                      frames[key], sc, args.show_hidden,
                                      fontsize=8 if cols > 1 else 9)
        titles[key] = ax_grid.set_title("", fontsize=12, loc="left")
    ax_legend = fig.add_subplot(gs[2, :])
    ax_legend.axis("off")
    ax_legend.legend(handles=legend_handles(args.show_hidden), loc="center",
                     ncol=6 if cols > 1 else 4, fontsize=8.5, frameon=False)
    header = (f"Earthquake at UW Seattle: {args.n} drones, {p} people, supplies from "
              f"{sc.depot.title()}  (seed {args.seed})")
    suptitle = fig.suptitle(header, fontsize=13)

    def verdict() -> str:
        if len(keys) < 2:
            return ""
        ranked = sorted(keys, key=lambda k: finish[k])
        return "   ".join(f"{i + 1}. {models[k].name} {clock(finish[k])}"
                           for i, k in enumerate(ranked))

    def draw(i):
        for key in keys:
            snap = panels[key].draw(i)
            name = models[key].name
            if snap.done:
                titles[key].set_text(f"{name}  -  DONE at {clock(finish[key])}")
                titles[key].set_color("#2ca02c")
            else:
                titles[key].set_text(f"{name}  -  {clock(snap.time)}")
                titles[key].set_color("black")
        last = i >= len(times) - 1
        suptitle.set_text(f"{header}\n{verdict()}" if last and verdict() else header)

    anim = FuncAnimation(fig, draw, frames=len(times) + 15, interval=args.interval,
                         blit=False, repeat=False)
    show_or_save(anim, args)


if __name__ == "__main__":
    main()
