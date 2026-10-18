"""
Prior probability-density surface of where people are on the UW campus.

This is the searcher's BELIEF, built only from public building data
(data/building_visitors.csv) and the campus geometry. It deliberately does
not use PeopleLayout, which places the simulation's true people; an
algorithm searching with that would be searching with the answer.

    python density.py info  [--cell-side 25] [--scale peak|mass] [--sigma-factor 1.0]
    python density.py show  [...] [--save density.png]
    python density.py estimate-csv     # add floor-area estimates for named buildings
                                       # missing from the CSV (keeps existing rows)

Model
-----
Every building b with daily visitors V_b contributes a 2-D Gaussian centred on
its footprint centroid c_b (cell units):

    sigma_b = max(min_sigma_cells, sigma_factor * sqrt(footprint_area / pi) / cell_side)

i.e. sigma_factor times the radius of a circle with the footprint's area. The
floor (0.5 cells by default) stops tiny footprints such as gatehouses from
becoming needle spikes that fall between cell centres or swamp everything.

    scale="peak"  (default)  height_b = V_b
    scale="mass"             height_b = V_b / (2 pi sigma_b^2)   (integral = V_b)

Peak scaling gives a building mass V_b * 2 pi sigma_b^2, so large footprints
get proportionally more; mass scaling makes each building's total exactly its
visitors (up to the campus edge and the cell-centre sampling).

    raw(x)  = sum_b height_b * exp(-|x - c_b|^2 / (2 sigma_b^2))   at campus cell centres
    prob(x) = raw(x) / sum over campus cells of raw

The sum is taken in log space (logaddexp), so far from every building the
surface is tiny but never exactly 0; this keeps empty areas from forming
flat plateaus of false peaks.

Daily visitors per building come from, in order:
  1. a row in data/building_visitors.csv whose name equals the OSM name
     (is_estimate tells measured figures from estimates); a name shared by
     several footprints is split between them by floor area;
  2. otherwise a fallback: floor area * FALLBACK_FACTOR * TYPE_DENSITY[type].

Peaks and basins
----------------
Cells are ordered by (log raw value, then lower row, then lower column), so
equal values (plateaus) are broken deterministically. A peak is a campus cell
that is higher in this order than each of its (up to 8) campus neighbours.
Every campus cell follows steepest ascent - move to the highest neighbour
while it is higher - and belongs to the basin of the peak it stops at.
Peaks on the campus edge (cells with fewer than 8 campus neighbours) are
usually artefacts of the boundary cutting through a hillside, so by default
(drop_edge_peaks=True) each is removed and its basin merged into the
neighbouring basin it joins across the highest saddle - the highest cell
pair where the two basins touch - or, for a detached piece of campus with no
neighbouring basin, into the nearest remaining peak.

A basin's remaining mass is the probability over its cells still unsearched,
so "searched" is derived from the caller's unsearched set, never stored.

Nothing in this module is random.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely

from campus_map import DATA_PATH, CampusMap, Grid, load_grid

VISITORS_PATH = Path(__file__).parent / "data" / "building_visitors.csv"
CSV_FIELDS = ["name", "daily_visitors", "source", "is_estimate"]

# Daily visitors per m^2 of floor area (footprint x levels), by building type.
# Calibrated against the measured UW figures in building_visitors.csv where
# possible; the rest are round assumptions.
TYPE_DENSITY = {
    "library":    0.12,   # UW gate counts: Odegaard 0.20, Suzzallo + Allen 0.07 per m^2
    "hospital":   0.11,   # UWMC Montlake: (staff + average inpatients) / floor area
    "university": 0.08,   # assumption: ~1 occupant per 20 m^2, ~1.6 visitors per seat per day
    "sports":     0.06,   # assumption: gyms, pools, indoor fields
    "office":     0.05,   # assumption: ~1 worker per 20 m^2, one visit a day
    "dormitory":  0.033,  # UW HFS beds / floor area: median over the 12 matched halls
    "apartments": 0.03,   # assumption: larger units than dormitories
    "stadium":    0.01,   # assumption: a non-event day
    "parking":    0.005,  # assumption
    "low_use":    0.002,  # sheds, plant, gatehouses, roofs, containers, bike houses
}
FALLBACK_FACTOR = 0.25    # buildings not in the CSV get a quarter of their type density
LOW_USE_TAGS = {"industrial", "service", "shed", "roof", "gatehouse", "bridge", "garage"}


def building_type(tag: str | None, name: str | None) -> str:
    """Coarse building type from the OSM building tag and name."""
    n = (name or "").lower()
    if any(w in n for w in ("bike house", "container", "shed", "gatehouse")):
        return "low_use"
    if "library" in n:
        return "library"
    if "garage" in n or "parking" in n or tag == "parking":
        return "parking"
    if tag == "hospital" or "medical center" in n:
        return "hospital"
    if tag in ("dormitory", "apartments", "office"):
        return tag
    if tag == "stadium" or "grandstand" in n:
        return "stadium"
    if any(w in n for w in ("ima building", "pavilion", "indoor", "tennis", "shellhouse", "pool")):
        return "sports"
    if tag in LOW_USE_TAGS:
        return "low_use"
    return "university"


# ------------------------------------------------------------ visitor table

@dataclass(frozen=True)
class VisitorRow:
    name: str
    daily_visitors: float
    source: str
    is_estimate: bool


def load_visitor_table(path: Path = VISITORS_PATH) -> dict[str, VisitorRow]:
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name = r["name"].strip()
            if name in rows:
                raise ValueError(f"{path}: {name!r} appears twice")
            flag = r["is_estimate"].strip().lower()
            if flag not in ("true", "false"):
                raise ValueError(f"{path}: is_estimate for {name!r} must be true or false")
            rows[name] = VisitorRow(name, float(r["daily_visitors"]), r["source"].strip(),
                                    flag == "true")
    return rows


@dataclass(frozen=True)
class BuildingSource:
    """Where one building's daily-visitor number came from, and its Gaussian."""
    index: int                   # position in CampusMap.buildings
    name: str | None
    tag: str | None
    kind: str                    # "measured" | "estimate" | "fallback"
    visitors: float
    centroid: tuple[float, float]   # cell units
    sigma: float                 # cell units


@dataclass
class Peak:
    id: int
    cell: tuple[int, int]
    height: float                # raw surface value at the peak
    buildings: tuple[str, ...]   # building(s) at the peak cell, else the nearest one
    distance_m: float            # 0 if the cell centre is inside a footprint
    basin: frozenset             # campus cells whose steepest ascent ends here
    mass: float                  # initial probability mass of the basin


# --------------------------------------------------------------- the surface

class DensitySurface:
    def __init__(self, cmap: CampusMap, grid: Grid,
                 visitors: dict[str, VisitorRow] | Path | str | None = None,
                 scale: str = "peak", sigma_factor: float = 1.0,
                 min_sigma_cells: float = 0.5, drop_edge_peaks: bool = True):
        if scale not in ("peak", "mass"):
            raise ValueError("scale must be 'peak' or 'mass'")
        if sigma_factor <= 0 or min_sigma_cells < 0:
            raise ValueError("sigma_factor must be > 0 and min_sigma_cells >= 0")
        if visitors is None or isinstance(visitors, (str, Path)):
            visitors = load_visitor_table(Path(visitors or VISITORS_PATH))
        self.grid = grid
        self.scale = scale
        self.sigma_factor = sigma_factor
        self.min_sigma_cells = min_sigma_cells
        self.drop_edge_peaks = drop_edge_peaks
        self.merged_edge_peaks: list[tuple[int, int]] = []   # edge peak cells removed
        self.k = k = grid.k
        side = grid.cell_side_m
        self._cmap = cmap

        self.sources = self._resolve_visitors(cmap, visitors, side)
        names = {n for _, n, _ in cmap.buildings if n}
        self.unmatched_rows = sorted(set(visitors) - names)

        # log of the sum of Gaussians at every cell centre, [y, x]
        centres = np.arange(k) + 0.5
        X, Y = np.meshgrid(centres, centres, indexing="xy")
        log_raw = np.full((k, k), -np.inf)
        for s in self.sources:
            if s.visitors <= 0:
                continue
            log_h = math.log(s.visitors)
            if scale == "mass":
                log_h -= math.log(2 * math.pi * s.sigma ** 2)
            d2 = (X - s.centroid[0]) ** 2 + (Y - s.centroid[1]) ** 2
            log_raw = np.logaddexp(log_raw, log_h - d2 / (2 * s.sigma ** 2))

        self.mask = np.zeros((k, k), dtype=bool)       # campus cells, [y, x]
        for x, y in grid.search_cells:
            self.mask[y, x] = True
        if not np.isfinite(log_raw[self.mask]).any():
            raise ValueError("no building has any visitors; the surface is empty")
        self.log_raw = np.where(self.mask, log_raw, -np.inf)
        self.raw_grid = np.where(self.mask, np.exp(log_raw), 0.0)
        total = self.raw_grid.sum()
        if total > 0:
            self.prob_grid = self.raw_grid / total
        else:  # everything underflowed: normalise in log space
            shifted = np.exp(self.log_raw - self.log_raw[self.mask].max())
            self.prob_grid = shifted / shifted.sum()
        cells = sorted(grid.search_cells)
        self.raw = {c: float(self.raw_grid[c[1], c[0]]) for c in cells}
        self.prob = {c: float(self.prob_grid[c[1], c[0]]) for c in cells}

        self.peaks = self._find_peaks(cells)
        self.basin_grid = np.full((k, k), -1, dtype=int)   # peak id per cell, -1 off campus
        for p in self.peaks:
            for x, y in p.basin:
                self.basin_grid[y, x] = p.id

    # ---------------------------------------------------------- building data

    def _resolve_visitors(self, cmap, visitors, side) -> list[BuildingSource]:
        floor = [poly.area * levels for poly, _, levels in cmap.buildings]
        share_floor: dict[str, float] = {}
        for (_, name, _), fa in zip(cmap.buildings, floor):
            if name in visitors:
                share_floor[name] = share_floor.get(name, 0.0) + fa
        out = []
        for i, ((poly, name, levels), tag) in enumerate(zip(cmap.buildings, cmap.building_tags)):
            if name in visitors:
                row = visitors[name]
                v = row.daily_visitors * (floor[i] / share_floor[name] if share_floor[name] else 1.0)
                kind = "estimate" if row.is_estimate else "measured"
            else:
                v = floor[i] * FALLBACK_FACTOR * TYPE_DENSITY[building_type(tag, name)]
                kind = "fallback"
            c = poly.centroid
            radius = math.sqrt(poly.area / math.pi) / side
            sigma = max(self.min_sigma_cells, self.sigma_factor * radius)
            out.append(BuildingSource(i, name, tag, kind, v, (c.x / side, c.y / side), sigma))
        return out

    def source_counts(self) -> dict[str, int]:
        counts = {"measured": 0, "estimate": 0, "fallback": 0}
        for s in self.sources:
            counts[s.kind] += 1
        return counts

    # ------------------------------------------------------- peaks and basins

    def _key(self, c: tuple[int, int]):
        x, y = c
        return (self.log_raw[y, x], -y, -x)   # higher value, then lower row, lower column

    def _neighbours(self, c):
        x, y = c
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dx or dy) and 0 <= x + dx < self.k and 0 <= y + dy < self.k \
                        and self.mask[y + dy, x + dx]:
                    yield (x + dx, y + dy)

    def _find_peaks(self, cells) -> list[Peak]:
        up = {}                                    # cell -> higher neighbour it climbs to
        tops = []
        for c in cells:
            best = max(self._neighbours(c), key=self._key, default=None)
            if best is not None and self._key(best) > self._key(c):
                up[c] = best
            else:
                tops.append(c)
        top_of = {t: t for t in tops}
        for c in cells:                            # follow steepest ascent, compressing paths
            path = []
            while c not in top_of:
                path.append(c)
                c = up[c]
            for p in path:
                top_of[p] = top_of[c]
        basins: dict[tuple, list] = {t: [] for t in tops}
        for c, t in top_of.items():
            basins[t].append(c)
        if self.drop_edge_peaks:
            self._merge_edge_peaks(basins, top_of)
            tops = list(basins)
        found = []
        for t in tops:
            mass = float(sum(self.prob[c] for c in basins[t]))
            found.append((t, frozenset(basins[t]), mass))
        found.sort(key=lambda f: (-f[2], f[0][1], f[0][0]))
        return [Peak(i, t, self.raw[t], *self._buildings_at(t), basin, mass)
                for i, (t, basin, mass) in enumerate(found)]

    def _is_edge(self, c) -> bool:
        return sum(1 for _ in self._neighbours(c)) < 8

    def _merge_edge_peaks(self, basins: dict, top_of: dict) -> None:
        """Merge each edge peak's basin into a neighbour (see module docstring)."""
        def mass(t):
            return sum(self.prob[c] for c in basins[t])
        queue = sorted((t for t in basins if self._is_edge(t)), key=lambda t: (mass(t), t[1], t[0]))
        for t in queue:
            if len(basins) == 1:
                break
            best = None                            # (saddle key, target peak)
            for c in basins[t]:
                for n in self._neighbours(c):
                    target = top_of[n]
                    if target != t:
                        saddle = min(self._key(c), self._key(n))
                        if best is None or saddle > best[0]:
                            best = (saddle, target)
            if best is not None:
                target = best[1]
            else:                                  # detached: nearest interior peak
                others = [u for u in basins if u != t and not self._is_edge(u)] or \
                         [u for u in basins if u != t]
                target = min(others, key=lambda u: ((u[0] - t[0]) ** 2 + (u[1] - t[1]) ** 2,
                                                    u[1], u[0]))
            for c in basins[t]:
                top_of[c] = target
            basins[target].extend(basins.pop(t))
            self.merged_edge_peaks.append(t)

    def _buildings_at(self, cell) -> tuple[tuple[str, ...], float]:
        side = self.grid.cell_side_m
        pt = shapely.Point((cell[0] + 0.5) * side, (cell[1] + 0.5) * side)
        label = lambda i: self._cmap.buildings[i][1] or f"(unnamed {self._cmap.building_tags[i]})"
        inside = [i for i, (poly, _, _) in enumerate(self._cmap.buildings) if poly.contains(pt)]
        if inside:
            return tuple(label(i) for i in inside), 0.0
        if not self._cmap.buildings:
            return (), math.inf
        dists = [poly.distance(pt) for poly, _, _ in self._cmap.buildings]
        i = int(np.argmin(dists))
        return (label(i),), float(dists[i])

    # ---------------------------------------------------------------- queries

    def cell_prob(self, cell: tuple[int, int]) -> float:
        return self.prob.get(cell, 0.0)

    def remaining_mass(self, peak_id: int, unsearched) -> float:
        """Probability over the peak's basin cells that are still in `unsearched`."""
        basin = self.peaks[peak_id].basin
        small, big = (unsearched, basin) if len(unsearched) < len(basin) else (basin, unsearched)
        return float(sum(self.prob[c] for c in small if c in big and c in self.prob))

    def peaks_by_mass(self, unsearched, min_mass: float = 0.0) -> list[tuple[Peak, float]]:
        """(peak, remaining mass) for peaks with remaining mass > min_mass, largest first."""
        out = [(p, self.remaining_mass(p.id, unsearched)) for p in self.peaks]
        out = [(p, m) for p, m in out if m > min_mass]
        out.sort(key=lambda pm: (-pm[1], pm[0].id))
        return out


def surface_for(cell_side_m: float = 25.0, scale: str = "peak", sigma_factor: float = 1.0,
                visitors_path: Path = VISITORS_PATH, map_path: Path = DATA_PATH,
                drop_edge_peaks: bool = True) -> DensitySurface:
    """The UW surface from the saved map and visitor table."""
    cmap, grid = load_grid(str(map_path), float(cell_side_m), "redmond")
    return DensitySurface(cmap, grid, visitors_path, scale=scale, sigma_factor=sigma_factor,
                          drop_edge_peaks=drop_edge_peaks)


# ------------------------------------------------------------- estimate rows

def estimate_rows(cmap: CampusMap, existing: dict[str, VisitorRow]) -> list[dict]:
    """Floor-area estimates for uniquely named buildings not already in the table.

    Unnamed footprints, names shared by several footprints, and low-use types
    (sheds, plant, parking) are left out; they take the fallback instead.
    """
    counts: dict[str, int] = {}
    for _, name, _ in cmap.buildings:
        if name:
            counts[name] = counts.get(name, 0) + 1
    rows = []
    for (poly, name, levels), tag, known in zip(cmap.buildings, cmap.building_tags,
                                                 cmap.building_levels_known):
        if not name or counts[name] > 1 or name in existing:
            continue
        kind = building_type(tag, name)
        if kind in ("low_use", "parking"):
            continue
        floor = poly.area * levels
        rows.append({"name": name, "daily_visitors": round(floor * TYPE_DENSITY[kind]),
                     "source": f"estimate: {floor:.0f} m^2 floor area (OSM footprint x {levels:g} "
                               f"levels{'' if known else ' - assumed, OSM has no level count'}) x "
                               f"{TYPE_DENSITY[kind]} visitors/m^2/day for type '{kind}' "
                               f"(OSM building={tag})",
                     "is_estimate": "true"})
    return sorted(rows, key=lambda r: r["name"])


def _write_estimates(path: Path = VISITORS_PATH) -> None:
    cmap = CampusMap.load(DATA_PATH)
    existing = load_visitor_table(path) if path.exists() else {}
    with open(path, newline="", encoding="utf-8") as f:
        kept = list(csv.DictReader(f)) if path.exists() else []
    new = estimate_rows(cmap, existing)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(kept + new)
    print(f"kept {len(kept)} rows, added {len(new)} estimates -> {path}")


# --------------------------------------------------------------------- CLI

def _info(s: DensitySurface) -> None:
    side = s.grid.cell_side_m
    counts = s.source_counts()
    print(f"{s.k}x{s.k} grid of {side:g} m cells, {len(s.prob)} on campus; "
          f"scale={s.scale}, sigma_factor={s.sigma_factor:g}")
    print(f"buildings: {sum(counts.values())} total - {counts['measured']} measured, "
          f"{counts['estimate']} estimated (CSV), {counts['fallback']} fallback")
    if s.unmatched_rows:
        print(f"CSV rows with no building of that name: {', '.join(s.unmatched_rows)}")
    edge = sum(1 for p in s.peaks if s._is_edge(p.cell))
    off = sum(1 for p in s.peaks if p.distance_m > side)
    merged = (f"{len(s.merged_edge_peaks)} edge peaks merged into neighbouring basins"
              if s.drop_edge_peaks else f"{edge} on the campus edge (kept)")
    print(f"peaks: {len(s.peaks)}  ({merged}, {off} more than one cell from any building)")
    print(f"\ntop 15 peaks by mass")
    print(f"  {'id':>3} {'cell':>10}  {'height':>9} {'mass':>6}  building")
    for p in s.peaks[:15]:
        where = ", ".join(p.buildings) + (f"  ({p.distance_m:.0f} m away)" if p.distance_m else "")
        print(f"  {p.id:>3} {str(p.cell):>10}  {p.height:>9.1f} {100 * p.mass:>5.1f}%  {where}")


def _show(s: DensitySurface, save: str | None) -> None:
    import matplotlib
    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm

    k = s.k
    cmap = LinearSegmentedColormap.from_list("prob", ["#f7fbff", "#9ec5f4", "#2a78d6", "#0d366b"])
    cmap.set_bad("#ffffff")
    fig, ax = plt.subplots(figsize=(11, 9))
    img = np.ma.masked_where(~s.mask, s.prob_grid)
    shown = ax.imshow(img, origin="lower", extent=(0, k, 0, k), cmap=cmap,
                      norm=PowerNorm(0.5, vmin=0, vmax=float(img.max())), interpolation="nearest")
    fig.colorbar(shown, ax=ax, shrink=0.7, label="probability per cell (square-root colour scale)")

    outlines = s._cmap.outline_cells(s.grid.cell_side_m)
    ax.add_collection(PolyCollection(outlines["buildings"], facecolors="none",
                                     edgecolors="#52514e", linewidths=0.35, alpha=0.6))
    ax.add_collection(PolyCollection(outlines["campus"], facecolors="none",
                                     edgecolors="#0b0b0b", linewidths=1.2))

    # basin boundaries: cell edges between two different basins
    b = s.basin_grid
    segs = []
    for y in range(k):
        for x in range(k):
            if b[y, x] < 0:
                continue
            if x + 1 < k and b[y, x + 1] >= 0 and b[y, x + 1] != b[y, x]:
                segs.append([(x + 1, y), (x + 1, y + 1)])
            if y + 1 < k and b[y + 1, x] >= 0 and b[y + 1, x] != b[y, x]:
                segs.append([(x, y + 1), (x + 1, y + 1)])
    ax.add_collection(LineCollection(segs, colors="#eb6834", linewidths=0.8))

    masses = np.array([p.mass for p in s.peaks])
    ax.scatter([p.cell[0] + 0.5 for p in s.peaks], [p.cell[1] + 0.5 for p in s.peaks],
               s=12 + 400 * masses / masses.max(), marker="^", c="#0b0b0b",
               edgecolors="white", linewidths=0.6, zorder=5)
    for p in s.peaks[:10]:
        ax.annotate(f"{p.buildings[0]} ({100 * p.mass:.1f}%)", (p.cell[0] + 0.5, p.cell[1] + 0.5),
                    xytext=(5, 5), textcoords="offset points", fontsize=7.5, color="#0b0b0b")
    ax.set_xlim(0, k)
    ax.set_ylim(0, k)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Prior density of people (scale={s.scale}, sigma_factor={s.sigma_factor:g}, "
                 f"{s.grid.cell_side_m:g} m cells): {len(s.peaks)} peaks, orange = basin boundaries",
                 fontsize=10)
    if save:
        fig.savefig(save, dpi=130, bbox_inches="tight")
        print(f"saved {save}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Prior density surface of people on campus.")
    parser.add_argument("command", choices=["info", "show", "estimate-csv"])
    parser.add_argument("--cell-side", type=float, default=25.0)
    parser.add_argument("--scale", choices=["peak", "mass"], default="peak")
    parser.add_argument("--sigma-factor", type=float, default=1.0)
    parser.add_argument("--keep-edge-peaks", action="store_true",
                        help="keep peaks on the campus edge instead of merging them")
    parser.add_argument("--save", metavar="PATH", help="write the 'show' figure to a PNG")
    args = parser.parse_args()
    if not DATA_PATH.exists():
        sys.exit(f"{DATA_PATH} is missing; run `python campus_map.py fetch` first")
    if args.command == "estimate-csv":
        _write_estimates()
        return
    s = surface_for(args.cell_side, args.scale, args.sigma_factor,
                    drop_edge_peaks=not args.keep_edge_peaks)
    if args.command == "info":
        _info(s)
    else:
        _show(s, args.save)


if __name__ == "__main__":
    main()
