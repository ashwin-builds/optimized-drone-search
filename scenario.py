"""
Real-world scenario: an earthquake damages the University of Washington
Seattle campus, and drones fly supplies in from Redmond or Bothell.

Campus geometry (campus="auto", the default):
  * "uw"      the real campus from OpenStreetMap (data/uw_campus.geojson, made
              by `python campus_map.py fetch`): its boundary, building
              footprints, people mostly inside buildings, and the depot exit
              where the line to the depot leaves campus. See campus_map.py.
  * "square"  a square of the same area with people spread evenly, and the
              depot exit on the square's edge (the original approximation).
  * "auto"    "uw" if the map file exists, otherwise "square".

The numbers below are round assumptions meant to be edited, not measured
values. The square-campus figures are:

    campus area      703 acres ~= 2.84 km^2  ->  1.69 km square
    cell             ~25 m (requested side; adjusted so k cells exactly span
                     the campus, so the modelled area never changes)
    Redmond depot    ~13.8 km from campus centre, almost due east
    Bothell depot    ~14.3 km from campus centre, north-east
    drone cruise     15 m/s (54 km/h)
    restock at depot 5 min
    search sweep     30 m wide at 5 m/s       ->  150 m^2/s
    people on campus 50,000 (daytime)
    needing aid      0.5% by default          ->  p = 250

Derived:
    tau_s = cell area / sweep rate               (total search work is the
                                                  same for any cell size)
    T     = 2 * (distance - centre-to-edge) / cruise + restock
            ~= 35 min round trip from the campus edge facing the depot
    flying inside campus                         ~= 1.7 s per 25 m cell

Drones leave and re-enter campus at the edge point facing the depot, and
start the run there, loaded.

Staging areas (real map only): depot="rainier_vista", "hub_yard", "denny_yard",
"parrington_lawn" are campus Mass Assembly Areas where relief supplies would be
staged, and "uw_tower" is next to the Emergency Operations Center just west of
campus (see campus_map.STAGING_LONLAT). An on-campus depot is its own exit, so a
depot run is the flight across campus plus the 5 min restock.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

from functools import lru_cache

from campus_map import DATA_PATH, DEPOT_LONLAT, STAGING_LONLAT, Grid, PeopleLayout, load_grid
from density import VISITORS_PATH, DensityLayout, DensitySurface

CAMPUS_AREA_M2 = 703 * 4046.86
CAMPUS_SIDE_M = math.sqrt(CAMPUS_AREA_M2)
DEPOT_DISTANCE_KM = {"redmond": 13.8, "bothell": 14.3}
# unit vector (east, north) from campus to depot, from straight-line coordinates
DEPOT_DIRECTION = {"redmond": (0.988, 0.150), "bothell": (0.592, 0.806)}
# people per run: normal around the expected (mean) count, SD = PEOPLE_SD_FRAC * mean;
# a draw below COMPLETION_FRAC * mean (metrics.py) is raised to PEOPLE_FLOOR_FRAC * mean,
# so the completion target is always reachable. Planners only know the mean.
PEOPLE_SD_FRAC = 0.10
PEOPLE_FLOOR_FRAC = 0.85
DEPOT_CHOICES = tuple(DEPOT_LONLAT)   # staging areas need the real map
DEPOT_NAMES = {"redmond": "Redmond", "bothell": "Bothell", "rainier_vista": "Rainier Vista",
               "hub_yard": "HUB Yard", "denny_yard": "Denny Yard",
               "parrington_lawn": "Parrington Lawn", "uw_tower": "UW Tower"}


CAMPUS_CHOICES = ("auto", "uw", "square")
PEOPLE_CHOICES = ("floor", "density")


@lru_cache(maxsize=16)
def _map_surface(map_path: str, cell_side_m: float, depot: str, scale: str) -> DensitySurface:
    cmap, grid = load_grid(map_path, cell_side_m, depot)
    return DensitySurface(cmap, grid, VISITORS_PATH, scale=scale)


@lru_cache(maxsize=16)
def _square_surface(k: int) -> DensitySurface:
    cells = frozenset((x, y) for x in range(k) for y in range(k))
    grid = Grid(k=k, cell_side_m=1.0, search_cells=cells, building_cells=frozenset(),
                indoor_weight={}, campus_area_m2=float(k * k), exit_point=(k, k / 2),
                exit_to_depot_m=0.0)
    return DensitySurface.uniform(grid)


@dataclass
class Scenario:
    depot: str = "redmond"
    campus: str = "auto"
    map_path: str = str(DATA_PATH)
    indoor_fraction: float = 0.8   # real map: share of people inside buildings
    people: str = "floor"          # true people: "floor" (PeopleLayout) or "density" (the prior)
    density_scale: str = "mass"    # prior surface scaling on the real map: "mass" or "peak"
    search_en_route: bool = True   # search unclaimed cells crossed on the way to a cell
    cell_side_target_m: float = 25.0
    cruise_mps: float = 15.0
    restock_min: float = 5.0
    sweep_width_m: float = 30.0
    search_speed_mps: float = 5.0
    population: int = 50_000
    aid_fraction: float = 0.005

    def __post_init__(self):
        if self.depot not in DEPOT_CHOICES:
            raise ValueError(f"depot must be one of {DEPOT_CHOICES}")
        if self.depot in STAGING_LONLAT and not self.uses_map:
            raise ValueError(f"the {self.depot} staging area needs the real campus map")

    @property
    def depot_name(self) -> str:
        return DEPOT_NAMES.get(self.depot, self.depot)

    @property
    def depot_on_campus(self) -> bool:
        return self.uses_map and self.map_grid[1].exit_to_depot_m == 0.0

    # ---------------------------------------------------------------- map

    @property
    def uses_map(self) -> bool:
        if self.campus not in CAMPUS_CHOICES:
            raise ValueError(f"campus must be one of {CAMPUS_CHOICES}")
        if self.campus == "uw" and not Path(self.map_path).exists():
            raise FileNotFoundError(f"{self.map_path} is missing; run `python campus_map.py fetch`")
        return self.campus == "uw" or (self.campus == "auto" and Path(self.map_path).exists())

    @property
    def map_grid(self):
        """(CampusMap, Grid) for the real campus at this cell size."""
        return load_grid(self.map_path, float(self.cell_side_target_m), self.depot)

    @property
    def density_surface(self) -> DensitySurface:
        """The searchers' prior: the building surface on the real map, uniform on the square."""
        if self.uses_map:
            return _map_surface(self.map_path, float(self.cell_side_target_m), self.depot,
                                self.density_scale)
        return _square_surface(self.k)

    @property
    def n_search_cells(self) -> int:
        return len(self.map_grid[1].search_cells) if self.uses_map else self.k ** 2

    # --------------------------------------------------------------- grid

    @property
    def k(self) -> int:
        if self.uses_map:
            return self.map_grid[1].k
        return max(1, round(CAMPUS_SIDE_M / self.cell_side_target_m))

    @property
    def cell_side_m(self) -> float:
        """Real map: the requested side. Square: k cells span the campus exactly."""
        if self.uses_map:
            return float(self.cell_side_target_m)
        return CAMPUS_SIDE_M / self.k

    @property
    def cell_area_km2(self) -> float:
        return (self.cell_side_m / 1000) ** 2

    @property
    def distance_km(self) -> float:
        return DEPOT_DISTANCE_KM[self.depot]

    @property
    def edge_to_depot_km(self) -> float:
        """Straight-line distance from the campus exit to the depot."""
        if self.uses_map:
            return self.map_grid[1].exit_to_depot_m / 1000
        return self.distance_km - self.edge_offset_m / 1000

    @property
    def depot_point(self) -> tuple[float, float]:
        """Where drones leave/enter campus: the edge point facing the depot (cell units)."""
        if self.uses_map:
            return self.map_grid[1].exit_point
        ux, uy = DEPOT_DIRECTION[self.depot]
        half = self.k / 2
        s = half / max(abs(ux), abs(uy))
        eps = 1e-6
        return (min(self.k - eps, max(eps, half + s * ux)),
                min(self.k - eps, max(eps, half + s * uy)))

    @property
    def edge_offset_m(self) -> float:
        """Distance from campus centre to the depot-facing edge point."""
        ux, uy = DEPOT_DIRECTION[self.depot]
        return CAMPUS_SIDE_M / 2 / max(abs(ux), abs(uy))

    @property
    def search_time_min(self) -> float:
        return self.cell_side_m ** 2 / (self.sweep_width_m * self.search_speed_mps) / 60

    @property
    def trip_time_min(self) -> float:
        if self.uses_map:
            off_campus_m = self.map_grid[1].exit_to_depot_m
        else:
            off_campus_m = self.distance_km * 1000 - self.edge_offset_m
        return 2 * off_campus_m / self.cruise_mps / 60 + self.restock_min

    @property
    def move_time_min(self) -> float:
        """Time to fly one cell side inside the search area (delivery flights)."""
        return self.cell_side_m / self.cruise_mps / 60

    @property
    def n_people(self) -> int:
        return round(self.population * self.aid_fraction)

    def model_kwargs(self) -> dict:
        """Grid and timing arguments for ContinuousModel (time unit: minutes)."""
        kwargs = dict(grid_size=self.k, cell_area=self.cell_area_km2,
                      search_time=self.search_time_min, trip_time=self.trip_time_min,
                      move_time_per_cell=self.move_time_min, depot_point=self.depot_point,
                      entry_cell=(min(self.k - 1, int(self.depot_point[0])),
                                  min(self.k - 1, int(self.depot_point[1]))))
        if self.people not in PEOPLE_CHOICES:
            raise ValueError(f"people must be one of {PEOPLE_CHOICES}")
        surface = self.density_surface
        kwargs.update(density=surface, search_en_route=self.search_en_route)
        if self.uses_map:
            grid = self.map_grid[1]
            kwargs.update(search_cells=grid.search_cells)
        if self.people == "density":
            kwargs.update(people_layout=DensityLayout(surface.prob))
        elif self.uses_map:
            kwargs.update(people_layout=PeopleLayout(self.map_grid[1], self.indoor_fraction))
        return kwargs

    def describe(self, people: int | None = None) -> str:
        people = self.n_people if people is None else people
        if self.uses_map:
            cmap, grid = self.map_grid
            where = (f"  map: real UW campus from OpenStreetMap (fetched {cmap.fetched}), "
                     f"{len(cmap.buildings)} buildings, {self.indoor_fraction:.0%} of people indoors\n"
                     f"  {grid.k}x{grid.k} grid of {self.cell_side_m:.0f} m cells, "
                     f"{len(grid.search_cells)} on campus "
                     f"({len(grid.search_cells) * self.cell_area_km2:.2f} km^2; "
                     f"true area {grid.campus_area_m2 / 1e6:.2f} km^2)\n")
        else:
            where = ("  map: square approximation (run `python campus_map.py fetch` for the real campus)\n"
                     f"  grid {self.k}x{self.k} cells of {self.cell_side_m:.1f} m "
                     f"({self.k ** 2 * self.cell_area_km2:.2f} km^2)\n")
        source = (f"{self.depot_name} (staging area on campus)" if self.depot_on_campus else
                  f"{self.depot_name} ({self.edge_to_depot_km:.1f} km from the campus exit)")
        return (f"UW Seattle campus, supplies from {source}\n"
                + where +
                f"  search {self.search_time_min * 60:.1f} s/cell "
                f"({self.n_search_cells * self.search_time_min:.0f} drone-min for the campus), "
                f"round trip {self.trip_time_min:.1f} min from the campus edge\n"
                f"  expected people needing aid: {people}"
                + (f" ({self.aid_fraction:.2%} of {self.population:,})"
                   if people == self.n_people else " (set directly)")
                + f"; each run draws a normal count, SD {PEOPLE_SD_FRAC:.0%} of that")


def draw_people(mean: float, seed: int) -> int:
    """People in the run with this seed: the same draw for every algorithm and depot."""
    from metrics import COMPLETION_FRAC

    n = round(random.Random(f"people:{seed}").gauss(mean, PEOPLE_SD_FRAC * mean))
    if n < COMPLETION_FRAC * mean:
        n = round(PEOPLE_FLOOR_FRAC * mean)
    return max(0, n)


def main():
    import argparse
    import statistics

    from continuous_model import CONTINUOUS_ALGORITHMS
    from metrics import COMPLETION_FRAC, completion_time

    parser = argparse.ArgumentParser(description="Run both algorithms on the UW scenario.")
    parser.add_argument("-n", type=int, default=20, help="number of drones (default 20)")
    parser.add_argument("-p", type=int, default=None,
                        help="expected people needing aid, the mean of each run's draw "
                             "(default: population x aid fraction)")
    parser.add_argument("--depot", choices=DEPOT_CHOICES, default="redmond")
    parser.add_argument("--aid-fraction", type=float, default=0.005)
    parser.add_argument("--cell-side", type=float, default=25.0,
                        help="target cell side in metres (default 25)")
    parser.add_argument("--campus", choices=CAMPUS_CHOICES, default="auto")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0, help="first seed")
    args = parser.parse_args()

    sc = Scenario(depot=args.depot, aid_fraction=args.aid_fraction,
                  cell_side_target_m=args.cell_side, campus=args.campus)
    p = sc.n_people if args.p is None else args.p
    print(sc.describe(p))
    print(f"  drones: {args.n}, expected people: {p} (each run draws its own count), "
          f"runs: {args.runs}\n")
    print(f"{'algorithm':>18} {f'{COMPLETION_FRAC:.0%} found (h)':>15} "
          f"{'wait found->supplied (min)':>27}")
    for M in CONTINUOUS_ALGORITHMS.values():
        done, waits = [], []
        for seed in range(args.seed, args.seed + args.runs):
            m = M(args.n, n_people=draw_people(p, seed), expected_people=p, seed=seed,
                  **sc.model_kwargs())
            m.run()
            done.append(completion_time(m) / 60)
            if m.people:
                waits.append(statistics.mean(x.supplied_step - x.found_step for x in m.people))
        w = f"{statistics.mean(waits):.1f}" if waits else "-"
        print(f"{M.name:>18} {statistics.mean(done):>15.2f} {w:>27}")


if __name__ == "__main__":
    main()
