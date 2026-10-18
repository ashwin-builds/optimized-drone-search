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
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from campus_map import DATA_PATH, PeopleLayout, load_grid

CAMPUS_AREA_M2 = 703 * 4046.86
CAMPUS_SIDE_M = math.sqrt(CAMPUS_AREA_M2)
DEPOT_DISTANCE_KM = {"redmond": 13.8, "bothell": 14.3}
# unit vector (east, north) from campus to depot, from straight-line coordinates
DEPOT_DIRECTION = {"redmond": (0.988, 0.150), "bothell": (0.592, 0.806)}


CAMPUS_CHOICES = ("auto", "uw", "square")


@dataclass
class Scenario:
    depot: str = "redmond"
    campus: str = "auto"
    map_path: str = str(DATA_PATH)
    indoor_fraction: float = 0.8   # real map: share of people inside buildings
    cell_side_target_m: float = 25.0
    cruise_mps: float = 15.0
    restock_min: float = 5.0
    sweep_width_m: float = 30.0
    search_speed_mps: float = 5.0
    population: int = 50_000
    aid_fraction: float = 0.005

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
    def people(self) -> int:
        return round(self.population * self.aid_fraction)

    def model_kwargs(self) -> dict:
        """Grid and timing arguments for ContinuousModel (time unit: minutes)."""
        kwargs = dict(grid_size=self.k, cell_area=self.cell_area_km2,
                      search_time=self.search_time_min, trip_time=self.trip_time_min,
                      move_time_per_cell=self.move_time_min, depot_point=self.depot_point,
                      entry_cell=(min(self.k - 1, int(self.depot_point[0])),
                                  min(self.k - 1, int(self.depot_point[1]))))
        if self.uses_map:
            grid = self.map_grid[1]
            kwargs.update(search_cells=grid.search_cells,
                          people_layout=PeopleLayout(grid, self.indoor_fraction))
        return kwargs

    def describe(self, people: int | None = None) -> str:
        people = self.people if people is None else people
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
        return (f"UW Seattle campus, supplies from {self.depot.title()} "
                f"({self.edge_to_depot_km:.1f} km from the campus exit)\n"
                + where +
                f"  search {self.search_time_min * 60:.1f} s/cell "
                f"({self.n_search_cells * self.search_time_min:.0f} drone-min for the campus), "
                f"round trip {self.trip_time_min:.1f} min from the campus edge\n"
                f"  people needing aid: {people}"
                + (f" ({self.aid_fraction:.2%} of {self.population:,})"
                   if people == self.people else " (set directly)"))


def main():
    import argparse
    import statistics

    from continuous_model import CONTINUOUS_ALGORITHMS

    parser = argparse.ArgumentParser(description="Run both algorithms on the UW scenario.")
    parser.add_argument("-n", type=int, default=20, help="number of drones (default 20)")
    parser.add_argument("-p", type=int, default=None,
                        help="people needing aid (default: population x aid fraction)")
    parser.add_argument("--depot", choices=sorted(DEPOT_DISTANCE_KM), default="redmond")
    parser.add_argument("--aid-fraction", type=float, default=0.005)
    parser.add_argument("--cell-side", type=float, default=25.0,
                        help="target cell side in metres (default 25)")
    parser.add_argument("--campus", choices=CAMPUS_CHOICES, default="auto")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0, help="first seed")
    args = parser.parse_args()

    sc = Scenario(depot=args.depot, aid_fraction=args.aid_fraction,
                  cell_side_target_m=args.cell_side, campus=args.campus)
    p = sc.people if args.p is None else args.p
    print(sc.describe(p))
    print(f"  drones: {args.n}, people: {p}, runs: {args.runs}\n")
    print(f"{'algorithm':>16} {'completion (h)':>15} {'wait found->supplied (min)':>27}")
    for M in CONTINUOUS_ALGORITHMS.values():
        done, waits = [], []
        for seed in range(args.seed, args.seed + args.runs):
            m = M(args.n, n_people=p, seed=seed, **sc.model_kwargs())
            done.append(m.run() / 60)
            if m.people:
                waits.append(statistics.mean(x.supplied_step - x.found_step for x in m.people))
        w = f"{statistics.mean(waits):.1f}" if waits else "-"
        print(f"{M.name:>16} {statistics.mean(done):>15.2f} {w:>27}")


if __name__ == "__main__":
    main()
