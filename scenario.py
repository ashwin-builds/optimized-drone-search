"""
Simplified real-world scenario: an earthquake damages the University of
Washington Seattle campus, and drones fly supplies in from Redmond or Bothell.

Geometry is deliberately crude: the campus is a square of the same area,
each depot is a straight-line distance away, and every number below is a
round assumption meant to be edited, not a measured value.

    campus area      703 acres ~= 2.84 km^2  ->  ~1.7 km square
    cell             100 m x 100 m (1 ha)    ->  k = 17, 289 cells
    Redmond depot    ~13.8 km straight line from campus
    Bothell depot    ~14.3 km straight line from campus
    drone cruise     15 m/s (54 km/h)
    restock at depot 5 min
    search sweep     30 m wide at 5 m/s       ->  150 m^2/s
    people on campus 50,000 (daytime)
    needing aid      0.5% by default          ->  p = 250

Derived:
    tau_s = cell area / sweep rate          ~= 1.1 min per cell
    T     = 2 * distance / cruise + restock ~= 36 min round trip
    rho   = T / tau_s                       ~= 33
"""

from __future__ import annotations

import math
from dataclasses import dataclass

CAMPUS_AREA_M2 = 703 * 4046.86
DEPOT_DISTANCE_KM = {"redmond": 13.8, "bothell": 14.3}


@dataclass
class Scenario:
    depot: str = "redmond"
    cell_side_m: float = 100.0
    cruise_mps: float = 15.0
    restock_min: float = 5.0
    sweep_width_m: float = 30.0
    search_speed_mps: float = 5.0
    population: int = 50_000
    aid_fraction: float = 0.005

    @property
    def k(self) -> int:
        return max(1, round(math.sqrt(CAMPUS_AREA_M2) / self.cell_side_m))

    @property
    def cell_area_km2(self) -> float:
        return (self.cell_side_m / 1000) ** 2

    @property
    def distance_km(self) -> float:
        return DEPOT_DISTANCE_KM[self.depot]

    @property
    def search_time_min(self) -> float:
        return self.cell_side_m ** 2 / (self.sweep_width_m * self.search_speed_mps) / 60

    @property
    def trip_time_min(self) -> float:
        return 2 * self.distance_km * 1000 / self.cruise_mps / 60 + self.restock_min

    @property
    def people(self) -> int:
        return round(self.population * self.aid_fraction)

    def model_kwargs(self) -> dict:
        """Grid and timing arguments for ContinuousModel (time unit: minutes)."""
        return dict(grid_size=self.k, cell_area=self.cell_area_km2,
                    search_time=self.search_time_min, trip_time=self.trip_time_min)

    def describe(self, people: int | None = None) -> str:
        people = self.people if people is None else people
        return (f"UW Seattle campus, supplies from {self.depot.title()} ({self.distance_km:g} km)\n"
                f"  grid {self.k}x{self.k} cells of {self.cell_side_m:g} m "
                f"({self.k ** 2 * self.cell_area_km2:.2f} km^2)\n"
                f"  search {self.search_time_min:.2f} min/cell, "
                f"round trip {self.trip_time_min:.1f} min, "
                f"rho = {self.trip_time_min / self.search_time_min:.1f}\n"
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
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0, help="first seed")
    args = parser.parse_args()

    sc = Scenario(depot=args.depot, aid_fraction=args.aid_fraction)
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
