"""
Discrete-time expected-value model of n drones searching a k x k grid for p people.

Time unit t: the round trip from the search area to the supply depot and back.
In one step t a drone can either
  * search one cell of area A (adjacent to where it is, if possible), or
  * fly back to the depot and return carrying one unit of supplies.

Distances inside the search area are assumed negligible compared to the depot
trip, so delivering supplies to an already-found person costs no time: at the
end of every step, each drone carrying supplies is matched one-to-one with a
found-but-unsupplied person.

Each step the controller picks r, the number of empty drones to send back, to
balance supply against expected demand:

    H + r = S * lambda + U

    H      = drones currently carrying supplies
    r      = empty drones sent back this step (arrive at end of step, loaded)
    S      = drones searching this step (= N - r, capped at unsearched cells)
    lambda = expected people found per searching drone = A * n / R
             (n = unfound people, R = unsearched area)
    U      = found people still waiting for supplies
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Person:
    id: int
    cell: tuple[int, int]
    found_step: int | None = None
    supplied_step: int | None = None
    supplied_by: int | None = None  # drone id; set exactly once


@dataclass
class Drone:
    id: int
    pos: tuple[int, int]
    has_supplies: bool


@dataclass
class StepRecord:
    step: int
    lam: float              # expected finds per searching drone at decision time
    carrying: int           # H at decision time
    waiting_before: int     # U at decision time
    searching: int          # S
    returning: int          # r
    idle: int
    found: int              # people actually found this step
    supplied: int           # deliveries made at end of this step
    unfound_after: int
    waiting_after: int
    unsearched_after: int


class DroneSearchModel:
    def __init__(
        self,
        n_drones: int,
        grid_size: int,
        n_people: int,
        cell_area: float = 1.0,
        start_with_supplies: bool = True,
        entry_cell: tuple[int, int] = (0, 0),
        seed: int | None = None,
    ):
        if n_drones < 1:
            raise ValueError("need at least one drone")
        self.k = grid_size
        self.A = cell_area
        self.rng = random.Random(seed)
        self.step_count = 0
        self.history: list[StepRecord] = []

        # People are placed uniformly at random; several may share a cell.
        self.people = [
            Person(i, (self.rng.randrange(grid_size), self.rng.randrange(grid_size)))
            for i in range(n_people)
        ]
        self.people_in_cell: dict[tuple[int, int], list[Person]] = {}
        for person in self.people:
            self.people_in_cell.setdefault(person.cell, []).append(person)

        self.unsearched: set[tuple[int, int]] = {
            (x, y) for x in range(grid_size) for y in range(grid_size)
        }
        self.waiting: list[Person] = []  # found, not yet supplied (FIFO)
        self.drones = [
            Drone(i, entry_cell, start_with_supplies) for i in range(n_drones)
        ]

    # ------------------------------------------------------------------ state

    @property
    def n_unfound(self) -> int:
        return sum(p.found_step is None for p in self.people)

    @property
    def remaining_area(self) -> float:
        return len(self.unsearched) * self.A

    @property
    def done(self) -> bool:
        return all(p.supplied_step is not None for p in self.people)

    def expected_finds_per_drone(self) -> float:
        """lambda = A * n / R: expected people one drone finds searching one cell."""
        if not self.unsearched:
            return 0.0
        return self.A * self.n_unfound / self.remaining_area

    # --------------------------------------------------------------- decision

    def choose_num_returning(self) -> tuple[int, int]:
        """Pick (r, S) minimising |H + r - (S * lambda + U)|.

        Only empty drones may return. Ties go to the smaller r (keep searching).
        """
        n = len(self.drones)
        carrying = sum(d.has_supplies for d in self.drones)
        empty = n - carrying
        lam = self.expected_finds_per_drone()
        waiting = len(self.waiting)
        cells = len(self.unsearched)

        best = None
        for r in range(empty + 1):
            s = min(n - r, cells)
            imbalance = abs(carrying + r - (s * lam + waiting))
            if best is None or imbalance < best[0] - 1e-12:
                best = (imbalance, r, s)
        _, r, s = best
        return r, s

    # ------------------------------------------------------------- movement

    def _pick_cell(self, drone: Drone, claimed: set[tuple[int, int]]):
        """An unclaimed unsearched cell adjacent to the drone, else the nearest one."""
        x, y = drone.pos
        adjacent = [
            c for c in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1), (x, y))
            if c in self.unsearched and c not in claimed
        ]
        if adjacent:
            return self.rng.choice(adjacent)
        candidates = self.unsearched - claimed
        if not candidates:
            return None
        best_d = min(abs(cx - x) + abs(cy - y) for cx, cy in candidates)
        nearest = [c for c in candidates if abs(c[0] - x) + abs(c[1] - y) == best_d]
        return self.rng.choice(nearest)

    # ------------------------------------------------------------------ step

    def step(self) -> StepRecord:
        self.step_count += 1
        lam = self.expected_finds_per_drone()
        carrying_before = sum(d.has_supplies for d in self.drones)
        waiting_before = len(self.waiting)
        r, s = self.choose_num_returning()

        # Loaded drones always stay in the field; returners come from the empty ones.
        empty = [d for d in self.drones if not d.has_supplies]
        loaded = [d for d in self.drones if d.has_supplies]
        self.rng.shuffle(empty)
        returners = empty[:r]
        field_drones = loaded + empty[r:]
        searchers = field_drones[:s]

        # Search: no two drones search the same cell in a step.
        claimed: set[tuple[int, int]] = set()
        found_now = 0
        for drone in searchers:
            cell = self._pick_cell(drone, claimed)
            if cell is None:
                break
            claimed.add(cell)
            drone.pos = cell
            self.unsearched.discard(cell)
            for person in self.people_in_cell.get(cell, []):
                person.found_step = self.step_count
                self.waiting.append(person)
                found_now += 1

        # Returners complete the round trip within this step and arrive loaded.
        for drone in returners:
            drone.has_supplies = True

        # Deliveries: each loaded drone serves at most one person, each person
        # is served by exactly one drone.
        supplied_now = 0
        for drone in self.drones:
            if not self.waiting:
                break
            if drone.has_supplies:
                person = self.waiting.pop(0)
                person.supplied_step = self.step_count
                person.supplied_by = drone.id
                drone.has_supplies = False
                supplied_now += 1

        record = StepRecord(
            step=self.step_count,
            lam=lam,
            carrying=carrying_before,
            waiting_before=waiting_before,
            searching=len(claimed),
            returning=r,
            idle=len(self.drones) - len(claimed) - r,
            found=found_now,
            supplied=supplied_now,
            unfound_after=self.n_unfound,
            waiting_after=len(self.waiting),
            unsearched_after=len(self.unsearched),
        )
        self.history.append(record)
        return record

    def run(self, max_steps: int = 100_000) -> int:
        """Step until every person is found and supplied. Returns steps taken (in units of t)."""
        while not self.done:
            if self.step_count >= max_steps:
                raise RuntimeError(f"did not finish within {max_steps} steps")
            self.step()
        return self.step_count


def time_to_completion(n_drones: int, grid_size: int, n_people: int, **kwargs) -> int:
    return DroneSearchModel(n_drones, grid_size, n_people, **kwargs).run()


if __name__ == "__main__":
    model = DroneSearchModel(n_drones=5, grid_size=10, n_people=8, seed=1)
    model.run()
    print(f"{'t':>3} {'lam':>5} {'H':>3} {'U':>3} | {'S':>3} {'r':>3} {'idle':>4} | "
          f"{'found':>5} {'supp':>4} | {'unfound':>7} {'wait':>4} {'cells':>5}")
    for h in model.history:
        print(f"{h.step:>3} {h.lam:>5.2f} {h.carrying:>3} {h.waiting_before:>3} | "
              f"{h.searching:>3} {h.returning:>3} {h.idle:>4} | {h.found:>5} {h.supplied:>4} | "
              f"{h.unfound_after:>7} {h.waiting_after:>4} {h.unsearched_after:>5}")
    print(f"completed in {model.step_count} steps of t")
