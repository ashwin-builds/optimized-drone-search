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

When there are more drones than unsearched cells, the balance is skipped:
loaded drones take the cells first, empty drones fill any cells left over, and
every remaining empty drone heads back for supplies.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Person:
    id: int
    cell: tuple[int, int]
    found_step: int | None = None
    supplied_step: int | None = None
    supplied_by: int | None = None  # drone id; set exactly once
    # used by the continuous model only
    xy: tuple[float, float] | None = None   # exact position, in cell units
    spot_frac: float = 1.0                  # how far into its cell's search it is spotted


@dataclass
class Drone:
    id: int
    pos: tuple[int, int]
    has_supplies: bool
    # What the drone did in the most recent step, for inspection/visualisation.
    action: str = "idle"    # "search" | "return" | "idle"
    carried: bool = False   # had supplies at the start of that step


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
    name = "expected value"
    rule = "r minimises |H + r - (S*lambda + U)|"
    surplus_rule = True  # N > cells: loaded search, leftover empty drones return

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
        self.waiting: deque[Person] = deque()  # found, not yet supplied (FIFO)
        self.n_unfound = n_people
        self.n_supplied = 0
        self.drones = [
            Drone(i, entry_cell, start_with_supplies) for i in range(n_drones)
        ]

    # ------------------------------------------------------------------ state

    @property
    def remaining_area(self) -> float:
        return len(self.unsearched) * self.A

    @property
    def done(self) -> bool:
        return self.n_supplied == len(self.people)

    def expected_finds_per_drone(self) -> float:
        """lambda = A * n / R: expected people one drone finds searching one cell."""
        if not self.unsearched:
            return 0.0
        return self.A * self.n_unfound / self.remaining_area

    # --------------------------------------------------------------- decision

    def choose_num_returning(self) -> tuple[int, int]:
        """Pick (r, S) minimising |H + r - (S * lambda + U)|.

        Only empty drones may return. Ties go to the smaller r (keep searching).
        With more drones than unsearched cells, every cell gets a searcher
        (loaded drones first) and all leftover empty drones return.
        """
        n = len(self.drones)
        carrying = sum(d.has_supplies for d in self.drones)
        empty = n - carrying
        lam = self.expected_finds_per_drone()
        waiting = len(self.waiting)
        cells = len(self.unsearched)

        if n > cells:
            empty_searchers = max(0, cells - carrying)
            return empty - empty_searchers, cells

        best = None
        for r in range(empty + 1):
            s = min(n - r, cells)
            imbalance = abs(carrying + r - (s * lam + waiting))
            if best is None or imbalance < best[0] - 1e-12:
                best = (imbalance, r, s)
        _, r, s = best
        return r, s

    # ------------------------------------------------------------- movement

    def _pick_cell(self, drone: Drone):
        """The nearest unsearched cell, scanning outward one Manhattan ring at a time.

        Ring 1 is the four neighbours, ring 2 the neighbours' neighbours, and so
        on; ties within a ring are broken at random. Cells are removed from
        self.unsearched as soon as they are picked, so no two drones can pick
        the same cell in a step.
        """
        if not self.unsearched:
            return None
        x, y = drone.pos
        for d in range(2 * self.k):
            ring = []
            for i in range(-d, d + 1):
                j = d - abs(i)
                for c in {(x + i, y + j), (x + i, y - j)}:
                    if c in self.unsearched:
                        ring.append(c)
            if ring:
                return self.rng.choice(ring)
        return None

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
        for drone in self.drones:
            drone.carried = drone.has_supplies
            drone.action = "idle"
        for drone in returners:
            drone.action = "return"

        # Search: no two drones search the same cell in a step.
        n_searched = 0
        found_now = 0
        for drone in searchers:
            cell = self._pick_cell(drone)
            if cell is None:
                break
            n_searched += 1
            drone.action = "search"
            drone.pos = cell
            self.unsearched.discard(cell)
            for person in self.people_in_cell.get(cell, []):
                person.found_step = self.step_count
                self.waiting.append(person)
                found_now += 1
        self.n_unfound -= found_now

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
                person = self.waiting.popleft()
                person.supplied_step = self.step_count
                person.supplied_by = drone.id
                drone.has_supplies = False
                supplied_now += 1
        self.n_supplied += supplied_now

        record = StepRecord(
            step=self.step_count,
            lam=lam,
            carrying=carrying_before,
            waiting_before=waiting_before,
            searching=n_searched,
            returning=r,
            idle=len(self.drones) - n_searched - r,
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


def time_to_completion(n_drones: int, grid_size: int, n_people: int,
                       model_cls: type[DroneSearchModel] = DroneSearchModel, **kwargs) -> int:
    return model_cls(n_drones, grid_size, n_people, **kwargs).run()


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
