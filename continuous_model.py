"""
Continuous-time (event-driven) version of the drone search model.

Instead of lockstep steps, every drone is always doing one thing with a known
finish time, and decides what to do next the moment it is free:

    search a cell        takes search_time  (tau_s)
    depot round trip     takes trip_time    (T), returns loaded

A person in a cell is spotted at a uniformly random moment during that cell's
search (or at the end of it with spot_at_end=True). Deliveries are instant: the
moment someone is spotted, any loaded drone drops supplies and carries on.

Decision rule (expected value). Whenever drones are free, choose how many of
the free empty drones to send back, r, minimising

    | H + F + r  -  (U + E[finds over the next T]) |

    H  loaded drones          F  drones already flying back
    U  people waiting         E  = n * min(1, S * A * T / (tau_s * R))
                                   (n unfound, R unsearched area including the
                                    unsearched part of cells in progress,
                                    S drones searching)

The horizon is T because a drone sent back now delivers at now + T. Ties go to
the smaller r. If free drones outnumber unclaimed cells, the surplus rule
applies as in the discrete model: loaded drones take cells first, empty ones
fill any left over, the remaining empty drones return.

Search first: if the whole fleet could finish the remaining search in less
than one round trip T (remaining cells * tau_s / N < T), the balance is
skipped and empty drones only go back for people already found
(r = min(free empty, U - F)). Searching is cheap next to a depot trip, so
holding drones back from a short search only delays finding the last people,
who then each wait a full trip. ContinuousKeepSearching reads "search first"
literally instead: in that phase no empty drone goes back (r = 0) until the
cells run out, trading longer waits for an earlier finish.

With trip_time == search_time and spot_at_end=True every drone stays in sync
and the rule reduces exactly to the discrete model's.

Person.found_step / supplied_step hold *times* in this model.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from drone_model import DroneSearchModel

EPS = 1e-9


@dataclass
class CDrone:
    id: int
    pos: tuple[int, int]
    has_supplies: bool
    status: str = "idle"        # "search" | "return" | "idle"
    started: float = 0.0        # start of the current search or trip
    until: float = 0.0          # when it ends
    cell: tuple[int, int] | None = None
    deliveries: int = 0


@dataclass
class Decision:
    time: float
    n: int          # unfound people
    R: float        # unsearched area
    E: float        # expected finds over the next T
    H: int
    F: int
    U: int
    S: int          # drones searching after the decision
    r: int          # free empty drones sent back
    mode: str       # "balance" | "search-first" | "surplus" | "fetch-found"


class ContinuousModel(DroneSearchModel):
    name = "expected value"
    rule = "search first if left < T, else min |H+F+r - (U+E[finds in T])|"
    keep_searching = False  # in the search-first phase: r = 0 instead of U - F

    def __init__(self, n_drones: int, grid_size: int, n_people: int,
                 search_time: float = 1.0, trip_time: float = 1.0,
                 cell_area: float = 1.0, start_with_supplies: bool = True,
                 spot_at_end: bool = False, entry_cell: tuple[int, int] = (0, 0),
                 seed: int | None = None):
        super().__init__(n_drones, grid_size, n_people, cell_area=cell_area,
                         start_with_supplies=start_with_supplies,
                         entry_cell=entry_cell, seed=seed)
        self.tau = search_time
        self.T = trip_time
        self.spot_at_end = spot_at_end
        self.drones = [CDrone(d.id, d.pos, d.has_supplies) for d in self.drones]
        self.time = 0.0
        self.completion_time: float | None = 0.0 if self.done else None
        self.searched: set[tuple[int, int]] = set()   # fully searched cells
        self.decisions: list[Decision] = []
        self._events: list[tuple] = []
        self._seq = 0
        self._started = False

    # ------------------------------------------------------------- quantities

    def remaining_area(self, t: float | None = None) -> float:
        """Unclaimed cells plus the unsearched fraction of cells in progress."""
        t = self.time if t is None else t
        partial = sum((d.until - t) / self.tau for d in self.drones if d.status == "search")
        return self.A * (len(self.unsearched) + partial)

    def search_time_left(self, t: float | None = None) -> float:
        """Time for the whole fleet to finish the remaining search."""
        return self.remaining_area(t) / self.A * self.tau / len(self.drones)

    def expected_finds(self, searchers: int, area: float) -> float:
        """Expected people found over the next T by `searchers` drones."""
        if area <= EPS or self.n_unfound == 0:
            return 0.0
        swept = searchers * self.A * self.T / self.tau
        return self.n_unfound * min(1.0, swept / area)

    # --------------------------------------------------------------- decision

    def choose_returners(self, free: list[CDrone], t: float) -> list[CDrone]:
        empty = self._shuffled_empty(free)
        H = sum(d.has_supplies for d in self.drones)
        F = sum(d.status == "return" for d in self.drones)
        U = len(self.waiting)
        busy = sum(d.status == "search" for d in self.drones)
        cells = len(self.unsearched)
        R = self.remaining_area(t)

        if len(free) > cells:
            empty_searchers = max(0, cells - (len(free) - len(empty)))
            r = len(empty) - min(len(empty), empty_searchers)
            S = busy + cells
            mode = "surplus"
        elif self.search_time_left(t) < self.T:
            r = 0 if self.keep_searching else min(len(empty), max(0, U - F))
            S = busy + len(free) - r
            mode = "search-first"
        else:
            best = None
            for r in range(len(empty) + 1):
                S = busy + len(free) - r
                imbalance = abs(H + F + r - (U + self.expected_finds(S, R)))
                if best is None or imbalance < best[0] - 1e-12:
                    best = (imbalance, r, S)
            _, r, S = best
            mode = "balance"

        self.decisions.append(Decision(t, self.n_unfound, R, self.expected_finds(S, R),
                                       H, F, U, S, r, mode))
        return empty[:r]

    # ----------------------------------------------------------------- engine

    def _shuffled_empty(self, free: list[CDrone]) -> list[CDrone]:
        """Free empty drones in random order; returners are taken from the front
        and the rest search in this order (as in the discrete model)."""
        empty = [d for d in free if not d.has_supplies]
        self.rng.shuffle(empty)
        self._empty_order = empty
        return empty

    def _push(self, t: float, kind: str, obj) -> None:
        self._seq += 1
        heapq.heappush(self._events, (t, self._seq, kind, obj))

    def _handle(self, kind: str, obj, t: float) -> None:
        if kind == "search_done":
            self.searched.add(obj.cell)
            obj.status, obj.cell = "idle", None
        elif kind == "arrive":
            obj.status, obj.has_supplies = "idle", True
        elif kind == "spot":
            obj.found_step = t
            self.n_unfound -= 1
            self.waiting.append(obj)

    def _deliver(self, t: float) -> None:
        if not self.waiting:
            return
        # idle loaded drones first, then loaded drones that are mid-search
        loaded = sorted((d for d in self.drones if d.has_supplies),
                        key=lambda d: d.status != "idle")
        for drone in loaded:
            if not self.waiting:
                break
            person = self.waiting.popleft()
            person.supplied_step = t
            person.supplied_by = drone.id
            drone.has_supplies = False
            drone.deliveries += 1
            self.n_supplied += 1

    def _dispatch(self, t: float) -> None:
        free = [d for d in self.drones if d.status == "idle"]
        if not free:
            return
        returners = self.choose_returners(free, t)
        for drone in returners:
            drone.status, drone.started, drone.until = "return", t, t + self.T
            self._push(drone.until, "arrive", drone)
        # loaded drones claim cells before empty ones
        searchers = [d for d in free if d.has_supplies] + self._empty_order[len(returners):]
        for drone in searchers:
            cell = self._pick_cell(drone)
            if cell is None:
                continue  # stays idle
            self.unsearched.discard(cell)
            drone.status, drone.cell, drone.pos = "search", cell, cell
            drone.started, drone.until = t, t + self.tau
            self._push(drone.until, "search_done", drone)
            for person in self.people_in_cell.get(cell, []):
                u = 1.0 if self.spot_at_end else self.rng.random()
                self._push(t + u * self.tau, "spot", person)

    def run_until(self, t_end: float) -> None:
        """Process every event up to and including time t_end."""
        if not self._started:
            self._started = True
            if not self.done:
                self._dispatch(0.0)
        while self._events and not self.done and self._events[0][0] <= t_end + EPS:
            t = self._events[0][0]
            while self._events and self._events[0][0] <= t + EPS:
                _, _, kind, obj = heapq.heappop(self._events)
                self._handle(kind, obj, t)
            self.time = t
            self._deliver(t)
            if self.done:
                self.completion_time = t
                break
            self._dispatch(t)
        if not self.done and math.isfinite(t_end):
            self.time = max(self.time, t_end)

    def run(self, max_time: float = math.inf) -> float:
        """Run to completion. Returns the time the last person was supplied."""
        self.run_until(max_time)
        if not self.done:
            raise RuntimeError("simulation stalled or hit max_time before finishing")
        return self.completion_time


class ContinuousFindThenFetch(ContinuousModel):
    """Baseline: fetch supplies only for people already found and not yet covered."""

    name = "find-then-fetch"
    rule = "r = min(free empty drones, U - F)"
    surplus_rule = False

    def choose_returners(self, free: list[CDrone], t: float) -> list[CDrone]:
        empty = self._shuffled_empty(free)
        F = sum(d.status == "return" for d in self.drones)
        U = len(self.waiting)
        r = min(len(empty), max(0, U - F))
        busy = sum(d.status == "search" for d in self.drones)
        S = busy + min(len(free) - r, len(self.unsearched))
        R = self.remaining_area(t)
        self.decisions.append(Decision(t, self.n_unfound, R, self.expected_finds(S, R),
                                       sum(d.has_supplies for d in self.drones), F, U, S, r,
                                       "fetch-found"))
        return empty[:r]


class ContinuousKeepSearching(ContinuousModel):
    """Expected value, but the search-first phase sends nobody back until the cells run out."""

    name = "keep searching"
    rule = "r = 0 while search left < T, else expected-value balance"
    keep_searching = True


CONTINUOUS_ALGORITHMS: dict[str, type[ContinuousModel]] = {
    "expected": ContinuousModel,
    "keep": ContinuousKeepSearching,
    "brute": ContinuousFindThenFetch,
}
