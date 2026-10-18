"""
Continuous-time (event-driven) version of the drone search model.

Every drone is always doing one thing with a known finish time and decides
what to do next the moment it is free:

    search a cell        takes search_time  (tau_s)
    depot round trip     takes trip_time    (T), returns loaded

A person is spotted part-way through their cell's search (Person.spot_frac,
uniform in [0, 1), or at the end with spot_at_end=True). Deliveries are
instant: the moment someone is spotted, any loaded drone drops supplies.

Prior density. `density` gives each search cell a probability of holding
people: a DensitySurface (density.py) on the real campus, or uniform when
omitted. It is the searchers' BELIEF; the true people are placed separately
(people_layout).

Decision rule (expected value). Whenever drones are free, choose how many of
the free empty drones to send back, r, minimising

    | H + F + r  -  (U + E[finds over the next T]) |

    H  loaded drones          F  drones already flying back
    U  people waiting         S  drones searching
    E  = n * min(1, M_top / M_rem)       (finds="density", the default)
         n      people believed still unfound: expected_people minus people
                found so far (expected_people defaults to the true count; a
                scenario with a random count passes only its mean)
         M_rem  prior mass still to search: unclaimed cells, cells claimed
                but not yet searched, and the unsearched part of cells in
                progress
         M_top  mass of the S * T / tau_s highest-probability unclaimed cells,
                i.e. the best cells S drones could sweep in the next T
    E  = n * min(1, S * A * T / (tau_s * R))   (finds="uniform", the old rule;
                R is the remaining area - the same thing under a uniform prior)

The horizon is T because a drone sent back now delivers at now + T. Ties go to
the smaller r. If free drones outnumber unclaimed cells, the surplus rule
applies as in the discrete model: loaded drones take cells first, empty ones
fill any left over, the remaining empty drones return.

ContinuousKeepSearching adds a search-first phase: if the whole fleet could
finish the remaining search in less than one round trip T (remaining cells *
tau_s / N < T), no empty drone goes back (r = 0) until the cells run out,
trading longer waits for an earlier finish. ContinuousModel itself always
uses the balance.

With trip_time == search_time and spot_at_end=True every drone stays in sync
and the rule reduces exactly to the discrete model's.

Physical movement (move_time_per_cell set). Flying inside the search area
takes (straight-line distance in cells) * move_time_per_cell:

  * Serving. A loaded drone must fly to the person's exact position. Whenever
    U people are waiting and there are free loaded drones (a drone mid-search
    finishes its cell first):
      1. take the U + 1 free loaded drones closest to any waiting person;
      2. among unassigned people, find the one whose nearest pool drone is
         furthest away and send that nearest drone to them;
      3. repeat until everyone has a drone or the pool is empty.
    Drones left over (including the "+1") make the usual search/return
    decision. In the balance equation, en-route drones and the people they
    are flying to cancel out: H counts loaded drones not yet committed, U
    unassigned people.
  * Searching while flying (search_en_route=True, the default). Flying over
    a cell takes about as long as searching it, so a drone heading for a
    distant cell searches every still-unclaimed cell its straight line
    crosses: each is claimed at once, searched for tau_s (with the usual spot
    events) and removed from `unsearched`. Cells already claimed or searched
    are flown over at move_time_per_cell per cell of line. The drone's own
    cell and its target are not part of the flight (the move into a
    neighbouring cell is part of the sweep). A flight is a fixed list of legs
    ("search" or "transit"), all scheduled when it starts.
    With search_en_route=False the old rule applies: any distance beyond one
    cell side is flown first ("transit"), then the target is searched.
  * Depot trips (depot_point set). A drone first flies to the point on the
    campus edge facing the depot ("exit"), then makes the off-campus round
    trip T, and comes back in at that same point.

Leaving move_time_per_cell as None keeps instant deliveries, free movement,
and depot trips that start and end where the drone is.

Randomness. self.rng (shared with the discrete model) places people in cells
and breaks ties. A separate stream derived from the seed gives each person an
exact position inside their cell and their spot_frac, so every algorithm sees
exactly the same people at the same moments of their cells' search.

Person.found_step / supplied_step hold *times* in this model.
"""

from __future__ import annotations

import heapq
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from drone_model import DroneSearchModel

EPS = 1e-9
STATUSES = ("idle", "transit", "search", "deliver", "exit", "return")


class SimulationInvariantError(AssertionError):
    """Raised by check=True when the simulator reaches an impossible state."""


@dataclass
class CDrone:
    id: int
    pos: tuple[int, int]             # cell it is in (for choosing cells to search)
    has_supplies: bool
    loc: tuple[float, float] = (0.5, 0.5)   # exact position, in cell units
    status: str = "idle"             # one of STATUSES
    started: float = 0.0             # start of the current activity
    until: float = 0.0               # when it ends
    cell: tuple[int, int] | None = None     # cell being flown to / searched
    search_start: float = 0.0
    deliveries: int = 0
    trips: int = 0
    route: list = field(default_factory=list)   # legs of the current search flight
    leg: int = 0                                 # index of the leg in progress
    origin: tuple[float, float] | None = None   # where the current flight started
    target: object = None                       # the Person it is flying to
    since: float = 0.0
    time_in: dict = field(default_factory=lambda: defaultdict(float))


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
    mode: str       # "balance" | "surplus" | "search-first" | "fetch-found"


def _center(cell: tuple[int, int]) -> tuple[float, float]:
    return (cell[0] + 0.5, cell[1] + 0.5)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class ContinuousModel(DroneSearchModel):
    name = "expected value"
    rule = "r minimises |H + F + r - (U + E[finds in T])|, density-based E"
    keep_searching = False  # r = 0 while the fleet can finish searching within T
    finds = "density"       # how E[finds in T] is computed: "density" or "uniform"

    def __init__(self, n_drones: int, grid_size: int, n_people: int,
                 search_time: float = 1.0, trip_time: float = 1.0,
                 cell_area: float = 1.0, start_with_supplies: bool = True,
                 spot_at_end: bool = False, entry_cell: tuple[int, int] = (0, 0),
                 move_time_per_cell: float | None = None,
                 depot_point: tuple[float, float] | None = None,
                 search_cells=None, people_layout=None, density=None,
                 search_en_route: bool = True, finds: str | None = None,
                 expected_people: float | None = None,
                 record: bool = False, check: bool = False, seed: int | None = None):
        super().__init__(n_drones, grid_size, n_people, cell_area=cell_area,
                         start_with_supplies=start_with_supplies,
                         entry_cell=entry_cell, seed=seed)
        if depot_point is not None and move_time_per_cell is None:
            raise ValueError("depot_point needs move_time_per_cell")
        self.tau = search_time
        self.T = trip_time
        self.spot_at_end = spot_at_end
        self.move_time = move_time_per_cell
        self.depot_point = depot_point
        self.record = record
        self.check = check
        self.search_en_route = search_en_route
        # what the planner believes the head count is (the true count unless told otherwise)
        self.expected_people = float(n_people if expected_people is None else expected_people)
        if finds is not None:
            self.finds = finds
        if self.finds not in ("density", "uniform"):
            raise ValueError("finds must be 'density' or 'uniform'")

        # a real map: only campus cells are searched, people placed by the layout
        self.search_cells = None if search_cells is None else frozenset(search_cells)
        if self.search_cells is not None:
            self.unsearched = set(self.search_cells)
        layout = random.Random(None if seed is None else f"layout:{seed}")
        if people_layout is None:
            for person in self.people:
                person.xy = (person.cell[0] + layout.random(), person.cell[1] + layout.random())
                person.spot_frac = layout.random()
        else:
            self.people_in_cell = {}
            for person, xy in zip(self.people, people_layout(layout, len(self.people))):
                person.xy = xy
                person.cell = (int(xy[0]), int(xy[1]))
                self.people_in_cell.setdefault(person.cell, []).append(person)
            for person in self.people:
                person.spot_frac = layout.random()
        if self.search_cells is not None and any(p.cell not in self.search_cells
                                                 for p in self.people):
            raise ValueError("people_layout put someone outside the search area")

        # prior probability per search cell, plus arrays for fast "best cell" queries
        area_cells = sorted(self.unsearched, key=lambda c: (c[1], c[0]))
        self.surface = density if hasattr(density, "peaks") else None
        prior = getattr(density, "prob", density)
        if prior is None:
            prior = {c: 1.0 / len(area_cells) for c in area_cells}
        missing = [c for c in area_cells if c not in prior]
        if missing:
            raise ValueError(f"density has no value for {len(missing)} search cells")
        self.density = {c: float(prior[c]) for c in area_cells}
        self._cells = area_cells
        self._cell_index = {c: i for i, c in enumerate(area_cells)}
        self._cell_xy = np.array([_center(c) for c in area_cells], dtype=float).reshape(-1, 2)
        self._cell_p = np.array([self.density[c] for c in area_cells], dtype=float)
        self._alive = np.ones(len(area_cells), dtype=bool)   # still in self.unsearched
        self._version = 0                                    # bumped on every claim
        self._top_cache = (-1, None)

        start = depot_point if depot_point is not None else _center(entry_cell)
        self.drones = [CDrone(d.id, self._cell_of(start), d.has_supplies, loc=start)
                       for d in self.drones]
        self.kits_created = sum(d.has_supplies for d in self.drones)
        self.time = 0.0
        self.completion_time: float | None = 0.0 if self.done else None
        self.searched: set[tuple[int, int]] = set()   # fully searched cells
        self.decisions: list[Decision] = []
        # replay log (record=True): drone activity segments and cell search windows
        self.segments: dict[int, list[list]] = {d.id: [] for d in self.drones}
        self.cell_windows: dict[tuple[int, int], tuple[float, float]] = {}
        self.assigned_at: dict[int, float] = {}   # person id -> when a drone was sent
        self._events: list[tuple] = []
        self._seq = 0
        self._started = False
        for drone in self.drones:
            self._log_segment(drone, 0.0, None, drone.loc, drone.loc)

    # ------------------------------------------------------------- quantities

    def _cell_of(self, xy: tuple[float, float]) -> tuple[int, int]:
        clamp = lambda v: min(self.k - 1, max(0, int(math.floor(v))))
        return (clamp(xy[0]), clamp(xy[1]))

    def _pending(self, t: float):
        """(cell, fraction still to search) for every claimed cell not yet searched."""
        for d in self.drones:
            if d.status not in ("transit", "search"):
                continue
            for i in range(d.leg, len(d.route)):
                kind, cell, t0, t1 = d.route[i][:4]
                if kind != "search":
                    continue
                frac = 1.0 if i > d.leg or d.status != "search" else \
                    min(1.0, max(0.0, (t1 - t) / self.tau))
                yield cell, frac

    def remaining_area(self, t: float | None = None) -> float:
        """Unclaimed cells, claimed cells still to search, and the unsearched part of cells in progress."""
        t = self.time if t is None else t
        return self.A * (len(self.unsearched) + sum(f for _, f in self._pending(t)))

    def remaining_mass(self, t: float | None = None) -> float:
        """Prior mass of everything still to search (unclaimed + pending)."""
        t = self.time if t is None else t
        return float(self._cell_p[self._alive].sum()) + \
            sum(self.density[c] * f for c, f in self._pending(t))

    def _top_mass(self, n_cells: float) -> float:
        """Mass of the n_cells highest-probability unclaimed cells (fractional n allowed)."""
        if self._top_cache[0] != self._version:
            p = np.sort(self._cell_p[self._alive])[::-1]
            self._top_cache = (self._version, np.concatenate([[0.0], np.cumsum(p)]))
        cum = self._top_cache[1]
        if n_cells >= len(cum) - 1:
            return float(cum[-1])
        whole = int(n_cells)
        return float(cum[whole] + (n_cells - whole) * (cum[whole + 1] - cum[whole]))

    def _claim(self, cell: tuple[int, int]) -> None:
        """Take a cell out of the unsearched pool (it is now some drone's to search)."""
        if self.check and cell in self.cell_windows:
            raise SimulationInvariantError(f"cell {cell} searched twice")
        self.unsearched.discard(cell)
        i = self._cell_index.get(cell)
        if i is not None and self._alive[i]:
            self._alive[i] = False
            self._version += 1

    def loaded_available(self) -> int:
        """H: loaded drones not already flying to someone."""
        return sum(d.has_supplies and d.status != "deliver" for d in self.drones)

    def flying_back(self) -> int:
        """F: drones on their way to the depot or back from it."""
        return sum(d.status in ("exit", "return") for d in self.drones)

    def searching(self) -> int:
        return sum(d.status in ("transit", "search") for d in self.drones)

    def search_time_left(self, t: float | None = None) -> float:
        """Time for the whole fleet to finish the remaining search."""
        return self.remaining_area(t) / self.A * self.tau / len(self.drones)

    @property
    def believed_unfound(self) -> float:
        """n for planning: expected head count minus people found so far, at least 0."""
        return max(0.0, self.expected_people - (len(self.people) - self.n_unfound))

    def expected_finds(self, searchers: int, area: float, mass: float | None = None) -> float:
        """Expected people found over the next T by `searchers` drones.

        finds="density": n * (mass of the best cells they can sweep) / (mass left).
        finds="uniform": n * (area they can sweep) / (area left).
        """
        n = self.believed_unfound
        if n <= 0:
            return 0.0
        if self.finds == "uniform":
            if area <= EPS:
                return 0.0
            swept = searchers * self.A * self.T / self.tau
            return n * min(1.0, swept / area)
        mass = self.remaining_mass() if mass is None else mass
        if mass <= 0:
            return 0.0
        return n * min(1.0, self._top_mass(searchers * self.T / self.tau) / mass)

    # --------------------------------------------------------------- decision

    def choose_returners(self, free: list[CDrone], t: float) -> list[CDrone]:
        empty = self._shuffled_empty(free)
        H = self.loaded_available()
        F = self.flying_back()
        U = len(self.waiting)
        busy = self.searching()
        cells = len(self.unsearched)
        R = self.remaining_area(t)
        M = self.remaining_mass(t) if self.finds == "density" else None

        if len(free) > cells:
            empty_searchers = max(0, cells - (len(free) - len(empty)))
            r = len(empty) - min(len(empty), empty_searchers)
            S = busy + cells
            mode = "surplus"
        elif self.keep_searching and self.search_time_left(t) < self.T:
            r = 0
            S = busy + len(free)
            mode = "search-first"
        else:
            best = None
            for r in range(len(empty) + 1):
                S = busy + len(free) - r
                imbalance = abs(H + F + r - (U + self.expected_finds(S, R, M)))
                if best is None or imbalance < best[0] - 1e-12:
                    best = (imbalance, r, S)
            _, r, S = best
            mode = "balance"

        self.decisions.append(Decision(t, round(self.believed_unfound), R,
                                       self.expected_finds(S, R, M), H, F, U, S, r, mode))
        return empty[:r]

    # ------------------------------------------------------------- bookkeeping

    def _shuffled_empty(self, free: list[CDrone]) -> list[CDrone]:
        """Free empty drones in random order; returners are taken from the front
        and the rest search in this order (as in the discrete model)."""
        empty = [d for d in free if not d.has_supplies]
        self.rng.shuffle(empty)
        self._empty_order = empty
        return empty

    def _log_segment(self, drone: CDrone, t: float, until: float | None,
                     xy0: tuple[float, float], xy1: tuple[float, float]) -> None:
        if not self.record:
            return
        segs = self.segments[drone.id]
        if segs and segs[-1][1] is None:
            segs[-1][1] = t
        segs.append([t, until, drone.status, drone.has_supplies, *xy0, *xy1,
                     self._group_label(drone)])

    def extra_metrics(self) -> dict:
        """Model-specific numbers for metrics.summarize (hook; none by default)."""
        return {}

    def _group_label(self, drone: CDrone):
        """Group id recorded in the replay log (None: no groups in this model)."""
        return None

    def _eligible(self, drone: CDrone, person) -> bool:
        """Whether a free loaded drone may serve this waiting person (hook)."""
        return True

    def _set_status(self, drone: CDrone, status: str, t: float, until: float | None = None,
                    to: tuple[float, float] | None = None) -> None:
        """Change what a drone is doing, keeping time accounting and the replay log."""
        drone.time_in[drone.status] += t - drone.since
        drone.since = t
        drone.status = status
        drone.started = t
        if until is not None:
            drone.until = until
        self._log_segment(drone, t, until, drone.loc, to if to is not None else drone.loc)

    def _push(self, t: float, kind: str, obj) -> None:
        self._seq += 1
        heapq.heappush(self._events, (t, self._seq, kind, obj))

    # ----------------------------------------------------------------- events

    def _handle(self, kind: str, obj, t: float) -> None:
        if kind == "leg_done":
            drone, i = obj
            leg_kind, cell, _, _, _, xy1 = drone.route[i]
            drone.loc = xy1
            if leg_kind == "search":
                self.searched.add(cell)
            if i + 1 < len(drone.route):
                self._start_leg(drone, i + 1, t)
            else:
                drone.pos, drone.cell, drone.route, drone.leg = cell, None, [], 0
                self._set_status(drone, "idle", t)
        elif kind == "exit_done":
            obj.loc = self.depot_point
            obj.pos = self._cell_of(self.depot_point)
            self._set_status(obj, "return", t, t + self.T)
            self._push(t + self.T, "arrive", obj)
        elif kind == "arrive":
            obj.has_supplies = True
            self.kits_created += 1
            self._set_status(obj, "idle", t)
        elif kind == "spot":
            obj.found_step = t
            self.n_unfound -= 1
            self.waiting.append(obj)
        elif kind == "delivered":
            person = obj.target
            if self.check and person.supplied_by is not None:
                raise SimulationInvariantError(f"person {person.id} supplied twice")
            person.supplied_step = t
            person.supplied_by = obj.id
            obj.loc, obj.pos = person.xy, person.cell
            obj.has_supplies, obj.target, obj.origin = False, None, None
            obj.deliveries += 1
            self.n_supplied += 1
            self._set_status(obj, "idle", t)

    def _assign(self, t: float) -> None:
        """Send free loaded drones to waiting people (see module docstring)."""
        free = [d for d in self.drones if d.status == "idle" and d.has_supplies]
        if not free or not self.waiting:
            return
        people = list(self.waiting)
        drone_xy = np.array([d.loc for d in free], dtype=float)
        people_xy = np.array([p.xy for p in people], dtype=float)
        dist = np.hypot(drone_xy[:, None, 0] - people_xy[None, :, 0],
                        drone_xy[:, None, 1] - people_xy[None, :, 1])  # [drone, person]
        eligible = np.array([[self._eligible(d, p) for p in people] for d in free])
        dist = np.where(eligible, dist, np.inf)        # ineligible pairs never match
        free = [d for d, ok in zip(free, eligible.any(axis=1)) if ok]
        dist = dist[eligible.any(axis=1)]
        if not free:
            return

        # pool: the U + 1 drones closest to any waiting person
        pool = np.argsort(dist.min(axis=1), kind="stable")[:len(people) + 1]
        d = np.full_like(dist, np.inf)
        d[pool] = dist[pool]
        while np.isfinite(d).any():
            nearest = d.min(axis=0)                     # each person's nearest pool drone
            nearest[~np.isfinite(nearest)] = -1.0       # already assigned
            j = int(np.argmax(nearest))                 # worst-served person first
            i = int(np.argmin(d[:, j]))
            drone, person = free[i], people[j]
            self.waiting.remove(person)
            self.assigned_at[person.id] = t
            drone.target, drone.origin = person, drone.loc
            until = t + dist[i, j] * self.move_time
            self._set_status(drone, "deliver", t, until, to=person.xy)
            self._push(until, "delivered", drone)
            d[i, :] = np.inf
            d[:, j] = np.inf

    def _deliver(self, t: float) -> None:
        """Instant deliveries (move_time_per_cell is None)."""
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

    def _send_back(self, drone: CDrone, t: float) -> None:
        drone.trips += 1
        if self.depot_point is None:
            self._set_status(drone, "return", t, t + self.T)
            self._push(t + self.T, "arrive", drone)
        else:
            drone.origin = drone.loc
            until = t + _dist(drone.loc, self.depot_point) * self.move_time
            self._set_status(drone, "exit", t, until, to=self.depot_point)
            self._push(until, "exit_done", drone)

    def _crossed(self, p: tuple[float, float], q: tuple[float, float]):
        """Cells the segment p -> q passes through, in order, as (cell, entry, exit)
        points; a line through a cell corner steps diagonally."""
        x, y = int(math.floor(p[0])), int(math.floor(p[1]))
        dx, dy = q[0] - p[0], q[1] - p[1]
        sx, sy = (1 if dx > 0 else -1), (1 if dy > 0 else -1)
        tx = ((x + (sx > 0)) - p[0]) / dx if dx else math.inf
        ty = ((y + (sy > 0)) - p[1]) / dy if dy else math.inf
        ddx, ddy = (abs(1 / dx) if dx else math.inf), (abs(1 / dy) if dy else math.inf)
        at = lambda s: (p[0] + s * dx, p[1] + s * dy)
        s, out = 0.0, []
        while True:
            nxt = min(tx, ty, 1.0)
            out.append(((x, y), at(s), at(nxt)))
            if nxt >= 1.0 - 1e-12:
                return out
            if abs(tx - ty) < 1e-12:
                x, y, tx, ty = x + sx, y + sy, tx + ddx, ty + ddy
            elif tx < ty:
                x, tx = x + sx, tx + ddx
            else:
                y, ty = y + sy, ty + ddy
            s = nxt

    def _can_claim_en_route(self, drone: CDrone, cell: tuple[int, int]) -> bool:
        """Whether `drone` may search `cell` while flying over it (hook for subclasses)."""
        return True

    def _route(self, drone: CDrone, target: tuple[int, int], t: float) -> list[tuple]:
        """Legs (kind, cell, t0, t1, xy0, xy1) of a flight to search `target`."""
        legs, now, here = [], t, drone.loc
        end = _center(target)
        if self.move_time is not None and not self.search_en_route:
            travel = max(0.0, _dist(here, end) - 1.0) * self.move_time
            if travel > EPS:
                legs.append(("transit", target, now, now + travel, here, end))
                now += travel
        elif self.move_time is not None:
            crossed = self._crossed(here, end)[1:-1]   # not the drone's own cell or the target
            for i, (cell, a, b) in enumerate(crossed):
                xy0 = here if i == 0 else a
                xy1 = end if i == len(crossed) - 1 else b
                if cell in self.unsearched and self._can_claim_en_route(drone, cell):
                    legs.append(("search", cell, now, now + self.tau, xy0, xy1))
                    now += self.tau
                else:
                    dur = _dist(a, b) * self.move_time
                    if legs and legs[-1][0] == "transit":     # merge consecutive fly-overs
                        k_, c_, t0, _, p0, _ = legs[-1]
                        legs[-1] = (k_, c_, t0, now + dur, p0, xy1)
                    elif dur > EPS:
                        legs.append(("transit", target, now, now + dur, xy0, xy1))
                    now += dur
        legs.append(("search", target, now, now + self.tau, end, end))
        return legs

    def _start_leg(self, drone: CDrone, i: int, t: float) -> None:
        kind, cell, _, t1, xy0, xy1 = drone.route[i]
        drone.leg, drone.cell = i, cell
        if kind == "transit":
            drone.origin = drone.loc
        else:
            drone.pos = cell
        self._set_status(drone, kind, t, t1, to=xy1)

    def _start_search(self, drone: CDrone, cell: tuple[int, int], t: float) -> None:
        """Fly to `cell` (searching on the way if enabled) and search it."""
        drone.route = self._route(drone, cell, t)
        for kind, c, t0, t1, _, _ in drone.route:
            if kind == "search":
                self._claim(c)
        drone.search_start = drone.route[-1][2]
        # all leg ends and spots are scheduled now (and in this order, which
        # keeps same-time ties identical to the single-cell flights)
        for i, (kind, c, t0, t1, _, _) in enumerate(drone.route):
            self._push(t1, "leg_done", (drone, i))
            if kind == "search":
                self.cell_windows[c] = (t0, t1)
                for person in self.people_in_cell.get(c, []):
                    frac = 1.0 if self.spot_at_end else person.spot_frac
                    self._push(t0 + frac * self.tau, "spot", person)
        self._start_leg(drone, 0, t)

    def _dispatch(self, t: float) -> None:
        if self.move_time is not None:
            self._assign(t)
        free = [d for d in self.drones if d.status == "idle"]
        if not free:
            return
        returners = self.choose_returners(free, t)
        for drone in returners:
            self._send_back(drone, t)
        # loaded drones claim cells before empty ones
        searchers = [d for d in free if d.has_supplies] + self._empty_order[len(returners):]
        for drone in searchers:
            cell = self._pick_cell(drone)
            if cell is not None:
                self._start_search(drone, cell, t)
            # otherwise it stays idle

    # ------------------------------------------------------------------ checks

    def _check(self, t: float) -> None:
        def fail(msg):
            raise SimulationInvariantError(f"t={t:.4f}: {msg}")

        loaded = sum(d.has_supplies for d in self.drones)
        if self.kits_created != self.n_supplied + loaded:
            fail(f"kits not conserved: created {self.kits_created}, "
                 f"used {self.n_supplied}, held {loaded}")
        if self.n_unfound != sum(p.found_step is None for p in self.people):
            fail("unfound count out of sync")
        if self.n_supplied != sum(p.supplied_step is not None for p in self.people):
            fail("supplied count out of sync")
        waiting = {p.id for p in self.waiting}
        targeted = [d.target.id for d in self.drones if d.status == "deliver"]
        if len(set(targeted)) != len(targeted):
            fail("two drones flying to the same person")
        supplied = {p.id for p in self.people if p.supplied_step is not None}
        found = {p.id for p in self.people if p.found_step is not None}
        if waiting & set(targeted) or waiting & supplied or set(targeted) & supplied:
            fail("a person is in more than one of waiting / targeted / supplied")
        if self.move_time is not None and waiting | set(targeted) | supplied != found:
            fail("found people not accounted for")
        claimed = []
        for d in self.drones:
            if d.status not in STATUSES:
                fail(f"drone {d.id} has unknown status {d.status}")
            if d.status in ("transit", "search"):
                if d.cell is None or d.cell in self.unsearched or not d.route:
                    fail(f"drone {d.id} searching an unclaimed cell")
                for i in range(d.leg, len(d.route)):
                    kind, cell = d.route[i][:2]
                    if kind == "search":
                        if cell in self.unsearched or cell in self.searched:
                            fail(f"drone {d.id} has cell {cell} queued but it is not claimed")
                        claimed.append(cell)
            if d.status == "deliver" and (not d.has_supplies or d.target is None):
                fail(f"drone {d.id} delivering without a kit or a target")
            if d.status in ("exit", "return") and d.has_supplies:
                fail(f"drone {d.id} flying to the depot with a kit")
            if d.status != "idle" and d.until < t - EPS:
                fail(f"drone {d.id} is {d.status} past its end time")
        if len(set(claimed)) != len(claimed):
            fail("two drones searching the same cell")
        if int(self._alive.sum()) != len(self.unsearched):   # every claim goes through _claim
            fail("unsearched set and the cell arrays disagree")

    # ------------------------------------------------------------------- run

    def _finish(self, t: float) -> None:
        self.completion_time = t
        for drone in self.drones:
            drone.time_in[drone.status] += t - drone.since
            drone.since = t
            if self.record:
                segs = self.segments[drone.id]
                if segs and segs[-1][1] is None:
                    segs[-1][1] = t

    def run_until(self, t_end: float) -> None:
        """Process every event up to and including time t_end."""
        if not self._started:
            self._started = True
            if not self.done:
                self._dispatch(0.0)
            else:
                self._finish(0.0)
        while self._events and not self.done and self._events[0][0] <= t_end + EPS:
            t = self._events[0][0]
            while self._events and self._events[0][0] <= t + EPS:
                _, _, kind, obj = heapq.heappop(self._events)
                self._handle(kind, obj, t)
            self.time = t
            if self.move_time is None:
                self._deliver(t)
            if self.done:
                self._finish(t)
                break
            self._dispatch(t)
            if self.check:
                self._check(t)
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
        F = self.flying_back()
        U = len(self.waiting)
        r = min(len(empty), max(0, U - F))
        S = self.searching() + min(len(free) - r, len(self.unsearched))
        R = self.remaining_area(t)
        self.decisions.append(Decision(t, round(self.believed_unfound), R, self.expected_finds(S, R),
                                       self.loaded_available(), F, U, S, r,
                                       "fetch-found"))
        return empty[:r]


class ContinuousKeepSearching(ContinuousModel):
    """Expected value, but the search-first phase sends nobody back until the cells run out."""

    name = "keep searching"
    rule = "r = 0 while search left < T, else expected-value balance"
    keep_searching = True


class ContinuousExpectedUniform(ContinuousModel):
    """The expected-value rule with the old uniform E (area swept / area left), for reference."""

    name = "expected (uniform E)"
    rule = "r minimises |H + F + r - (U + E[finds in T])|, uniform E"
    finds = "uniform"


class ContinuousGreedyDensity(ContinuousModel):
    """Each free drone takes the unclaimed cell with the best prior probability per
    unit of time, prob / (travel + tau); depot runs as in the expected-value rule."""

    name = "greedy density"
    rule = "cell maximising prob / (travel + tau); expected-value depot runs"

    def _pick_cell(self, drone: CDrone):
        idx = np.flatnonzero(self._alive)
        if len(idx) == 0:
            return None
        d = self._cell_xy[idx] - np.asarray(drone.loc)
        travel = np.hypot(d[:, 0], d[:, 1]) * (self.move_time or 0.0)
        rate = self._cell_p[idx] / (travel + self.tau)
        return self._cells[idx[int(np.argmax(rate))]]   # ties: first in (row, column) order


# the algorithms offered in the dashboard, benchmarks and CLIs; the other classes above
# (keep searching, uniform E, greedy density) stay importable for reference and tests
CONTINUOUS_ALGORITHMS: dict[str, type[ContinuousModel]] = {
    "expected": ContinuousModel,
    "brute": ContinuousFindThenFetch,
}

# the peak-group model lives in peaks_model.py, which imports this module and
# adds itself to CONTINUOUS_ALGORITHMS (works whichever module is imported first)
import peaks_model  # noqa: E402,F401
