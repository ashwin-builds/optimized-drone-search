"""
Peak-group search: drones work the prior density surface peak by peak.

Built on ContinuousModel (search while flying, density-based expected value,
the same depot and delivery mechanics) and on the DensitySurface's peaks and
basins (density.py), which every Scenario passes in as `density`.

Groups and peaks
----------------
At t = 0 the top m peaks by remaining mass each get a group, which claims
that peak. A peak is unclaimed, claimed by exactly one group, or exhausted:

    exhausted(peak)  <=>  remaining_mass(peak) < k,   k = k_frac * (largest initial peak mass)

remaining_mass is the prior over the basin's cells still unclaimed, so
exhaustion is derived from what has been searched (by anyone), never stored;
a peak scavengers have already cleared is therefore never picked again.

Group sizes (size_strategy), with m = max(1, n // group_size), group_size = 10:
  fixed    m groups of group_size drones (fewer if n is smaller); leftovers scavenge
  mass     m groups sharing all n drones in proportion to peak mass, >= 2 each
  reserve  scavenger_share of the drones scavenge from the start; the rest
           are shared as in "mass"
(m shrinks if there are fewer non-exhausted peaks, or too few drones for 2 each.)

Group behaviour
---------------
A group drone picks the unclaimed cell of its claimed basin with the best
prob / (travel + tau), searching every unclaimed cell it flies over on the
way (any basin: that is the en-route exception). Whenever a group has a free
drone it re-plans from the free drones' mean position x:

    in-basin rate   max over its basin's unclaimed cells of prob / (travel + tau)
    move rate (q)   remaining_mass(q) / (travel(x -> peak q) + cells_left(q) * tau / (size / 2))

(split_rule="cell", the default; split_rule="basin" instead compares the
move rate with the group's rate on its own basin, remaining_mass(own) /
(travel + cells_left(own) * tau / size), so both sides are group rates)
over candidate peaks q: unclaimed and not exhausted, whose basins border the
group's basin; if none qualify, the two nearest such peaks. If the best move
rate beats the in-basin rate the group splits into two halves (lower drone
ids first), each claiming one of the two best candidates ranked by
mass / distance; with a single candidate, or a single drone present, the
whole group moves. Leaving a basin releases its claim. If the group's basin
is exhausted and there is no candidate, the group dissolves and its drones
become scavengers. A group whose drones have all left also dissolves.
Every split, move and dissolve goes to `group_log`.

Scavengers
----------
Scavengers search the nearest unclaimed cell (the ring search, with en-route
search) but never take, or search en route, a cell in a basin some group
currently claims. A released basin is scavenger territory. With nothing left
to take a scavenger idles until the next event.

People and deliveries
---------------------
`found_by[person]` records the group (or None: scavengers) whose drone was
searching the cell when the person was spotted; `owner[person]` is who must
serve them now (a split hands each waiting person to the nearer half, a
dissolve to the scavengers). A group's loaded drones serve only its own
people; scavengers' loaded drones serve anyone. The worst-served-first
assignment is unchanged, restricted to eligible drone/person pairs.

Depot runs
----------
Each group balances |H + F + r - (U + E)| over its own drones: U its waiting
people, E = n_unfound * (mass of the best cells of its basin its S drones can
sweep in T) / (prior mass left). The scavengers run one pooled balance: E over
unclaimed territory, U = their own waiting people plus each group's waiting
people beyond that group's H + F.

After a depot run (rejoin):
  group       back to the group it left; if that group split, to the half
              with fewer drones (repeatedly); if it dissolved, scavenge
  scavengers  every drone sent to the depot becomes a scavenger

The six size x rejoin combinations are registered as peaks-<size>-<rejoin>.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from continuous_model import (CDrone, ContinuousModel, Decision, SimulationInvariantError,
                              _center, _dist)

SIZE_STRATEGIES = ("fixed", "mass", "reserve")
REJOIN_MODES = ("group", "scavengers")
AWAY = ("exit", "return")


@dataclass
class Group:
    id: int
    peak: int                        # claimed peak id (last one, once inactive)
    members: set = field(default_factory=set)   # drone ids, present and away
    active: bool = True
    children: tuple = ()             # the two halves, once it has split
    dissolved: bool = False


def share_out(total: int, weights: list[float], minimum: int) -> list[int]:
    """Split `total` in proportion to `weights`, at least `minimum` each (largest remainder)."""
    m = len(weights)
    wsum = sum(weights) or 1.0
    ideal = [total * w / wsum for w in weights]
    alloc = [max(minimum, int(x)) for x in ideal]
    while sum(alloc) > total:      # take from whoever is most over their ideal
        i = max((i for i in range(m) if alloc[i] > minimum), key=lambda i: (alloc[i] - ideal[i], -i))
        alloc[i] -= 1
    while sum(alloc) < total:      # give to whoever is most under
        i = max(range(m), key=lambda i: (ideal[i] - alloc[i], -i))
        alloc[i] += 1
    return alloc


class PeakGroupModel(ContinuousModel):
    name = "peak groups"
    rule = "groups work peaks of the prior; scavengers take the rest"
    size_strategy = "mass"
    rejoin = "group"
    k_frac = 0.10
    scavenger_share = 0.3
    group_size = 10
    split_rule = "cell"      # "cell": best in-basin cell rate; "basin": own-basin group rate

    def __init__(self, *args, size_strategy: str | None = None, rejoin: str | None = None,
                 k_frac: float | None = None, scavenger_share: float | None = None,
                 split_rule: str | None = None, group_size: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        for name, value in (("size_strategy", size_strategy), ("rejoin", rejoin),
                            ("k_frac", k_frac), ("scavenger_share", scavenger_share),
                            ("split_rule", split_rule), ("group_size", group_size)):
            if value is not None:
                setattr(self, name, value)
        if self.size_strategy not in SIZE_STRATEGIES or self.rejoin not in REJOIN_MODES \
                or self.split_rule not in ("cell", "basin"):
            raise ValueError(f"size_strategy in {SIZE_STRATEGIES}, rejoin in {REJOIN_MODES}, "
                             f"split_rule in ('cell', 'basin')")
        if self.surface is None:
            raise ValueError("the peak-group model needs density=<DensitySurface>")
        if self.move_time is None:
            raise ValueError("the peak-group model needs move_time_per_cell")
        s = self.surface
        self.peaks = {p.id: p for p in s.peaks}
        self._peak_xy = {p.id: _center(p.cell) for p in s.peaks}
        self._cell_peak = np.array([s.basin_grid[c[1], c[0]] for c in self._cells], dtype=int)
        self._basin_idx = {pid: np.flatnonzero(self._cell_peak == pid) for pid in self.peaks}
        self._basin_left = {pid: float(self._cell_p[idx].sum()) for pid, idx in self._basin_idx.items()}
        self.k_mass = self.k_frac * max(p.mass for p in s.peaks)   # exhaustion threshold "k"
        self._adjacent = self._basin_adjacency()

        self.groups: dict[int, Group] = {}
        self.drone_group: dict[int, int | None] = {d.id: None for d in self.drones}
        self.claims: dict[int, int] = {}                 # peak id -> group id
        self.cell_owner: dict[tuple, int | None] = {}    # group searching each claimed cell
        self.found_by: dict[int, int | None] = {}
        self.owner: dict[int, int | None] = {}
        self.group_log: list[dict] = []
        self._claims_version = 0
        self._cache: dict = {}
        self.scavenger_idle = 0.0          # drone-minutes scavengers spent idle
        self._form_groups()

    # ------------------------------------------------------------- set-up

    def _basin_adjacency(self) -> dict[int, set]:
        b = self.surface.basin_grid
        adj = {pid: set() for pid in self.peaks}
        k = b.shape[0]
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):   # each neighbour pair once
            a = b[0:k - dy, max(0, -dx):k - max(0, dx)]
            c = b[dy:k, max(0, dx):k + min(0, dx)]
            mask = (a >= 0) & (c >= 0) & (a != c)
            for p, q in zip(a[mask], c[mask]):
                adj[int(p)].add(int(q))
                adj[int(q)].add(int(p))
        return adj

    def _form_groups(self) -> None:
        n = len(self.drones)
        ranked = [p for p in sorted(self.peaks, key=lambda q: (-self._basin_left[q], q))
                  if not self._exhausted(p)]
        m = min(max(1, n // self.group_size), len(ranked))
        if self.size_strategy == "fixed":
            sizes = [min(self.group_size, n - i * self.group_size) for i in range(m)]
        else:
            pool = n - (round(self.scavenger_share * n) if self.size_strategy == "reserve" else 0)
            while m > 1 and pool < 2 * m:
                m -= 1
            sizes = share_out(pool, [self._basin_left[p] for p in ranked[:m]], min(2, pool)) \
                if m else []
        ids = sorted(d.id for d in self.drones)
        start = 0
        for pid, size in zip(ranked[:m], sizes):
            if size <= 0:
                continue
            g = self._new_group(pid, set(ids[start:start + size]))
            self.group_log.append({"t": 0.0, "event": "form", "group": g.id, "peak": pid,
                                   "drones": len(g.members)})
            start += size

    def _new_group(self, peak: int, members: set) -> Group:
        g = Group(len(self.groups), peak, set(members))
        self.groups[g.id] = g
        self.claims[peak] = g.id
        self._claims_version += 1
        for d in members:
            self.drone_group[d] = g.id
        return g

    # ------------------------------------------------------------ queries

    def peak_remaining(self, pid: int) -> float:
        return max(0.0, self._basin_left[pid])

    def _exhausted(self, pid: int) -> bool:
        return self.peak_remaining(pid) < self.k_mass

    def _cells_left(self, pid: int) -> int:
        return int(self._alive[self._basin_idx[pid]].sum())

    def _cached(self) -> dict:
        """Scratch cache, emptied whenever cells or peak claims change."""
        state = (self._version, self._claims_version)
        if self._cache.get("state") != state:
            self._cache = {"state": state}
        return self._cache

    def _claimed_mask(self) -> np.ndarray:
        cache = self._cached()
        if "claimed" not in cache:
            cache["claimed"] = np.isin(self._cell_peak, list(self.claims))
        return cache["claimed"]

    def _top_of(self, which, idx: np.ndarray, n_cells: float) -> float:
        """Mass of the n_cells best unclaimed cells among `idx` (cached per claim state)."""
        cache = self._cached()
        key = ("top", which)
        cum = cache.get(key)
        if cum is None:
            p = np.sort(self._cell_p[idx][self._alive[idx]])[::-1]
            cum = cache[key] = np.concatenate([[0.0], np.cumsum(p)])
        if n_cells >= len(cum) - 1:
            return float(cum[-1])
        whole = int(n_cells)
        return float(cum[whole] + (n_cells - whole) * (cum[whole + 1] - cum[whole]))

    def _finds(self, which, idx, searchers: int, mass: float) -> float:
        if self.n_unfound == 0 or mass <= 0:
            return 0.0
        return self.n_unfound * min(1.0, self._top_of(which, idx, searchers * self.T / self.tau) / mass)

    def _members(self, gid: int | None) -> list[CDrone]:
        return [d for d in self.drones if self.drone_group[d.id] == gid]

    def _set_status(self, drone: CDrone, status: str, t: float, until=None, to=None) -> None:
        if drone.status == "idle" and getattr(self, "drone_group", {}).get(drone.id, 0) is None:
            self.scavenger_idle += t - drone.since
        super()._set_status(drone, status, t, until, to)

    def _finish(self, t: float) -> None:
        for d in self.drones:
            if d.status == "idle" and self.drone_group[d.id] is None:
                self.scavenger_idle += t - d.since
        super()._finish(t)

    def extra_metrics(self) -> dict:
        events = [e["event"] for e in self.group_log]
        splits = [e["t"] for e in self.group_log if e["event"] == "split"]
        total = len(self.drones) * (self.completion_time or 0.0) or 1.0
        return {"groups_formed": events.count("form"), "splits": len(splits),
                "moves": events.count("move"), "dissolves": events.count("dissolve"),
                "first_split": splits[0] if splits else float("nan"),
                "scavenger_idle_share": self.scavenger_idle / total}

    def _group_label(self, drone: CDrone):
        gid = getattr(self, "drone_group", {}).get(drone.id)
        return -1 if gid is None else gid

    # ---------------------------------------------------- lineage helpers

    def _resolve(self, gid: int | None, person=None) -> int | None:
        """The active group now responsible for what `gid` was doing.

        After a split: the half nearer `person`, or (for a returning drone) the
        half with fewer drones; after a dissolve: None (scavengers).
        """
        while gid is not None:
            g = self.groups[gid]
            if g.active:
                return gid
            if g.dissolved or not g.children:
                return None
            a, b = (self.groups[c] for c in g.children)
            if person is not None:
                gid = min((a, b), key=lambda h: (_dist(person.xy, self._peak_xy[h.peak]), h.id)).id
            else:
                gid = min((a, b), key=lambda h: (len(h.members), h.id)).id
        return None

    def _set_group(self, drone_id: int, gid: int | None) -> None:
        old = self.drone_group[drone_id]
        if old is not None:
            self.groups[old].members.discard(drone_id)
        if gid is not None:
            self.groups[gid].members.add(drone_id)
        self.drone_group[drone_id] = gid

    # ---------------------------------------------------------- re-planning

    def _rate_in_basin(self, pid: int, pos) -> float:
        idx = self._basin_idx[pid]
        idx = idx[self._alive[idx]]
        if len(idx) == 0:
            return 0.0
        d = self._cell_xy[idx] - np.asarray(pos)
        travel = np.hypot(d[:, 0], d[:, 1]) * self.move_time
        return float((self._cell_p[idx] / (travel + self.tau)).max())

    def _candidates(self, g: Group, pos) -> list[int]:
        ok = lambda q: q != g.peak and q not in self.claims and not self._exhausted(q)
        cands = sorted(q for q in self._adjacent[g.peak] if ok(q))
        if not cands:
            cands = sorted((q for q in self.peaks if ok(q)),
                           key=lambda q: (_dist(pos, self._peak_xy[q]), q))[:2]
        return cands

    def _update_groups(self, t: float) -> None:
        for g in [g for g in self.groups.values() if g.active]:
            present = [d for d in self._members(g.id) if d.status not in AWAY]
            if not g.members:
                self._dissolve(g, t, "no drones left")
                continue
            free = [d for d in present if d.status == "idle"]
            if not free:
                continue
            pos = tuple(np.mean([d.loc for d in free], axis=0))
            cands = self._candidates(g, pos)
            if not cands:
                if self._exhausted(g.peak):
                    self._dissolve(g, t, "no candidate peak left")
                continue
            half = max(1.0, len(present) / 2)
            move = {q: self.peak_remaining(q) /
                    (_dist(pos, self._peak_xy[q]) * self.move_time + self._cells_left(q) * self.tau / half)
                    for q in cands}
            if self.split_rule == "cell":
                stay = self._rate_in_basin(g.peak, pos)
            else:
                left = self._cells_left(g.peak)
                stay = 0.0 if left == 0 else self.peak_remaining(g.peak) / (
                    _dist(pos, self._peak_xy[g.peak]) * self.move_time
                    + left * self.tau / max(1, len(present)))
            if max(move.values()) <= stay:
                continue
            ranked = sorted(cands, key=lambda q: (-self.peak_remaining(q) /
                                                  max(_dist(pos, self._peak_xy[q]), 0.5), q))
            if len(present) >= 2 and len(ranked) >= 2:
                self._split(g, present, ranked[0], ranked[1], t)
            else:
                self._move(g, ranked[0], t)

    def _release(self, g: Group) -> None:
        if self.claims.get(g.peak) == g.id:
            del self.claims[g.peak]
            self._claims_version += 1

    def _move(self, g: Group, peak: int, t: float) -> None:
        self.group_log.append({"t": t, "event": "move", "group": g.id, "from": g.peak, "to": peak,
                               "drones": len(g.members)})
        self._release(g)
        g.peak = peak
        self.claims[peak] = g.id
        self._claims_version += 1

    def _split(self, g: Group, present: list[CDrone], pa: int, pb: int, t: float) -> None:
        ids = sorted(d.id for d in present)
        half = (len(ids) + 1) // 2
        self._release(g)
        g.active = False
        for d in ids:
            g.members.discard(d)
        a = self._new_group(pa, set(ids[:half]))
        b = self._new_group(pb, set(ids[half:]))
        g.children = (a.id, b.id)
        for pid, o in self.owner.items():     # waiting people go to the nearer half
            if o == g.id:
                self.owner[pid] = self._resolve(g.id, self.people[pid])
        self.group_log.append({"t": t, "event": "split", "group": g.id, "from": g.peak,
                               "into": [(a.id, pa, len(a.members)), (b.id, pb, len(b.members))],
                               "away": len(g.members)})

    def _dissolve(self, g: Group, t: float, why: str) -> None:
        self._release(g)
        g.active, g.dissolved = False, True
        for d in [d for d in self._members(g.id) if d.status not in AWAY]:
            self._set_group(d.id, None)
        for pid, o in self.owner.items():
            if o == g.id:
                self.owner[pid] = None
        self.group_log.append({"t": t, "event": "dissolve", "group": g.id, "peak": g.peak,
                               "why": why, "away": len(g.members)})

    # -------------------------------------------------------- engine hooks

    def _claim(self, cell) -> None:
        i = self._cell_index.get(cell)
        was = i is not None and self._alive[i]
        super()._claim(cell)
        if was:
            self._basin_left[int(self._cell_peak[i])] -= self._cell_p[i]

    def _can_claim_en_route(self, drone: CDrone, cell) -> bool:
        if self.drone_group[drone.id] is not None:
            return True
        i = self._cell_index.get(cell)
        return i is None or int(self._cell_peak[i]) not in self.claims

    def _start_search(self, drone: CDrone, cell, t: float) -> None:
        gid = self.drone_group[drone.id]
        if self.check:
            pid = int(self._cell_peak[self._cell_index[cell]])
            if gid is not None and self.groups[gid].peak != pid:
                raise SimulationInvariantError(f"group {gid} drone {drone.id} searching basin {pid}")
            if gid is None and pid in self.claims:
                raise SimulationInvariantError(f"scavenger {drone.id} taking claimed basin {pid}")
        super()._start_search(drone, cell, t)
        for kind, c, *_ in drone.route:
            if kind == "search":
                self.cell_owner[c] = gid
                if self.check and gid is None and int(self._cell_peak[self._cell_index[c]]) in self.claims:
                    raise SimulationInvariantError(f"scavenger {drone.id} searched claimed cell {c}")

    def _handle(self, kind: str, obj, t: float) -> None:
        if kind == "spot":
            finder = self._resolve(self.cell_owner.get(obj.cell), obj)
            self.found_by[obj.id] = self.cell_owner.get(obj.cell)
            self.owner[obj.id] = finder
        super()._handle(kind, obj, t)
        if kind == "arrive" and self.rejoin == "group":
            self._set_group(obj.id, self._resolve(self.drone_group[obj.id]))

    def _send_back(self, drone: CDrone, t: float) -> None:
        if self.rejoin == "scavengers":
            self._set_group(drone.id, None)
        super()._send_back(drone, t)

    def _eligible(self, drone: CDrone, person) -> bool:
        gid = self.drone_group[drone.id]
        return gid is None or self.owner.get(person.id) == gid

    # ------------------------------------------------------------ deciding

    def choose_returners(self, free: list[CDrone], t: float) -> list[CDrone]:
        empty = self._shuffled_empty(free)
        mass = self.remaining_mass(t)
        returners: list[CDrone] = []
        uncovered = 0                        # group demand their own drones can't meet
        for g in sorted((g for g in self.groups.values() if g.active), key=lambda g: g.id):
            members = self._members(g.id)
            H = sum(d.has_supplies and d.status != "deliver" for d in members)
            F = sum(d.status in AWAY for d in members)
            U = sum(1 for p in self.waiting if self.owner.get(p.id) == g.id)
            uncovered += max(0, U - H - F)
            g_free = [d for d in free if self.drone_group[d.id] == g.id]
            if not g_free:
                continue
            g_empty = [d for d in empty if self.drone_group[d.id] == g.id]
            busy = sum(d.status in ("transit", "search") for d in members)
            idx = self._basin_idx[g.peak]
            r, S, E = self._balance(H, F, U, busy, len(g_free), g_empty,
                                    lambda s: self._finds(("basin", g.peak), idx, s, mass))
            returners += g_empty[:r]
            self.decisions.append(Decision(t, self.n_unfound, 0.0, E, H, F, U, S, r, f"group {g.id}"))

        scav = self._members(None)
        s_free = [d for d in free if self.drone_group[d.id] is None]
        if s_free:
            H = sum(d.has_supplies and d.status != "deliver" for d in scav)
            F = sum(d.status in AWAY for d in scav)
            U = sum(1 for p in self.waiting if self.owner.get(p.id) is None) + uncovered
            s_empty = [d for d in empty if self.drone_group[d.id] is None]
            busy = sum(d.status in ("transit", "search") for d in scav)
            idx = np.flatnonzero(~self._claimed_mask())
            r, S, E = self._balance(H, F, U, busy, len(s_free), s_empty,
                                    lambda s: self._finds(("free",), idx, s, mass))
            returners += s_empty[:r]
            self.decisions.append(Decision(t, self.n_unfound, 0.0, E, H, F, U, S, r, "scavengers"))
        return returners

    @staticmethod
    def _balance(H, F, U, busy, n_free, empty, finds):
        best = None
        for r in range(len(empty) + 1):
            S = busy + n_free - r
            E = finds(S)
            imbalance = abs(H + F + r - (U + E))
            if best is None or imbalance < best[0] - 1e-12:
                best = (imbalance, r, S, E)
        return best[1:]

    def _choose_cell(self, drone: CDrone):
        gid = self.drone_group[drone.id]
        if gid is not None:
            idx = self._basin_idx[self.groups[gid].peak]
            idx = idx[self._alive[idx]]
            if len(idx) == 0:
                return None
            d = self._cell_xy[idx] - np.asarray(drone.loc)
            rate = self._cell_p[idx] / (np.hypot(d[:, 0], d[:, 1]) * self.move_time + self.tau)
            return self._cells[idx[int(np.argmax(rate))]]
        # scavenger: nearest unclaimed cell outside claimed basins, ring by ring
        if not (self._alive & ~self._claimed_mask()).any():
            return None                     # nothing to take: idle until the next event
        claimed = self.claims
        x, y = drone.pos
        for r in range(2 * self.k):
            ring = []
            for i in range(-r, r + 1):
                j = r - abs(i)
                for c in {(x + i, y + j), (x + i, y - j)}:
                    if c in self.unsearched and int(self._cell_peak[self._cell_index[c]]) not in claimed:
                        ring.append(c)
            if ring:
                return self.rng.choice(sorted(ring))
        return None

    def _dispatch(self, t: float) -> None:
        self._update_groups(t)
        self._assign(t)
        free = [d for d in self.drones if d.status == "idle"]
        if not free:
            return
        returners = self.choose_returners(free, t)
        gone = {d.id for d in returners}
        for drone in returners:
            self._send_back(drone, t)
        searchers = [d for d in free if d.has_supplies] + \
                    [d for d in self._empty_order if d.id not in gone]
        for drone in searchers:
            cell = self._choose_cell(drone)
            if cell is not None:
                self._start_search(drone, cell, t)

    # -------------------------------------------------------------- checks

    def _check(self, t: float) -> None:
        super()._check(t)

        def fail(msg):
            raise SimulationInvariantError(f"t={t:.4f}: {msg}")

        seen = set()
        for g in self.groups.values():
            if seen & g.members:
                fail(f"a drone is in two groups ({seen & g.members})")
            seen |= g.members
            for d in g.members:
                if self.drone_group[d] != g.id:
                    fail(f"drone {d} listed in group {g.id} but assigned to {self.drone_group[d]}")
        for d, gid in self.drone_group.items():
            if gid is not None and d not in self.groups[gid].members:
                fail(f"drone {d} assigned to group {gid} but not a member")
            if gid is not None and not self.groups[gid].active and \
                    next(x for x in self.drones if x.id == d).status not in AWAY:
                fail(f"drone {d} is present but in inactive group {gid}")
        active = {g.id: g for g in self.groups.values() if g.active}
        if sorted(self.claims.values()) != sorted(active):
            fail("claims and active groups disagree")
        for pid, gid in self.claims.items():
            if active[gid].peak != pid:
                fail(f"peak {pid} claimed by group {gid}, which holds peak {active[gid].peak}")


def _variant(size: str, rejoin: str) -> type[PeakGroupModel]:
    return type(f"Peaks{size.title()}{rejoin.title()}", (PeakGroupModel,), {
        "name": f"peaks {size}/{rejoin}", "size_strategy": size, "rejoin": rejoin,
        "__module__": __name__, "__doc__": f"Peak groups: {size} sizes, rejoin {rejoin}."})


# module-level classes so benchmark.py's process pool can pickle them
PeaksFixedGroup = _variant("fixed", "group")
PeaksFixedScavengers = _variant("fixed", "scavengers")
PeaksMassGroup = _variant("mass", "group")
PeaksMassScavengers = _variant("mass", "scavengers")
PeaksReserveGroup = _variant("reserve", "group")
PeaksReserveScavengers = _variant("reserve", "scavengers")

PEAK_ALGORITHMS = {
    "peaks-fixed-group": PeaksFixedGroup,
    "peaks-fixed-scavengers": PeaksFixedScavengers,
    "peaks-mass-group": PeaksMassGroup,
    "peaks-mass-scavengers": PeaksMassScavengers,
    "peaks-reserve-group": PeaksReserveGroup,
    "peaks-reserve-scavengers": PeaksReserveScavengers,
}

# register with the simulator's algorithm table (see the end of continuous_model.py)
from continuous_model import CONTINUOUS_ALGORITHMS  # noqa: E402

CONTINUOUS_ALGORITHMS.update(PEAK_ALGORITHMS)
