"""
Tests for the peak-group model (peaks_model.py) on small made-up campuses.
"""

import math
import pickle

import pytest
from shapely.geometry import Polygon, mapping

import campus_map
from campus_map import CampusMap
from continuous_model import CONTINUOUS_ALGORITHMS, SimulationInvariantError
from density import DensityLayout, DensitySurface, VisitorRow
from peaks_model import (PEAK_ALGORITHMS, PeakGroupModel, PeaksFixedGroup, PeaksMassGroup,
                         share_out)

LON0, LAT0 = -122.31, 47.65
M_PER_DEG_LAT = math.pi / 180 * campus_map.EARTH_R
M_PER_DEG_LON = M_PER_DEG_LAT * math.cos(math.radians(LAT0))


def ll(x, y):
    return (LON0 + x / M_PER_DEG_LON, LAT0 + y / M_PER_DEG_LAT)


def box(x0, y0, x1, y1):
    return Polygon([ll(x0, y0), ll(x1, y0), ll(x1, y1), ll(x0, y1)])


def world(buildings, w=1000, h=300):
    """buildings: (cx, cy, name, visitors) with 50 m square footprints; 25 m cells."""
    feats = [{"type": "Feature", "properties": {"role": "campus"},
              "geometry": mapping(box(0, 0, w, h))}]
    for cx, cy, name, _ in buildings:
        feats.append({"type": "Feature", "geometry": mapping(box(cx - 25, cy - 25, cx + 25, cy + 25)),
                      "properties": {"role": "building", "name": name, "building": "university",
                                     "levels": 2}})
    cmap = CampusMap({"type": "FeatureCollection", "features": feats})
    grid = cmap.grid(25.0, "redmond")
    vis = {n: VisitorRow(n, v, "test", False) for _, _, n, v in buildings}
    return grid, DensitySurface(cmap, grid, vis, scale="mass")


def make(cls, grid, surface, n=10, p=0, **kw):
    args = dict(search_time=1.0, trip_time=40.0, move_time_per_cell=0.2, cell_area=1.0,
                density=surface, search_cells=grid.search_cells, entry_cell=(0, 0),
                people_layout=DensityLayout(surface.prob),
                depot_point=(0.01, 0.01), seed=0, check=True)
    return cls(n, grid.k, p, **(args | kw))


# five separate buildings with clearly different visitor counts
FIVE = [(112.5, 162.5, "A", 5000), (312.5, 162.5, "B", 3000), (512.5, 162.5, "C", 1500),
        (712.5, 162.5, "D", 800), (912.5, 162.5, "E", 400)]
# a strong middle peak with weaker neighbours either side
ROW = [(212.5, 162.5, "W", 1500), (512.5, 162.5, "A", 4000), (812.5, 162.5, "E", 2000)]


def peak_named(surface, name):
    return next(p.id for p in surface.peaks if name in p.buildings)


def claim_basin(model, pid):
    """Pretend every cell of a basin has been searched."""
    for c in sorted(model.peaks[pid].basin):
        model._claim(c)
        model.cell_windows[c] = (0.0, 0.0)


# ------------------------------------------------------------- set-up

def test_share_out_is_proportional_with_a_minimum():
    assert share_out(20, [5, 3, 2], 2) == [10, 6, 4]
    assert sum(share_out(25, [100, 1, 1], 2)) == 25
    assert min(share_out(25, [100, 1, 1], 2)) == 2


@pytest.mark.parametrize("strategy, groups, scavengers", [
    ("fixed", [10, 10], 5),
    ("mass", None, 0),
    ("reserve", None, 8),        # round(0.3 * 25) = 8 scavenge
])
def test_initial_groups_follow_the_size_strategy(strategy, groups, scavengers):
    grid, s = world(FIVE)
    m = make(PeakGroupModel, grid, s, n=25, size_strategy=strategy)
    sizes = [len(g.members) for g in m.groups.values()]
    assert len(m.groups) == 2                                       # m = 25 // 10
    assert sorted(m.claims) == sorted([peak_named(s, "A"), peak_named(s, "B")])
    assert sum(gid is None for gid in m.drone_group.values()) == scavengers
    if groups:
        assert sizes == groups
    else:  # shared out by mass (A : B), at least 2 each
        pool = 25 - scavengers
        a, b = (m.peaks[g.peak].mass for g in m.groups.values())
        assert sizes == share_out(pool, [a, b], 2) and sum(sizes) == pool
        assert sizes[0] > sizes[1] >= 2


def test_needs_a_density_surface():
    grid, s = world(FIVE)
    with pytest.raises(ValueError, match="DensitySurface"):
        make(PeakGroupModel, grid, s, density=s.prob)


# ------------------------------------------------------- split / dissolve

def test_group_splits_when_its_basin_is_empty_and_neighbours_are_worth_more():
    grid, s = world(ROW)
    m = make(PeaksFixedGroup, grid, s, n=10)
    a, w, e = (peak_named(s, x) for x in "AWE")
    (g,) = m.groups.values()
    assert g.peak == a and len(g.members) == 10
    claim_basin(m, a)
    for d in m.drones:
        d.loc = (20.5, 6.5)
    m._update_groups(1.0)
    (ev,) = [x for x in m.group_log if x["event"] == "split"]
    assert ev["from"] == a and {p for _, p, _ in ev["into"]} == {w, e}
    halves = [m.groups[h] for h, _, _ in ev["into"]]
    assert [len(h.members) for h in halves] == [5, 5]
    assert not g.active and a not in m.claims
    assert {m.claims[w], m.claims[e]} == {h.id for h in halves}


def test_group_moves_whole_when_only_one_candidate():
    grid, s = world(ROW)
    m = make(PeaksFixedGroup, grid, s, n=10)
    a, w, e = (peak_named(s, x) for x in "AWE")
    claim_basin(m, a)
    claim_basin(m, w)                       # W already cleared: exhausted, never a candidate
    m._update_groups(1.0)
    (g,) = [g for g in m.groups.values() if g.active]
    assert g.peak == e and len(g.members) == 10
    assert [x["event"] for x in m.group_log] == ["form", "move"]


def test_peak_cleared_by_scavengers_is_never_claimed():
    grid, s = world(FIVE)
    m = make(PeaksFixedGroup, grid, s, n=10)          # one group on A
    a, b = peak_named(s, "A"), peak_named(s, "B")
    claim_basin(m, b)                                  # scavengers swept B
    assert m._exhausted(b)
    claim_basin(m, a)
    for _ in range(4):                                 # several re-plans
        for d in m.drones:
            if m.drone_group[d.id] is not None:
                d.status = "idle"
        m._update_groups(1.0)
    claimed_ever = {x.get("to") for x in m.group_log} | \
        {p for x in m.group_log for _, p, _ in x.get("into", [])}
    assert b not in claimed_ever and b not in m.claims


def test_group_dissolves_when_every_candidate_is_below_k():
    grid, s = world(ROW)
    m = make(PeaksFixedGroup, grid, s, n=10)
    for name in "AWE":
        claim_basin(m, peak_named(s, name))
    m._update_groups(1.0)
    (g,) = m.groups.values()
    assert g.dissolved and not m.claims
    assert all(gid is None for gid in m.drone_group.values())
    assert m.group_log[-1]["event"] == "dissolve"


def test_k_is_relative_to_the_largest_peak():
    grid, s = world(FIVE)
    m = make(PeakGroupModel, grid, s, k_frac=0.2)
    assert m.k_mass == pytest.approx(0.2 * max(p.mass for p in s.peaks))
    small = peak_named(s, "E")
    assert m._exhausted(small) == (m.peak_remaining(small) < m.k_mass)


# ------------------------------------------------------- people & deliveries

def test_delivery_eligibility_group_only_vs_scavengers_to_anyone():
    grid, s = world(FIVE)
    m = make(PeakGroupModel, grid, s, n=25, p=3, size_strategy="fixed")
    g0, g1 = sorted(m.groups)
    group_drone = next(d for d in m.drones if m.drone_group[d.id] == g0)
    scav = next(d for d in m.drones if m.drone_group[d.id] is None)
    mine, theirs, nobody = m.people
    m.owner = {mine.id: g0, theirs.id: g1, nobody.id: None}
    assert m._eligible(group_drone, mine)
    assert not m._eligible(group_drone, theirs) and not m._eligible(group_drone, nobody)
    assert all(m._eligible(scav, p) for p in m.people)


def test_assignment_only_pairs_eligible_drones_and_people():
    grid, s = world(FIVE)
    m = make(PeakGroupModel, grid, s, n=25, p=1, size_strategy="fixed")
    person = m.people[0]
    person.found_step = 0.0
    m.waiting.append(person)
    m.owner[person.id] = None                          # found by scavengers
    for d in m.drones:
        d.has_supplies, d.status = True, "idle"
    near = next(d for d in m.drones if m.drone_group[d.id] is not None)
    near.loc = person.xy                               # a group drone right on top of them
    far = next(d for d in m.drones if m.drone_group[d.id] is None)
    far.loc = (39.5, 10.5)
    others = [d for d in m.drones if d not in (near, far)]
    for d in others:
        d.has_supplies = False
    m._assign(1.0)
    assert near.status == "idle" and far.status == "deliver" and far.target is person


# ------------------------------------------------------------- rejoin

def test_rejoin_group_returns_to_the_group_or_the_smaller_half():
    grid, s = world(ROW)
    m = make(PeaksFixedGroup, grid, s, n=10)
    (g,) = m.groups.values()
    away = m.drones[9]
    away.has_supplies = False
    m._send_back(away, 0.0)
    assert m.drone_group[away.id] == g.id              # still the group's while away
    claim_basin(m, peak_named(s, "A"))
    m._update_groups(1.0)                              # splits 9 present drones: 5 + 4
    a, b = (m.groups[h] for h in g.children)
    assert (len(a.members), len(b.members)) == (5, 4)
    m._handle("arrive", away, 50.0)
    assert m.drone_group[away.id] == b.id and away.id in b.members


def test_rejoin_group_after_dissolve_becomes_a_scavenger():
    grid, s = world(ROW)
    m = make(PeaksFixedGroup, grid, s, n=10)
    away = m.drones[0]
    away.has_supplies = False
    m._send_back(away, 0.0)
    for name in "AWE":
        claim_basin(m, peak_named(s, name))
    m._update_groups(1.0)
    m._handle("arrive", away, 50.0)
    assert m.drone_group[away.id] is None


def test_rejoin_scavengers_turns_every_depot_runner_into_a_scavenger():
    grid, s = world(ROW)
    m = make(PEAK_ALGORITHMS["peaks-fixed-scavengers"], grid, s, n=10)
    d = m.drones[3]
    d.has_supplies = False
    m._send_back(d, 0.0)
    assert m.drone_group[d.id] is None
    m._handle("arrive", d, 50.0)
    assert m.drone_group[d.id] is None


# ----------------------------------------------------------- whole runs

def test_no_stall_when_groups_claim_every_cell():
    # two peaks, two groups claim both basins (all of campus); 5 scavengers have nothing
    grid, s = world([(250, 150, "L", 3000), (750, 150, "R", 3000)])
    m = make(PeakGroupModel, grid, s, n=25, p=30, size_strategy="fixed")
    assert len(m.claims) == len(s.peaks) == 2
    assert m._choose_cell(next(d for d in m.drones if m.drone_group[d.id] is None)) is None
    m.run()                                            # raises if it stalls
    assert m.n_supplied == 30


@pytest.mark.parametrize("key", list(PEAK_ALGORITHMS))
def test_every_variant_runs_with_checks(key):
    grid, s = world(FIVE)
    m = make(PEAK_ALGORITHMS[key], grid, s, n=23, p=40, record=True, seed=3)
    m.run()
    assert m.n_supplied == 40
    assert set(m.found_by) == {p.id for p in m.people}
    # the replay log carries each drone's group (-1 = scavenger)
    labels = {seg[-1] for segs in m.segments.values() for seg in segs}
    assert -1 in labels or all(gid is not None for gid in m.drone_group.values())
    assert labels - {-1} <= set(m.groups)


def test_seeded_runs_are_deterministic():
    grid, s = world(FIVE)
    runs = [make(PeaksMassGroup, grid, s, n=23, p=40, seed=5, check=False) for _ in range(2)]
    for m in runs:
        m.run()
    assert runs[0].completion_time == runs[1].completion_time
    assert runs[0].group_log == runs[1].group_log
    assert runs[0].found_by == runs[1].found_by


def test_variants_are_registered_and_picklable():
    for key, cls in PEAK_ALGORITHMS.items():
        assert CONTINUOUS_ALGORITHMS[key] is cls
        assert pickle.loads(pickle.dumps(cls)) is cls
        size, rejoin = key.split("-")[1:]
        assert (cls.size_strategy, cls.rejoin) == (size, rejoin)


def test_check_catches_a_group_drone_outside_its_basin():
    grid, s = world(FIVE)
    m = make(PeaksFixedGroup, grid, s, n=10)
    d = m.drones[0]
    other = next(p for p in s.peaks if p.id != m.groups[0].peak)
    with pytest.raises(SimulationInvariantError, match="searching basin"):
        m._start_search(d, sorted(other.basin)[0], 0.0)


def test_split_rule_basin_keeps_the_group_on_a_big_untouched_peak():
    # at t = 0 the default rule compares one cell against a whole neighbouring
    # basin and splits at once; the basin rule compares group rates and stays
    # strong neighbours either side of the biggest peak
    grid, s = world([(212.5, 162.5, "W", 2500), (512.5, 162.5, "A", 4000),
                     (812.5, 162.5, "E", 2500)])
    real_timing = dict(search_time=0.07, move_time_per_cell=0.028, trip_time=33.0)  # UW, 25 m
    for rule, expect_split in (("cell", True), ("basin", False)):
        m = make(PeaksFixedGroup, grid, s, n=10, split_rule=rule, **real_timing)
        m._update_groups(0.0)
        assert any(e["event"] == "split" for e in m.group_log) == expect_split, rule


def test_split_rule_basin_still_splits_once_the_basin_is_empty():
    grid, s = world(ROW)
    m = make(PeaksFixedGroup, grid, s, n=10, split_rule="basin")
    claim_basin(m, peak_named(s, "A"))
    m._update_groups(1.0)
    assert [e["event"] for e in m.group_log] == ["form", "split"]


def test_peaks_model_can_be_imported_first():
    import subprocess
    import sys
    out = subprocess.run([sys.executable, "-c", "import peaks_model, continuous_model as c; "
                          "print(sorted(k for k in c.CONTINUOUS_ALGORITHMS if k.startswith('peaks')))"],
                         capture_output=True, text=True, cwd=__import__("os").path.dirname(__file__))
    assert out.returncode == 0 and "peaks-mass-group" in out.stdout
