import math

import pytest

from brute_force import FindThenFetchModel
from continuous_model import ContinuousFindThenFetch, ContinuousKeepSearching, ContinuousModel
from drone_model import DroneSearchModel
from scenario import Scenario

CONFIGS = [(1, 5, 3), (5, 20, 15), (10, 8, 40), (20, 20, 80), (6, 12, 40)]


@pytest.mark.parametrize("discrete,continuous", [(DroneSearchModel, ContinuousModel),
                                                 (DroneSearchModel, ContinuousKeepSearching),
                                                 (FindThenFetchModel, ContinuousFindThenFetch)])
@pytest.mark.parametrize("n,k,p", CONFIGS)
def test_reduces_to_discrete_model_when_trip_equals_search(discrete, continuous, n, k, p):
    for seed in range(15):
        expected = discrete(n, k, p, seed=seed).run()
        got = continuous(n, k, p, spot_at_end=True, seed=seed).run()
        assert got == pytest.approx(expected)


@pytest.mark.parametrize("model_cls", [ContinuousModel, ContinuousKeepSearching,
                                       ContinuousFindThenFetch])
@pytest.mark.parametrize("seed", range(10))
def test_everyone_supplied_once_with_long_trips(model_cls, seed):
    model = model_cls(8, 10, 30, search_time=1.0, trip_time=12.5, seed=seed)
    model.run()
    assert all(p.supplied_by is not None for p in model.people)
    for p in model.people:
        assert p.found_step <= p.supplied_step <= model.completion_time
    # a drone delivers at most one kit per trip, plus the one it starts with
    trips = {d.id: 0 for d in model.drones}
    for p in model.people:
        trips[p.supplied_by] += 1
    returns = sum(dec.r for dec in model.decisions)
    assert sum(trips.values()) <= returns + len(model.drones)


def test_people_spotted_during_their_cells_search():
    model = ContinuousModel(3, 6, 20, search_time=2.0, trip_time=7.0, seed=4)
    spots = {}
    orig = model._push

    def push(t, kind, obj):
        if kind == "spot":
            spots[obj.id] = t
        orig(t, kind, obj)

    model._push = push
    starts = {}
    model.run_until(0.0)
    while not model.done:
        for d in model.drones:
            if d.status == "search":
                for person in model.people_in_cell.get(d.cell, []):
                    starts.setdefault(person.id, d.started)
        model.run_until(model.time + 0.5)
    for pid, t in spots.items():
        assert starts[pid] <= t <= starts[pid] + 2.0 + 1e-9


def test_uw_scenario_numbers():
    sc = Scenario()
    assert sc.k == 67
    assert sc.k * sc.cell_side_m == pytest.approx(math.sqrt(703 * 4046.86))
    assert sc.search_time_min == pytest.approx(sc.cell_side_m ** 2 / 150 / 60)
    assert sc.trip_time_min == pytest.approx(2 * (13_800 - sc.edge_offset_m) / 15 / 60 + 5)
    assert sc.people == 250
    # the depot entry point is on the east edge for Redmond
    assert sc.depot_point[0] == pytest.approx(sc.k, abs=1e-5)
    model = ContinuousModel(20, n_people=sc.people, seed=0, check=True, **sc.model_kwargs())
    assert model.run() > 0


@pytest.mark.parametrize("cell", [100.0, 50.0, 25.0, 10.0])
def test_cell_size_keeps_campus_area_and_total_search_work(cell):
    sc = Scenario(cell_side_target_m=cell)
    assert sc.k ** 2 * sc.cell_area_km2 == pytest.approx(703 * 4046.86 / 1e6)
    assert sc.k ** 2 * sc.search_time_min == pytest.approx(703 * 4046.86 / 150 / 60)


def test_expected_value_always_balances():
    # even when the fleet could finish searching within one trip
    model = ContinuousModel(4, 4, 10, search_time=1.0, trip_time=10.0, seed=0)
    model.run()
    assert {d.mode for d in model.decisions} <= {"balance", "surplus"}
    assert model.decisions[0].mode == "balance"


def test_keep_searching_enters_search_first_when_fleet_can_finish_within_one_trip():
    # 4 drones, 16 cells, tau 1 -> 4 time units of search left < T = 10
    model = ContinuousKeepSearching(4, 4, 10, search_time=1.0, trip_time=10.0, seed=0)
    model.run_until(0.0)
    assert model.decisions[0].mode == "search-first"
    # long search compared with a trip -> balance
    model = ContinuousKeepSearching(4, 20, 10, search_time=1.0, trip_time=10.0, seed=0)
    model.run_until(0.0)
    assert model.decisions[0].mode == "balance"


def test_keep_searching_sends_nobody_back_while_cells_remain():
    model = ContinuousKeepSearching(4, 6, 30, search_time=1.0, trip_time=10.0, seed=1)
    model.run()
    for dec in model.decisions:
        if dec.mode == "search-first":
            assert dec.r == 0


def _setup_assignment(drone_cells, person_cells):
    """A model with free loaded drones and found people at the given cells."""
    model = ContinuousModel(len(drone_cells), 80, len(person_cells), move_time_per_cell=1.0,
                            seed=0)
    for drone, cell in zip(model.drones, drone_cells):
        drone.pos, drone.loc, drone.has_supplies, drone.status = cell, cell, True, "idle"
    for person, cell in zip(model.people, person_cells):
        person.cell, person.xy, person.found_step = cell, cell, 0.0
        model.waiting.append(person)
    model.n_unfound = 0
    return model


def test_worst_served_person_gets_their_nearest_drone_first():
    # A at x=0, B at x=4. d0 at x=1 is nearest to both; d1 at x=-2.
    # B's nearest drone (d0, 3 away) is further than A's (d0, 1 away), so B is
    # served first and takes d0; A then gets d1. Nearest-first would do the opposite.
    model = _setup_assignment([(1, 0), (-2, 0)], [(0, 0), (4, 0)])
    model._assign(0.0)
    a, b = model.people
    assert model.drones[0].target is b
    assert model.drones[1].target is a
    assert model.drones[0].until == pytest.approx(3.0)
    assert model.drones[1].until == pytest.approx(2.0)


def test_pool_is_u_plus_one_closest_and_leftovers_stay_free():
    # 2 people -> pool of 3 closest drones; the 4th is never considered,
    # and the unused pool drone stays idle to make its own decision.
    model = _setup_assignment([(1, 0), (5, 0), (9, 0), (30, 0)], [(0, 0), (6, 0)])
    model._assign(0.0)
    assigned = [d.id for d in model.drones if d.status == "deliver"]
    assert sorted(assigned) == [0, 1]
    assert model.drones[2].status == "idle" and model.drones[3].status == "idle"
    assert not model.waiting


def test_fewer_drones_than_people_leaves_the_rest_waiting():
    model = _setup_assignment([(0, 0)], [(1, 0), (10, 0)])
    model._assign(0.0)
    # the person whose nearest drone is furthest (x=10) is served first
    assert model.drones[0].target is model.people[1]
    assert list(model.waiting) == [model.people[0]]


@pytest.mark.parametrize("model_cls", [ContinuousModel, ContinuousKeepSearching,
                                       ContinuousFindThenFetch])
@pytest.mark.parametrize("seed", range(10))
def test_moving_to_serve_takes_travel_time(model_cls, seed):
    model = model_cls(6, 12, 25, search_time=1.0, trip_time=15.0, move_time_per_cell=0.3,
                      seed=seed)
    flights = []
    orig = model._push

    def push(t, kind, obj):
        if kind == "delivered":
            flights.append((obj.started, t, obj.origin, obj.target.xy))
        orig(t, kind, obj)

    model._push = push
    model.run()
    assert all(p.supplied_by is not None for p in model.people)
    assert len(flights) == len(model.people)
    for start, end, origin, target in flights:
        dist = math.hypot(origin[0] - target[0], origin[1] - target[1])
        assert end - start == pytest.approx(dist * 0.3)
    for p in model.people:
        assert p.supplied_step >= p.found_step
