import pytest

from brute_force import FindThenFetchModel
from continuous_model import ContinuousFindThenFetch, ContinuousModel
from drone_model import DroneSearchModel
from scenario import Scenario

CONFIGS = [(1, 5, 3), (5, 20, 15), (10, 8, 40), (20, 20, 80), (6, 12, 40)]


@pytest.mark.parametrize("discrete,continuous", [(DroneSearchModel, ContinuousModel),
                                                 (FindThenFetchModel, ContinuousFindThenFetch)])
@pytest.mark.parametrize("n,k,p", CONFIGS)
def test_reduces_to_discrete_model_when_trip_equals_search(discrete, continuous, n, k, p):
    for seed in range(15):
        expected = discrete(n, k, p, seed=seed).run()
        got = continuous(n, k, p, spot_at_end=True, seed=seed).run()
        assert got == pytest.approx(expected)


@pytest.mark.parametrize("model_cls", [ContinuousModel, ContinuousFindThenFetch])
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
    assert sc.k == 17
    assert sc.search_time_min == pytest.approx(10_000 / 150 / 60)
    assert sc.trip_time_min == pytest.approx(2 * 13_800 / 15 / 60 + 5)
    assert sc.people == 250
    model = ContinuousModel(20, n_people=sc.people, seed=0, **sc.model_kwargs())
    assert model.run() > 0
