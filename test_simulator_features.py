import math

import pytest

from benchmark import BenchmarkConfig, mean_ci, run_benchmark, summarize_rows
from continuous_model import (CONTINUOUS_ALGORITHMS, ContinuousModel, SimulationInvariantError)
from metrics import STATUSES, curves, summarize
from scenario import Scenario

SC = Scenario()


def uw(cls, n=12, p=40, seed=0, **kw):
    return cls(n, n_people=p, seed=seed, **SC.model_kwargs(), **kw)


@pytest.mark.parametrize("key", list(CONTINUOUS_ALGORITHMS))
@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("depot", ["redmond", "bothell"])
def test_built_in_checks_pass_on_the_uw_scenario(key, seed, depot):
    sc = Scenario(depot=depot)
    model = CONTINUOUS_ALGORITHMS[key](10, n_people=50, seed=seed, check=True,
                                       **sc.model_kwargs())
    model.run()
    assert model.n_supplied == 50


def test_checks_catch_a_broken_state():
    model = uw(ContinuousModel, check=True)
    model.run_until(5.0)
    model.drones[0].has_supplies = not model.drones[0].has_supplies  # conjure / lose a kit
    with pytest.raises(SimulationInvariantError, match="kits not conserved"):
        model.run_until(10.0)


def test_every_algorithm_sees_the_same_people_and_spot_moments():
    models = [uw(cls, seed=3) for cls in CONTINUOUS_ALGORITHMS.values()]
    ref = [(p.xy, p.spot_frac) for p in models[0].people]
    for m in models[1:]:
        assert [(p.xy, p.spot_frac) for p in m.people] == ref
    # people sit inside their cell
    for p in models[0].people:
        assert p.cell == (int(p.xy[0]), int(p.xy[1]))


def test_people_are_spotted_at_their_fraction_of_the_cell_search():
    model = uw(ContinuousModel, record=True, seed=2)
    model.run()
    for p in model.people:
        start, end = model.cell_windows[p.cell]
        assert p.found_step == pytest.approx(start + p.spot_frac * (end - start))


def test_search_moves_beyond_a_neighbouring_cell_cost_flight_time():
    model = uw(ContinuousModel, record=True, seed=1)
    model.run()
    transits = [s for segs in model.segments.values() for s in segs if s[2] == "transit"]
    assert transits, "expected some long search moves"
    for t0, t1, _, _, x0, y0, x1, y1 in transits:
        dist = math.hypot(x1 - x0, y1 - y0)
        assert t1 - t0 == pytest.approx((dist - 1.0) * model.move_time)


def test_depot_trips_leave_and_return_through_the_depot_edge():
    model = uw(ContinuousModel, record=True, seed=0)
    model.run()
    dp = SC.depot_point
    for segs in model.segments.values():
        for i, (t0, t1, status, loaded, x0, y0, x1, y1) in enumerate(segs):
            if status == "exit":
                assert (x1, y1) == pytest.approx(dp)
                assert not loaded
                assert segs[i + 1][2] == "return"
            if status == "return":
                assert t1 - t0 == pytest.approx(model.T)
                assert (x0, y0) == pytest.approx(dp)
    # everyone starts at the depot edge, loaded
    starts = [segs[0] for segs in model.segments.values()]
    assert all((s[4], s[5]) == pytest.approx(dp) and s[3] for s in starts)


@pytest.mark.parametrize("key", list(CONTINUOUS_ALGORITHMS))
def test_metrics_are_consistent(key):
    model = uw(CONTINUOUS_ALGORITHMS[key], seed=5)
    model.run()
    m = summarize(model)
    c = curves(model)
    assert m["completion"] == pytest.approx(max(c["supplied"]))
    assert m["all_found"] == pytest.approx(max(c["found"]))
    assert m["mean_wait"] <= m["p90_wait"] <= m["max_wait"]
    assert sum(m[f"share_{s}"] for s in STATUSES) == pytest.approx(1.0)
    # every drone's time adds up to the run length
    for d in model.drones:
        assert sum(d.time_in.values()) == pytest.approx(model.completion_time)
    assert m["trips"] == sum(dec.r for dec in model.decisions)


def test_mean_ci_matches_hand_calculation():
    m, ci = mean_ci([1.0, 2.0, 3.0, 4.0])
    assert m == 2.5
    assert ci == pytest.approx(3.182 * math.sqrt(5 / 3) / 2)


def test_benchmark_pairs_algorithms_on_the_same_seeds():
    cfg = BenchmarkConfig(drones=[8], people=[20], runs=3, algos=["expected", "brute"],
                          baseline="brute", check=True)
    rows = run_benchmark(cfg, workers=1)
    assert len(rows) == 6
    summary = {s["algo"]: s for s in summarize_rows(rows, "brute")}
    by = {(r["algo"], r["seed"]): r for r in rows}
    diffs = [by[("expected", s)]["completion"] - by[("brute", s)]["completion"] for s in range(3)]
    assert summary["expected"]["completion_diff"] == pytest.approx(sum(diffs) / 3)
    assert summary["expected"]["faster"] + summary["expected"]["slower"] <= 3
    assert "completion_diff" not in summary["brute"]


def test_dashboard_api_returns_replay_data():
    import server
    out = server.run_scenario({"drones": 6, "people": 15, "algos": ["expected", "brute"],
                               "check": True})
    assert set(out["results"]) == {"expected", "brute"}
    rep = out["results"]["expected"]["replay"]
    assert len(rep["people"]) == 15
    assert len(rep["cells"]) == out["scenario"]["k"] ** 2
    for x, y, found, sent, supplied in rep["people"]:
        assert found <= sent <= supplied
    with pytest.raises(ValueError):
        server.run_scenario({"drones": 0, "people": 10})
