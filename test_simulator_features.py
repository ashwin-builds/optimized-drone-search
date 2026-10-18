import math

import pytest

from benchmark import BenchmarkConfig, mean_ci, run_benchmark, summarize_rows
from continuous_model import (CONTINUOUS_ALGORITHMS, ContinuousModel, SimulationInvariantError)
from metrics import STATUSES, curves, summarize
from scenario import Scenario, draw_people

SC = Scenario(campus="square")   # the real map has its own tests in test_campus_map.py


def uw(cls, n=12, p=40, seed=0, **kw):
    return cls(n, n_people=p, seed=seed, **(SC.model_kwargs() | kw))


@pytest.mark.parametrize("key", list(CONTINUOUS_ALGORITHMS))
@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("depot", ["redmond", "bothell"])
def test_built_in_checks_pass_on_the_uw_scenario(key, seed, depot):
    sc = Scenario(depot=depot, campus="square")
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
    # the old rule: no searching on the way, any distance beyond one cell is flown
    model = uw(ContinuousModel, record=True, seed=1, search_en_route=False)
    model.run()
    transits = [s for segs in model.segments.values() for s in segs if s[2] == "transit"]
    assert transits, "expected some long search moves"
    for t0, t1, _, _, x0, y0, x1, y1, _ in transits:
        dist = math.hypot(x1 - x0, y1 - y0)
        assert t1 - t0 == pytest.approx((dist - 1.0) * model.move_time)


def test_depot_trips_leave_and_return_through_the_depot_edge():
    model = uw(ContinuousModel, record=True, seed=0)
    model.run()
    dp = SC.depot_point
    for segs in model.segments.values():
        for i, (t0, t1, status, loaded, x0, y0, x1, y1, _) in enumerate(segs):
            if status == "exit":
                assert (x1, y1) == pytest.approx(dp)
                assert not loaded
                if i + 1 < len(segs):  # the run can end mid-flight
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
    assert m["all_served"] == pytest.approx(max(c["supplied"]))
    assert m["all_found"] == pytest.approx(max(c["found"]))
    # completion: the ceil(0.8 * expected)-th find (expected = the true count here)
    assert m["completion"] == pytest.approx(c["found"][math.ceil(0.8 * len(model.people)) - 1])
    assert m["completion"] <= m["all_found"] <= m["all_served"]
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
    # 15 is the expected head count; the run draws its own (seed 0), the same for every algorithm
    assert out["expected_people"] == 15 and out["people"] == draw_people(15, 0)
    assert len(rep["people"]) == out["people"] == len(out["results"]["brute"]["replay"]["people"])
    sc = Scenario()
    assert len(rep["cells"]) == sc.n_search_cells
    assert (out["scenario"]["map"] is not None) == sc.uses_map
    for x, y, found, sent, supplied in rep["people"]:
        assert found <= sent <= supplied
    with pytest.raises(ValueError):
        server.run_scenario({"drones": 0, "people": 10})


# ------------------------------------------------- search while flying (phase A)

from continuous_model import ContinuousGreedyDensity, ContinuousExpectedUniform


def _line_model(**kw):
    """A 12x3 strip, move 0.5 per cell, tau 1; one drone starting at the west end."""
    args = dict(search_time=1.0, trip_time=50.0, move_time_per_cell=0.5, entry_cell=(0, 1),
                record=True, check=True, seed=0)
    return ContinuousModel(1, 12, 0, **(args | kw))


def test_crossed_cells_follow_the_line_and_step_diagonally_through_corners():
    m = _line_model()
    cells = [c for c, _, _ in m._crossed((0.5, 1.5), (5.5, 1.5))]
    assert cells == [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]
    diag = [c for c, _, _ in m._crossed((0.5, 0.5), (3.5, 3.5))]
    assert diag == [(0, 0), (1, 1), (2, 2), (3, 3)]       # exact corners: no side cells
    # slope 3.2: crosses x = 1 at y = 2.1, so it visits three cells in column 0
    steep = [c for c, _, _ in m._crossed((0.5, 0.5), (1.5, 3.7))]
    assert steep == [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3)]


def test_flying_to_a_distant_cell_searches_every_unclaimed_cell_on_the_way():
    m = _line_model()
    m.unsearched = {(x, 1) for x in range(12)}          # only the middle row is searchable
    m._alive[:] = [c in m.unsearched for c in m._cells]
    m._version += 1
    d = m.drones[0]
    d.loc, d.pos = (0.5, 1.5), (0, 1)
    m._start_search(d, (5, 1), 0.0)
    legs = [(k, c, t0, t1) for k, c, t0, t1, _, _ in d.route]
    assert legs == [("search", (x, 1), x - 1.0, float(x)) for x in range(1, 6)]
    assert all((x, 1) not in m.unsearched for x in range(1, 6))
    # replay log and time accounting follow the legs (no people, so the model
    # counts as finished; process the scheduled events directly)
    import heapq
    while m._events and m._events[0][0] <= 6.0:
        t, _, kind, obj = heapq.heappop(m._events)
        m._handle(kind, obj, t)
    assert m.searched >= {(x, 1) for x in range(1, 6)}
    assert d.status == "idle" and sum(d.time_in.values()) == pytest.approx(d.since) == 5.0
    segs = m.segments[0]
    assert [s[2] for s in segs[1:]] == ["search"] * 5 + ["idle"]
    assert all(a[1] == pytest.approx(b[0]) for a, b in zip(segs[1:], segs[2:]))   # contiguous


def test_cells_already_searched_are_flown_over_at_move_time():
    m = _line_model()
    m.unsearched = {(x, 1) for x in range(12)} - {(2, 1), (3, 1)}
    m._alive[:] = [c in m.unsearched for c in m._cells]
    m._version += 1
    d = m.drones[0]
    d.loc, d.pos = (0.5, 1.5), (0, 1)
    m._start_search(d, (5, 1), 0.0)
    legs = [(k, c, round(t0, 9), round(t1, 9)) for k, c, t0, t1, _, _ in d.route]
    assert legs == [("search", (1, 1), 0.0, 1.0),
                    ("transit", (5, 1), 1.0, 2.0),          # two cells x 0.5
                    ("search", (4, 1), 2.0, 3.0),
                    ("search", (5, 1), 3.0, 4.0)]


def test_without_en_route_search_the_old_flight_rule_applies():
    m = _line_model(search_en_route=False)
    d = m.drones[0]
    d.loc, d.pos = (0.5, 1.5), (0, 1)
    m._start_search(d, (5, 1), 0.0)
    assert [(k, c) for k, c, *_ in d.route] == [("transit", (5, 1)), ("search", (5, 1))]
    assert d.route[0][3] == pytest.approx((5 - 1) * 0.5)


def test_people_in_en_route_cells_are_spotted_during_that_cells_search():
    sc = Scenario(campus="square")
    m = ContinuousModel(6, n_people=60, seed=3, record=True, check=True, **sc.model_kwargs())
    m.run()
    for p in m.people:
        t0, t1 = m.cell_windows[p.cell]
        assert t0 <= p.found_step <= t1 + 1e-9


def test_density_expected_finds_uses_the_best_cells():
    # 10 cells in a row; cell 0 holds half the prior, the rest share the other half
    prior = {(x, 0): (0.5 if x == 0 else 0.5 / 9) for x in range(10)}
    m = ContinuousModel(2, 10, 0, search_time=1.0, trip_time=1.0, density=prior,
                        search_cells=set(prior), seed=0)
    m.n_unfound = 20
    # one drone sweeps T / tau = 1 cell in the horizon: the best one, half the mass
    assert m.expected_finds(1, m.remaining_area()) == pytest.approx(20 * 0.5)
    assert m.expected_finds(3, m.remaining_area()) == pytest.approx(20 * (0.5 + 2 * 0.5 / 9))
    uniform = ContinuousExpectedUniform(2, 10, 0, search_time=1.0, trip_time=1.0, density=prior,
                                        search_cells=set(prior), seed=0)
    uniform.n_unfound = 20
    assert uniform.expected_finds(1, uniform.remaining_area()) == pytest.approx(20 * 0.1)


def test_greedy_takes_the_best_probability_per_unit_time():
    prior = {(x, 0): 0.0 for x in range(10)}
    prior.update({(2, 0): 0.3, (9, 0): 0.7})
    m = ContinuousGreedyDensity(1, 10, 0, search_time=1.0, trip_time=50.0, density=prior,
                                search_cells=set(prior), move_time_per_cell=0.1,
                                entry_cell=(0, 0), seed=0)
    d = m.drones[0]
    # rates: 0.3 / (1.5 + 1) = 0.12 vs 0.7 / (9.0 * 0.1 + 1) = 0.37
    assert m._pick_cell(d) == (9, 0)
    m.move_time = 1.0      # far travel now costs too much: 0.3/3 = 0.1 > 0.7/10
    assert m._pick_cell(d) == (2, 0)


def test_people_density_samples_the_prior():
    sc = Scenario(campus="square", people="density")
    kw = sc.model_kwargs()
    m = ContinuousModel(5, n_people=3000, seed=1, **kw)
    # a uniform prior: every quadrant gets about a quarter
    k = sc.k
    q = sum(p.cell[0] < k // 2 and p.cell[1] < k // 2 for p in m.people) / 3000
    assert q == pytest.approx(0.25, abs=0.03)


def test_floor_and_density_placements_differ_on_the_real_map():
    pytest.importorskip("scipy")
    import campus_map
    if not campus_map.DATA_PATH.exists():
        pytest.skip("UW map not downloaded")
    floor = ContinuousModel(5, n_people=50, seed=2, **Scenario(people="floor").model_kwargs())
    dens = ContinuousModel(5, n_people=50, seed=2, **Scenario(people="density").model_kwargs())
    assert [p.xy for p in floor.people] != [p.xy for p in dens.people]
    assert floor.density == dens.density      # same prior either way


def test_benchmark_csv_keeps_columns_only_some_models_have(tmp_path):
    from benchmark import write_csv
    rows = [{"algo": "expected", "completion": 1.0},
            {"algo": "peaks-mass-group", "completion": 2.0, "splits": 3}]
    path = tmp_path / "rows.csv"
    write_csv(rows, str(path))
    lines = path.read_text().splitlines()
    assert lines[0] == "algo,completion,splits" and lines[1].endswith(",")


# ---------------------------------------- random head count, 80% completion

def test_people_draw_is_seeded_normal_with_a_floor():
    from metrics import COMPLETION_FRAC
    from scenario import PEOPLE_FLOOR_FRAC, PEOPLE_SD_FRAC

    draws = [draw_people(250, s) for s in range(4000)]
    assert draws == [draw_people(250, s) for s in range(4000)]
    assert min(draws) >= COMPLETION_FRAC * 250
    assert round(PEOPLE_FLOOR_FRAC * 250) in draws          # low draws are raised to 85%
    kept = [d for d in draws if d != round(PEOPLE_FLOOR_FRAC * 250)]
    assert abs(sum(draws) / len(draws) - 250) < 2
    sd = (sum((d - 250) ** 2 for d in kept) / len(kept)) ** 0.5
    assert 0.8 * PEOPLE_SD_FRAC * 250 < sd < 1.1 * PEOPLE_SD_FRAC * 250


def test_planner_uses_the_expected_count_not_the_true_one():
    a = uw(ContinuousModel, p=40)
    b = uw(ContinuousModel, p=40, expected_people=80)
    assert a.believed_unfound == 40 and b.believed_unfound == 80
    assert b.expected_finds(10, a.remaining_area()) == pytest.approx(
        2 * a.expected_finds(10, a.remaining_area()))
    c = uw(ContinuousModel, p=40, expected_people=30)
    for person in c.people[:35]:            # found more than expected: nothing left to plan for
        person.found_step = 0.0
    c.n_unfound = 5
    assert c.believed_unfound == 0 and c.expected_finds(10, c.remaining_area()) == 0


def test_completion_counts_finds_against_the_expected_count():
    model = uw(ContinuousModel, p=40, expected_people=30, seed=2)
    model.run()
    found = curves(model)["found"]
    assert summarize(model)["completion"] == pytest.approx(found[24 - 1])   # ceil(0.8 * 30)
    assert summarize(model)["n_people"] == 40


def test_benchmark_runs_share_each_seeds_head_count():
    cfg = BenchmarkConfig(drones=[6], people=[30], runs=3, algos=["expected", "brute"],
                          baseline="brute", campus="square")
    rows = run_benchmark(cfg, workers=1)
    for seed in range(3):
        counts = {r["n_people"] for r in rows if r["seed"] == seed}
        assert counts == {draw_people(30, seed)}
