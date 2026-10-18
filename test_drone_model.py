import pytest

from drone_model import DroneSearchModel


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("n_drones,k,p", [(1, 5, 3), (4, 10, 10), (10, 8, 40), (3, 6, 0)])
def test_everyone_found_and_supplied_once(seed, n_drones, k, p):
    model = DroneSearchModel(n_drones, k, p, seed=seed)
    model.run()
    assert all(person.supplied_by is not None for person in model.people)
    assert all(person.supplied_step >= person.found_step for person in model.people)
    # one delivery per drone per step
    per_step = {}
    for person in model.people:
        key = (person.supplied_step, person.supplied_by)
        assert key not in per_step
        per_step[key] = person.id


def test_returns_only_empty_drones():
    model = DroneSearchModel(5, 10, 20, seed=0)
    while not model.done:
        empty = sum(not d.has_supplies for d in model.drones)
        rec = model.step()
        assert rec.returning <= empty
        assert rec.searching + rec.returning + rec.idle == 5


def test_lambda_is_density_times_cell_area():
    model = DroneSearchModel(2, 4, 8, cell_area=2.0, seed=0)
    assert model.expected_finds_per_drone() == pytest.approx(2.0 * 8 / (16 * 2.0))


def test_no_supplies_start_sends_drones_back_when_people_are_dense():
    # lambda = 1 per drone: with 4 empty drones, balance is r = S -> r = 2
    model = DroneSearchModel(4, 4, 16, start_with_supplies=False, seed=0)
    r, s = model.choose_num_returning()
    assert (r, s) == (2, 2)


def test_surplus_drones_loaded_search_empty_return():
    model = DroneSearchModel(6, 4, 5, seed=0)
    model.unsearched = set(list(model.unsearched)[:3])  # 6 drones, 3 cells
    for d in model.drones[:2]:
        d.has_supplies = True
    for d in model.drones[2:]:
        d.has_supplies = False
    # 2 loaded + 1 empty search the 3 cells, the other 3 empty drones return
    assert model.choose_num_returning() == (3, 3)
    for d in model.drones[:4]:
        d.has_supplies = True
    # 4 loaded > 3 cells: 3 loaded search, 1 loaded idles, both empty return
    assert model.choose_num_returning() == (2, 3)


from brute_force import FindThenFetchModel


@pytest.mark.parametrize("seed", range(20))
def test_brute_force_completes_and_supplies_everyone_once(seed):
    model = FindThenFetchModel(4, 10, 15, seed=seed)
    model.run()
    assert all(person.supplied_by is not None for person in model.people)
    pairs = {(p.supplied_step, p.supplied_by) for p in model.people}
    assert len(pairs) == len(model.people)


def test_same_seed_gives_same_people_layout():
    a = DroneSearchModel(4, 10, 15, seed=7)
    b = FindThenFetchModel(4, 10, 15, seed=7)
    assert [p.cell for p in a.people] == [p.cell for p in b.people]


def test_brute_force_only_fetches_for_people_already_found():
    model = FindThenFetchModel(4, 4, 16, start_with_supplies=False, seed=0)
    assert model.choose_num_returning() == (0, 4)  # nobody found yet -> all search
