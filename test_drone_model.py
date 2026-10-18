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
