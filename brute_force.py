"""
Find-then-fetch baseline: no forecasting.

Every drone searches. Supplies are fetched only for people who have already
been found: each step, one empty drone is sent back per person still waiting
for supplies. Expected future finds (lambda) are ignored entirely.

Everything else (grid, people placement, movement, instant deliveries,
one-to-one drone/person matching) is inherited from DroneSearchModel, so with
the same seed both models see the same people layout.
"""

from __future__ import annotations

from drone_model import DroneSearchModel


class FindThenFetchModel(DroneSearchModel):
    name = "find-then-fetch"
    rule = "r = min(empty drones, U)"
    surplus_rule = False

    def choose_num_returning(self) -> tuple[int, int]:
        n = len(self.drones)
        empty = n - sum(d.has_supplies for d in self.drones)
        r = min(empty, len(self.waiting))
        return r, min(n - r, len(self.unsearched))


ALGORITHMS: dict[str, type[DroneSearchModel]] = {
    "expected": DroneSearchModel,
    "brute": FindThenFetchModel,
}
