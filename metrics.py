"""
Metrics for a finished continuous-model run.

    summarize(model)  -> flat dict of numbers (one row of a benchmark table)
    curves(model)     -> sorted found / supplied times, for "people vs time" plots

Times are in the model's time unit (minutes for the UW scenario).
"""

from __future__ import annotations

import math
import statistics

from continuous_model import STATUSES, ContinuousModel

# completion: the time COMPLETION_FRAC of the expected head count has been found
COMPLETION_FRAC = 0.8

METRICS = {
    "completion": f"time {COMPLETION_FRAC:.0%} of the expected number of people are found",
    "all_served": "time the last person is supplied",
    "n_people": "people in this run",
    "all_found": "time the last person is found",
    "mean_wait": "mean time from found to supplied",
    "p90_wait": "90th percentile of found-to-supplied",
    "max_wait": "longest found-to-supplied",
    "mean_supplied_at": "mean time from the start until a person is supplied",
    "trips": "depot round trips started",
    "unused_kits": "kits on drones or on their way at the end (fetched but never needed)",
    **{f"share_{s}": f"share of drone time spent {s}" for s in STATUSES},
}


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def completion_time(model: ContinuousModel, frac: float = COMPLETION_FRAC) -> float:
    """Time the ceil(frac * expected_people)-th person is found (the last find if fewer)."""
    found = sorted(p.found_step for p in model.people if p.found_step is not None)
    if not found:
        return 0.0
    target = max(1, math.ceil(frac * model.expected_people - 1e-9))
    return found[min(target, len(found)) - 1]


def summarize(model: ContinuousModel) -> dict:
    if not model.done:
        raise ValueError("run the model to completion first")
    people = model.people
    waits = [p.supplied_step - p.found_step for p in people]
    total = sum(sum(d.time_in.values()) for d in model.drones) or 1.0
    row = {
        "completion": completion_time(model),
        "all_served": model.completion_time,
        "n_people": len(people),
        "all_found": max((p.found_step for p in people), default=0.0),
        "mean_wait": statistics.mean(waits) if waits else 0.0,
        "p90_wait": _percentile(waits, 0.9),
        "max_wait": max(waits, default=0.0),
        "mean_supplied_at": statistics.mean(p.supplied_step for p in people) if people else 0.0,
        "trips": sum(d.trips for d in model.drones),
        "unused_kits": sum(d.has_supplies or d.status in ("exit", "return")
                           for d in model.drones),
    }
    for status in STATUSES:
        row[f"share_{status}"] = sum(d.time_in.get(status, 0.0) for d in model.drones) / total
    row.update(model.extra_metrics())
    return row


def curves(model: ContinuousModel) -> dict:
    return {
        "found": sorted(p.found_step for p in model.people if p.found_step is not None),
        "supplied": sorted(p.supplied_step for p in model.people if p.supplied_step is not None),
    }
