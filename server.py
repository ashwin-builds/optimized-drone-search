"""
Local dashboard for the drone search simulator.

    python server.py                 # then open http://localhost:8000
    python server.py --port 8080

Endpoints (JSON):
    GET  /api/options              algorithms, depots, defaults
    POST /api/run                  run one scenario for several algorithms (with replay data)
    POST /api/benchmark            start a benchmark job -> {"job": id}
    POST /api/demo                 start the depot comparison behind the demo -> {"job": id}
    GET  /api/benchmark/<id>       progress, then summary and per-run rows
    GET  /api/benchmark/<id>.csv   per-run rows as CSV
Everything else is served from ./web.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import threading
import uuid
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from benchmark import BenchmarkConfig, run_benchmark, summarize_rows
from continuous_model import CONTINUOUS_ALGORITHMS, STATUSES
from metrics import COMPLETION_FRAC, METRICS, curves, summarize
from campus_map import DATA_PATH
from scenario import (CAMPUS_CHOICES, DEPOT_CHOICES, DEPOT_DISTANCE_KM, DEPOT_NAMES, STAGING_LONLAT,
                      Scenario, draw_people)

WEB_DIR = Path(__file__).parent / "web"
MAX_RUN_PEOPLE = 5000
MAX_BENCH_JOBS = 20000
# the demo: UW map, supplies from Redmond vs staged on campus at Rainier Vista
DEMO = {"drones": 20, "people": 250, "seed": 0, "campus": "auto", "cell_side": 25.0,
        "depot": "rainier_vista", "compare_depots": ["redmond", "rainier_vista"],
        "algos": ["expected", "peaks-best", "brute"], "baseline": "brute",
        "runs": 8}
JOBS: dict[str, dict] = {}


def _r(x: float, nd: int = 3) -> float:
    return round(float(x), nd)


def _scenario(params: dict) -> Scenario:
    campus = params.get("campus", "auto")
    if campus not in CAMPUS_CHOICES:
        raise ValueError(f"campus must be one of {CAMPUS_CHOICES}")
    return Scenario(depot=params.get("depot", "redmond"), campus=campus,
                    cell_side_target_m=float(params.get("cell_side", 25.0)))


def scenario_info(sc: Scenario) -> dict:
    info = {
        "k": sc.k, "cell_side_m": sc.cell_side_m,
        "area_km2": sc.n_search_cells * sc.cell_area_km2,
        "depot": sc.depot, "distance_km": round(sc.edge_to_depot_km, 1), "depot_point": sc.depot_point,
        "depot_name": sc.depot_name, "depot_on_campus": sc.depot_on_campus,
        "search_time_min": sc.search_time_min, "trip_time_min": sc.trip_time_min,
        "move_time_min": sc.move_time_min, "map": None,
    }
    if sc.uses_map:
        cmap, grid = sc.map_grid
        mask = ["".join("1" if (x, y) in grid.search_cells else "0" for x in range(grid.k))
                for y in range(grid.k)]
        info["map"] = {"fetched": cmap.fetched, "source": cmap.source,
                       "n_buildings": len(cmap.buildings), "indoor_fraction": sc.indoor_fraction,
                       "mask": mask, **cmap.outline_cells(sc.cell_side_m)}
    return info


def run_scenario(params: dict) -> dict:
    sc = _scenario(params)
    n = int(params.get("drones", 20))
    p = int(params.get("people", 60))
    seed = int(params.get("seed", 0))
    algos = [a for a in params.get("algos", list(CONTINUOUS_ALGORITHMS))
             if a in CONTINUOUS_ALGORITHMS]
    if not (1 <= n <= 500) or not (0 <= p <= MAX_RUN_PEOPLE) or not algos:
        raise ValueError("drones must be 1-500, people 0-5000, and at least one algorithm")
    results = {}
    actual = draw_people(p, seed)   # p is the expected head count; this seed's draw
    for key in algos:
        cls = CONTINUOUS_ALGORITHMS[key]
        model = cls(n, n_people=actual, expected_people=p, seed=seed, record=True,
                    check=bool(params.get("check")), **sc.model_kwargs())
        model.run()
        # [x, y, found, drone sent, supplied]
        people = [[_r(p.xy[0]), _r(p.xy[1]), _r(p.found_step),
                   _r(model.assigned_at.get(p.id, p.supplied_step)), _r(p.supplied_step)]
                  for p in model.people]
        results[key] = {
            "name": cls.name,
            "rule": cls.rule,
            "metrics": summarize(model),
            "curves": {k: [_r(v) for v in vals] for k, vals in curves(model).items()},
            "replay": {
                "cells": [[x, y, _r(a), _r(b)] for (x, y), (a, b) in model.cell_windows.items()],
                "people": people,
                "segments": {d: [[_r(t0), _r(t1), s, int(ld), _r(x0), _r(y0), _r(x1), _r(y1),
                                  g]
                                 for t0, t1, s, ld, x0, y0, x1, y1, g in segs]
                             for d, segs in model.segments.items()},
                "decisions": [[_r(d.time), d.mode, d.H, d.F, d.U, d.S, d.r, _r(d.E, 2), d.n,
                               _r(d.R, 4)] for d in model.decisions],
            },
        }
    return {"scenario": scenario_info(sc), "drones": n, "people": actual, "expected_people": p,
            "completion_frac": COMPLETION_FRAC, "seed": seed, "results": results}


def depot_label(key: str) -> str:
    if key in DEPOT_DISTANCE_KM:
        return f"{DEPOT_NAMES[key]} ({DEPOT_DISTANCE_KM[key]} km)"
    where = "EOC, just off campus" if key == "uw_tower" else "on-campus staging"
    return f"{DEPOT_NAMES[key]} ({where})" if key in STAGING_LONLAT else key


def start_job(work_fn, total: int, config: dict) -> str:
    """Run work_fn(progress) in a thread; it returns (rows, summary)."""
    job_id = uuid.uuid4().hex[:12]
    job = {"done": 0, "total": total, "finished": False, "error": None, "config": config}
    JOBS[job_id] = job

    def progress(done, _total):
        job["done"] = done

    def work():
        try:
            job["rows"], job["summary"] = work_fn(progress)
        except Exception as exc:  # reported to the page
            job["error"] = f"{type(exc).__name__}: {exc}"
        job["finished"] = True

    threading.Thread(target=work, daemon=True).start()
    return job_id


def start_demo(params: dict) -> str:
    """Same drones, people and seeds for each depot, so the depots compare seed by seed."""
    d = DEMO | {k: v for k, v in params.items() if k in ("drones", "people", "runs")}
    cfgs = {depot: BenchmarkConfig(drones=[int(d["drones"])], people=[int(d["people"])],
                                   runs=int(d["runs"]), algos=list(d["algos"]),
                                   baseline=d["baseline"], depot=depot,
                                   campus=_scenario(d).campus)
            for depot in d["compare_depots"]}
    per = len(cfgs[d["compare_depots"][0]].jobs())

    def work(progress):
        rows, summary = [], {}
        for i, (depot, cfg) in enumerate(cfgs.items()):
            r = run_benchmark(cfg, progress=lambda done, _t, i=i: progress(i * per + done, 0))
            for row in r:
                row["depot"] = depot
            rows += r
            summary[depot] = {"name": DEPOT_NAMES[depot],
                              "rows": summarize_rows(r, d["baseline"])}
        return rows, summary

    return start_job(work, per * len(cfgs), {k: d[k] for k in ("drones", "people", "runs", "algos")})


def start_benchmark(params: dict) -> str:
    cfg = BenchmarkConfig(
        drones=[int(x) for x in params.get("drones", [20])],
        people=[int(x) for x in params.get("people", [60])],
        runs=int(params.get("runs", 20)),
        seed=int(params.get("seed", 0)),
        algos=[a for a in params.get("algos", list(CONTINUOUS_ALGORITHMS))
               if a in CONTINUOUS_ALGORITHMS],
        baseline=params.get("baseline", "brute"),
        depot=params.get("depot", "redmond"),
        cell_side=float(params.get("cell_side", 25.0)),
        campus=_scenario(params).campus,
        check=bool(params.get("check")),
    )
    if cfg.baseline not in cfg.algos:
        cfg.algos.append(cfg.baseline)
    total = len(cfg.jobs())
    if not total or total > MAX_BENCH_JOBS:
        raise ValueError(f"benchmark must have 1-{MAX_BENCH_JOBS} runs (this one has {total})")

    def work(progress):
        rows = run_benchmark(cfg, progress=progress)
        return rows, summarize_rows(rows, cfg.baseline)

    return start_job(work, total, cfg.__dict__)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    def _json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, allow_nan=False, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        if self.path == "/api/options":
            return self._json({
                "algorithms": {k: {"name": c.name, "rule": c.rule}
                               for k, c in CONTINUOUS_ALGORITHMS.items()},
                "depots": {k: depot_label(k) for k in DEPOT_CHOICES},
                "staging_depots": list(STAGING_LONLAT),
                "demo": DEMO,
                "completion_frac": COMPLETION_FRAC,
                "metrics": METRICS,
                "statuses": STATUSES,
                "map_available": DATA_PATH.exists(),
            })
        if self.path.startswith("/api/benchmark/"):
            job_id = self.path.rsplit("/", 1)[1]
            as_csv = job_id.endswith(".csv")
            job = JOBS.get(job_id.removesuffix(".csv"))
            if job is None:
                return self._json({"error": "unknown job"}, HTTPStatus.NOT_FOUND)
            if as_csv:
                rows = job.get("rows") or []
                buf = io.StringIO()
                if rows:
                    fields = list(dict.fromkeys(k for r in rows for k in r))
                    writer = csv.DictWriter(buf, fieldnames=fields, restval="")
                    writer.writeheader()
                    writer.writerows(rows)
                body = buf.getvalue().encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/csv")
                self.send_header("Content-Disposition", "attachment; filename=benchmark.csv")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            payload = {k: v for k, v in job.items() if k != "rows"}
            return self._json(_clean(payload))
        return super().do_GET()

    def do_POST(self):
        try:
            params = self._body()
            if self.path == "/api/run":
                return self._json(_clean(run_scenario(params)))
            if self.path == "/api/benchmark":
                return self._json({"job": start_benchmark(params)})
            if self.path == "/api/demo":
                return self._json({"job": start_demo(params)})
            return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except (ValueError, KeyError, TypeError) as exc:
            return self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def _clean(obj):
    """Replace NaN/inf (e.g. a CI from one run) with None so the JSON is valid."""
    if isinstance(obj, float):
        return obj if obj == obj and abs(obj) != float("inf") else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(description="Serve the simulator dashboard locally.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"dashboard on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
