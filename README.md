# Optimized drone search

Monte Carlo simulator for n drones searching an area for p people and flying
them supplies from a distant depot. The main scenario is an earthquake at the
UW Seattle campus with supplies from Redmond or Bothell (`scenario.py`).

```bash
pip install -r requirements.txt

python server.py                      # dashboard at http://localhost:8000
python benchmark.py --drones 10 20 50 --people 60 250 --runs 30 --csv runs.csv
python scenario.py -n 20 -p 60        # quick comparison table
python continuous_viz.py -n 20 -p 60 --algo expected keep brute   # matplotlib replay
pytest
```

| File | What it is |
|---|---|
| `continuous_model.py` | Event-driven simulator and the three algorithms (`expected`, `keep`, `brute`) |
| `scenario.py` | UW campus geometry, depot distances and directions, drone speeds |
| `metrics.py` | Per-run metrics and found/served curves |
| `benchmark.py` | Seed-paired sweeps with 95% confidence intervals |
| `server.py`, `web/` | Local dashboard (replay, time curves, metrics, benchmarks) |
| `drone_model.py`, `brute_force.py`, `visualize.py`, `compare.py` | Original discrete-step model |

New algorithms: subclass `ContinuousModel`, override `choose_returners`, and add
the class to `CONTINUOUS_ALGORITHMS`. Run with `check=True` (or `--check`) while
developing; it raises if kits stop being conserved or drone states go wrong.
