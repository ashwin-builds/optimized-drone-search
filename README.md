# Optimized drone search

Monte Carlo simulator for n drones searching an area for p people and flying
them supplies from a distant depot. The main scenario is an earthquake at the
UW Seattle campus with supplies from Redmond or Bothell (`scenario.py`).

```bash
pip install -r requirements.txt
python campus_map.py fetch            # one-off: real UW campus from OpenStreetMap -> data/

python server.py                      # dashboard at http://localhost:8000
python benchmark.py --drones 10 20 50 --people 60 250 --runs 30 --csv runs.csv
python scenario.py -n 20 -p 60        # quick comparison table
python continuous_viz.py -n 20 -p 60 --algo expected keep brute   # matplotlib replay
pytest
```

| File | What it is |
|---|---|
| `continuous_model.py` | Event-driven simulator and the three algorithms (`expected`, `keep`, `brute`) |
| `scenario.py` | UW scenario: depot distances, drone speeds; real map or square approximation |
| `campus_map.py` | Downloads the UW boundary and building footprints from OpenStreetMap and rasterises them |
| `metrics.py` | Per-run metrics and found/served curves |
| `benchmark.py` | Seed-paired sweeps with 95% confidence intervals |
| `server.py`, `web/` | Local dashboard (replay, time curves, metrics, benchmarks) |
| `drone_model.py`, `brute_force.py`, `visualize.py`, `compare.py` | Original discrete-step model |

New algorithms: subclass `ContinuousModel`, override `choose_returners`, and add
the class to `CONTINUOUS_ALGORITHMS`. Run with `check=True` (or `--check`) while
developing; it raises if kits stop being conserved or drone states go wrong.

## Campus map

`python campus_map.py fetch` saves the University of Washington campus boundary
and building footprints to `data/uw_campus.geojson`. When that file exists,
every tool uses it (`--campus auto`, the default): only campus cells are
searched, 80% of people are placed inside buildings in proportion to floor
area, and drones leave for the depot where the straight line to it crosses
the campus boundary. `--campus square` keeps the original square
approximation.

Map data © OpenStreetMap contributors, available under the
[Open Database License](https://www.openstreetmap.org/copyright).
