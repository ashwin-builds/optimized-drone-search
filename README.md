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
| `continuous_model.py` | Event-driven simulator and the algorithms (`expected`, `expected-uniform`, `keep`, `brute`, `greedy-density`) |
| `scenario.py` | UW scenario: depot distances, drone speeds; real map or square approximation |
| `campus_map.py` | Downloads the UW boundary and building footprints from OpenStreetMap and rasterises them |
| `density.py`, `data/building_visitors.csv` | Prior density surface of where people are (the searcher's belief), its peaks and basins |
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

## Density surface

`density.py` builds the searcher's prior belief of where people are, from
public building data only (it never looks at the simulation's true people).
Each building is a 2-D Gaussian at its footprint centroid, with σ equal to the
footprint's equivalent radius (times `--sigma-factor`), scaled by its daily
visitors from `data/building_visitors.csv`. Measured figures (library gate
counts, residence hall beds, HUB footfall) are marked `is_estimate=false`;
everything else is floor area × a per-type density and says so. Buildings
not in the CSV get a low fallback density.

`--scale peak` makes each Gaussian's height the visitor count (large
footprints get proportionally more mass); `--scale mass` makes its integral
the visitor count. Peaks are cells higher than all 8 campus neighbours; every
cell belongs to the basin its steepest ascent reaches, and
`remaining_mass(peak, unsearched)` is a basin's probability not yet searched.

```bash
python density.py info --scale mass          # peaks, top 15 by mass, data sources
python density.py show --save density.png    # heatmap, peaks, basin boundaries
```

## Simulator defaults and algorithms

Every model now:

* **searches while flying** (`search_en_route=True`): a drone heading for a
  distant cell searches every unclaimed cell its straight line crosses, taking
  `tau` per cell; cells already searched or claimed are flown over at cruise
  speed. `search_en_route=False` restores the old fly-then-search rule.
* **uses the density surface for expected value**: E[finds in T] =
  unfound × (prior mass of the best cells S drones can sweep in T) / (prior
  mass left). `finds="uniform"` (or the `expected-uniform` algorithm) keeps
  the old area-based rule.
* **knows the prior**: `Scenario` passes the density surface (mass scaling,
  cells integrated) on the real map and a uniform one on the square campus.

`Scenario(people="floor")` (default) places the true people by building floor
area, so the prior is imperfect; `people="density"` samples them from the prior
itself, so it is exactly right.

| Algorithm | What it does |
|---|---|
| `expected` | depot runs balance \|H + F + r − (U + E)\|; drones search the nearest unclaimed cell |
| `expected-uniform` | the same with the old uniform E, for reference |
| `keep` | expected value, but nobody returns while the fleet can finish searching within a trip |
| `brute` | find-then-fetch: fetch only for people already found |
| `greedy-density` | each free drone takes the cell with the best prob / (travel + tau); expected-value depot runs |

```bash
python benchmark.py --drones 10 20 50 --people 60 250 --runs 30 \
    --algos expected greedy-density "expected:search_en_route=False" \
    --baseline expected --compare-to greedy-density --placement density
```
