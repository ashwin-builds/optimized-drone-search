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
python continuous_viz.py -n 20 -p 60 --algo expected peaks-best brute   # matplotlib replay
pytest
```

| File | What it is |
|---|---|
| `continuous_model.py` | Event-driven simulator and the algorithms (`expected`, `brute`) |
| `peaks_model.py` | Peak-group search: groups work density peaks, scavengers take the rest (`peaks-best`) |
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

## Staging areas and the demo

Besides Redmond and Bothell, `--depot` (and the dashboard's Depot menu) can put the
supplies on or next to campus: `rainier_vista`, `hub_yard`, `denny_yard`,
`parrington_lawn` are campus Mass Assembly Areas that UW EH&S lists for staging relief
supplies, and `uw_tower` is beside the Emergency Operations Center just west of campus.
Positions are approximate (about 50 m). An on-campus depot is its own exit, so a depot
run is the flight across campus plus the 5 min restock; these depots need the real map.

**Run demo** on the dashboard runs 20 drones / 250 people on the UW map with supplies at
Rainier Vista (the replay), and compares Redmond with Rainier Vista over 8 seeds for
`expected`, `peaks-best` and find-then-fetch, leading with mean wait.
`peaks-best` is the best peak-group setting so far (reserve sizes, runners rejoin as
scavengers, `split_rule=basin`).

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
| `peaks-best` | peak groups (below) with the best settings found: reserve sizes, runners rejoin as scavengers, `split_rule=basin` |
| `brute` | find-then-fetch: fetch only for people already found |

These are the only registered algorithms. The keep-searching, uniform-E and
greedy-density classes (`ContinuousKeepSearching`, `ContinuousExpectedUniform`,
`ContinuousGreedyDensity`) stay in `continuous_model.py` for reference and tests.

```bash
python benchmark.py --drones 10 20 50 --people 60 250 --runs 30 \
    --algos expected peaks-best "expected:search_en_route=False" \
    --baseline brute --compare-to expected --placement density
```

### People and completion

`--people` (and the dashboard's Expected people) is the **expected** number of
people needing aid. Each seed draws its own count from a normal distribution
around it (SD 10% of the mean); a draw below 80% of the mean is raised to 85%.
Every algorithm and depot sees the same draw for a seed, and the planners only
know the mean: E uses (expected people − people found so far).

**Completion** is the time 80% of the expected number of people have been
found (`metrics.COMPLETION_FRAC`). The time everyone is served is reported
separately as `all_served`.

## Peak-group search (`peaks_model.py`)

At the start, the top m = n // 10 peaks of the density surface each get a group
of drones, which claims the peak. Group drones search the best cells of their
basin (prob / (travel + tau)), searching whatever they fly over on the way.
When a group's best in-basin rate drops below the rate of moving to an
unclaimed, non-exhausted neighbouring peak, it splits in two (or moves, if it
has one drone or one candidate); when no candidate peak has mass ≥ k it
dissolves. A peak is exhausted once its unsearched mass is below
`k = k_frac × the largest peak's initial mass`, so peaks that scavengers
cleared are never claimed. Scavengers search the nearest cell outside claimed
basins. Group drones deliver only to people their group found; scavengers
deliver to anyone. Each group and the scavengers run their own depot-run
balance.

| Option | Values |
|---|---|
| size strategy | `fixed`: groups of 10, leftovers scavenge · `mass`: all drones shared by peak mass (≥ 2 each) · `reserve`: `scavenger_share` (0.3) scavenge, the rest by mass |
| rejoin | `group`: back to the group after a depot run (the smaller half if it split, scavenger if it dissolved) · `scavengers`: depot runners become scavengers |
| `k_frac` | exhaustion threshold relative to the largest peak (default 0.10) |
| `split_rule` | `cell` (spec default): best single in-basin cell rate vs moving half the group, which at t = 0 almost always splits at once · `basin`: the group's rate on its own basin vs half the group on the neighbour |

Every split, move and dissolve is in `model.group_log`; the replay log and both
viewers colour drones by group (grey = scavenger). Other settings run as
options on `peaks-best`:

```bash
python benchmark.py --drones 20 --people 250 --runs 30 \
    --algos peaks-best "peaks-best:k_frac=0.2" "peaks-best:size_strategy=mass,rejoin=group" \
    --baseline expected
```
