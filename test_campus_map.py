"""
Tests for the real-campus pipeline. These use a small made-up map (an L-shaped
"campus" with a few buildings near UW) so they run without the OpenStreetMap
download; they check the machinery, not UW's actual geography.
"""

import json
import math
import random

import pytest
from shapely.geometry import Polygon, mapping

import campus_map
from campus_map import CampusMap, PeopleLayout
from continuous_model import CONTINUOUS_ALGORITHMS
from metrics import summarize
from scenario import Scenario

LON0, LAT0 = -122.31, 47.65
M_PER_DEG_LAT = math.pi / 180 * campus_map.EARTH_R
M_PER_DEG_LON = M_PER_DEG_LAT * math.cos(math.radians(LAT0))


def ll(x_m, y_m):
    """metres east/north of (LON0, LAT0) -> (lon, lat)"""
    return (LON0 + x_m / M_PER_DEG_LON, LAT0 + y_m / M_PER_DEG_LAT)


def box(x0, y0, x1, y1):
    return Polygon([ll(x0, y0), ll(x1, y0), ll(x1, y1), ll(x0, y1)])


# L-shape: 600 x 300 strip plus a 300 x 300 block on top-right -> 270,000 m^2
CAMPUS = Polygon([ll(0, 0), ll(600, 0), ll(600, 600), ll(300, 600), ll(300, 300), ll(0, 300)])
BUILDINGS = [
    (box(50, 50, 150, 150), "Big Hall", 4),         # 10,000 m^2 footprint, 4 floors
    (box(400, 400, 450, 450), "Small Lab", None),   # unknown levels -> 2
    (box(500, 100, 510, 110), "Kiosk", 1),          # smaller than a 25 m cell
    (box(100, 400, 200, 500), "Off Campus", 3),     # outside the L
]


@pytest.fixture
def map_path(tmp_path):
    feats = [{"type": "Feature", "properties": {"role": "campus"}, "geometry": mapping(CAMPUS)}]
    for poly, name, levels in BUILDINGS[:3]:  # fetch() drops off-campus buildings
        feats.append({"type": "Feature", "geometry": mapping(poly),
                      "properties": {"role": "building", "name": name, "levels": levels}})
    path = tmp_path / "campus.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "fetched": "test",
                                "features": feats}))
    campus_map.load_grid.cache_clear()
    return path


def test_projection_is_metric(map_path):
    cmap = CampusMap.load(map_path)
    assert cmap.campus.area == pytest.approx(270_000, rel=2e-3)
    assert cmap.width_m == pytest.approx(600, rel=1e-3)
    assert cmap.height_m == pytest.approx(600, rel=1e-3)
    assert len(cmap.buildings) == 3


def test_grid_follows_the_campus_shape(map_path):
    grid = CampusMap.load(map_path).grid(25.0, "redmond")
    assert grid.k == 24
    assert len(grid.search_cells) * 625 == pytest.approx(270_000, rel=0.01)
    assert (2, 2) in grid.search_cells           # bottom-left of the L
    assert (4, 20) not in grid.search_cells      # the missing top-left quarter
    # Big Hall covers 4x4 cells of 4 floors; the kiosk's area lands in one cell
    big = [c for c in grid.indoor_weight if 2 <= c[0] <= 5 and 2 <= c[1] <= 5]
    assert len(big) == 16 and all(grid.indoor_weight[c] == pytest.approx(625 * 4) for c in big)
    assert grid.indoor_weight[(20, 4)] == pytest.approx(100, rel=0.01)
    assert grid.indoor_weight[(16, 16)] == pytest.approx(625 * 2)  # default 2 levels


def test_depot_exit_is_on_the_boundary_facing_the_depot(map_path):
    cmap = CampusMap.load(map_path)
    for depot in ("redmond", "bothell"):
        grid = cmap.grid(25.0, depot)
        ex, ey = grid.exit_point[0] * 25, grid.exit_point[1] * 25
        assert cmap.campus.boundary.distance(__import__("shapely").Point(ex, ey)) < 1e-6
        dx, dy = cmap.lonlat_to_m(*campus_map.DEPOT_LONLAT[depot])
        # no other campus point is on the depot side of the exit along that line
        c = cmap.campus.centroid
        assert math.hypot(ex - dx, ey - dy) < math.hypot(c.x - dx, c.y - dy)
        assert 10_000 < grid.exit_to_depot_m < 20_000


def test_people_mostly_indoors_weighted_by_floor_area(map_path):
    grid = CampusMap.load(map_path).grid(25.0, "redmond")
    layout = PeopleLayout(grid, indoor_fraction=0.8)
    pts = layout(random.Random(1), 4000)
    cells = [(int(x), int(y)) for x, y in pts]
    assert all(c in grid.search_cells for c in cells)
    indoor = sum(c in grid.building_cells for c in cells) / len(cells)
    assert indoor == pytest.approx(0.8, abs=0.03)
    big = sum(2 <= c[0] <= 5 and 2 <= c[1] <= 5 for c in cells)
    small = sum(16 <= c[0] <= 17 and 16 <= c[1] <= 17 for c in cells)
    # floor area 40,000 vs 5,000 m^2 -> about 8x as many people
    assert 5 < big / max(small, 1) < 12


@pytest.mark.parametrize("key", list(CONTINUOUS_ALGORITHMS))
def test_scenario_on_a_real_map_runs_and_stays_on_campus(map_path, key):
    sc = Scenario(campus="uw", map_path=str(map_path))
    assert sc.uses_map and sc.k == 24 and sc.n_search_cells == 432
    model = CONTINUOUS_ALGORITHMS[key](8, n_people=40, seed=2, check=True, record=True,
                                       **sc.model_kwargs())
    model.run()
    assert set(model.cell_windows) <= model.search_cells
    assert model.searched <= model.search_cells
    assert all(p.cell in model.search_cells for p in model.people)
    assert summarize(model)["completion"] > 0


def test_same_seed_same_people_across_algorithms_on_a_map(map_path):
    sc = Scenario(campus="uw", map_path=str(map_path))
    xs = [[p.xy for p in cls(5, n_people=30, seed=4, **sc.model_kwargs()).people]
          for cls in CONTINUOUS_ALGORITHMS.values()]
    assert xs[0] == xs[1] == xs[2]


def test_missing_map_is_reported_not_silently_replaced(tmp_path):
    with pytest.raises(FileNotFoundError, match="campus_map.py fetch"):
        Scenario(campus="uw", map_path=str(tmp_path / "nope.geojson")).k
    assert not Scenario(campus="auto", map_path=str(tmp_path / "nope.geojson")).uses_map


def test_fetch_builds_the_map_from_an_overpass_response(tmp_path, monkeypatch):
    def way(poly, tags, wid):
        return {"type": "way", "id": wid, "tags": tags,
                "geometry": [{"lon": x, "lat": y} for x, y in poly.exterior.coords]}
    # campus as a multipolygon relation split into two member ways
    coords = list(CAMPUS.exterior.coords)
    relation = {"type": "relation", "id": 1,
                "tags": {"amenity": "university", "name": "University of Washington"},
                "members": [{"type": "way", "role": "outer",
                             "geometry": [{"lon": x, "lat": y} for x, y in coords[:4]]},
                            {"type": "way", "role": "outer",
                             "geometry": [{"lon": x, "lat": y} for x, y in coords[3:]]}]}
    elements = [relation] + [way(p, {"building": "university", "name": n,
                                     "building:levels": str(lv) if lv else ""}, i + 10)
                             for i, (p, n, lv) in enumerate(BUILDINGS)]
    monkeypatch.setattr(campus_map, "_overpass", lambda q: {"elements": elements})
    path = campus_map.fetch(tmp_path / "out.geojson")
    data = json.loads(path.read_text())
    roles = [f["properties"]["role"] for f in data["features"]]
    names = [f["properties"].get("name") for f in data["features"]]
    assert roles.count("campus") == 1 and roles.count("building") == 3
    assert "Off Campus" not in names
    assert "OpenStreetMap" in data["source"]
    cmap = CampusMap.load(path)
    assert cmap.campus.area == pytest.approx(270_000, rel=2e-3)
    levels = {n: lv for _, n, lv in cmap.buildings}
    assert levels == {"Big Hall": 4.0, "Small Lab": 2.0, "Kiosk": 1.0}


def test_overpass_retries_across_mirrors_then_explains(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(req.full_url)
        raise OSError("HTTP Error 502: Bad Gateway")

    monkeypatch.setattr(campus_map.urllib.request, "urlopen", fake_urlopen)
    waits = []
    with pytest.raises(RuntimeError, match="overpass-turbo"):
        campus_map._overpass("q", sleep=waits.append)
    assert len(calls) == len(campus_map.OVERPASS_URLS) * len(campus_map.RETRY_WAITS_S)
    assert waits == [w for w in campus_map.RETRY_WAITS_S if w]


def test_overpass_uses_the_first_mirror_that_answers(monkeypatch):
    import io

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout):
        if "overpass-api.de" in req.full_url:
            raise OSError("HTTP Error 502: Bad Gateway")
        return Resp(b'{"elements": []}')

    monkeypatch.setattr(campus_map.urllib.request, "urlopen", fake_urlopen)
    assert campus_map._overpass("q", sleep=lambda s: None) == {"elements": []}


def test_fetch_from_a_saved_export_handles_non_ascii_names(tmp_path):
    coords = [{"lon": x, "lat": y} for x, y in CAMPUS.exterior.coords]
    hall = [{"lon": x, "lat": y} for x, y in BUILDINGS[0][0].exterior.coords]
    export = {"elements": [
        {"type": "way", "id": 1, "geometry": coords,
         "tags": {"amenity": "university", "name": "University of Washington"}},
        {"type": "way", "id": 2, "geometry": hall,
         "tags": {"building": "yes", "name": "Kāne Hall – Café", "building:levels": "3"}},
    ]}
    src = tmp_path / "export.json"
    src.write_text(json.dumps(export), encoding="utf-8")
    out = campus_map.fetch(tmp_path / "map.geojson", from_files=[src])
    cmap = CampusMap.load(out)
    assert [(n, lv) for _, n, lv in cmap.buildings] == [("Kāne Hall – Café", 3.0)]
