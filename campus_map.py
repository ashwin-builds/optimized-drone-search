"""
The real UW Seattle campus, from OpenStreetMap.

    python campus_map.py fetch        # download boundary + buildings -> data/uw_campus.geojson
    python campus_map.py info         # summarise the saved map at the default cell size

The saved file holds the campus boundary (amenity=university, "University of
Washington") and every building footprint whose centre lies on campus, in
longitude/latitude. Map data (c) OpenStreetMap contributors, available under
the Open Database License (https://www.openstreetmap.org/copyright).

CampusMap projects it to metres (x east, y north) and rasterises it onto the
simulation's square grid:

  * search cells   cells whose centre is on campus; only these are searched
  * people         a share of people (indoor_fraction) are inside buildings,
                   placed in proportion to floor area (footprint x levels,
                   2 levels when OSM doesn't say); the rest are outdoors,
                   spread evenly over campus cells that are not buildings
  * depot exit     where the straight line from the campus centre to the
                   depot crosses the campus boundary
"""

from __future__ import annotations

import json
import math
import random
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, mapping, shape
from shapely.ops import linemerge, polygonize, unary_union

DATA_PATH = Path(__file__).parent / "data" / "uw_campus.geojson"
OVERPASS_URLS = ["https://overpass-api.de/api/interpreter",
                 "https://overpass.kumi.systems/api/interpreter"]
# south, west, north, east around the Seattle campus
UW_BBOX = (47.643, -122.322, 47.664, -122.293)
# depots (lon, lat): UW Bothell campus and downtown Redmond
DEPOT_LONLAT = {"redmond": (-122.1215, 47.6740), "bothell": (-122.1906, 47.7589)}
DEFAULT_LEVELS = 2.0
EARTH_R = 6_371_008.8

# ---------------------------------------------------------------------- fetch

_QUERY = """
[out:json][timeout:120];
(
  way["amenity"="university"]["name"="University of Washington"]({bbox});
  relation["amenity"="university"]["name"="University of Washington"]({bbox});
)->.campus;
.campus out geom;
(
  way["building"]({bbox});
  relation["building"]["type"="multipolygon"]({bbox});
);
out geom;
"""


def _overpass(query: str) -> dict:
    data = urllib.parse.urlencode({"data": query}).encode()
    errors = []
    for url in OVERPASS_URLS:
        try:
            req = urllib.request.Request(url, data=data,
                                         headers={"User-Agent": "optimized-drone-search/1.0"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.load(resp)
        except Exception as exc:  # try the next mirror
            errors.append(f"{url}: {exc}")
    raise RuntimeError("could not reach Overpass:\n  " + "\n  ".join(errors))


def _way_polygon(el: dict) -> Polygon | None:
    pts = [(g["lon"], g["lat"]) for g in el.get("geometry", [])]
    if len(pts) < 4 or pts[0] != pts[-1]:
        return None
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)       # repair self-touching outlines
    return None if poly.is_empty or poly.area <= 0 else poly


def _relation_polygon(el: dict):
    """Assemble a multipolygon relation from its member ways (outer minus inner)."""
    lines = {"outer": [], "inner": []}
    for m in el.get("members", []):
        if m.get("type") == "way" and m.get("geometry") and m.get("role") in lines:
            lines[m["role"]].append(LineString([(g["lon"], g["lat"]) for g in m["geometry"]]))
    if not lines["outer"]:
        return None
    outer = unary_union(list(polygonize(linemerge(lines["outer"]))))
    if lines["inner"]:
        inner = unary_union(list(polygonize(linemerge(lines["inner"]))))
        outer = outer.difference(inner)
    return outer if not outer.is_empty else None


def _levels(tags: dict) -> float | None:
    for key in ("building:levels", "levels"):
        try:
            v = float(str(tags.get(key, "")).split(";")[0])
            if v > 0:
                return v
        except ValueError:
            pass
    return None


def fetch(path: Path = DATA_PATH) -> Path:
    s, w, n, e = UW_BBOX
    raw = _overpass(_QUERY.format(bbox=f"{s},{w},{n},{e}"))
    campus_parts, buildings = [], []
    for el in raw["elements"]:
        tags = el.get("tags", {})
        geom = _way_polygon(el) if el["type"] == "way" else _relation_polygon(el)
        if geom is None:
            continue
        if tags.get("amenity") == "university":
            campus_parts.append(geom)
        elif "building" in tags:
            buildings.append((geom, tags))
    if not campus_parts:
        raise RuntimeError("OpenStreetMap returned no 'University of Washington' campus boundary")
    campus = max(campus_parts, key=lambda g: g.area)
    features = [{"type": "Feature", "properties": {"role": "campus", "name": "University of Washington"},
                 "geometry": mapping(campus)}]
    for geom, tags in buildings:
        if not campus.contains(geom.representative_point()):
            continue
        features.append({"type": "Feature", "geometry": mapping(geom), "properties": {
            "role": "building", "name": tags.get("name"), "building": tags.get("building"),
            "levels": _levels(tags)}})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "type": "FeatureCollection",
        "source": "(c) OpenStreetMap contributors, ODbL 1.0 - https://www.openstreetmap.org/copyright",
        "fetched": date.today().isoformat(),
        "features": features,
    }, separators=(",", ":")))
    return path


# ------------------------------------------------------------------- the map


def _project(lonlat: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    """Local equirectangular projection to metres; error < 0.1% over a few km."""
    lon, lat = lonlat[..., 0], lonlat[..., 1]
    x = np.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_R
    return np.stack([x, y], axis=-1)


@dataclass
class Grid:
    """A campus map rasterised at one cell size. Coordinates are in cell units."""
    k: int
    cell_side_m: float
    search_cells: frozenset            # (x, y) cells on campus
    building_cells: frozenset          # campus cells with a building at their centre
    indoor_weight: dict                # cell -> floor area weight (m^2 of floors)
    campus_area_m2: float              # true polygon area
    exit_point: tuple                  # depot-facing boundary point, cell units
    exit_to_depot_m: float


class CampusMap:
    def __init__(self, geojson: dict):
        feats = geojson["features"]
        campus = [f for f in feats if f["properties"].get("role") == "campus"]
        if len(campus) != 1:
            raise ValueError("map needs exactly one campus feature")
        self.source = geojson.get("source", "")
        self.fetched = geojson.get("fetched", "")
        geom_ll = shape(campus[0]["geometry"])
        minx, miny, maxx, maxy = geom_ll.bounds
        self.lon0, self.lat0 = minx, miny       # origin at the south-west corner
        self.campus = self._to_m(geom_ll)
        self.buildings = []
        self.building_tags = []    # OSM building=* value, parallel to self.buildings
        self.building_levels_known = []   # False where levels is the DEFAULT_LEVELS guess
        for f in feats:
            if f["properties"].get("role") != "building":
                continue
            g = self._to_m(shape(f["geometry"]))
            levels = f["properties"].get("levels") or DEFAULT_LEVELS
            self.buildings.append((g, f["properties"].get("name"), float(levels)))
            self.building_tags.append(f["properties"].get("building"))
            self.building_levels_known.append(bool(f["properties"].get("levels")))
        self.width_m = self.campus.bounds[2]
        self.height_m = self.campus.bounds[3]

    @classmethod
    def load(cls, path: Path = DATA_PATH) -> "CampusMap":
        return cls(json.loads(Path(path).read_text()))

    def _to_m(self, geom):
        return shapely.transform(geom, lambda xy: _project(xy, self.lon0, self.lat0))

    def lonlat_to_m(self, lon: float, lat: float) -> tuple[float, float]:
        x, y = _project(np.array([lon, lat], dtype=float), self.lon0, self.lat0)
        return float(x), float(y)

    def grid(self, cell_side_m: float, depot: str) -> Grid:
        k = max(1, math.ceil(max(self.width_m, self.height_m) / cell_side_m))
        centres = (np.arange(k) + 0.5) * cell_side_m
        cx, cy = np.meshgrid(centres, centres, indexing="xy")      # [y, x]
        on_campus = shapely.contains_xy(self.campus, cx, cy)
        search = {(int(x), int(y)) for y, x in zip(*np.nonzero(on_campus))}

        weight: dict[tuple[int, int], float] = {}
        cell_area = cell_side_m ** 2
        for poly, _, levels in self.buildings:
            x0, y0, x1, y1 = poly.bounds
            ix = np.arange(max(0, int(x0 // cell_side_m)), min(k, int(x1 // cell_side_m) + 1))
            iy = np.arange(max(0, int(y0 // cell_side_m)), min(k, int(y1 // cell_side_m) + 1))
            if len(ix) == 0 or len(iy) == 0:
                continue
            gx, gy = np.meshgrid(ix, iy, indexing="xy")
            inside = shapely.contains_xy(poly, (gx + 0.5) * cell_side_m, (gy + 0.5) * cell_side_m)
            cells = [(int(a), int(b)) for a, b in zip(gx[inside], gy[inside])]
            if not cells:  # smaller than a cell: put its floor area in the cell under its centre
                c = poly.representative_point()
                cells = [(min(k - 1, int(c.x // cell_side_m)), min(k - 1, int(c.y // cell_side_m)))]
                share = poly.area * levels
            else:
                share = cell_area * levels
            for cell in cells:
                if cell in search:
                    weight[cell] = weight.get(cell, 0.0) + share

        dx, dy = self.lonlat_to_m(*DEPOT_LONLAT[depot])
        centre = self.campus.centroid
        crossing = LineString([(centre.x, centre.y), (dx, dy)]).intersection(self.campus.boundary)
        pts = [crossing] if isinstance(crossing, Point) else list(getattr(crossing, "geoms", []))
        if not pts:
            raise ValueError("could not find where the depot line leaves campus")
        exit_pt = min(pts, key=lambda p: math.hypot(p.x - dx, p.y - dy))
        return Grid(k=k, cell_side_m=cell_side_m, search_cells=frozenset(search),
                    building_cells=frozenset(weight), indoor_weight=weight,
                    campus_area_m2=self.campus.area,
                    exit_point=(exit_pt.x / cell_side_m, exit_pt.y / cell_side_m),
                    exit_to_depot_m=math.hypot(exit_pt.x - dx, exit_pt.y - dy))

    def outline_cells(self, cell_side_m: float, tolerance_m: float = 2.0) -> dict:
        """Campus boundary and building outlines in cell units (for drawing)."""
        def rings(geom):
            geom = geom.simplify(tolerance_m)
            polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
            out = []
            for p in polys:
                if p.is_empty:
                    continue
                for ring in [p.exterior, *p.interiors]:
                    out.append([[round(x / cell_side_m, 3), round(y / cell_side_m, 3)]
                                for x, y in ring.coords])
            return out
        return {"campus": rings(self.campus),
                "buildings": [r for poly, _, _ in self.buildings for r in rings(poly)]}


@lru_cache(maxsize=16)
def load_grid(path: str, cell_side_m: float, depot: str) -> tuple[CampusMap, Grid]:
    cmap = CampusMap.load(Path(path))
    return cmap, cmap.grid(cell_side_m, depot)


class PeopleLayout:
    """Picklable people sampler for ContinuousModel(people_layout=...)."""

    def __init__(self, grid: Grid, indoor_fraction: float):
        self.indoor_fraction = indoor_fraction
        self.indoor_cells = sorted(grid.indoor_weight)
        self.indoor_w = [grid.indoor_weight[c] for c in self.indoor_cells]
        outdoor = sorted(grid.search_cells - grid.building_cells)
        self.outdoor_cells = outdoor or sorted(grid.search_cells)

    def __call__(self, rng: random.Random, n: int) -> list[tuple[float, float]]:
        out = []
        for _ in range(n):
            indoor = self.indoor_cells and rng.random() < self.indoor_fraction
            if indoor:
                cell = rng.choices(self.indoor_cells, weights=self.indoor_w)[0]
            else:
                cell = self.outdoor_cells[rng.randrange(len(self.outdoor_cells))]
            out.append((cell[0] + rng.random(), cell[1] + rng.random()))
        return out


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "info"
    if cmd == "fetch":
        path = fetch()
        print(f"saved {path}")
        cmd = "info"
    if cmd == "info":
        cmap, grid = load_grid(str(DATA_PATH), 25.0, "redmond")
        named = sum(1 for _, n, _ in cmap.buildings if n)
        print(f"{DATA_PATH} (fetched {cmap.fetched})")
        print(f"  campus area     {cmap.campus.area / 1e6:.3f} km^2, "
              f"{cmap.width_m:.0f} m x {cmap.height_m:.0f} m bounding box")
        print(f"  buildings       {len(cmap.buildings)} ({named} named)")
        print(f"  25 m grid       {grid.k}x{grid.k}, {len(grid.search_cells)} campus cells, "
              f"{len(grid.building_cells)} with buildings")
        for depot in DEPOT_LONLAT:
            g = load_grid(str(DATA_PATH), 25.0, depot)[1]
            print(f"  {depot:<8} exit at ({g.exit_point[0] * 25:.0f} m, {g.exit_point[1] * 25:.0f} m), "
                  f"{g.exit_to_depot_m / 1000:.1f} km to the depot")


if __name__ == "__main__":
    main()
