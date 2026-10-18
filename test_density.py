"""
Tests for density.py. Most use small made-up campuses (rectangles with a few
box buildings near UW) so they need no network; the last few check the real
UW data when data/uw_campus.geojson is present.
"""

import csv
import json
import math

import numpy as np
import pytest
from shapely.geometry import Polygon, mapping

import campus_map
from campus_map import CampusMap
from density import (TYPE_DENSITY, DensitySurface, VisitorRow, building_type, estimate_rows,
                     load_visitor_table, surface_for)

LON0, LAT0 = -122.31, 47.65
M_PER_DEG_LAT = math.pi / 180 * campus_map.EARTH_R
M_PER_DEG_LON = M_PER_DEG_LAT * math.cos(math.radians(LAT0))


def ll(x_m, y_m):
    return (LON0 + x_m / M_PER_DEG_LON, LAT0 + y_m / M_PER_DEG_LAT)


def box(x0, y0, x1, y1):
    return Polygon([ll(x0, y0), ll(x1, y0), ll(x1, y1), ll(x0, y1)])


def make_map(buildings, w=1000, h=600):
    """buildings: list of (x0, y0, x1, y1, name, tag, levels) in metres."""
    feats = [{"type": "Feature", "properties": {"role": "campus"},
              "geometry": mapping(box(0, 0, w, h))}]
    for x0, y0, x1, y1, name, tag, levels in buildings:
        feats.append({"type": "Feature", "geometry": mapping(box(x0, y0, x1, y1)),
                      "properties": {"role": "building", "name": name, "building": tag,
                                     "levels": levels}})
    cmap = CampusMap({"type": "FeatureCollection", "features": feats})
    return cmap, cmap.grid(25.0, "redmond")


def table(**visitors):
    return {n: VisitorRow(n, v, "test", False) for n, v in visitors.items()}


def surface(buildings, visitors, **kw):
    cmap, grid = make_map(buildings, **kw.pop("size", {}))
    return DensitySurface(cmap, grid, visitors, **kw)


def centroid_cell(x0, y0, x1, y1, side=25.0):
    return (int((x0 + x1) / 2 // side), int((y0 + y1) / 2 // side))


# ------------------------------------------------------------- the surface

def test_prob_sums_to_one_and_is_zero_off_campus():
    # L-shaped campus so some grid cells are off campus
    feats = [{"type": "Feature", "properties": {"role": "campus"}, "geometry": mapping(
        Polygon([ll(0, 0), ll(600, 0), ll(600, 600), ll(300, 600), ll(300, 300), ll(0, 300)]))},
        {"type": "Feature", "geometry": mapping(box(100, 100, 200, 200)),
         "properties": {"role": "building", "name": "Hall", "building": "university", "levels": 3}}]
    cmap = CampusMap({"type": "FeatureCollection", "features": feats})
    grid = cmap.grid(25.0, "redmond")
    s = DensitySurface(cmap, grid, table(Hall=1000))
    assert s.prob_grid.sum() == pytest.approx(1.0)
    assert sum(s.prob.values()) == pytest.approx(1.0)
    off = ~s.mask
    assert off.any() and (s.prob_grid[off] == 0).all() and (s.raw_grid[off] == 0).all()
    assert set(s.prob) == set(grid.search_cells)
    assert s.cell_prob((4, 20)) == 0.0           # in the missing quarter of the L
    assert s.cell_prob((5, 5)) == s.prob[(5, 5)] == s.prob_grid[5, 5]


def test_single_building_gives_one_peak_at_its_centroid():
    b = (412.5, 262.5, 512.5, 362.5)   # centre at a cell centre, not on a cell corner
    s = surface([(*b, "Hall", "university", 3)], table(Hall=800))
    assert len(s.peaks) == 1
    p = s.peaks[0]
    assert p.cell == centroid_cell(*b)
    assert p.buildings == ("Hall",) and p.distance_m == 0.0
    assert p.basin == frozenset(s.prob)          # the whole campus drains to it
    assert p.mass == pytest.approx(1.0)
    assert p.height == pytest.approx(800, rel=0.01)   # peak scaling: height = visitors


def test_two_far_apart_buildings_give_two_peaks_and_basins():
    a, b = (112.5, 262.5, 212.5, 362.5), (812.5, 262.5, 912.5, 362.5)
    s = surface([(*a, "West", "university", 3), (*b, "East", "university", 3)],
                table(West=1000, East=600))
    assert len(s.peaks) == 2
    assert {p.cell for p in s.peaks} == {centroid_cell(*a), centroid_cell(*b)}
    west, east = s.peaks                          # ids are ordered by mass
    assert west.buildings == ("West",) and east.buildings == ("East",)
    assert west.basin.isdisjoint(east.basin)
    assert west.basin | east.basin == frozenset(s.prob)
    assert west.mass + east.mass == pytest.approx(1.0)
    assert (s.basin_grid[s.mask] >= 0).all() and (s.basin_grid[~s.mask] == -1).all()


def test_two_overlapping_buildings_merge_into_one_peak():
    # 100 m boxes, centres 50 m apart: separation < sigma (~56 m) -> one hill
    s = surface([(400, 250, 500, 350, "A", "university", 3),
                 (450, 250, 550, 350, "B", "university", 3)], table(A=1000, B=1000))
    assert len(s.peaks) == 1
    assert set(s.peaks[0].buildings) <= {"A", "B"}


def test_peak_scaling_favours_wide_buildings_mass_scaling_does_not():
    small, wide = (125, 275, 175, 325), (650, 200, 850, 400)   # 50 m vs 200 m squares
    blds = [(*small, "Small", "university", 2), (*wide, "Wide", "university", 2)]
    vis = table(Small=1000, Wide=1000)
    peak = surface(blds, vis, scale="peak", size={"w": 1100, "h": 600})
    mass = surface(blds, vis, scale="mass", size={"w": 1100, "h": 600})
    by = lambda s: {p.buildings[0]: p.mass for p in s.peaks}
    pm, mm = by(peak), by(mass)
    # peak: mass ~ sigma^2, and the wide building's sigma is 4x the small one's
    assert pm["Wide"] / pm["Small"] > 8
    assert mm["Wide"] == pytest.approx(mm["Small"], rel=0.05)
    # mass scaling: each building's total is its visitors (up to sampling)
    assert sum(mass.raw.values()) == pytest.approx(2000, rel=0.02)


def test_sigma_follows_footprint_and_factor_with_a_floor():
    blds = [(400, 250, 500, 350, "Hall", "university", 3), (100, 100, 105, 105, "Hut", "shed", 1)]
    s = surface(blds, table(Hall=1, Hut=1), sigma_factor=2.0)
    hall, hut = s.sources
    assert hall.sigma == pytest.approx(2.0 * math.sqrt(10_000 / math.pi) / 25, rel=1e-3)
    assert hut.sigma == 0.5                      # floored
    assert hall.centroid == pytest.approx((18.0, 12.0), abs=1e-3)


# ---------------------------------------------------------------- queries

def test_remaining_mass_tracks_the_unsearched_set():
    a, b = (100, 250, 200, 350), (800, 250, 900, 350)
    s = surface([(*a, "West", "university", 3), (*b, "East", "university", 3)],
                table(West=1000, East=600))
    west, east = s.peaks
    unsearched = set(s.prob)
    assert s.remaining_mass(west.id, unsearched) == pytest.approx(west.mass)
    # searching the other basin leaves this one untouched
    unsearched -= east.basin
    assert s.remaining_mass(west.id, unsearched) == pytest.approx(west.mass)
    assert s.remaining_mass(east.id, unsearched) == 0.0
    # searching part of a basin reduces it; all of it zeroes it
    some = set(list(sorted(west.basin))[:10])
    assert s.remaining_mass(west.id, unsearched - some) == pytest.approx(
        west.mass - sum(s.prob[c] for c in some))
    assert s.remaining_mass(west.id, unsearched - west.basin) == 0.0


def test_peaks_by_mass_orders_by_remaining_mass():
    a, b = (100, 250, 200, 350), (800, 250, 900, 350)
    s = surface([(*a, "West", "university", 3), (*b, "East", "university", 3)],
                table(West=1000, East=600))
    west, east = s.peaks
    full = s.peaks_by_mass(set(s.prob))
    assert [p.id for p, _ in full] == [west.id, east.id]
    # search most of the bigger basin and the order flips
    unsearched = set(s.prob) - set(sorted(west.basin, key=s.cell_prob, reverse=True)[:60])
    flipped = s.peaks_by_mass(unsearched)
    assert flipped[0][0].id == east.id and flipped[0][1] > flipped[1][1]
    assert s.peaks_by_mass(set(s.prob) - west.basin) == [(east, pytest.approx(east.mass))]
    assert s.peaks_by_mass(set(s.prob), min_mass=0.9) == []


def test_deterministic_across_constructions():
    blds = [(100, 250, 200, 350, "West", "university", 3),
            (450, 250, 550, 350, "Mid", "dormitory", 6),
            (800, 250, 900, 350, "East", "university", 3)]
    s1 = surface(blds, table(West=1000, Mid=700))
    s2 = surface(blds, table(West=1000, Mid=700))
    assert np.array_equal(s1.raw_grid, s2.raw_grid)
    assert np.array_equal(s1.prob_grid, s2.prob_grid)
    assert np.array_equal(s1.basin_grid, s2.basin_grid)
    assert [(p.id, p.cell, p.basin, p.mass) for p in s1.peaks] == \
           [(p.id, p.cell, p.basin, p.mass) for p in s2.peaks]


def test_plateau_ties_are_broken_deterministically():
    # two identical buildings placed symmetrically: equal values, but one peak each
    blds = [(150, 250, 250, 350, "L", "university", 3), (750, 250, 850, 350, "R", "university", 3)]
    s = surface(blds, table(L=500, R=500))
    assert len(s.peaks) == 2
    assert s.peaks[0].mass == pytest.approx(s.peaks[1].mass, rel=1e-6)
    # every campus cell belongs to exactly one basin
    assert sum(len(p.basin) for p in s.peaks) == len(s.prob)


# ------------------------------------------------------------ visitor data

def test_visitor_sources_measured_estimate_fallback_and_shared_names():
    blds = [(100, 100, 200, 200, "Library", "university", 4),
            (300, 100, 400, 200, "Twin", "dormitory", 4),
            (500, 100, 550, 150, "Twin", "dormitory", 4),     # same name, 1/4 the floor area
            (700, 100, 800, 200, None, "yes", None)]          # unnamed -> fallback
    vis = {"Library": VisitorRow("Library", 900, "gate count", False),
           "Twin": VisitorRow("Twin", 500, "estimate", True),
           "Gone": VisitorRow("Gone", 10, "no such building", False)}
    s = surface(blds, vis)
    lib, t1, t2, unnamed = s.sources
    assert (lib.kind, lib.visitors) == ("measured", 900)
    assert t1.kind == t2.kind == "estimate"
    assert t1.visitors == pytest.approx(400) and t2.visitors == pytest.approx(100)
    assert unnamed.kind == "fallback"
    assert unnamed.visitors == pytest.approx(10_000 * 2 * 0.25 * TYPE_DENSITY["university"])
    assert s.source_counts() == {"measured": 1, "estimate": 2, "fallback": 1}
    assert s.unmatched_rows == ["Gone"]


def test_building_type_rules():
    assert building_type("university", "Odegaard Undergraduate Library") == "library"
    assert building_type("yes", "South Campus Garage") == "parking"
    assert building_type("hospital", "Brotman Baty Surgery Pavilion") == "hospital"
    assert building_type("yes", "IMA Building") == "sports"
    assert building_type("dormitory", "Maple Hall") == "dormitory"
    assert building_type("yes", "Hub Bike House") == "low_use"
    assert building_type("industrial", "Power Plant") == "low_use"
    assert building_type("university", "Kane Hall") == "university"


def test_csv_loading_validates(tmp_path):
    path = tmp_path / "v.csv"
    path.write_text("name,daily_visitors,source,is_estimate\nA,10,x,true\nB,5,y,false\n",
                    encoding="utf-8")
    t = load_visitor_table(path)
    assert t["A"].is_estimate and not t["B"].is_estimate and t["B"].daily_visitors == 5
    path.write_text("name,daily_visitors,source,is_estimate\nA,10,x,maybe\n", encoding="utf-8")
    with pytest.raises(ValueError, match="is_estimate"):
        load_visitor_table(path)
    path.write_text("name,daily_visitors,source,is_estimate\nA,1,x,true\nA,2,y,true\n",
                    encoding="utf-8")
    with pytest.raises(ValueError, match="twice"):
        load_visitor_table(path)


def test_estimate_rows_skip_known_shared_and_low_use_buildings():
    cmap, _ = make_map([(100, 100, 200, 200, "Hall", "university", 3),
                        (300, 100, 400, 200, "Known", "university", 3),
                        (500, 100, 520, 120, "Shed 1", "shed", 1),
                        (600, 100, 700, 200, "Dup", "university", 2),
                        (800, 100, 900, 200, "Dup", "university", 2)])
    rows = estimate_rows(cmap, table(Known=5))
    assert [r["name"] for r in rows] == ["Hall"]
    assert rows[0]["is_estimate"] == "true"
    assert rows[0]["daily_visitors"] == round(30_000 * TYPE_DENSITY["university"])


def test_surface_does_not_use_the_simulations_people_layout():
    import density
    src = open(density.__file__, encoding="utf-8").read()
    assert "PeopleLayout" not in src.split('"""', 2)[2]   # mentioned in the docstring only


# --------------------------------------------------------- the real UW data

real = pytest.mark.skipif(not campus_map.DATA_PATH.exists(), reason="UW map not downloaded")


@real
def test_visitor_csv_names_all_match_the_map():
    t = load_visitor_table()
    names = {n for _, n, _ in CampusMap.load(campus_map.DATA_PATH).buildings if n}
    assert set(t) <= names
    with open(campus_map.DATA_PATH.parent / "building_visitors.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:   # estimates must say so, and every row cites a source or reasoning
        assert r["source"]
        if r["source"].startswith("estimate"):
            assert r["is_estimate"] == "true"


@real
@pytest.mark.parametrize("scale", ["peak", "mass"])
def test_uw_surface(scale):
    s = surface_for(scale=scale)
    assert sum(s.prob.values()) == pytest.approx(1.0)
    assert len(s.peaks) > 20
    assert sum(p.mass for p in s.peaks) == pytest.approx(1.0)
    top = {b for p in s.peaks[:10] for b in p.buildings}
    assert {"Kane Hall", "Husky Union Building"} <= top
