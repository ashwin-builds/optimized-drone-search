"""
Two-drone person mapping on a shared 2D ground grid (coordinates match the Drone Fusion Map).

  Ground frame: metres, x = east, y = north, origin = the intersection where the person in
  black stands. C1 (high_level.mp4) looks north up the x = 0 street; C2 (low_level.mp4) looks
  west along the y = 0 street.

  * Perceptron (perceptron.inc) detects people in each drone's video.
  * Each drone's camera motion is removed (optical flow + periodic SIFT re-anchor to frame 0).
  * Each detection's feet point is projected onto the ground grid with a per-drone homography
    calibrated from the map (MAP below).
  * A LOW track and a HIGH track on the same spot become ONE person with one ID and one mark.
    The link is sticky, and the shared person also corrects any registration drift.
  * The map panel warps BOTH drone frames onto the grid every frame and blends them, then draws
    the drones, their look lines, the grid cells people occupy, and the people.

Usage
-----
  pip install "perceptron>=0.4.0" opencv-python numpy
  export PERCEPTRON_API_KEY=sk_live_...        (Windows: set PERCEPTRON_API_KEY=sk_live_...)

  python drone_grid.py                         # finds *low_level*.mp4 / *high_level*.mp4 for you
  python drone_grid.py --low path/low.mp4 --high path/high.mp4
  python drone_grid.py --save out.mp4 --log people.jsonl
  python drone_grid.py --detector blob         # offline fallback, no API key needed
  python drone_grid.py --grid-only             # just the grid, no drone feeds on the left
  python drone_grid.py --width 2560            # bigger output (16:9); default 1920x1080
  python drone_grid.py --footage --footprints  # optional: footage under the grid, camera outlines

Perceptron calls are cached to detections_cache.json, so after the first run playback is real-time.
Keys: q / Esc = quit, space = pause, f = fullscreen.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------------------------------------------------
# MAP: ground layout + calibration (edit here to match a new scene)
# ----------------------------------------------------------------------------------------------
# For each drone: pixel positions in its FIRST video frame paired with ground positions (metres)
# on the map. The first three are the people at t = 0 from the Drone Fusion Map; the last two
# set the scale across the street (streets ~7.6 m wide).
MAP = {
    "x_range": (-16.0, 36.0),               # area shown on the grid (metres); wider = smaller nodes
    # street centre lines; a person within half_width of one is kept on it (map matching)
    "streets": [{"axis": "x", "at": 0.0, "half_width": 5.0},     # C1's north-south street
                {"axis": "y", "at": 0.0, "half_width": 5.0}],    # C2's east-west street
    "y_range": (-40.0, 12.0),
    "drones": {
        "HIGH": {
            "name": "C1",
            "start": (0.0, -36.0),               # drone ground position at t = 0
            "points": [
                [[972, 470], [0.0, 0.0]],        # person in black (seen by both drones)
                [[985, 785], [0.0, -18.5]],      # blue walker
                [[987, 963], [0.0, -28.9]],      # blue walker
                [[855, 785], [-7.63, -18.5]],    # across-street scale
                [[1115, 785], [7.63, -18.5]],
            ],
            # crop sent to Perceptron (people are ~10 px tall here, so send the street at full res)
            "detect_crop": (700, 280, 1260, 1080),
            "map_clip": None,                    # draw this drone's whole frame on the map
        },
        "LOW": {
            "name": "C2",
            "start": (24.0, 0.0),
            "points": [
                [[979, 584], [0.0, 0.0]],        # person in black (seen by both drones)
                [[982, 686], [7.5, 0.0]],        # orange walker
                [[983, 806], [13.5, 0.0]],       # orange walker
                [[818, 584], [0.0, -3.8]],       # across-street scale
                [[1142, 584], [0.0, 3.8]],
            ],
            "detect_crop": None,
            # only paint C2's footage along its street (x_min, y_min, x_max, y_max) in metres;
            # further out its oblique view stretches buildings across the map
            "map_clip": (-14.0, -9.0, 30.0, 9.0),
        },
    },
}

# Road areas in frame 0 (only used by the offline blob detector to ignore rubble/buildings).
ROAD_RECTS = {
    "LOW":  [(820, 200, 1100, 1080), (700, 440, 1250, 590)],
    "HIGH": [(925, 250, 1030, 1080), (850, 400, 1080, 495)],
}

GRID_CELL_M = 2.0       # one occupancy cell = 2 m x 2 m
MAJOR_M = 10.0          # labelled grid lines every 10 m (as on the map)
FUSE_GATE_M = 3.0       # LOW and HIGH tracks closer than this are the same person
TRACK_GATE_M = 3.5      # max jump of a track between frames
MAX_MISSES = 15         # frames a track survives without a detection
MIN_HITS = 2            # detector passes before a new track gets an ID (filters one-off false positives)
LOW_WEIGHT = 0.5        # how much to trust LOW vs HIGH when both see a person
SMOOTH_ACCEL = 0.002    # motion smoothing: how fast a node may change speed (m/frame^2); lower = smoother
SMOOTH_MEAS_M = 0.8     # how noisy one detection's ground position is (m)
ALIGN_FRAMES = 24       # learn the C2->C1 alignment from the shared person for this many frames, then lock it


def snap_to_street(xy, streets, current=None):
    """Keep a person on the street they're walking along: zero the across-street offset.
    A person sticks to `current` (index of their street) while still within it, so nobody
    flips between streets while crossing an intersection. Returns (position, street index)."""
    def offset(i):
        st = streets[i]
        return abs(xy[0 if st["axis"] == "x" else 1] - st["at"])
    choice = None
    if current is not None and offset(current) <= streets[current]["half_width"]:
        choice = current
    else:
        near = [i for i in range(len(streets)) if offset(i) <= streets[i]["half_width"]]
        choice = min(near, key=offset) if near else None
    if choice is None:
        return np.asarray(xy, float), None
    out = np.array(xy, float)
    out[0 if streets[choice]["axis"] == "x" else 1] = streets[choice]["at"]
    return out, choice



# ----------------------------------------------------------------------------------------------
# Detectors
# ----------------------------------------------------------------------------------------------
class PerceptronDetector:
    """People detection with Perceptron. Results are cached per (video, frame)."""

    def __init__(self, cache_path="detections_cache.json", model=None, crops=None, fallback=None):
        from perceptron import detect, image   # imported lazily so --detector blob needs no SDK
        self._detect, self._image = detect, image
        self.fallback, self.warned = fallback, False
        self.model = model
        self.crops = crops or {}                # drone -> (x0, y0, x1, y1) or None
        self.cache_path = Path(cache_path)
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}

    def __call__(self, frame_bgr, drone, frame_idx, H_t_to_0=None):
        key = f"{drone}:{frame_idx}"
        if key in self.cache:
            return [tuple(b) for b in self.cache[key]]
        x0, y0, crop = 0, 0, frame_bgr
        if self.crops.get(drone):
            x0, y0, x1, y1 = self.crops[drone]
            crop = frame_bgr[y0:y1, x0:x1]
        ok, jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        kwargs = {"model": self.model} if self.model else {}
        try:
            res = self._detect(self._image(jpg.tobytes()), classes=["person"], **kwargs)
        except Exception as e:                      # bad key, network, quota... keep running
            if self.fallback is None:
                raise
            if not self.warned:
                print(f"[perceptron] request failed ({type(e).__name__}: {e}); "
                      "using the offline blob detector for frames that fail", file=sys.stderr)
                self.warned = True
            return self.fallback[drone](frame_bgr, drone, frame_idx, H_t_to_0)
        h, w = crop.shape[:2]
        boxes = [(b.top_left.x + x0, b.top_left.y + y0, b.bottom_right.x + x0, b.bottom_right.y + y0, 1.0)
                 for b in (res.boxes_to_pixels(w, h) or [])]
        self.cache[key] = boxes
        self.cache_path.write_text(json.dumps(self.cache))
        return boxes


class BlobDetector:
    """Offline fallback: small dark or saturated blobs on the road. Good enough to test the
    pipeline without an API key; Perceptron is the real detector."""

    def __init__(self, road_rects, min_area, max_area):
        self.road_rects, self.min_area, self.max_area = road_rects, min_area, max_area

    def __call__(self, frame_bgr, drone, frame_idx, H_t_to_0=None):
        h, w = frame_bgr.shape[:2]
        road = np.zeros((h, w), np.uint8)
        for x0, y0, x1, y1 in self.road_rects[drone]:
            road[y0:y1, x0:x1] = 255
        if H_t_to_0 is not None:   # road rectangles are in frame-0 coords; move them to frame t
            road = cv2.warpPerspective(road, np.linalg.inv(H_t_to_0), (w, h), flags=cv2.INTER_NEAREST)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        v, s = hsv[..., 2].astype(np.int16), hsv[..., 1].astype(np.int16)
        bg_v = cv2.medianBlur(hsv[..., 2], 31).astype(np.int16)
        bg_s = cv2.medianBlur(hsv[..., 1], 31).astype(np.int16)
        mask = (((bg_v - v) > 45) | ((s - bg_s) > 70)).astype(np.uint8) * 255
        mask &= road
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 3), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        boxes = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if self.min_area <= area <= self.max_area and 0.9 * bw <= bh <= 5 * bw:
                boxes.append((int(x), int(y), int(x + bw), int(y + bh), 1.0))
        return boxes


# ----------------------------------------------------------------------------------------------
# Camera motion: homography from frame t to frame 0 of the same video
# ----------------------------------------------------------------------------------------------
class CameraStabilizer:
    """Every frame: Lucas-Kanade feature tracking from the previous frame (fast).
    Every `anchor_every` frames: a full SIFT match against frame 0, so drift can't build up."""

    def __init__(self, frame0, scale=0.5, anchor_every=12):
        self.scale, self.anchor_every, self.count = scale, anchor_every, 0
        if hasattr(cv2, "SIFT_create"):
            self.feat, self.matcher = cv2.SIFT_create(nfeatures=3000), cv2.BFMatcher(cv2.NORM_L2)
        else:                                       # older OpenCV builds without SIFT
            self.feat, self.matcher = cv2.ORB_create(nfeatures=4000), cv2.BFMatcher(cv2.NORM_HAMMING)
        self.kp0, self.des0 = self.feat.detectAndCompute(self._prep(frame0), None)
        self.S = np.diag([scale, scale, 1.0])
        self.S_inv = np.linalg.inv(self.S)
        self.H_small = np.eye(3)
        self.prev, self.prev_pts = None, None

    def _prep(self, f):
        return cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), None, fx=self.scale, fy=self.scale)

    def _anchor(self, g):
        kp, des = self.feat.detectAndCompute(g, None)
        if des is None or len(kp) < 20 or self.des0 is None:
            return None
        pairs = self.matcher.knnMatch(des, self.des0, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 20:
            return None
        src = np.float32([kp[m.queryIdx].pt for m in good])
        dst = np.float32([self.kp0[m.trainIdx].pt for m in good])
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
        return H if H is not None and inl.sum() >= 15 else None

    def __call__(self, frame):
        g = self._prep(frame)
        H = self._anchor(g) if self.count % self.anchor_every == 0 else None
        if H is None and self.prev is not None and self.prev_pts is not None and len(self.prev_pts) >= 20:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, self.prev_pts, None, winSize=(21, 21), maxLevel=3)
            ok = st.ravel() == 1
            if ok.sum() >= 20:
                H_t_prev, _ = cv2.findHomography(nxt[ok], self.prev_pts[ok], cv2.RANSAC, 2.0)
                if H_t_prev is not None:
                    H = self.H_small @ H_t_prev
        if H is not None:
            self.H_small = H
        self.prev = g
        self.prev_pts = cv2.goodFeaturesToTrack(g, maxCorners=600, qualityLevel=0.01, minDistance=8)
        self.count += 1
        return self.S_inv @ self.H_small @ self.S


def project(H, pts):
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


# ----------------------------------------------------------------------------------------------
# Tracking + cross-drone fusion on the ground plane
# ----------------------------------------------------------------------------------------------
def match(a, b, gate):
    """Greedy closest-first pairs (i, j) between point sets a and b with distance under gate."""
    if len(a) == 0 or len(b) == 0:
        return []
    cost = np.linalg.norm(np.asarray(a, float)[:, None, :] - np.asarray(b, float)[None, :, :], axis=2)
    pairs, used_i, used_j = [], set(), set()
    for flat in np.argsort(cost, axis=None):
        i, j = np.unravel_index(flat, cost.shape)
        if cost[i, j] >= gate:
            break
        if i not in used_i and j not in used_j:
            pairs.append((int(i), int(j))); used_i.add(i); used_j.add(j)
    return pairs


@dataclass
class LocalTrack:
    lid: int
    xy: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    hits: int = 1
    misses: int = 0
    det_idx: int | None = None


class LocalTracker:
    """Tracks one drone's detections on the ground so each has a stable per-drone id."""

    def __init__(self):
        self.tracks: list[LocalTrack] = []
        self.next_id = 1

    def update(self, ground_pts, fresh=True):
        """fresh=False when the points were carried forward by optical flow, not re-detected."""
        pairs = match([t.xy + t.vel for t in self.tracks], ground_pts, TRACK_GATE_M)
        for t in self.tracks:
            t.det_idx = None
        hit_t, hit_o = set(), set()
        for ti, oi in pairs:
            t = self.tracks[ti]
            new = np.asarray(ground_pts[oi], float)
            t.vel = 0.7 * t.vel + 0.3 * (new - t.xy)
            t.xy, t.hits, t.misses, t.det_idx = new, t.hits + int(fresh), 0, oi
            hit_t.add(ti); hit_o.add(oi)
        for ti, t in enumerate(self.tracks):
            if ti not in hit_t:
                t.misses += 1
                t.xy = t.xy + t.vel
                t.vel *= 0.8
        for oi, p in enumerate(ground_pts):
            if oi not in hit_o:
                self.tracks.append(LocalTrack(self.next_id, np.asarray(p, float), det_idx=oi))
                self.next_id += 1
        self.tracks = [t for t in self.tracks if t.misses <= MAX_MISSES]
        return self.tracks


class Smooth2D:
    """Constant-velocity Kalman filter for one 2D point. Detections jitter by a metre or so
    from frame to frame; this turns them into a steady glide. Lower `accel` = smoother."""

    def __init__(self, xy, accel=SMOOTH_ACCEL, meas=SMOOTH_MEAS_M):
        self.x = np.array([xy[0], xy[1], 0.0, 0.0])
        self.P = np.diag([1.0, 1.0, 0.5, 0.5])
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        self.Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        self.Q = accel ** 2 * np.diag([0.25, 0.25, 1.0, 1.0])
        self.R = meas ** 2 * np.eye(2)

    def step(self, z=None):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        if z is not None:
            y = np.asarray(z, float) - self.Hm @ self.x
            S = self.Hm @ self.P @ self.Hm.T + self.R
            K = self.P @ self.Hm.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.P = (np.eye(4) - K @ self.Hm) @ self.P
        return self.x[:2].copy()


@dataclass
class Person:
    gid: int
    xy: np.ndarray
    seen_by: set = field(default_factory=set)
    now_by: set = field(default_factory=set)
    trail: list = field(default_factory=list)
    misses: int = 0
    kf: Smooth2D | None = None
    street: int | None = None


class CrossDroneFusion:
    """Links LOW and HIGH tracks of the same person (sticky), so they share one global ID.
    Linked pairs also measure LOW->HIGH registration error; that offset is applied to LOW."""

    def __init__(self, streets=()):
        self.streets = list(streets)
        self.frames = 0
        self.offset = np.zeros(2)
        self.links = {}                    # LOW lid -> HIGH lid
        self.gid = {}                      # (drone, lid) -> global person id
        self.people: dict[int, Person] = {}
        self.next_gid = 1

    def corrected(self, drone, xy):
        return xy + self.offset if drone == "LOW" else xy

    def _new_gid(self):
        self.next_gid += 1
        return self.next_gid - 1

    def update(self, low_tracks, high_tracks):
        low = {t.lid: t for t in low_tracks}
        high = {t.lid: t for t in high_tracks}
        self.links = {l: h for l, h in self.links.items() if l in low and h in high}

        diffs = [high[h].xy - low[l].xy for l, h in self.links.items()
                 if low[l].det_idx is not None and high[h].det_idx is not None]
        self.frames += 1
        if diffs and self.frames <= ALIGN_FRAMES:   # align once, then lock so C2's own motion shows
            self.offset = 0.85 * self.offset + 0.15 * np.mean(diffs, axis=0)

        free_l = [t for t in low_tracks if t.lid not in self.links and t.misses == 0 and t.hits >= MIN_HITS]
        linked_h = set(self.links.values())
        free_h = [t for t in high_tracks if t.lid not in linked_h and t.misses == 0 and t.hits >= MIN_HITS]
        for i, j in match([self.corrected("LOW", t.xy) for t in free_l], [t.xy for t in free_h], FUSE_GATE_M):
            lt, ht = free_l[i], free_h[j]
            self.links[lt.lid] = ht.lid
            gl, gh = self.gid.get(("LOW", lt.lid)), self.gid.get(("HIGH", ht.lid))
            keep = min(g for g in (gl, gh) if g) if (gl or gh) else self._new_gid()
            for g in (gl, gh):
                if g and g != keep:                       # two ids were one person: merge
                    old = self.people.pop(g, None)
                    if old and keep in self.people:
                        self.people[keep].seen_by |= old.seen_by
            self.gid[("LOW", lt.lid)] = self.gid[("HIGH", ht.lid)] = keep

        for drone, tracks in (("LOW", low_tracks), ("HIGH", high_tracks)):
            for t in tracks:
                if (drone, t.lid) not in self.gid and t.hits >= MIN_HITS:
                    self.gid[(drone, t.lid)] = self._new_gid()

        members: dict[int, dict] = {}
        for drone, tracks in (("LOW", low_tracks), ("HIGH", high_tracks)):
            for t in tracks:
                g = self.gid.get((drone, t.lid))
                if g:
                    members.setdefault(g, {})[drone] = t

        for g, mem in members.items():
            p = self.people.setdefault(g, Person(g, np.zeros(2)))
            seen = {d: t for d, t in mem.items() if t.misses == 0}
            p.now_by = set(seen)
            p.seen_by |= set(seen)
            src = seen or mem
            if "LOW" in src and "HIGH" in src:
                z = LOW_WEIGHT * self.corrected("LOW", src["LOW"].xy) + (1 - LOW_WEIGHT) * src["HIGH"].xy
            else:
                d, t = next(iter(src.items()))
                z = self.corrected(d, t.xy)
            if self.streets:
                z, p.street = snap_to_street(z, self.streets, p.street)
            if p.kf is None:
                p.kf = Smooth2D(z)
            p.xy = p.kf.step(z if seen else None)       # smoothed position is what gets drawn
            p.misses = 0 if seen else p.misses + 1
            p.trail.append(p.xy.copy())
            p.trail[:] = p.trail[-60:]
        for g in list(self.people):
            if g not in members:
                del self.people[g]

        det_to_person = {"LOW": {}, "HIGH": {}}
        for g, mem in members.items():
            for d, t in mem.items():
                if t.det_idx is not None:
                    det_to_person[d][t.det_idx] = self.people[g]
        return [p for p in self.people.values() if p.misses <= 6], det_to_person


# ----------------------------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------------------------
# One simple colour rule everywhere (grid circles, trails, video boxes, labels):
#   blue   = seen by C1 only      orange = seen by C2 only      purple = seen by both, counted once
COLOR_C1 = (221, 138, 55)      # BGR for #378add
COLOR_C2 = (48, 90, 216)       # BGR for #d85a30
COLOR_BOTH = (214, 102, 111)   # BGR for #6f66d6
COLOR_UNKNOWN = (150, 150, 150)
DRONE_COLORS = {"HIGH": COLOR_C1, "LOW": COLOR_C2}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def person_color(p):
    if p is None:
        return COLOR_UNKNOWN
    if {"LOW", "HIGH"} <= p.seen_by:
        return COLOR_BOTH
    return COLOR_C1 if "HIGH" in p.seen_by else COLOR_C2


def person_text(p, names):
    return f"P{p.gid}  " + " + ".join(names[d] for d in ("HIGH", "LOW") if d in p.seen_by)


def label(img, text, org, color, scale=0.5, thick=1, pad=4):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x, y - th - 2 * pad), (x + tw + 2 * pad, y + pad), color, -1)
    cv2.putText(img, text, (x + pad, y - pad + 1), FONT, scale, (255, 255, 255), thick, cv2.LINE_AA)


def dashed_line(img, p0, p1, color, dash=10, gap=8, thick=2):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    n = np.linalg.norm(p1 - p0)
    if n < 1:
        return
    u = (p1 - p0) / n
    s = 0.0
    while s < n:
        a, b = p0 + u * s, p0 + u * min(s + dash, n)
        cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), color, thick, cv2.LINE_AA)
        s += dash + gap


class MapView:
    """The shared ground grid (x east, y north, metres), sized to fill a W x H panel."""

    def __init__(self, x_range, y_range, W, H):
        self.x0, self.x1 = x_range
        self.y0, self.y1 = y_range
        self.W, self.H = W, H
        self.u = H / 1080                                   # UI scale: 1.0 on a 1080-tall panel
        u = self.u
        self.L, self.R, self.T, self.B = int(70 * u), int(24 * u), int(76 * u), int(120 * u)
        self.ppm = min((W - self.L - self.R) / (self.x1 - self.x0), (H - self.T - self.B) / (self.y1 - self.y0))
        self.mw = int((self.x1 - self.x0) * self.ppm)
        self.mh = int((self.y1 - self.y0) * self.ppm)
        self.L = (W - self.mw) // 2 + int(10 * u)           # centre the map horizontally
        self.S = np.array([[self.ppm, 0, self.L - self.x0 * self.ppm],
                           [0, -self.ppm, self.T + self.y1 * self.ppm],
                           [0, 0, 1]], float)

    def to_px(self, xy):
        return (int(round(self.L + (xy[0] - self.x0) * self.ppm)),
                int(round(self.T + (self.y1 - xy[1]) * self.ppm)))

    def _combine(self, img, layers):
        """layers: (frame, H_frame_to_ground, opacity, clip) bottom first (only with --footage)."""
        map_area = np.zeros((self.H, self.W), bool)
        map_area[self.T:self.T + self.mh, self.L:self.L + self.mw] = True
        for frame, H, alpha, clip in layers:
            inside = map_area
            if clip:
                (ax, ay), (bx, by) = self.to_px((clip[0], clip[3])), self.to_px((clip[2], clip[1]))
                inside = np.zeros_like(map_area)
                inside[max(ay, 0):by, max(ax, 0):bx] = True
                inside &= map_area
            h, w = frame.shape[:2]
            uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            denom = H[2, 0] * uu + H[2, 1] * vv + H[2, 2]
            dc = H[2] @ [w / 2, h / 2, 1]
            valid = ((denom * dc > 0) & (np.abs(denom) > 0.25 * abs(dc))).astype(np.uint8) * 255
            M = self.S @ H
            warped = cv2.warpPerspective(frame, M, (self.W, self.H), flags=cv2.INTER_LINEAR)
            cover = cv2.warpPerspective(valid, M, (self.W, self.H), flags=cv2.INTER_NEAREST)
            cover[~inside] = 0
            cover = cv2.erode(cover, np.ones((9, 9), np.uint8))
            a = cv2.GaussianBlur(cover, (25, 25), 0).astype(np.float32)[..., None] / 255 * alpha
            a[~map_area] = 0
            img[:] = (a * warped + (1 - a) * img).astype(np.uint8)
        region = img[self.T:self.T + self.mh, self.L:self.L + self.mw]
        region[:] = (region * 0.6).astype(np.uint8)

    def draw(self, people, drones, footprints, stats, names, raw=None, layers=None):
        u = self.u
        fs, ft = 0.62 * u, max(1, int(round(2 * u)))       # label font size / thickness
        img = np.zeros((self.H, self.W, 3), np.uint8)
        if layers:
            self._combine(img, layers)
        view = img[self.T:self.T + self.mh, self.L:self.L + self.mw]      # drawing here clips to the map
        off = np.array([self.L, self.T])

        # grid: 2 m cells faint, 10 m lines brighter and labelled
        tick = (150, 150, 150)
        for x in np.arange(np.ceil(self.x0 / GRID_CELL_M) * GRID_CELL_M, self.x1 + 1e-6, GRID_CELL_M):
            major = abs(x / MAJOR_M - round(x / MAJOR_M)) < 1e-6
            px = self.to_px((x, 0))[0] - self.L
            cv2.line(view, (px, 0), (px, self.mh), (85, 85, 85) if major else (34, 34, 34), 1)
            if major:
                t = f"{int(x)}"
                (tw, _), _ = cv2.getTextSize(t, FONT, 0.55 * u, 1)
                cv2.putText(img, t, (px + self.L - tw // 2, self.T + self.mh + int(26 * u)), FONT, 0.55 * u, tick, 1, cv2.LINE_AA)
        for y in np.arange(np.ceil(self.y0 / GRID_CELL_M) * GRID_CELL_M, self.y1 + 1e-6, GRID_CELL_M):
            major = abs(y / MAJOR_M - round(y / MAJOR_M)) < 1e-6
            py = self.to_px((0, y))[1] - self.T
            cv2.line(view, (0, py), (self.mw, py), (85, 85, 85) if major else (34, 34, 34), 1)
            if major:
                t = f"{int(y)}"
                (tw, _), _ = cv2.getTextSize(t, FONT, 0.55 * u, 1)
                cv2.putText(img, t, (self.L - tw - int(10 * u), py + self.T + int(6 * u)), FONT, 0.55 * u, tick, 1, cv2.LINE_AA)
        cv2.putText(img, "x (m)", (self.L + self.mw - int(50 * u), self.T + self.mh + int(52 * u)), FONT, 0.5 * u, tick, 1, cv2.LINE_AA)
        cv2.putText(img, "y (m)", (max(4, self.L - int(44 * u)), self.T - int(12 * u)), FONT, 0.5 * u, tick, 1, cv2.LINE_AA)
        cv2.rectangle(img, (self.L, self.T), (self.L + self.mw, self.T + self.mh), (120, 120, 120), 1)

        for d, poly in (footprints or {}).items():                      # only with --footprints
            pts = np.array([self.to_px(p) for p in poly], np.int32) - off
            cv2.polylines(view, [pts], True, tuple(int(c * 0.55) for c in DRONE_COLORS[d]), 1, cv2.LINE_AA)

        # drones: white disc with an X, ringed in the drone's colour, dashed line to what they watch
        placed = []
        rd = int(18 * u)
        for d, (pos, look) in drones.items():
            q = tuple(int(v) for v in np.array(self.to_px(pos)) - off)
            t = tuple(int(v) for v in np.array(self.to_px(look)) - off)
            dashed_line(view, q, t, (170, 170, 170), int(10 * u), int(8 * u), max(1, int(2 * u)))
            cv2.circle(view, q, rd, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(view, q, rd, DRONE_COLORS[d], max(2, int(3 * u)), cv2.LINE_AA)
            k = int(rd * 0.5)
            cv2.line(view, (q[0] - k, q[1] - k), (q[0] + k, q[1] + k), (30, 30, 30), max(2, int(3 * u)), cv2.LINE_AA)
            cv2.line(view, (q[0] - k, q[1] + k), (q[0] + k, q[1] - k), (30, 30, 30), max(2, int(3 * u)), cv2.LINE_AA)
            label(view, names[d], (q[0] + rd + int(8 * u), q[1] + int(9 * u)), DRONE_COLORS[d], fs, ft, int(5 * u))
            placed.append((q[0] - rd, q[1] - rd, q[0] + rd + int(70 * u), q[1] + rd))

        # people: trail + big circle in the colour rule above, label in the same colour
        rp = int(16 * u)
        for p in people:
            q = np.array(self.to_px(p.xy)) - off
            placed.append((q[0] - rp - 4, q[1] - rp - 4, q[0] + rp + 4, q[1] + rp + 4))
        for p in sorted(people, key=lambda p: p.gid):
            c = person_color(p)
            if len(p.trail) > 1:
                trail = np.array([self.to_px(q) for q in p.trail], np.int32) - off
                cv2.polylines(view, [trail], False, tuple(int(v * 0.7) for v in c), max(2, int(4 * u)), cv2.LINE_AA)
        for p in sorted(people, key=lambda p: p.gid):
            c = person_color(p)
            q = tuple(int(v) for v in np.array(self.to_px(p.xy)) - off)
            cv2.circle(view, q, rp, c, -1, cv2.LINE_AA)
            cv2.circle(view, q, rp, (255, 255, 255), max(2, int(2 * u)), cv2.LINE_AA)
            text = person_text(p, names)
            (tw, th), _ = cv2.getTextSize(text, FONT, fs, ft)
            g = rp + int(8 * u)
            own = (q[0] - rp - 4, q[1] - rp - 4, q[0] + rp + 4, q[1] + rp + 4)
            for dx, dy in ((g, -g // 2), (-tw - g - 10, -g // 2), (g, g + th), (-tw - g - 10, g + th),
                           (-tw // 2, -g - 4), (-tw // 2, g + th + 10)):
                box = (q[0] + dx, q[1] + dy - th - 10, q[0] + dx + tw + 10, q[1] + dy + 5)
                if not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3]
                           for b in placed if b != own):
                    break
            placed.append(box)
            label(view, text, (q[0] + dx, q[1] + dy), c, fs, ft, int(5 * u))

        # header
        cv2.putText(img, "Shared ground grid", (self.L, int(34 * u)), FONT, 0.9 * u, (240, 240, 240), max(1, int(2 * u)), cv2.LINE_AA)
        txt = f"t = {stats['t']:.1f} s    people: {stats['unique']}    seen by both: {stats['merged']}"
        (tw, _), _ = cv2.getTextSize(txt, FONT, 0.6 * u, 1)
        cv2.putText(img, txt, (self.L + self.mw - tw, int(34 * u)), FONT, 0.6 * u, (200, 200, 200), max(1, int(u)), cv2.LINE_AA)

        # legend: the three colours
        y = self.H - int(28 * u)
        x = self.L
        for c, text in ((COLOR_C1, "seen by C1"), (COLOR_C2, "seen by C2"), (COLOR_BOTH, "seen by both (counted once)")):
            cv2.circle(img, (x + int(10 * u), y - int(6 * u)), int(10 * u), c, -1, cv2.LINE_AA)
            cv2.circle(img, (x + int(10 * u), y - int(6 * u)), int(10 * u), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, text, (x + int(28 * u), y), FONT, 0.6 * u, (220, 220, 220), max(1, int(u)), cv2.LINE_AA)
            (tw, _), _ = cv2.getTextSize(text, FONT, 0.6 * u, max(1, int(u)))
            x += tw + int(64 * u)
        return img


def draw_camera(frame, drone, name, dets, det_to_person, names, ui=1.0):
    """One drone feed with Perceptron's boxes, labelled and coloured exactly like the grid.
    `ui` is how much bigger to draw so text stays readable after the frame is shrunk."""
    out = frame.copy()
    pad = int((14 if drone == "HIGH" else 8) * ui)     # C1's people are tiny, so give their boxes room
    th = max(2, int(3 * ui))
    for k, (x1, y1, x2, y2, _) in enumerate(dets):
        p = det_to_person.get(k)
        c = person_color(p)
        cv2.rectangle(out, (int(x1) - pad, int(y1) - pad), (int(x2) + pad, int(y2) + pad), c, th + 1)
        if p:
            label(out, person_text(p, names), (x2 + pad + 8 * ui, (y1 + y2) / 2 + 14 * ui), c,
                  1.15 * ui, th, int(8 * ui))
    label(out, f"{name}  ({drone.lower()} drone)", (24 * ui, 64 * ui), DRONE_COLORS[drone], 1.3 * ui, th, int(10 * ui))
    return out


# ----------------------------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------------------------
def find_video(arg, pattern):
    if arg:
        if not Path(arg).exists():
            raise SystemExit(f"Video not found: {arg}  (current folder: {Path.cwd()})")
        return arg
    here = Path(__file__).resolve().parent
    for folder in (Path.cwd(), here, here.parent):
        hits = sorted(folder.glob(pattern))
        if hits:
            return str(hits[0])
    for folder in (here.parent, Path.cwd()):
        hits = sorted(folder.rglob(pattern))
        if hits:
            return str(hits[0])
    raise SystemExit(f"Could not find a video matching {pattern} near {here}. "
                     f"Pass the paths, e.g. --low path/to/low_level.mp4 --high path/to/high_level.mp4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--low", default=None, help="C2 video (default: finds *low_level*.mp4)")
    ap.add_argument("--high", default=None, help="C1 video (default: finds *high_level*.mp4)")
    ap.add_argument("--detector", choices=["perceptron", "blob"], default="perceptron")
    ap.add_argument("--model", default=None, help="Perceptron model id, e.g. perceptron-mk1.5 or isaac-0.2-2b-preview")
    ap.add_argument("--detect-every", type=int, default=4,
                    help="call the detector every N frames; boxes are tracked with optical flow in between")
    ap.add_argument("--save", default=None, help="write the combined view to this .mp4")
    ap.add_argument("--log", default=None, help="write per-frame people/drone positions to this .jsonl")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--footage", action="store_true", help="also paint the drone footage under the grid")
    ap.add_argument("--grid-only", action="store_true", help="show just the grid, without the drone feeds")
    ap.add_argument("--width", type=int, default=1920, help="output width in pixels (16:9); default 1920x1080")
    ap.add_argument("--footprints", action="store_true", help="outline each camera's view on the grid")
    args = ap.parse_args()

    paths = {"LOW": find_video(args.low, "*low_level*.mp4"), "HIGH": find_video(args.high, "*high_level*.mp4")}
    caps = {d: cv2.VideoCapture(p) for d, p in paths.items()}
    frames0 = {}
    for d, c in caps.items():
        ok, f = c.read()
        if not ok:
            raise SystemExit(f"OpenCV could not decode {paths[d]}. Re-encode it with: "
                             f"ffmpeg -i in.mp4 -c:v libx264 -pix_fmt yuv420p out.mp4")
        frames0[d] = f
    print(f"C1 (high) = {paths['HIGH']}\nC2 (low)  = {paths['LOW']}")
    fps = caps["LOW"].get(cv2.CAP_PROP_FPS) or 24

    cfg = MAP["drones"]
    names = {d: cfg[d]["name"] for d in cfg}
    H0_ground = {}
    for d in ("LOW", "HIGH"):
        src = np.float32([a for a, _ in cfg[d]["points"]])
        dst = np.float32([b for _, b in cfg[d]["points"]])
        H0_ground[d], _ = cv2.findHomography(src, dst, 0)
    out_w = args.width - args.width % 4
    out_h = int(round(out_w * 9 / 16)) // 2 * 2
    grid_w = out_w if args.grid_only else out_w // 2      # 50/50 split: grid on the right half
    mapview = MapView(MAP["x_range"], MAP["y_range"], grid_w, out_h)

    stab = {d: CameraStabilizer(frames0[d]) for d in frames0}
    blob = {"LOW": BlobDetector(ROAD_RECTS, 300, 5000), "HIGH": BlobDetector(ROAD_RECTS, 50, 900)}
    detector = blob
    if args.detector == "perceptron":
        if not (os.environ.get("PERCEPTRON_API_KEY") or os.environ.get("FAL_KEY")) and not Path("detections_cache.json").exists():
            print("[perceptron] no PERCEPTRON_API_KEY set; using the offline blob detector", file=sys.stderr)
        else:
            try:
                p = PerceptronDetector(model=args.model, fallback=blob,
                                       crops={d: cfg[d]["detect_crop"] for d in cfg})
                detector = {"LOW": p, "HIGH": p}
            except ImportError:
                print('[perceptron] SDK not installed (pip install "perceptron>=0.4.0"); '
                      "using the offline blob detector", file=sys.stderr)

    show = not args.no_window
    if show:
        try:
            cv2.namedWindow("Drone people grid", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Drone people grid", out_w, out_h)
        except cv2.error:
            show = False
            args.save = args.save or "drone_grid_output.mp4"
            print("[display] this OpenCV build has no GUI (opencv-python-headless?). "
                  f"Writing {args.save} instead. For a live window: "
                  "pip uninstall -y opencv-python-headless && pip install opencv-python", file=sys.stderr)

    local = {"LOW": LocalTracker(), "HIGH": LocalTracker()}
    fusion = CrossDroneFusion(MAP.get("streets", []))
    writer, logf = None, (open(args.log, "w") if args.log else None)
    prev_gray = {"LOW": None, "HIGH": None}
    dets = {"LOW": [], "HIGH": []}
    drone_kf = {}
    center0 = {d: project(H0_ground[d], [[frames0[d].shape[1] / 2, frames0[d].shape[0] / 2]])[0] for d in frames0}
    frame_idx, paused, canvas, tic = 0, False, None, time.time()

    while True:
        if not paused:
            frames = {}
            for d, c in caps.items():
                ok, f = c.read()
                if ok:
                    frames[d] = f
            if len(frames) < 2:
                break
            tic = time.time()
            footprints, ground_H, look = {}, {}, {}
            for d, f in frames.items():
                H_t0 = stab[d](f)
                H_t_ground = H0_ground[d] @ H_t0
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                fresh = frame_idx % args.detect_every == 0 or prev_gray[d] is None
                if fresh:
                    dets[d] = detector[d](f, d, frame_idx, H_t0)
                elif dets[d]:
                    # carry boxes forward with Lucas-Kanade optical flow on the box centres
                    ctr = np.float32([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in dets[d]]).reshape(-1, 1, 2)
                    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray[d], gray, ctr, None, winSize=(21, 21), maxLevel=3)
                    moved = []
                    for b, p0, p1, ok in zip(dets[d], ctr.reshape(-1, 2), nxt.reshape(-1, 2), st.ravel()):
                        dx, dy = (p1 - p0) if ok else (0.0, 0.0)
                        moved.append((b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy, b[4]))
                    dets[d] = moved
                prev_gray[d] = gray
                ground_H[d] = H_t_ground

                feet = [((b[0] + b[2]) / 2, b[3]) for b in dets[d]]   # bottom-centre touches the ground
                local[d].update(list(project(H_t_ground, feet)) if feet else [], fresh)
                h, w = f.shape[:2]
                v_top = h * 0.3 if d == "LOW" else 0                  # C2's top rows are near its horizon
                footprints[d] = project(H_t_ground, [[0, v_top], [w, v_top], [w, h], [0, h]])
                look[d] = project(H_t_ground, [[w / 2, h / 2]])[0]

            people, det_to_person = fusion.update(local["LOW"].tracks, local["HIGH"].tracks)

            # C2 (LOW) uses the same drift correction as its detections everywhere on the map
            T = np.array([[1, 0, fusion.offset[0]], [0, 1, fusion.offset[1]], [0, 0, 1]])
            footprints["LOW"] = footprints["LOW"] + fusion.offset
            look["LOW"] = look["LOW"] + fusion.offset
            layers = None
            if args.footage:
                layers = [(frames["HIGH"], ground_H["HIGH"], 1.0, cfg["HIGH"]["map_clip"]),
                          (frames["LOW"], T @ ground_H["LOW"], 0.9, cfg["LOW"]["map_clip"])]
            raw = {d: [fusion.corrected(d, t.xy) for t in local[d].tracks if t.misses == 0] for d in local}
            # drone position = start position + how far its view centre has slid over the ground,
            # smoothed the same way as the people so the drones glide too
            drones = {}
            for d in ("HIGH", "LOW"):
                moved = look[d] - center0[d] - (fusion.offset if d == "LOW" else 0)
                pos = np.array(cfg[d]["start"]) + moved
                if d not in drone_kf:
                    drone_kf[d] = Smooth2D(pos, accel=0.001, meas=1.0)
                pos = drone_kf[d].step(pos)
                # dashed scan line: to the person both drones see (as on the map), else straight ahead
                shared = [p for p in people if {"LOW", "HIGH"} <= p.now_by]
                drones[d] = (pos, shared[0].xy if shared else pos + (look[d] - pos) * 0.5)

            stats = {"t": frame_idx / fps, "unique": len(people),
                     "merged": sum(1 for p in people if {"LOW", "HIGH"} <= p.seen_by)}
            right = mapview.draw(people, drones, footprints if args.footprints else None, stats, names, raw, layers)
            canvas = right

        if not paused and not args.grid_only:
            # split screen: the two drone feeds stacked on the left half, the grid on the right half
            feed_w, feed_h = out_w // 2, out_h // 2
            feeds = []
            for d in ("HIGH", "LOW"):
                fw = frames[d].shape[1]
                ui = (0.62 * mapview.u / 1.15) * (fw / feed_w)      # same on-screen text size as the grid
                f = draw_camera(frames[d], d, names[d], dets[d], det_to_person[d], names, ui)
                feeds.append(cv2.resize(f, (feed_w, feed_h), interpolation=cv2.INTER_AREA))
            left = np.vstack(feeds)
            canvas = np.hstack([left, right])
            cv2.line(canvas, (feed_w, 0), (feed_w, out_h), (70, 70, 70), 2)
            cv2.line(canvas, (0, feed_h), (feed_w, feed_h), (70, 70, 70), 2)

        if not paused:
            if args.save:
                if writer is None:
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                             (canvas.shape[1], canvas.shape[0]))
                writer.write(canvas)
            if logf:   # one line per frame, ready for the database step of the pipeline
                logf.write(json.dumps({
                    "frame": frame_idx, "t": round(frame_idx / fps, 3),
                    "people": [{"id": f"P{p.gid}", "x": round(float(p.xy[0]), 2), "y": round(float(p.xy[1]), 2),
                                "seen_by": sorted(names[d] for d in p.now_by)} for p in people],
                    "drones": {names[d]: [round(float(v), 2) for v in drones[d][0]] for d in drones},
                }) + "\n")
            frame_idx += 1

        if show and canvas is not None:
            cv2.imshow("Drone people grid", canvas)
            wait = 30 if paused else max(1, int(1000 / fps - (time.time() - tic) * 1000))
            key = cv2.waitKey(wait) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("f"):        # toggle fullscreen
                full = cv2.getWindowProperty("Drone people grid", cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
                cv2.setWindowProperty("Drone people grid", cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL if full else cv2.WINDOW_FULLSCREEN)
            if key == ord(" "):
                paused = not paused

    if writer:
        writer.release()
        print("saved", args.save)
    if logf:
        logf.close()
        print("saved", args.log)
    print("people at end:", [(f"P{p.gid}", sorted(names[d] for d in p.seen_by), np.round(p.xy, 1).tolist())
                             for p in fusion.people.values()])
    if show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
