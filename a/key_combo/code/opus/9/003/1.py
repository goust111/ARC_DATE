from __future__ import annotations

import importlib.util
import random
from itertools import combinations
from pathlib import Path

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("_arc_utils_shared", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load arc_utils from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC = _load_arc_utils()
Color = _ARC.Color
bbox_coords = _ARC.bbox_coords
in_bounds = _ARC.in_bounds


PALETTE = tuple(c for c in Color.NOT_BLACK if c != Color.GREY)
CORNERS = ("tl", "tr", "br", "bl")
CORNER_TO_DIR = {
    "tl": (-1, 0),
    "tr": (0, 1),
    "br": (1, 0),
    "bl": (0, -1),
}
SIZE_CHOICES = tuple(
    (h, w)
    for h in range(4, 9)
    for w in range(4, 9)
)
GRID_H_RANGE = (18, 27)
GRID_W_RANGE = (18, 27)
OBJECT_COUNT_RANGE = (2, 4)
MIN_BBOX_GAP = 2
MIN_RAY_PIXELS = 3
MAX_GENERATION_ATTEMPTS = 600
CROSSING_PROBABILITY = 0.72
EXTRA_BOX_ATTEMPTS = 160


def _corner_points(bbox: tuple[int, int, int, int]) -> dict[str, tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {
        "tl": (r0, c0),
        "tr": (r0, c1),
        "br": (r1, c1),
        "bl": (r1, c0),
    }


def _corner_from_markers(
    bbox: tuple[int, int, int, int],
    markers: set[tuple[int, int]],
) -> str:
    points = _corner_points(bbox)
    missing = [name for name, pt in points.items() if pt not in markers]
    assert len(missing) == 1, "markers must occupy exactly three bbox corners"
    return missing[0]


def _bbox_from_gap(
    gap: tuple[int, int],
    corner: str,
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    r, c = gap
    if corner == "tl":
        return r, c, r + height - 1, c + width - 1
    if corner == "tr":
        return r, c - width + 1, r + height - 1, c
    if corner == "br":
        return r - height + 1, c - width + 1, r, c
    if corner == "bl":
        return r - height + 1, c, r, c + width - 1
    raise ValueError(f"unknown corner {corner}")


def _point_in_bbox(pt: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = pt
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _bboxes_too_close(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int = MIN_BBOX_GAP,
) -> bool:
    row_separated = a[2] + gap < b[0] or b[2] + gap < a[0]
    col_separated = a[3] + gap < b[1] or b[3] + gap < a[1]
    return not (row_separated or col_separated)


def _area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _make_object(
    color: int,
    bbox: tuple[int, int, int, int],
    missing_corner: str,
) -> dict:
    points = _corner_points(bbox)
    markers = {pt for name, pt in points.items() if name != missing_corner}
    return {
        "color": int(color),
        "bbox": tuple(map(int, bbox)),
        "missing_corner": missing_corner,
        "gap": points[missing_corner],
        "markers": markers,
        "area": _area(bbox),
    }


def _render_input(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for obj in objects:
        for r, c in obj["markers"]:
            grid[r, c] = obj["color"]
    return grid


def _build_base(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for obj in objects:
        color = int(obj["color"])
        r0, c0, r1, c1 = obj["bbox"]
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
        out[obj["gap"]] = Color.BLACK
        if r1 - r0 >= 2 and c1 - c0 >= 2:
            out[r0 + 1 : r1, c0 + 1 : c1] = Color.GREY
    for obj in objects:
        for r, c in obj["markers"]:
            out[r, c] = obj["color"]
    return out


def _ray_corridor(shape: tuple[int, int], obj: dict) -> list[tuple[int, int]]:
    dr, dc = CORNER_TO_DIR[obj["missing_corner"]]
    r, c = obj["gap"]
    path = []
    while in_bounds(shape, r, c):
        path.append((int(r), int(c)))
        r += dr
        c += dc
    return path


def _trace_ray(base: np.ndarray, obj: dict) -> list[tuple[int, int]]:
    dr, dc = CORNER_TO_DIR[obj["missing_corner"]]
    r, c = obj["gap"]
    path: list[tuple[int, int]] = []
    while in_bounds(base.shape, r, c) and int(base[r, c]) == Color.BLACK:
        path.append((int(r), int(c)))
        r += dr
        c += dc
    return path


def _render_output(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    base = _build_base(shape, objects)
    out = base.copy()
    winners: dict[tuple[int, int], tuple[int, int]] = {}
    for obj in objects:
        for pt in _trace_ray(base, obj):
            prev = winners.get(pt)
            cand = (int(obj["area"]), int(obj["color"]))
            if prev is None or cand[0] < prev[0]:
                winners[pt] = cand
    for (r, c), (_area_value, color) in winners.items():
        out[r, c] = color
    for obj in objects:
        for r, c in obj["markers"]:
            out[r, c] = obj["color"]
    return out


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(np.bincount(g.ravel()).argmax())
    assert bg == Color.BLACK, "expected black background"

    objects: list[dict] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        coords = np.argwhere(g == color)
        assert coords.shape == (3, 2), "each color must appear exactly three times"
        bbox = bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        assert r0 < r1 and c0 < c1, "three markers must define a nondegenerate bbox"
        markers = {tuple(map(int, rc)) for rc in coords}
        points = _corner_points(bbox)
        assert markers <= set(points.values()), "markers must be bbox corners"
        missing_corner = _corner_from_markers(bbox, markers)
        objects.append(
            {
                "color": color,
                "bbox": bbox,
                "missing_corner": missing_corner,
                "gap": points[missing_corner],
                "markers": markers,
                "area": _area(bbox),
            }
        )
    assert 2 <= len(objects) <= 4, "expected 2 to 4 marker groups"
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each three-corner marker group into a grey-filled rectangular frame with one missing corner, then emit a colored ray from that gap.

    Input structure:
    - The grid has a black background and 2 to 4 foreground colors.
    - Each foreground color appears exactly three times, and those three pixels occupy three corners of one non-overlapping bounding box.
    - The absent fourth corner is the gap indicator for that rectangle; box interiors and all space between boxes are black in the input.

    Rule:
    - For each color, compute the bbox of its three pixels and draw the rectangle border in that same color.
    - Leave the missing corner pixel black as a one-cell gap; interpret that gap as belonging to the clockwise-adjacent side, so TL/TR/BR/BL gaps fire up/right/down/left respectively.
    - Fill every strict interior cell of the box with grey.
    - From each gap pixel, extend a ray outward through black cells only; if several rays cover one pixel, keep the color of the box with the smaller bbox area.
    - Restore the original three marker pixels on top of the constructed frame.

    Key insight:
    - The absent corner is not repaired: it simultaneously identifies the completed rectangle and encodes which side emits the outward ray, while ray conflicts are resolved by comparing rectangle areas.
    """
    objects = _extract_objects(grid)
    return _render_output(np.asarray(grid).shape, objects)


def _choose_size(rng: random.Random, used_areas: set[int]) -> tuple[int, int] | None:
    candidates = [(h, w) for h, w in SIZE_CHOICES if h * w not in used_areas]
    if not candidates:
        return None
    return rng.choice(candidates)


def _sample_random_object(
    rng: random.Random,
    shape: tuple[int, int],
    color: int,
    used_areas: set[int],
) -> dict | None:
    H, W = shape
    corner = rng.choice(CORNERS)
    size = _choose_size(rng, used_areas)
    if size is None:
        return None
    h, w = size

    if corner == "tl":
        r0_choices = list(range(MIN_RAY_PIXELS - 1, H - h + 1))
        c0_choices = list(range(0, W - w + 1))
        if not r0_choices or not c0_choices:
            return None
        r0 = rng.choice(r0_choices)
        c0 = rng.choice(c0_choices)
        bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
    elif corner == "tr":
        r0_choices = list(range(0, H - h + 1))
        c1_choices = list(range(w - 1, W - MIN_RAY_PIXELS + 1))
        if not r0_choices or not c1_choices:
            return None
        r0 = rng.choice(r0_choices)
        c1 = rng.choice(c1_choices)
        bbox = (r0, c1 - w + 1, r0 + h - 1, c1)
    elif corner == "br":
        r1_choices = list(range(h - 1, H - MIN_RAY_PIXELS + 1))
        c1_choices = list(range(w - 1, W))
        if not r1_choices or not c1_choices:
            return None
        r1 = rng.choice(r1_choices)
        c1 = rng.choice(c1_choices)
        bbox = (r1 - h + 1, c1 - w + 1, r1, c1)
    else:
        r1_choices = list(range(h - 1, H))
        c0_choices = list(range(MIN_RAY_PIXELS - 1, W - w + 1))
        if not r1_choices or not c0_choices:
            return None
        r1 = rng.choice(r1_choices)
        c0 = rng.choice(c0_choices)
        bbox = (r1 - h + 1, c0, r1, c0 + w - 1)
    return _make_object(color, bbox, corner)


def _corridors_compatible(shape: tuple[int, int], objects: list[dict], candidate: dict) -> bool:
    candidate_corridor = _ray_corridor(shape, candidate)
    for obj in objects:
        if _bboxes_too_close(obj["bbox"], candidate["bbox"]):
            return False
        if any(_point_in_bbox(pt, obj["bbox"]) for pt in candidate_corridor[1:]):
            return False
        if any(_point_in_bbox(pt, candidate["bbox"]) for pt in _ray_corridor(shape, obj)[1:]):
            return False
    return True


def _sample_cross_pair(
    rng: random.Random,
    shape: tuple[int, int],
    colors: tuple[int, int],
) -> list[dict] | None:
    H, W = shape
    for _ in range(220):
        used_areas: set[int] = set()
        cross_r = rng.randint(4, H - 5)
        cross_c = rng.randint(4, W - 5)

        v_corner = rng.choice(("tl", "br"))
        h_corner = rng.choice(("tr", "bl"))

        v_size = _choose_size(rng, used_areas)
        if v_size is None:
            return None
        vh, vw = v_size
        used_areas.add(vh * vw)

        h_size = _choose_size(rng, used_areas)
        if h_size is None:
            return None
        hh, hw = h_size

        if v_corner == "tl":
            gap_r_choices = list(range(max(cross_r + 2, MIN_RAY_PIXELS - 1), H - vh + 1))
            bbox_v = None
            if gap_r_choices and cross_c + vw - 1 < W:
                gap_v = (rng.choice(gap_r_choices), cross_c)
                bbox_v = _bbox_from_gap(gap_v, v_corner, vh, vw)
        else:
            gap_r_choices = list(range(vh - 1, min(cross_r - 1, H - MIN_RAY_PIXELS) + 1))
            bbox_v = None
            if gap_r_choices and cross_c - vw + 1 >= 0:
                gap_v = (rng.choice(gap_r_choices), cross_c)
                bbox_v = _bbox_from_gap(gap_v, v_corner, vh, vw)

        if h_corner == "tr":
            gap_c_choices = list(range(hw - 1, min(cross_c - 2, W - MIN_RAY_PIXELS) + 1))
            bbox_h = None
            if gap_c_choices and cross_r + hh - 1 < H:
                gap_h = (cross_r, rng.choice(gap_c_choices))
                bbox_h = _bbox_from_gap(gap_h, h_corner, hh, hw)
        else:
            gap_c_choices = list(range(max(cross_c + 2, MIN_RAY_PIXELS - 1), W - hw + 1))
            bbox_h = None
            if gap_c_choices and cross_r - hh + 1 >= 0:
                gap_h = (cross_r, rng.choice(gap_c_choices))
                bbox_h = _bbox_from_gap(gap_h, h_corner, hh, hw)

        if bbox_v is None or bbox_h is None:
            continue
        if _bboxes_too_close(bbox_v, bbox_h):
            continue

        vertical = _make_object(colors[0], bbox_v, v_corner)
        horizontal = _make_object(colors[1], bbox_h, h_corner)
        objects = [vertical, horizontal]
        cross_pt = (horizontal["gap"][0], vertical["gap"][1])
        if any(_point_in_bbox(cross_pt, obj["bbox"]) for obj in objects):
            continue
        if cross_pt == vertical["gap"] or cross_pt == horizontal["gap"]:
            continue
        if not _corridors_compatible(shape, [vertical], horizontal):
            continue
        base = _build_base(shape, objects)
        paths = [_trace_ray(base, obj) for obj in objects]
        if any(len(path) < MIN_RAY_PIXELS for path in paths):
            continue
        if cross_pt not in set(paths[0]) or cross_pt not in set(paths[1]):
            continue
        return objects
    return None


def _intersection_pixels(paths: list[list[tuple[int, int]]]) -> set[tuple[int, int]]:
    counts: dict[tuple[int, int], int] = {}
    for path in paths:
        for pt in path:
            counts[pt] = counts.get(pt, 0) + 1
    return {pt for pt, count in counts.items() if count >= 2}


def _validate_semantics(grid: np.ndarray, require_crossing: bool) -> bool:
    try:
        objects = _extract_objects(grid)
    except AssertionError:
        return False

    areas = [int(obj["area"]) for obj in objects]
    if len(set(areas)) != len(areas):
        return False
    if any(obj["color"] == Color.GREY for obj in objects):
        return False
    for a, b in combinations(objects, 2):
        if _bboxes_too_close(a["bbox"], b["bbox"]):
            return False

    base = _build_base(grid.shape, objects)
    paths = [_trace_ray(base, obj) for obj in objects]
    if any(len(path) < MIN_RAY_PIXELS for path in paths):
        return False

    for obj, path in zip(objects, paths, strict=True):
        if path[0] != obj["gap"]:
            return False
        if any(base[r, c] != Color.BLACK for r, c in path):
            return False
        if any(_point_in_bbox(pt, other["bbox"]) for other in objects if other is not obj for pt in path):
            return False

    intersections = _intersection_pixels(paths)
    if require_crossing and not intersections:
        return False

    out = _render_output(grid.shape, objects)
    if np.array_equal(out, grid):
        return False

    if intersections:
        for pt in intersections:
            contenders = [obj for obj, path in zip(objects, paths, strict=True) if pt in set(path)]
            winner = min(contenders, key=lambda obj: obj["area"])
            if out[pt] != winner["color"]:
                return False

    try:
        if not np.array_equal(main(grid), out):
            return False
    except AssertionError:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    H = rng.randint(*GRID_H_RANGE)
    W = rng.randint(*GRID_W_RANGE)
    shape = (H, W)
    count = rng.randint(*OBJECT_COUNT_RANGE)
    colors = tuple(rng.sample(PALETTE, count))
    require_crossing = count >= 2 and rng.random() < CROSSING_PROBABILITY

    objects: list[dict] = []
    used_areas: set[int] = set()
    next_color_index = 0

    if require_crossing:
        pair = _sample_cross_pair(rng, shape, colors[:2])
        if pair is None:
            return None
        objects.extend(pair)
        used_areas.update(int(obj["area"]) for obj in pair)
        next_color_index = 2

    while len(objects) < count:
        candidate = None
        for _ in range(EXTRA_BOX_ATTEMPTS):
            candidate = _sample_random_object(rng, shape, colors[next_color_index], used_areas)
            if candidate is None:
                break
            if _corridors_compatible(shape, objects, candidate):
                base = _build_base(shape, objects + [candidate])
                if len(_trace_ray(base, candidate)) >= MIN_RAY_PIXELS:
                    break
            candidate = None
        if candidate is None:
            return None
        objects.append(candidate)
        used_areas.add(int(candidate["area"]))
        next_color_index += 1

    rng.shuffle(objects)
    grid = _render_input(shape, objects)
    if not _validate_semantics(grid, require_crossing=require_crossing):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the missing-corner frame task")
