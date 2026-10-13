from __future__ import annotations

import random
from collections import Counter

import numpy as np


MARKER_COLORS = (1, 2, 3, 6)
SIDE_ORDER = ("top", "right", "bottom", "left")
FIXED_SIDE_COLORS = {"top": 4, "right": 7, "bottom": 5, "left": 9}
ORIENTATIONS = ("TLBR", "TRBL")
GEN_MAX_ATTEMPTS = 500


def _edge_midpoints(shape: tuple[int, int]) -> dict[str, tuple[int, int]]:
    h, w = shape
    return {
        "top": (0, w // 2),
        "right": (h // 2, w - 1),
        "bottom": (h - 1, w // 2),
        "left": (h // 2, 0),
    }


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _intersection_area(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> int:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _nearest_side(
    bbox: tuple[int, int, int, int], shape: tuple[int, int]
) -> str | None:
    h, w = shape
    r0, c0, r1, c1 = bbox
    center_r2 = r0 + r1
    center_c2 = c0 + c1
    distances = {
        "top": center_r2,
        "right": 2 * (w - 1) - center_c2,
        "bottom": 2 * (h - 1) - center_r2,
        "left": center_c2,
    }
    best = min(distances.values())
    winners = [side for side, value in distances.items() if value == best]
    if len(winners) != 1:
        return None
    return winners[0]


def _marker_points(
    bbox: tuple[int, int, int, int], orientation: str
) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if orientation == "TLBR":
        return (r0, c0), (r1, c1)
    assert orientation == "TRBL"
    return (r0, c1), (r1, c0)


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    rectangles: list[dict] = []
    for color in MARKER_COLORS:
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        assert coords.shape == (2, 2), "each marker color must appear exactly twice"
        pts = [(int(r), int(c)) for r, c in coords]
        rows = {pt[0] for pt in pts}
        cols = {pt[1] for pt in pts}
        assert len(rows) == 2 and len(cols) == 2, "marker pair must define a diagonal"
        r0 = min(rows)
        r1 = max(rows)
        c0 = min(cols)
        c1 = max(cols)
        orientation = "TLBR" if set(pts) == {(r0, c0), (r1, c1)} else "TRBL"
        rectangles.append(
            {
                "marker_color": color,
                "bbox": (r0, c0, r1, c1),
                "orientation": orientation,
                "area": _bbox_area((r0, c0, r1, c1)),
            }
        )
    assert 2 <= len(rectangles) <= 4, "expected 2 to 4 marker pairs"
    return rectangles


def _render_output(
    shape: tuple[int, int],
    rectangles: list[dict],
    side_colors: dict[str, int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    ordered = sorted(
        rectangles,
        key=lambda rect: (
            -int(rect["area"]),
            int(rect["bbox"][0]),
            int(rect["bbox"][1]),
            int(rect["marker_color"]),
        ),
    )
    for rect in ordered:
        bbox = rect["bbox"]
        fill_side = _nearest_side(bbox, shape)
        assert fill_side is not None, "rectangle center must have a unique nearest side"
        fill_color = int(side_colors[fill_side])
        r0, c0, r1, c1 = bbox
        out[r0 : r1 + 1, c0 : c1 + 1] = fill_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each diagonal-marker rectangle with the color of its uniquely nearest grid side.

    Input structure:
    - The grid has black background, plus 2 to 4 marker-color pairs where each color appears exactly twice.
    - Each same-color pair sits on opposite corners of an axis-aligned rectangle, using either diagonal orientation.
    - The four edge midpoints contain single-pixel side indicators that encode the top, right, bottom, and left fill colors.

    Rule:
    - For every marker color, take its two pixels as opposite rectangle corners and recover the inclusive bounding box.
    - Compute that rectangle center and compare its distances to the top, right, bottom, and left grid edges.
    - Fill the whole rectangle with the indicator color from the unique nearest side.
    - When rectangles overlap, paint larger areas first and smaller areas later so the smaller rectangle remains on top.
    - Remove the four edge indicators in the output and leave all uncovered cells black.

    Key insight:
    - Marker colors only identify which two pixels belong together; the actual paint color is chosen from spatial position via the rectangle center's nearest side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    side_colors = {
        side: int(g[pos]) for side, pos in _edge_midpoints(g.shape).items()
    }
    rectangles = _extract_rectangles(g)
    return _render_output(g.shape, rectangles, side_colors)


def _sample_shape() -> tuple[int, int]:
    return random.choice((13, 15, 17, 19)), random.choice((13, 15, 17, 19))


def _sample_side_plan(count: int) -> list[str]:
    if count == 2:
        sides = [random.choice(("top", "bottom")), random.choice(("left", "right"))]
    elif count == 3:
        first = random.choice(("top", "bottom"))
        second = random.choice(("left", "right"))
        remaining = [side for side in SIDE_ORDER if side not in {first, second}]
        sides = [first, second, random.choice(remaining)]
    else:
        sides = list(SIDE_ORDER)
    random.shuffle(sides)
    return sides


def _sample_bbox_for_side(
    shape: tuple[int, int], side: str
) -> tuple[int, int, int, int] | None:
    h, w = shape
    max_h = min(9, h - 2)
    max_w = min(9, w - 2)
    for _ in range(250):
        rect_h = random.randint(3, max_h)
        rect_w = random.randint(3, max_w)
        r0_lo, r0_hi = 1, h - rect_h - 1
        c0_lo, c0_hi = 1, w - rect_w - 1
        if r0_lo > r0_hi or c0_lo > c0_hi:
            return None

        row_margin = max(1, h // 6)
        col_margin = max(1, w // 6)

        if side == "top":
            r0_hi = min(r0_hi, max(1, h // 4))
            c0_lo = max(c0_lo, col_margin)
            c0_hi = min(c0_hi, w - rect_w - 1 - col_margin)
        elif side == "bottom":
            r0_lo = max(r0_lo, h - rect_h - 1 - max(1, h // 4))
            c0_lo = max(c0_lo, col_margin)
            c0_hi = min(c0_hi, w - rect_w - 1 - col_margin)
        elif side == "left":
            c0_hi = min(c0_hi, max(1, w // 4))
            r0_lo = max(r0_lo, row_margin)
            r0_hi = min(r0_hi, h - rect_h - 1 - row_margin)
        else:
            assert side == "right"
            c0_lo = max(c0_lo, w - rect_w - 1 - max(1, w // 4))
            r0_lo = max(r0_lo, row_margin)
            r0_hi = min(r0_hi, h - rect_h - 1 - row_margin)

        if r0_lo > r0_hi or c0_lo > c0_hi:
            continue

        r0 = random.randint(r0_lo, r0_hi)
        c0 = random.randint(c0_lo, c0_hi)
        bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
        if _nearest_side(bbox, shape) == side:
            return bbox
    return None


def _build_input(shape: tuple[int, int], rectangles: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for side, pos in _edge_midpoints(shape).items():
        grid[pos] = FIXED_SIDE_COLORS[side]
    for rect in rectangles:
        for r, c in _marker_points(rect["bbox"], rect["orientation"]):
            grid[r, c] = int(rect["marker_color"])
    return grid


def _same_rectangle_set(a: list[dict], b: list[dict]) -> bool:
    key = lambda rect: (
        int(rect["marker_color"]),
        tuple(int(v) for v in rect["bbox"]),
        str(rect["orientation"]),
    )
    return sorted(key(rect) for rect in a) == sorted(key(rect) for rect in b)


def _layout_valid(
    shape: tuple[int, int], rectangles: list[dict], want_overlap: bool
) -> bool:
    total_area = shape[0] * shape[1]
    coverage = np.zeros(shape, dtype=int)
    marker_pts: set[tuple[int, int]] = set()
    side_set = set()
    orientations = {str(rect["orientation"]) for rect in rectangles}
    overlap_pairs = 0
    partial_overlap_seen = False
    overlap_pixels = 0

    for rect in rectangles:
        bbox = rect["bbox"]
        r0, c0, r1, c1 = bbox
        if min(r0, c0) <= 0 or r1 >= shape[0] - 1 or c1 >= shape[1] - 1:
            return False
        for pt in _marker_points(bbox, str(rect["orientation"])):
            if pt in marker_pts:
                return False
            marker_pts.add(pt)
        nearest = _nearest_side(bbox, shape)
        if nearest is None:
            return False
        side_set.add(nearest)
        coverage[r0 : r1 + 1, c0 : c1 + 1] += 1

    for i, rect_a in enumerate(rectangles):
        for rect_b in rectangles[i + 1 :]:
            inter = _intersection_area(rect_a["bbox"], rect_b["bbox"])
            if inter == 0:
                continue
            overlap_pairs += 1
            overlap_pixels += inter
            area_a = int(rect_a["area"])
            area_b = int(rect_b["area"])
            if area_a == area_b:
                return False
            if inter >= min(area_a, area_b):
                return False
            if rect_a["target_side"] == rect_b["target_side"]:
                return False
            partial_overlap_seen = True

    union_area = int((coverage > 0).sum())
    if not (0.08 * total_area <= union_area <= 0.46 * total_area):
        return False
    if overlap_pixels > 0.12 * total_area:
        return False
    if len(side_set) < 2:
        return False
    if len(rectangles) >= 2 and orientations != set(ORIENTATIONS):
        return False
    if want_overlap and not partial_overlap_seen:
        return False
    if not want_overlap and overlap_pairs:
        return False
    return True


def _semantic_checks(
    grid: np.ndarray,
    rectangles: list[dict],
    expected: np.ndarray,
    want_overlap: bool,
) -> bool:
    decoded = _extract_rectangles(grid)
    if not _same_rectangle_set(rectangles, decoded):
        return False

    mids = _edge_midpoints(grid.shape)
    for side, pos in mids.items():
        if int(grid[pos]) != FIXED_SIDE_COLORS[side]:
            return False

    counts = Counter(map(int, grid.ravel()))
    if counts[0] <= grid.size // 2:
        return False
    for color in FIXED_SIDE_COLORS.values():
        if counts[color] != 1:
            return False
    for color in MARKER_COLORS:
        if counts[color] not in (0, 2):
            return False

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return False

    if len(set(map(int, expected.ravel())) - {0}) < 2:
        return False

    overlap_mask = np.zeros(grid.shape, dtype=int)
    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        overlap_mask[r0 : r1 + 1, c0 : c1 + 1] += 1
    if want_overlap and int((overlap_mask >= 2).sum()) == 0:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    shape = _sample_shape()
    count = random.choices((2, 3, 4), weights=(2, 4, 4), k=1)[0]
    want_overlap = random.random() < (0.6 if count == 2 else 0.75)
    side_plan = _sample_side_plan(count)
    marker_colors = random.sample(MARKER_COLORS, count)
    orientations = list(ORIENTATIONS)
    while len(orientations) < count:
        orientations.append(random.choice(ORIENTATIONS))
    random.shuffle(orientations)

    rectangles: list[dict] = []
    for idx in range(count):
        bbox = _sample_bbox_for_side(shape, side_plan[idx])
        if bbox is None:
            return None
        rectangles.append(
            {
                "marker_color": marker_colors[idx],
                "bbox": bbox,
                "orientation": orientations[idx],
                "target_side": side_plan[idx],
                "area": _bbox_area(bbox),
            }
        )

    if not _layout_valid(shape, rectangles, want_overlap):
        return None

    grid = _build_input(shape, rectangles)
    expected = _render_output(shape, rectangles, FIXED_SIDE_COLORS)
    if not _semantic_checks(grid, rectangles, expected, want_overlap):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-pairing nearest-side input")
