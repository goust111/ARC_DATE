from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _mode_preserve_order(values: np.ndarray) -> int:
    flat = [int(v) for v in np.asarray(values).ravel()]
    counts: dict[int, int] = {}
    order: list[int] = []
    for value in flat:
        if value not in counts:
            counts[value] = 1
            order.append(value)
        else:
            counts[value] += 1
    best = max(counts.values())
    for value in order:
        if counts[value] == best:
            return value
    raise AssertionError("failed to compute the background color")


def _pair_cells_by_color(grid: np.ndarray, bg: int) -> dict[int, tuple[Coord, Coord]]:
    pairs: dict[int, list[Coord]] = {}
    for r, c in np.argwhere(grid != bg):
        color = int(grid[r, c])
        pairs.setdefault(color, []).append((int(r), int(c)))

    assert pairs, "expected at least one foreground pair"
    out: dict[int, tuple[Coord, Coord]] = {}
    for color, coords in pairs.items():
        assert len(coords) == 2, "each non-background color must appear exactly twice"
        out[int(color)] = (coords[0], coords[1])
    return out


def _pair_params(a: Coord, b: Coord) -> tuple[int, int, int]:
    r0, c0 = a
    r1, c1 = b
    mid_r = (int(r0) + int(r1)) // 2
    mid_c = (int(c0) + int(c1)) // 2
    dist = max(abs(int(r1) - int(r0)), abs(int(c1) - int(c0)))
    assert dist >= 1
    return mid_r, mid_c, dist


def _square_cells(center_r: int, center_c: int, dist: int) -> set[Coord]:
    """
    Rasterize the requested square as a centered Chebyshev ring.

    With midpoint rounding on a discrete grid, odd `dist` values map naturally to a
    `dist x dist` ring, while even `dist` values become the centered `(dist+1) x (dist+1)`
    ring whose geometric side length is still `dist`.
    """
    radius = int(dist) // 2
    if radius == 0:
        return {(int(center_r), int(center_c))}

    top = int(center_r) - radius
    bottom = int(center_r) + radius
    left = int(center_c) - radius
    right = int(center_c) + radius

    cells: set[Coord] = set()
    for c in range(left, right + 1):
        cells.add((top, c))
        cells.add((bottom, c))
    for r in range(top, bottom + 1):
        cells.add((r, left))
        cells.add((r, right))
    cells.add((int(center_r), int(center_c)))
    return cells


def _paint_cells(out: np.ndarray, cells: set[Coord], color: int, bg: int) -> None:
    h, w = out.shape
    for r, c in cells:
        assert 0 <= r < h and 0 <= c < w, "square must fit inside the grid"
        existing = int(out[r, c])
        assert existing in (int(bg), int(color)), "different output squares must not overlap"
        out[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each same-color pixel pair by a same-color midpoint marker plus a centered square outline.

    Input structure:
    - The background is the dominant color.
    - Each foreground color appears exactly twice as isolated pixels.

    Rule:
    - For each color pair, compute the integer midpoint by flooring the row and column averages.
    - Let `D` be the larger of the row and column separations.
    - Draw that pair's centered square outline using the discrete radius `D // 2`, and also color the midpoint.
    - Start from a blank background canvas, so original pair pixels survive only when they lie on the new square.

    Key insight:
    - Color identifies the pair, the midpoint chooses where the new object is centered, and the
      pair spacing chooses the square size.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _mode_preserve_order(g)

    out = np.full_like(g, bg)
    for color, (a, b) in _pair_cells_by_color(g, bg).items():
        mid_r, mid_c, dist = _pair_params(a, b)
        _paint_cells(out, _square_cells(mid_r, mid_c, dist), int(color), int(bg))
    return out


# --------------- generator (c) ---------------

# Canvas, palette, and retry policy.
COLORS = tuple(range(10))
MIN_SIDE_BY_COUNT = {1: 10, 2: 13, 3: 16, 4: 19}
MAX_SIDE = 25
MAX_GENERATION_ATTEMPTS = 400

# Pair count and placement spacing.
PAIR_COUNT_RANGE = (1, 4)
FOOTPRINT_GAP = 1
PLACEMENT_RETRIES = 160

# Pair geometry.
PAIR_LAYOUTS = (
    "horizontal",
    "vertical",
    "diag_desc",
    "diag_asc",
    "row_dominant",
    "col_dominant",
)
DIST_MIN = 3
DIST_MAX = 9


def _boxes_separated(a: BBox, b: BBox, gap: int) -> bool:
    return (
        a[2] + int(gap) < b[0]
        or b[2] + int(gap) < a[0]
        or a[3] + int(gap) < b[1]
        or b[3] + int(gap) < a[1]
    )


def _bbox_in_bounds(box: BBox, h: int, w: int) -> bool:
    r0, c0, r1, c1 = box
    return 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w


def _merge_bbox(box: BBox, points: tuple[Coord, Coord]) -> BBox:
    r0, c0, r1, c1 = box
    pr = [points[0][0], points[1][0]]
    pc = [points[0][1], points[1][1]]
    return min(r0, *pr), min(c0, *pc), max(r1, *pr), max(c1, *pc)


def _render_input(shape: tuple[int, int], specs: list[dict], bg: int) -> np.ndarray:
    h, w = shape
    grid = np.full((int(h), int(w)), int(bg), dtype=int)
    for spec in specs:
        color = int(spec["color"])
        a, b = spec["points"]
        assert grid[a[0], a[1]] == bg and grid[b[0], b[1]] == bg
        grid[a[0], a[1]] = color
        grid[b[0], b[1]] = color
    return grid


def _render_expected(shape: tuple[int, int], specs: list[dict], bg: int) -> np.ndarray:
    h, w = shape
    out = np.full((int(h), int(w)), int(bg), dtype=int)
    for spec in specs:
        color = int(spec["color"])
        center_r = int(spec["center"][0])
        center_c = int(spec["center"][1])
        dist = int(spec["dist"])
        _paint_cells(out, _square_cells(center_r, center_c, dist), color, int(bg))
    return out


def _sample_delta(layout: str, max_dist: int) -> tuple[int, int, int] | None:
    if max_dist < DIST_MIN:
        return None

    dist = random.randint(DIST_MIN, max_dist)
    if layout == "horizontal":
        return 0, random.choice((-dist, dist)), dist
    if layout == "vertical":
        return random.choice((-dist, dist)), 0, dist
    if layout == "diag_desc":
        return dist, dist, dist
    if layout == "diag_asc":
        return dist, -dist, dist
    if layout == "row_dominant":
        if dist < 2:
            return None
        minor = random.randint(1, dist - 1)
        return random.choice((-dist, dist)), random.choice((-minor, minor)), dist
    if layout == "col_dominant":
        if dist < 2:
            return None
        minor = random.randint(1, dist - 1)
        return random.choice((-minor, minor)), random.choice((-dist, dist)), dist
    raise AssertionError(f"unknown layout: {layout}")


def _pair_from_center(center: Coord, delta_r: int, delta_c: int) -> tuple[Coord, Coord]:
    center_r, center_c = center
    a = (int(center_r) - int(delta_r) // 2, int(center_c) - int(delta_c) // 2)
    b = (a[0] + int(delta_r), a[1] + int(delta_c))
    return a, b


def _output_box(center: Coord, dist: int) -> BBox:
    radius = int(dist) // 2
    r, c = center
    return int(r) - radius, int(c) - radius, int(r) + radius, int(c) + radius


def _candidate_spec(h: int, w: int, color: int, layout: str) -> dict | None:
    max_dist = min(DIST_MAX, h - 2, w - 2)
    sampled = _sample_delta(layout, max_dist)
    if sampled is None:
        return None
    delta_r, delta_c, dist = sampled

    center_r = random.randint(1, h - 2)
    center_c = random.randint(1, w - 2)
    center = (center_r, center_c)
    a, b = _pair_from_center(center, delta_r, delta_c)
    out_box = _output_box(center, dist)
    if not _bbox_in_bounds(out_box, h, w):
        return None
    if not (0 <= a[0] < h and 0 <= a[1] < w and 0 <= b[0] < h and 0 <= b[1] < w):
        return None

    footprint = _merge_bbox(out_box, (a, b))
    return {
        "color": int(color),
        "layout": layout,
        "dist": int(dist),
        "center": center,
        "points": (a, b),
        "footprint": footprint,
    }


def _sample_specs(h: int, w: int, bg: int, count: int) -> list[dict] | None:
    colors = random.sample([color for color in COLORS if color != bg], count)
    layouts = [random.choice(PAIR_LAYOUTS) for _ in range(count)]
    # Place larger pairs first so dense cases fail less often.
    order = list(range(count))
    random.shuffle(order)
    order.sort(key=lambda idx: layouts[idx], reverse=False)

    specs: list[dict] = []
    used_boxes: list[BBox] = []

    for idx in order:
        color = int(colors[idx])
        layout = layouts[idx]
        placed = None
        for _ in range(PLACEMENT_RETRIES):
            candidate = _candidate_spec(h, w, color, layout)
            if candidate is None:
                continue
            footprint = candidate["footprint"]
            if not all(_boxes_separated(footprint, other, FOOTPRINT_GAP) for other in used_boxes):
                continue
            placed = candidate
            break
        if placed is None:
            return None
        specs.append(placed)
        used_boxes.append(placed["footprint"])

    random.shuffle(specs)
    return specs


def _attempt_shape(count: int) -> tuple[int, int]:
    side_lo = MIN_SIDE_BY_COUNT[count]
    h = random.randint(side_lo, MAX_SIDE)
    w = random.randint(side_lo, MAX_SIDE)
    return h, w


def _has_enough_diversity(specs: list[dict]) -> bool:
    layouts = Counter(str(spec["layout"]) for spec in specs)
    # Single-pair cases are allowed, but multi-pair samples should not collapse to one motif too often.
    if len(specs) <= 1:
        return True
    return len(layouts) >= 2 or random.random() < 0.35


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(*PAIR_COUNT_RANGE)
    h, w = _attempt_shape(count)
    bg = random.choice(COLORS)

    specs = _sample_specs(h, w, bg, count)
    if specs is None or not _has_enough_diversity(specs):
        return None

    inp = _render_input((h, w), specs, bg)
    expected = _render_expected((h, w), specs, bg)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
