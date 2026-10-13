from __future__ import annotations

import numpy as np

from arc_utils import colors_present, find_components, get_bg, in_bounds
import random


def _role_colors(g: np.ndarray, bg: int) -> tuple[int, int]:
    fg = colors_present(g, exclude={bg})
    assert len(fg) == 2, "expected exactly two foreground colors"

    counts = {color: int(np.count_nonzero(g == color)) for color in fg}
    marker_count = min(counts.values())
    body_count = max(counts.values())
    markers = [color for color, count in counts.items() if count == marker_count]
    bodies = [color for color, count in counts.items() if count == body_count]
    assert marker_count < body_count, "fracture marker color must be the minority foreground color"
    assert len(markers) == 1 and len(bodies) == 1, "foreground roles must be unique"
    return markers[0], bodies[0]


def _set_body(out: np.ndarray, r: int, c: int, color: int) -> None:
    assert in_bounds(out.shape, r, c), "shifted object pixel must stay in bounds"
    assert int(out[r, c]) != int(color), "shifted object pixels must not collide"
    out[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove each fracture marker line and push the surrounding object pixels one cell away from it.

    Input structure
    - A background grid with 8-connected foreground objects made from a body color and a rarer marker color.
    - Each object has a non-square bbox; the marker pixels form one complete fracture line across the shorter bbox axis.

    Rule
    - Erase all marker pixels.
    - For a wide object, each row has one marker; body pixels left of it shift one cell left and body pixels right of it shift one cell right.
    - For a tall object, each column has one marker; body pixels above it shift one cell up and body pixels below it shift one cell down.

    Key insight
    - The marker line is a crack through the object, so the two sides separate by one pixel perpendicular to the crack.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    marker, body = _role_colors(g, bg)
    out = np.full_like(g, bg)

    objects = find_components(
        g,
        bg=bg,
        connectivity=8,
        monochromatic=False,
        colors=[marker, body],
    )
    assert objects, "expected at least one fractured object"

    for obj in objects:
        coords = obj["coords"]
        r0, c0, r1, c1 = obj["bbox"]
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        assert height != width, "fractured object bbox must determine an orientation"

        vals = g[coords[:, 0], coords[:, 1]]
        assert bool(np.any(vals == marker)), "each object must contain marker pixels"
        assert bool(np.any(vals == body)), "each object must contain body pixels"

        marker_coords = coords[vals == marker]
        if width > height:
            marker_by_row: dict[int, list[int]] = {}
            for r, c in marker_coords:
                marker_by_row.setdefault(int(r), []).append(int(c))
            assert set(marker_by_row) == set(range(r0, r1 + 1)), "wide object needs one marker in each row"
            assert all(len(cols) == 1 for cols in marker_by_row.values()), "wide object marker row must be unique"

            for r, c in coords[vals == body]:
                rr = int(r)
                cc = int(c)
                marker_c = marker_by_row[rr][0]
                assert cc != marker_c, "body pixel cannot occupy the marker column"
                shifted_c = cc - 1 if cc < marker_c else cc + 1
                _set_body(out, rr, shifted_c, body)
        else:
            marker_by_col: dict[int, list[int]] = {}
            for r, c in marker_coords:
                marker_by_col.setdefault(int(c), []).append(int(r))
            assert set(marker_by_col) == set(range(c0, c1 + 1)), "tall object needs one marker in each column"
            assert all(len(rows) == 1 for rows in marker_by_col.values()), "tall object marker column must be unique"

            for r, c in coords[vals == body]:
                rr = int(r)
                cc = int(c)
                marker_r = marker_by_col[cc][0]
                assert rr != marker_r, "body pixel cannot occupy the marker row"
                shifted_r = rr - 1 if rr < marker_r else rr + 1
                _set_body(out, shifted_r, cc, body)

    return out

# --------------- generator (o5x) ---------------

# Canvas, palette, and retry budget.
BACKGROUND_COLOR = 0
FOREGROUND_COLORS = tuple(range(1, 10))
MAX_GRID_SIZE = 19
MIN_GRID_SIZE_BY_COUNT = (8, 8, 10, 12, 14, 16, 18, 19)
GENERATION_RETRIES = 300

# Object counts, dimensions, and orientations.
MIN_OBJECT_COUNT = 1
MAX_OBJECT_COUNT = 7
SHORT_SIDE_SIZES = (2, 3, 4, 5, 6)
LONG_SIDE_FACTOR = 2
WIDE = "wide"
TALL = "tall"
ORIENTATIONS = [WIDE, TALL]

# Dense layouts try smaller rectangles first while preserving random tie order.
DENSE_COUNT_THRESHOLD = 6
DENSE_PREFERRED_MAX_SHORT_SIDE = 3
PACKED_COUNT_THRESHOLD = 4
PACKED_PREFERRED_MAX_SHORT_SIDE = 4

# Placement and retry constraints.
SHIFT_PADDING = 1
MIN_CHEB_GAP = 2

BBox = tuple[int, int, int, int]
Location = tuple[int, int]
ObjectSpec = tuple[int, int, int, str]


def _interval_sep(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return b0 - a1
    if b1 < a0:
        return a0 - b1
    return 0


def _cheb_gap_at_least_2(a: BBox, b: BBox) -> bool:
    dr = _interval_sep(a[0], a[2], b[0], b[2])
    dc = _interval_sep(a[1], a[3], b[1], b[3])
    return max(dr, dc) >= MIN_CHEB_GAP


def _overlap(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expanded_bbox(r: int, c: int, h: int, w: int) -> BBox:
    if w > h:
        return r, c - SHIFT_PADDING, r + h - 1, c + w - 1 + SHIFT_PADDING
    return r - SHIFT_PADDING, c, r + h - 1 + SHIFT_PADDING, c + w - 1


def _object_dims(short_side: int, orient: str) -> tuple[int, int]:
    if orient == WIDE:
        return short_side, LONG_SIDE_FACTOR * short_side
    return LONG_SIDE_FACTOR * short_side, short_side


def _locations(n: int, h: int, w: int, originals: list[BBox], expanded: list[BBox]) -> list[Location]:
    if w > h:
        row_range = range(0, n - h + 1)
        col_range = range(SHIFT_PADDING, n - w - SHIFT_PADDING + 1)
    else:
        row_range = range(SHIFT_PADDING, n - h - SHIFT_PADDING + 1)
        col_range = range(0, n - w + 1)

    locs: list[Location] = []
    for r in row_range:
        for c in col_range:
            orig = (r, c, r + h - 1, c + w - 1)
            exp = _expanded_bbox(r, c, h, w)
            if any(not _cheb_gap_at_least_2(orig, old) for old in originals):
                continue
            if any(_overlap(exp, old) for old in expanded):
                continue
            locs.append((r, c))
    random.shuffle(locs)
    return locs


def _draw_object(grid: np.ndarray, expected: np.ndarray, r: int, c: int, s: int, orient: str, body: int, marker: int) -> None:
    h, w = _object_dims(s, orient)
    grid[r : r + h, c : c + w] = body
    bits = [0, 1] + [random.randrange(2) for _ in range(s - 2)]
    random.shuffle(bits)

    if orient == WIDE:
        rows = r + np.arange(s)
        marker_cols = c + s - 1 + np.asarray(bits)
        grid[rows, marker_cols] = marker
        for row, marker_c in zip(rows, marker_cols):
            expected[row, c - SHIFT_PADDING : marker_c - SHIFT_PADDING] = body
            expected[row, marker_c + SHIFT_PADDING + 1 : c + w + SHIFT_PADDING] = body
    else:
        cols = c + np.arange(s)
        marker_rows = r + s - 1 + np.asarray(bits)
        grid[marker_rows, cols] = marker
        for col, marker_r in zip(cols, marker_rows):
            expected[r - SHIFT_PADDING : marker_r - SHIFT_PADDING, col] = body
            expected[marker_r + SHIFT_PADDING + 1 : r + h + SHIFT_PADDING, col] = body


def _size_order(count: int) -> list[int]:
    sizes = list(SHORT_SIDE_SIZES)
    random.shuffle(sizes)
    if count >= DENSE_COUNT_THRESHOLD:
        return sorted(sizes, key=lambda s: (s > DENSE_PREFERRED_MAX_SHORT_SIDE, random.random()))
    if count >= PACKED_COUNT_THRESHOLD:
        return sorted(sizes, key=lambda s: (s > PACKED_PREFERRED_MAX_SHORT_SIDE, random.random()))
    return sizes


def _place_objects(n: int, count: int) -> list[ObjectSpec] | None:
    """Pick non-overlapping object specs; return None if this attempt cannot fit them."""
    originals: list[BBox] = []
    expanded: list[BBox] = []
    placed: list[ObjectSpec] = []

    for _ in range(count):
        choices: list[tuple[int, str, list[Location]]] = []
        for s in _size_order(count):
            for orient in random.sample(ORIENTATIONS, 2):
                h, w = _object_dims(s, orient)
                locs = _locations(n, h, w, originals, expanded)
                if locs:
                    choices.append((s, orient, locs))
        if not choices:
            return None

        s, orient, locs = random.choice(choices)
        r, c = random.choice(locs)
        h, w = _object_dims(s, orient)
        originals.append((r, c, r + h - 1, c + w - 1))
        expanded.append(_expanded_bbox(r, c, h, w))
        placed.append((r, c, s, orient))

    return placed


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(MIN_OBJECT_COUNT, MAX_OBJECT_COUNT)
    n = random.randint(MIN_GRID_SIZE_BY_COUNT[count], MAX_GRID_SIZE)

    colors = list(FOREGROUND_COLORS)
    body = random.choice(colors)
    colors.remove(body)
    marker = random.choice(colors)

    placed = _place_objects(n, count)
    if placed is None:
        return None

    grid = np.full((n, n), BACKGROUND_COLOR, dtype=int)
    expected = np.full((n, n), BACKGROUND_COLOR, dtype=int)
    for r, c, s, orient in placed:
        _draw_object(grid, expected, r, c, s, orient, body, marker)

    try:
        out = main(grid)
    except Exception:
        return None
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
