from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
ORANGE = 7
RECT_COLORS = (1, 2, 3, 4, 6, 8, 9)
LAYER_LENGTHS = (1, 2, 3, 4, 5)

MIN_RECT_COUNT = 2
MAX_RECT_COUNT = 5
MIN_RECT_HEIGHT = 5
MAX_RECT_HEIGHT = 9
MIN_RECT_WIDTH = 5
MAX_RECT_WIDTH = 9

MIN_MARGIN = 1
MAX_MARGIN = 4
MAX_GRID_SIDE = 28
MAX_GEN_TRIES = 500


def _find_anchors(grid: np.ndarray) -> list[dict[str, int]]:
    mask = grid == ORANGE
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    anchors: list[dict[str, int]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = {rr for rr, _ in cells}
            assert len(rows) == 1, "each orange marker must be horizontal"
            row = rows.pop()
            cols = sorted(cc for _, cc in cells)
            assert cols == list(range(cols[0], cols[0] + len(cols))), "marker pixels must be contiguous"
            anchors.append({"row": int(row), "col": int(cols[0]), "length": int(len(cols))})

    return anchors


def _recover_rectangles(grid: np.ndarray) -> list[dict[str, int]]:
    anchors = _find_anchors(grid)
    assert MIN_RECT_COUNT <= len(anchors) <= MAX_RECT_COUNT

    used_layers: set[int] = set()
    rectangles: list[dict[str, int]] = []

    for anchor in anchors:
        layer = int(anchor["length"])
        assert layer in LAYER_LENGTHS, "anchor length must be in 1..5"
        assert layer not in used_layers, "anchor lengths must be unique"
        used_layers.add(layer)

        r0 = int(anchor["row"]) - 1
        c0 = int(anchor["col"]) - 1
        assert r0 >= 0 and c0 >= 0, "anchor must sit one cell inside the rectangle"

        row_hi = min(grid.shape[0], r0 + 2)
        col_hi = min(grid.shape[1], c0 + layer + 2)
        window = grid[r0:row_hi, c0:col_hi]
        colors = sorted(int(v) for v in np.unique(window) if int(v) not in {BG, GREY, ORANGE})
        assert len(colors) == 1, "anchor neighborhood must reveal exactly one rectangle color"
        color = int(colors[0])

        coords = np.argwhere(grid == color)
        assert coords.size > 0
        top, left = map(int, coords.min(axis=0))
        bottom, right = map(int, coords.max(axis=0))
        assert top == r0 and left == c0, "color bbox must start at the anchor-implied corner"
        assert right - c0 + 1 >= layer + 1, "rectangle must be wide enough to contain its anchor"
        assert bottom - r0 + 1 >= 3, "rectangle must be at least three cells tall"

        patch = grid[r0 : bottom + 1, c0 : right + 1]
        allowed = np.isin(patch, [color, GREY, ORANGE])
        assert bool(allowed.all()), "rectangle interior may only contain its color, grey overlaps, or orange anchor"

        anchor_cols = slice(c0 + 1, c0 + 1 + layer)
        assert np.all(grid[r0 + 1, anchor_cols] == ORANGE), "anchor must match the inferred rectangle"

        rectangles.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(bottom),
                "c1": int(right),
                "color": int(color),
                "layer": int(layer),
            }
        )

    return rectangles


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve grey overlap cells by painting each rectangle in ascending anchor-length order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rectangles = _recover_rectangles(g)
    out = np.full_like(g, BG)
    for rect in sorted(rectangles, key=lambda rect: rect["layer"]):
        out[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = int(rect["color"])
    return out


def _anchor_cells(rect: dict[str, int]) -> list[tuple[int, int]]:
    row = int(rect["r0"]) + 1
    start = int(rect["c0"]) + 1
    length = int(rect["layer"])
    return [(row, start + dc) for dc in range(length)]


def _rect_coverage(rect: dict[str, int], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = True
    return mask


def _render_pair(rectangles: list[dict[str, int]], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = shape
    coverages = [_rect_coverage(rect, shape) for rect in rectangles]
    counts = np.zeros(shape, dtype=int)
    for mask in coverages:
        counts += mask.astype(int)

    if int((counts >= 2).sum()) == 0:
        return None

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

    for rect, mask in zip(rectangles, coverages):
        inp[(counts == 1) & mask] = int(rect["color"])

    inp[counts >= 2] = GREY

    for rect in rectangles:
        for r, c in _anchor_cells(rect):
            if counts[r, c] != 1:
                return None
            inp[r, c] = ORANGE

        if not np.any(counts[rect["r1"], rect["c0"] : rect["c1"] + 1] == 1):
            return None
        if not np.any(counts[rect["r0"] : rect["r1"] + 1, rect["c1"]] == 1):
            return None

    for rect in sorted(rectangles, key=lambda rect: rect["layer"]):
        out[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = int(rect["color"])

    for rect in rectangles:
        coords = np.argwhere(inp == rect["color"])
        if coords.size == 0:
            return None
        top, left = map(int, coords.min(axis=0))
        bottom, right = map(int, coords.max(axis=0))
        if (top, left, bottom, right) != (rect["r0"], rect["c0"], rect["r1"], rect["c1"]):
            return None

    return inp, out


def _make_first_rect(layer: int, color: int) -> dict[str, int]:
    height = random.randint(MIN_RECT_HEIGHT, MAX_RECT_HEIGHT)
    width = random.randint(max(MIN_RECT_WIDTH, layer + 2), MAX_RECT_WIDTH)
    return {
        "r0": 0,
        "c0": 0,
        "r1": height - 1,
        "c1": width - 1,
        "color": int(color),
        "layer": int(layer),
    }


def _append_vertical(prev: dict[str, int], layer: int, color: int) -> dict[str, int] | None:
    row_lo = int(prev["r0"]) + 2
    row_hi = int(prev["r1"]) - 1
    if row_lo > row_hi:
        return None

    r0 = random.randint(row_lo, row_hi)
    c0 = int(prev["c1"])
    r1 = int(prev["r1"]) + random.randint(2, 5)
    width = random.randint(max(MIN_RECT_WIDTH, layer + 2), MAX_RECT_WIDTH)
    c1 = c0 + width - 1

    return {
        "r0": int(r0),
        "c0": int(c0),
        "r1": int(r1),
        "c1": int(c1),
        "color": int(color),
        "layer": int(layer),
    }


def _append_horizontal(prev: dict[str, int], layer: int, color: int) -> dict[str, int] | None:
    col_lo = int(prev["c0"]) + int(prev["layer"]) + 1
    col_hi = int(prev["c1"])
    if col_lo > col_hi:
        return None

    c0 = random.randint(col_lo, col_hi)
    r0 = int(prev["r1"])
    c1 = int(prev["c1"]) + random.randint(2, 5)
    height = random.randint(MIN_RECT_HEIGHT, MAX_RECT_HEIGHT)
    r1 = r0 + height - 1

    return {
        "r0": int(r0),
        "c0": int(c0),
        "r1": int(r1),
        "c1": int(c1),
        "color": int(color),
        "layer": int(layer),
    }


def _shift_rectangles(rectangles: list[dict[str, int]]) -> tuple[list[dict[str, int]], tuple[int, int]] | None:
    max_r = max(int(rect["r1"]) for rect in rectangles)
    max_c = max(int(rect["c1"]) for rect in rectangles)

    top = random.randint(MIN_MARGIN, MAX_MARGIN)
    left = random.randint(MIN_MARGIN, MAX_MARGIN)
    bottom = random.randint(MIN_MARGIN, MAX_MARGIN)
    right = random.randint(MIN_MARGIN, MAX_MARGIN)

    height = top + max_r + 1 + bottom
    width = left + max_c + 1 + right
    if height > MAX_GRID_SIDE or width > MAX_GRID_SIDE:
        return None

    shifted: list[dict[str, int]] = []
    for rect in rectangles:
        shifted.append(
            {
                "r0": int(rect["r0"]) + top,
                "c0": int(rect["c0"]) + left,
                "r1": int(rect["r1"]) + top,
                "c1": int(rect["c1"]) + left,
                "color": int(rect["color"]),
                "layer": int(rect["layer"]),
            }
        )

    return shifted, (height, width)


def _gen_attempt() -> np.ndarray | None:
    rect_count = random.randint(MIN_RECT_COUNT, MAX_RECT_COUNT)
    layers = random.sample(LAYER_LENGTHS, rect_count)
    colors = random.sample(RECT_COLORS, rect_count)
    patterns = [random.choice(("V", "H")) for _ in range(rect_count - 1)]

    rectangles = [_make_first_rect(layers[0], colors[0])]
    for layer, color, pattern in zip(layers[1:], colors[1:], patterns):
        prev = rectangles[-1]
        rect = _append_vertical(prev, layer, color) if pattern == "V" else _append_horizontal(prev, layer, color)
        if rect is None:
            return None
        rectangles.append(rect)

    shifted = _shift_rectangles(rectangles)
    if shifted is None:
        return None
    rectangles, shape = shifted

    rendered = _render_pair(rectangles, shape)
    if rendered is None:
        return None
    inp, out = rendered

    try:
        if np.array_equal(main(inp), out):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid overlap-filter puzzle")
