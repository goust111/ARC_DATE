from __future__ import annotations

import math

import numpy as np


BG = 0
ARC_MAX = 30
COLORS = tuple(range(1, 10))

RECT_COUNT_RANGE = (3, 5)
MAX_RECT_HEIGHT = 13
MAX_RECT_WIDTH = 13
PLACEMENT_ATTEMPTS = 200
GENERATION_RETRIES = 300

_RNG = np.random.default_rng()


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(lo, hi + 1))


def _extract_rectangles(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    H, W = g.shape

    rectangles: list[dict[str, int]] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != BG):
        coords = np.argwhere(g == color)
        assert coords.size > 0

        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        area = int(coords.shape[0])

        assert area == h * w, "each color must form a solid rectangle"
        assert bool(np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color)), "rectangle bbox must be fully filled"

        rr0 = max(0, r0 - 1)
        rr1 = min(H, r1 + 2)
        cc0 = max(0, c0 - 1)
        cc1 = min(W, c1 + 2)
        halo_area = int(np.count_nonzero(g[rr0:rr1, cc0:cc1] != BG))
        assert halo_area == area, "rectangles must not overlap or touch"

        rectangles.append(
            {
                "color": int(color),
                "r0": r0,
                "c0": c0,
                "height": h,
                "width": w,
                "area": area,
            }
        )

    assert len(rectangles) >= 2, "expected multiple rectangles"

    rectangles.sort(key=lambda item: item["area"])
    areas = [item["area"] for item in rectangles]
    heights = [item["height"] for item in rectangles]
    widths = [item["width"] for item in rectangles]

    assert len(areas) == len(set(areas)), "rectangle areas must be unique"
    assert len(heights) == len(set(heights)), "rectangle heights must be unique"
    assert len(widths) == len(set(widths)), "rectangle widths must be unique"

    for inner, outer in zip(rectangles, rectangles[1:]):
        assert outer["height"] > inner["height"], "nested rectangles need strictly increasing heights"
        assert outer["width"] > inner["width"], "nested rectangles need strictly increasing widths"

    row_parity = rectangles[0]["height"] % 2
    col_parity = rectangles[0]["width"] % 2
    assert all(item["height"] % 2 == row_parity for item in rectangles), "all heights must share one center parity"
    assert all(item["width"] % 2 == col_parity for item in rectangles), "all widths must share one center parity"

    return rectangles


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered solid rectangles into concentric nested rectangles centered on the smallest one.

    Input structure:
    - The background is Black.
    - Each non-Black color forms one solid monochrome rectangle.
    - Rectangles are pairwise separated, have distinct colors, distinct widths, distinct heights, and distinct areas.

    Rule:
    - Sort the rectangles by area from largest to smallest.
    - Use the smallest rectangle's original center as the common alignment anchor.
    - Redraw the rectangles on top of one another with the shared center, painting the largest first and the smallest last.
    - Crop the result to the minimal bounding box of the largest rectangle.

    Key insight:
    - The input positions are discarded except for the smallest rectangle's center, which determines the common center used to build a concentric stack of rectangular color rings.
    """
    rectangles = _extract_rectangles(np.asarray(grid, dtype=int))
    anchor = rectangles[0]
    outer = rectangles[-1]

    # Doubled coordinates preserve exact centers for both odd and even side lengths.
    anchor_r2 = 2 * anchor["r0"] + anchor["height"]
    anchor_c2 = 2 * anchor["c0"] + anchor["width"]
    outer_top = (anchor_r2 - outer["height"]) // 2
    outer_left = (anchor_c2 - outer["width"]) // 2

    out = np.full((outer["height"], outer["width"]), BG, dtype=int)
    for item in sorted(rectangles, key=lambda rect: rect["area"], reverse=True):
        assert (anchor_r2 - item["height"]) % 2 == 0
        assert (anchor_c2 - item["width"]) % 2 == 0

        abs_r = (anchor_r2 - item["height"]) // 2
        abs_c = (anchor_c2 - item["width"]) // 2
        r = abs_r - outer_top
        c = abs_c - outer_left

        h = item["height"]
        w = item["width"]
        assert 0 <= r <= out.shape[0] - h
        assert 0 <= c <= out.shape[1] - w
        out[r : r + h, c : c + w] = item["color"]

    return out


def _parity_values(max_dim: int, parity: int) -> list[int]:
    start = 1 if parity == 1 else 2
    return list(range(start, max_dim + 1, 2))


def _sample_size_chain(count: int) -> list[tuple[int, int]]:
    for _ in range(80):
        height_parity = _randint(0, 1)
        width_parity = _randint(0, 1)

        height_values = _parity_values(MAX_RECT_HEIGHT, height_parity)
        width_values = _parity_values(MAX_RECT_WIDTH, width_parity)
        if len(height_values) < count or len(width_values) < count:
            continue

        heights = sorted(int(v) for v in _RNG.choice(height_values, size=count, replace=False).tolist())
        widths = sorted(int(v) for v in _RNG.choice(width_values, size=count, replace=False).tolist())
        dims = list(zip(heights, widths))

        if any(h == w for h, w in dims):
            continue
        return dims

    raise RuntimeError("failed to sample a valid rectangle chain")


def _choose_canvas(dims: list[tuple[int, int]]) -> tuple[int, int] | None:
    max_h = max(h for h, _w in dims)
    max_w = max(w for _h, w in dims)
    halo_cells = sum((h + 2) * (w + 2) for h, w in dims)

    min_h = max(max_h + 2, int(math.ceil(math.sqrt(halo_cells))))
    min_w = max(max_w + 2, int(math.ceil(halo_cells / min_h)))
    if min_h > ARC_MAX or min_w > ARC_MAX:
        return None

    hi_h = min(ARC_MAX, min_h + 6)
    hi_w = min(ARC_MAX, min_w + 6)
    H = _randint(min_h, hi_h)
    W = _randint(min_w, hi_w)
    if H * W < halo_cells:
        return None
    return H, W


def _place_rectangles(dims: list[tuple[int, int]], colors: list[int], shape: tuple[int, int]) -> np.ndarray | None:
    H, W = shape
    grid = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    order = [int(i) for i in _RNG.permutation(len(dims))]
    for idx in order:
        h, w = dims[idx]
        color = int(colors[idx])

        row_lo, row_hi = 1, H - h - 1
        col_lo, col_hi = 1, W - w - 1
        if row_lo > row_hi or col_lo > col_hi:
            return None

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r = _randint(row_lo, row_hi)
            c = _randint(col_lo, col_hi)
            if occupied[r - 1 : r + h + 1, c - 1 : c + w + 1].any():
                continue

            grid[r : r + h, c : c + w] = color
            occupied[r : r + h, c : c + w] = True
            placed = True
            break

        if not placed:
            return None

    return grid


def _gen_attempt() -> np.ndarray | None:
    count = _randint(*RECT_COUNT_RANGE)
    dims = _sample_size_chain(count)
    canvas = _choose_canvas(dims)
    if canvas is None:
        return None

    colors = [int(c) for c in _RNG.choice(COLORS, size=count, replace=False).tolist()]
    grid = _place_rectangles(dims, colors, canvas)
    if grid is None:
        return None

    rectangles = _extract_rectangles(grid)
    if len(rectangles) != count:
        return None

    out = main(grid)
    if max(grid.shape) > ARC_MAX or max(out.shape) > ARC_MAX:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
