from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GRID_SIDE_MIN = 12
GRID_SIDE_MAX = 24
RECT_COUNT_MIN = 3
RECT_COUNT_MAX = 5
RECT_SIDE_MAX = 7
SPEC_RETRIES = 200
PLACEMENT_RETRIES = 300
GEN_RETRIES = 400


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _extract_rectangles(grid: np.ndarray) -> list[dict[str, int]]:
    bg = _mode_color(grid)
    rects: list[dict[str, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        assert coords.size > 0
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        patch = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        assert coords.shape[0] == h * w
        assert bool((patch == color).all())
        rects.append(
            {
                "color": color,
                "r0": int(r0),
                "c0": int(c0),
                "h": h,
                "w": w,
                "area": h * w,
            }
        )

    assert RECT_COUNT_MIN <= len(rects) <= RECT_COUNT_MAX
    areas = [rect["area"] for rect in rects]
    assert len(areas) == len(set(areas))
    return rects


def _build_nested(rects: list[dict[str, int]]) -> np.ndarray:
    ordered = sorted(rects, key=lambda rect: rect["area"])
    layers = len(ordered) - 1
    core = ordered[0]
    out_h = core["h"] + 2 * layers
    out_w = core["w"] + 2 * layers
    out = np.full((out_h, out_w), int(ordered[-1]["color"]), dtype=int)

    for idx in range(len(ordered) - 2, 0, -1):
        inset = len(ordered) - 1 - idx
        out[inset : out_h - inset, inset : out_w - inset] = int(ordered[idx]["color"])

    out[layers : layers + core["h"], layers : layers + core["w"]] = int(core["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Nest scattered solid rectangles by area, keeping the smallest rectangle as the core.

    The input contains 3 to 5 non-overlapping monochrome filled rectangles on a black
    square background. Rectangle colors are distinct and their areas are pairwise distinct.
    The output ignores the input positions: it centers the smallest rectangle unchanged,
    then wraps it with 1-pixel rectangular border layers in ascending area order, using
    each larger rectangle only for its color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    rects = _extract_rectangles(g)
    return _build_nested(rects)


def _sample_core_dims(max_side: int) -> tuple[int, int] | None:
    hi = min(max_side, 5)
    if hi < 1:
        return None
    for _ in range(SPEC_RETRIES):
        if random.random() < 0.7 and hi >= 2:
            short_hi = min(3, hi)
            long_lo = min(2, hi)
            h = random.randint(1, short_hi)
            w = random.randint(long_lo, hi)
            if random.random() < 0.5:
                h, w = w, h
            if h != w:
                return h, w
        else:
            h = random.randint(1, hi)
            w = random.randint(1, hi)
            if h * w <= 12:
                return h, w
    return None


def _sample_other_dims(core_area: int, used_areas: set[int], max_side: int) -> tuple[int, int] | None:
    for _ in range(SPEC_RETRIES):
        if random.random() < 0.55:
            short_hi = min(3, max_side)
            long_lo = min(3, max_side)
            long = random.randint(long_lo, max_side)
            short = random.randint(1, short_hi)
            h, w = (short, long) if random.random() < 0.5 else (long, short)
        else:
            h = random.randint(1, max_side)
            w = random.randint(1, max_side)
        area = h * w
        if area <= core_area or area in used_areas:
            continue
        return h, w
    return None


def _sample_rectangles(side: int, n_rects: int) -> list[dict[str, int]] | None:
    max_side = min(RECT_SIDE_MAX, side - 2)
    if max_side < 2:
        return None

    core_dims = _sample_core_dims(max_side)
    if core_dims is None:
        return None
    core_h, core_w = core_dims
    used_areas = {core_h * core_w}
    specs = [{"h": core_h, "w": core_w, "area": core_h * core_w}]

    while len(specs) < n_rects:
        dims = _sample_other_dims(specs[0]["area"], used_areas, max_side)
        if dims is None:
            return None
        h, w = dims
        specs.append({"h": h, "w": w, "area": h * w})
        used_areas.add(h * w)

    if sum(spec["area"] for spec in specs) > (side * side) // 3:
        return None

    colors = random.sample(PALETTE, n_rects)
    random.shuffle(colors)
    for spec, color in zip(specs, colors):
        spec["color"] = color
    return specs


def _place_rectangles(side: int, rects: list[dict[str, int]]) -> np.ndarray | None:
    grid = np.full((side, side), BG, dtype=int)
    blocked = np.zeros((side, side), dtype=bool)

    for rect in sorted(rects, key=lambda rect: rect["area"], reverse=True):
        h = int(rect["h"])
        w = int(rect["w"])
        if side - h - 1 < 1 or side - w - 1 < 1:
            return None
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(1, side - h - 1)
            c0 = random.randint(1, side - w - 1)
            rr0 = r0 - 1
            rr1 = r0 + h + 1
            cc0 = c0 - 1
            cc1 = c0 + w + 1
            if bool(blocked[rr0:rr1, cc0:cc1].any()):
                continue
            grid[r0 : r0 + h, c0 : c0 + w] = int(rect["color"])
            blocked[rr0:rr1, cc0:cc1] = True
            rect["r0"] = r0
            rect["c0"] = c0
            placed = True
            break
        if not placed:
            return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    side = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    n_rects = random.randint(RECT_COUNT_MIN, RECT_COUNT_MAX)
    rects = _sample_rectangles(side, n_rects)
    if rects is None:
        return None

    grid = _place_rectangles(side, rects)
    if grid is None:
        return None

    expected = _build_nested(rects)
    try:
        got = main(grid)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None

    color_counts = Counter(map(int, grid.flat))
    assert color_counts[BG] > max(count for color, count in color_counts.items() if color != BG)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
