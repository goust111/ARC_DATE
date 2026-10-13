from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 8  # ARC azure
PALETTE = tuple(color for color in range(10) if color != BG)

# Generator configuration.
GRID_H_RANGE = (11, 24)
GRID_W_RANGE = (11, 24)
RECT_COUNT_RANGE = (2, 5)
MAX_RECT_HEIGHT = 11
MAX_RECT_WIDTH = 11
OVERLAP_PROB = 0.5
GEN_RETRIES = 400
PLACEMENT_RETRIES = 400


def _bresenham(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    """Return the rasterized line between two endpoints, inclusive."""
    points: list[tuple[int, int]] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr
    r, c = r0, c0
    while True:
        points.append((r, c))
        if r == r1 and c == c1:
            return points
        err2 = err * 2
        if err2 > -dr:
            err -= dr
            c += sc
        if err2 < dc:
            err += dc
            r += sr


def _pattern_points(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0, r1 + 1):
        pts.add((r, c0))
        pts.add((r, c1))
    pts.update(_bresenham(r0, c0, r1, c1))
    pts.update(_bresenham(r0, c1, r1, c0))
    return pts


def _corners_to_rectangle(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = {tuple(map(int, pt)) for pt in points}
    assert len(pts) == 4, "each foreground color must occupy exactly four distinct cells"

    rows = sorted({r for r, _ in pts})
    cols = sorted({c for _, c in pts})
    assert len(rows) == 2 and len(cols) == 2, "four points must use exactly two rows and two columns"

    r0, r1 = rows
    c0, c1 = cols
    assert r0 < r1 and c0 < c1, "rectangle corners must be distinct"
    expected = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    assert pts == expected, "pixels must be the four corners of an axis-aligned rectangle"
    return r0, c0, r1, c1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect four-corner virtual rectangles and draw their borders plus both diagonals.

    Rectangles are processed from larger visible area to smaller visible area so
    later, smaller rectangles can overwrite earlier strokes where they overlap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    rectangles: list[tuple[int, int, int, int, int, int]] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != BG):
        points = np.argwhere(g == color)
        assert len(points) == 4, "each non-background color must appear exactly four times"
        r0, c0, r1, c1 = _corners_to_rectangle(map(tuple, points))
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        rectangles.append((area, r0, c0, r1, c1, color))

    rectangles.sort(key=lambda item: (-item[0], item[1], item[2], item[3], item[4]))

    for _area, r0, c0, r1, c1, color in rectangles:
        for r, c in _pattern_points(r0, c0, r1, c1):
            out[r, c] = color
    return out


def _sample_dimensions(rng: random.Random, h: int, w: int, force_large: bool) -> tuple[int, int] | None:
    max_h = min(MAX_RECT_HEIGHT, h - 1)
    max_w = min(MAX_RECT_WIDTH, w - 1)
    if max_h < 2 or max_w < 2:
        return None

    for _ in range(80):
        mode = rng.choice(("square", "wide", "tall", "generic"))
        if mode == "square":
            low = 5 if force_large else 3
            low = min(low, min(max_h, max_w))
            if low < 2:
                low = 2
            side = rng.randint(low, min(max_h, max_w))
            rect_h = side
            rect_w = side
        elif mode == "wide":
            h_low = 3 if force_large else 2
            h_high = min(max_h, 6 if force_large else 7)
            if h_low > h_high:
                continue
            rect_h = rng.randint(h_low, h_high)
            w_low = max(rect_h + 1, 5 if force_large else 4)
            if w_low > max_w:
                continue
            rect_w = rng.randint(w_low, max_w)
        elif mode == "tall":
            w_low = 3 if force_large else 2
            w_high = min(max_w, 6 if force_large else 7)
            if w_low > w_high:
                continue
            rect_w = rng.randint(w_low, w_high)
            h_low = max(rect_w + 1, 5 if force_large else 4)
            if h_low > max_h:
                continue
            rect_h = rng.randint(h_low, max_h)
        else:
            h_low = 4 if force_large else 2
            w_low = 4 if force_large else 2
            if h_low > max_h or w_low > max_w:
                continue
            rect_h = rng.randint(h_low, max_h)
            rect_w = rng.randint(w_low, max_w)

        if rect_h == 2 and rect_w == 2:
            continue
        return rect_h, rect_w
    return None


def _random_rect(
    rng: random.Random,
    h: int,
    w: int,
    used_corners: set[tuple[int, int]],
    used_areas: set[int],
    existing_patterns: set[tuple[int, int]],
    require_overlap: bool,
    force_large: bool,
) -> dict[str, object] | None:
    for _ in range(PLACEMENT_RETRIES):
        dims = _sample_dimensions(rng, h, w, force_large)
        if dims is None:
            return None
        rect_h, rect_w = dims

        r0 = rng.randint(0, h - rect_h)
        c0 = rng.randint(0, w - rect_w)
        r1 = r0 + rect_h - 1
        c1 = c0 + rect_w - 1
        area = rect_h * rect_w
        if area in used_areas:
            continue

        corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        if corners & used_corners:
            continue

        pattern = _pattern_points(r0, c0, r1, c1)
        overlap = bool(pattern & existing_patterns)
        if require_overlap != overlap:
            continue

        return {
            "r0": r0,
            "c0": c0,
            "r1": r1,
            "c1": c1,
            "area": area,
            "corners": corners,
            "pattern": pattern,
        }
    return None


def _valid_input(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or not np.all((0 <= g) & (g <= 9)):
        return False

    colors = [int(c) for c in np.unique(g) if int(c) != BG]
    if not (RECT_COUNT_RANGE[0] <= len(colors) <= len(PALETTE)):
        return False

    areas: set[int] = set()
    for color in colors:
        points = np.argwhere(g == color)
        if len(points) != 4:
            return False
        try:
            r0, c0, r1, c1 = _corners_to_rectangle(map(tuple, points))
        except AssertionError:
            return False
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        if area in areas:
            return False
        areas.add(area)
        if len(_pattern_points(r0, c0, r1, c1) - {tuple(p) for p in points}) == 0:
            return False

    try:
        out = main(g)
    except AssertionError:
        return False
    return out.shape == g.shape and np.any(out != g)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)

    max_rects = min(RECT_COUNT_RANGE[1], len(PALETTE), max(RECT_COUNT_RANGE[0], (h * w) // 55))
    rect_count = rng.randint(RECT_COUNT_RANGE[0], max_rects)
    colors = rng.sample(PALETTE, rect_count)
    need_overlap = rect_count >= 2 and rng.random() < OVERLAP_PROB

    rects: list[dict[str, object]] = []
    used_corners: set[tuple[int, int]] = set()
    used_areas: set[int] = set()
    existing_patterns: set[tuple[int, int]] = set()
    overlap_done = False

    for idx, color in enumerate(colors):
        require_overlap = need_overlap and not overlap_done and idx > 0
        force_large = idx == 0 and need_overlap
        rect = _random_rect(
            rng,
            h,
            w,
            used_corners,
            used_areas,
            existing_patterns,
            require_overlap=require_overlap,
            force_large=force_large,
        )
        if rect is None:
            return None
        rect["color"] = color
        rects.append(rect)
        used_corners.update(rect["corners"])
        used_areas.add(int(rect["area"]))
        existing_patterns |= rect["pattern"]
        overlap_done = overlap_done or require_overlap

    if need_overlap and not overlap_done:
        return None

    for rect in rects:
        color = int(rect["color"])
        for r, c in rect["corners"]:
            grid[r, c] = color

    return grid if _valid_input(grid) else None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
