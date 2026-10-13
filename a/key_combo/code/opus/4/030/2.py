from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox, components_mask, flood_fill, get_bg


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


GEN_OPENINGS = ("right", "left", "top", "bottom")
GEN_COLOR_POOL = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
GEN_MAX_ATTEMPTS = 400
GEN_PATTERN_SIZE_RANGE = (4, 8)
_GEN_CALL_COUNT = 0


def _outline_bbox(grid: np.ndarray) -> BBox:
    mask = grid == Color.GREY
    box = bbox(mask)
    assert box is not None, "expected a grey outline"
    return box


def _detect_opening(grid: np.ndarray) -> tuple[str, BBox]:
    box = _outline_bbox(grid)
    r0, c0, r1, c1 = box
    crop = (grid[r0 : r1 + 1, c0 : c1 + 1] == Color.GREY)
    h, w = crop.shape

    expected = np.zeros_like(crop, dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True

    assert np.all(crop <= expected), "outline must stay on its perimeter"
    missing = np.argwhere(expected & ~crop)
    assert 2 <= len(missing) <= 3, "opening must be a 2-3 pixel gap"

    rows = missing[:, 0]
    cols = missing[:, 1]
    if np.all(rows == 0):
        opening = "top"
        coords = sorted(int(c) for c in cols)
        limit = w
    elif np.all(rows == h - 1):
        opening = "bottom"
        coords = sorted(int(c) for c in cols)
        limit = w
    elif np.all(cols == 0):
        opening = "left"
        coords = sorted(int(r) for r in rows)
        limit = h
    elif np.all(cols == w - 1):
        opening = "right"
        coords = sorted(int(r) for r in rows)
        limit = h
    else:
        raise AssertionError("opening must lie on exactly one side")

    assert coords == list(range(coords[0], coords[0] + len(coords))), "opening must be contiguous"
    assert 0 < coords[0] and coords[-1] < limit - 1, "opening must not remove corners"
    return opening, box


def _inner_bbox(box: BBox) -> BBox:
    r0, c0, r1, c1 = box
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "outline must have an interior"
    return r0 + 1, c0 + 1, r1 - 1, c1 - 1


def _pattern_mask(grid: np.ndarray) -> np.ndarray:
    return (grid != Color.BLACK) & (grid != Color.GREY)


def _mirror_coord(coord: Coord, axis: str, inner: BBox) -> Coord:
    r, c = coord
    ir0, ic0, ir1, ic1 = inner
    if axis == "vertical":
        return r, ic0 + ic1 - c
    if axis == "horizontal":
        return ir0 + ir1 - r, c
    raise ValueError(f"unknown axis: {axis}")


def _axis_for_opening(opening: str) -> str:
    return "vertical" if opening in {"left", "right"} else "horizontal"


def _apply_axis_completion(grid: np.ndarray, axis: str, box: BBox | None = None) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    if box is None:
        box = _outline_bbox(out)
    inner = _inner_bbox(box)
    ir0, ic0, ir1, ic1 = inner

    pattern = np.argwhere(_pattern_mask(out))
    assert len(pattern) > 0, "expected an internal colored pattern"

    for rr, cc in pattern:
        r = int(rr)
        c = int(cc)
        tr, tc = _mirror_coord((r, c), axis, inner)
        if out[tr, tc] != Color.GREY:
            out[tr, tc] = int(out[r, c])

    interior = out[ir0 : ir1 + 1, ic0 : ic1 + 1].copy()
    while True:
        zeros = np.argwhere(interior == Color.BLACK)
        if len(zeros) == 0:
            break
        flood_fill(interior, int(zeros[0, 0]), int(zeros[0, 1]), Color.GREY, connectivity=4)
    out[ir0 : ir1 + 1, ic0 : ic1 + 1] = interior
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the cup interior using the opening side to choose the symmetry axis, then flood-fill the remaining cavity.

    Input structure:
    - The background is black.
    - One grey one-pixel-thick rectangular cup outline appears, with a contiguous 2-3 pixel opening on exactly one side.
    - Inside the outline, only the half opposite that opening contains a small 4-connected multicolor pattern that stays at least one cell away from the outline.

    Rule:
    - Detect which side of the grey outline contains the opening.
    - Use a vertical interior midline for left/right openings and a horizontal interior midline for top/bottom openings.
    - Reflect every colored pattern pixel across that axis, preserving its color, then fill every remaining black cell inside the outline's interior box with grey.
    - Keep the exterior background and the original grey outline unchanged.

    Key insight:
    - The opening direction does not say where to draw; it selects which symmetry axis the incomplete interior pattern must use before the leftover interior void is grey-filled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert int(get_bg(g)) == Color.BLACK, "expected black background"

    opening, box = _detect_opening(g)
    return _apply_axis_completion(g, _axis_for_opening(opening), box)


def _safe_source_cells(box: BBox, opening: str) -> list[Coord]:
    r0, c0, r1, c1 = box
    rows = list(range(r0 + 2, r1 - 1))
    cols = list(range(c0 + 2, c1 - 1))
    center_r2 = (r0 + 1) + (r1 - 1)
    center_c2 = (c0 + 1) + (c1 - 1)

    if opening == "right":
        cols = [c for c in cols if 2 * c < center_c2]
    elif opening == "left":
        cols = [c for c in cols if 2 * c > center_c2]
    elif opening == "top":
        rows = [r for r in rows if 2 * r > center_r2]
    elif opening == "bottom":
        rows = [r for r in rows if 2 * r < center_r2]
    else:
        raise ValueError(f"unknown opening: {opening}")

    return [(r, c) for r in rows for c in cols]


def _neighbors4(coord: Coord) -> list[Coord]:
    r, c = coord
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _grow_connected_shape(allowed: set[Coord], size: int) -> list[Coord] | None:
    if len(allowed) < size:
        return None

    start = random.choice(tuple(allowed))
    cells = {start}
    cursor = start
    while len(cells) < size:
        frontier = {
            nxt
            for cell in cells
            for nxt in _neighbors4(cell)
            if nxt in allowed and nxt not in cells
        }
        if not frontier:
            return None
        walk = [nxt for nxt in _neighbors4(cursor) if nxt in frontier]
        nxt = random.choice(walk) if walk and random.random() < 0.6 else random.choice(tuple(frontier))
        cells.add(nxt)
        cursor = nxt
    return list(cells)


def _assign_colors(coords: list[Coord]) -> dict[Coord, int]:
    n_colors = 2 if len(coords) <= 4 else random.choice((2, 3))
    palette = random.sample(GEN_COLOR_POOL, n_colors)
    counts = [1] * n_colors
    for _ in range(len(coords) - n_colors):
        counts[random.randrange(n_colors)] += 1
    color_list = []
    for color, count in zip(palette, counts):
        color_list.extend([color] * count)
    random.shuffle(color_list)
    return {coord: color for coord, color in zip(coords, color_list)}


def _paint_outline_with_gap(grid: np.ndarray, box: BBox, opening: str) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = Color.GREY
    grid[r1, c0 : c1 + 1] = Color.GREY
    grid[r0 : r1 + 1, c0] = Color.GREY
    grid[r0 : r1 + 1, c1] = Color.GREY

    gap = random.randint(2, 3)
    if opening in {"top", "bottom"}:
        start = random.randint(c0 + 1, c1 - gap)
        row = r0 if opening == "top" else r1
        grid[row, start : start + gap] = Color.BLACK
    else:
        start = random.randint(r0 + 1, r1 - gap)
        col = c0 if opening == "left" else c1
        grid[start : start + gap, col] = Color.BLACK


def _build_input(opening: str) -> np.ndarray | None:
    inner_h = random.randint(5, 7)
    inner_w = random.randint(5, 7)
    outer_h, outer_w = inner_h + 2, inner_w + 2
    height = random.randint(outer_h + 2, outer_h + 6)
    width = random.randint(outer_w + 2, outer_w + 6)

    grid = np.full((height, width), Color.BLACK, dtype=int)
    r0 = random.randint(1, height - outer_h - 1)
    c0 = random.randint(1, width - outer_w - 1)
    box = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
    _paint_outline_with_gap(grid, box, opening)

    allowed = _safe_source_cells(box, opening)
    coords = _grow_connected_shape(set(allowed), random.randint(*GEN_PATTERN_SIZE_RANGE))
    if coords is None:
        return None

    for (r, c), color in _assign_colors(coords).items():
        grid[r, c] = color
    return grid


def _pattern_semantics(grid: np.ndarray, opening: str, box: BBox) -> bool:
    mask = _pattern_mask(grid)
    comps = components_mask(mask, connectivity=4)
    if len(comps) != 1:
        return False
    coords = [tuple(map(int, rc)) for rc in comps[0]]
    if not (4 <= len(coords) <= 8):
        return False
    colors = sorted(set(int(grid[r, c]) for r, c in coords))
    if len(colors) not in {2, 3}:
        return False

    allowed = set(_safe_source_cells(box, opening))
    if any(coord not in allowed for coord in coords):
        return False

    axis = _axis_for_opening(opening)
    inner = _inner_bbox(box)
    mirrored = {_mirror_coord(coord, axis, inner) for coord in coords}
    if mirrored & set(coords):
        return False
    if any(int(grid[r, c]) != Color.BLACK for r, c in mirrored):
        return False
    return True


def _fill_only(grid: np.ndarray, box: BBox) -> np.ndarray:
    out = grid.copy()
    ir0, ic0, ir1, ic1 = _inner_bbox(box)
    interior = out[ir0 : ir1 + 1, ic0 : ic1 + 1]
    interior[interior == Color.BLACK] = Color.GREY
    out[ir0 : ir1 + 1, ic0 : ic1 + 1] = interior
    return out


def _needs_fill_after_mirror(grid: np.ndarray, opening: str, box: BBox) -> bool:
    mirrored = _apply_axis_completion(grid, _axis_for_opening(opening), box)
    ir0, ic0, ir1, ic1 = _inner_bbox(box)
    return np.any(mirrored[ir0 : ir1 + 1, ic0 : ic1 + 1] == Color.GREY)


def _mirror_without_fill(grid: np.ndarray, opening: str, box: BBox) -> np.ndarray:
    out = grid.copy()
    axis = _axis_for_opening(opening)
    inner = _inner_bbox(box)
    for rr, cc in np.argwhere(_pattern_mask(out)):
        r = int(rr)
        c = int(cc)
        tr, tc = _mirror_coord((r, c), axis, inner)
        if out[tr, tc] != Color.GREY:
            out[tr, tc] = int(out[r, c])
    return out


def _semantic_checks(grid: np.ndarray, opening: str) -> bool:
    try:
        detected, box = _detect_opening(grid)
        if detected != opening:
            return False
        if not _pattern_semantics(grid, opening, box):
            return False

        expected = _apply_axis_completion(grid, _axis_for_opening(opening), box)
        solved = main(grid)
        if not np.array_equal(solved, expected):
            return False
        if np.array_equal(expected, grid):
            return False

        mirror_only = _mirror_without_fill(grid, opening, box)
        if np.array_equal(mirror_only, expected):
            return False
        if not np.any((mirror_only == Color.BLACK) & (expected == Color.GREY)):
            return False

        fill_only = _fill_only(grid, box)
        if np.array_equal(fill_only, expected):
            return False
        if not np.any((_pattern_mask(expected)) & (fill_only == Color.GREY)):
            return False

        wrong_axis = "horizontal" if _axis_for_opening(opening) == "vertical" else "vertical"
        if np.array_equal(_apply_axis_completion(grid, wrong_axis, box), expected):
            return False

        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    opening = GEN_OPENINGS[_GEN_CALL_COUNT % len(GEN_OPENINGS)]
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _build_input(opening)
        if grid is not None and _semantic_checks(grid, opening):
            _GEN_CALL_COUNT += 1
            return grid
    raise RuntimeError("failed to generate a valid cup-mirror instance")
