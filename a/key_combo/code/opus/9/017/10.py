from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
OBJ = 1
MARK = 2

MARKER_ARMS = ((-1, 0), (1, 0), (0, -1), (0, 1))
SMALL_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
)
BIG_SHAPES = (
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (1, 0), (2, 0), (3, 0)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 1), (2, 1), (2, 0)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)),
)

MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors(r: int, c: int, connectivity: int) -> tuple[tuple[int, int], ...]:
    if connectivity == 4:
        return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
    return (
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    )


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _marker_cells(center_r: int, center_c: int) -> set[tuple[int, int]]:
    return {(center_r + dr, center_c + dc) for dr, dc in MARKER_ARMS}


def _find_markers(grid: np.ndarray) -> list[tuple[int, int]]:
    comps = _components(grid == MARK, connectivity=8)
    centers: list[tuple[int, int]] = []
    covered: set[tuple[int, int]] = set()
    for coords in comps:
        assert coords.shape[0] == 4, "each red marker must have four arm pixels"
        r0, c0, r1, c1 = _bbox(coords)
        assert (r1 - r0, c1 - c0) == (2, 2), "each marker must occupy a 3x3 box"
        cr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        expected = _marker_cells(cr, cc)
        actual = {tuple(map(int, rc)) for rc in coords}
        assert actual == expected, "red components must be hollow plus markers"
        assert int(grid[cr, cc]) == BG, "marker center must stay background in the input"
        centers.append((cr, cc))
        covered |= expected

    assert len(centers) == 2, "expected exactly two red hollow markers"
    assert int(np.count_nonzero(grid == MARK)) == len(covered), "unexpected extra red pixels"
    centers.sort(key=lambda rc: rc[1])
    assert centers[0][0] == centers[1][0], "marker centers must share one row"
    assert centers[1][1] - centers[0][1] >= 4, "markers must leave a horizontal gap"
    row = centers[0][0]
    assert np.all(grid[row, centers[0][1] + 2 : centers[1][1] - 1] == BG), "connector gap must be empty"
    return centers


def _reflect_cells(cells: set[tuple[int, int]], axis_r: int) -> set[tuple[int, int]]:
    return {(2 * axis_r - r, c) for r, c in cells}


def _blue_components(grid: np.ndarray) -> list[dict]:
    parts = []
    for coords in _components(grid == OBJ, connectivity=4):
        cells = {tuple(map(int, rc)) for rc in coords}
        parts.append({"coords": coords, "cells": cells, "area": int(coords.shape[0])})
    return parts


def _build_expected(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    centers = _find_markers(g)
    axis_r = centers[0][0]
    left_c = centers[0][1]
    right_c = centers[1][1]

    out = g.copy()
    out[axis_r, left_c + 2 : right_c - 1] = MARK

    mirror_cells: set[tuple[int, int]] = set()
    blue_parts = _blue_components(g)
    assert blue_parts, "expected at least one blue component"
    for part in blue_parts:
        rows = {int(r) for r, _ in part["cells"]}
        assert max(rows) < axis_r, "all blue objects must be strictly above the axis row"
        area = part["area"]
        assert area != 3, "area-3 blue objects are intentionally excluded"
        if area <= 2:
            reflected = _reflect_cells(part["cells"], axis_r)
            for rr, cc in reflected:
                assert _in_bounds(g.shape, rr, cc), "reflected small object must stay inside the grid"
                assert rr > axis_r, "reflected small objects must land below the axis"
                assert int(g[rr, cc]) == BG, "reflected small object must land on empty background"
            assert mirror_cells.isdisjoint(reflected), "different reflected small objects must not overlap"
            mirror_cells |= reflected
        else:
            assert area >= 4, "blue objects must be either small (<=2) or large (>=4)"

    for rr, cc in mirror_cells:
        out[rr, cc] = OBJ
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect the two hollow markers to define a horizontal mirror axis, then reflect only the small blue objects across it.

    Input structure:
    - The background is black, with exactly two red hollow-plus markers made of the four arm cells of a 3x3 box.
    - The two marker centers lie on the same row with only empty background in the horizontal gap between them.
    - Blue 4-connected objects of mixed sizes lie strictly above that row, with small objects of area 1 or 2 and large objects of area at least 4.

    Rule:
    - Detect the two red hollow markers and use their shared center row as the axis row.
    - Paint the empty horizontal gap between the markers red, connecting only the adjacent marker pair on that row.
    - Find every blue 4-connected object and measure its area.
    - Keep every original blue object, but for each small object of area at most 2, copy its pixels to the row-reflected positions across the axis row.
    - Leave large blue objects unchanged, and preserve the markers, connector, and background everywhere else.

    Key insight:
    - The red connector is not just decoration: once it is inferred from the two hollow markers, its row becomes the mirror axis, and the blue objects are filtered by area before reflection.
    """
    return _build_expected(np.asarray(grid, dtype=int))


def _dilate8(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    h, w = shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _shape_size(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _translate(shape: tuple[tuple[int, int], ...], r0: int, c0: int) -> set[tuple[int, int]]:
    return {(r0 + r, c0 + c) for r, c in shape}


def _candidate_positions(
    grid_shape: tuple[int, int],
    axis_r: int,
    shape: tuple[tuple[int, int], ...],
    forbidden_original: set[tuple[int, int]],
    blocked_original: set[tuple[int, int]],
    blocked_mirror: set[tuple[int, int]],
) -> list[set[tuple[int, int]]]:
    h, w = grid_shape
    shape_h, shape_w = _shape_size(shape)
    candidates: list[set[tuple[int, int]]] = []
    for r0 in range(0, axis_r - shape_h + 1):
        for c0 in range(0, w - shape_w + 1):
            cells = _translate(shape, r0, c0)
            if cells & forbidden_original:
                continue
            if cells & blocked_original:
                continue
            reflected = _reflect_cells(cells, axis_r)
            if any(not _in_bounds(grid_shape, rr, cc) or rr <= axis_r for rr, cc in reflected):
                continue
            if reflected & blocked_mirror:
                continue
            candidates.append(cells)
    return candidates


def _render_input(
    h: int,
    w: int,
    axis_r: int,
    left_c: int,
    right_c: int,
    objects: list[set[tuple[int, int]]],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for center_c in (left_c, right_c):
        for rr, cc in _marker_cells(axis_r, center_c):
            grid[rr, cc] = MARK
    for cells in objects:
        for rr, cc in cells:
            grid[rr, cc] = OBJ
    return grid


def _analyze_instance(grid: np.ndarray) -> dict:
    centers = _find_markers(grid)
    axis_r = centers[0][0]
    blue_parts = _blue_components(grid)
    small_parts = [part for part in blue_parts if part["area"] <= 2]
    large_parts = [part for part in blue_parts if part["area"] >= 4]
    return {
        "centers": centers,
        "axis_r": axis_r,
        "blue_parts": blue_parts,
        "small_parts": small_parts,
        "large_parts": large_parts,
    }


def _validate_instance(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if set(map(int, np.unique(grid))) - {BG, OBJ, MARK}:
        return False

    try:
        info = _analyze_instance(grid)
    except AssertionError:
        return False

    axis_r = info["axis_r"]
    centers = info["centers"]
    left_c = centers[0][1]
    right_c = centers[1][1]

    if not (3 <= len(info["small_parts"]) <= 4):
        return False
    if not (1 <= len(info["large_parts"]) <= 2):
        return False
    if len(info["blue_parts"]) != len(info["small_parts"]) + len(info["large_parts"]):
        return False

    corridor = {
        (r, c)
        for r in range(axis_r)
        for c in range(left_c - 1, right_c + 2)
        if _in_bounds(grid.shape, r, c)
    }
    if any(cell in corridor for part in info["blue_parts"] for cell in part["cells"]):
        return False

    small_mirror_cells: set[tuple[int, int]] = set()
    all_mirror_cells: set[tuple[int, int]] = set()
    for part in info["blue_parts"]:
        rows = {r for r, _ in part["cells"]}
        if max(rows) >= axis_r:
            return False
        reflected = _reflect_cells(part["cells"], axis_r)
        if any(not _in_bounds(grid.shape, rr, cc) or rr <= axis_r for rr, cc in reflected):
            return False
        if any(int(grid[rr, cc]) != BG for rr, cc in reflected):
            return False
        if reflected & all_mirror_cells:
            return False
        all_mirror_cells |= reflected
        if part["area"] <= 2:
            small_mirror_cells |= reflected

    # Anti-shortcut: reflecting all blue objects must differ from the correct output.
    large_only_mirror = set()
    for part in info["large_parts"]:
        large_only_mirror |= _reflect_cells(part["cells"], axis_r)
    if not large_only_mirror:
        return False
    if large_only_mirror & small_mirror_cells:
        return False

    expected = _build_expected(grid)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    if not np.all(actual[axis_r, left_c + 2 : right_c - 1] == MARK):
        return False
    return True


def _sample_marker_columns(width: int) -> tuple[int, int] | None:
    options = []
    for left_c in range(6, width - 11):
        for right_c in range(left_c + 6, width - 6):
            if (left_c - 2) >= 4 and (width - (right_c + 2)) >= 4:
                options.append((left_c, right_c))
    if not options:
        return None
    return random.choice(options)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(17, 24)
    w = random.randint(17, 24)
    axis_r = random.randint(6, h - 7)
    marker_cols = _sample_marker_columns(w)
    if marker_cols is None:
        return None
    left_c, right_c = marker_cols

    marker_cells = _marker_cells(axis_r, left_c) | _marker_cells(axis_r, right_c)
    forbidden_original = {
        (r, c)
        for r in range(axis_r)
        for c in range(left_c - 1, right_c + 2)
        if _in_bounds((h, w), r, c)
    }
    blocked_original = _dilate8(marker_cells, (h, w))
    blocked_mirror: set[tuple[int, int]] = set()

    object_specs: list[tuple[str, tuple[tuple[int, int], ...]]] = []
    for _ in range(random.randint(1, 2)):
        object_specs.append(("large", random.choice(BIG_SHAPES)))
    for _ in range(random.randint(3, 4)):
        object_specs.append(("small", random.choice(SMALL_SHAPES)))
    random.shuffle(object_specs)
    object_specs.sort(key=lambda item: (item[0] != "large", -len(item[1])))

    placed: list[tuple[str, set[tuple[int, int]]]] = []
    for kind, shape in object_specs:
        candidates = _candidate_positions(
            (h, w),
            axis_r,
            shape,
            forbidden_original,
            blocked_original,
            blocked_mirror,
        )
        if not candidates:
            return None
        cells = random.choice(candidates)
        placed.append((kind, cells))
        blocked_original |= _dilate8(cells, (h, w))
        blocked_mirror |= _dilate8(_reflect_cells(cells, axis_r), (h, w))

    grid = _render_input(h, w, axis_r, left_c, right_c, [cells for _, cells in placed])
    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
