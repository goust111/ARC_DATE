from __future__ import annotations

from collections import deque
import random

import numpy as np


# Lightweight copies of the arc_utils color and coordinate conventions so this
# file stays directly loadable through SourceFileLoader without path setup.
class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREY = 5
    GRAY = 5


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int):
    yield int(r) - 1, int(c)
    yield int(r) + 1, int(c)
    yield int(r), int(c) - 1
    yield int(r), int(c) + 1


def _find_axis(grid: np.ndarray) -> tuple[str, int]:
    h, w = map(int, grid.shape)
    red_rows = [r for r in range(h) if np.all(grid[r, :] == Color.RED)]
    red_cols = [c for c in range(w) if np.all(grid[:, c] == Color.RED)]
    assert (len(red_rows), len(red_cols)) in {(1, 0), (0, 1)}

    if red_cols:
        axis = int(red_cols[0])
        assert w % 2 == 1 and axis == w // 2
        return "vertical", axis

    axis = int(red_rows[0])
    assert h % 2 == 1 and axis == h // 2
    return "horizontal", axis


def _mirror_coord(shape: tuple[int, int], orientation: str, axis: int, r: int, c: int) -> tuple[int, int]:
    if orientation == "vertical":
        mr, mc = int(r), int(2 * axis - c)
    else:
        mr, mc = int(2 * axis - r), int(c)
    assert in_bounds(shape, mr, mc)
    return mr, mc


def _wormhole_fill_mask(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    seed_coords = np.argwhere(g == Color.BLUE)
    assert seed_coords.shape == (1, 2)

    orientation, axis = _find_axis(g)
    fill = np.zeros(g.shape, dtype=bool)
    sr, sc = map(int, seed_coords[0])
    fill[sr, sc] = True
    q: deque[tuple[int, int]] = deque([(sr, sc)])

    while q:
        r, c = q.popleft()

        for nr, nc in neighbors4(r, c):
            if in_bounds(g.shape, nr, nc) and g[nr, nc] == Color.BLACK and not fill[nr, nc]:
                fill[nr, nc] = True
                q.append((int(nr), int(nc)))

        mr, mc = _mirror_coord(g.shape, orientation, axis, r, c)
        if g[mr, mc] == Color.BLACK and not fill[mr, mc]:
            fill[mr, mc] = True
            q.append((mr, mc))

    return fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood from the blue seed through black space, with every filled cell also activating its mirror across the central red axis.

    Input structure:
    - A full red row or column sits exactly at the center and splits the grid into two mirror halves.
    - The background is black, several gray wall segments block motion, and the wall layouts differ across the axis.
    - Exactly one blue seed pixel lies off the red axis.

    Rule:
    - Start from the blue seed and repeatedly add two kinds of cells until no new cell appears.
    - Any filled cell may expand to its four-neighbor black cells, and its mirror position across the red axis is also filled when that mirrored cell is black.
    - Recolor every finally reached cell blue; keep the red axis, gray walls, and all other cells unchanged.

    Key insight:
    - Mirroring creates wormhole links between the two halves, so regions can be opened on one side, explored there, and then mirrored back to unlock new rooms on the original side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert set(map(int, np.unique(g))) <= {Color.BLACK, Color.BLUE, Color.RED, Color.GREY}

    _find_axis(g)
    fill = _wormhole_fill_mask(g)

    out = g.copy()
    out[fill] = Color.BLUE
    return out


def _local_flood_only(grid: np.ndarray, filled: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = filled.copy()
    q: deque[tuple[int, int]] = deque(map(tuple, np.argwhere(out)))

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(int(r), int(c)):
            if in_bounds(g.shape, nr, nc) and g[nr, nc] == Color.BLACK and not out[nr, nc]:
                out[nr, nc] = True
                q.append((int(nr), int(nc)))
    return out


def _mirror_once(grid: np.ndarray, filled: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    orientation, axis = _find_axis(g)
    out = filled.copy()
    for r, c in np.argwhere(filled):
        mr, mc = _mirror_coord(g.shape, orientation, axis, int(r), int(c))
        if g[mr, mc] == Color.BLACK:
            out[mr, mc] = True
    return out


def _round_based_fill(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    seed = np.zeros(g.shape, dtype=bool)
    sr, sc = map(int, np.argwhere(g == Color.BLUE)[0])
    seed[sr, sc] = True

    filled = seed
    while True:
        expanded = _local_flood_only(g, filled)
        mirrored = _mirror_once(g, expanded)
        if np.array_equal(mirrored, filled):
            return filled
        filled = mirrored


def _single_mirror_round_fill(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    filled = np.zeros(g.shape, dtype=bool)
    sr, sc = map(int, np.argwhere(g == Color.BLUE)[0])
    filled[sr, sc] = True
    filled = _local_flood_only(g, filled)
    filled = _mirror_once(g, filled)
    filled = _local_flood_only(g, filled)
    return filled


def _choose_room_heights() -> list[int]:
    while True:
        heights = [random.choice((3, 4)) for _ in range(5)]
        if sum(heights) + 4 <= 30:
            return heights


def _band_bounds(room_heights: list[int]) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    r = 0
    for i, height in enumerate(room_heights):
        bounds.append((r, r + int(height) - 1))
        r += int(height)
        if i != len(room_heights) - 1:
            r += 1
    return bounds


def _sample_edge_pattern() -> tuple[set[int], set[int]]:
    patterns = [
        ({0, 2}, {1}),
        ({0, 2}, {1}),
        ({0, 2}, {1, 3}),
        ({0, 3}, {1, 2}),
        ({0, 1, 3}, {2}),
    ]
    left_edges, right_edges = random.choice(patterns)
    return set(left_edges), set(right_edges)


def _sample_door(side_start: int, side_end: int) -> tuple[int, int]:
    width = 2 if side_end - side_start + 1 >= 6 and random.random() < 0.7 else 1
    lo = int(side_start)
    hi = int(side_end) - width + 1
    start = random.randint(lo, hi)
    return start, width


def _sprinkle_pillars(grid: np.ndarray, bounds: list[tuple[int, int]], axis_col: int) -> None:
    half_w = axis_col
    seed = tuple(map(int, np.argwhere(grid == Color.BLUE)[0]))
    for _ in range(random.randint(0, 2)):
        band = random.randrange(len(bounds))
        r0, r1 = bounds[band]
        side = random.choice(("left", "right"))
        cells = {
            (1, 1),
            (1, 2),
            (2, 1),
        }
        shapes = [shape for shape in cells if (r1 - r0 + 1) >= shape[0] and half_w >= shape[1] + 2]
        if not shapes:
            continue
        h, w = random.choice(shapes)

        if side == "left":
            c_min, c_max = 0, axis_col - 1
        else:
            c_min, c_max = axis_col + 1, grid.shape[1] - 1

        rr = random.randint(r0, r1 - h + 1)
        cc = random.randint(c_min, c_max - w + 1)
        patch = grid[rr : rr + h, cc : cc + w]
        if np.any(patch != Color.BLACK):
            continue
        if any((rr + dr, cc + dc) == seed for dr in range(h) for dc in range(w)):
            continue
        grid[rr : rr + h, cc : cc + w] = Color.GREY


def _build_vertical_candidate() -> np.ndarray:
    room_heights = _choose_room_heights()
    bounds = _band_bounds(room_heights)
    h = int(bounds[-1][1] + 1)
    half_w = random.randint(5, 9)
    w = 2 * half_w + 1
    axis_col = half_w

    grid = np.full((h, w), Color.BLACK, dtype=int)
    grid[:, axis_col] = Color.RED

    left_edges, right_edges = _sample_edge_pattern()
    for band in range(len(bounds) - 1):
        wall_r = bounds[band][1] + 1
        grid[wall_r, :axis_col] = Color.GREY
        grid[wall_r, axis_col + 1 :] = Color.GREY

        if band in left_edges:
            start, width = _sample_door(0, axis_col - 1)
            grid[wall_r, start : start + width] = Color.BLACK
        if band in right_edges:
            start, width = _sample_door(axis_col + 1, w - 1)
            grid[wall_r, start : start + width] = Color.BLACK

    seed_band = 0
    seed_r = (bounds[seed_band][0] + bounds[seed_band][1]) // 2
    seed_c = max(1, axis_col // 2)
    grid[seed_r, seed_c] = Color.BLUE

    _sprinkle_pillars(grid, bounds, axis_col)
    return grid


def _random_transform(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int)
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = out.T
    return np.array(out, dtype=int, copy=True)


def _walls_are_asymmetric(grid: np.ndarray) -> bool:
    orientation, axis = _find_axis(grid)
    if orientation == "vertical":
        left = grid[:, :axis] == Color.GREY
        right = np.fliplr(grid[:, axis + 1 :] == Color.GREY)
    else:
        left = grid[:axis, :] == Color.GREY
        right = np.flipud(grid[axis + 1 :, :] == Color.GREY)
    return left.shape == right.shape and not np.array_equal(left, right)


def _validate_candidate(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if np.count_nonzero(g == Color.BLUE) != 1:
        return False
    try:
        _find_axis(g)
    except AssertionError:
        return False
    if not _walls_are_asymmetric(g):
        return False

    oracle_fill = _round_based_fill(g)
    if np.count_nonzero(oracle_fill) <= 8:
        return False
    if np.array_equal(oracle_fill, _single_mirror_round_fill(g)):
        return False

    solver_out = main(g)
    expected = g.copy()
    expected[oracle_fill] = Color.BLUE
    if not np.array_equal(solver_out, expected):
        return False

    if np.count_nonzero(expected == Color.BLACK) == 0:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        candidate = _random_transform(_build_vertical_candidate())
        if _validate_candidate(candidate):
            return candidate
    raise RuntimeError("generate_input: failed to construct a valid mirror-wormhole flood sample")
