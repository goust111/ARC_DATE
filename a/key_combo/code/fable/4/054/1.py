from __future__ import annotations

from collections import Counter

import numpy as np


OCCLUDER = 8
MAX_ATTEMPTS = 400
ALL_COLORS = tuple(range(10))
RNG = np.random.default_rng()


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    h, w = shape
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _trace_path(mask: np.ndarray) -> list[tuple[int, int]]:
    cells = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    assert cells, "expected a non-empty path"

    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell in cells:
        adjacency[cell] = [nbr for nbr in _neighbors4(mask.shape, *cell) if nbr in cells]
        assert len(adjacency[cell]) <= 2, "path must be branch-free"

    endpoints = sorted(cell for cell, nbrs in adjacency.items() if len(nbrs) == 1)
    assert len(endpoints) == 2, "path must have exactly two endpoints"

    start = endpoints[0]
    order = [start]
    prev = None
    cur = start
    while True:
        nxts = [nbr for nbr in adjacency[cur] if nbr != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path traversal must be unique"
        prev, cur = cur, nxts[0]
        order.append(cur)

    assert len(order) == len(cells), "path traversal missed cells"
    return order


def _infer_period(colors: list[int], unknown: int = OCCLUDER) -> list[int]:
    n = len(colors)
    for period in range(1, n + 1):
        pattern: list[int | None] = [None] * period
        ok = True
        for idx, color in enumerate(colors):
            if color == unknown:
                continue
            slot = idx % period
            seen = pattern[slot]
            if seen is None:
                pattern[slot] = int(color)
            elif seen != int(color):
                ok = False
                break
        if ok and all(value is not None for value in pattern):
            return [int(value) for value in pattern]
    raise AssertionError("failed to infer a fully observed shortest period")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover gray-occluded snake cells by inferring the repeating color cycle along the path order.

    Input structure:
    - The background is the most common color.
    - All non-background cells form one 4-connected branch-free snake path with exactly two endpoints.
    - Snake colors repeat with a fixed period along the path, and some path cells are replaced by color 8.

    Rule:
    - Find the path endpoints and traverse the snake cell-by-cell from one endpoint to the other.
    - Infer the unique shortest repeating color period from the visible non-8 path cells.
    - Replace every color-8 path cell by the color for its path index modulo that period, leaving all other cells unchanged.

    Key insight:
    - The hidden structure is one-dimensional: the relevant neighborhood is the snake traversal order, not the 2D row or column layout around each gray cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    path_mask = g != bg
    order = _trace_path(path_mask)
    colors = [int(g[r, c]) for r, c in order]
    pattern = _infer_period(colors)

    out = g.copy()
    for idx, (r, c) in enumerate(order):
        if int(out[r, c]) == OCCLUDER:
            out[r, c] = pattern[idx % len(pattern)]
    return out


def _build_horizontal_snake(height: int, width: int) -> list[tuple[int, int]] | None:
    max_lanes = min(4, (height + 1) // 2)
    if max_lanes < 2 or width < 4:
        return None

    lanes = int(RNG.integers(2, max_lanes + 1))
    top = int(RNG.integers(0, height - (2 * lanes - 1) + 1))
    direction = int(RNG.choice(np.array([-1, 1], dtype=int)))

    if direction == 1:
        start_col = int(RNG.integers(0, width - 2))
    else:
        start_col = int(RNG.integers(2, width))

    path: list[tuple[int, int]] = []
    current_col = start_col

    for lane_idx in range(lanes):
        row = top + 2 * lane_idx
        if direction == 1:
            if current_col + 2 >= width:
                return None
            end_col = int(RNG.integers(current_col + 2, width))
            lane_cols = list(range(current_col, end_col + 1))
        else:
            if current_col - 2 < 0:
                return None
            end_col = int(RNG.integers(0, current_col - 1))
            lane_cols = list(range(current_col, end_col - 1, -1))

        if lane_idx == 0:
            path.extend((row, c) for c in lane_cols)
        else:
            path.extend((row, c) for c in lane_cols[1:])

        current_col = end_col
        if lane_idx + 1 < lanes:
            path.append((row + 1, current_col))
            path.append((row + 2, current_col))
            direction *= -1

    return path


def _build_snake(height: int, width: int) -> list[tuple[int, int]] | None:
    if int(RNG.integers(0, 2)) == 0:
        return _build_horizontal_snake(height, width)

    transposed = _build_horizontal_snake(width, height)
    if transposed is None:
        return None
    return [(c, r) for r, c in transposed]


def _primitive_cycle(colors: np.ndarray) -> bool:
    seq = list(map(int, colors))
    for period in range(1, len(seq)):
        if len(seq) % period == 0 and seq == seq[:period] * (len(seq) // period):
            return False
    return True


def _choose_cycle(bg: int, path_len: int) -> np.ndarray:
    usable = [color for color in ALL_COLORS if color not in {bg, OCCLUDER}]
    max_period = min(5, len(usable), path_len // 3)
    assert max_period >= 2

    while True:
        period = int(RNG.integers(2, max_period + 1))
        cycle = RNG.choice(np.array(usable, dtype=int), size=period, replace=False)
        if _primitive_cycle(cycle):
            return cycle.astype(int)


def _choose_occluded_indices(path_len: int, period: int) -> set[int]:
    anchor_hi = min(path_len - 1, 3 * period)
    anchor_len = int(RNG.integers(2 * period, anchor_hi + 1))
    anchor_start = int(RNG.integers(0, path_len - anchor_len + 1))
    anchor_stop = anchor_start + anchor_len

    intervals = []
    if anchor_start > 0:
        intervals.append((0, anchor_start))
    if anchor_stop < path_len:
        intervals.append((anchor_stop, path_len))

    occluded: set[int] = set()
    RNG.shuffle(intervals)
    for lo, hi in intervals:
        if hi - lo <= 0:
            continue
        if occluded and RNG.random() < 0.35:
            continue
        seg_len = int(RNG.integers(1, min(period, hi - lo) + 1))
        seg_start = int(RNG.integers(lo, hi - seg_len + 1))
        occluded.update(range(seg_start, seg_start + seg_len))

    if not occluded:
        outside = [idx for idx in range(path_len) if idx < anchor_start or idx >= anchor_stop]
        occluded.add(int(RNG.choice(np.array(outside, dtype=int))))
    return occluded


def _render_case(
    shape: tuple[int, int],
    bg: int,
    path: list[tuple[int, int]],
    cycle: np.ndarray,
    occluded: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    clean = np.full(shape, bg, dtype=int)
    for idx, (r, c) in enumerate(path):
        clean[r, c] = int(cycle[idx % len(cycle)])

    inp = clean.copy()
    for idx in occluded:
        r, c = path[idx]
        inp[r, c] = OCCLUDER
    return inp, clean


def _gen_attempt() -> np.ndarray | None:
    height = int(RNG.integers(10, 17))
    width = int(RNG.integers(10, 17))
    bg = int(RNG.choice(np.array([c for c in ALL_COLORS if c != OCCLUDER], dtype=int)))

    path = _build_snake(height, width)
    if path is None:
        return None
    if path[0] > path[-1]:
        path = list(reversed(path))
    if len(path) > (height * width) // 2:
        return None

    cycle = _choose_cycle(bg, len(path))
    occluded = _choose_occluded_indices(len(path), len(cycle))
    inp, clean = _render_case((height, width), bg, path, cycle, occluded)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, clean):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid snake-period restoration input")
