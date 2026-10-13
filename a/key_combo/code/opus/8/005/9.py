from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
PINK = 6
ORANGE = 7
TEAL = 8
PATTERN_COLORS = (BLACK, PINK, ORANGE, TEAL)

_SWAP_TABLE = np.arange(10, dtype=int)
_SWAP_TABLE[BLACK] = TEAL
_SWAP_TABLE[TEAL] = BLACK
_SWAP_TABLE[PINK] = ORANGE
_SWAP_TABLE[ORANGE] = PINK

_GEN_ATTEMPTS = 300


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    n = int(flags.size)
    while i < n:
        if not bool(flags[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and bool(flags[j + 1]):
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs


def _clean_paired_edges(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    clean_a = np.zeros_like(a, dtype=bool)
    clean_b = np.zeros_like(b, dtype=bool)

    runs_a = [run for run in _runs(a) if run[1] - run[0] + 1 >= 2]
    runs_b = [run for run in _runs(b) if run[1] - run[0] + 1 >= 2]

    for a0, a1 in runs_a:
        for b0, b1 in runs_b:
            lo = max(a0, b0)
            hi = min(a1, b1)
            if lo <= hi:
                clean_a[lo : hi + 1] = True
                clean_b[lo : hi + 1] = True
    return clean_a, clean_b


def _apply_swap(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = grid.copy()
    out[mask] = _SWAP_TABLE[out[mask]]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove unsupported edge markers, then swap colors along the opposite-side spans they still mark.

    Input structure:
    - The grid is a 2D pattern using black, teal, pink, and orange.
    - Yellow marker pixels lie only on the four outer edges, where some form short 2-3 pixel runs and some are noise.
    - Left/right markers indicate rows; top/bottom markers indicate columns.

    Rule:
    - Read the yellow pixels on each edge and group consecutive edge pixels into 8-connected edge runs.
    - Remove unsupported marker pixels: only rows or columns belonging to a length-at-least-2 run that also overlaps a length-at-least-2 run on the opposite edge are kept yellow; deleted edge pixels revert to the color just inside the grid.
    - Rows supported by both left and right edges select their strictly interior horizontal span, and columns supported by both top and bottom edges select their strictly interior vertical span.
    - Apply one color swap to every selected interior pixel: black <-> teal and pink <-> orange.
    - Keep the surviving yellow markers and all unselected cells unchanged.

    Key insight:
    - The useful edge markers are the opposite-edge-supported runs, so denoising the borders must happen before the interior color swap spans are chosen.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    out = g.copy()

    left = g[:, 0] == YELLOW
    right = g[:, w - 1] == YELLOW
    top = g[0, :] == YELLOW
    bottom = g[h - 1, :] == YELLOW

    clean_left, clean_right = _clean_paired_edges(left, right)
    clean_top, clean_bottom = _clean_paired_edges(top, bottom)

    for r in np.flatnonzero(left & ~clean_left):
        out[int(r), 0] = int(g[int(r), 1]) if w > 1 else BLACK
    for r in np.flatnonzero(right & ~clean_right):
        out[int(r), w - 1] = int(g[int(r), w - 2]) if w > 1 else BLACK
    for c in np.flatnonzero(top & ~clean_top):
        out[0, int(c)] = int(g[1, int(c)]) if h > 1 else BLACK
    for c in np.flatnonzero(bottom & ~clean_bottom):
        out[h - 1, int(c)] = int(g[h - 2, int(c)]) if h > 1 else BLACK

    swap_mask = np.zeros_like(g, dtype=bool)
    if w > 2:
        for r in np.flatnonzero(clean_left & clean_right):
            swap_mask[int(r), 1 : w - 1] = True
    if h > 2:
        for c in np.flatnonzero(clean_top & clean_bottom):
            swap_mask[1 : h - 1, int(c)] = True

    swap_mask &= out != YELLOW
    out = _apply_swap(out, swap_mask)
    return out


def _base_pattern(h: int, w: int) -> np.ndarray:
    block_h = random.choice((2, 3))
    block_w = random.choice((2, 3))
    coarse_h = (h + block_h - 1) // block_h
    coarse_w = (w + block_w - 1) // block_w

    coarse = np.random.choice(PATTERN_COLORS, size=(coarse_h, coarse_w))
    if len(set(map(int, coarse.flatten()))) < 3:
        for _ in range(6):
            rr = random.randrange(coarse_h)
            cc = random.randrange(coarse_w)
            coarse[rr, cc] = random.choice(PATTERN_COLORS)

    grid = np.repeat(np.repeat(coarse, block_h, axis=0), block_w, axis=1)[:h, :w]

    if random.random() < 0.5:
        for _ in range(random.randint(1, 2)):
            rr = random.randrange(h)
            grid[rr, :] = random.choice(PATTERN_COLORS)
    else:
        for _ in range(random.randint(1, 2)):
            cc = random.randrange(w)
            grid[:, cc] = random.choice(PATTERN_COLORS)

    if h > 1:
        grid[0, :] = grid[1, :]
        grid[h - 1, :] = grid[h - 2, :]
    if w > 1:
        grid[:, 0] = grid[:, 1]
        grid[:, w - 1] = grid[:, w - 2]
    return grid.astype(int)


def _sample_intervals(length: int) -> list[tuple[int, int]]:
    candidates = [(s, s + size - 1) for size in (2, 3) for s in range(1, length - size)]
    random.shuffle(candidates)
    intervals: list[tuple[int, int]] = []
    target = random.randint(1, 2)
    for start, end in candidates:
        if all(end < s0 - 1 or start > e0 + 1 for s0, e0 in intervals):
            intervals.append((start, end))
            if len(intervals) == target:
                break
    intervals.sort()
    return intervals


def _expand_intervals(intervals: list[tuple[int, int]]) -> set[int]:
    positions: set[int] = set()
    for start, end in intervals:
        positions.update(range(start, end + 1))
    return positions


def _add_noise(
    positions: set[int],
    forbidden: set[int],
    length: int,
    intervals: list[tuple[int, int]],
) -> tuple[set[int], bool]:
    positions = set(positions)
    added_attached = False
    count = random.choices([0, 1, 2], weights=[2, 4, 2])[0]
    for _ in range(count):
        use_attached = bool(intervals) and random.random() < 0.55
        if use_attached:
            attached = []
            for start, end in intervals:
                for idx in (start - 1, end + 1):
                    if 1 <= idx < length - 1 and idx not in positions and idx not in forbidden:
                        attached.append(idx)
            if attached:
                idx = random.choice(attached)
                positions.add(idx)
                added_attached = True
                continue

        isolated = []
        for idx in range(1, length - 1):
            if idx in positions or idx in forbidden:
                continue
            if (idx - 1) in positions or (idx + 1) in positions:
                continue
            isolated.append(idx)
        if isolated:
            positions.add(random.choice(isolated))
    return positions, added_attached


def _paint_edge(grid: np.ndarray, edge: str, positions: set[int]) -> None:
    if edge == "left":
        for r in positions:
            grid[int(r), 0] = YELLOW
        return
    if edge == "right":
        for r in positions:
            grid[int(r), -1] = YELLOW
        return
    if edge == "top":
        for c in positions:
            grid[0, int(c)] = YELLOW
        return
    for c in positions:
        grid[-1, int(c)] = YELLOW


def _build_case() -> np.ndarray | None:
    h = random.randint(10, 16)
    w = random.randint(10, 16)
    base = _base_pattern(h, w)

    row_intervals = _sample_intervals(h)
    col_intervals = _sample_intervals(w)
    if not row_intervals or not col_intervals:
        return None

    left = _expand_intervals(row_intervals)
    right = _expand_intervals(row_intervals)
    top = _expand_intervals(col_intervals)
    bottom = _expand_intervals(col_intervals)

    left, left_attached = _add_noise(left, right, h, row_intervals)
    right, right_attached = _add_noise(right, left, h, row_intervals)
    top, top_attached = _add_noise(top, bottom, w, col_intervals)
    bottom, bottom_attached = _add_noise(bottom, top, w, col_intervals)

    all_noise = (left - _expand_intervals(row_intervals)) | (right - _expand_intervals(row_intervals))
    all_noise |= (top - _expand_intervals(col_intervals)) | (bottom - _expand_intervals(col_intervals))
    if not all_noise:
        return None

    grid = base.copy()
    _paint_edge(grid, "left", left)
    _paint_edge(grid, "right", right)
    _paint_edge(grid, "top", top)
    _paint_edge(grid, "bottom", bottom)

    out = main(grid)
    if np.array_equal(grid, out):
        return None

    row_hits = np.flatnonzero((out[:, 0] == YELLOW) & (out[:, -1] == YELLOW))
    col_hits = np.flatnonzero((out[0, :] == YELLOW) & (out[-1, :] == YELLOW))
    if row_hits.size < 2 or col_hits.size < 2:
        return None

    removed_noise = int(np.sum(grid == YELLOW) - np.sum(out == YELLOW))
    if removed_noise <= 0:
        return None

    if not (left_attached or right_attached or top_attached or bottom_attached) and random.random() < 0.6:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _build_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
