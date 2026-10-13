from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
START = 5
LEGEND_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (12, 19)
GRID_W_RANGE = (6, 19)
LEGEND_LEN_OPTIONS = (3, 4, 5)
MAX_GENERATION_ATTEMPTS = 600

_RNG = np.random.default_rng()


def _legend_info(grid: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    top = np.asarray(grid[0], dtype=int)
    nz = np.flatnonzero(top != BG)
    assert len(nz) >= 3, "top row must contain a legend run"
    c0 = int(nz[0])
    c1 = int(nz[-1])
    assert np.array_equal(nz, np.arange(c0, c1 + 1)), "legend must be one contiguous top-edge run"
    seq = tuple(int(v) for v in top[c0 : c1 + 1])
    assert 3 <= len(seq) <= 5, "legend length must be 3..5"
    assert len(set(seq)) == len(seq), "legend colors must be distinct"
    assert all(color != BG for color in seq), "legend colors must be non-background"
    return c0, c1, seq


def _start_info(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == START)
    assert coords.shape[0] == 1, "expected exactly one gray start pixel"
    r, c = map(int, coords[0])
    h, w = grid.shape
    assert 0 < r < h - 1, "start must be off the top/bottom edges"
    assert 0 < c < w - 1, "start must be off the side edges"
    return r, c


def _validate_input(grid: np.ndarray) -> tuple[int, int, tuple[int, ...], int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h <= 30 and w <= 30, "ARC grids must fit within 30x30"
    c0, c1, legend = _legend_info(g)
    sr, sc = _start_info(g)

    allowed = np.zeros_like(g, dtype=bool)
    allowed[0, c0 : c1 + 1] = True
    allowed[sr, sc] = True
    assert not np.any((g != BG) & ~allowed), "only the legend and one gray start pixel may be foreground"
    assert int(g[sr, sc]) == START
    return c0, c1, legend, sr, sc


def _trace_path(
    shape: tuple[int, int], start: tuple[int, int], legend: Iterable[int]
) -> tuple[list[tuple[int, int, int]], list[int]]:
    h, w = map(int, shape)
    r, c = map(int, start)
    colors = tuple(int(v) for v in legend)
    assert colors

    color_i = 0
    dc = 1
    segment_lengths: list[int] = []
    current_segment_len = 0
    path: list[tuple[int, int, int]] = []

    while True:
        path.append((r, c, colors[color_i]))
        current_segment_len += 1

        nr = r - 1
        if nr < 0:
            break

        nc = c + dc
        if nc < 0 or nc >= w:
            segment_lengths.append(current_segment_len)
            current_segment_len = 0
            dc = -dc
            color_i = (color_i + 1) % len(colors)
            nc = c + dc

        r, c = nr, nc

    segment_lengths.append(current_segment_len)
    return path, segment_lengths


def _render_output(
    grid: np.ndarray, legend_span: tuple[int, int], path: list[tuple[int, int, int]]
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for r, c, color in path:
        out[r, c] = int(color)
    c0, c1 = legend_span
    out[0, c0 : c1 + 1] = grid[0, c0 : c1 + 1]
    return out


def _oracle_output(grid: np.ndarray) -> tuple[np.ndarray, list[int], list[tuple[int, int, int]]]:
    c0, c1, legend, sr, sc = _validate_input(grid)
    path, segment_lengths = _trace_path(grid.shape, (sr, sc), legend)
    out = _render_output(grid, (c0, c1), path)
    return out, segment_lengths, path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the gray launch pixel diagonally upward, switching to the next legend color after each side-wall bounce.

    Input structure:
    - The background is black.
    - The top row contains exactly one contiguous non-black legend run of 3 to 5 distinct colors.
    - Away from the top edge there is exactly one isolated gray start pixel in the interior.

    Rule:
    - Read the top-edge legend colors from left to right as a cyclic sequence.
    - Starting at the gray pixel, move one step up-right each turn, painting every visited cell with the current segment color.
    - When the next step would leave the grid through the left or right side, reverse only the horizontal direction and advance the legend to its next color before continuing upward.
    - Stop after reaching the top row; keep the original legend run unchanged, and replace the gray start pixel with the first legend color.

    Key insight:
    - Geometry decides where each diagonal segment ends, but color changes happen only on side-wall bounces, not on every visited pixel.
    """
    out, _segment_lengths, _path = _oracle_output(np.asarray(grid, dtype=int))
    return out


def _wrong_pixel_cycle_output(grid: np.ndarray) -> np.ndarray:
    c0, c1, legend, sr, sc = _validate_input(grid)
    path, _segment_lengths = _trace_path(grid.shape, (sr, sc), legend)
    out = np.asarray(grid, dtype=int).copy()
    for i, (r, c, _segment_color) in enumerate(path):
        out[r, c] = int(legend[i % len(legend)])
    out[0, c0 : c1 + 1] = grid[0, c0 : c1 + 1]
    return out


def _mode_shape(mode: str) -> tuple[int, int, int]:
    if mode == "straight":
        h = int(_RNG.integers(12, 19))
        w = int(_RNG.integers(11, 19))
        legend_len = int(_RNG.choice([3, 4, 5]))
    elif mode == "single":
        h = int(_RNG.integers(12, 19))
        w = int(_RNG.integers(8, 16))
        legend_len = int(_RNG.choice([3, 4, 5]))
    elif mode == "multi":
        h = int(_RNG.integers(13, 19))
        w = int(_RNG.integers(6, 12))
        legend_len = int(_RNG.choice([3, 4, 5]))
    else:
        assert mode == "wrap"
        h = int(_RNG.integers(14, 19))
        w = int(_RNG.integers(6, 9))
        legend_len = int(_RNG.choice([3, 4]))
    return h, w, legend_len


def _candidate_starts(
    h: int, w: int, legend_len: int, mode: str
) -> list[tuple[int, int, list[int], list[tuple[int, int, int]]]]:
    base_legend = tuple(LEGEND_POOL[:legend_len])
    candidates = []

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            path, segment_lengths = _trace_path((h, w), (r, c), base_legend)
            bounces = len(segment_lengths) - 1

            if mode == "straight" and bounces != 0:
                continue
            if mode == "single" and bounces != 1:
                continue
            if mode == "multi" and not (2 <= bounces < legend_len):
                continue
            if mode == "wrap" and bounces < legend_len:
                continue

            if len(path) < 4:
                continue
            if bounces > 0 and max(segment_lengths) < 2:
                continue

            candidates.append((r, c, segment_lengths, path))

    return candidates


def _legend_positions(w: int, legend_len: int, landing_c: int, overlap: bool) -> list[int]:
    positions = []
    for c0 in range(w - legend_len + 1):
        hits = c0 <= landing_c < c0 + legend_len
        if hits == overlap:
            positions.append(c0)
    return positions


def _random_legend(legend_len: int) -> tuple[int, ...]:
    colors = list(_RNG.choice(LEGEND_POOL, size=legend_len, replace=False))
    _RNG.shuffle(colors)
    return tuple(int(v) for v in colors)


def _build_input(
    h: int, w: int, legend_len: int, start: tuple[int, int], path: list[tuple[int, int, int]]
) -> np.ndarray | None:
    overlap = bool(_RNG.integers(0, 2))
    landing_c = int(path[-1][1])
    c0_options = _legend_positions(w, legend_len, landing_c, overlap)
    if not c0_options:
        c0_options = _legend_positions(w, legend_len, landing_c, not overlap)
    if not c0_options:
        return None

    c0 = int(_RNG.choice(c0_options))
    legend = _random_legend(legend_len)
    sr, sc = start

    grid = np.zeros((h, w), dtype=int)
    grid[0, c0 : c0 + legend_len] = np.asarray(legend, dtype=int)
    grid[sr, sc] = START

    correct, segment_lengths, traced_path = _oracle_output(grid)
    assert len(segment_lengths) >= 1
    assert len(traced_path) == len(path)

    wrong = _wrong_pixel_cycle_output(grid)
    if np.array_equal(correct, wrong):
        return None

    if np.array_equal(correct, grid):
        return None

    bounces = len(segment_lengths) - 1
    if bounces > 0:
        path_colors = {int(correct[r, c]) for r, c, _ in traced_path[:-1]}
        if len(path_colors) < 2:
            return None

    return grid


def _attempt_generate() -> np.ndarray | None:
    mode = random.choices(
        population=["straight", "single", "multi", "wrap"],
        weights=[1, 2, 3, 2],
        k=1,
    )[0]
    h, w, legend_len = _mode_shape(mode)
    candidates = _candidate_starts(h, w, legend_len, mode)
    if not candidates:
        return None

    chosen = candidates[int(_RNG.integers(len(candidates)))]
    sr, sc, _segment_lengths, path = chosen
    grid = _build_input(h, w, legend_len, (sr, sc), path)
    if grid is None:
        return None

    expected, segment_lengths, _path = _oracle_output(grid)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    bounces = len(segment_lengths) - 1
    if mode == "straight" and bounces != 0:
        return None
    if mode == "single" and bounces != 1:
        return None
    if mode == "multi" and not (2 <= bounces < legend_len):
        return None
    if mode == "wrap" and bounces < legend_len:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
