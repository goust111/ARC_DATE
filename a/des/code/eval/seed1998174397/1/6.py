from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
PALETTE = (1, 2, 3, 4, 6, 8, 9)

SEGMENT_MIN_LEN = 2
SEGMENT_MAX_LEN = 4
SEGMENT_COUNT_MIN = 2
SEGMENT_COUNT_MAX = 6
MAX_GENERATION_ATTEMPTS = 300


def _band_runs(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = 0
    while start < len(values):
        color = int(values[start])
        end = start + 1
        while end < len(values) and int(values[end]) == color:
            end += 1
        runs.append((color, end - start))
        start = end
    return runs


def _validate_band(values: np.ndarray) -> None:
    runs = _band_runs(values)
    assert runs, "expected a non-empty edge band"
    for _color, run_len in runs:
        assert SEGMENT_MIN_LEN <= run_len <= SEGMENT_MAX_LEN, "edge-band segment lengths must be 2-4"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the top row and left column as weaving instructions for the interior canvas.

    The top band defines the vertical warp colors, the left band defines the
    horizontal weft colors, and orange markers flip the local over/under order
    so the warp color reappears at those crossings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 3 and w >= 3, "grid must include bands plus a non-empty canvas"
    assert int(g[0, 0]) == GREY, "top-left corner must be grey"

    top_band = g[0, 1:]
    left_band = g[1:, 0]
    _validate_band(top_band)
    _validate_band(left_band)

    canvas = g[1:, 1:]
    assert np.all((canvas == BLACK) | (canvas == ORANGE)), "canvas must contain only black background and orange markers"

    warp = np.broadcast_to(top_band[np.newaxis, :], canvas.shape)
    woven = np.broadcast_to(left_band[:, np.newaxis], canvas.shape).copy()
    markers = canvas == ORANGE
    woven[markers] = warp[markers]

    out = g.copy()
    out[1:, 1:] = woven
    return out


def _sample_band(rng: random.Random) -> np.ndarray:
    segment_count = rng.randint(SEGMENT_COUNT_MIN, SEGMENT_COUNT_MAX)
    lengths = [rng.randint(SEGMENT_MIN_LEN, SEGMENT_MAX_LEN) for _ in range(segment_count)]

    colors: list[int] = []
    prev_color: int | None = None
    for _ in range(segment_count):
        choices = [color for color in PALETTE if color != prev_color]
        color = rng.choice(choices)
        colors.append(color)
        prev_color = color

    values: list[int] = []
    for color, length in zip(colors, lengths):
        values.extend([color] * length)
    return np.array(values, dtype=int)


def _choose_marker_count(max_count: int, area: int, rng: random.Random) -> int:
    mode = rng.choice(("sparse", "medium", "dense"))
    if mode == "sparse":
        lo, hi = 1, min(3, max_count)
    elif mode == "medium":
        lo, hi = min(2, max_count), min(max_count, max(2, area // 7))
    else:
        lo, hi = min(3, max_count), min(max_count, max(3, area // 4))

    if lo > hi:
        lo, hi = 1, max_count
    return rng.randint(lo, hi)


def _pick_marker_cells(row_colors: np.ndarray, col_colors: np.ndarray, rng: random.Random) -> set[tuple[int, int]] | None:
    h = len(row_colors)
    w = len(col_colors)

    differing = [(r, c) for r in range(h) for c in range(w) if int(row_colors[r]) != int(col_colors[c])]
    if len(differing) < 2:
        return None

    parity_pools = [
        [(r, c) for r, c in differing if (r + c) % 2 == parity]
        for parity in range(2)
    ]
    usable_pools = [pool for pool in parity_pools if pool]
    if not usable_pools:
        return None

    pool = list(rng.choice(usable_pools))
    rng.shuffle(pool)
    area = h * w
    max_count = min(len(pool), len(differing) - 1, max(1, area // 3))
    if max_count <= 0:
        return None

    count = _choose_marker_count(max_count, area, rng)
    markers = set(pool[:count])

    mark_mask = np.zeros((h, w), dtype=bool)
    for r, c in markers:
        mark_mask[r, c] = True
    visible_default = np.any((row_colors[:, None] != col_colors[None, :]) & ~mark_mask)
    if not visible_default:
        return None

    return markers


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    top_band = _sample_band(rng)
    left_band = _sample_band(rng)
    markers = _pick_marker_cells(left_band, top_band, rng)
    if markers is None:
        return None

    h = len(left_band) + 1
    w = len(top_band) + 1
    grid = np.full((h, w), BLACK, dtype=int)
    grid[0, 0] = GREY
    grid[0, 1:] = top_band
    grid[1:, 0] = left_band
    for r, c in markers:
        grid[r + 1, c + 1] = ORANGE

    out = main(grid)
    assert np.any(grid[1:, 1:] == ORANGE), "generator must place at least one marker"
    assert np.any(out[1:, 1:] != left_band[:, None]), "generator must create at least one visible warp-over-weft crossing"
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
