from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
LINE_COLORS = tuple(color for color in range(10) if color not in {BG, GREY})

MIN_GRID = 7
MAX_GRID = 18
MAX_LINES = 5
MAX_RETRIES = 400

_RNG = random.Random()


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _seed_pairs(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    colors = sorted(int(color) for color in np.unique(grid) if int(color) != BG)
    assert colors, "expected at least one seed pair"

    horizontal: list[tuple[int, int]] = []
    vertical: list[tuple[int, int]] = []
    for color in colors:
        coords = np.argwhere(grid == color)
        assert coords.shape[0] == 2, f"color {color} must appear exactly twice"
        (r0, c0), (r1, c1) = map(tuple, coords.tolist())
        assert (r0 in {0, h - 1} or c0 in {0, w - 1}) and (r1 in {0, h - 1} or c1 in {0, w - 1}), (
            f"color {color} must lie on the border"
        )

        if r0 == r1:
            assert {c0, c1} == {0, w - 1}, f"horizontal pair for color {color} must use left/right border"
            horizontal.append((r0, color))
        elif c0 == c1:
            assert {r0, r1} == {0, h - 1}, f"vertical pair for color {color} must use top/bottom border"
            vertical.append((c0, color))
        else:
            raise AssertionError(f"color {color} does not form a valid opposite-side pair")

    return horizontal, vertical


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw full-width/full-height lines from same-color border seed pairs, then
    fill odd-indexed rectangular regions of the interior with grey.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 3 and w >= 3

    horizontal, vertical = _seed_pairs(g)

    out = g.copy()
    line_mask = np.zeros((h, w), dtype=bool)

    for row, color in sorted(horizontal):
        out[row, :] = color
        line_mask[row, :] = True

    # Mixed-orientation inputs are resolved deterministically by drawing
    # vertical lines after horizontal ones. The generator avoids crossings.
    for col, color in sorted(vertical):
        out[:, col] = color
        line_mask[:, col] = True

    if h <= 2 or w <= 2:
        return out

    interior_mask = out[1:-1, 1:-1] == BG
    regions = []
    for comp in _components4(interior_mask):
        top = min(r for r, _ in comp) + 1
        left = min(c for _, c in comp) + 1
        regions.append(((top, left), comp))

    regions.sort(key=lambda item: item[0])
    for index, (_corner, comp) in enumerate(regions, start=1):
        if index % 2 == 0:
            continue
        for r, c in comp:
            out[r + 1, c + 1] = GREY

    return out


def _sample_spaced_positions(size: int, count: int, rng: random.Random) -> list[int]:
    candidates = list(range(2, size - 2))
    for _ in range(200):
        positions = sorted(rng.sample(candidates, count))
        if all(b - a >= 2 for a, b in zip(positions, positions[1:])):
            return positions
    raise RuntimeError("failed to sample non-degenerate line positions")


def _gen_attempt(rng: random.Random) -> np.ndarray:
    mode = rng.choice(("horizontal", "vertical"))
    h = rng.randint(MIN_GRID, MAX_GRID)
    w = rng.randint(MIN_GRID, MAX_GRID)

    axis_size = h if mode == "horizontal" else w
    max_count = min(MAX_LINES, len(LINE_COLORS), (axis_size - 3) // 2)
    assert max_count >= 1
    count = rng.randint(1, max_count)
    positions = _sample_spaced_positions(axis_size, count, rng)
    colors = rng.sample(LINE_COLORS, count)

    grid = np.full((h, w), BG, dtype=int)
    if mode == "horizontal":
        for row, color in zip(positions, colors):
            grid[row, 0] = color
            grid[row, w - 1] = color
    else:
        for col, color in zip(positions, colors):
            grid[0, col] = color
            grid[h - 1, col] = color

    _validate_generated(grid, mode)
    return grid


def _validate_generated(grid: np.ndarray, mode: str) -> None:
    h, w = grid.shape
    assert np.all(grid[1:-1, 1:-1] == BG)

    horizontal, vertical = _seed_pairs(grid)
    if mode == "horizontal":
        assert horizontal and not vertical
        rows = sorted(row for row, _ in horizontal)
        assert all(2 <= row <= h - 3 for row in rows)
        assert all(b - a >= 2 for a, b in zip(rows, rows[1:]))
    else:
        assert vertical and not horizontal
        cols = sorted(col for col, _ in vertical)
        assert all(2 <= col <= w - 3 for col in cols)
        assert all(b - a >= 2 for a, b in zip(cols, cols[1:]))

    out = main(grid)
    assert out.shape == grid.shape
    assert np.any(out == GREY), "odd-indexed region fill should always be visible"


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        try:
            return _gen_attempt(_RNG)
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input")
