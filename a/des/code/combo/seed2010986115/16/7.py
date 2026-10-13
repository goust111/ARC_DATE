from __future__ import annotations

from collections import Counter

import numpy as np


GREY = 5
BLACK = 0
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 500


def _checkerboard(h: int, w: int, even_color: int, odd_color: int) -> np.ndarray:
    rows = np.arange(h, dtype=int)[:, None]
    cols = np.arange(w, dtype=int)[None, :]
    return np.where((rows + cols) % 2 == 0, even_color, odd_color).astype(int)


def _majority_color(values: np.ndarray) -> int:
    colors, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _infer_checkerboard_colors(grid: np.ndarray) -> tuple[int, int]:
    rows = np.arange(grid.shape[0], dtype=int)[:, None]
    cols = np.arange(grid.shape[1], dtype=int)[None, :]
    even_mask = (rows + cols) % 2 == 0
    odd_mask = ~even_mask

    even_color = _majority_color(grid[even_mask])
    odd_color = _majority_color(grid[odd_mask])
    assert even_color != odd_color, "expected two different checkerboard colors"
    return even_color, odd_color


def _extract_markers(grid: np.ndarray) -> tuple[list[tuple[int, int, int]], Counter]:
    even_bg, odd_bg = _infer_checkerboard_colors(grid)
    markers: list[tuple[int, int, int]] = []
    counts: Counter = Counter()

    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            expected = even_bg if (r + c) % 2 == 0 else odd_bg
            color = int(grid[r, c])
            if color == expected:
                continue
            markers.append((r, c, color))
            counts[color] += 1

    assert markers, "expected at least one colored marker"
    return markers, counts


def _diamond_outline(center_r: int, center_c: int, radius: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for dr in range(-radius, radius + 1):
        dc = radius - abs(dr)
        cells.append((center_r + dr, center_c + dc))
        if dc != 0:
            cells.append((center_r + dr, center_c - dc))
    return cells


def _filled_diamond(center_r: int, center_c: int, radius: int, h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for dr in range(-radius, radius + 1):
        rr = center_r + dr
        if rr < 0 or rr >= h:
            continue
        max_dc = radius - abs(dr)
        for dc in range(-max_dc, max_dc + 1):
            cc = center_c + dc
            if 0 <= cc < w:
                cells.append((rr, cc))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find all colored marker pixels on the checkerboard and draw a diamond
    outline around each marker. The radius equals the global count of that
    marker color, and only outline pixels with the same checkerboard parity as
    the center are painted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    markers, color_counts = _extract_markers(g)
    h, w = g.shape

    for r, c, color in markers:
        radius = int(color_counts[color])
        center_parity = (r + c) % 2
        for rr, cc in _diamond_outline(r, c, radius):
            if 0 <= rr < h and 0 <= cc < w and (rr + cc) % 2 == center_parity:
                out[rr, cc] = int(color)

    return out


def _choose_counts(rng: np.random.Generator, n_colors: int) -> list[int]:
    counts = [int(rng.choice([2, 4]))]
    while len(counts) < n_colors:
        if rng.random() < 0.35:
            counts.append(int(rng.choice([1, 3, 5])))
        else:
            counts.append(int(rng.choice([2, 4])))

    if len(set(counts)) == 1:
        alternatives = [n for n in (1, 2, 3, 4, 5) if n != counts[0]]
        counts[-1] = int(rng.choice(alternatives))

    rng.shuffle(counts)
    return counts


def _size_floor(counts: list[int]) -> int:
    pressure = sum(count * (1 + 2 * count * (count + 1)) for count in counts)
    return max(18, int(np.ceil(np.sqrt(pressure * 1.8))))


def _place_groups(
    rng: np.random.Generator, h: int, w: int, counts: list[int]
) -> list[list[tuple[int, int]]] | None:
    groups: list[list[tuple[int, int]]] = [[] for _ in counts]
    occupied = np.zeros((h, w), dtype=bool)
    order = sorted(range(len(counts)), key=lambda idx: (-counts[idx], idx))

    for idx in order:
        radius = int(counts[idx])
        border = radius + 1
        if h <= 2 * border or w <= 2 * border:
            return None

        for _ in range(radius):
            placed = False
            for _ in range(PLACEMENT_ATTEMPTS):
                r = int(rng.integers(border, h - border))
                c = int(rng.integers(border, w - border))
                footprint = _filled_diamond(r, c, radius, h, w)
                if any(occupied[rr, cc] for rr, cc in footprint):
                    continue

                groups[idx].append((r, c))
                for rr, cc in footprint:
                    occupied[rr, cc] = True
                placed = True
                break

            if not placed:
                return None

    return groups


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_colors = int(rng.choice([2, 3]))
    counts = _choose_counts(rng, n_colors)

    size_floor = _size_floor(counts)
    if size_floor > 34:
        return None

    h = int(rng.integers(size_floor, min(35, size_floor + 6)))
    w = int(rng.integers(size_floor, min(35, size_floor + 6)))
    groups = _place_groups(rng, h, w, counts)
    if groups is None:
        return None

    if rng.integers(0, 2) == 0:
        even_bg, odd_bg = GREY, BLACK
    else:
        even_bg, odd_bg = BLACK, GREY
    grid = _checkerboard(h, w, even_bg, odd_bg)

    colors = [int(c) for c in rng.choice(MARKER_COLORS, size=n_colors, replace=False)]
    for color, cells in zip(colors, groups):
        for r, c in cells:
            grid[r, c] = color

    if np.array_equal(main(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
