from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Square:
    color: int
    r0: int
    c0: int
    r1: int
    c1: int
    side: int


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_square_components(grid: np.ndarray, bg: int) -> list[Square]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    squares: list[Square] = []

    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            side = r1 - r0 + 1
            assert side == c1 - c0 + 1, "each object must be a square"
            assert 1 <= side <= 3, "square side must be in 1..3"
            assert len(cells) == side * side, "each square must be solid"
            assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), "square bbox must be fully filled"
            squares.append(Square(color=color, r0=r0, c0=c0, r1=r1, c1=c1, side=side))

    assert squares and len(squares) % 2 == 0, "expected an even number of squares"
    return squares


def _overlap_1d(a0: int, a1: int, b0: int, b1: int) -> int:
    return min(a1, b1) - max(a0, b0) + 1


def _horizontal_candidate(grid: np.ndarray, bg: int, a: Square, b: Square) -> bool:
    left, right = (a, b) if a.c0 < b.c0 else (b, a)
    gap = right.c0 - left.c1 - 1
    if gap <= 0:
        return False
    if _overlap_1d(left.r0, left.r1, right.r0, right.r1) <= 0:
        return False
    corridor = grid[min(left.r0, right.r0) : max(left.r1, right.r1) + 1, left.c1 + 1 : right.c0]
    return corridor.size > 0 and np.all(corridor == bg)


def _vertical_candidate(grid: np.ndarray, bg: int, a: Square, b: Square) -> bool:
    top, bottom = (a, b) if a.r0 < b.r0 else (b, a)
    gap = bottom.r0 - top.r1 - 1
    if gap <= 0:
        return False
    if _overlap_1d(top.c0, top.c1, bottom.c0, bottom.c1) <= 0:
        return False
    corridor = grid[top.r1 + 1 : bottom.r0, min(top.c0, bottom.c0) : max(top.c1, bottom.c1) + 1]
    return corridor.size > 0 and np.all(corridor == bg)


def _pair_squares(grid: np.ndarray, bg: int, squares: list[Square]) -> list[tuple[Square, Square, str]]:
    neighbors: list[list[tuple[int, str]]] = [[] for _ in squares]

    for i in range(len(squares)):
        for j in range(i + 1, len(squares)):
            horizontal = _horizontal_candidate(grid, bg, squares[i], squares[j])
            vertical = _vertical_candidate(grid, bg, squares[i], squares[j])
            assert not (horizontal and vertical), "a pair cannot be both horizontal and vertical"
            if horizontal:
                neighbors[i].append((j, "h"))
                neighbors[j].append((i, "h"))
            elif vertical:
                neighbors[i].append((j, "v"))
                neighbors[j].append((i, "v"))

    for options in neighbors:
        assert len(options) == 1, "each square must have exactly one facing partner"

    pairs: list[tuple[Square, Square, str]] = []
    used: set[int] = set()
    for i, options in enumerate(neighbors):
        if i in used:
            continue
        j, axis = options[0]
        assert j not in used
        used.add(i)
        used.add(j)
        pairs.append((squares[i], squares[j], axis))

    assert len(used) == len(squares)
    return pairs


def _growth_lengths(gap: int, left_speed: int, right_speed: int) -> tuple[int, int]:
    full_ticks, remainder = divmod(gap, left_speed + right_speed)
    left = full_ticks * left_speed
    right = full_ticks * right_speed
    if remainder:
        left += max(0, remainder - right_speed)
        right += max(0, remainder - left_speed)
    return left, right


def _paint_pair(out: np.ndarray, a: Square, b: Square, axis: str, bg: int) -> None:
    if axis == "h":
        left, right = (a, b) if a.c0 < b.c0 else (b, a)
        gap = right.c0 - left.c1 - 1
        left_len, right_len = _growth_lengths(gap, left.side, right.side)

        if left_len:
            target = out[left.r0 : left.r1 + 1, left.c1 + 1 : left.c1 + 1 + left_len]
            assert np.all(target == bg)
            target[:] = left.color
        if right_len:
            target = out[right.r0 : right.r1 + 1, right.c0 - right_len : right.c0]
            assert np.all(target == bg)
            target[:] = right.color
        return

    top, bottom = (a, b) if a.r0 < b.r0 else (b, a)
    gap = bottom.r0 - top.r1 - 1
    top_len, bottom_len = _growth_lengths(gap, top.side, bottom.side)

    if top_len:
        target = out[top.r1 + 1 : top.r1 + 1 + top_len, top.c0 : top.c1 + 1]
        assert np.all(target == bg)
        target[:] = top.color
    if bottom_len:
        target = out[bottom.r0 - bottom_len : bottom.r0, bottom.c0 : bottom.c1 + 1]
        assert np.all(target == bg)
        target[:] = bottom.color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow every facing square pair inward until their thickness-sized turns either meet exactly or collide on a same-tick frontier.

    Input structure:
    - The grid has a black background and an even number of separated monochrome solid squares with side length 1 to 3.
    - Each square belongs to exactly one facing pair, aligned either horizontally with overlapping rows or vertically with overlapping columns.
    - The gap rectangle between paired squares is clear background, and different pairs do not interfere.

    Rule:
    - For each pair, treat the side length of each square as both its bar thickness and its per-tick advance speed toward its partner.
    - Repeat the simultaneous inward growth until the remaining gap is either consumed exactly or the two new chunks would overlap in the same tick.
    - Paint each square's completed inward bar in its own color and leave any same-tick contested cells black.

    Key insight:
    - The final split is set by tick-wise race times, not by the geometric midpoint: larger squares claim more of the gap because they advance in thicker, faster chunks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    squares = _find_square_components(g, bg)
    pairs = _pair_squares(g, bg, squares)

    out = g.copy()
    for first, second, axis in pairs:
        _paint_pair(out, first, second, axis, bg)
    return out


_PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
_MAX_GENERATION_ATTEMPTS = 400


def _distribute(total: int, slots: int, rng: np.random.Generator) -> list[int]:
    parts = [0] * slots
    for _ in range(total):
        parts[int(rng.integers(0, slots))] += 1
    return parts


def _sample_gap(rng: np.random.Generator, s1: int, s2: int) -> int:
    cycle = s1 + s2
    if float(rng.random()) < 0.5:
        return cycle * int(rng.integers(1, 4))
    safe_ticks = int(rng.integers(0, 3))
    remainder = int(rng.integers(1, cycle))
    return safe_ticks * cycle + remainder


def _sample_specs(pair_count: int, rng: np.random.Generator) -> list[tuple[int, int, int]]:
    specs: list[tuple[int, int, int]] = []
    for _ in range(pair_count):
        s1 = int(rng.integers(1, 4))
        s2 = int(rng.integers(1, 4))
        specs.append((s1, s2, _sample_gap(rng, s1, s2)))
    return specs


def _build_horizontal_input(rng: np.random.Generator) -> np.ndarray | None:
    pair_count = int(rng.integers(2, 5))
    specs = _sample_specs(pair_count, rng)

    band_heights = [max(s1, s2) for s1, s2, _ in specs]
    widths = [s1 + gap + s2 for s1, s2, gap in specs]
    min_height = sum(band_heights) + (pair_count + 1)
    min_width = max(widths) + 2
    if min_height > 30 or min_width > 30:
        return None

    height = int(rng.integers(min_height, 31))
    width = int(rng.integers(min_width, 31))
    colors = list(map(int, rng.choice(_PALETTE, size=2 * pair_count, replace=False)))
    row_extras = _distribute(height - min_height, pair_count + 1, rng)

    grid = np.zeros((height, width), dtype=int)
    row = 1 + row_extras[0]
    for idx, (s1, s2, gap) in enumerate(specs):
        band_height = max(s1, s2)
        total_width = s1 + gap + s2
        left = int(rng.integers(1, width - total_width + 1))

        color_a = colors[2 * idx]
        color_b = colors[2 * idx + 1]
        r1 = row + (band_height - s1) // 2
        r2 = row + (band_height - s2) // 2
        c1 = left
        c2 = left + s1 + gap

        grid[r1 : r1 + s1, c1 : c1 + s1] = color_a
        grid[r2 : r2 + s2, c2 : c2 + s2] = color_b
        row += band_height + 1 + row_extras[idx + 1]

    return grid


def _build_vertical_input(rng: np.random.Generator) -> np.ndarray | None:
    pair_count = int(rng.integers(2, 5))
    specs = _sample_specs(pair_count, rng)

    band_widths = [max(s1, s2) for s1, s2, _ in specs]
    heights = [s1 + gap + s2 for s1, s2, gap in specs]
    min_width = sum(band_widths) + (pair_count + 1)
    min_height = max(heights) + 2
    if min_height > 30 or min_width > 30:
        return None

    height = int(rng.integers(min_height, 31))
    width = int(rng.integers(min_width, 31))
    colors = list(map(int, rng.choice(_PALETTE, size=2 * pair_count, replace=False)))
    col_extras = _distribute(width - min_width, pair_count + 1, rng)

    grid = np.zeros((height, width), dtype=int)
    col = 1 + col_extras[0]
    for idx, (s1, s2, gap) in enumerate(specs):
        band_width = max(s1, s2)
        total_height = s1 + gap + s2
        top = int(rng.integers(1, height - total_height + 1))

        color_a = colors[2 * idx]
        color_b = colors[2 * idx + 1]
        c1 = col + (band_width - s1) // 2
        c2 = col + (band_width - s2) // 2
        r1 = top
        r2 = top + s1 + gap

        grid[r1 : r1 + s1, c1 : c1 + s1] = color_a
        grid[r2 : r2 + s2, c2 : c2 + s2] = color_b
        col += band_width + 1 + col_extras[idx + 1]

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        mode = "horizontal" if float(rng.random()) < 0.5 else "vertical"
        grid = _build_horizontal_input(rng) if mode == "horizontal" else _build_vertical_input(rng)
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(grid, out):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
