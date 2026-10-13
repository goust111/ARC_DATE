from __future__ import annotations

from collections import Counter, deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5

PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _bands(indices: np.ndarray) -> list[tuple[int, int]]:
    vals = [int(x) for x in indices]
    assert vals, "expected at least one non-separator band"
    out: list[tuple[int, int]] = []
    start = prev = vals[0]
    for x in vals[1:]:
        if x == prev + 1:
            prev = x
        else:
            out.append((start, prev))
            start = prev = x
    out.append((start, prev))
    return out


def _component_size(mask: np.ndarray) -> int:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 0
    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen)


def _separator_color(g: np.ndarray) -> int:
    candidates = []
    for color in np.unique(g):
        color = int(color)
        if np.any(np.all(g == color, axis=1)) and np.any(np.all(g == color, axis=0)):
            candidates.append(color)
    assert len(candidates) == 1, f"expected one separator color, got {candidates}"
    return candidates[0]


def _cell_bands(g: np.ndarray, sep: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_bands = _bands(np.where(~np.all(g == sep, axis=1))[0])
    col_bands = _bands(np.where(~np.all(g == sep, axis=0))[0])
    row_heights = {r1 - r0 + 1 for r0, r1 in row_bands}
    col_widths = {c1 - c0 + 1 for c0, c1 in col_bands}
    assert len(row_heights) == 1 and len(col_widths) == 1, "cells must have equal size"
    return row_bands, col_bands


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor unique-color cell objects by checkerboard cell parity.

    Input structure:
    - Black background.
    - Grey separator rows and columns split the grid into equal rectangular cells.
    - Each cell contains exactly one 4-connected monochrome object.
    - Colors that appear in exactly one cell are "unique"; colors appearing in multiple cells are repeats.

    Rule:
    - Keep repeated-color objects unchanged.
    - Recolor each unique-color object to green if its cell index sum is even, otherwise red.
    - Leave separator lines unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _separator_color(g)
    row_bands, col_bands = _cell_bands(g, sep)

    cells: list[tuple[int, int, int, np.ndarray]] = []
    color_counts: Counter[int] = Counter()
    for i, (r0, r1) in enumerate(row_bands):
        for j, (c0, c1) in enumerate(col_bands):
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            assert not np.any(cell == sep), "separator color must stay outside cells"
            colors = [int(color) for color in np.unique(cell) if int(color) != BLACK]
            assert len(colors) == 1, f"expected one object color per cell, got {colors}"
            color = colors[0]
            mask = cell == color
            assert _component_size(mask) == int(mask.sum()), "cell object must be 4-connected"
            cells.append((i, j, color, mask))
            color_counts[color] += 1

    out = g.copy()
    idx = 0
    for i, (r0, r1) in enumerate(row_bands):
        for j, (c0, c1) in enumerate(col_bands):
            _ci, _cj, color, mask = cells[idx]
            idx += 1
            if color_counts[color] != 1:
                continue
            cell_out = out[r0 : r1 + 1, c0 : c1 + 1]
            cell_out[mask] = GREEN if (i + j) % 2 == 0 else RED
    return out


def _random_shape(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    inner_h = h - 2 if h >= 5 else h
    inner_w = w - 2 if w >= 5 else w
    assert inner_h >= 2 and inner_w >= 2

    max_area = min(inner_h * inner_w - 1, 6)
    target = int(rng.integers(1, max_area + 1))
    base = np.zeros((inner_h, inner_w), dtype=bool)

    start = (int(rng.integers(inner_h)), int(rng.integers(inner_w)))
    cells = {start}
    frontier = set()
    while len(cells) < target:
        frontier.clear()
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < inner_h and 0 <= cc < inner_w and (rr, cc) not in cells:
                    frontier.add((rr, cc))
        if not frontier:
            break
        choices = list(frontier)
        cells.add(choices[int(rng.integers(len(choices)))])

    for r, c in cells:
        base[r, c] = True

    mask = np.zeros((h, w), dtype=bool)
    row_off = 1 if h >= 5 else 0
    col_off = 1 if w >= 5 else 0
    mask[row_off : row_off + inner_h, col_off : col_off + inner_w] = base
    return mask


def _partition_repeated_positions(rng: np.random.Generator, positions: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    remaining = positions[:]
    groups: list[list[tuple[int, int]]] = []
    while remaining:
        if len(remaining) <= 3:
            size = len(remaining)
        else:
            size = int(rng.integers(2, min(4, len(remaining)) + 1))
            if len(remaining) - size == 1:
                size += 1
        group = remaining[:size]
        remaining = remaining[size:]
        assert len(group) >= 2
        groups.append(group)
    return groups


def _expected_output(inp: np.ndarray) -> np.ndarray:
    return main(inp)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n_rows = int(rng.integers(2, 4))
    n_cols = int(rng.integers(2, 4))
    total = n_rows * n_cols
    cell_h = int(rng.integers(4, 8))
    cell_w = int(rng.integers(4, 8))

    positions = [(i, j) for i in range(n_rows) for j in range(n_cols)]
    even_positions = [p for p in positions if (p[0] + p[1]) % 2 == 0]
    odd_positions = [p for p in positions if (p[0] + p[1]) % 2 == 1]
    if not even_positions or not odd_positions or total < 4:
        return None

    unique_count = int(rng.integers(2, total - 1))
    unique_count = min(unique_count, total - 2)
    unique_count = min(unique_count, len(PALETTE) - 1)
    if unique_count < 2:
        return None

    for _ in range(20):
        unique_positions = [even_positions[int(rng.integers(len(even_positions)))], odd_positions[int(rng.integers(len(odd_positions)))]]
        remaining_choices = [p for p in positions if p not in unique_positions]
        extra = unique_count - 2
        if extra > len(remaining_choices):
            continue
        if extra:
            pick = rng.choice(len(remaining_choices), size=extra, replace=False)
            unique_positions.extend(remaining_choices[int(i)] for i in np.atleast_1d(pick))
        unique_positions = list(dict.fromkeys(unique_positions))
        if len(unique_positions) < unique_count:
            continue
        repeated_positions = [p for p in positions if p not in unique_positions]
        if len(repeated_positions) >= 2:
            break
    else:
        return None

    repeated_groups = _partition_repeated_positions(rng, repeated_positions)
    needed_colors = len(unique_positions) + len(repeated_groups)
    if needed_colors > len(PALETTE):
        return None
    colors = list(rng.choice(PALETTE, size=needed_colors, replace=False))

    unique_color_map = {pos: int(color) for pos, color in zip(unique_positions, colors[: len(unique_positions)])}
    rep_colors = colors[len(unique_positions) :]
    repeated_color_map: dict[tuple[int, int], int] = {}
    for color, group in zip(rep_colors, repeated_groups):
        for pos in group:
            repeated_color_map[pos] = int(color)

    height = n_rows * cell_h + (n_rows - 1)
    width = n_cols * cell_w + (n_cols - 1)
    inp = np.full((height, width), BLACK, dtype=int)
    for r in range(cell_h, height, cell_h + 1):
        inp[r, :] = GREY
    for c in range(cell_w, width, cell_w + 1):
        inp[:, c] = GREY

    for i in range(n_rows):
        for j in range(n_cols):
            r0 = i * (cell_h + 1)
            c0 = j * (cell_w + 1)
            pos = (i, j)
            color = unique_color_map[pos] if pos in unique_color_map else repeated_color_map[pos]
            mask = _random_shape(rng, cell_h, cell_w)
            inp[r0 : r0 + cell_h, c0 : c0 + cell_w][mask] = color

    try:
        out = _expected_output(inp)
    except Exception:
        return None

    unique_out_colors = set(map(int, np.unique(out))) - {BLACK, GREY}
    if RED not in unique_out_colors or GREEN not in unique_out_colors:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
