from __future__ import annotations

import random

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Generator configuration.
BG = 0
GRID_SIZE_RANGE = (20, 28)
GROUP_COUNT_CHOICES = (2, 3)
PALETTE_SIZE_CHOICES = (2, 3)
GROUP_POS_COUNT_RANGE = (1, 4)
MAX_GENERATION_ATTEMPTS = 400
MAX_LOCAL_ATTEMPTS = 200


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _color_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    seen = np.zeros((height, width), dtype=bool)
    comps: list[dict] = []

    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            stack = [(int(r), int(c))]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            rows = [row for row, _ in cells]
            cols = [col for _, col in cells]
            comps.append(
                {
                    "color": color,
                    "cells": tuple(cells),
                    "top_left": (min(rows), min(cols)),
                }
            )
    return comps


def _band_groups(values: np.ndarray, bg: int) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for idx, value in enumerate(values.tolist(), start=1):
        color = int(value)
        if color == int(bg):
            continue
        groups.setdefault(color, []).append(int(idx))
    return groups


def _palette_info(grid: np.ndarray, bg: int) -> tuple[np.ndarray, list[int]]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape

    interior = np.argwhere(g[1:, 1:] != bg)
    assert interior.size > 0, "expected a bottom-right palette"
    interior = interior + 1

    r0 = int(interior[:, 0].min())
    c0 = int(interior[:, 1].min())
    r1 = int(interior[:, 0].max())
    c1 = int(interior[:, 1].max())
    assert r1 == height - 1 and c1 == width - 1, "palette must sit in the bottom-right corner"

    size_h = int(r1 - r0 + 1)
    size_w = int(c1 - c0 + 1)
    assert size_h == size_w and size_h in (2, 3), "palette must be 2x2 or 3x3"

    palette = g[r0 : r1 + 1, c0 : c1 + 1]
    entries = _color_components(palette, bg)
    entries.sort(key=lambda comp: comp["top_left"])
    colors = [int(comp["color"]) for comp in entries]
    assert 2 <= len(colors) <= 3, "palette must expose 2-3 visible entries"
    assert len(colors) == len(set(colors)), "palette entries must use distinct colors"

    mask = np.zeros_like(g, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask, colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode active rows/columns from matching border colors and paint their intersections
    with colors remapped by the bottom-right palette order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _dominant_color(g)
    separator = int(g[0, 0])
    assert separator != int(bg), "top-left separator must differ from the background"

    top_groups = _band_groups(g[0, 1:], bg)
    left_groups = _band_groups(g[1:, 0], bg)
    assert top_groups, "expected colored cells on the top band"
    assert left_groups, "expected colored cells on the left band"
    assert set(top_groups) == set(left_groups), "top and left bands must encode the same groups"

    order_by_top = sorted(top_groups, key=lambda color: top_groups[int(color)][0])
    order_by_left = sorted(left_groups, key=lambda color: left_groups[int(color)][0])
    assert order_by_top == order_by_left, "group order must agree between the two bands"

    palette_mask, palette_colors = _palette_info(g, bg)
    assert len(palette_colors) == len(order_by_top), "palette size must match the number of groups"

    out = g.copy()
    for band_color, mapped_color in zip(order_by_top, palette_colors, strict=False):
        for row in left_groups[int(band_color)]:
            for col in top_groups[int(band_color)]:
                if palette_mask[int(row), int(col)]:
                    continue
                out[int(row), int(col)] = int(mapped_color)
    return out


def _sample_palette_positions(size: int, count: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(size) for c in range(size)]
    for _ in range(MAX_LOCAL_ATTEMPTS):
        positions = random.sample(cells, count)
        rows = {r for r, _ in positions}
        cols = {c for _, c in positions}
        if 0 in rows and size - 1 in rows and 0 in cols and size - 1 in cols:
            return sorted((int(r), int(c)) for r, c in positions)
    raise RuntimeError("failed to sample palette positions")


def _sample_group_positions(limit: int, group_count: int) -> list[list[int]]:
    for _ in range(MAX_LOCAL_ATTEMPTS):
        counts = [random.randint(*GROUP_POS_COUNT_RANGE) for _ in range(group_count)]
        if not any(count > 1 for count in counts):
            counts[random.randrange(group_count)] = 2
        if sum(counts) > limit:
            continue

        first_positions = sorted(random.sample(range(1, limit + 1), group_count))
        groups = [[int(first)] for first in first_positions]
        used = set(first_positions)
        ok = True

        for idx in range(group_count - 1, -1, -1):
            need = counts[idx] - 1
            if need == 0:
                continue
            choices = [pos for pos in range(first_positions[idx] + 1, limit + 1) if pos not in used]
            if len(choices) < need:
                ok = False
                break
            picks = random.sample(choices, need)
            groups[idx].extend(int(pos) for pos in picks)
            used.update(picks)

        if not ok:
            continue

        groups = [sorted(group) for group in groups]
        has_gap = any(
            len(group) > 1 and any(b - a > 1 for a, b in zip(group, group[1:], strict=False))
            for group in groups
        )
        if has_gap or random.random() < 0.35:
            return groups

    raise RuntimeError("failed to sample band positions")


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    height = random.randint(*GRID_SIZE_RANGE)
    width = random.randint(*GRID_SIZE_RANGE)
    palette_size = random.choice(PALETTE_SIZE_CHOICES)
    group_count = random.choice(GROUP_COUNT_CHOICES)

    max_row = height - palette_size - 1
    max_col = width - palette_size - 1
    assert max_row >= group_count and max_col >= group_count

    color_pool = list(range(1, 10))
    sampled = random.sample(color_pool, 1 + 2 * group_count)
    separator = int(sampled[0])
    band_colors = [int(color) for color in sampled[1 : 1 + group_count]]
    marker_colors = [int(color) for color in sampled[1 + group_count :]]

    top_groups = _sample_group_positions(max_col, group_count)
    left_groups = _sample_group_positions(max_row, group_count)
    if not any(len(rows) * len(cols) > 1 for rows, cols in zip(left_groups, top_groups, strict=False)):
        raise RuntimeError("sampled only trivial intersections")

    palette_positions = _sample_palette_positions(palette_size, group_count)
    palette_row0 = height - palette_size
    palette_col0 = width - palette_size

    grid = np.full((height, width), BG, dtype=int)
    grid[0, 0] = separator

    for color, cols in zip(band_colors, top_groups, strict=False):
        for col in cols:
            grid[0, int(col)] = int(color)

    for color, rows in zip(band_colors, left_groups, strict=False):
        for row in rows:
            grid[int(row), 0] = int(color)

    for color, (pr, pc) in zip(marker_colors, palette_positions, strict=False):
        grid[palette_row0 + int(pr), palette_col0 + int(pc)] = int(color)

    expected = grid.copy()
    for rows, cols, color in zip(left_groups, top_groups, marker_colors, strict=False):
        for row in rows:
            for col in cols:
                expected[int(row), int(col)] = int(color)

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid, expected = _build_case()
        except RuntimeError:
            continue

        try:
            observed = main(grid)
        except AssertionError:
            continue

        if np.array_equal(observed, expected):
            return grid

    raise RuntimeError("failed to generate a valid input")
