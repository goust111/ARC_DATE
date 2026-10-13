from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
AZURE = 8
MAROON = 9

NOISE_COLORS = [BLUE, RED, GREEN, YELLOW, MAGENTA, MAROON]

GRID_H_RANGE = (18, 22)
GRID_W_RANGE = (20, 28)
MIN_PANEL_WIDTH = 9
FRAME_SIZE_RANGE = (4, 7)
NOISE_COUNT_RANGE = (8, 15)
GEN_RETRIES = 400
PANEL_RETRIES = 100


def _draw_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _find_divider(grid: np.ndarray) -> tuple[int, int]:
    same_as_top = np.all(grid == grid[0:1, :], axis=0)
    candidates = np.flatnonzero(same_as_top & (grid[0, :] != BLACK))
    assert len(candidates) == 1, "expected exactly one full-height non-black divider"
    divider_col = int(candidates[0])
    return divider_col, int(grid[0, divider_col])


def _frame_bbox(panel: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(panel == AZURE)
    assert coords.size > 0, "expected an azure frame in each panel"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))

    expected = np.zeros_like(panel, dtype=bool)
    _draw_frame(expected, r0, c0, r1, c1, True)
    assert np.array_equal(panel == AZURE, expected), "azure pixels must form one hollow rectangle"
    return r0, c0, r1, c1


def _majority_noise_color(panel: np.ndarray) -> int:
    values = panel[(panel != BLACK) & (panel != AZURE)]
    assert values.size > 0, "expected noise markers in each panel"
    counts = Counter(int(v) for v in values)
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    assert len(winners) == 1, "noise colors must have a unique majority"
    return int(winners[0])


def _paint_result_frame(
    out: np.ndarray,
    row_offset: int,
    col_offset: int,
    bbox: tuple[int, int, int, int],
    border_color: int,
    fill_color: int,
) -> None:
    r0, c0, r1, c1 = bbox
    r0 += row_offset
    r1 += row_offset
    c0 += col_offset
    c1 += col_offset
    _draw_frame(out, r0, c0, r1, c1, border_color)
    out[r0 + 1 : r1, c0 + 1 : c1] = fill_color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    divider_col, divider_color = _find_divider(g)
    left = g[:, :divider_col]
    right = g[:, divider_col + 1 :]
    assert left.size > 0 and right.size > 0, "divider must split the grid into two panels"

    left_bbox = _frame_bbox(left)
    right_bbox = _frame_bbox(right)
    left_major = _majority_noise_color(left)
    right_major = _majority_noise_color(right)
    assert left_major != right_major, "panel majority colors must differ"

    out = np.zeros_like(g)
    out[:, divider_col] = divider_color
    _paint_result_frame(out, 0, 0, left_bbox, left_major, right_major)
    _paint_result_frame(out, 0, divider_col + 1, right_bbox, right_major, left_major)
    return out


def _noise_counts(total: int, n_colors: int) -> list[int]:
    assert 2 <= n_colors <= 3
    if n_colors == 2:
        majority = random.randint(total // 2 + 1, total - 1)
        return [majority, total - majority]

    for _ in range(100):
        majority = random.randint(max(3, total // 3 + 1), total - 2)
        remaining = total - majority
        first = random.randint(1, remaining - 1)
        second = remaining - first
        if majority > max(first, second):
            return [majority, first, second]
    return [total - 2, 1, 1]


def _candidate_noise_cells(height: int, width: int, bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            if r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1:
                continue
            cells.append((r, c))
    return cells


def _sample_isolated_cells(candidates: list[tuple[int, int]], n: int) -> list[tuple[int, int]] | None:
    for _ in range(80):
        order = candidates[:]
        random.shuffle(order)
        chosen: list[tuple[int, int]] = []
        blocked: set[tuple[int, int]] = set()
        for cell in order:
            if cell in blocked:
                continue
            chosen.append(cell)
            if len(chosen) == n:
                return chosen
            r, c = cell
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    blocked.add((r + dr, c + dc))
    return None


def _build_panel(height: int, width: int, majority_color: int) -> np.ndarray | None:
    if width < MIN_PANEL_WIDTH:
        return None

    max_frame_h = min(FRAME_SIZE_RANGE[1], height - 2)
    max_frame_w = min(FRAME_SIZE_RANGE[1], width - 2)
    if max_frame_h < FRAME_SIZE_RANGE[0] or max_frame_w < FRAME_SIZE_RANGE[0]:
        return None

    for _ in range(PANEL_RETRIES):
        panel = np.zeros((height, width), dtype=int)

        frame_h = random.randint(FRAME_SIZE_RANGE[0], max_frame_h)
        frame_w = random.randint(FRAME_SIZE_RANGE[0], max_frame_w)
        r0 = random.randint(1, height - frame_h - 1)
        c0 = random.randint(1, width - frame_w - 1)
        r1 = r0 + frame_h - 1
        c1 = c0 + frame_w - 1
        bbox = (r0, c0, r1, c1)
        _draw_frame(panel, r0, c0, r1, c1, AZURE)

        n_colors = random.choice([2, 3])
        other_colors = random.sample([c for c in NOISE_COLORS if c != majority_color], n_colors - 1)
        counts = _noise_counts(random.randint(*NOISE_COUNT_RANGE), n_colors)
        cells = _sample_isolated_cells(_candidate_noise_cells(height, width, bbox), sum(counts))
        if cells is None:
            continue

        random.shuffle(cells)
        colors = [majority_color] + other_colors
        index = 0
        for color, count in zip(colors, counts):
            for r, c in cells[index : index + count]:
                panel[r, c] = color
            index += count
        return panel

    return None


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    divider_col = random.randint(MIN_PANEL_WIDTH, width - MIN_PANEL_WIDTH - 1)
    left_width = divider_col
    right_width = width - divider_col - 1

    left_major, right_major = random.sample(NOISE_COLORS, 2)
    left_panel = _build_panel(height, left_width, left_major)
    right_panel = _build_panel(height, right_width, right_major)
    if left_panel is None or right_panel is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[:, :divider_col] = left_panel
    grid[:, divider_col] = GRAY
    grid[:, divider_col + 1 :] = right_panel

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
