import random
from typing import Iterable

import numpy as np


BG = 0
SEP = 3
PALETTE = [1, 2, 4, 5, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def _separator_color(grid: np.ndarray) -> int:
    row_colors = []
    for row in grid:
        if np.all(row == row[0]) and int(row[0]) != BG:
            row_colors.append(int(row[0]))
    col_colors = []
    for col in grid.T:
        if np.all(col == col[0]) and int(col[0]) != BG:
            col_colors.append(int(col[0]))
    common = set(row_colors) & set(col_colors)
    assert len(common) == 1, "expected one separator color shared by full rows and columns"
    return common.pop()


def _intervals(seps: list[int], n: int) -> list[tuple[int, int]]:
    bounds = [-1] + seps + [n]
    parts = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a > 1:
            parts.append((a + 1, b))
    assert parts, "expected at least one panel interval"
    return parts


def _panel_intervals(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    sep = _separator_color(grid)
    row_seps = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == sep)]
    col_seps = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == sep)]
    row_parts = _intervals(row_seps, grid.shape[0])
    col_parts = _intervals(col_seps, grid.shape[1])
    heights = {r1 - r0 for r0, r1 in row_parts}
    widths = {c1 - c0 for c0, c1 in col_parts}
    assert len(heights) == 1 and heights == widths, "panels must be equal squares"
    return row_parts, col_parts


def _panel_direction(mask: np.ndarray) -> str:
    n = mask.shape[0]
    half = n // 2
    rows, cols = np.where(mask)
    assert len(rows) > 0, "panel must contain a foreground pattern"

    top = bool(np.all(rows < half))
    bottom = bool(np.all(rows >= half))
    left = bool(np.all(cols < half))
    right = bool(np.all(cols >= half))
    flags = {"up": top, "down": bottom, "left": left, "right": right}
    found = [name for name, ok in flags.items() if ok]
    assert len(found) == 1, "pattern must be confined to exactly one side"
    return found[0]


def _mirror_fill(panel: np.ndarray) -> np.ndarray:
    mask = panel != BG
    direction = _panel_direction(mask)
    out = panel.copy()
    n = panel.shape[0]
    half = n // 2

    if direction == "up":
        src = panel[:half, :]
        dst = np.flipud(src)
        fill = dst != BG
        out[half:, :][fill] = dst[fill]
    elif direction == "down":
        src = panel[half:, :]
        dst = np.flipud(src)
        fill = dst != BG
        out[:half, :][fill] = dst[fill]
    elif direction == "left":
        src = panel[:, :half]
        dst = np.fliplr(src)
        fill = dst != BG
        out[:, half:][fill] = dst[fill]
    else:
        src = panel[:, half:]
        dst = np.fliplr(src)
        fill = dst != BG
        out[:, :half][fill] = dst[fill]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    row_parts, col_parts = _panel_intervals(g)
    out = g.copy()

    for r0, r1 in row_parts:
        for c0, c1 in col_parts:
            panel = g[r0:r1, c0:c1]
            assert panel.shape[0] == panel.shape[1] and panel.shape[0] % 2 == 0, "expected even square panels"
            out[r0:r1, c0:c1] = _mirror_fill(panel)
    return out


def _grow_connected_mask(h: int, w: int, min_cells: int, max_cells: int) -> np.ndarray:
    total = h * w
    target = random.randint(min_cells, min(max_cells, total))
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    frontier = {start}

    while int(mask.sum()) < target:
        candidates = set()
        for r, c in frontier:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    candidates.add((nr, nc))
        if not candidates:
            break
        cell = random.choice(list(candidates))
        mask[cell] = True
        frontier.add(cell)

    return mask


def _spans_both_sides(mask: np.ndarray, axis: str) -> bool:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return False
    h, w = mask.shape
    if axis in {"up", "down"}:
        return int(cols.min()) < w // 2 and int(cols.max()) >= w // 2
    return int(rows.min()) < h // 2 and int(rows.max()) >= h // 2


def _colorize(mask: np.ndarray, colors: Iterable[int]) -> np.ndarray:
    color_list = list(colors)
    out = np.zeros(mask.shape, dtype=int)
    coords = list(zip(*np.where(mask)))
    random.shuffle(coords)
    if not coords:
        return out

    chunk_count = random.randint(1, min(3, len(color_list)))
    chosen = random.sample(color_list, chunk_count)
    splits = sorted(random.sample(range(1, len(coords)), chunk_count - 1)) if len(coords) > 1 and chunk_count > 1 else []
    groups = np.split(np.array(coords, dtype=int), splits)
    for color, group in zip(chosen, groups):
        for r, c in group:
            out[int(r), int(c)] = color
    return out


def _make_half_pattern(size: int, direction: str) -> np.ndarray:
    half = size // 2
    if direction in {"up", "down"}:
        h, w = half, size
    else:
        h, w = size, half

    min_cells = max(4, (h * w) // 5)
    max_cells = max(min_cells, (h * w * 3) // 5)
    for _ in range(80):
        mask = _grow_connected_mask(h, w, min_cells, max_cells)
        if _spans_both_sides(mask, direction):
            return _colorize(mask, PALETTE)
    raise RuntimeError("failed to build an unambiguous half pattern")


def _place_pattern(size: int, direction: str) -> np.ndarray:
    panel = np.zeros((size, size), dtype=int)
    half_pat = _make_half_pattern(size, direction)
    half = size // 2
    if direction == "up":
        panel[:half, :] = half_pat
    elif direction == "down":
        panel[half:, :] = half_pat
    elif direction == "left":
        panel[:, :half] = half_pat
    else:
        panel[:, half:] = half_pat
    return panel


def _gen_attempt() -> np.ndarray:
    panel_size = random.choice([6, 8, 10])
    panel_rows = random.randint(2, 4)
    panel_cols = random.randint(2, 4)
    h = panel_rows * panel_size + (panel_rows - 1)
    w = panel_cols * panel_size + (panel_cols - 1)
    grid = np.zeros((h, w), dtype=int)

    for r in range(1, panel_rows):
        grid[r * panel_size + (r - 1), :] = SEP
    for c in range(1, panel_cols):
        grid[:, c * panel_size + (c - 1)] = SEP

    directions = ["up", "down", "left", "right"]
    total = panel_rows * panel_cols
    assigned = directions.copy()
    assigned.extend(random.choice(directions) for _ in range(total - 4))
    random.shuffle(assigned)

    k = 0
    for pr in range(panel_rows):
        for pc in range(panel_cols):
            r0 = pr * (panel_size + 1)
            c0 = pc * (panel_size + 1)
            grid[r0 : r0 + panel_size, c0 : c0 + panel_size] = _place_pattern(panel_size, assigned[k])
            k += 1

    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            return _gen_attempt()
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input")
