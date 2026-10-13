from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


SEPARATOR_COLOR = 5
PANEL_SIZE_RANGE = (3, 5)
PANEL_ROW_RANGE = (2, 4)
GENERATION_ATTEMPTS = 400
PATTERN_ATTEMPTS = 80


def _regular_intervals(separators: list[int], length: int) -> list[tuple[int, int]]:
    bounds = [-1, *sorted(separators), int(length)]
    intervals = [(a + 1, b) for a, b in zip(bounds, bounds[1:]) if b > a + 1]
    assert intervals, "expected non-empty intervals"
    return intervals


def _separator_color(grid: np.ndarray) -> int:
    row_counts: Counter[int] = Counter()
    col_counts: Counter[int] = Counter()

    for r in range(grid.shape[0]):
        row = grid[r]
        if np.all(row == row[0]):
            row_counts[int(row[0])] += 1

    for c in range(grid.shape[1]):
        col = grid[:, c]
        if np.all(col == col[0]):
            col_counts[int(col[0])] += 1

    candidates = set(row_counts) & set(col_counts)
    assert candidates, "expected a separator color spanning full rows and columns"
    return max(
        candidates,
        key=lambda color: (
            row_counts[color] + col_counts[color],
            int(np.count_nonzero(grid == color)),
        ),
    )


def _extract_structure(
    grid: np.ndarray,
) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], np.ndarray]:
    sep = _separator_color(grid)
    row_separators = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == sep)]
    col_separators = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == sep)]

    row_intervals = _regular_intervals(row_separators, grid.shape[0])
    col_intervals = _regular_intervals(col_separators, grid.shape[1])
    widths = [c1 - c0 for c0, c1 in col_intervals]
    assert len(set(widths)) == 1, "panel columns must have equal width"
    panel_size = int(widths[0])

    heights = [r1 - r0 for r0, r1 in row_intervals]
    assert len(row_intervals) >= 2, "expected panel rows plus one legend row"
    assert heights[-1] == 1, "bottom legend must occupy one row"
    assert all(h == panel_size for h in heights[:-1]), "panel rows must be square"
    assert len(col_intervals) == panel_size, "legend encoding expects one column group per local row"

    legend_row = grid[row_intervals[-1][0], :]
    legend = np.zeros((panel_size, panel_size), dtype=int)
    for local_r, (c0, c1) in enumerate(col_intervals):
        segment = legend_row[c0:c1]
        assert len(segment) == panel_size
        legend[local_r] = segment

    return panel_size, row_intervals[:-1], col_intervals, legend


def _majority_template(panels: np.ndarray) -> np.ndarray:
    n_panels, height, width = panels.shape
    assert n_panels >= 3, "need several panels for majority voting"
    template = np.zeros((height, width), dtype=int)

    for r in range(height):
        for c in range(width):
            values, counts = np.unique(panels[:, r, c], return_counts=True)
            best = int(np.argmax(counts))
            assert int(np.sum(counts == counts[best])) == 1, "majority must be unique"
            template[r, c] = int(values[best])

    return template


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the repeated master panel, identify the unique damaged copy, and
    replace the damaged positions with colors from the bottom legend row.

    Input structure:
    - Equal-size square panels occupy the upper part of the grid and are split by
      grey separator rows and columns.
    - The bottom contains a 1-row legend. It is partitioned by the same separator
      columns into `panel_size` groups of width `panel_size`; group r, offset c
      encodes the color for local coordinate (r, c) inside a panel.
    - All upper panels copy the same master pattern, except one damaged panel in
      which several cells are replaced by the background color.

    Rule:
    - Recover the master by majority vote at each local coordinate across all
      upper panels.
    - Find the unique panel that differs from that master.
    - Output only that damaged panel crop, recoloring each differing position with
      its legend color and leaving all other positions as the recovered master.
    """
    g = np.asarray(grid, dtype=int)
    panel_size, panel_rows, col_intervals, legend = _extract_structure(g)

    panels: list[np.ndarray] = []
    for r0, r1 in panel_rows:
        for c0, c1 in col_intervals:
            panel = g[r0:r1, c0:c1]
            assert panel.shape == (panel_size, panel_size)
            panels.append(panel.copy())

    panel_stack = np.stack(panels, axis=0)
    master = _majority_template(panel_stack)
    mismatch_counts = np.sum(panel_stack != master, axis=(1, 2))
    damaged = np.flatnonzero(mismatch_counts > 0)
    assert len(damaged) == 1, "expected exactly one damaged panel"

    damaged_panel = panel_stack[int(damaged[0])]
    damage_mask = damaged_panel != master
    out = master.copy()
    out[damage_mask] = legend[damage_mask]
    return out


def _neighbors4(r: int, c: int, height: int, width: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < height:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < width:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, height, width):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _grow_blob(allowed: np.ndarray, target: int) -> np.ndarray | None:
    components = [comp for comp in _components(allowed) if len(comp) >= target]
    if not components:
        return None

    comp = random.choice(components)
    comp_set = set(comp)
    seed = random.choice(comp)
    chosen = {seed}
    frontier = {cell for cell in _neighbors4(*seed, *allowed.shape) if cell in comp_set}

    while len(chosen) < target:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        chosen.add(cell)
        for nxt in _neighbors4(*cell, *allowed.shape):
            if nxt in comp_set and nxt not in chosen:
                frontier.add(nxt)

    mask = np.zeros_like(allowed, dtype=bool)
    for r, c in chosen:
        mask[r, c] = True
    return mask


def _make_pattern(size: int, palette: list[int]) -> np.ndarray:
    for _ in range(PATTERN_ATTEMPTS):
        colors = random.sample(palette, random.randint(2, min(4, len(palette))))
        mode = random.choice(("rows", "cols", "checker", "quadrants", "rings", "diagonal"))
        grid = np.full((size, size), colors[0], dtype=int)

        if mode == "rows":
            row_colors = [random.choice(colors) for _ in range(size)]
            for r in range(size):
                grid[r, :] = row_colors[r]
            accent_cols = random.sample(range(size), random.randint(1, min(2, size)))
            for c in accent_cols:
                grid[:, c] = random.choice(colors)
        elif mode == "cols":
            col_colors = [random.choice(colors) for _ in range(size)]
            for c in range(size):
                grid[:, c] = col_colors[c]
            accent_rows = random.sample(range(size), random.randint(1, min(2, size)))
            for r in accent_rows:
                grid[r, :] = random.choice(colors)
        elif mode == "checker":
            period = random.randint(2, min(3, len(colors)))
            for r in range(size):
                for c in range(size):
                    grid[r, c] = colors[(r + c) % period]
            if random.random() < 0.7:
                grid[np.arange(size), np.arange(size)] = random.choice(colors)
        elif mode == "quadrants":
            split_r = random.randint(1, size - 1)
            split_c = random.randint(1, size - 1)
            quad_colors = [colors[i % len(colors)] for i in range(4)]
            grid[:split_r, :split_c] = quad_colors[0]
            grid[:split_r, split_c:] = quad_colors[1]
            grid[split_r:, :split_c] = quad_colors[2]
            grid[split_r:, split_c:] = quad_colors[3]
            if random.random() < 0.6:
                grid[split_r - 1 : split_r + 1, :] = random.choice(colors)
        elif mode == "rings":
            for layer in range((size + 1) // 2):
                color = colors[layer % len(colors)]
                grid[layer : size - layer, layer] = color
                grid[layer : size - layer, size - layer - 1] = color
                grid[layer, layer : size - layer] = color
                grid[size - layer - 1, layer : size - layer] = color
            if size >= 3:
                grid[size // 2, :] = random.choice(colors)
        else:
            for r in range(size):
                for c in range(size):
                    grid[r, c] = colors[(r + 2 * c) % len(colors)]
            if random.random() < 0.6:
                grid[:, size // 2] = random.choice(colors)

        if len(set(map(int, grid.flat))) >= 2:
            return grid

    raise RuntimeError("failed to create a non-degenerate pattern")


def _rectangle_mask(allowed: np.ndarray, max_size: int) -> np.ndarray | None:
    height, width = allowed.shape
    options: list[np.ndarray] = []
    for r0 in range(height):
        for r1 in range(r0, height):
            for c0 in range(width):
                for c1 in range(c0, width):
                    mask = np.zeros_like(allowed, dtype=bool)
                    mask[r0 : r1 + 1, c0 : c1 + 1] = allowed[r0 : r1 + 1, c0 : c1 + 1]
                    count = int(mask.sum())
                    if 1 <= count <= max_size:
                        options.append(mask)
    if not options:
        return None
    return random.choice(options)


def _damage_mask(master: np.ndarray, legend: np.ndarray) -> np.ndarray:
    allowed = legend != master
    candidates = [(int(r), int(c)) for r, c in np.argwhere(allowed)]
    assert candidates, "legend must differ from master somewhere"

    size = master.shape[0]
    max_damage = min(len(candidates), max(1, (size * size) // 2))
    mode = random.choice(("sparse", "blob", "rectangle"))

    if mode == "blob" and max_damage >= 2:
        target = random.randint(2, max_damage)
        blob = _grow_blob(allowed, target)
        if blob is not None and blob.any():
            return blob

    if mode == "rectangle":
        rect = _rectangle_mask(allowed, max_damage)
        if rect is not None and rect.any():
            return rect

    count = random.randint(1, max_damage)
    chosen = random.sample(candidates, count)
    mask = np.zeros_like(allowed, dtype=bool)
    for r, c in chosen:
        mask[r, c] = True
    return mask


def _build_input(
    panel_rows: int,
    panel_size: int,
    background: int,
    master: np.ndarray,
    legend: np.ndarray,
    damage_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    width = panel_size * panel_size + (panel_size - 1)
    height = panel_rows * panel_size + panel_rows + 1
    grid = np.full((height, width), background, dtype=int)

    for k in range(1, panel_size):
        grid[:, k * (panel_size + 1) - 1] = SEPARATOR_COLOR
    for k in range(1, panel_rows + 1):
        grid[k * (panel_size + 1) - 1, :] = SEPARATOR_COLOR

    damaged_row = random.randrange(panel_rows)
    damaged_col = random.randrange(panel_size)

    for pr in range(panel_rows):
        for pc in range(panel_size):
            r0 = pr * (panel_size + 1)
            c0 = pc * (panel_size + 1)
            panel = master.copy()
            if pr == damaged_row and pc == damaged_col:
                panel[damage_mask] = background
            grid[r0 : r0 + panel_size, c0 : c0 + panel_size] = panel

    legend_row = panel_rows * (panel_size + 1)
    for local_r in range(panel_size):
        c0 = local_r * (panel_size + 1)
        grid[legend_row, c0 : c0 + panel_size] = legend[local_r]

    expected = master.copy()
    expected[damage_mask] = legend[damage_mask]
    return grid, expected


def _generate_one() -> np.ndarray | None:
    panel_size = random.randint(*PANEL_SIZE_RANGE)
    panel_rows = random.randint(*PANEL_ROW_RANGE)

    non_separator = [color for color in range(10) if color != SEPARATOR_COLOR]
    background = random.choice(non_separator)
    usable = [color for color in non_separator if color != background]
    master = _make_pattern(panel_size, usable)
    legend = _make_pattern(panel_size, usable)
    damage_mask = _damage_mask(master, legend)
    grid, expected = _build_input(panel_rows, panel_size, background, master, legend, damage_mask)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
