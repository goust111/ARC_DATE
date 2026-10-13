from __future__ import annotations

import random

import numpy as np


AZURE = 8
GREY = 5
RED = 2

DIRECTIONS = {
    (-1, 0): 0,   # up: no rotation
    (0, 1): -1,   # right: clockwise 90 degrees
    (1, 0): 2,    # down: 180 degrees
    (0, -1): 1,   # left: counter-clockwise 90 degrees
}
GEN_COLORS = [0, 1, 3, 4, 6, 7, 9]
PANEL_SIZES = (5, 7, 9)
GEN_ATTEMPTS = 300


def _runs_excluding(indices: set[int], n: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i in range(n):
        if i in indices:
            if start is not None:
                runs.append((start, i))
                start = None
        elif start is None:
            start = i
    if start is not None:
        runs.append((start, n))
    return runs


def _panel_intervals(g: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = g.shape
    row_seps = {r for r in range(h) if np.all(g[r, :] == GREY)}
    col_seps = {c for c in range(w) if np.all(g[:, c] == GREY)}
    row_intervals = _runs_excluding(row_seps, h)
    col_intervals = _runs_excluding(col_seps, w)

    assert row_intervals and col_intervals, "expected at least one panel"
    heights = {r1 - r0 for r0, r1 in row_intervals}
    widths = {c1 - c0 for c0, c1 in col_intervals}
    assert len(heights) == 1 and len(widths) == 1, "panels must be equal sized"
    side = next(iter(heights))
    assert side == next(iter(widths)), "panels must be square"
    assert side % 2 == 1 and side >= 3, "panels must have an odd side length"
    return row_intervals, col_intervals


def _pointer_cells(side: int, direction: tuple[int, int], length: int) -> set[tuple[int, int]]:
    center = side // 2
    dr, dc = direction
    return {(center + i * dr, center + i * dc) for i in range(length)}


def _read_rotation(block: np.ndarray) -> int:
    side = block.shape[0]
    red_cells = {tuple(map(int, rc)) for rc in np.argwhere(block == RED)}
    assert red_cells, "each panel must contain a red clock hand"
    for direction, krot in DIRECTIONS.items():
        for length in (2, 3):
            if red_cells == _pointer_cells(side, direction, length):
                return krot
    raise AssertionError("red clock hand must be a 2- or 3-pixel cardinal segment from the center")


def _rotate_non_red_panel(block: np.ndarray, krot: int) -> np.ndarray:
    payload = np.full(block.shape, AZURE, dtype=int)
    keep = (block != AZURE) & (block != RED)
    payload[keep] = block[keep]
    return np.rot90(payload, krot)


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each panel's colored pattern according to its red clock hand.

    Input structure:
    - Azure is the background. Full Grey separator rows and columns divide the grid into equal odd-sided square panels.
    - Each panel contains one non-red foreground pattern and one Red clock hand: a 2- or 3-pixel line segment from the panel center.
    - The Red hand points exactly up, right, down, or left.

    Rule:
    - In each panel, read the Red hand direction as a rotation instruction: up = 0, right = clockwise 90, down = 180, left = counter-clockwise 90.
    - Erase the Red hand and rotate every non-Red foreground pixel in that panel around the panel center by the indicated angle.
    - Keep all Azure background cells and all Grey separator lines fixed.

    Key insight:
    The Red clock hand is an in-panel direction marker; its orientation directly encodes how much to rotate the panel's actual foreground pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    row_intervals, col_intervals = _panel_intervals(g)

    out = g.copy()
    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            block = g[r0:r1, c0:c1]
            krot = _read_rotation(block)
            out[r0:r1, c0:c1] = _rotate_non_red_panel(block, krot)
    return out


# --------------- generator ---------------


def _neighbors4(cell: tuple[int, int], side: int):
    r, c = cell
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < side and 0 <= nc < side:
            yield nr, nc


def _grow_cells(side: int, blocked: set[tuple[int, int]], target: int) -> set[tuple[int, int]] | None:
    available = [(r, c) for r in range(side) for c in range(side) if (r, c) not in blocked]
    if len(available) < target:
        return None

    start = random.choice(available)
    cells = {start}
    frontier = {nb for nb in _neighbors4(start, side) if nb not in blocked}
    while len(cells) < target:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nb in _neighbors4(cell, side):
            if nb not in blocked and nb not in cells:
                frontier.add(nb)
    return cells


def _make_line_cells(side: int, blocked: set[tuple[int, int]], target: int) -> set[tuple[int, int]] | None:
    for _ in range(80):
        r = random.randrange(side)
        c = random.randrange(side)
        if (r, c) in blocked:
            continue
        cells = {(r, c)}
        dr, dc = random.choice(((1, 0), (0, 1)))
        while len(cells) < target:
            if random.random() < 0.25:
                dr, dc = dc, dr
            nr, nc = r + dr, c + dc
            if not (0 <= nr < side and 0 <= nc < side) or (nr, nc) in blocked:
                dr, dc = -dr, -dc
                nr, nc = r + dr, c + dc
            if not (0 <= nr < side and 0 <= nc < side) or (nr, nc) in blocked:
                break
            r, c = nr, nc
            cells.add((r, c))
        if len(cells) == target:
            return cells
    return None


def _paint_cells(side: int, cells: set[tuple[int, int]]) -> np.ndarray:
    panel = np.full((side, side), AZURE, dtype=int)
    colors = random.sample(GEN_COLORS, random.randint(1, min(3, len(GEN_COLORS))))
    ordered = sorted(cells, key=lambda rc: (rc[0] + rc[1], rc[0] - rc[1], rc[0]))
    shift = random.randrange(len(colors))
    for i, (r, c) in enumerate(ordered):
        if len(colors) == 1:
            color = colors[0]
        else:
            color = colors[(i + shift) % len(colors)]
        panel[r, c] = color

    if len(colors) > 1:
        for color in colors:
            if color not in set(map(int, panel.ravel())):
                r, c = random.choice(tuple(cells))
                panel[r, c] = color
    return panel


def _make_pattern(side: int, blocked: set[tuple[int, int]], krot: int) -> np.ndarray | None:
    max_cells = min(side * side - len(blocked), 3 * side - 2)
    min_cells = min(max_cells, max(4, side - 1))
    if min_cells > max_cells:
        return None

    for _ in range(120):
        target = random.randint(min_cells, max_cells)
        if random.random() < 0.45:
            cells = _make_line_cells(side, blocked, target)
        else:
            cells = _grow_cells(side, blocked, target)
        if cells is None:
            continue

        if random.random() < 0.35 and len(cells) + 1 < side * side - len(blocked):
            free = [
                (r, c)
                for r in range(side)
                for c in range(side)
                if (r, c) not in blocked and (r, c) not in cells
            ]
            if free:
                cells.add(random.choice(free))

        panel = _paint_cells(side, cells)
        if krot == 0 or not np.array_equal(panel, np.rot90(panel, krot)):
            return panel
    return None


def _valid_panel_counts(side: int) -> list[tuple[int, int]]:
    choices: list[tuple[int, int]] = []
    for rows in range(1, 6):
        for cols in range(1, 6):
            height = rows * side + rows - 1
            width = cols * side + cols - 1
            panels = rows * cols
            if height <= 30 and width <= 30 and 4 <= panels <= 12:
                choices.append((rows, cols))
    return choices


def _stitch_grid(panels: list[list[np.ndarray]], side: int) -> np.ndarray:
    rows = len(panels)
    cols = len(panels[0])
    h = rows * side + rows - 1
    w = cols * side + cols - 1
    grid = np.full((h, w), AZURE, dtype=int)
    for i in range(1, rows):
        grid[i * side + i - 1, :] = GREY
    for j in range(1, cols):
        grid[:, j * side + j - 1] = GREY
    for pr in range(rows):
        for pc in range(cols):
            r0 = pr * (side + 1)
            c0 = pc * (side + 1)
            grid[r0 : r0 + side, c0 : c0 + side] = panels[pr][pc]
    return grid


def _gen_attempt() -> np.ndarray | None:
    side = random.choice(PANEL_SIZES)
    counts = _valid_panel_counts(side)
    if not counts:
        return None
    panel_rows, panel_cols = random.choice(counts)
    panel_count = panel_rows * panel_cols

    direction_keys = list(DIRECTIONS)
    directions = direction_keys[:]
    while len(directions) < panel_count:
        directions.append(random.choice(direction_keys))
    random.shuffle(directions)

    panels: list[list[np.ndarray]] = []
    direction_index = 0
    for _pr in range(panel_rows):
        row_panels: list[np.ndarray] = []
        for _pc in range(panel_cols):
            direction = directions[direction_index]
            direction_index += 1
            krot = DIRECTIONS[direction]
            length = random.randint(2, min(3, side // 2 + 1))
            hand = _pointer_cells(side, direction, length)
            panel = _make_pattern(side, hand, krot)
            if panel is None:
                return None
            for r, c in hand:
                panel[r, c] = RED
            row_panels.append(panel)
        panels.append(row_panels)

    grid = _stitch_grid(panels, side)
    try:
        solved = main(grid)
    except AssertionError:
        return None

    expected_panels = [
        [_rotate_non_red_panel(panel, _read_rotation(panel)) for panel in row]
        for row in panels
    ]
    expected = _stitch_grid(expected_panels, side)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid seed2047415201/15/2 input")
