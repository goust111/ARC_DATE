from __future__ import annotations

import random

import numpy as np


ORANGE = 7
GREY = 5
FOREGROUND_COLORS = (0, 1, 2, 3, 4, 6, 8, 9)

PANEL_HEIGHT_RANGE = (4, 8)
PANEL_WIDTH_RANGE = (4, 8)
SHAPE_ATTEMPTS = 80
GENERATION_ATTEMPTS = 300

LAYOUTS_BY_BRANCH = {
    "horizontal": (
        np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
        np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0]], dtype=bool),
        np.array([[0, 1, 1], [0, 1, 1], [0, 1, 1]], dtype=bool),
    ),
    "vertical": (
        np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=bool),
        np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0]], dtype=bool),
        np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1]], dtype=bool),
    ),
}


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _intervals(separators: list[int], length: int) -> list[tuple[int, int]]:
    bounds = [-1, *separators, int(length)]
    spans = [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]
    sizes = {end - start for start, end in spans}
    assert len(spans) == 3 and len(sizes) == 1 and min(sizes) > 0
    return spans


def _panel_state(panel: np.ndarray) -> tuple[np.ndarray, int] | None:
    if np.all(panel == ORANGE):
        return None

    colors = set(map(int, np.unique(panel)))
    colors.discard(ORANGE)
    assert len(colors) == 1, "each non-empty panel must contain one foreground color"
    color = int(next(iter(colors)))
    mask = panel == color
    assert mask.any(), "non-empty panel must expose a foreground mask"
    return mask, color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each non-empty panel's shape into adjacent blank panels using a mirrored template.

    Input structure:
    - Orange background with two grey separator rows and two grey separator columns.
    - The separators partition the grid into a 3x3 array of equal-sized panels.
    - Every panel is either blank orange or contains one monochrome foreground object.
    - Each blank panel has exactly one non-empty orthogonal neighbor.

    Rule:
    - Keep all existing non-empty panels unchanged.
    - For each blank panel, find its unique non-empty orthogonal neighbor.
    - Copy that neighbor's object mask into the blank panel using the same color.
    - Mirror the mask horizontally for left/right neighbors, vertically for up/down neighbors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]
    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    assert len(sep_rows) == 2 and len(sep_cols) == 2, "expected a 3x3 panel grid"

    row_spans = _intervals(sep_rows, g.shape[0])
    col_spans = _intervals(sep_cols, g.shape[1])

    states: list[list[tuple[np.ndarray, int] | None]] = [[None] * 3 for _ in range(3)]
    for pr, (r0, r1) in enumerate(row_spans):
        for pc, (c0, c1) in enumerate(col_spans):
            states[pr][pc] = _panel_state(g[r0:r1, c0:c1])

    out = g.copy()
    blank_count = 0
    for pr, (r0, r1) in enumerate(row_spans):
        for pc, (c0, c1) in enumerate(col_spans):
            if states[pr][pc] is not None:
                continue

            blank_count += 1
            filled_neighbors: list[tuple[int, int]] = []
            for nr, nc in _neighbors4(pr, pc, 3, 3):
                if states[nr][nc] is not None:
                    filled_neighbors.append((nr, nc))
            assert len(filled_neighbors) == 1, "blank panel must have a unique non-empty neighbor"

            nr, nc = filled_neighbors[0]
            neighbor_mask, neighbor_color = states[nr][nc]
            if nr == pr:
                new_mask = np.fliplr(neighbor_mask)
            else:
                new_mask = np.flipud(neighbor_mask)

            panel = np.full((r1 - r0, c1 - c0), ORANGE, dtype=int)
            panel[new_mask] = int(neighbor_color)
            out[r0:r1, c0:c1] = panel

    assert blank_count > 0, "expected at least one blank panel"
    return out


def _grow_connected_mask(height: int, width: int, area: int) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    frontier = set(_neighbors4(start[0], start[1], height, width))

    while int(mask.sum()) < area:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        if mask[cell]:
            continue
        mask[cell] = True
        for nb in _neighbors4(cell[0], cell[1], height, width):
            if not mask[nb]:
                frontier.add(nb)
    return mask


def _random_shape(panel_h: int, panel_w: int) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        box_h = random.randint(2, panel_h)
        box_w = random.randint(2, panel_w)
        box_area = box_h * box_w
        low = max(3, (box_area + 4) // 5)
        high = min(box_area - 1, max(low, (3 * box_area) // 5))
        if low > high:
            continue

        target_area = random.randint(low, high)
        inner = _grow_connected_mask(box_h, box_w, target_area)
        if inner is None:
            continue

        top = random.randint(0, panel_h - box_h)
        left = random.randint(0, panel_w - box_w)
        full = np.zeros((panel_h, panel_w), dtype=bool)
        full[top : top + box_h, left : left + box_w] = inner

        if np.array_equal(full, np.fliplr(full)):
            continue
        if np.array_equal(full, np.flipud(full)):
            continue
        return full
    return None


def _choose_layout() -> np.ndarray:
    branch = random.choice(("horizontal", "vertical"))
    return random.choice(LAYOUTS_BY_BRANCH[branch]).copy()


def _render_grid(panel_data: list[list[tuple[np.ndarray, int] | None]], panel_h: int, panel_w: int) -> np.ndarray:
    grid_h = 3 * panel_h + 2
    grid_w = 3 * panel_w + 2
    grid = np.full((grid_h, grid_w), ORANGE, dtype=int)
    grid[panel_h, :] = GREY
    grid[2 * panel_h + 1, :] = GREY
    grid[:, panel_w] = GREY
    grid[:, 2 * panel_w + 1] = GREY

    for pr in range(3):
        for pc in range(3):
            state = panel_data[pr][pc]
            if state is None:
                continue
            mask, color = state
            r0 = pr * (panel_h + 1)
            c0 = pc * (panel_w + 1)
            panel = np.full((panel_h, panel_w), ORANGE, dtype=int)
            panel[mask] = int(color)
            grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = panel

    return grid


def _expected_from_layout(panel_data: list[list[tuple[np.ndarray, int] | None]]) -> list[list[tuple[np.ndarray, int] | None]]:
    expected = [[panel_data[r][c] for c in range(3)] for r in range(3)]
    for pr in range(3):
        for pc in range(3):
            if panel_data[pr][pc] is not None:
                continue

            filled_neighbors = [(nr, nc) for nr, nc in _neighbors4(pr, pc, 3, 3) if panel_data[nr][nc] is not None]
            assert len(filled_neighbors) == 1
            nr, nc = filled_neighbors[0]
            src_mask, src_color = panel_data[nr][nc]
            if nr == pr:
                expected[pr][pc] = (np.fliplr(src_mask), src_color)
            else:
                expected[pr][pc] = (np.flipud(src_mask), src_color)
    return expected


def _make_example() -> tuple[np.ndarray, np.ndarray] | None:
    panel_h = random.randint(*PANEL_HEIGHT_RANGE)
    panel_w = random.randint(*PANEL_WIDTH_RANGE)
    layout = _choose_layout()

    filled = [(r, c) for r in range(3) for c in range(3) if layout[r, c]]
    colors = random.sample(FOREGROUND_COLORS, len(filled))

    panel_data: list[list[tuple[np.ndarray, int] | None]] = [[None] * 3 for _ in range(3)]
    for color, (pr, pc) in zip(colors, filled, strict=False):
        shape = _random_shape(panel_h, panel_w)
        if shape is None:
            return None
        panel_data[pr][pc] = (shape, int(color))

    inp = _render_grid(panel_data, panel_h, panel_w)
    expected = _render_grid(_expected_from_layout(panel_data), panel_h, panel_w)
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        example = _make_example()
        if example is None:
            continue
        inp, expected = example
        try:
            if np.array_equal(main(inp), expected):
                return inp
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input")
