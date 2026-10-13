import random
from collections import deque

import numpy as np


BG_COLOR = 0
ANCHOR_COLOR = 2
SEED_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)

SINGLE_AXIS_LAYOUTS = (1, 2, 3)
BOTH_AXIS_LAYOUTS = ((1, 1), (1, 2), (2, 1), (1, 3), (3, 1))
OUTER_MARGIN_RANGE = (1, 3)
CELL_SIZE_RANGE = (2, 4)
MAX_GENERATION_ATTEMPTS = 300


def _background_color(grid: np.ndarray) -> int:
    return int(np.bincount(grid.ravel()).argmax())


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_edge(comp: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)


def _find_anchor_color(grid: np.ndarray, bg: int) -> int:
    candidates: list[int] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        comps = _components(grid == color)
        if comps and all(len(comp) == 1 for comp in comps):
            candidates.append(color)
    assert len(candidates) == 1, "expected a unique singleton anchor color"
    return candidates[0]


def _draw_separator_mask(anchor_mask: np.ndarray) -> np.ndarray:
    h, w = anchor_mask.shape
    lines = anchor_mask.copy()

    for r in range(h):
        cols = np.flatnonzero(anchor_mask[r])
        if cols.size == 1:
            raise AssertionError("anchor rows must contain at least two anchors")
        for c0, c1 in zip(cols, cols[1:]):
            lines[r, int(c0) : int(c1) + 1] = True

    for c in range(w):
        rows = np.flatnonzero(anchor_mask[:, c])
        if rows.size == 1:
            raise AssertionError("anchor columns must contain at least two anchors")
        for r0, r1 in zip(rows, rows[1:]):
            lines[int(r0) : int(r1) + 1, c] = True

    return lines


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect the red anchor pixels into an orthogonal partition, then flood each
    enclosed region with the color of its unique 2-cell seed block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    anchor_color = _find_anchor_color(g, bg)
    separator_mask = _draw_separator_mask(g == anchor_color)

    out = g.copy()
    h, w = g.shape
    for comp in _components(~separator_mask):
        if _touches_edge(comp, h, w):
            continue
        seed_colors = {
            int(g[r, c])
            for r, c in comp
            if int(g[r, c]) not in (bg, anchor_color)
        }
        assert len(seed_colors) == 1, "each enclosed region must contain exactly one seed color"
        fill_color = next(iter(seed_colors))
        for r, c in comp:
            if int(out[r, c]) == bg:
                out[r, c] = fill_color
    return out


def _choose_layout() -> tuple[int, int]:
    mode = random.choice(("horizontal_only", "vertical_only", "both"))
    if mode == "horizontal_only":
        return random.choice(SINGLE_AXIS_LAYOUTS), 0
    if mode == "vertical_only":
        return 0, random.choice(SINGLE_AXIS_LAYOUTS)
    return random.choice(BOTH_AXIS_LAYOUTS)


def _sample_sizes(count: int) -> list[int]:
    return [random.randint(*CELL_SIZE_RANGE) for _ in range(count)]


def _line_positions(start: int, cell_sizes: list[int]) -> list[int]:
    positions = [start]
    cursor = start
    for size in cell_sizes:
        cursor += size + 1
        positions.append(cursor)
    return positions


def _place_seed(grid: np.ndarray, r0: int, r1: int, c0: int, c1: int, color: int) -> None:
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    orientations: list[str] = []
    if width >= 2:
        orientations.append("horizontal")
    if height >= 2:
        orientations.append("vertical")
    assert orientations, "seed region must fit a domino"

    orientation = random.choice(orientations)
    if orientation == "horizontal":
        rr = random.randint(r0, r1)
        cc = random.randint(c0, c1 - 1)
        grid[rr, cc] = color
        grid[rr, cc + 1] = color
    else:
        rr = random.randint(r0, r1 - 1)
        cc = random.randint(c0, c1)
        grid[rr, cc] = color
        grid[rr + 1, cc] = color


def _gen_attempt() -> np.ndarray | None:
    internal_h, internal_v = _choose_layout()
    row_cells = internal_h + 1
    col_cells = internal_v + 1
    region_count = row_cells * col_cells
    if region_count > len(SEED_COLORS):
        return None

    cell_heights = _sample_sizes(row_cells)
    cell_widths = _sample_sizes(col_cells)
    top_margin = random.randint(*OUTER_MARGIN_RANGE)
    bottom_margin = random.randint(*OUTER_MARGIN_RANGE)
    left_margin = random.randint(*OUTER_MARGIN_RANGE)
    right_margin = random.randint(*OUTER_MARGIN_RANGE)

    frame_height = sum(cell_heights) + len(cell_heights) + 1
    frame_width = sum(cell_widths) + len(cell_widths) + 1
    grid_h = top_margin + frame_height + bottom_margin
    grid_w = left_margin + frame_width + right_margin

    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    row_lines = _line_positions(top_margin, cell_heights)
    col_lines = _line_positions(left_margin, cell_widths)

    for r in row_lines:
        for c in col_lines:
            grid[r, c] = ANCHOR_COLOR

    seed_colors = list(SEED_COLORS)
    random.shuffle(seed_colors)
    color_iter = iter(seed_colors[:region_count])

    for i in range(row_cells):
        for j in range(col_cells):
            ir0 = row_lines[i] + 1
            ir1 = row_lines[i + 1] - 1
            ic0 = col_lines[j] + 1
            ic1 = col_lines[j + 1] - 1
            _place_seed(grid, ir0, ir1, ic0, ic1, next(color_iter))

    anchor_mask = grid == ANCHOR_COLOR
    assert np.sum(anchor_mask) == len(row_lines) * len(col_lines)
    assert region_count >= 2, "generator should create at least two regions"

    solved = main(grid)
    assert solved.shape == grid.shape
    assert np.any((grid == BG_COLOR) & (solved != BG_COLOR)), "output must fill some background"
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
