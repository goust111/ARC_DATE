from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAGENTA = 6
ORANGE = 7
AZURE = 8

TOWER_COLORS = (RED, BLUE, YELLOW, GREEN)
GEN_HEIGHT_RANGE = (8, 18)
GEN_WIDTH_RANGE = (8, 18)
GEN_RETRY_LIMIT = 400
LAYOUT_RETRY_LIMIT = 80
MIN_GAP = 2


def _tower_order(towers: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Left-to-right, then top-to-bottom."""
    return sorted(towers, key=lambda item: (item[1], item[0]))


def _extract_towers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    coords = np.argwhere(g != BLACK)
    assert coords.size > 0, "expected at least one tower"
    assert int(coords.shape[0]) <= len(TOWER_COLORS), "too many towers for the palette"

    towers: list[tuple[int, int, int]] = []
    seen_colors: set[int] = set()
    for r, c in coords:
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        assert 0 < r < h - 1 and 0 < c < w - 1, "towers cannot sit on the edge"
        assert color in TOWER_COLORS, "unexpected tower color"
        assert color not in seen_colors, "tower colors must be distinct"
        seen_colors.add(color)
        towers.append((r, c, color))
    return _tower_order(towers)


def _mix_tower_colors(colors: list[int]) -> int:
    """Fold the original tower colors that reach one cell into the visible mixed color."""
    assert colors, "expected at least one reaching tower"
    state = set(map(int, colors))
    if len(state) == 1:
        return int(colors[0])
    if GREEN in state:
        return AZURE
    if state == {RED, BLUE}:
        return MAGENTA
    if state == {RED, YELLOW}:
        return ORANGE
    if state == {BLUE, YELLOW}:
        return GREEN
    if state == {RED, BLUE, YELLOW}:
        return AZURE
    raise AssertionError(f"unsupported ray color combination: {sorted(state)}")


def _reaching_colors(
    towers: list[tuple[int, int, int]],
    r: int,
    c: int,
) -> list[int]:
    return [color for tr, tc, color in towers if tr == r or tc == c]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each colored tower across its row and column, mixing crossing signals into new colors.

    Input structure:
    - Black is the background.
    - Every non-black cell is a single-pixel tower placed away from the grid edge.
    - Tower colors are distinct and come from blue, red, yellow, and green.

    Rule:
    - Each tower emits a straight ray of its own color upward, downward, leftward, and rightward until the grid edge.
    - Every non-tower cell takes the mix of all tower colors whose row or column reaches that cell.
    - Blue+red becomes magenta, red+yellow becomes orange, blue+yellow becomes green, and any mixed cell that includes green together with another color becomes azure.
    - When three or more towers reach the same cell, fold those tower colors in left-to-right then top-to-bottom tower order; tower cells themselves stay unchanged.

    Key insight:
    - The input stores only the signal sources, so each output pixel is determined entirely by which tower rows and columns pass through that location.
    """
    g = np.asarray(grid, dtype=int)
    towers = _extract_towers(g)
    h, w = map(int, g.shape)
    tower_lookup = {(r, c): color for r, c, color in towers}

    out = np.full((h, w), BLACK, dtype=int)
    for r in range(h):
        for c in range(w):
            if (r, c) in tower_lookup:
                out[r, c] = int(tower_lookup[(r, c)])
                continue
            colors = _reaching_colors(towers, r, c)
            if colors:
                out[r, c] = _mix_tower_colors(colors)
    return out


def _sample_spaced_values(limit: int, count: int) -> list[int] | None:
    values = list(range(1, int(limit) - 1))
    for _ in range(LAYOUT_RETRY_LIMIT):
        random.shuffle(values)
        chosen: list[int] = []
        for value in values:
            if all(abs(value - old) >= MIN_GAP for old in chosen):
                chosen.append(int(value))
                if len(chosen) == count:
                    chosen.sort()
                    return chosen
    return None


def _layout_pair(h: int, w: int) -> list[tuple[int, int, int]] | None:
    rows = _sample_spaced_values(h, 2)
    cols = _sample_spaced_values(w, 2)
    if rows is None or cols is None:
        return None

    paired_cols = cols[:]
    random.shuffle(paired_cols)
    colors = random.sample(TOWER_COLORS, 2)
    return [
        (rows[0], paired_cols[0], colors[0]),
        (rows[1], paired_cols[1], colors[1]),
    ]


def _layout_triple(h: int, w: int) -> list[tuple[int, int, int]] | None:
    colors = random.sample(TOWER_COLORS, 3)
    if random.random() < 0.5:
        rows = _sample_spaced_values(h, 2)
        cols = _sample_spaced_values(w, 3)
        if rows is None or cols is None:
            return None
        main_row = random.choice(rows)
        other_row = rows[0] if rows[1] == main_row else rows[1]
        positions = [
            (main_row, cols[0]),
            (main_row, cols[2]),
            (other_row, cols[1]),
        ]
    else:
        rows = _sample_spaced_values(h, 3)
        cols = _sample_spaced_values(w, 2)
        if rows is None or cols is None:
            return None
        main_col = random.choice(cols)
        other_col = cols[0] if cols[1] == main_col else cols[1]
        positions = [
            (rows[0], main_col),
            (rows[2], main_col),
            (rows[1], other_col),
        ]

    random.shuffle(colors)
    return [(r, c, color) for (r, c), color in zip(positions, colors)]


def _layout_quad(h: int, w: int) -> list[tuple[int, int, int]] | None:
    rows = _sample_spaced_values(h, 3)
    cols = _sample_spaced_values(w, 3)
    if rows is None or cols is None:
        return None

    colors = list(TOWER_COLORS)
    random.shuffle(colors)
    positions = [
        (rows[1], cols[0]),
        (rows[1], cols[2]),
        (rows[0], cols[1]),
        (rows[2], cols[1]),
    ]
    return [(r, c, color) for (r, c), color in zip(positions, colors)]


def _render_input(shape: tuple[int, int], towers: list[tuple[int, int, int]]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for r, c, color in towers:
        grid[int(r), int(c)] = int(color)
    return grid


def _max_signal_count(towers: list[tuple[int, int, int]], shape: tuple[int, int]) -> int:
    h, w = map(int, shape)
    tower_cells = {(r, c) for r, c, _ in towers}
    best = 0
    for r in range(h):
        for c in range(w):
            if (r, c) in tower_cells:
                continue
            best = max(best, len(_reaching_colors(towers, r, c)))
    return best


def _is_valid_generated(grid: np.ndarray, mode: str) -> bool:
    try:
        towers = _extract_towers(grid)
        out = main(grid)
    except Exception:
        return False

    if out.shape != grid.shape or np.array_equal(out, grid):
        return False

    tower_colors = {color for _, _, color in towers}
    derived_colors = set(map(int, np.unique(out))) - tower_colors - {BLACK}
    if not derived_colors:
        return False

    max_signals = _max_signal_count(towers, grid.shape)
    if mode == "pair" and max_signals != 2:
        return False
    if mode == "triple" and max_signals < 3:
        return False
    if mode == "quad" and max_signals < 4:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(("pair", "triple", "quad"))
    h = random.randint(*GEN_HEIGHT_RANGE)
    w = random.randint(*GEN_WIDTH_RANGE)

    if mode == "pair":
        towers = _layout_pair(h, w)
    elif mode == "triple":
        towers = _layout_triple(h, w)
    else:
        towers = _layout_quad(h, w)

    if towers is None:
        return None

    grid = _render_input((h, w), towers)
    return grid if _is_valid_generated(grid, mode) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower input")
