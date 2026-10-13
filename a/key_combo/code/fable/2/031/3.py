from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
try:
    if str(_ARC_UTILS_DIR) not in sys.path:
        sys.path.append(str(_ARC_UTILS_DIR))
    from arc_utils import bbox as _arc_bbox
    from arc_utils import crop_bbox as _arc_crop_bbox
    from arc_utils import get_bg as _arc_get_bg
except Exception:
    _arc_bbox = None
    _arc_crop_bbox = None
    _arc_get_bg = None


Coord = tuple[int, int]
State = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    if _arc_get_bg is not None:
        return int(_arc_get_bg(grid))
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    if _arc_bbox is not None:
        box = _arc_bbox(mask)
        return None if box is None else tuple(map(int, box))
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    if _arc_crop_bbox is not None:
        return np.asarray(_arc_crop_bbox(grid, box), dtype=grid.dtype)
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _regions_between(size: int, separator_lines: list[int]) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    start = 0
    for line in sorted(separator_lines):
        assert start < line < size - 1
        regions.append((start, line - 1))
        start = line + 1
    assert start < size
    regions.append((start, size - 1))
    return regions


def _snake_coords(n_rows: int, n_cols: int) -> list[Coord]:
    order: list[Coord] = []
    for r in range(n_rows):
        cols = range(n_cols) if r % 2 == 0 else range(n_cols - 1, -1, -1)
        for c in cols:
            order.append((r, c))
    return order


def _row_major_coords(n_rows: int, n_cols: int) -> list[Coord]:
    return [(r, c) for r in range(n_rows) for c in range(n_cols)]


def _l_mask(length: int, orientation: int) -> np.ndarray:
    mask = np.zeros((length, length), dtype=bool)
    if orientation == 0:
        mask[0, :] = True
        mask[:, 0] = True
    elif orientation == 1:
        mask[0, :] = True
        mask[:, -1] = True
    elif orientation == 2:
        mask[-1, :] = True
        mask[:, -1] = True
    elif orientation == 3:
        mask[-1, :] = True
        mask[:, 0] = True
    else:
        raise ValueError("orientation must be in 0..3")
    return mask


def _render_l(cell_h: int, cell_w: int, length: int, orientation: int, fg: int, bg: int) -> np.ndarray:
    assert 2 <= length <= min(cell_h, cell_w)
    patch = np.full((cell_h, cell_w), bg, dtype=int)
    mask = _l_mask(length, orientation)
    r0 = (cell_h - length) // 2
    c0 = (cell_w - length) // 2
    window = patch[r0 : r0 + length, c0 : c0 + length]
    window[mask] = fg
    return patch


def _recognize_state(block: np.ndarray, bg: int, separator: int) -> tuple[State | None, int | None]:
    mask = block != bg
    if not np.any(mask):
        return None, None

    colors = sorted(set(map(int, block[mask].ravel())))
    assert separator not in colors
    assert len(colors) == 1

    box = _bbox(mask)
    assert box is not None
    cropped = _crop_bbox(block, box)
    cropped_mask = cropped != bg
    length_h, length_w = cropped_mask.shape
    assert length_h == length_w
    length = int(length_h)
    assert length >= 2

    matches = [
        orientation
        for orientation in range(4)
        if np.array_equal(cropped_mask, _l_mask(length, orientation))
    ]
    assert len(matches) == 1
    return (matches[0], length), colors[0]


def _infer_board(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    height, width = map(int, g.shape)
    bg = _get_bg(g)

    separator_candidates: list[tuple[int, list[int], list[int]]] = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        row_lines = [r for r in range(height) if np.all(g[r, :] == color)]
        col_lines = [c for c in range(width) if np.all(g[:, c] == color)]
        if row_lines or col_lines:
            separator_candidates.append((color, row_lines, col_lines))

    assert len(separator_candidates) == 1
    separator, row_lines, col_lines = separator_candidates[0]
    assert row_lines and col_lines

    row_regions = _regions_between(height, row_lines)
    col_regions = _regions_between(width, col_lines)
    assert len(row_regions) >= 2 and len(col_regions) >= 2
    assert len({r1 - r0 + 1 for r0, r1 in row_regions}) == 1
    assert len({c1 - c0 + 1 for c0, c1 in col_regions}) == 1
    return bg, separator, row_regions, col_regions


def _step(state: State, grow: int) -> State:
    orientation, length = state
    return (orientation + 1) % 4, length + grow


def _inverse_step(state: State, grow: int) -> State:
    orientation, length = state
    return (orientation - 1) % 4, length - grow


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the one empty cell of a separator board by following a snake-ordered rotate-and-grow L-sequence.

    Input structure:
    - Solid separator rows and columns split the canvas into equal rectangular cells.
    - Exactly one cell is pure background, while every other cell contains one monochrome L-corner sprite from the same family.
    - The sprite color stays fixed across cells, but orientation and arm length vary from cell to cell.

    Rule:
    - Read the cells in boustrophedon order: left-to-right on the first row, right-to-left on the next, alternating by row.
    - Consecutive sprites differ by one composite step: rotate the L 90 degrees clockwise and increase both arms by one fixed amount.
    - Infer the missing state from its snake-order predecessor or successor, then fill only that empty cell and keep the separators and all existing cells unchanged.

    Key insight:
    - A plain row-major scan breaks the temporal order at every row turn; only the snake order makes all visible neighbors share the same rotate-and-grow step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 0 < g.shape[0] <= 30 and 0 < g.shape[1] <= 30

    bg, separator, row_regions, col_regions = _infer_board(g)
    cell_h = row_regions[0][1] - row_regions[0][0] + 1
    cell_w = col_regions[0][1] - col_regions[0][0] + 1

    states_by_cell: dict[Coord, State] = {}
    blank_cells: list[Coord] = []
    fg_color: int | None = None

    for rr, (r0, r1) in enumerate(row_regions):
        for cc, (c0, c1) in enumerate(col_regions):
            block = g[r0 : r1 + 1, c0 : c1 + 1]
            state, color = _recognize_state(block, bg, separator)
            if state is None:
                blank_cells.append((rr, cc))
                continue
            if fg_color is None:
                fg_color = int(color)
            else:
                assert fg_color == int(color)
            states_by_cell[(rr, cc)] = state

    assert fg_color is not None
    assert len(blank_cells) == 1

    snake = _snake_coords(len(row_regions), len(col_regions))
    states = [states_by_cell.get(cell) for cell in snake]
    blank_index = states.index(None)

    observed_grows: set[int] = set()
    for a, b in zip(states, states[1:]):
        if a is None or b is None:
            continue
        assert b[0] == (a[0] + 1) % 4
        grow = b[1] - a[1]
        assert grow > 0
        observed_grows.add(grow)

    assert len(observed_grows) == 1
    grow = observed_grows.pop()

    candidates: list[State] = []
    if blank_index > 0 and states[blank_index - 1] is not None:
        candidates.append(_step(states[blank_index - 1], grow))
    if blank_index + 1 < len(states) and states[blank_index + 1] is not None:
        backward = _inverse_step(states[blank_index + 1], grow)
        assert backward[1] >= 2
        candidates.append(backward)

    assert candidates
    assert all(candidate == candidates[0] for candidate in candidates[1:])
    missing_orientation, missing_length = candidates[0]
    assert 2 <= missing_length <= min(cell_h, cell_w)

    out = g.copy()
    blank_r, blank_c = blank_cells[0]
    r0, r1 = row_regions[blank_r]
    c0, c1 = col_regions[blank_c]
    out[r0 : r1 + 1, c0 : c1 + 1] = _render_l(
        cell_h, cell_w, missing_length, missing_orientation, fg_color, bg
    )
    return out


LAYOUTS = ((2, 3), (3, 2))
COLOR_POOL = tuple(range(10))
GROW_CHOICES = (1,)
MIN_CELL_DIM = 8
GEN_RETRIES = 600


def _render_board(
    n_rows: int,
    n_cols: int,
    cell_h: int,
    cell_w: int,
    bg: int,
    separator: int,
    fg: int,
    states: list[State],
) -> np.ndarray:
    height = n_rows * cell_h + (n_rows - 1)
    width = n_cols * cell_w + (n_cols - 1)
    board = np.full((height, width), bg, dtype=int)
    for r in range(1, n_rows):
        board[r * cell_h + (r - 1), :] = separator
    for c in range(1, n_cols):
        board[:, c * cell_w + (c - 1)] = separator

    for (rr, cc), state in zip(_snake_coords(n_rows, n_cols), states):
        r0 = rr * (cell_h + 1)
        c0 = cc * (cell_w + 1)
        board[r0 : r0 + cell_h, c0 : c0 + cell_w] = _render_l(
            cell_h, cell_w, state[1], state[0], fg, bg
        )
    return board


def _row_major_has_violation(
    n_rows: int,
    n_cols: int,
    states_by_cell: dict[Coord, State],
    blank_cell: Coord,
    grow: int,
) -> bool:
    sequence = [
        None if cell == blank_cell else states_by_cell[cell]
        for cell in _row_major_coords(n_rows, n_cols)
    ]
    for a, b in zip(sequence, sequence[1:]):
        if a is None or b is None:
            continue
        if b != _step(a, grow):
            return True
    return False


def _visible_orientation_repeat(states: list[State], blank_index: int) -> bool:
    by_orientation: dict[int, set[int]] = {}
    for index, (orientation, length) in enumerate(states):
        if index == blank_index:
            continue
        by_orientation.setdefault(orientation, set()).add(length)
    return any(len(lengths) >= 2 for lengths in by_orientation.values())


def _sample_dimensions(n_rows: int, n_cols: int, grow: int, n_states: int) -> tuple[int, int, int] | None:
    max_h = (30 - (n_rows - 1)) // n_rows
    max_w = (30 - (n_cols - 1)) // n_cols
    h_options = [h for h in range(MIN_CELL_DIM, max_h + 1)]
    w_options = [w for w in range(MIN_CELL_DIM, max_w + 1)]
    random.shuffle(h_options)
    random.shuffle(w_options)
    for cell_h in h_options:
        for cell_w in w_options:
            max_length = min(cell_h, cell_w) - 1
            max_start = max_length - grow * (n_states - 1)
            if max_start < 2:
                continue
            start_length = random.randint(2, max_start)
            return cell_h, cell_w, start_length
    return None


def _gen_attempt() -> np.ndarray | None:
    n_rows, n_cols = random.choice(LAYOUTS)
    n_states = n_rows * n_cols
    grow = random.choice(GROW_CHOICES)
    dims = _sample_dimensions(n_rows, n_cols, grow, n_states)
    if dims is None:
        return None
    cell_h, cell_w, start_length = dims

    bg, separator, fg = random.sample(COLOR_POOL, 3)
    start_orientation = random.randrange(4)
    states = [
        ((start_orientation + idx) % 4, start_length + idx * grow)
        for idx in range(n_states)
    ]
    full = _render_board(n_rows, n_cols, cell_h, cell_w, bg, separator, fg, states)
    if max(full.shape) > 30:
        return None

    blank_index = random.randrange(n_states)
    blank_cell = _snake_coords(n_rows, n_cols)[blank_index]
    states_by_cell = {
        cell: state
        for cell, state in zip(_snake_coords(n_rows, n_cols), states)
    }

    inp = full.copy()
    r0 = blank_cell[0] * (cell_h + 1)
    c0 = blank_cell[1] * (cell_w + 1)
    inp[r0 : r0 + cell_h, c0 : c0 + cell_w] = bg

    if not _row_major_has_violation(n_rows, n_cols, states_by_cell, blank_cell, grow):
        return None
    if not _visible_orientation_repeat(states, blank_index):
        return None
    if _get_bg(inp) != bg:
        return None
    if not np.array_equal(main(inp), full):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid snake-board input")
