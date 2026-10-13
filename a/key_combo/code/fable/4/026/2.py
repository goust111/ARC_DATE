from __future__ import annotations

import random

import numpy as np


SPIRAL_ORDER = [
    (1, 1),
    (0, 1),
    (0, 2),
    (1, 2),
    (2, 2),
    (2, 1),
    (2, 0),
    (1, 0),
    (0, 0),
]

MAX_GENERATION_ATTEMPTS = 400
PALETTE = tuple(range(10))


def _rot_cw(grid: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(grid, -int(turns) % 4)


def _consecutive_groups(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    groups = [[indices[0]]]
    for value in indices[1:]:
        if value == groups[-1][-1] + 1:
            groups[-1].append(value)
        else:
            groups.append([value])
    return groups


def _find_layout(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    for divider_color in map(int, np.unique(grid)):
        divider_rows = [r for r in range(h) if np.all(grid[r, :] == divider_color)]
        divider_cols = [c for c in range(w) if np.all(grid[:, c] == divider_color)]
        row_groups = _consecutive_groups(divider_rows)
        col_groups = _consecutive_groups(divider_cols)
        if len(row_groups) != 2 or len(col_groups) != 2:
            continue

        row_spans = [
            (0, row_groups[0][0] - 1),
            (row_groups[0][-1] + 1, row_groups[1][0] - 1),
            (row_groups[1][-1] + 1, h - 1),
        ]
        col_spans = [
            (0, col_groups[0][0] - 1),
            (col_groups[0][-1] + 1, col_groups[1][0] - 1),
            (col_groups[1][-1] + 1, w - 1),
        ]
        if any(lo > hi for lo, hi in row_spans + col_spans):
            continue

        row_heights = [hi - lo + 1 for lo, hi in row_spans]
        col_widths = [hi - lo + 1 for lo, hi in col_spans]
        if len(set(row_heights)) != 1 or len(set(col_widths)) != 1:
            continue
        return divider_color, row_spans, col_spans

    raise AssertionError("failed to parse a 3x3 divider grid")


def _extract_cells(grid: np.ndarray) -> list[list[np.ndarray]]:
    _divider_color, row_spans, col_spans = _find_layout(grid)
    cells: list[list[np.ndarray]] = []
    for r0, r1 in row_spans:
        row_cells = []
        for c0, c1 in col_spans:
            row_cells.append(grid[r0 : r1 + 1, c0 : c1 + 1].copy())
        cells.append(row_cells)
    return cells


def _rotation_offset(reference: np.ndarray, candidate: np.ndarray) -> int:
    matches = [turn for turn in range(4) if np.array_equal(_rot_cw(reference, turn), candidate)]
    assert len(matches) == 1, "cells must be one unique quarter-turn rotation of the same sprite"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Use the lone spiral-rotation anomaly as a pointer and return the later target cell it indicates.

    Input structure:
    - The input is partitioned into a 3x3 array of equal cells by one uniform divider color.
    - Every cell contains the same non-rotationally-symmetric multicolor sprite in one quarter-turn orientation.
    - Along the hidden square spiral starting at the center and moving up, right, down, left with step lengths 1,1,2,2, the correct cells advance by +90 degrees each step, except for one anomalous cell.

    Rule:
    - Read the nine cell contents without the divider lines.
    - For some global phase offset, eight cells fit the spiral rule exactly and one cell is rotated an extra 90, 180, or 270 degrees beyond its expected orientation.
    - Treat that deviation as a pointer step count, continue forward along the same spiral by deviation/90 cells, and output the raw content of the landing cell.

    Key insight:
    - The bad rotation is not repaired; its angular difference identifies both which cell is the anomaly and how far to move forward on the same implicit spiral indexing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    cells = _extract_cells(g)
    reference = cells[0][0]

    actual_turns = np.zeros((3, 3), dtype=int)
    for r in range(3):
        for c in range(3):
            actual_turns[r, c] = _rotation_offset(reference, cells[r][c])

    candidates: list[tuple[int, int]] = []
    for phase in range(4):
        mismatches: list[tuple[int, int, int, int]] = []
        for spiral_index, (r, c) in enumerate(SPIRAL_ORDER):
            expected_turn = (phase + spiral_index) % 4
            if int(actual_turns[r, c]) != expected_turn:
                mismatches.append((spiral_index, r, c, expected_turn))
        if len(mismatches) != 1:
            continue

        spiral_index, _r, _c, expected_turn = mismatches[0]
        deviation = (int(actual_turns[_r, _c]) - expected_turn) % 4
        if deviation in {1, 2, 3} and spiral_index + deviation < len(SPIRAL_ORDER):
            candidates.append((spiral_index, deviation))

    assert len(candidates) == 1, "expected exactly one anomaly-pointer interpretation"
    anomaly_index, deviation = candidates[0]
    target_r, target_c = SPIRAL_ORDER[anomaly_index + deviation]
    return cells[target_r][target_c].copy()


def _neighbors4(r: int, c: int, size: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size:
            out.append((nr, nc))
    return out


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _grow_connected_mask(size: int, target_area: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    start = (random.randrange(size), random.randrange(size))
    mask[start] = True
    frontier = set(_neighbors4(start[0], start[1], size))

    while int(mask.sum()) < target_area:
        assert frontier, "connected growth frontier unexpectedly vanished"
        r, c = random.choice(tuple(frontier))
        frontier.remove((r, c))
        mask[r, c] = True
        for nr, nc in _neighbors4(r, c, size):
            if not mask[nr, nc]:
                frontier.add((nr, nc))

    return mask


def _sample_sprite(size: int, cell_bg: int, sprite_colors: list[int]) -> np.ndarray | None:
    target_area = random.randint(size + 1, min(size * size - 2, (size * size * 3) // 5))
    mask = _grow_connected_mask(size, target_area)
    r0, c0, r1, c1 = _bbox(mask)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None

    sprite = np.full((size, size), cell_bg, dtype=int)
    sprite[mask] = sprite_colors[0]

    coords = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
    random.shuffle(coords)
    accent_count = random.randint(1, min(3, len(sprite_colors) - 1))
    accent_cells = coords[:accent_count]
    for idx, (r, c) in enumerate(accent_cells, start=1):
        sprite[r, c] = sprite_colors[idx]

    if len({sprite_colors[0], *[int(sprite[r, c]) for r, c in accent_cells]}) < 2:
        return None
    if any(np.array_equal(sprite, _rot_cw(sprite, turn)) for turn in (1, 2, 3)):
        return None
    if len({tuple(_rot_cw(sprite, turn).ravel()) for turn in range(4)}) != 4:
        return None
    return sprite


def _render_input(
    sprite: np.ndarray,
    divider_color: int,
    divider_thickness: int,
    base_turn: int,
    anomaly_index: int,
    deviation: int,
) -> tuple[np.ndarray, np.ndarray]:
    cell_size = int(sprite.shape[0])
    grid_size = 3 * cell_size + 2 * divider_thickness
    inp = np.full((grid_size, grid_size), divider_color, dtype=int)

    cells: dict[tuple[int, int], np.ndarray] = {}
    for spiral_index, (gr, gc) in enumerate(SPIRAL_ORDER):
        turn = (base_turn + spiral_index) % 4
        if spiral_index == anomaly_index:
            turn = (turn + deviation) % 4
        cell = _rot_cw(sprite, turn)
        cells[(gr, gc)] = cell

        r0 = gr * (cell_size + divider_thickness)
        c0 = gc * (cell_size + divider_thickness)
        inp[r0 : r0 + cell_size, c0 : c0 + cell_size] = cell

    target_cell = cells[SPIRAL_ORDER[anomaly_index + deviation]].copy()
    return inp, target_cell


def _gen_attempt() -> np.ndarray | None:
    divider_thickness = random.randint(1, 2)
    max_cell_size = (30 - 2 * divider_thickness) // 3
    cell_size = random.randint(5, min(8, max_cell_size))

    colors = random.sample(PALETTE, 6)
    divider_color = colors[0]
    cell_bg = colors[1]
    sprite_colors = colors[2:]

    sprite = _sample_sprite(cell_size, cell_bg, sprite_colors)
    if sprite is None:
        return None

    valid_pairs = [(idx, dev) for idx in range(9) for dev in (1, 2, 3) if idx + dev < 9]
    anomaly_index, deviation = random.choice(valid_pairs)
    base_turn = random.randrange(4)

    inp, expected = _render_input(
        sprite=sprite,
        divider_color=divider_color,
        divider_thickness=divider_thickness,
        base_turn=base_turn,
        anomaly_index=anomaly_index,
        deviation=deviation,
    )

    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anomaly-pointer input")
