from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5
AZURE = 8

ROTATION_TO_COLOR = {1: AZURE, 2: RED, 3: GREEN}
COLOR_TO_ROTATION = {value: key for key, value in ROTATION_TO_COLOR.items()}
PATTERN_COLORS = [1, 4, 6, 7, 9]

MAX_GRID_SIZE = 30
CELL_SIZE_CHOICES = [3, 4, 5]
MAX_GENERATION_ATTEMPTS = 300


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected at least one non-background pixel"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _grid_geometry(grid: np.ndarray) -> tuple[tuple[int, int], list[int], list[int], int]:
    r0, c0, r1, c1 = _bbox(grid != BLACK)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]

    line_rows = [r for r in range(sub.shape[0]) if bool(np.all(sub[r] != BLACK))]
    line_cols = [c for c in range(sub.shape[1]) if bool(np.all(sub[:, c] != BLACK))]
    assert len(line_rows) >= 2 and len(line_cols) >= 2, "expected a visible lattice"

    row_steps = np.diff(line_rows)
    col_steps = np.diff(line_cols)
    assert len(set(map(int, row_steps))) == 1, "row grid lines must be evenly spaced"
    assert len(set(map(int, col_steps))) == 1, "column grid lines must be evenly spaced"

    step = int(row_steps[0])
    assert step == int(col_steps[0]) and step >= 2, "cells must be uniform squares"
    assert line_rows[0] == 0 and line_rows[-1] == sub.shape[0] - 1
    assert line_cols[0] == 0 and line_cols[-1] == sub.shape[1] - 1
    return (r0, c0), line_rows, line_cols, step - 1


def _paint_side(
    canvas: np.ndarray,
    line_rows: list[int],
    line_cols: list[int],
    cell_r: int,
    cell_c: int,
    side: str,
    color: int,
) -> None:
    top = int(line_rows[cell_r])
    bottom = int(line_rows[cell_r + 1])
    left = int(line_cols[cell_c])
    right = int(line_cols[cell_c + 1])

    if side == "top":
        canvas[top, left + 1 : right] = int(color)
    elif side == "bottom":
        canvas[bottom, left + 1 : right] = int(color)
    elif side == "left":
        canvas[top + 1 : bottom, left] = int(color)
    elif side == "right":
        canvas[top + 1 : bottom, right] = int(color)
    else:
        raise ValueError(f"unknown side: {side}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each cell's interior pattern by the amount encoded on its unique colored border, then restore the lattice.

    Input structure:
    - The background is black, and the non-black area is one grey lattice of equal square cells with 1-pixel grid lines.
    - Every cell interior has the same size and contains a single-color asymmetric pattern on black.
    - In each cell, exactly one border segment between grid intersections is recolored from grey to Azure, Red, or Green.

    Rule:
    - Locate the lattice and split it into equal square cells.
    - For each cell, read the unique non-grey border segment; Azure means rotate the cell interior 90 degrees clockwise, Red means 180 degrees, and Green means 270 degrees.
    - Replace the cell interior by the rotated version inside the same cell.
    - Restore every grid-line pixel, including all colored border segments, back to grey.

    Key insight:
    - The colored border is not part of the motif; it is a local control signal that encodes how many clockwise quarter-turns to apply inside that specific cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    (base_r, base_c), line_rows, line_cols, _cell_size = _grid_geometry(g)
    sub = g[base_r : base_r + line_rows[-1] + 1, base_c : base_c + line_cols[-1] + 1]

    out = g.copy()
    out_sub = out[base_r : base_r + sub.shape[0], base_c : base_c + sub.shape[1]]
    out_sub[line_rows, :] = GREY
    out_sub[:, line_cols] = GREY

    for cell_r in range(len(line_rows) - 1):
        for cell_c in range(len(line_cols) - 1):
            top = int(line_rows[cell_r])
            bottom = int(line_rows[cell_r + 1])
            left = int(line_cols[cell_c])
            right = int(line_cols[cell_c + 1])

            borders = {
                "top": sub[top, left + 1 : right],
                "bottom": sub[bottom, left + 1 : right],
                "left": sub[top + 1 : bottom, left],
                "right": sub[top + 1 : bottom, right],
            }

            colored = []
            for side, seg in borders.items():
                uniques = np.unique(seg)
                assert uniques.size == 1, "each border segment must be monochrome"
                color = int(uniques[0])
                if color != GREY:
                    colored.append((side, color))
            assert len(colored) == 1, "each cell must have exactly one colored border"

            turns = COLOR_TO_ROTATION[int(colored[0][1])]
            inner = sub[top + 1 : bottom, left + 1 : right]
            out_sub[top + 1 : bottom, left + 1 : right] = np.rot90(inner, -turns)

    return out


def _neighbors(cell: tuple[int, int], size: int) -> list[tuple[int, int]]:
    r, c = cell
    result = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = int(r + dr)
        nc = int(c + dc)
        if 0 <= nr < size and 0 <= nc < size:
            result.append((nr, nc))
    return result


def _fallback_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    coords_by_size = {
        3: [(0, 1), (1, 1), (1, 2), (2, 0)],
        4: [(0, 1), (1, 1), (1, 2), (2, 2), (3, 0)],
        5: [(0, 1), (1, 1), (1, 2), (2, 2), (3, 2), (3, 3)],
    }
    for r, c in coords_by_size[size]:
        mask[r, c] = True
    return mask


def _random_mask(size: int) -> np.ndarray:
    max_area = min(6, size * size - 2)
    for _ in range(200):
        area = random.randint(3, max_area)
        start = (random.randrange(size), random.randrange(size))
        cells = {start}
        frontier = set(_neighbors(start, size))

        while frontier and len(cells) < area:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nbr in _neighbors(cell, size):
                if nbr not in cells:
                    frontier.add(nbr)

        if len(cells) != area:
            continue

        mask = np.zeros((size, size), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if np.max(mask.sum(axis=1)) >= size or np.max(mask.sum(axis=0)) >= size:
            continue
        if len(set(np.where(mask.any(axis=1))[0])) < 2 or len(set(np.where(mask.any(axis=0))[0])) < 2:
            continue

        rotations = {np.rot90(mask, k).tobytes() for k in range(4)}
        if len(rotations) != 4:
            continue
        return mask

    return np.rot90(_fallback_mask(size), random.randrange(4))


def _make_pattern(size: int, color: int) -> np.ndarray:
    tile = np.zeros((size, size), dtype=int)
    tile[_random_mask(size)] = int(color)
    return tile


def _build_groups(rows: int, cols: int) -> list[list[tuple[tuple[int, int], str]]]:
    groups: list[list[tuple[tuple[int, int], str]]] = []
    primary = random.choice(["vertical", "horizontal"])

    if primary == "vertical":
        paired_rows = rows - (rows % 2)
        for r in range(0, paired_rows, 2):
            for c in range(cols):
                groups.append([((r, c), "bottom"), ((r + 1, c), "top")])
        if rows % 2 == 1:
            last = rows - 1
            paired_cols = cols - (cols % 2)
            for c in range(0, paired_cols, 2):
                groups.append([((last, c), "right"), ((last, c + 1), "left")])
            if cols % 2 == 1:
                groups.append([((last, cols - 1), random.choice(["bottom", "right"]))])
    else:
        paired_cols = cols - (cols % 2)
        for c in range(0, paired_cols, 2):
            for r in range(rows):
                groups.append([((r, c), "right"), ((r, c + 1), "left")])
        if cols % 2 == 1:
            last = cols - 1
            paired_rows = rows - (rows % 2)
            for r in range(0, paired_rows, 2):
                groups.append([((r, last), "bottom"), ((r + 1, last), "top")])
            if rows % 2 == 1:
                groups.append([((rows - 1, last), random.choice(["bottom", "right"]))])

    seen = {}
    for group_index, group in enumerate(groups):
        for cell, side in group:
            assert cell not in seen, f"cell assigned twice: {cell}"
            seen[cell] = (group_index, side)
    assert len(seen) == rows * cols, "every cell must be assigned exactly one control edge"
    return groups


def _assign_rotations(groups: list[list[tuple[tuple[int, int], str]]]) -> list[int]:
    assert len(groups) >= 3, "need enough control edges to cover all rotation colors"
    weights = [len(group) for group in groups]
    order = list(range(len(groups)))
    random.shuffle(order)
    order.sort(key=lambda idx: weights[idx], reverse=True)

    assigned = [0] * len(groups)
    totals = {1: 0, 2: 0, 3: 0}

    for rot, idx in zip((1, 2, 3), order[:3]):
        assigned[idx] = rot
        totals[rot] += weights[idx]

    for idx in order[3:]:
        best = min(totals.values())
        options = [rot for rot, total in totals.items() if total == best]
        rot = random.choice(options)
        assigned[idx] = rot
        totals[rot] += weights[idx]

    return assigned


def _lattice_dimensions(cell_size: int) -> tuple[int, int]:
    max_cells = min(5, (MAX_GRID_SIZE - 1) // (cell_size + 1))
    choices = [
        (rows, cols)
        for rows in range(2, max_cells + 1)
        for cols in range(2, max_cells + 1)
        if rows * cols >= 6
    ]
    return random.choice(choices)


def _place_lattice(lattice_h: int, lattice_w: int) -> tuple[int, int, int, int]:
    spare_h = MAX_GRID_SIZE - lattice_h
    spare_w = MAX_GRID_SIZE - lattice_w
    top = random.randint(0, min(2, spare_h))
    bottom = random.randint(0, min(2, spare_h - top))
    left = random.randint(0, min(2, spare_w))
    right = random.randint(0, min(2, spare_w - left))
    return top, bottom, left, right


def _render_lattice(
    rows: int,
    cols: int,
    cell_size: int,
    patterns: dict[tuple[int, int], np.ndarray],
    groups: list[list[tuple[tuple[int, int], str]]],
    rotations: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    lattice_h = rows * (cell_size + 1) + 1
    lattice_w = cols * (cell_size + 1) + 1
    top, bottom, left, right = _place_lattice(lattice_h, lattice_w)
    total_h = lattice_h + top + bottom
    total_w = lattice_w + left + right

    inp = np.full((total_h, total_w), BLACK, dtype=int)
    out = np.full((total_h, total_w), BLACK, dtype=int)

    line_rows = [top + r * (cell_size + 1) for r in range(rows + 1)]
    line_cols = [left + c * (cell_size + 1) for c in range(cols + 1)]

    inp[line_rows, left : left + lattice_w] = GREY
    inp[top : top + lattice_h, line_cols] = GREY
    out[line_rows, left : left + lattice_w] = GREY
    out[top : top + lattice_h, line_cols] = GREY

    cell_turns: dict[tuple[int, int], int] = {}
    for group, turns in zip(groups, rotations):
        color = ROTATION_TO_COLOR[int(turns)]
        for (cell_r, cell_c), side in group:
            cell_turns[(cell_r, cell_c)] = int(turns)
            _paint_side(inp, line_rows, line_cols, cell_r, cell_c, side, color)

    for cell_r in range(rows):
        for cell_c in range(cols):
            tile = patterns[(cell_r, cell_c)]
            r0 = int(line_rows[cell_r] + 1)
            c0 = int(line_cols[cell_c] + 1)
            inp[r0 : r0 + cell_size, c0 : c0 + cell_size] = tile
            out[r0 : r0 + cell_size, c0 : c0 + cell_size] = np.rot90(tile, -cell_turns[(cell_r, cell_c)])

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    cell_size = random.choice(CELL_SIZE_CHOICES)
    rows, cols = _lattice_dimensions(cell_size)

    patterns = {
        (r, c): _make_pattern(cell_size, random.choice(PATTERN_COLORS))
        for r in range(rows)
        for c in range(cols)
    }
    groups = _build_groups(rows, cols)
    rotations = _assign_rotations(groups)
    inp, expected = _render_lattice(rows, cols, cell_size, patterns, groups, rotations)

    k = random.randrange(4)
    if k:
        inp = np.rot90(inp, k).copy()
        expected = np.rot90(expected, k).copy()

    if inp.shape[0] > MAX_GRID_SIZE or inp.shape[1] > MAX_GRID_SIZE:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid lattice-rotation puzzle")
