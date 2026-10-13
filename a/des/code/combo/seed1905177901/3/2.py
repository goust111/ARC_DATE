from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GEN_ATTEMPTS = 400


def _cells_of_color(grid: np.ndarray, color: int) -> set[tuple[int, int]]:
    return {tuple(map(int, cell)) for cell in np.argwhere(grid == color)}


def _paint(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _shape_key(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(_normalize(cells)))


def _same_shape(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    return len(a) == len(b) and _shape_key(a) == _shape_key(b)


def _translate(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _in_bounds(cells: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    rows, cols = shape
    return all(0 <= r < rows and 0 <= c < cols for r, c in cells)


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    seen = set()
    stack = [next(iter(cells))]
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        r, c = cell
        for dr, dc in DIRS4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                stack.append(nxt)
    return seen == cells


def _is_edge_cell(r: int, c: int, shape: tuple[int, int]) -> bool:
    rows, cols = shape
    return r == 0 or c == 0 or r == rows - 1 or c == cols - 1


def _expected_overlap_output(
    shape: tuple[int, int],
    red_cells: set[tuple[int, int]],
    blue_cells: set[tuple[int, int]],
) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    overlap = red_cells & blue_cells
    _paint(out, red_cells - blue_cells, RED)
    _paint(out, blue_cells - red_cells, BLUE)
    _paint(out, overlap, YELLOW)
    return out


def _candidate_outputs(
    shape: tuple[int, int],
    red_cells: set[tuple[int, int]],
    blue_cells: set[tuple[int, int]],
) -> list[np.ndarray]:
    rows, cols = shape
    outputs: list[np.ndarray] = []

    if _same_shape(red_cells, blue_cells) and _is_connected(red_cells) and _is_connected(blue_cells):
        out = np.full(shape, BLACK, dtype=int)
        _paint(out, red_cells, YELLOW)
        outputs.append(out)

    for full_color, full_cells, other_color, visible_other in (
        (RED, red_cells, BLUE, blue_cells),
        (BLUE, blue_cells, RED, red_cells),
    ):
        if not _is_connected(full_cells):
            continue
        if not visible_other or len(visible_other) >= len(full_cells):
            continue
        for dr in range(-(rows - 1), rows):
            for dc in range(-(cols - 1), cols):
                if dr == 0 and dc == 0:
                    continue
                shifted = _translate(full_cells, dr, dc)
                if not _in_bounds(shifted, shape):
                    continue
                overlap = shifted & full_cells
                if not overlap or len(overlap) == len(full_cells):
                    continue
                if shifted - full_cells != visible_other:
                    continue

                out = np.full(shape, BLACK, dtype=int)
                _paint(out, full_cells - shifted, full_color)
                _paint(out, shifted - full_cells, other_color)
                _paint(out, overlap, YELLOW)
                outputs.append(out)

    unique: list[np.ndarray] = []
    seen: set[bytes] = set()
    for out in outputs:
        key = out.tobytes()
        if key not in seen:
            seen.add(key)
            unique.append(out)
    return unique


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay two identical translated shapes.

    The input contains a red object and a blue object derived from the same
    4-connected shape, plus a green edge marker. If the two objects are visibly
    disjoint, move blue onto red and color the shared shape yellow. Otherwise one
    of the objects is fully visible while the other is partially occluded by the
    overlap; reconstruct the hidden copy and paint overlap cells yellow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    allowed = {BLACK, BLUE, RED, GREEN}
    assert set(map(int, np.unique(g))) <= allowed

    marker = np.argwhere(g == GREEN)
    assert marker.shape[0] == 1
    mr, mc = map(int, marker[0])
    assert _is_edge_cell(mr, mc, g.shape)

    red_cells = _cells_of_color(g, RED)
    blue_cells = _cells_of_color(g, BLUE)
    assert red_cells and blue_cells

    outputs = _candidate_outputs(g.shape, red_cells, blue_cells)
    assert len(outputs) == 1, "expected a unique reconstruction"
    return outputs[0]


def _random_shape(rng: random.Random) -> set[tuple[int, int]] | None:
    for _ in range(100):
        box_h = rng.randint(2, 5)
        box_w = rng.randint(2, 5)
        max_area = min(10, box_h * box_w - 1)
        min_area = max(4, max(box_h, box_w))
        if min_area > max_area:
            continue

        target_area = rng.randint(min_area, max_area)
        cells = {(rng.randrange(box_h), rng.randrange(box_w))}
        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(tuple(frontier)))

        if len(cells) != target_area:
            continue
        cells = _normalize(cells)
        if len({r for r, _ in cells}) < 2 or len({c for _, c in cells}) < 2:
            continue
        return cells
    return None


def _shape_hw(cells: set[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1


def _all_positions(cells: set[tuple[int, int]], shape: tuple[int, int]) -> list[tuple[int, int]]:
    rows, cols = shape
    h, w = _shape_hw(cells)
    return [(r, c) for r in range(rows - h + 1) for c in range(cols - w + 1)]


def _well_separated(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r1, c1 in a:
        for r2, c2 in b:
            if abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1:
                return False
    return True


def _edge_choices(shape: tuple[int, int], occupied: set[tuple[int, int]]) -> list[tuple[int, int]]:
    rows, cols = shape
    cells = []
    for r in range(rows):
        for c in range(cols):
            if _is_edge_cell(r, c, shape) and (r, c) not in occupied:
                cells.append((r, c))
    return cells


def _render_input(
    shape: tuple[int, int],
    red_cells: set[tuple[int, int]],
    blue_cells: set[tuple[int, int]],
    marker_cell: tuple[int, int],
    top_color: int | None,
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    if top_color is None:
        _paint(grid, red_cells, RED)
        _paint(grid, blue_cells, BLUE)
    elif top_color == RED:
        _paint(grid, blue_cells - red_cells, BLUE)
        _paint(grid, red_cells, RED)
    else:
        _paint(grid, red_cells - blue_cells, RED)
        _paint(grid, blue_cells, BLUE)
    grid[marker_cell] = GREEN
    return grid


def _gen_disjoint_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    cells = _random_shape(rng)
    if cells is None:
        return None

    shape = (rng.randint(12, 18), rng.randint(12, 18))
    positions = _all_positions(cells, shape)
    rng.shuffle(positions)

    for red_pos in positions:
        red_cells = _translate(cells, *red_pos)
        blue_positions = positions[:]
        rng.shuffle(blue_positions)
        for blue_pos in blue_positions:
            if blue_pos == red_pos:
                continue
            blue_cells = _translate(cells, *blue_pos)
            if not _well_separated(red_cells, blue_cells):
                continue

            occupied = red_cells | blue_cells
            markers = _edge_choices(shape, occupied)
            if not markers:
                continue
            marker_cell = rng.choice(markers)
            inp = _render_input(shape, red_cells, blue_cells, marker_cell, top_color=None)
            expected = np.full(shape, BLACK, dtype=int)
            _paint(expected, red_cells, YELLOW)
            return inp, expected
    return None


def _gen_overlap_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    cells = _random_shape(rng)
    if cells is None:
        return None

    shape = (rng.randint(10, 18), rng.randint(10, 18))
    positions = _all_positions(cells, shape)
    if len(positions) < 2:
        return None

    pairs = []
    for red_pos in positions:
        red_cells = _translate(cells, *red_pos)
        for blue_pos in positions:
            if blue_pos == red_pos:
                continue
            blue_cells = _translate(cells, *blue_pos)
            overlap = red_cells & blue_cells
            if not overlap or len(overlap) == len(cells):
                continue
            if len(red_cells - blue_cells) < 2 or len(blue_cells - red_cells) < 2:
                continue
            pairs.append((red_cells, blue_cells))
    if not pairs:
        return None

    rng.shuffle(pairs)
    for red_cells, blue_cells in pairs:
        occupied = red_cells | blue_cells
        markers = _edge_choices(shape, occupied)
        if not markers:
            continue
        marker_cell = rng.choice(markers)
        top_color = rng.choice([RED, BLUE])
        inp = _render_input(shape, red_cells, blue_cells, marker_cell, top_color=top_color)
        expected = _expected_overlap_output(shape, red_cells, blue_cells)
        return inp, expected
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    branch = rng.choice(["disjoint", "overlap"])
    result = _gen_disjoint_case(rng) if branch == "disjoint" else _gen_overlap_case(rng)
    if result is None:
        return None

    inp, expected = result
    try:
        out = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(out, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
