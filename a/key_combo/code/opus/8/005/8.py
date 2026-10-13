from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
TEAL = 8

PANEL_SIZE = 8
INPUT_SHAPE = (PANEL_SIZE, PANEL_SIZE * 2)
GEN_RETRY_LIMIT = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _turn_clockwise(dr: int, dc: int) -> tuple[int, int]:
    return int(dc), int(-dr)


def _draw_spiral(obstacles: np.ndarray) -> np.ndarray:
    h, w = obstacles.shape
    path = np.zeros((h, w), dtype=bool)
    if h == 0 or w == 0 or bool(obstacles[0, 0]):
        return path

    def blocked(r: int, c: int) -> bool:
        return not (0 <= r < h and 0 <= c < w) or bool(path[r, c]) or bool(obstacles[r, c])

    r, c = 0, 0
    dr, dc = 0, 1
    path[r, c] = True

    while True:
        nr, nc = r + dr, c + dc
        if blocked(nr, nc):
            ndr, ndc = _turn_clockwise(dr, dc)
            tr, tc = r + ndr, c + ndc
            if blocked(tr, tc):
                break
            dr, dc = ndr, ndc
            r, c = tr, tc
        else:
            r, c = nr, nc
        path[r, c] = True

    return path


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = grid.shape
    assert h == PANEL_SIZE, "expected 8 rows"
    assert w == PANEL_SIZE * 2, "expected two 8x8 panels"
    return grid[:, :PANEL_SIZE], grid[:, PANEL_SIZE:]


def _solve(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    box = _bbox(left == TEAL)
    assert box is not None, "expected an incomplete teal square in the left panel"
    r0, c0, r1, c1 = box

    obstacle = (left == BLUE) | (right == RED)
    out = np.zeros_like(left)
    out[obstacle] = GREY

    out[r0, c0 : c1 + 1] = TEAL
    out[r1, c0 : c1 + 1] = TEAL
    out[r0 : r1 + 1, c0] = TEAL
    out[r0 : r1 + 1, c1] = TEAL

    if r1 - r0 >= 2 and c1 - c0 >= 2:
        inner_obstacles = obstacle[r0 + 1 : r1, c0 + 1 : c1]
        spiral = _draw_spiral(inner_obstacles)
        out[r0 + 1 : r1, c0 + 1 : c1][spiral] = GREEN

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the two aligned panels to build obstacles, complete the teal square, and draw a spiral that turns away from those obstacles.

    Input structure:
    - The input is an 8x16 black grid split into two aligned 8x8 panels.
    - The left panel contains blue foreground pixels and one incomplete teal square outline with a few missing border cells.
    - The right panel contains red foreground pixels at aligned coordinates, with no teal pixels.

    Rule:
    - Split the input into left and right 8x8 panels and mark an obstacle wherever the left panel has blue or the right panel has red at the same aligned cell.
    - Infer the teal square's bbox from the visible teal outline in the left panel and complete that full border in the output.
    - Treat the square interior as the spiral canvas, then start from its upper-left interior cell and draw a clockwise green path.
    - Whenever the next step would leave the interior, hit an obstacle, or hit an already drawn green cell, turn clockwise; if the turned step is also blocked, stop.
    - Output a single 8x8 grid containing the completed teal border, gray obstacles, and the green spiral on black background.

    Key insight:
    - The two panels are not merged by color replacement; their Boolean union becomes a collision map that changes where the spiral can travel inside the completed square.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    left, right = _split_panels(g)
    return _solve(left, right)


def _border_coords(r0: int, c0: int, side: int) -> list[tuple[int, int]]:
    r1 = r0 + side - 1
    c1 = c0 + side - 1
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        coords.append((r, c1))
    return coords


def _sides_for_cell(r: int, c: int, r0: int, c0: int, r1: int, c1: int) -> list[int]:
    sides = []
    if r == r0:
        sides.append(0)
    if r == r1:
        sides.append(1)
    if c == c0:
        sides.append(2)
    if c == c1:
        sides.append(3)
    return sides


def _choose_missing_border_cells(r0: int, c0: int, side: int, count: int) -> set[tuple[int, int]]:
    r1 = r0 + side - 1
    c1 = c0 + side - 1
    candidates = _border_coords(r0, c0, side)
    counts = {0: side, 1: side, 2: side, 3: side}
    removed: set[tuple[int, int]] = set()

    while len(removed) < count:
        choices = [
            cell
            for cell in candidates
            if cell not in removed
            and all(counts[idx] > 1 for idx in _sides_for_cell(cell[0], cell[1], r0, c0, r1, c1))
        ]
        if not choices:
            raise RuntimeError("failed to remove border cells without erasing a whole side")
        cell = random.choice(choices)
        removed.add(cell)
        for idx in _sides_for_cell(cell[0], cell[1], r0, c0, r1, c1):
            counts[idx] -= 1

    return removed


def _all_panel_cells() -> list[tuple[int, int]]:
    return [(r, c) for r in range(PANEL_SIZE) for c in range(PANEL_SIZE)]


def _place_union_pixels(
    allowed: list[tuple[int, int]],
    interior: list[tuple[int, int]],
    outside: list[tuple[int, int]],
    n_blue: int,
    n_red: int,
    require_inner: bool,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    max_union = min(len(allowed), n_blue + n_red, 8)
    min_union = max(n_blue, n_red)
    if min_union > max_union:
        return None

    union_size = random.randint(min_union, max_union)
    overlap = n_blue + n_red - union_size
    if overlap < 0 or overlap > min(n_blue, n_red):
        return None

    inner_max = min(len(interior), union_size)
    if require_inner and inner_max == 0:
        return None

    if require_inner:
        upper = min(inner_max, max(1, union_size - 1))
        inner_count = random.randint(1, upper)
    else:
        inner_count = random.randint(0, inner_max)

    outer_count = union_size - inner_count
    if outer_count > len(outside):
        return None

    union_cells = set(random.sample(interior, inner_count)) | set(random.sample(outside, outer_count))
    union_list = list(union_cells)
    overlap_cells = set(random.sample(union_list, overlap))
    remaining = [cell for cell in union_list if cell not in overlap_cells]
    random.shuffle(remaining)

    blue_only_count = n_blue - overlap
    red_only_count = n_red - overlap
    blue_only = set(remaining[:blue_only_count])
    red_only = set(remaining[blue_only_count : blue_only_count + red_only_count])

    blue = overlap_cells | blue_only
    red = overlap_cells | red_only
    if len(blue) != n_blue or len(red) != n_red or blue | red != union_cells:
        return None
    return blue, red


def _empty_spiral_mask(box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=bool)
    spiral = _draw_spiral(np.zeros((r1 - r0 - 1, c1 - c0 - 1), dtype=bool))
    mask[r0 + 1 : r1, c0 + 1 : c1] = spiral
    return mask


def _generate_candidate() -> np.ndarray | None:
    left = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=int)
    right = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=int)

    side = random.choice([6, 6, 7])
    r0 = random.randint(0, PANEL_SIZE - side)
    c0 = random.randint(0, PANEL_SIZE - side)
    r1 = r0 + side - 1
    c1 = c0 + side - 1
    box = (r0, c0, r1, c1)

    missing = _choose_missing_border_cells(r0, c0, side, random.randint(2, 4))
    for r, c in _border_coords(r0, c0, side):
        if (r, c) not in missing:
            left[r, c] = TEAL

    start = (r0 + 1, c0 + 1)
    border = set(_border_coords(r0, c0, side))
    interior = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if (r, c) != start
    ]
    outside = [
        cell
        for cell in _all_panel_cells()
        if cell not in border and not (r0 < cell[0] < r1 and c0 < cell[1] < c1)
    ]
    allowed = interior + outside

    inner_limit = ((side - 2) * (side - 2)) // 3
    require_effect = random.random() < 0.7

    n_blue = random.randint(3, 6)
    n_red = random.randint(3, 6)
    placed = _place_union_pixels(allowed, interior, outside, n_blue, n_red, require_effect)
    if placed is None:
        return None
    blue_cells, red_cells = placed

    union_inner = len((blue_cells | red_cells) & set(interior))
    if union_inner > inner_limit:
        return None

    for r, c in blue_cells:
        left[r, c] = BLUE
    for r, c in red_cells:
        right[r, c] = RED

    grid = np.concatenate([left, right], axis=1)
    out = main(grid)

    if np.sum(grid[:, :PANEL_SIZE] == BLUE) != n_blue:
        return None
    if np.sum(grid[:, PANEL_SIZE:] == RED) != n_red:
        return None
    if out.shape != (PANEL_SIZE, PANEL_SIZE):
        return None
    if tuple(np.argwhere(left == TEAL).shape) == (0, 2):
        return None
    if out[start] == GREY:
        return None
    if np.sum(out == GREEN) < 4:
        return None

    if require_effect:
        if union_inner == 0:
            return None
        if np.array_equal(out == GREEN, _empty_spiral_mask(box)):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
