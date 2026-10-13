from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
ORANGE = 7
TEAL = 8
OBSTACLE_COLORS = (RED, GREEN, YELLOW, ORANGE)
PATTERN_COLORS = (BLUE, TEAL)
PERIOD_CHOICES = (2, 3)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    h, w = grid.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(cells, dtype=int))
    return components


def _bracket_candidates(box: tuple[int, int, int, int]) -> list[set[tuple[int, int]]]:
    r0, c0, r1, c1 = box
    return [
        {(r0, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0, r1 + 1)},
        {(r0, c) for c in range(c0, c1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)},
        {(r1, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0, r1 + 1)},
        {(r1, c) for c in range(c0, c1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)},
    ]


def _is_corner_bracket(coords: np.ndarray) -> bool:
    box = _bbox_coords(coords)
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    area = int(coords.shape[0])
    if min(h, w) < 2:
        return False
    if area not in (3, 4):
        return False
    if area != h + w - 1:
        return False
    coord_set = {tuple(map(int, cell)) for cell in coords}
    return any(coord_set == candidate for candidate in _bracket_candidates(box))


def _find_bracket_and_marker(
    grid: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int], int]:
    gray_components = _components_of_color(grid, GRAY)
    brackets = [coords for coords in gray_components if _is_corner_bracket(coords)]
    assert len(brackets) == 1, "expected exactly one gray corner bracket"

    bracket = brackets[0]
    box = _bbox_coords(bracket)
    bracket_cells = {tuple(map(int, cell)) for cell in bracket}
    r0, c0, r1, c1 = box
    markers = [
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if (r, c) not in bracket_cells and int(grid[r, c]) not in (BLUE, TEAL, GRAY)
    ]
    assert len(markers) == 1, "expected one marker inside the bracket bbox"
    marker = markers[0]
    return bracket, box, marker, int(grid[marker])


def _pattern_from_tile(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    rr = np.arange(h)[:, None] % tile.shape[0]
    cc = np.arange(w)[None, :] % tile.shape[1]
    return tile[rr, cc]


def _infer_pattern_tile(
    grid: np.ndarray,
    bracket_box: tuple[int, int, int, int],
) -> tuple[int, int, np.ndarray]:
    visible = np.isin(grid, PATTERN_COLORS)
    r0, c0, r1, c1 = bracket_box
    visible[r0 : r1 + 1, c0 : c1 + 1] = False
    coords = np.argwhere(visible)
    assert len(coords) > 0, "expected visible pattern cells"

    candidates = []
    for pr in PERIOD_CHOICES:
        for pc in PERIOD_CHOICES:
            tile = np.full((pr, pc), -1, dtype=int)
            ok = True
            for r, c in coords:
                rr = int(r) % pr
                cc = int(c) % pc
                color = int(grid[r, c])
                if tile[rr, cc] == -1:
                    tile[rr, cc] = color
                elif tile[rr, cc] != color:
                    ok = False
                    break
            if ok and np.all(tile != -1):
                candidates.append((pr, pc, tile))

    assert candidates, "failed to infer any valid period"
    min_area = min(pr * pc for pr, pc, _tile in candidates)
    best = [(pr, pc, tile) for pr, pc, tile in candidates if pr * pc == min_area]
    assert len(best) == 1, "expected a unique minimal repeating unit"
    return best[0]


def _rectangular_components_of_color(
    grid: np.ndarray,
    color: int,
) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    rectangles = []
    for coords in _components_of_color(grid, color):
        if coords.shape[0] < 4:
            continue
        box = _bbox_coords(coords)
        r0, c0, r1, c1 = box
        if coords.shape[0] == (r1 - r0 + 1) * (c1 - c0 + 1):
            rectangles.append((coords, box))
    return rectangles


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore only the marker-selected rectangle color back to the hidden periodic tiling.

    Input structure:
    - The grid contains a full-canvas blue/teal periodic tiling with row and column periods 2 or 3.
    - Several solid colored rectangles occlude the tiling; their colors come from red, green, yellow, and orange.
    - Exactly one gray 4-connected corner bracket appears on an uncovered area, and inside its bbox there is exactly one non-pattern marker pixel.

    Rule:
    - Find the gray corner bracket and read the color of the unique non-blue/non-teal/non-gray marker pixel inside its bbox.
    - Infer the smallest periodic blue/teal tile from the still-visible tiling cells outside the bracket bbox.
    - Restore every solid rectangular occluder of the marker's color by replacing it with the inferred tiling at those coordinates.
    - Leave all other occluders, the bracket, and the marker unchanged.

    Key insight:
    - The bracket is a color selector for which cover layer to remove, but the recovered content must come from global periodic-pattern inference rather than direct deletion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _bracket, bracket_box, marker, selected_color = _find_bracket_and_marker(g)
    pr, pc, tile = _infer_pattern_tile(g, bracket_box)
    _ = (pr, pc)
    restored_pattern = _pattern_from_tile(tile, g.shape)

    out = g.copy()
    for coords, _box in _rectangular_components_of_color(g, selected_color):
        rows = coords[:, 0]
        cols = coords[:, 1]
        out[rows, cols] = restored_pattern[rows, cols]
    out[marker] = selected_color
    return out


GRID_RANGE = (14, 18)
OBSTACLE_COUNT_CHOICES = (3, 4)
RECT_SIDE_RANGE = (2, 4)
BRACKET_SHAPES = ((2, 2), (2, 3), (3, 2))
GEN_ATTEMPTS = 500


def _exact_periods(tile: np.ndarray) -> tuple[int, int]:
    pr, pc = tile.shape
    row_period = next(
        p for p in range(1, pr + 1) if all(np.array_equal(tile[r], tile[(r + p) % pr]) for r in range(pr))
    )
    col_period = next(
        p for p in range(1, pc + 1) if all(np.array_equal(tile[:, c], tile[:, (c + p) % pc]) for c in range(pc))
    )
    return row_period, col_period


def _random_tile(pr: int, pc: int) -> np.ndarray:
    while True:
        bits = np.random.randint(0, 2, size=(pr, pc))
        if bits.min() == bits.max():
            continue
        tile = np.where(bits == 0, BLUE, TEAL).astype(int)
        if _exact_periods(tile) != (pr, pc):
            continue
        return tile


def _expand_box(box: tuple[int, int, int, int], shape: tuple[int, int], margin: int = 1) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = box
    return (
        max(0, r0 - margin),
        max(0, c0 - margin),
        min(h - 1, r1 + margin),
        min(w - 1, c1 + margin),
    )


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _random_free_box(
    shape: tuple[int, int],
    box_h: int,
    box_w: int,
    forbidden: Iterable[tuple[int, int, int, int]],
    *,
    margin: int,
) -> tuple[int, int, int, int] | None:
    h, w = shape
    positions = [
        (r0, c0)
        for r0 in range(h - box_h + 1)
        for c0 in range(w - box_w + 1)
    ]
    random.shuffle(positions)
    for r0, c0 in positions:
        box = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
        expanded = _expand_box(box, shape, margin=margin)
        if any(_boxes_overlap(expanded, other) for other in forbidden):
            continue
        return box
    return None


def _place_obstacles(
    grid: np.ndarray,
    colors: list[int],
) -> list[tuple[int, tuple[int, int, int, int]]]:
    placed: list[tuple[int, tuple[int, int, int, int]]] = []
    forbidden: list[tuple[int, int, int, int]] = []
    shape = grid.shape
    for color in colors:
        box = None
        for _ in range(80):
            box_h = random.randint(*RECT_SIDE_RANGE)
            box_w = random.randint(*RECT_SIDE_RANGE)
            box = _random_free_box(shape, box_h, box_w, forbidden, margin=1)
            if box is not None:
                break
        if box is None:
            raise AssertionError("failed to place obstacle")
        r0, c0, r1, c1 = box
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
        placed.append((color, box))
        forbidden.append(box)
    return placed


def _choose_bracket_layout(
    shape: tuple[int, int],
    forbidden: list[tuple[int, int, int, int]],
) -> tuple[tuple[int, int, int, int], list[tuple[int, int]], tuple[int, int]] | None:
    for _ in range(120):
        box_h, box_w = random.choice(BRACKET_SHAPES)
        box = _random_free_box(shape, box_h, box_w, forbidden, margin=1)
        if box is None:
            continue
        candidates = _bracket_candidates(box)
        bracket_cells = list(random.choice(candidates))
        empty_cells = [
            (r, c)
            for r in range(box[0], box[2] + 1)
            for c in range(box[1], box[3] + 1)
            if (r, c) not in bracket_cells
        ]
        marker = random.choice(empty_cells)
        return box, bracket_cells, marker
    return None


def _covers_all_residues(
    grid: np.ndarray,
    bracket_box: tuple[int, int, int, int],
    pr: int,
    pc: int,
) -> bool:
    visible = np.isin(grid, PATTERN_COLORS)
    r0, c0, r1, c1 = bracket_box
    visible[r0 : r1 + 1, c0 : c1 + 1] = False
    seen = {(int(r) % pr, int(c) % pc) for r, c in np.argwhere(visible)}
    return len(seen) == pr * pc


def _expected_output(
    inp: np.ndarray,
    full_pattern: np.ndarray,
    selected_color: int,
) -> np.ndarray:
    out = inp.copy()
    for coords, _box in _rectangular_components_of_color(inp, selected_color):
        rows = coords[:, 0]
        cols = coords[:, 1]
        out[rows, cols] = full_pattern[rows, cols]
    return out


def _validate_case(
    inp: np.ndarray,
    expected: np.ndarray,
    selected_color: int,
    obstacle_boxes: list[tuple[int, tuple[int, int, int, int]]],
    bracket_box: tuple[int, int, int, int],
    marker: tuple[int, int],
    pr: int,
    pc: int,
) -> bool:
    try:
        actual = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    candidates = []
    visible = np.isin(inp, PATTERN_COLORS)
    r0, c0, r1, c1 = bracket_box
    visible[r0 : r1 + 1, c0 : c1 + 1] = False
    coords = np.argwhere(visible)
    for cand_r in PERIOD_CHOICES:
        for cand_c in PERIOD_CHOICES:
            seen = {}
            ok = True
            for rr, cc in coords:
                key = (int(rr) % cand_r, int(cc) % cand_c)
                val = int(inp[rr, cc])
                if key in seen and seen[key] != val:
                    ok = False
                    break
                seen[key] = val
            if ok and len(seen) == cand_r * cand_c:
                candidates.append((cand_r, cand_c))
    if not candidates:
        return False
    min_area = min(r * c for r, c in candidates)
    if candidates.count((pr, pc)) != 1:
        return False
    if sum(1 for r, c in candidates if r * c == min_area) != 1:
        return False
    if min_area != pr * pc:
        return False
    if not _covers_all_residues(inp, bracket_box, pr, pc):
        return False

    if int(actual[marker]) != selected_color:
        return False
    if not np.array_equal(actual[r0 : r1 + 1, c0 : c1 + 1][inp[r0 : r1 + 1, c0 : c1 + 1] == GRAY], inp[r0 : r1 + 1, c0 : c1 + 1][inp[r0 : r1 + 1, c0 : c1 + 1] == GRAY]):
        return False

    restored = 0
    preserved = 0
    for color, box in obstacle_boxes:
        rr0, cc0, rr1, cc1 = box
        before = inp[rr0 : rr1 + 1, cc0 : cc1 + 1]
        after = actual[rr0 : rr1 + 1, cc0 : cc1 + 1]
        if color == selected_color:
            if np.array_equal(before, after):
                return False
            restored += 1
        else:
            if not np.array_equal(before, after):
                return False
            preserved += 1
    return restored >= 1 and preserved >= 2


def _generate_case() -> np.ndarray | None:
    h = random.randint(*GRID_RANGE)
    w = random.randint(*GRID_RANGE)
    pr = random.choice(PERIOD_CHOICES)
    pc = random.choice(PERIOD_CHOICES)
    base_tile = _random_tile(pr, pc)
    phase_r = random.randrange(pr)
    phase_c = random.randrange(pc)
    shifted_tile = np.empty_like(base_tile)
    for r in range(pr):
        for c in range(pc):
            shifted_tile[r, c] = base_tile[(r + phase_r) % pr, (c + phase_c) % pc]
    full_pattern = _pattern_from_tile(shifted_tile, (h, w))
    inp = full_pattern.copy()

    obstacle_count = random.choice(OBSTACLE_COUNT_CHOICES)
    colors = random.sample(list(OBSTACLE_COLORS), obstacle_count)
    try:
        obstacles = _place_obstacles(inp, colors)
    except AssertionError:
        return None

    forbidden = [box for _color, box in obstacles]
    bracket_layout = _choose_bracket_layout((h, w), forbidden)
    if bracket_layout is None:
        return None
    bracket_box, bracket_cells, marker = bracket_layout

    selected_color = random.choice(colors)
    for r, c in bracket_cells:
        inp[r, c] = GRAY
    inp[marker] = selected_color

    if not _covers_all_residues(inp, bracket_box, pr, pc):
        return None

    expected = _expected_output(inp, full_pattern, selected_color)
    if not _validate_case(
        inp,
        expected,
        selected_color,
        obstacles,
        bracket_box,
        marker,
        pr,
        pc,
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
