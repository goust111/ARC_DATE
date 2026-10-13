from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
NON_BACKGROUND_COLORS = tuple(range(1, 10))
RIGHT = "right"
LEFT = "left"
SIDES = (RIGHT, LEFT)
GENERATION_ATTEMPTS = 700
BLOCK_CLEARANCE = 1
SIDE_GAP = 3
MAX_FUEL = 28

_GEN_COUNTER = 0


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = map(int, mask.shape)
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []

    for sr, sc in np.argwhere(mask):
        r0 = int(sr)
        c0 = int(sc)
        if seen[r0, c0]:
            continue

        stack = [(r0, c0)]
        seen[r0, c0] = True
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))

    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_roles(
    grid: np.ndarray,
) -> tuple[tuple[int, int], int, np.ndarray, int, tuple[int, int, int, int]]:
    assert grid.ndim == 2, "expected a 2D grid"
    assert max(grid.shape) <= 30, "ARC grids must fit within 30x30"

    comps = _components(grid != BACKGROUND)
    assert len(comps) == 2, "expected exactly two foreground components"

    marker_comps = [comp for comp in comps if len(comp) == 1]
    block_comps = [comp for comp in comps if len(comp) > 1]
    assert len(marker_comps) == 1 and len(block_comps) == 1, "expected one marker and one fuel block"

    marker_r, marker_c = map(int, marker_comps[0][0])
    marker_color = int(grid[marker_r, marker_c])

    block = block_comps[0]
    block_values = grid[block[:, 0], block[:, 1]]
    block_color = int(block_values[0])
    assert np.all(block_values == block_color), "fuel block must be monochrome"

    box = _bbox(block)
    assert box[1] + box[3] != 2 * marker_c, "fuel block center column cannot match the marker column"
    return (marker_r, marker_c), marker_color, block, block_color, box


def _side_from_bbox(
    marker: tuple[int, int],
    block_box: tuple[int, int, int, int],
) -> str:
    _mr, mc = marker
    _r0, c0, _r1, c1 = block_box
    return RIGHT if c0 + c1 > 2 * mc else LEFT


def _spiral_directions(side: str) -> tuple[tuple[int, int], ...]:
    if side == RIGHT:
        return ((0, 1), (1, 0), (0, -1), (-1, 0))
    assert side == LEFT, "unknown side"
    return ((0, -1), (1, 0), (0, 1), (-1, 0))


def _spiral_cells(
    shape: tuple[int, int],
    marker: tuple[int, int],
    side: str,
    count: int,
) -> list[tuple[int, int]]:
    height, width = map(int, shape)
    r, c = marker
    cells: list[tuple[int, int]] = []
    seen = {marker}
    directions = _spiral_directions(side)
    step_len = 1
    turn = 0

    while len(cells) < count:
        dr, dc = directions[turn % 4]
        for _ in range(step_len):
            if len(cells) == count:
                break
            r += dr
            c += dc
            assert 0 <= r < height and 0 <= c < width, "spiral must fit inside the grid"
            assert (r, c) not in seen, "square spiral should not cross itself"
            seen.add((r, c))
            cells.append((r, c))
        step_len += 1
        turn += 1

    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the fuel block and spend its area drawing a same-colored square spiral from the marker.

    Input structure:
    - The grid background is black and contains exactly two foreground components:
      one single-cell center marker and one remote monochrome connected fuel block.
    - The fuel block's bounding-box center column is not the marker column, so the
      block is unambiguously on the marker's left or right side.

    Rule:
    - Clear every cell of the fuel block to black.
    - Starting from the marker, first step horizontally toward the fuel block and
      then walk square-spiral legs of lengths 1, 2, 3, ... .
    - A right-side fuel block uses clockwise turns; a left-side fuel block uses
      counterclockwise turns. Paint exactly as many spiral cells as the block area,
      all in the block color.
    - Keep the marker cell in its original color.

    Key insight:
    - The distant block is fuel rather than a shape to copy: its side chooses the
      spiral handedness, while its pixel count is the amount of ink to spend.
    """
    g = np.asarray(grid, dtype=int)
    marker, marker_color, block, block_color, block_box = _extract_roles(g)
    side = _side_from_bbox(marker, block_box)

    out = g.copy()
    out[block[:, 0], block[:, 1]] = BACKGROUND

    for r, c in _spiral_cells(g.shape, marker, side, len(block)):
        out[r, c] = block_color
    out[marker] = marker_color
    return out


def _dilated_mask(
    shape: tuple[int, int],
    cells: list[tuple[int, int]],
    radius: int,
) -> np.ndarray:
    height, width = map(int, shape)
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    mask[rr, cc] = True
    return mask


def _candidate_area_mask(
    shape: tuple[int, int],
    marker: tuple[int, int],
    side: str,
    reserved_cells: list[tuple[int, int]],
) -> np.ndarray:
    _mr, mc = marker
    height, width = map(int, shape)
    allowed = np.ones(shape, dtype=bool)

    if side == RIGHT:
        allowed[:, : mc + SIDE_GAP + 1] = False
    else:
        allowed[:, max(0, mc - SIDE_GAP) :] = False

    allowed &= ~_dilated_mask(shape, reserved_cells, BLOCK_CLEARANCE)

    # Avoid tiny edge slivers that make the fuel look like noise rather than a block.
    if height > 4 and width > 4:
        allowed[0, :] = False
        allowed[-1, :] = False
        allowed[:, 0] = False
        allowed[:, -1] = False
    return allowed


def _grow_connected_blob(component: np.ndarray, area: int) -> set[tuple[int, int]] | None:
    comp_set = {(int(r), int(c)) for r, c in component}
    if len(comp_set) < area:
        return None

    seed = random.choice(tuple(comp_set))
    coords = {seed}
    frontier = {
        nxt
        for nxt in _neighbors4(*seed)
        if nxt in comp_set
    }

    while len(coords) < area:
        if not frontier:
            return None

        options = tuple(frontier)
        weights = []
        for r, c in options:
            touching = sum(nbr in coords for nbr in _neighbors4(r, c))
            dist = abs(r - seed[0]) + abs(c - seed[1])
            weights.append((touching + 1) ** 3 / (dist + 1))

        cell = random.choices(options, weights=weights, k=1)[0]
        frontier.remove(cell)
        coords.add(cell)
        for nxt in _neighbors4(*cell):
            if nxt in comp_set and nxt not in coords:
                frontier.add(nxt)

    return coords


def _blob_is_clear(coords: set[tuple[int, int]], area: int, marker: tuple[int, int], side: str) -> bool:
    arr = np.array(sorted(coords), dtype=int)
    r0, c0, r1, c1 = _bbox(arr)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    density = area / (height * width)
    if height > 8 or width > 8 or density < 0.38:
        return False

    _mr, mc = marker
    if side == RIGHT and c0 + c1 <= 2 * mc:
        return False
    if side == LEFT and c0 + c1 >= 2 * mc:
        return False
    return True


def _choose_area(mode: int) -> int:
    exact_segment_endings = (6, 10, 15, 21, 28)
    partial_segment_stops = tuple(
        n for n in range(6, MAX_FUEL + 1) if n not in exact_segment_endings
    )
    if mode == 0:
        return random.choice(exact_segment_endings)
    return random.choice(partial_segment_stops)


def _semantic_check(grid: np.ndarray, expected_side: str, expected_area: int) -> bool:
    try:
        marker, marker_color, block, block_color, block_box = _extract_roles(grid)
        side = _side_from_bbox(marker, block_box)
        if side != expected_side or len(block) != expected_area:
            return False
        if marker_color == block_color:
            return False

        spiral = _spiral_cells(grid.shape, marker, side, len(block))
        if set(map(tuple, block.tolist())) & set(spiral):
            return False

        expected = np.zeros_like(grid)
        expected[marker] = marker_color
        for r, c in spiral:
            expected[r, c] = block_color

        out = main(grid)
        if not np.array_equal(out, expected):
            return False
        if np.array_equal(out, grid):
            return False
        if int(np.count_nonzero(out == block_color)) != expected_area:
            return False
        if any(int(out[r, c]) != BACKGROUND for r, c in block):
            return False
    except AssertionError:
        return False
    return True


def _marker_for_side(height: int, width: int, side: str) -> tuple[int, int] | None:
    if height < 14 or width < 16:
        return None

    row_min = 5
    row_max = height - 6
    if row_min > row_max:
        return None

    if side == RIGHT:
        col_min = 5
        col_max = width - 9
    else:
        col_min = 8
        col_max = width - 6
    if col_min > col_max:
        return None

    return random.randint(row_min, row_max), random.randint(col_min, col_max)


def _gen_attempt(side: str, area_mode: int) -> np.ndarray | None:
    height = random.randint(16, 25)
    width = random.randint(17, 26)
    marker = _marker_for_side(height, width, side)
    if marker is None:
        return None

    area = _choose_area(area_mode)
    try:
        spiral = _spiral_cells((height, width), marker, side, area)
    except AssertionError:
        return None

    allowed = _candidate_area_mask((height, width), marker, side, [marker, *spiral])
    candidates = [comp for comp in _components(allowed) if len(comp) >= area]
    if not candidates:
        return None

    random.shuffle(candidates)
    blob: set[tuple[int, int]] | None = None
    for component in candidates[:8]:
        for _ in range(30):
            coords = _grow_connected_blob(component, area)
            if coords is not None and _blob_is_clear(coords, area, marker, side):
                blob = coords
                break
        if blob is not None:
            break
    if blob is None:
        return None

    marker_color, block_color = random.sample(NON_BACKGROUND_COLORS, 2)
    grid = np.zeros((height, width), dtype=int)
    grid[marker] = marker_color
    for r, c in blob:
        grid[r, c] = block_color

    if _semantic_check(grid, side, area):
        return grid
    return None


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    side = SIDES[_GEN_COUNTER % len(SIDES)]
    area_mode = _GEN_COUNTER % 3
    _GEN_COUNTER += 1

    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(side, area_mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fuel spiral input")
