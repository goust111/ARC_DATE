import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

FRAME_SIZE = 4
GAP_OFFSETS = (1, 2)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

SYMMETRIC_SHAPES = (
    {(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)},
    {(-2, 0), (-1, 0), (0, -1), (0, 0), (0, 1), (1, 0), (2, 0)},
    {(-1, 0), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (1, 0)},
)

INWARD_SIDES = {
    (-1, -1): ("right", "bottom"),
    (-1, 1): ("left", "bottom"),
    (1, -1): ("right", "top"),
    (1, 1): ("left", "top"),
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: set[tuple[int, int]] | list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _reflect_point(
    r: int,
    c: int,
    axis_r: int,
    axis_c: int,
    flip_r: bool,
    flip_c: bool,
) -> tuple[int, int]:
    if flip_r:
        r = 2 * axis_r - r
    if flip_c:
        c = 2 * axis_c - c
    return r, c


def _frame_border(r0: int, c0: int) -> set[tuple[int, int]]:
    cells = set()
    for r in range(r0, r0 + FRAME_SIZE):
        for c in range(c0, c0 + FRAME_SIZE):
            if r in (r0, r0 + FRAME_SIZE - 1) or c in (c0, c0 + FRAME_SIZE - 1):
                cells.add((r, c))
    return cells


def _gap_cell(r0: int, c0: int, side: str, offset: int) -> tuple[int, int]:
    if side == "top":
        return r0, c0 + offset
    if side == "bottom":
        return r0 + FRAME_SIZE - 1, c0 + offset
    if side == "left":
        return r0 + offset, c0
    if side == "right":
        return r0 + offset, c0 + FRAME_SIZE - 1
    raise AssertionError(f"unknown side: {side}")


def _frame_from_params(r0: int, c0: int, side: str, offset: int) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    border = _frame_border(r0, c0)
    gap = _gap_cell(r0, c0, side, offset)
    border.remove(gap)
    return border, gap


def _gap_direction(gap: tuple[int, int], bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r, c = gap
    r0, c0, r1, c1 = bbox
    if r == r0:
        return -1, 0
    if r == r1:
        return 1, 0
    if c == c0:
        return 0, -1
    if c == c1:
        return 0, 1
    raise AssertionError("gap is not on the frame boundary")


def _detect_axes(green_cells: set[tuple[int, int]]) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(green_cells)
    axis_r = (r0 + r1) // 2
    axis_c = (c0 + c1) // 2
    assert all((2 * axis_r - r, c) in green_cells for r, c in green_cells)
    assert all((r, 2 * axis_c - c) in green_cells for r, c in green_cells)
    return axis_r, axis_c


def _detect_frame(grid: np.ndarray) -> tuple[set[tuple[int, int]], tuple[int, int], tuple[int, int, int, int]]:
    blue_mask = grid == BLUE
    comps = _components(blue_mask)
    assert len(comps) == 1, "expected exactly one blue component"
    cells = set(comps[0])
    assert len(cells) == 11, "expected a 4x4 one-gap frame"
    bbox = _bbox(cells)
    r0, c0, r1, c1 = bbox
    assert (r1 - r0 + 1, c1 - c0 + 1) == (FRAME_SIZE, FRAME_SIZE)
    full_border = _frame_border(r0, c0)
    gap_candidates = list(full_border - cells)
    assert len(gap_candidates) == 1, "expected exactly one missing border cell"
    gap = gap_candidates[0]
    gr, gc = gap
    assert (gr in (r0, r1) and c0 < gc < c1) or (gc in (c0, c1) and r0 < gr < r1)
    return cells, gap, bbox


def _all_variants(
    walls: set[tuple[int, int]],
    gap: tuple[int, int],
    axis_r: int,
    axis_c: int,
) -> list[tuple[set[tuple[int, int]], tuple[int, int], tuple[int, int, int, int]]]:
    variants = []
    for flip_r in (False, True):
        for flip_c in (False, True):
            new_wall = {
                _reflect_point(r, c, axis_r, axis_c, flip_r, flip_c)
                for r, c in walls
            }
            new_gap = _reflect_point(gap[0], gap[1], axis_r, axis_c, flip_r, flip_c)
            new_bbox = _bbox(new_wall | {new_gap})
            variants.append((new_wall, new_gap, new_bbox))
    return variants


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect the one-gap frame across the marker's symmetry axes, then flood red from each gap until blocked.

    Input structure:
    - The grid has black background, one green object with both horizontal and vertical mirror symmetry, and one blue 4x4 rectangular frame with exactly one non-corner border gap.
    - The green object supplies the horizontal and vertical reflection axes.
    - The blue frame sits fully inside one quadrant relative to those axes, with a 2x2 black interior.

    Rule:
    - Detect the green object's symmetry axes and reflect the blue gap-frame across the horizontal axis, vertical axis, and both axes, producing four blue frame copies including the original.
    - For every copy, fill its 2x2 interior red and recolor its gap cell red.
    - From each red gap cell, cast a one-cell-wide red ray outward in the gap's facing direction, stopping before any green pixel, any blue frame-wall pixel, or the grid boundary.

    Key insight:
    - The symmetric green object is not transformed; it acts as an axis donor that determines where all four gap-frames appear and therefore how their outward red rays interact.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    green_cells = {tuple(map(int, rc)) for rc in np.argwhere(g == GREEN)}
    assert green_cells, "expected a green symmetric object"
    axis_r, axis_c = _detect_axes(green_cells)

    input_walls, input_gap, _ = _detect_frame(g)
    variants = _all_variants(input_walls, input_gap, axis_r, axis_c)

    out = np.full_like(g, BLACK)
    for r, c in green_cells:
        out[r, c] = GREEN

    all_walls: set[tuple[int, int]] = set()
    for wall_cells, _, bbox in variants:
        r0, c0, r1, c1 = bbox
        assert (r1 - r0 + 1, c1 - c0 + 1) == (FRAME_SIZE, FRAME_SIZE)
        all_walls |= wall_cells
    assert all_walls.isdisjoint(green_cells), "reflected frames must not overlap the green object"

    for r, c in all_walls:
        out[r, c] = BLUE

    obstacles = all_walls | green_cells

    for _, gap, bbox in variants:
        r0, c0, r1, c1 = bbox
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                assert (r, c) not in obstacles
                out[r, c] = RED
        assert gap not in green_cells
        out[gap] = RED

    for _, gap, bbox in variants:
        dr, dc = _gap_direction(gap, bbox)
        r, c = gap[0] + dr, gap[1] + dc
        while 0 <= r < h and 0 <= c < w and (r, c) not in obstacles:
            out[r, c] = RED
            r += dr
            c += dc

    return out


def _shape_extents(shape: set[tuple[int, int]]) -> tuple[int, int]:
    rows = [abs(r) for r, _ in shape]
    cols = [abs(c) for _, c in shape]
    return max(rows), max(cols)


def _paint_shape(grid: np.ndarray, center_r: int, center_c: int, shape: set[tuple[int, int]], color: int) -> None:
    for dr, dc in shape:
        grid[center_r + dr, center_c + dc] = color


def _validate_layout(
    h: int,
    w: int,
    green_cells: set[tuple[int, int]],
    variants: list[tuple[set[tuple[int, int]], tuple[int, int], tuple[int, int, int, int]]],
) -> bool:
    for wall_cells, gap, bbox in variants:
        all_cells = wall_cells | {gap}
        if any(not (0 <= r < h and 0 <= c < w) for r, c in all_cells):
            return False
        r0, c0, r1, c1 = bbox
        if (r1 - r0 + 1, c1 - c0 + 1) != (FRAME_SIZE, FRAME_SIZE):
            return False
        if all_cells & green_cells:
            return False
    frame_masks = [wall_cells | {gap} for wall_cells, gap, _ in variants]
    for i in range(len(frame_masks)):
        for j in range(i + 1, len(frame_masks)):
            if frame_masks[i] & frame_masks[j]:
                return False
    if len({tuple(sorted(mask)) for mask in frame_masks}) != 4:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.choice((19, 21, 23, 25))
    w = random.choice((19, 21, 23, 25))
    axis_r = h // 2
    axis_c = w // 2
    shape = random.choice(SYMMETRIC_SHAPES)
    half_h, half_w = _shape_extents(shape)

    quad_r, quad_c = random.choice(tuple(INWARD_SIDES))
    side = random.choice(INWARD_SIDES[(quad_r, quad_c)])
    gap_offset = random.choice(GAP_OFFSETS)

    top_min = 1
    top_max = axis_r - half_h - FRAME_SIZE - 1
    bottom_min = axis_r + half_h + 2
    bottom_max = h - FRAME_SIZE - 1
    left_min = 1
    left_max = axis_c - half_w - FRAME_SIZE - 1
    right_min = axis_c + half_w + 2
    right_max = w - FRAME_SIZE - 1

    if quad_r == -1:
        if top_min > top_max:
            return None
        r0 = random.randint(top_min, top_max)
    else:
        if bottom_min > bottom_max:
            return None
        r0 = random.randint(bottom_min, bottom_max)

    if quad_c == -1:
        if left_min > left_max:
            return None
        c0 = random.randint(left_min, left_max)
    else:
        if right_min > right_max:
            return None
        c0 = random.randint(right_min, right_max)

    walls, gap = _frame_from_params(r0, c0, side, gap_offset)
    green_cells = {(axis_r + dr, axis_c + dc) for dr, dc in shape}
    variants = _all_variants(walls, gap, axis_r, axis_c)

    if not _validate_layout(h, w, green_cells, variants):
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    _paint_shape(grid, axis_r, axis_c, shape, GREEN)
    for r, c in walls:
        grid[r, c] = BLUE

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if not np.any(out == RED):
        return None
    if np.count_nonzero(out == BLUE) != 44:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid reflected-gap-frame input")
