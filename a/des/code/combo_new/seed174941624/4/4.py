from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
AZURE = 8
PALETTE = [1, 2, 3, 4, 6, 7, 9]

IDENTITY = "identity"
HFLIP = "hflip"
VFLIP = "vflip"
ROT180 = "rot180"

ICON_MASKS = {
    IDENTITY: np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    HFLIP: np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [1, 0, 0],
        ],
        dtype=bool,
    ),
    VFLIP: np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=bool,
    ),
    ROT180: np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=bool,
    ),
}

QUADRANTS = ((0, 0), (0, 1), (1, 0), (1, 1))
MAX_GENERATION_ATTEMPTS = 400
OBJECT_PLACEMENT_ATTEMPTS = 200
SHAPE_ATTEMPTS = 200
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _find_cross(g: np.ndarray) -> tuple[int, int]:
    rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]
    cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    assert len(rows) == 1 and len(cols) == 1
    return rows[0], cols[0]


def _transform_mask(mask: np.ndarray, op: str) -> np.ndarray:
    if op == IDENTITY:
        return mask.copy()
    if op == HFLIP:
        return np.fliplr(mask)
    if op == VFLIP:
        return np.flipud(mask)
    if op == ROT180:
        return np.rot90(mask, 2)
    raise AssertionError(f"unknown operation: {op}")


def _decode_icon(patch: np.ndarray) -> str:
    assert patch.shape == (3, 3)
    assert np.all((patch == BG) | (patch == AZURE))
    mask = patch == AZURE
    for op, icon in ICON_MASKS.items():
        if np.array_equal(mask, icon):
            return op
    raise AssertionError("unknown corner icon")


def _quadrant_bounds(shape: tuple[int, int], mid_r: int, mid_c: int, quadrant: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    if quadrant == (0, 0):
        return 0, mid_r, 0, mid_c
    if quadrant == (0, 1):
        return 0, mid_r, mid_c + 1, w
    if quadrant == (1, 0):
        return mid_r + 1, h, 0, mid_c
    if quadrant == (1, 1):
        return mid_r + 1, h, mid_c + 1, w
    raise AssertionError("invalid quadrant")


def _corner_patch(g: np.ndarray, quadrant: tuple[int, int]) -> np.ndarray:
    h, w = g.shape
    if quadrant == (0, 0):
        return g[:3, :3]
    if quadrant == (0, 1):
        return g[:3, w - 3 :]
    if quadrant == (1, 0):
        return g[h - 3 :, :3]
    if quadrant == (1, 1):
        return g[h - 3 :, w - 3 :]
    raise AssertionError("invalid quadrant")


def _component_masks(region: np.ndarray) -> list[tuple[int, tuple[int, int, int, int], np.ndarray]]:
    h, w = region.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[tuple[int, tuple[int, int, int, int], np.ndarray]] = []
    for r in range(h):
        for c in range(w):
            color = int(region[r, c])
            if visited[r, c] or color in (BG, GREY, AZURE):
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(region[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))
            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in cells:
                mask[rr - r0, cc - c0] = True
            components.append((color, (r0, r1, c0, c1), mask))
    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply a quadrant-specific geometric transform to each colored object, using the Azure corner icons as the operation code.

    Input structure:
    - The background is black.
    - One full grey row and one full grey column form a cross that splits the grid into four equal quadrants.
    - Each corner contains a 3x3 Azure icon: a solid square means keep the shapes unchanged, a left-heavy icon means horizontal flip, a top-heavy icon means vertical flip, and a main-diagonal icon means rotate 180 degrees.
    - Each quadrant contains one or more connected non-black, non-grey, non-Azure shapes.

    Rule:
    - Decode the Azure icon in each corner to determine the transform assigned to that quadrant.
    - For every connected shape in that quadrant, compute its tight bounding box and apply the quadrant's transform inside that box.
    - Keep the transformed shape in the same quadrant and at the same bounding-box position.
    - Preserve the grey cross and the Azure icons unchanged.

    Key insight:
    - The corner sprites are not decoration: they are a visual code that independently tells you how to transform the objects in each quadrant.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    mid_r, mid_c = _find_cross(g)

    ops = {quadrant: _decode_icon(_corner_patch(g, quadrant)) for quadrant in QUADRANTS}
    out = g.copy()

    for quadrant in QUADRANTS:
        r0, r1, c0, c1 = _quadrant_bounds(g.shape, mid_r, mid_c, quadrant)
        region = g[r0:r1, c0:c1]
        components = _component_masks(region)
        placements: list[tuple[int, int, int, np.ndarray]] = []
        to_clear: list[tuple[int, int]] = []

        for color, (cr0, cr1, cc0, cc1), mask in components:
            transformed = _transform_mask(mask, ops[quadrant])
            placements.append((color, r0 + cr0, c0 + cc0, transformed))
            for rr in range(cr0, cr1 + 1):
                for cc in range(cc0, cc1 + 1):
                    if mask[rr - cr0, cc - cc0]:
                        to_clear.append((r0 + rr, c0 + cc))

        for rr, cc in to_clear:
            out[rr, cc] = BG
        for color, rr0, cc0, transformed in placements:
            patch = out[rr0 : rr0 + transformed.shape[0], cc0 : cc0 + transformed.shape[1]]
            patch[transformed] = color

    return out


def _overlaps_with_gap(
    bbox: tuple[int, int, int, int],
    others: list[tuple[int, int, int, int]],
    gap: int = 1,
) -> bool:
    r0, r1, c0, c1 = bbox
    for or0, or1, oc0, oc1 in others:
        if not (r1 + gap < or0 or or1 + gap < r0 or c1 + gap < oc0 or oc1 + gap < c0):
            return True
    return False


def _random_shape(max_h: int, max_w: int) -> np.ndarray | None:
    max_h = min(max_h, 5)
    max_w = min(max_w, 5)
    if max_h < 2 or max_w < 2:
        return None

    for _ in range(SHAPE_ATTEMPTS):
        h = random.randint(2, max_h)
        w = random.randint(2, max_w)
        area = h * w
        min_cells = max(4, (area + 2) // 3)
        max_cells = min(area - 1, max(min_cells, int(round(area * 0.7))))
        if min_cells > max_cells:
            continue
        target = random.randint(min_cells, max_cells)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier = set()
        for dr, dc in NEIGHBORS4:
            nr, nc = start[0] + dr, start[1] + dc
            if 0 <= nr < h and 0 <= nc < w:
                frontier.add((nr, nc))

        while len(cells) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for dr, dc in NEIGHBORS4:
                nr, nc = cell[0] + dr, cell[1] + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        if len(cells) != target:
            continue

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r0 == r1 or c0 == c1:
            continue

        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in cells:
            patch[r - r0, c - c0] = True

        if patch.all():
            continue
        if np.array_equal(patch, np.fliplr(patch)):
            continue
        if np.array_equal(patch, np.flipud(patch)):
            continue
        if np.array_equal(patch, np.rot90(patch, 2)):
            continue
        return patch

    return None


def _choose_object_count(avail_h: int, avail_w: int) -> int:
    area = avail_h * avail_w
    if area >= 36 and random.random() < 0.25:
        return 3
    if area >= 20 and random.random() < 0.55:
        return 2
    return 1


def _placement_window(
    shape: tuple[int, int],
    mid_r: int,
    mid_c: int,
    quadrant: tuple[int, int],
) -> tuple[int, int, int, int]:
    r0, r1, c0, c1 = _quadrant_bounds(shape, mid_r, mid_c, quadrant)
    top, left = quadrant
    row_lead = 4 if top == 0 else 1
    row_tail = 1 if top == 0 else 4
    col_lead = 4 if left == 0 else 1
    col_tail = 1 if left == 0 else 4
    return r0 + row_lead, r1 - row_tail, c0 + col_lead, c1 - col_tail


def _paint_icon(grid: np.ndarray, quadrant: tuple[int, int], op: str) -> None:
    mask = ICON_MASKS[op]
    h, w = grid.shape
    if quadrant == (0, 0):
        patch = grid[:3, :3]
    elif quadrant == (0, 1):
        patch = grid[:3, w - 3 :]
    elif quadrant == (1, 0):
        patch = grid[h - 3 :, :3]
    else:
        patch = grid[h - 3 :, w - 3 :]
    patch[mask] = AZURE


def _stamp_shape(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    patch = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
    patch[mask] = color


def _place_quadrant_objects(
    grid: np.ndarray,
    expected: np.ndarray,
    quadrant: tuple[int, int],
    op: str,
    color: int,
    mid_r: int,
    mid_c: int,
) -> bool:
    pr0, pr1, pc0, pc1 = _placement_window(grid.shape, mid_r, mid_c, quadrant)
    avail_h = pr1 - pr0
    avail_w = pc1 - pc0
    if avail_h < 2 or avail_w < 2:
        return False

    object_count = _choose_object_count(avail_h, avail_w)
    bboxes: list[tuple[int, int, int, int]] = []

    for _ in range(object_count):
        placed = False
        for _ in range(OBJECT_PLACEMENT_ATTEMPTS):
            mask = _random_shape(avail_h, avail_w)
            if mask is None:
                return False
            h, w = mask.shape
            if h > avail_h or w > avail_w:
                continue
            r0 = random.randint(pr0, pr1 - h)
            c0 = random.randint(pc0, pc1 - w)
            bbox = (r0, r0 + h - 1, c0, c0 + w - 1)
            if _overlaps_with_gap(bbox, bboxes):
                continue

            _stamp_shape(grid, mask, r0, c0, color)
            transformed = _transform_mask(mask, op)
            _stamp_shape(expected, transformed, r0, c0, color)
            bboxes.append(bbox)
            placed = True
            break

        if not placed:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    quad_h = random.randint(9, 13)
    quad_w = random.randint(9, 13)
    h = 2 * quad_h + 1
    w = 2 * quad_w + 1
    mid_r = quad_h
    mid_c = quad_w

    grid = np.full((h, w), BG, dtype=int)
    grid[mid_r, :] = GREY
    grid[:, mid_c] = GREY
    expected = grid.copy()

    ops = dict(zip(QUADRANTS, random.sample(list(ICON_MASKS.keys()), len(QUADRANTS))))
    colors = random.sample(PALETTE, len(QUADRANTS))

    for quadrant, op in ops.items():
        _paint_icon(grid, quadrant, op)
        _paint_icon(expected, quadrant, op)

    for quadrant, color in zip(QUADRANTS, colors):
        if not _place_quadrant_objects(grid, expected, quadrant, ops[quadrant], color, mid_r, mid_c):
            return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
