from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

SHIFT = 2
DIR_BY_SIDE = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}
ALL_SIDES = tuple(DIR_BY_SIDE)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GRID_MIN = 11
GRID_MAX = 16
SHAPE_MIN_AREA = 6
SHAPE_MAX_AREA = 15
SHAPE_MIN_MARGIN = 3
GEN_ATTEMPTS = 500
LOCAL_BBOX_MIN = 3
LOCAL_BBOX_MAX = 6


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        cells = []
        while stack:
            rr, cc = stack.pop()
            cells.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(cells, dtype=int))
    return comps


def _translate_mask(mask: np.ndarray, dr: int, dc: int) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)

    if dr >= 0:
        src_r = slice(0, h - dr)
        dst_r = slice(dr, h)
    else:
        src_r = slice(-dr, h)
        dst_r = slice(0, h + dr)

    if dc >= 0:
        src_c = slice(0, w - dc)
        dst_c = slice(dc, w)
    else:
        src_c = slice(-dc, w)
        dst_c = slice(0, w + dc)

    out[dst_r, dst_c] = mask[src_r, src_c]
    return out


def _overlap_mask(mask: np.ndarray, dr: int, dc: int) -> np.ndarray:
    return mask & _translate_mask(mask, dr, dc)


def _decode_marker_direction(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    markers = np.argwhere(grid == RED)
    assert len(markers) == 1, "expected exactly one red marker"
    r, c = map(int, markers[0])
    h, w = grid.shape

    on_top = r == 0
    on_bottom = r == h - 1
    on_left = c == 0
    on_right = c == w - 1
    assert on_top or on_bottom or on_left or on_right, "marker must lie on the border"
    assert sum((on_top, on_bottom, on_left, on_right)) == 1, "marker must not be on a corner"

    if on_top:
        return (r, c), DIR_BY_SIDE["top"]
    if on_bottom:
        return (r, c), DIR_BY_SIDE["bottom"]
    if on_left:
        return (r, c), DIR_BY_SIDE["left"]
    return (r, c), DIR_BY_SIDE["right"]


def _shape_mask(grid: np.ndarray) -> np.ndarray:
    mask = grid == GREEN
    comps = _components(mask)
    assert len(comps) == 1, "expected one green 4-connected shape"
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal the shape core by intersecting the green shape with its inward 2-cell translation.

    Input structure:
    - The grid is a black background with one irregular 4-connected green shape fully inside the border.
    - Exactly one red marker pixel lies on a non-corner border cell.
    - The marker's border side is the only cue for the translation direction.

    Rule:
    - Read the red marker side: top means shift the shape down 2, bottom means up 2, left means right 2, and right means left 2.
    - Form a virtual copy of the green shape translated by that fixed 2-cell inward offset.
    - Output a clean black grid of the same size, coloring blue exactly where the original shape and the translated copy overlap.

    Key insight:
    - The red border marker does not name a color operation; it encodes a geometric parameter, and the answer is the pixelwise intersection of one shape with its own translated copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    allowed = {BLACK, RED, GREEN}
    assert set(map(int, np.unique(g))).issubset(allowed), "unexpected colors present"

    _marker, (dr, dc) = _decode_marker_direction(g)
    shape = _shape_mask(g)
    overlap = _overlap_mask(shape, dr * SHIFT, dc * SHIFT)

    out = np.zeros_like(g, dtype=int)
    out[overlap] = BLUE
    return out


def _grow_connected_cells(height: int, width: int, area: int) -> set[tuple[int, int]] | None:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    return cells


def _cells_to_mask(cells: set[tuple[int, int]], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _is_irregular(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    bbox_h = int(r1 - r0 + 1)
    bbox_w = int(c1 - c0 + 1)
    area = int(mask.sum())

    if bbox_h < 3 and bbox_w < 3:
        return False
    if bbox_h * bbox_w == area:
        return False

    row_counts = np.sum(mask, axis=1)
    col_counts = np.sum(mask, axis=0)
    return len(set(map(int, row_counts[row_counts > 0]))) > 1 or len(set(map(int, col_counts[col_counts > 0]))) > 1


def _make_candidate_shape(height: int, width: int, area: int) -> np.ndarray | None:
    bbox_h = random.randint(LOCAL_BBOX_MIN, min(LOCAL_BBOX_MAX, height - 2 * SHAPE_MIN_MARGIN))
    bbox_w = random.randint(LOCAL_BBOX_MIN, min(LOCAL_BBOX_MAX, width - 2 * SHAPE_MIN_MARGIN))
    max_area = min(SHAPE_MAX_AREA, area, bbox_h * bbox_w - 1)
    if max_area < SHAPE_MIN_AREA:
        return None

    local_area = random.randint(SHAPE_MIN_AREA, max_area)
    cells = _grow_connected_cells(bbox_h, bbox_w, local_area)
    if cells is None:
        return None

    local_mask = _cells_to_mask(cells, bbox_h, bbox_w)
    if not _is_irregular(local_mask):
        return None

    top = random.randint(SHAPE_MIN_MARGIN, height - SHAPE_MIN_MARGIN - bbox_h)
    left = random.randint(SHAPE_MIN_MARGIN, width - SHAPE_MIN_MARGIN - bbox_w)
    full = np.zeros((height, width), dtype=bool)
    full[top : top + bbox_h, left : left + bbox_w] = local_mask
    return full


def _marker_position(side: str, height: int, width: int) -> tuple[int, int]:
    if side == "top":
        return 0, random.randint(1, width - 2)
    if side == "bottom":
        return height - 1, random.randint(1, width - 2)
    if side == "left":
        return random.randint(1, height - 2), 0
    return random.randint(1, height - 2), width - 1


def _is_marker_clear(marker: tuple[int, int], shape: np.ndarray) -> bool:
    h, w = shape.shape
    mr, mc = marker
    for nr, nc in _neighbors4(mr, mc):
        if 0 <= nr < h and 0 <= nc < w and shape[nr, nc]:
            return False
    return not shape[mr, mc]


def _semantic_checks(shape: np.ndarray, side: str, require_alt: bool) -> bool:
    outputs = {
        side_name: _overlap_mask(shape, dr * SHIFT, dc * SHIFT)
        for side_name, (dr, dc) in DIR_BY_SIDE.items()
    }
    target = outputs[side]
    target_area = int(target.sum())
    if target_area == 0:
        return False
    if target_area >= int(shape.sum()):
        return False
    if any(np.array_equal(target, other) for other_side, other in outputs.items() if other_side != side):
        return False

    alt_nonempty = [
        other
        for other_side, other in outputs.items()
        if other_side != side and int(other.sum()) > 0
    ]
    if require_alt and not alt_nonempty:
        return False
    if require_alt and not any(not np.array_equal(target, other) for other in alt_nonempty):
        return False
    return True


def _build_grid(shape: np.ndarray, side: str) -> np.ndarray | None:
    h, w = shape.shape
    marker = _marker_position(side, h, w)
    if not _is_marker_clear(marker, shape):
        return None
    grid = np.zeros((h, w), dtype=int)
    grid[shape] = GREEN
    grid[marker] = RED
    return grid


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    side = random.choice(ALL_SIDES)
    require_alt = random.random() < 0.6

    shape = _make_candidate_shape(height, width, SHAPE_MAX_AREA)
    if shape is None:
        return None
    if not _semantic_checks(shape, side, require_alt):
        return None

    grid = _build_grid(shape, side)
    if grid is None:
        return None

    pts = np.argwhere(shape)
    if pts[:, 0].min() < SHAPE_MIN_MARGIN or pts[:, 0].max() > height - SHAPE_MIN_MARGIN - 1:
        return None
    if pts[:, 1].min() < SHAPE_MIN_MARGIN or pts[:, 1].max() > width - SHAPE_MIN_MARGIN - 1:
        return None
    if len(_components(shape)) != 1:
        return None

    expected = np.zeros_like(grid)
    dr, dc = DIR_BY_SIDE[side]
    expected[_overlap_mask(shape, dr * SHIFT, dc * SHIFT)] = BLUE
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
