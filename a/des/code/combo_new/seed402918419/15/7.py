import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
ORANGE = 7

MARKER_COLORS = (GREEN, YELLOW, BLUE, ORANGE)
GEN_ATTEMPTS = 300
GRID_SIDES = tuple(range(11, 30, 2))


SHAPE_LIBRARY = (
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
)


def _apply_op(mask: np.ndarray, marker_color: int) -> np.ndarray:
    if marker_color == GREEN:
        return mask.copy()
    if marker_color == YELLOW:
        return np.fliplr(mask)
    if marker_color == BLUE:
        return np.flipud(mask)
    if marker_color == ORANGE:
        return np.rot90(mask, 2)
    raise AssertionError(f"unknown marker color: {marker_color}")


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_cross(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    row_candidates = [r for r in range(h) if np.all(grid[r, 1 : w - 1] == RED)]
    col_candidates = [c for c in range(w) if np.all(grid[1 : h - 1, c] == RED)]
    assert len(row_candidates) == 1
    assert len(col_candidates) == 1
    cr = row_candidates[0]
    cc = col_candidates[0]
    assert 0 < cr < h - 1 and 0 < cc < w - 1
    return cr, cc


def _quadrant_specs(cr: int, cc: int, h: int, w: int) -> list[tuple[slice, slice, tuple[int, int]]]:
    return [
        (slice(0, cr), slice(0, cc), (0, 0)),
        (slice(0, cr), slice(cc + 1, w), (0, 1)),
        (slice(cr + 1, h), slice(0, cc), (1, 0)),
        (slice(cr + 1, h), slice(cc + 1, w), (1, 1)),
    ]


def _extract_object(subgrid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    gray_mask = subgrid == GRAY
    parts = _components(gray_mask)
    assert len(parts) == 1
    coords = parts[0]
    r0, c0, r1, c1 = _bbox(coords)
    patch = gray_mask[r0 : r1 + 1, c0 : c1 + 1]
    return patch, (r0, c0, r1, c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each quadrant's gray object using the row-end and column-end marker operations of a central cross.

    Input structure:
    - The grid has odd height and odd width with black background.
    - The exact middle row and exact middle column form a red cross, except the four outer endpoints are replaced by one green, one yellow, one blue, and one orange marker.
    - The cross divides the board into four quadrants, and each quadrant contains exactly one 4-connected gray foreground object.

    Rule:
    - The top endpoint controls the two upper quadrants and the bottom endpoint controls the two lower quadrants.
    - The left endpoint controls the two left quadrants and the right endpoint controls the two right quadrants.
    - For each quadrant, first apply that quadrant's column operation to the gray object, then apply that quadrant's row operation; use the object's own bounding-box center as the in-place flip or 180-degree rotation anchor.
    - Keep the black background, the red cross, and the four endpoint markers unchanged.

    Key insight:
    - The four endpoint colors are a row/column lookup table: each quadrant's final shape comes from composing one row-controlled transformation with one column-controlled transformation on the local object only.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h % 2 == 1 and w % 2 == 1

    cr, cc = _find_cross(g)
    assert (cr, cc) == (h // 2, w // 2)
    assert g[cr, cc] == RED

    top = int(g[0, cc])
    bottom = int(g[h - 1, cc])
    left = int(g[cr, 0])
    right = int(g[cr, w - 1])
    markers = [top, bottom, left, right]
    assert set(markers) == set(MARKER_COLORS)
    assert len(set(markers)) == 4

    expected_cross = np.zeros_like(g, dtype=bool)
    expected_cross[cr, :] = True
    expected_cross[:, cc] = True
    endpoint_mask = np.zeros_like(g, dtype=bool)
    endpoint_mask[0, cc] = True
    endpoint_mask[h - 1, cc] = True
    endpoint_mask[cr, 0] = True
    endpoint_mask[cr, w - 1] = True

    assert np.all(g[expected_cross & ~endpoint_mask] == RED)
    non_cross = ~expected_cross
    assert np.all(np.isin(g[non_cross], [BLACK, GRAY]))

    row_ops = {0: top, 1: bottom}
    col_ops = {0: left, 1: right}
    out = g.copy()

    for row_slice, col_slice, (qr, qc) in _quadrant_specs(cr, cc, h, w):
        sub = g[row_slice, col_slice]
        patch, (r0, c0, r1, c1) = _extract_object(sub)
        transformed = _apply_op(_apply_op(patch, col_ops[qc]), row_ops[qr])
        target = out[row_slice, col_slice]
        box = target[r0 : r1 + 1, c0 : c1 + 1]
        box[:] = BLACK
        box[transformed] = GRAY

    return out


def _random_shape() -> np.ndarray:
    shape = random.choice(SHAPE_LIBRARY)
    op = random.choice(MARKER_COLORS)
    return _apply_op(shape, op)


def _place_shape(area: np.ndarray, shape: np.ndarray) -> None:
    ah, aw = area.shape
    sh, sw = shape.shape
    assert ah >= sh + 2 and aw >= sw + 2
    r0 = random.randint(1, ah - sh - 1)
    c0 = random.randint(1, aw - sw - 1)
    box = area[r0 : r0 + sh, c0 : c0 + sw]
    box[shape] = GRAY


def _iter_quadrants(grid: np.ndarray, cr: int, cc: int) -> Iterable[tuple[np.ndarray, tuple[int, int]]]:
    h, w = grid.shape
    for row_slice, col_slice, key in _quadrant_specs(cr, cc, h, w):
        yield grid[row_slice, col_slice], key


def _valid_sample(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except AssertionError:
        return False

    cr, cc = _find_cross(grid)
    if np.array_equal(out, grid):
        return False

    changed_quadrants = 0
    for row_slice, col_slice, _key in _quadrant_specs(cr, cc, *grid.shape):
        subgrid = grid[row_slice, col_slice]
        patch, (r0, c0, r1, c1) = _extract_object(subgrid)
        if patch.shape[0] > 3 or patch.shape[1] > 3:
            return False
        if np.array_equal(subgrid, out[row_slice, col_slice]):
            return False
        out_patch = out[row_slice, col_slice][r0 : r1 + 1, c0 : c1 + 1] == GRAY
        if not np.array_equal(out_patch, patch):
            changed_quadrants += 1
    return changed_quadrants == 4


def _gen_attempt() -> np.ndarray | None:
    h = random.choice(GRID_SIDES)
    w = random.choice(GRID_SIDES)
    cr, cc = h // 2, w // 2
    if cr < 5 or cc < 5:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    grid[cr, :] = RED
    grid[:, cc] = RED

    top, bottom, left, right = random.sample(MARKER_COLORS, 4)
    grid[0, cc] = top
    grid[h - 1, cc] = bottom
    grid[cr, 0] = left
    grid[cr, w - 1] = right

    row_ops = {0: top, 1: bottom}
    col_ops = {0: left, 1: right}

    for area, (qr, qc) in _iter_quadrants(grid, cr, cc):
        shape = _random_shape()
        final_shape = _apply_op(_apply_op(shape, col_ops[qc]), row_ops[qr])
        if np.array_equal(shape, final_shape):
            return None
        _place_shape(area, shape)

    if not _valid_sample(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
