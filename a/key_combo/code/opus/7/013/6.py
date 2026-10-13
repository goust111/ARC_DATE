from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_h, bbox_w, colors_present, coords_where, crop_bbox, get_bg


BG = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
MARKER = Color.ORANGE

GRID_MIN = 18
GRID_MAX = 26
BOX_H_RANGE = (5, 11)
BOX_W_RANGE = (5, 11)
BOX_DIFF_MIN = 2
INNER_FILL_FRAC = (0.22, 0.48)
GEN_RETRIES = 400


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _marker_coords(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    coords = coords_where(grid == MARKER)
    assert coords.shape == (2, 2), "expected exactly two orange marker pixels"
    return tuple(map(int, coords[0])), tuple(map(int, coords[1]))


def _parallelogram_bbox_from_markers(
    p: tuple[int, int], q: tuple[int, int]
) -> tuple[int, int, int, int]:
    dr = int(q[0] - p[0])
    dc = int(q[1] - p[1])
    abs_dr = abs(dr)
    abs_dc = abs(dc)
    assert abs_dr > 0 and abs_dc > 0, "marker offset must have both row and column components"
    assert abs_dr != abs_dc, "marker offset magnitudes must be unequal"

    diag_len = min(abs_dr, abs_dc)
    straight_len = abs(abs_dr - abs_dc)
    sr = _sign(dr)
    sc = _sign(dc)

    if abs_dr > abs_dc:
        straight_vec = (sr * straight_len, 0)
    else:
        straight_vec = (0, sc * straight_len)
    diag_vec = (sr * diag_len, sc * diag_len)

    vertices = np.array(
        [
            p,
            q,
            (p[0] + straight_vec[0], p[1] + straight_vec[1]),
            (p[0] + diag_vec[0], p[1] + diag_vec[1]),
        ],
        dtype=int,
    )
    r0, c0 = vertices.min(axis=0)
    r1, c1 = vertices.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mirror_by_aspect(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape
    if w >= h:
        return np.fliplr(crop).copy()
    return np.flipud(crop).copy()


def _expected_output(grid: np.ndarray) -> np.ndarray:
    p, q = _marker_coords(grid)
    box = _parallelogram_bbox_from_markers(p, q)
    return _mirror_by_aspect(crop_bbox(grid, box))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the bounding box induced by the marker-defined parallelogram, then mirror it along the axis chosen by that box's aspect ratio.

    Input structure:
    - The grid has a black background, exactly two orange marker pixels, and blue/red motif pixels inside the marker-defined region.
    - The marker-to-marker offset has non-zero row and column parts with unequal magnitudes.
    - Outside the cropped region there are no non-background pixels besides the markers.

    Rule:
    - Decompose the marker offset into a 45-degree diagonal vector plus the remaining horizontal or vertical vector.
    - Use those two vectors as adjacent sides of a parallelogram whose opposite corners are the two markers.
    - Take the axis-aligned bounding box of the four parallelogram vertices, crop the input to that box, then mirror the crop left-right when it is wider than tall, otherwise mirror it top-bottom; square crops also use left-right mirroring.

    Key insight:
    - The orange pixels are geometric controls: they first define a parallelogram, that geometry determines the crop, and the crop's aspect ratio determines the mirror axis.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    assert int(bg) == BG, "expected black background"
    assert set(colors_present(g, exclude={BG})) <= {BLUE, RED, MARKER}, "unexpected foreground colors"
    return _expected_output(g)


def _grow_mask(h: int, w: int, target: int) -> np.ndarray | None:
    cells = [(r, c) for r in range(h) for c in range(w)]
    seed = random.choice(cells)
    mask = np.zeros((h, w), dtype=bool)
    mask[seed] = True
    frontier = {seed}

    while int(mask.sum()) < target:
        candidates: set[tuple[int, int]] = set()
        for r, c in frontier:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                    candidates.add((rr, cc))
        if not candidates:
            return None
        pick = random.choice(sorted(candidates))
        mask[pick] = True
        frontier.add(pick)
    return mask


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    assert len(coords) >= 2
    random.shuffle(coords)
    split = random.randint(1, len(coords) - 1)
    first = coords[:split]
    second = coords[split:]
    for r, c in first:
        patch[r, c] = BLUE
    for r, c in second:
        patch[r, c] = RED
    return patch


def _random_inner_patch(inner_h: int, inner_w: int) -> np.ndarray | None:
    area = inner_h * inner_w
    lo = max(4, int(np.ceil(area * INNER_FILL_FRAC[0])))
    hi = min(area - 1, int(np.floor(area * INNER_FILL_FRAC[1])))
    if lo > hi:
        return None

    for _ in range(80):
        target = random.randint(lo, hi)
        mask = _grow_mask(inner_h, inner_w, target)
        if mask is None:
            continue
        patch = _colorize_mask(mask)
        if np.array_equal(patch, np.fliplr(patch)):
            continue
        if np.array_equal(patch, np.flipud(patch)):
            continue
        if np.count_nonzero(patch == BLUE) == 0 or np.count_nonzero(patch == RED) == 0:
            continue
        return patch
    return None


def _build_grid(
    grid_h: int,
    grid_w: int,
    box_h: int,
    box_w: int,
    sr: int,
    sc: int,
    anchor_r: int,
    anchor_c: int,
    inner_patch: np.ndarray,
) -> np.ndarray:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    p = (anchor_r, anchor_c)
    q = (anchor_r + sr * (box_h - 1), anchor_c + sc * (box_w - 1))
    grid[p] = MARKER
    grid[q] = MARKER

    r0, r1 = sorted((p[0], q[0]))
    c0, c1 = sorted((p[1], q[1]))
    grid[r0 + 1 : r1, c0 + 1 : c1] = inner_patch
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False

    colors = set(map(int, np.unique(grid)))
    if not colors <= {BG, BLUE, RED, MARKER}:
        return False
    if np.count_nonzero(grid == MARKER) != 2:
        return False
    if np.count_nonzero(grid == BLUE) == 0 or np.count_nonzero(grid == RED) == 0:
        return False

    p, q = _marker_coords(grid)
    box = _parallelogram_bbox_from_markers(p, q)
    r0, c0, r1, c1 = box
    box_h = bbox_h(box)
    box_w = bbox_w(box)
    if abs(box_h - box_w) < BOX_DIFF_MIN:
        return False

    outside = grid.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = BG
    if np.count_nonzero(outside) != 0:
        return False

    crop = crop_bbox(grid, box)
    inner = crop[1:-1, 1:-1]
    if inner.size == 0:
        return False
    if np.any(crop[1:-1, 0] != BG) or np.any(crop[1:-1, -1] != BG):
        return False
    if np.any(crop[0, 1:-1] != BG) or np.any(crop[-1, 1:-1] != BG):
        return False
    if np.count_nonzero(inner) == 0:
        return False

    expected = _expected_output(grid)
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    if box_w > box_h and not np.array_equal(actual, np.fliplr(crop)):
        return False
    if box_h > box_w and not np.array_equal(actual, np.flipud(crop)):
        return False

    if np.array_equal(actual, crop):
        return False
    return True


def _sample_box_dims() -> tuple[int, int]:
    branch = random.choice(("wide", "tall"))
    if branch == "wide":
        box_h = random.randint(*BOX_H_RANGE)
        lo = max(BOX_W_RANGE[0], box_h + BOX_DIFF_MIN)
        hi = BOX_W_RANGE[1]
        if lo > hi:
            return _sample_box_dims()
        box_w = random.randint(lo, hi)
        return box_h, box_w

    box_w = random.randint(*BOX_W_RANGE)
    lo = max(BOX_H_RANGE[0], box_w + BOX_DIFF_MIN)
    hi = BOX_H_RANGE[1]
    if lo > hi:
        return _sample_box_dims()
    box_h = random.randint(lo, hi)
    return box_h, box_w


def _gen_attempt() -> np.ndarray | None:
    box_h, box_w = _sample_box_dims()
    inner_patch = _random_inner_patch(box_h - 2, box_w - 2)
    if inner_patch is None:
        return None

    grid_h = random.randint(max(GRID_MIN, box_h + 4), GRID_MAX)
    grid_w = random.randint(max(GRID_MIN, box_w + 4), GRID_MAX)
    sr = random.choice((-1, 1))
    sc = random.choice((-1, 1))

    row_lo = box_h - 1 if sr < 0 else 0
    row_hi = grid_h - box_h if sr > 0 else grid_h - 1
    col_lo = box_w - 1 if sc < 0 else 0
    col_hi = grid_w - box_w if sc > 0 else grid_w - 1
    if row_lo > row_hi or col_lo > col_hi:
        return None

    anchor_r = random.randint(row_lo, row_hi)
    anchor_c = random.randint(col_lo, col_hi)
    grid = _build_grid(grid_h, grid_w, box_h, box_w, sr, sc, anchor_r, anchor_c, inner_patch)
    if _semantic_checks(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
