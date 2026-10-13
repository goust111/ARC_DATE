from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import Color, bbox, components_mask, crop_bbox, in_bounds, neighbors4


BG = Color.BLACK
TILE_COLOR = Color.TEAL
CONTOUR_COLOR = Color.GREY
LEGEND_POOL = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)

GRID_MIN = 17
GRID_MAX = 20
OUTER_MIN = 9
OUTER_MAX = 11
MARGIN = 3
GEN_ATTEMPTS = 400

TILE_PATTERNS = (
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
    np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
)


def _legend_mask(grid: np.ndarray) -> np.ndarray:
    return (grid != BG) & (grid != TILE_COLOR) & (grid != CONTOUR_COLOR)


def _legend_colors(grid: np.ndarray) -> list[int]:
    coords = np.argwhere(_legend_mask(grid))
    assert 3 <= len(coords) <= 4, "expected a 3-4 cell color legend"
    coords = sorted((int(r), int(c)) for r, c in coords)
    cols = {c for _, c in coords}
    assert len(cols) == 1, "legend must be vertical"
    colors = [int(grid[r, c]) for r, c in coords]
    assert len(set(colors)) == len(colors), "legend colors must be distinct"
    return colors


def _tile_mask_and_anchor(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int], tuple[int, int, int, int]]:
    tile_box = bbox(grid == TILE_COLOR)
    assert tile_box is not None, "missing teal control tile"
    tile = crop_bbox(grid, tile_box)
    assert set(map(int, np.unique(tile))) <= {BG, TILE_COLOR}
    return tile == TILE_COLOR, (tile_box[0], tile_box[1]), tile_box


def _flood_from_border(passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    seen = np.zeros_like(passable, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            if in_bounds(passable.shape, nr, nc) and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return seen


def _interior_mask(grid: np.ndarray) -> np.ndarray:
    passable = grid == BG
    exterior = _flood_from_border(passable)
    return passable & (~exterior)


def _depth_from_boundary(interior: np.ndarray) -> np.ndarray:
    depth = np.zeros(interior.shape, dtype=int)
    queue: deque[tuple[int, int]] = deque()

    for r, c in np.argwhere(interior):
        rr, cc = int(r), int(c)
        if any((not in_bounds(interior.shape, nr, nc)) or (not interior[nr, nc]) for nr, nc in neighbors4(rr, cc)):
            depth[rr, cc] = 1
            queue.append((rr, cc))

    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            if in_bounds(interior.shape, nr, nc) and interior[nr, nc] and depth[nr, nc] == 0:
                depth[nr, nc] = depth[r, c] + 1
                queue.append((nr, nc))

    return depth


def _tiled_foreground(shape: tuple[int, int], tile_fg: np.ndarray, anchor: tuple[int, int]) -> np.ndarray:
    h, w = shape
    th, tw = tile_fg.shape
    row_ids = (np.arange(h) - int(anchor[0])) % th
    col_ids = (np.arange(w) - int(anchor[1])) % tw
    return tile_fg[row_ids[:, None], col_ids[None, :]]


def _shape_boundary(shape_mask: np.ndarray) -> np.ndarray:
    h, w = shape_mask.shape
    boundary = np.zeros_like(shape_mask, dtype=bool)
    for r, c in np.argwhere(shape_mask):
        rr, cc = int(r), int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = rr + dr, cc + dc
                if not in_bounds(shape_mask.shape, nr, nc) or not shape_mask[nr, nc]:
                    boundary[rr, cc] = True
                    dr = dc = 2
                    break
            if boundary[rr, cc]:
                break
    return boundary


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the teal control motif across the grid, crop it by the grey mask, and recolor visible hits by nesting depth.

    Input structure:
    - Black is the background.
    - A small teal-on-black control tile sits in the upper-left area.
    - A one-cell-wide grey contour encloses an irregular black interior region.
    - A vertical legend of 3 or 4 colored cells sits near the lower-right corner.

    Rule:
    - Extract the minimal bounding box of the teal control tile, preserving both teal and black cells inside it.
    - Repeat that tile over the full grid using its original top-left alignment.
    - Find the black cells strictly inside the grey contour by 4-connected flood fill from the border.
    - For each interior cell, compute its 4-connected distance from the nearest non-interior cell; this depth starts at 1 just inside the contour.
    - Recolor only interior cells whose tiled copy lands on a teal tile pixel, using the legend color at index `(depth - 1) mod len(legend)`.
    - Keep tiled black cells black, preserve the grey contour, and leave the control tile and legend unchanged.

    Key insight:
    - The final colors depend on three separate structures at once: the repeated tile says where paint exists, the contour says where it is visible, and the inward depth says which legend color each visible tile pixel receives.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    tile_fg, anchor, tile_box = _tile_mask_and_anchor(g)
    legend = np.array(_legend_colors(g), dtype=int)
    interior = _interior_mask(g)
    depth = _depth_from_boundary(interior)
    tiled_fg = _tiled_foreground(g.shape, tile_fg, anchor)

    out = np.full_like(g, BG)
    out[g == CONTOUR_COLOR] = CONTOUR_COLOR

    visible = interior & tiled_fg
    if np.any(visible):
        out[visible] = legend[(depth[visible] - 1) % len(legend)]

    r0, c0, r1, c1 = tile_box
    out[r0 : r1 + 1, c0 : c1 + 1] = g[r0 : r1 + 1, c0 : c1 + 1]
    legend_mask = _legend_mask(g)
    out[legend_mask] = g[legend_mask]
    return out


def _carve_notches(local: np.ndarray) -> np.ndarray:
    h, w = local.shape
    carved = local.copy()
    sides = ["top", "bottom", "left", "right"]
    notch_count = random.randint(2, 4)
    used: set[tuple[str, int, int]] = set()

    for _ in range(20):
        if len(used) >= notch_count:
            break
        side = random.choice(sides)
        if side in {"top", "bottom"}:
            width = random.randint(1, min(3, w - 2))
            start = random.randint(1, w - width - 1)
            signature = (side, start, width)
            if signature in used:
                continue
            used.add(signature)
            rows = slice(0, 1) if side == "top" else slice(h - 1, h)
            carved[rows, start : start + width] = False
        else:
            height = random.randint(1, min(3, h - 2))
            start = random.randint(1, h - height - 1)
            signature = (side, start, height)
            if signature in used:
                continue
            used.add(signature)
            cols = slice(0, 1) if side == "left" else slice(w - 1, w)
            carved[start : start + height, cols] = False

    return carved


def _semantic_instance_ok(grid: np.ndarray) -> bool:
    tile_fg, anchor, tile_box = _tile_mask_and_anchor(grid)
    legend = np.array(_legend_colors(grid), dtype=int)
    legend_mask = _legend_mask(grid)

    contour = grid == CONTOUR_COLOR
    if len(components_mask(contour, connectivity=8)) != 1:
        return False

    interior = _interior_mask(grid)
    if np.count_nonzero(interior) < 35:
        return False
    if len(components_mask(interior, connectivity=4)) != 1:
        return False

    depth = _depth_from_boundary(interior)
    max_depth = int(depth.max(initial=0))
    if max_depth < len(legend):
        return False

    tiled_fg = _tiled_foreground(grid.shape, tile_fg, anchor)
    visible_fg = interior & tiled_fg
    visible_bg = interior & (~tiled_fg)
    if not np.any(visible_fg) or not np.any(visible_bg):
        return False

    used_depths = set(map(int, depth[visible_fg]))
    if len(used_depths) < 2:
        return False
    used_colors = set(map(int, legend[(depth[visible_fg] - 1) % len(legend)]))
    if used_colors != set(map(int, legend)):
        return False

    special = np.zeros_like(grid, dtype=bool)
    r0, c0, r1, c1 = tile_box
    special[r0 : r1 + 1, c0 : c1 + 1] = True
    special[legend_mask] = True
    special[contour] = True
    if not np.any((~interior) & tiled_fg & (~special)):
        return False

    return not np.array_equal(main(grid), grid)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid = np.zeros((h, w), dtype=int)

    tile_fg = random.choice(TILE_PATTERNS)
    grid[:3, :3] = np.where(tile_fg, TILE_COLOR, BG)

    legend_len = random.choice((3, 4))
    legend = random.sample(list(LEGEND_POOL), legend_len)
    grid[h - legend_len :, w - 1] = legend

    outer_h_choices = [size for size in range(OUTER_MIN, OUTER_MAX + 1) if size <= h - 2 * MARGIN]
    outer_w_choices = [size for size in range(OUTER_MIN, OUTER_MAX + 1) if size <= w - 2 * MARGIN]
    if not outer_h_choices or not outer_w_choices:
        return None

    outer_h = random.choice(outer_h_choices)
    outer_w = random.choice(outer_w_choices)
    top = random.randint(MARGIN, h - outer_h - MARGIN)
    left = random.randint(MARGIN, w - outer_w - MARGIN)

    local = np.ones((outer_h, outer_w), dtype=bool)
    local = _carve_notches(local)
    if np.count_nonzero(local) < outer_h * outer_w - 1:
        pass

    shape_mask = np.zeros((h, w), dtype=bool)
    shape_mask[top : top + outer_h, left : left + outer_w] = local
    boundary = _shape_boundary(shape_mask)
    interior = shape_mask & (~boundary)

    if np.count_nonzero(interior) < 35:
        return None
    if len(components_mask(shape_mask, connectivity=4)) != 1:
        return None
    if len(components_mask(boundary, connectivity=8)) != 1:
        return None
    if len(components_mask(interior, connectivity=4)) != 1:
        return None

    grid[boundary] = CONTOUR_COLOR

    if not _semantic_instance_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mask-tiling ARC input")
