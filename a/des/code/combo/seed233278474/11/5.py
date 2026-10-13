from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
ORANGE = 7

CONTOUR_COLORS = (GREEN, BLUE, RED)
LAYER_COLORS = (GREEN, BLUE, RED)
GEN_MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _component_masks(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for rr, cc in np.argwhere(mask):
        r, c = int(rr), int(cc)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        cells: list[tuple[int, int]] = []
        while q:
            cr, cc2 = q.popleft()
            cells.append((cr, cc2))
            for nr, nc in _neighbors4(cr, cc2, h, w):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comp = np.zeros_like(mask, dtype=bool)
        for cr, cc2 in cells:
            comp[cr, cc2] = True
        comps.append(comp)
    return comps


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    assert pts.size > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _flood_outside(blocked: np.ndarray) -> np.ndarray:
    blocked = np.asarray(blocked, dtype=bool)
    h, w = blocked.shape
    outside = np.zeros_like(blocked, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not blocked[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not blocked[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if blocked[nr, nc] or outside[nr, nc]:
                continue
            outside[nr, nc] = True
            q.append((nr, nc))

    return outside


def _enclosed_region(contour: np.ndarray) -> np.ndarray:
    contour = np.asarray(contour, dtype=bool)
    outside = _flood_outside(contour)
    return (~outside) & (~contour)


def _solid_boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask, 1, constant_values=False)
    interior = (
        mask
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return mask & (~interior)


def _simple_cycle(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    if len(_component_masks(mask)) != 1:
        return False
    h, w = mask.shape
    for rr, cc in pts:
        r, c = int(rr), int(cc)
        degree = sum(1 for nr, nc in _neighbors4(r, c, h, w) if mask[nr, nc])
        if degree != 2:
            return False
    return True


def _find_marker_center(grid: np.ndarray) -> tuple[int, int]:
    ys = np.argwhere(grid == YELLOW)
    assert len(ys) == 5, "expected exactly one yellow cross"
    h, w = grid.shape
    centers: list[tuple[int, int]] = []
    for rr, cc in ys:
        r, c = int(rr), int(cc)
        if all(0 <= nr < h and 0 <= nc < w and grid[nr, nc] == YELLOW for nr, nc in _neighbors4(r, c, h, w)):
            centers.append((r, c))
    assert len(centers) == 1, "expected a unique cross center"
    return centers[0]


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.all((g >= 0) & (g <= 9))

    anchor_r, anchor_c = _find_marker_center(g)
    contour_mask = np.isin(g, CONTOUR_COLORS)
    components = _component_masks(contour_mask)
    assert len(components) in (2, 3), "expected two or three nested contours"
    for comp in components:
        assert _simple_cycle(comp), "each contour must be a simple closed loop"

    enclosed = [_enclosed_region(comp) for comp in components]
    levels: list[int] = []
    for i, comp in enumerate(components):
        rr, cc = np.argwhere(comp)[0]
        r, c = int(rr), int(cc)
        level = 1 + sum(1 for j, region in enumerate(enclosed) if i != j and region[r, c])
        levels.append(level)

    assert sorted(levels) == list(range(1, len(components) + 1)), "contours must form a strict nesting chain"

    bg_cells = g == BG
    depth = np.zeros_like(g, dtype=int)
    for region in enclosed:
        depth += region.astype(int)
    assert depth[bg_cells].max() == len(components), "nesting depth mismatch"

    out = g.copy()
    odd_fill = bg_cells & (depth > 0) & ((depth % 2) == 1)
    even_fill = bg_cells & ((depth % 2) == 0) & (depth > 0)
    out[odd_fill] = GRAY
    out[even_fill] = ORANGE

    deepest = bg_cells & (depth == len(components))
    assert np.any(deepest), "innermost region must be non-empty"
    stamp = _crop_mask(deepest)

    out[g == YELLOW] = BG
    sh, sw = stamp.shape
    assert anchor_r + sh <= g.shape[0] and anchor_c + sw <= g.shape[1], "stamp placement must fit"
    target = out[anchor_r : anchor_r + sh, anchor_c : anchor_c + sw]
    target[stamp] = MAGENTA
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid)


def _dilate(mask: np.ndarray, steps: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    for _ in range(steps):
        padded = np.pad(out, 1, constant_values=False)
        out = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
        )
    return out


def _region_interior(mask: np.ndarray) -> np.ndarray:
    boundary = _solid_boundary(mask)
    return mask & (~boundary)


def _valid_region(mask: np.ndarray) -> bool:
    if not np.any(mask):
        return False
    boundary = _solid_boundary(mask)
    interior = mask & (~boundary)
    if not np.any(interior):
        return False
    if len(_component_masks(mask)) != 1:
        return False
    return _simple_cycle(boundary)


def _make_inner_region() -> np.ndarray:
    h = random.randint(5, 9)
    w = random.randint(5, 9)
    mask = np.ones((h, w), dtype=bool)
    assert _valid_region(mask)
    return mask


def _nested_regions(depth: int) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    inner = _make_inner_region()
    regions = [inner]
    offsets = [(0, 0)]
    for _ in range(depth - 1):
        prev = regions[0]
        top_margin = random.randint(2, 4)
        bottom_margin = random.randint(2, 4)
        left_margin = random.randint(2, 4)
        right_margin = random.randint(2, 4)
        outer = np.ones(
            (prev.shape[0] + top_margin + bottom_margin, prev.shape[1] + left_margin + right_margin),
            dtype=bool,
        )
        regions = [outer] + regions
        offsets = [(0, 0)] + [(r + top_margin, c + left_margin) for r, c in offsets]
    return regions, offsets


def _anchor_for_corner(corner: str, grid_h: int, grid_w: int, stamp_h: int, stamp_w: int) -> tuple[int, int]:
    if corner == "tl":
        return 1, 1
    if corner == "tr":
        return 1, grid_w - stamp_w - 1
    if corner == "bl":
        return grid_h - stamp_h - 1, 1
    if corner == "br":
        return grid_h - stamp_h - 1, grid_w - stamp_w - 1
    raise ValueError(corner)


def _placement_ranges(
    corner: str,
    grid_h: int,
    grid_w: int,
    outer_h: int,
    outer_w: int,
    stamp_h: int,
    stamp_w: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if corner == "tl":
        return (stamp_h + 2, grid_h - outer_h - 1), (stamp_w + 2, grid_w - outer_w - 1)
    if corner == "tr":
        return (stamp_h + 2, grid_h - outer_h - 1), (1, grid_w - stamp_w - outer_w - 3)
    if corner == "bl":
        return (1, grid_h - stamp_h - outer_h - 3), (stamp_w + 2, grid_w - outer_w - 1)
    if corner == "br":
        return (1, grid_h - stamp_h - outer_h - 3), (1, grid_w - stamp_w - outer_w - 3)
    raise ValueError(corner)


def _draw_cross(grid: np.ndarray, r: int, c: int) -> None:
    grid[r, c] = YELLOW
    grid[r - 1, c] = YELLOW
    grid[r + 1, c] = YELLOW
    grid[r, c - 1] = YELLOW
    grid[r, c + 1] = YELLOW


def _gen_attempt() -> np.ndarray | None:
    depth = random.choice([2, 3])
    outer_to_inner, offsets = _nested_regions(depth)
    outer = outer_to_inner[0]
    inner = outer_to_inner[-1]
    stamp = _region_interior(inner)
    if not np.any(stamp):
        return None
    stamp = _crop_mask(stamp)
    stamp_h, stamp_w = stamp.shape
    outer_h, outer_w = outer.shape

    grid_h_min = max(18, outer_h + stamp_h + 6)
    grid_w_min = max(18, outer_w + stamp_w + 6)
    grid_h = random.randint(grid_h_min, grid_h_min + 8)
    grid_w = random.randint(grid_w_min, grid_w_min + 8)

    corner = random.choice(["tl", "tr", "bl", "br"])
    anchor_r, anchor_c = _anchor_for_corner(corner, grid_h, grid_w, stamp_h, stamp_w)
    row_range, col_range = _placement_ranges(corner, grid_h, grid_w, outer_h, outer_w, stamp_h, stamp_w)
    if row_range[0] > row_range[1] or col_range[0] > col_range[1]:
        return None

    top = random.randint(*row_range)
    left = random.randint(*col_range)
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    for color, region, (dr, dc) in zip(LAYER_COLORS[:depth], outer_to_inner, offsets):
        boundary = _solid_boundary(region)
        rr = top + dr
        cc = left + dc
        patch = grid[rr : rr + region.shape[0], cc : cc + region.shape[1]]
        if np.any((patch != BG) & boundary):
            return None
        patch[boundary] = color

    cross_rows = [anchor_r - 1, anchor_r, anchor_r + 1]
    cross_cols = [anchor_c - 1, anchor_c, anchor_c + 1]
    if min(cross_rows) < 0 or min(cross_cols) < 0 or max(cross_rows) >= grid_h or max(cross_cols) >= grid_w:
        return None
    if np.any(grid[np.ix_(cross_rows, [anchor_c])] != BG):
        return None
    if np.any(grid[np.ix_([anchor_r], cross_cols)] != BG):
        return None
    _draw_cross(grid, anchor_r, anchor_c)

    try:
        out = _solve(grid)
    except AssertionError:
        return None

    if np.any(out == YELLOW):
        return None
    if not np.any(out == MAGENTA):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
