# Variant of 8f3a5a89
# Dimension: D1 — 路径/区域发现机制 (path)
# Change: boundary_trace + flood_fill——从标记像素出发做4-连通洪填，找到被结构墙阻挡的可达区域，再在... → 直线射线扫描代替洪填

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


# -----------------------------------------------------------------------------
# Design analysis for the D1 variant
#
# a) Concrete behavior trace
#    - Input colors stay the same: bg=8, wall=1, marker=6, fill=7.
#    - The marker remains a unique corner pixel, and wall-colored border-touching
#      components are the structural blockers.
#    - The key change is the reachable-region mechanism:
#      original: a 4-connected flood-fill from the marker explores around bends.
#      variant: a one-bend ray fan. First sweep straight rays from the marker
#      along its row and column; then each cell on those seed rays emits one
#      orthogonal straight ray. The reachable region is the union of all swept
#      pixels. This keeps the task about wall-blocked line-of-sight rather than
#      connectivity.
#    - Small trace on a bottom-left marker:
#      original: if a top-hanging wall blocks column 6, flood-fill can still go
#      around the side and reach cells above that wall through neighboring cols.
#      variant: the bottom-row seed ray reaches column 6, but the upward ray at
#      column 6 stops under the wall; no later recursive turn is allowed, so the
#      hidden cap stays unreachable. That is the intended visible difference.
#
# b) Complexity calibration
#    - A literal single-origin 4-ray scan from a fixed bottom-left corner would
#      collapse to a thin L-shape and make the task visually trivial.
#    - The variant therefore uses a corner-anchored fan-out with exactly one
#      extra orthogonal sweep. This is still clearly "straight-ray scanning"
#      instead of flood-fill, but it yields 2D regions large enough to infer
#      from 2-4 ARC examples.
#    - The generator enforces structural blockers on the top/right borders so
#      the ray region is typically much smaller than the flood-fill region.
#
# c) Old assumptions that no longer hold
#    - The original `_flood(...)` assumed reachability was purely 4-connected.
#      That premise is false here; replacing only the final border-paint stage
#      would silently preserve old behavior, so reach computation must change.
#    - The original "structural if border-touching and adjacent to reach0"
#      relied on an initial flood-fill. In this variant that bootstrap is gone,
#      so structure is identified directly as border-touching wall components.
#    - Any code that treated "not reached" as "behind a closed enclosure" is now
#      weaker: a cell can be flood-reachable yet still ray-unreachable. The
#      generator intentionally creates such cells to make the new rule visible.
#
# d) Placement legality
#    - Structural walls are non-overlapping top/right anchored bars/rectangles.
#      They never cover the marker and always stay in-bounds.
#    - Visible decorations are placed only on ray-reachable non-fill cells, so
#      the solver keeps them blue and they do not erase the orange boundary.
#    - Hidden decorations are placed strictly off the border and outside the
#      ray region, so the solver erases them to bg.
#
# e) Visual observability
#    - Samples are rejected unless ray reach is substantially smaller than the
#      same-grid flood-fill reach. This forces the D1 change to show up in the
#      examples as "straight visibility from the corner" rather than generic
#      connectivity.
#    - Top and right blockers create clear shadow zones above/behind them.
#
# f) Visual cleanliness
#    - Only four colors are used.
#    - Structural blockers live on the top/right sides; decorations are small
#      interior blobs; orange is reserved for the final border. This prevents
#      ambiguous overlaps.
#
# g) Color inference
#    - No hidden color remapping exists. Blue always denotes wall material,
#      magenta is always the start marker, and orange always marks the boundary
#      extracted from the ray-scanned region.
#
# Main/new implicit assumptions and generator guarantees
#    - Solver assumption: border-touching blue components are the only blockers
#      for ray stopping. Generator guarantee: all true structural walls are
#      border-touching and decorations never touch the border.
#    - Solver assumption: the marker is a unique non-bg corner pixel. Generator
#      guarantee: marker is always a single magenta pixel at bottom-left.
#    - Solver assumption: non-structural blue cells may be traversed by rays.
#      Generator guarantee: visible decorations are small isolated blobs that do
#      not create contradictory border-touching components.
# -----------------------------------------------------------------------------


def _flood(start: tuple[int, int], passable, h: int, w: int) -> set[tuple[int, int]]:
    q = deque([start])
    vis = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and passable(nr, nc):
                vis.add((nr, nc))
                q.append((nr, nc))
    return vis


def _ray_cells(
    start: tuple[int, int],
    dr: int,
    dc: int,
    passable,
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    r, c = start
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            break
        if not passable(r, c):
            break
        out.append((r, c))
    return out


def _ray_region(
    start: tuple[int, int],
    passable,
    h: int,
    w: int,
) -> set[tuple[int, int]]:
    horizontal_seed = {start}
    vertical_seed = {start}

    for dc in (-1, 1):
        horizontal_seed.update(_ray_cells(start, 0, dc, passable, h, w))
    for dr in (-1, 1):
        vertical_seed.update(_ray_cells(start, dr, 0, passable, h, w))

    reach = set(horizontal_seed) | set(vertical_seed)

    for cell in list(horizontal_seed):
        for dr in (-1, 1):
            reach.update(_ray_cells(cell, dr, 0, passable, h, w))
    for cell in list(vertical_seed):
        for dc in (-1, 1):
            reach.update(_ray_cells(cell, 0, dc, passable, h, w))

    return reach


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint an inner border around the region seen by straight ray scans from a corner marker.

    Input structure:
    - bg = most common color; wall color (blue) appears as large border-touching blockers plus small interior decorations.
    - A unique single pixel (magenta) sits on one corner and acts as the ray origin.
    - Border-touching wall components are the real structures; non-border wall fragments are decorative material.

    Rule:
    - Identify structural walls as the wall-colored components that touch the grid border.
    - Starting from the marker, sweep straight rays along the marker's row and column; then let those seed-ray cells emit one orthogonal straight ray each. Rays pass through bg and decorations but stop at structural walls.
    - Erase decorative wall pixels that were never swept, and paint orange (color 7) on bg cells lying on the inner boundary of the swept region.

    Key insight: Reachability is no longer based on 4-connected motion around corners; it is the union of sight-lines cast from the marker's corner fan, clipped by border-attached walls.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

    counts = Counter(int(v) for v in g.ravel())
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    marker = None
    for r, c in corners:
        v = int(g[r, c])
        if v != bg and counts.get(v, 0) == 1:
            marker = (r, c, v)
            break
    assert marker is not None, "no unique single-pixel corner marker found"

    mr, mc, marker_col = marker

    other_cols = [c for c in counts.keys() if c not in (bg, marker_col)]
    assert other_cols, "no non-bg non-marker colors found; cannot determine wall color"
    wall_col = max(other_cols, key=lambda c: counts[c])

    structural = np.zeros((h, w), dtype=bool)
    for comp_arr in components_mask(g == wall_col, connectivity=4):
        comp = [(int(r), int(c)) for r, c in comp_arr]
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            for r, c in comp:
                structural[r, c] = True

    reach = _ray_region(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col) or (int(g[r, c]) == wall_col and not structural[r, c]),
        h,
        w,
    )

    for r in range(h):
        for c in range(w):
            if int(out[r, c]) == wall_col and not structural[r, c] and (r, c) not in reach:
                out[r, c] = bg

    fill_col = 7
    for r, c in reach:
        if int(out[r, c]) != bg:
            continue
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            out[r, c] = fill_col
            continue
        if any(structural[nr, nc] for nr, nc in neighbors8(r, c) if 0 <= nr < h and 0 <= nc < w):
            out[r, c] = fill_col
            continue
        if any((nr, nc) not in reach for nr, nc in neighbors4(r, c) if 0 <= nr < h and 0 <= nc < w):
            out[r, c] = fill_col

    return out


# --------------- generator ---------------

BG = 8
WALL = 1
MARKER = 6
FILL = 7

GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (10, 20)

NUM_STRUCTS_RANGE = (2, 5)
STRUCT_RETRIES = 12

RAY_REACH_RATIO_RANGE = (0.28, 0.82)
MIN_FLOOD_ADVANTAGE = 10

NUM_DECOR_RANGE = (0, 3)
DECOR_AREA_RANGE = (1, 4)
DECOR_BORDER = 1
DECOR_PADDING = 1

GEN_ATTEMPTS = 400


def _shape_top_rect(h, w, rng):
    rh = rng.randint(2, max(2, min(6, h - 3)))
    rw = rng.randint(2, max(2, min(6, w - 3)))
    c0 = rng.randint(1, max(1, w - rw - 1))
    return [(r, c) for r in range(rh) for c in range(c0, c0 + rw)]


def _shape_top_drop(h, w, rng):
    depth = rng.randint(3, max(3, h - 3))
    c = rng.randint(1, w - 2)
    return [(r, c) for r in range(depth)]


def _shape_right_rect(h, w, rng):
    rh = rng.randint(2, max(2, min(6, h - 3)))
    rw = rng.randint(2, max(2, min(6, w - 3)))
    r0 = rng.randint(1, max(1, h - rh - 1))
    c0 = w - rw
    return [(r, c) for r in range(r0, r0 + rh) for c in range(c0, w)]


def _shape_right_bar(h, w, rng):
    length = rng.randint(3, max(3, w - 3))
    r = rng.randint(1, h - 2)
    return [(r, c) for c in range(w - length, w)]


def _shape_top_right_corner(h, w, rng):
    rh = rng.randint(2, max(2, min(5, h - 3)))
    rw = rng.randint(2, max(2, min(5, w - 3)))
    return [(r, c) for r in range(rh) for c in range(w - rw, w)]


SHAPE_FACTORIES = [
    _shape_top_rect,
    _shape_top_drop,
    _shape_right_rect,
    _shape_right_bar,
    _shape_top_right_corner,
]


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = _dilate4(mask)
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _flood_mask(start, blocked, h, w):
    reach = np.zeros((h, w), dtype=bool)
    reach[start] = True
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not reach[nr, nc] and not blocked[nr, nc]:
                reach[nr, nc] = True
                q.append((nr, nc))
    return reach


def _ray_mask(start, blocked, h, w):
    passable = lambda r, c: not blocked[r, c]
    reach = np.zeros((h, w), dtype=bool)
    for r, c in _ray_region(start, passable, h, w):
        reach[r, c] = True
    return reach


def _paint_output(structural: np.ndarray, reach: np.ndarray, marker_rc: tuple[int, int]) -> np.ndarray:
    out = np.full(structural.shape, BG, dtype=int)
    out[structural] = WALL
    out[marker_rc] = MARKER

    candidate = reach & (out == BG)

    edge = np.zeros_like(candidate)
    edge[0, :] = True
    edge[-1, :] = True
    edge[:, 0] = True
    edge[:, -1] = True

    near_struct = _dilate8(structural)
    near_unreach = _dilate4(~reach)
    fill = candidate & (edge | near_struct | near_unreach)
    out[fill] = FILL
    return out


def _place_structural_walls(h, w, rng):
    structural = np.zeros((h, w), dtype=bool)
    num = rng.randint(*NUM_STRUCTS_RANGE)
    placed = 0
    for _ in range(num):
        for _try in range(STRUCT_RETRIES):
            cells = rng.choice(SHAPE_FACTORIES)(h, w, rng)
            if any(structural[r, c] for r, c in cells):
                continue
            for r, c in cells:
                structural[r, c] = True
            placed += 1
            break
    if placed == 0:
        return None
    ratio = float(structural.mean())
    if not (0.05 <= ratio <= 0.20):
        return None
    return structural


def _validate_structural(structural: np.ndarray, h: int, w: int) -> bool:
    for comp_arr in components_mask(structural, connectivity=4):
        cmask = np.zeros((h, w), dtype=bool)
        cmask[comp_arr[:, 0], comp_arr[:, 1]] = True
        if not _touches_border(cmask):
            return False
    return True


def _place_decor(allowed_mask, count, rng):
    h, w = allowed_mask.shape
    placed = np.zeros((h, w), dtype=bool)
    if count == 0:
        return placed
    for _ in range(count):
        area = rng.randint(*DECOR_AREA_RANGE)
        shapes = [(1, area), (area, 1)]
        if area >= 3:
            shapes.append((2, 2))
        rng.shuffle(shapes)
        for ph, pw in shapes:
            if ph > h or pw > w:
                continue
            patch = np.ones((ph, pw), dtype=int)
            canvas = np.where(allowed_mask & ~placed, 0, 9)
            try:
                r0, c0 = random_free_location(
                    canvas,
                    patch,
                    transparent=0,
                    border=DECOR_BORDER,
                    padding=DECOR_PADDING,
                    padding_connectivity=8,
                )
            except ValueError:
                continue
            placed[r0:r0 + ph, c0:c0 + pw] = True
            break
        else:
            return None
    return placed


def _gen_attempt():
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    marker_rc = (h - 1, 0)

    structural = _place_structural_walls(h, w, rng)
    if structural is None or structural[marker_rc]:
        return None
    if not _validate_structural(structural, h, w):
        return None

    ray_reach = _ray_mask(marker_rc, structural, h, w)
    reach_frac = float(ray_reach.mean())
    if not (RAY_REACH_RATIO_RANGE[0] <= reach_frac <= RAY_REACH_RATIO_RANGE[1]):
        return None

    flood_reach = _flood_mask(marker_rc, structural, h, w)
    flood_advantage = int(flood_reach.sum() - ray_reach.sum())
    if flood_advantage < max(MIN_FLOOD_ADVANTAGE, int(0.10 * h * w)):
        return None

    out = _paint_output(structural, ray_reach, marker_rc)

    fill_mask = out == FILL
    if not fill_mask.any():
        return None
    if len(components_mask(fill_mask, connectivity=4)) != 1:
        return None

    remaining_bg = (out == BG) & ray_reach
    if int(remaining_bg.sum()) < 4:
        return None

    n_decor = rng.randint(*NUM_DECOR_RANGE)
    n_visible = rng.randint(0, n_decor)
    n_hidden = n_decor - n_visible

    visible_allowed = ray_reach & (out == BG)
    hidden_allowed = (~ray_reach) & (~structural)
    hidden_allowed[0, :] = False
    hidden_allowed[-1, :] = False
    hidden_allowed[:, 0] = False
    hidden_allowed[:, -1] = False

    visible_placed = _place_decor(visible_allowed, n_visible, rng)
    if visible_placed is None:
        return None
    hidden_placed = _place_decor(hidden_allowed, n_hidden, rng)
    if hidden_placed is None:
        return None

    out_final = out.copy()
    out_final[visible_placed] = WALL

    inp = out_final.copy()
    inp[inp == FILL] = BG
    inp[hidden_placed] = WALL

    if get_bg(inp) != BG:
        return None
    if not np.array_equal(main(inp), out_final):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError("failed to generate valid input for 8f3a5a89 D1 ray variant")
