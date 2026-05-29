# Variant of 8f3a5a89
# Dimension: D2 — 内边界涂色规则 (trajectory)
# Change: inner_border——可达区域中，位于网格边缘、8-邻接结构墙、或4-邻接不可达像素的背景格子涂橙色，形成1像素宽... → 外边界涂色（在不可达侧涂色）

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


# Design analysis for the D2 variant
# ----------------------------------
# variant_spec:
#   core_change:
#     original: orange is painted on reachable background cells that lie on the
#       inside of the reach/unreach divider.
#     variant: orange is painted on unreachable background cells that lie on
#       the outside of that same divider.
#   concrete_behavior:
#     - The magenta marker still sits in one corner and defines the side from
#       which flood-fill starts.
#     - Blue border-touching components that are adjacent to the first flood are
#       still the real walls; other blue pixels are still decorations.
#     - Hidden decorations behind structural walls are still erased.
#     - The only visual inversion is which side of the divider receives orange:
#       cells from the unreachable background are colored if they are on a grid
#       edge, 8-neighbor a structural wall, or 4-neighbor the reachable region.
#   behavior_trace:
#     - Small example: a full vertical wall splits the board into a left
#       reachable strip (with the bottom-left marker) and a right unreachable
#       strip.
#     - Original rule: paint the left strip's edge cells near the wall and the
#       outer frame, producing an inner orange ring on the reachable side.
#     - Variant rule: keep the same wall detection, but instead paint the right
#       strip's cells near the wall and the outer frame, producing the mirrored
#       outer ring.
#     - Difference: the orange region is disjoint from the original one and
#       proves that the solver must understand which side is unreachable.
#   complexity_calibration:
#     - The palette and wall semantics stay unchanged, so the task does not add
#       new object types or ambiguous color mappings.
#     - The generator must reject layouts where structural walls fail to cut off
#       any background cells; otherwise the "outside" ring would disappear and
#       the D2 change would not be observable.
#     - We keep 1-4 structural components and 0-3 decoration blobs so examples
#       still show repeated evidence without exploding the number of cases.
#   implicit_assumptions_review:
#     - Marker detection by a unique corner pixel still holds.
#     - Choosing the most frequent non-bg, non-marker color as the wall color
#       still holds because the palette is unchanged.
#     - Structural-wall recovery from border-touching + adjacency to reach0
#       still holds; the variant does not change which blue pixels are barriers.
#     - The old fill loop assumed candidate cells were inside `reach`; that
#       assumption is invalid now and must be flipped to unreachable background.
#     - Hidden-decoration erasure must still happen before painting, otherwise a
#       hidden blue blob could incorrectly block or suppress outer-border paint.
#   placement_legality:
#     - Structural shapes remain border-touching and non-overlapping.
#     - Visible decorations stay inside the reachable background.
#     - Hidden decorations stay off the border so the solver never mistakes them
#       for structural walls.
#     - The generator validates that the orange mask is non-empty and 4-
#       connected, and that some unreachable background exists.
#   visual_observability:
#     - Across samples, orange moves to the opposite side of the same blue
#       divider. This is visible even when decorations are absent.
#     - The generator does not force a single wall shape, so the rule must be
#       inferred as "paint the other side of the separator", not memorized from
#       one silhouette.
#   visual_cleanliness:
#     - Orange is only painted on background cells, never over the marker or
#       walls, so layers remain readable.
#     - Decorations remain small and sparse to avoid cluttering the region whose
#       side classification the solver must infer.
#   color_inference:
#     - No new color mapping is introduced; orange remains the sole output-only
#       signal, and its placement is determined by geometry rather than color id.


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


def _mask_from_coords(coords: set[tuple[int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


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


def _outer_fill_mask(bg_mask: np.ndarray, reach_mask: np.ndarray, wall_mask: np.ndarray) -> np.ndarray:
    candidate = (~reach_mask) & bg_mask

    edge = np.zeros_like(candidate)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True

    near_wall = _dilate8(wall_mask)
    near_reach = _dilate4(reach_mask)
    return candidate & (edge | near_wall | near_reach)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint an outer border in the unreachable side of the region split by blue walls.

    Input structure:
    - The most common color is background, while blue wall-colored pixels divide the grid.
    - A unique corner marker pixel identifies the flood-fill start side.
    - Some blue components are real border-connected barriers; other blue fragments are decorations.

    Rule:
    - Flood-fill from the marker corner to recover the background reachable from that side.
    - Keep only border-touching blue components that touch the first reachable area as structural walls, and erase hidden blue decorations behind them.
    - Paint orange on unreachable background cells that sit on the grid edge, 8-touch a structural wall, or 4-touch the reachable region, forming a 1-cell outer border.

    Key insight: The divider itself is unchanged from the original task; the variant is solved by identifying the same reach/unreach partition and coloring the opposite side of it.
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
    other_cols = [c for c in counts if c not in (bg, marker_col)]
    assert other_cols, "no non-bg non-marker colors found; cannot determine wall color"
    wall_col = max(other_cols, key=lambda c: counts[c])

    reach0 = _flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col),
        h,
        w,
    )

    walls: set[tuple[int, int]] = set()
    for comp_arr in components_mask(g == wall_col, connectivity=4):
        comp = [(int(r), int(c)) for r, c in comp_arr]
        touches_border = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)
        if not touches_border:
            continue
        adjacent_to_reach0 = any(any((nr, nc) in reach0 for nr, nc in neighbors4(r, c)) for r, c in comp)
        if adjacent_to_reach0:
            walls |= set(comp)

    reach = _flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col) or (int(g[r, c]) == wall_col and (r, c) not in walls),
        h,
        w,
    )

    for r in range(h):
        for c in range(w):
            if int(out[r, c]) == wall_col and (r, c) not in walls and (r, c) not in reach:
                out[r, c] = bg

    reach_mask = _mask_from_coords(reach, h, w)
    wall_mask = _mask_from_coords(walls, h, w)
    fill_mask = _outer_fill_mask(out == bg, reach_mask, wall_mask)
    out[fill_mask] = 7
    return out


# --------------- generator ---------------

BG = 8
WALL = 1
MARKER = 6
FILL = 7

GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (10, 20)

NUM_STRUCTS_RANGE = (1, 4)
STRUCT_SHAPE_RETRIES = 8
RECT_MAX_DIM_DIV = 3
TRI_MAX_SIZE = 6
TRI_MIN_DIM_DIV = 3

REACH_RATIO_RANGE = (0.38, 0.87)

NUM_DECOR_RANGE = (0, 3)
DECOR_AREA_RANGE = (1, 4)
DECOR_BORDER = 1
DECOR_PADDING = 1

GEN_ATTEMPTS = 300
MIN_FILL_CELLS = 5


def _shape_vertical_line(h, w, rng):
    c = rng.randint(2, w - 2)
    mode = rng.choice(["top", "bottom", "full"])
    if mode == "full":
        r0, r1 = 0, h - 1
    elif mode == "top":
        r0, r1 = 0, rng.randint(max(1, h // 3), h - 2)
    else:
        r0, r1 = rng.randint(1, h - max(2, h // 3)), h - 1
    return [(r, c) for r in range(r0, r1 + 1)]


def _shape_horizontal_line(h, w, rng):
    r = rng.randint(1, h - 2)
    if rng.choice(["left", "right"]) == "left":
        c0, c1 = 0, rng.randint(max(1, w // 3), w - 2)
    else:
        c0, c1 = rng.randint(1, w - max(2, w // 3)), w - 1
    return [(r, c) for c in range(c0, c1 + 1)]


def _shape_rectangle(h, w, rng):
    rh = rng.randint(2, max(2, h // RECT_MAX_DIM_DIV))
    rw = rng.randint(2, max(2, w // RECT_MAX_DIM_DIV))
    anchor = rng.choice(["top", "right", "tr", "top_mid", "right_mid", "bottom_right"])
    if anchor == "top":
        r0, c0 = 0, rng.randint(1, max(1, w - rw - 1))
    elif anchor == "right":
        r0, c0 = rng.randint(1, max(1, h - rh - 1)), w - rw
    elif anchor == "tr":
        r0, c0 = 0, w - rw
    elif anchor == "top_mid":
        r0, c0 = 0, rng.randint(2, max(2, w - rw - 2))
    elif anchor == "right_mid":
        r0, c0 = rng.randint(2, max(2, h - rh - 2)), w - rw
    else:
        r0, c0 = h - rh, w - rw
    return [(r, c) for r in range(r0, r0 + rh) for c in range(c0, c0 + rw)]


def _shape_left_diamond(h, w, rng):
    max_size = min(TRI_MAX_SIZE, h // TRI_MIN_DIM_DIV, w // TRI_MIN_DIM_DIV)
    if max_size < 3:
        return None
    size = rng.randint(3, max_size)
    r_center = rng.randint(size, h - size - 1)
    top = r_center - (size - 1)
    cells = []
    for i in range(size):
        cells.extend((top + i, c) for c in range(i + 1))
    for i in range(size - 1):
        cells.extend((top + size + i, c) for c in range(size - 1 - i))
    return cells


def _shape_top_triangle(h, w, rng):
    max_size = min(TRI_MAX_SIZE, h // TRI_MIN_DIM_DIV, w // TRI_MIN_DIM_DIV)
    if max_size < 2:
        return None
    size = rng.randint(2, max_size)
    c_start = rng.randint(1, max(1, w - 2 * size))
    cells = []
    for i in range(size):
        cells.extend((i, c) for c in range(c_start, c_start + size - i))
    return cells


def _shape_right_triangle(h, w, rng):
    max_size = min(TRI_MAX_SIZE, h // 2, w // 2)
    if max_size < 2:
        return None
    size = rng.randint(2, max_size)
    r_start = rng.randint(0, max(0, h - size - 1))
    cells = []
    for i in range(size):
        length = i + 1
        cells.extend((r_start + i, c) for c in range(w - length, w))
    return cells


SHAPE_FACTORIES = [
    _shape_vertical_line,
    _shape_horizontal_line,
    _shape_rectangle,
    _shape_left_diamond,
    _shape_top_triangle,
    _shape_right_triangle,
]


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


def _touches_border(mask):
    return mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any()


def _place_structural_walls(h, w, marker_rc, rng):
    num = rng.randint(*NUM_STRUCTS_RANGE)
    structural = np.zeros((h, w), dtype=bool)
    placed = 0
    for _ in range(num):
        for _try in range(STRUCT_SHAPE_RETRIES):
            cells = rng.choice(SHAPE_FACTORIES)(h, w, rng)
            if not cells:
                continue
            if marker_rc in cells or any(structural[r, c] for r, c in cells):
                continue
            for r, c in cells:
                structural[r, c] = True
            placed += 1
            break
    return structural if placed > 0 else None


def _validate_structural(structural, reach, h, w):
    reach_nbr = _dilate4(reach)
    for comp_arr in components_mask(structural, connectivity=4):
        cmask = np.zeros((h, w), dtype=bool)
        cmask[comp_arr[:, 0], comp_arr[:, 1]] = True
        if not _touches_border(cmask):
            return False
        if not (cmask & reach_nbr).any():
            return False
    return True


def _paint_output(structural, reach, marker_rc):
    out = np.full(structural.shape, BG, dtype=int)
    out[structural] = WALL
    out[marker_rc] = MARKER
    fill = _outer_fill_mask(out == BG, reach, structural)
    out[fill] = FILL
    return out


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
            placed[r0 : r0 + ph, c0 : c0 + pw] = True
            break
        else:
            return None
    return placed


def _gen_attempt():
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    marker_rc = (h - 1, 0)

    structural = _place_structural_walls(h, w, marker_rc, rng)
    if structural is None or structural[marker_rc]:
        return None

    reach = _flood_mask(marker_rc, structural, h, w)
    reach_frac = reach.mean()
    if not (REACH_RATIO_RANGE[0] <= reach_frac <= REACH_RATIO_RANGE[1]):
        return None

    if not _validate_structural(structural, reach, h, w):
        return None

    unreachable_bg = (~reach) & (~structural)
    if not unreachable_bg.any():
        return None

    out = _paint_output(structural, reach, marker_rc)
    fill_mask = out == FILL
    if int(fill_mask.sum()) < MIN_FILL_CELLS or len(components_mask(fill_mask, connectivity=4)) != 1:
        return None

    n_decor = rng.randint(*NUM_DECOR_RANGE)
    n_visible = rng.randint(0, n_decor)
    n_hidden = n_decor - n_visible

    visible_allowed = reach & (out == BG)
    hidden_allowed = (~reach) & (~structural)
    hidden_allowed[0, :] = hidden_allowed[-1, :] = False
    hidden_allowed[:, 0] = hidden_allowed[:, -1] = False

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
    raise RuntimeError("failed to generate valid input for 8f3a5a89 D2 outer-border variant")
