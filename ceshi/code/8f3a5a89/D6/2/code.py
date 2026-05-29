# Variant of 8f3a5a89
# Dimension: D6 — 装饰墙处理规则 (transform)
# Change: 可达区域外的装饰墙擦除为背景色，可达区域内的装饰墙保留为墙色... → 装饰墙被边界色环绕

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


# -----------------------------------------------------------------------------
# Design analysis for the D6 variant
# -----------------------------------------------------------------------------
# variant_spec:
#   changed_dimension: D6 only
#   unchanged_core:
#     - marker is still one unique corner pixel and still seeds the flood fill
#     - border-touching blue components that face the marker side are still the
#       only structural walls
#     - unreachable decorations are still erased
#     - the main orange boundary of the marker-reachable region is still painted
#       exactly as in the base task
#   new_behavior:
#     - every reachable non-structural wall pixel/component remains blue
#     - additionally, each such reachable decoration paints orange on its
#       4-neighbor background cells, producing a small detached ring/halo
#   concrete_pixel_effect:
#     - original: a visible interior decoration stays blue and does not affect
#       nearby azure cells
#     - variant: the same blue decoration stays blue, but its north/south/east/
#       west azure neighbors turn orange, so the output contains extra local
#       orange loops besides the global reachable-region border
#   traced_example:
#     - if a reachable interior cell pattern is `888 / 818 / 888`, the base task
#       outputs the same local pattern, while this variant outputs
#       `878 / 717 / 878` restricted to the 4-neighbors of the center blue pixel
#       (the center itself remains blue)
#   complexity_calibration:
#     - reachable decorations are kept small and sparse so the new rule is easy
#       to notice and does not dominate the original border logic
#     - visible decorations are generated as isolated single pixels; this is a
#       deliberate D6-only adaptation so the orange surround is unambiguous in
#       2-4 examples
#     - hidden decorations still vary independently, preserving the original
#       "erase the blocked side" contrast
#   implicit-assumption_audit:
#     - the first flood still treats every blue cell as blocking; this remains
#       valid because decorations must not help define structural walls
#     - the second flood still walks through non-structural blue cells; this is
#       now more important, because reachable decorations must be recognized as
#       being on the marker side before their surrounds are painted
#     - when painting orange, background-only writes are still required; without
#       that guard the new surround step could overwrite walls or the marker
#     - structural-wall detection still relies on border-touch + adjacency to the
#       first reach set, so the generator must keep all visible decorations away
#       from the border
#   placement_legality:
#     - visible decorations are sampled only from deep interior reachable azure
#       cells obtained by eroding the post-border-fill interior by two 4-neighbor
#       layers; that guarantees space for a detached orange surround
#     - selected visible decorations are spaced apart by repeated 4-dilation so
#       their surrounds do not merge
#     - hidden decorations are still non-border and non-structural, so the solver
#       will not misclassify them as structural walls
#   visual_observability:
#     - every generated puzzle contains at least one visible decoration
#     - generator validation treats each tiny surround as an 8-connected halo,
#       which matches the intended "small ring" visual effect even though the
#       painted cells came from a 4-neighbor rule
#     - generator validation rejects outputs where the new orange surround merges
#       into the main orange border, because that would hide the D6 change
#   cleanliness_and_color_reasoning:
#     - no new colors are introduced; orange already denotes "boundary"
#     - the new orange-on-azure pattern appears only around reachable
#       decorations, giving a spatially local clue instead of a color-id trick


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint the marker-reachable inner border, and also surround reachable decorations.

    Input structure:
    - The background is the most common color, with one unique corner marker pixel.
    - Blue wall-color pixels split into two roles: border-touching structural barriers
      and interior decoration fragments.
    - Decorations may lie on the reachable side or be hidden behind the structural walls.

    Rule:
    - Flood-fill from the marker corner, first treating all blue as blocking so the
      true barrier-facing side is identified.
    - Reclassify border-touching blue components adjacent to that first reach as
      structural walls; flood again while allowing passage through other blue pixels.
    - Erase unreachable decorations, paint the original orange inner border of the
      reachable region, and additionally paint orange on reachable background cells
      that are 4-neighbors of reachable decorations.

    Key insight: Structural walls are still the only real separators, but reachable
    decoration walls now act like tiny local boundaries that receive their own orange surrounds.
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

    visible_decor = {
        (r, c)
        for r in range(h)
        for c in range(w)
        if int(g[r, c]) == wall_col and (r, c) not in walls and (r, c) in reach
    }

    for r in range(h):
        for c in range(w):
            if int(out[r, c]) == wall_col and (r, c) not in walls and (r, c) not in reach:
                out[r, c] = bg

    fill_col = 7
    for r, c in reach:
        if int(out[r, c]) != bg:
            continue
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            out[r, c] = fill_col
            continue
        if any((nr, nc) in walls for nr, nc in neighbors8(r, c)):
            out[r, c] = fill_col
            continue
        if any((nr, nc) not in reach for nr, nc in neighbors4(r, c)):
            out[r, c] = fill_col

    for r, c in visible_decor:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in reach and int(out[nr, nc]) == bg:
                out[nr, nc] = fill_col

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

NUM_DECOR_RANGE = (1, 3)
NUM_VISIBLE_DECOR_RANGE = (1, 2)
DECOR_AREA_RANGE = (1, 4)
DECOR_BORDER = 1
DECOR_PADDING = 1

GEN_ATTEMPTS = 300


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


def _dilate4(mask):
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _dilate8(mask):
    m = _dilate4(mask)
    m[1:, 1:] |= mask[:-1, :-1]
    m[1:, :-1] |= mask[:-1, 1:]
    m[:-1, 1:] |= mask[1:, :-1]
    m[:-1, :-1] |= mask[1:, 1:]
    return m


def _erode4(mask, steps):
    out = mask.copy()
    for _ in range(steps):
        eroded = out.copy()
        eroded[0, :] = False
        eroded[-1, :] = False
        eroded[:, 0] = False
        eroded[:, -1] = False
        eroded[1:, :] &= out[:-1, :]
        eroded[:-1, :] &= out[1:, :]
        eroded[:, 1:] &= out[:, :-1]
        eroded[:, :-1] &= out[:, 1:]
        out = eroded
    return out


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


def _paint_output(structural, reach, marker_rc, visible_decor=None):
    h, w = structural.shape
    out = np.full((h, w), BG, dtype=int)
    out[structural] = WALL
    out[marker_rc] = MARKER
    if visible_decor is not None:
        out[visible_decor] = WALL

    candidate = reach & (out == BG)

    edge = np.zeros_like(candidate)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True

    near_struct = _dilate8(structural)
    near_unreach = _dilate4(~reach)

    fill = candidate & (edge | near_struct | near_unreach)
    out[fill] = FILL

    if visible_decor is not None:
        for r, c in np.argwhere(visible_decor):
            r = int(r)
            c = int(c)
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and reach[nr, nc] and out[nr, nc] == BG:
                    out[nr, nc] = FILL

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
            placed[r0:r0 + ph, c0:c0 + pw] = True
            break
        else:
            return None
    return placed


def _place_visible_decor(allowed_mask, count, rng):
    placed = np.zeros_like(allowed_mask, dtype=bool)
    candidates = allowed_mask.copy()
    for _ in range(count):
        coords = np.argwhere(candidates)
        if len(coords) == 0:
            return None
        idx = rng.randrange(len(coords))
        r, c = map(int, coords[idx])
        placed[r, c] = True
        block = np.zeros_like(candidates, dtype=bool)
        block[r, c] = True
        for _ in range(3):
            block = _dilate4(block)
        candidates &= ~block
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

    out_base = _paint_output(structural, reach, marker_rc)
    base_fill_mask = out_base == FILL
    if not base_fill_mask.any() or len(components_mask(base_fill_mask, connectivity=4)) != 1:
        return None

    deep_visible_allowed = _erode4(reach & (out_base == BG), 2)
    max_visible = min(NUM_VISIBLE_DECOR_RANGE[1], int(deep_visible_allowed.sum()))
    if max_visible < NUM_VISIBLE_DECOR_RANGE[0]:
        return None
    n_visible = rng.randint(NUM_VISIBLE_DECOR_RANGE[0], max_visible)
    visible_placed = _place_visible_decor(deep_visible_allowed, n_visible, rng)
    if visible_placed is None:
        return None

    hidden_cap = min(1, NUM_DECOR_RANGE[1] - n_visible)
    n_hidden = rng.randint(0, hidden_cap)

    hidden_allowed = (~reach) & (~structural)
    hidden_allowed[0, :] = hidden_allowed[-1, :] = False
    hidden_allowed[:, 0] = hidden_allowed[:, -1] = False

    hidden_placed = _place_decor(hidden_allowed, n_hidden, rng)
    if hidden_placed is None:
        hidden_placed = np.zeros_like(hidden_allowed, dtype=bool)

    out_final = _paint_output(structural, reach, marker_rc, visible_placed)

    fill_mask = out_final == FILL
    fill_comps = components_mask(fill_mask, connectivity=8)
    if len(fill_comps) != 1 + n_visible:
        return None

    decor_fill_mask = (out_final == FILL) & (out_base != FILL)
    decor_fill_comps = components_mask(decor_fill_mask, connectivity=8)
    if len(decor_fill_comps) != n_visible:
        return None
    if any(len(comp) != 4 for comp in decor_fill_comps):
        return None

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
    raise RuntimeError("failed to generate valid input for 8f3a5a89 D6 variant")
