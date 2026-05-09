# Combo variant of 8f3a5a89 (index 11)
# combo_spec: D6/2+D20/0
# Name: 装饰墙环绕涂色+多独立边界环

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


# -----------------------------------------------------------------------------
# Design analysis for the combo variant
# -----------------------------------------------------------------------------
# fused_rule:
#   - Keep the original structural-wall inference: only border-touching blue
#     components adjacent to the marker side are true barriers.
#   - Keep erasing unreachable blue decorations.
#   - Keep the large orange inner border of the marker-reachable region.
#   - Add D6/2: each reachable decoration wall is also surrounded on its
#     reachable 4-neighbor background cells by orange.
#   - Add D20/0: do not require all orange pixels to form one component; the
#     intended output is one large border plus one or more detached small halos.
# concrete_trace:
#   - Local patch before decoration processing:
#       88888
#       88888
#       88188
#       88888
#       88888
#     where the center `1` is a reachable interior decoration and the area is
#     far from the main reachable-region border.
#   - Base task output keeps the blue `1` unchanged and paints only the large
#     reachable-region border elsewhere.
#   - This combo keeps that `1`, but also paints its N/S/E/W azure neighbors to
#     `7`, yielding a detached 4-cell orange halo. The global border still
#     exists, so the output becomes "one big ring + small local rings".
# complexity_calibration:
#   - Visible decorations are restricted to isolated single pixels. That makes
#     each small halo visually stable and easy to infer from 2-4 examples.
#   - We require exactly 2 visible decorations so every example visibly shows
#     one large border plus multiple detached small rings, making D20/0 easy to
#     infer without cluttering the canvas.
#   - Hidden decorations remain optional, preserving the original contrast
#     between reachable-side retention and blocked-side erasure.
# implicit_assumption_audit:
#   - The first flood still treats every blue pixel as blocking. This remains
#     necessary so interior decorations cannot masquerade as structural walls.
#   - Structural-wall identification still assumes "touches border" plus
#     "adjacent to the first reachable side"; the generator must therefore keep
#     visible decorations away from the border.
#   - The second flood still walks through non-structural blue cells; otherwise
#     reachable decorations would be mistaken as blocked-side objects.
#   - Orange writes must remain background-only. Without that guard, the halo
#     step could overwrite walls or the marker.
#   - D20/0 removes only a generator-side invariant. The solver already permits
#     disconnected orange pieces once the D6/2 surround rule is added.
# placement_legality:
#   - A visible decoration center is legal only if its four orthogonal neighbors
#     are all reachable azure cells and none of those halo cells is 8-adjacent
#     to the main border. This guarantees a detached small ring.
#   - Multiple visible decorations are greedily spaced so their halos are also
#     8-disconnected from one another.
#   - Hidden decorations are sampled only from unreachable non-border cells, so
#     the solver will erase them instead of reclassifying them as structural.
# visual_observability:
#   - Every generated puzzle contains exactly two visible decorations.
#   - Validation requires the final orange mask to split into exactly
#     `1 + n_visible` 8-connected components: one large border component and one
#     detached 4-cell halo per visible decoration.
# cleanliness_and_inferability:
#   - No new colors are introduced. Orange still means "boundary", but now that
#     concept applies both to the reachable-region perimeter and to reachable
#     interior wall decorations.


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
    Paint the marker-side inner boundary, while also giving reachable decorations their own halos.

    Input structure:
    - Azure is the background, magenta is a unique corner marker, and blue pixels split into
      border-touching structural walls versus interior decoration walls.
    - Structural walls, together with the grid edge, separate the marker-reachable side from
      blocked pockets.
    - Decoration walls may appear on either side; only the reachable-side ones survive.

    Rule:
    - Flood from the marker twice: first with all blue blocking to locate the marker-facing side,
      then again while allowing passage through non-structural blue decorations.
    - Erase unreachable decorations, keep reachable decorations blue, and paint the standard
      orange inner border on reachable background cells that touch an edge, a structural wall,
      or unreachable space.
    - Additionally paint orange on reachable background cells that are 4-neighbors of reachable
      decorations, producing detached mini-halos around them.

    Key insight: Border-touching blue components still define the true enclosure, but reachable
    interior blue decorations now behave like tiny secondary boundaries, so the output can
    legitimately contain several independent orange rings instead of just one connected border.
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
    other_cols = [col for col in counts if col not in (bg, marker_col)]
    assert other_cols, "no non-bg non-marker colors found; cannot determine wall color"
    wall_col = max(other_cols, key=lambda col: counts[col])

    reach0 = _flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col),
        h,
        w,
    )

    walls: set[tuple[int, int]] = set()
    for comp_arr in components_mask(g == wall_col, connectivity=4):
        comp = [(int(r), int(c)) for r, c in comp_arr]
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        if any(any((nr, nc) in reach0 for nr, nc in neighbors4(r, c)) for r, c in comp):
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

NUM_VISIBLE_DECOR_RANGE = (2, 2)
NUM_HIDDEN_DECOR_RANGE = (0, 1)
DECOR_AREA_RANGE = (1, 4)
DECOR_BORDER = 1
DECOR_PADDING = 1

GEN_ATTEMPTS = 400


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
    out = _dilate4(mask)
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _place_structural_walls(h, w, marker_rc, rng):
    num = rng.randint(*NUM_STRUCTS_RANGE)
    structural = np.zeros((h, w), dtype=bool)
    placed = 0
    for _ in range(num):
        for _ in range(STRUCT_SHAPE_RETRIES):
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
    edge[0, :] = True
    edge[-1, :] = True
    edge[:, 0] = True
    edge[:, -1] = True

    near_struct = _dilate8(structural)
    near_unreach = _dilate4(~reach)

    fill = candidate & (edge | near_struct | near_unreach)
    out[fill] = FILL

    if visible_decor is not None:
        for r, c in np.argwhere(visible_decor):
            for nr, nc in neighbors4(int(r), int(c)):
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


def _decor_halo(center, h, w):
    r, c = center
    halo = []
    for nr, nc in neighbors4(r, c):
        if 0 <= nr < h and 0 <= nc < w:
            halo.append((nr, nc))
    return halo


def _find_visible_decor_candidates(base_out, reach):
    h, w = base_out.shape
    base_fill = base_out == FILL
    near_base_fill = _dilate8(base_fill)
    candidates: list[tuple[tuple[int, int], tuple[tuple[int, int], ...]]] = []
    for r, c in np.argwhere(reach & (base_out == BG)):
        r = int(r)
        c = int(c)
        halo = _decor_halo((r, c), h, w)
        if len(halo) != 4:
            continue
        if any(not reach[nr, nc] or base_out[nr, nc] != BG for nr, nc in halo):
            continue
        if any(near_base_fill[nr, nc] for nr, nc in halo):
            continue
        candidates.append(((r, c), tuple(halo)))
    return candidates


def _select_visible_decor(base_out, reach, rng):
    candidates = _find_visible_decor_candidates(base_out, reach)
    if not candidates:
        return None

    rng.shuffle(candidates)
    h, w = base_out.shape
    chosen: list[tuple[tuple[int, int], tuple[tuple[int, int], ...]]] = []
    forbidden = np.zeros((h, w), dtype=bool)
    limit = NUM_VISIBLE_DECOR_RANGE[1]

    for center, halo in candidates:
        pts = [center, *halo]
        if any(forbidden[r, c] for r, c in pts):
            continue
        chosen.append((center, halo))
        local = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            local[r, c] = True
        forbidden |= _dilate8(local)
        if len(chosen) == limit:
            break

    target = NUM_VISIBLE_DECOR_RANGE[0]
    if len(chosen) < target:
        return None

    visible = np.zeros((h, w), dtype=bool)
    for center, _halo in chosen[:target]:
        visible[center] = True
    return visible


def _gen_attempt():
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    marker_rc = (h - 1, 0)

    structural = _place_structural_walls(h, w, marker_rc, rng)
    if structural is None or structural[marker_rc]:
        return None

    reach = _flood_mask(marker_rc, structural, h, w)
    reach_frac = float(reach.mean())
    if not (REACH_RATIO_RANGE[0] <= reach_frac <= REACH_RATIO_RANGE[1]):
        return None

    if not _validate_structural(structural, reach, h, w):
        return None

    out_base = _paint_output(structural, reach, marker_rc)
    base_fill_mask = out_base == FILL
    if not base_fill_mask.any():
        return None
    if len(components_mask(base_fill_mask, connectivity=4)) != 1:
        return None

    visible_placed = _select_visible_decor(out_base, reach, rng)
    if visible_placed is None:
        return None

    hidden_allowed = (~reach) & (~structural)
    hidden_allowed[0, :] = False
    hidden_allowed[-1, :] = False
    hidden_allowed[:, 0] = False
    hidden_allowed[:, -1] = False

    n_hidden = rng.randint(*NUM_HIDDEN_DECOR_RANGE)
    hidden_placed = _place_decor(hidden_allowed, n_hidden, rng)
    if hidden_placed is None:
        hidden_placed = np.zeros_like(hidden_allowed, dtype=bool)

    out_final = _paint_output(structural, reach, marker_rc, visible_placed)

    fill_mask = out_final == FILL
    fill_comps8 = components_mask(fill_mask, connectivity=8)
    n_visible = int(visible_placed.sum())
    if len(fill_comps8) != 1 + n_visible:
        return None

    extra_fill = (out_final == FILL) & (out_base != FILL)
    extra_comps8 = components_mask(extra_fill, connectivity=8)
    if len(extra_comps8) != n_visible:
        return None
    if any(len(comp) != 4 for comp in extra_comps8):
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
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 8f3a5a89 combo variant 11")
