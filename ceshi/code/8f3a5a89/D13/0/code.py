# Variant of 8f3a5a89
# Dimension: D13 — 边界涂色的邻接条件 (trajectory)
# Change: 三种条件的并集：(1) 位于网格边缘 (2) 8-邻接结构墙 (3) 4-邻接不可达像素... → 统一为4-邻接不可达/墙壁
#
# Design analysis
# a) Concrete behavior:
#    - The blue border-touching components still define the blocked side exactly as in the
#      base task; the magenta marker still starts at the bottom-left corner.
#    - The only changed stage is the orange painting test on reachable background cells.
#      Old rule: paint if on grid edge, or 8-near a structural wall, or 4-near an
#      unreachable cell. New rule: ignore off-grid space and ignore diagonal wall contact;
#      paint only if some in-bounds 4-neighbor is a structural wall or lies outside the
#      marker-reachable set.
#    - Small example: if a reachable cell touches a wall only diagonally, old logic paints
#      it because of the 8-neighbor test; new logic leaves it azure because none of its
#      four orthogonal neighbors is blocked. Likewise, a reachable edge cell with only open
#      interior neighbors was orange before and is now left unpainted.
# b) Complexity calibration:
#    - The rest of the task stays unchanged so the object vocabulary is stable: one corner
#      marker, 1-4 structural wall components, optional decorative wall blobs.
#    - To make D13 visible, the generator rejects samples where old and new paint masks are
#      identical. It also prefers samples with at least one diagonal-only omission so the
#      thinner ring is not reduced to a trivial “outer edge no longer paints” effect.
# c) Implicit assumptions audit for main():
#    - Marker detection still assumes the corner marker is the unique singleton non-bg
#      color; generator keeps exactly one magenta corner pixel.
#    - Wall detection still assumes blue is the dominant non-bg non-marker color; generator
#      keeps structural walls much larger than decorative blobs.
#    - Structural-wall recovery still depends on border-touching + adjacency to first reach;
#      generator still validates every structural component against that criterion.
#    - The new paint loop assumes “off-grid” should not count as blocked; generator therefore
#      must not rely on exposed outer edges alone to create orange cells.
# d) Placement legality:
#    - Structural shapes remain non-overlapping and border-touching.
#    - Hidden decorations are forced away from the border so they cannot be mistaken for
#      structural components.
#    - Visible decorations are kept away from both old and new border masks so they do not
#      obscure the D13 signal.
# e) Visual observability:
#    - Across samples, reachable outer edges now often stay azure instead of becoming orange.
#    - Concave/diagonal wall corners can show visible missing orange corner pixels compared
#      with the base task.
# f) Visual cleanliness:
#    - The palette stays at four semantic colors.
#    - Decorations remain small and isolated, and border cells are not overloaded with extra
#      wall blobs.
# g) Color inferability:
#    - No new color mapping is introduced; only the spatial criterion for where orange
#      appears changes.

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, random_free_location


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
    Paint a thinner inner border around the region reachable from the corner marker.

    Input structure:
    - The most common color is the background, and blue pixels are the wall-colored objects.
    - A unique one-cell magenta marker sits on a corner and marks the flood-fill start.
    - Border-touching wall components are the real barriers; interior wall fragments are decorations.

    Rule:
    - Flood from the marker through background and through non-structural wall decorations, but not through structural walls.
    - Remove wall-colored decorations that stay on the unreachable side of the structural barriers.
    - Paint orange only on reachable background cells whose in-bounds 4-neighbors include a structural wall or any unreachable cell.

    Key insight: The border is now defined strictly by orthogonal contact with blocked cells; diagonal wall touches and exposed outer edges no longer count.
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

    fill_col = 7
    for r, c in reach:
        if int(out[r, c]) != bg:
            continue
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and ((nr, nc) in walls or (nr, nc) not in reach):
                out[r, c] = fill_col
                break

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

NUM_DECOR_RANGE = (0, 3)
DECOR_AREA_RANGE = (1, 4)
DECOR_BORDER = 1
DECOR_PADDING = 1

MIN_SIGNAL_TOTAL = 3
MIN_EDGE_SIGNAL = 2
MIN_DIAG_SIGNAL = 1
MIN_FILL_CELLS = 8
MIN_INTERIOR_BG = 3

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
    m = _dilate4(mask)
    m[1:, 1:] |= mask[:-1, :-1]
    m[1:, :-1] |= mask[:-1, 1:]
    m[:-1, 1:] |= mask[1:, :-1]
    m[:-1, :-1] |= mask[1:, 1:]
    return m


def _edge_mask(h, w):
    edge = np.zeros((h, w), dtype=bool)
    edge[0, :] = True
    edge[-1, :] = True
    edge[:, 0] = True
    edge[:, -1] = True
    return edge


def _base_output(structural, marker_rc):
    out = np.full(structural.shape, BG, dtype=int)
    out[structural] = WALL
    out[marker_rc] = MARKER
    return out


def _fill_mask_variant(structural, reach, marker_rc):
    out = _base_output(structural, marker_rc)
    candidate = reach & (out == BG)
    near_blocked = _dilate4((~reach) | structural)
    return candidate & near_blocked


def _fill_mask_original(structural, reach, marker_rc):
    h, w = structural.shape
    out = _base_output(structural, marker_rc)
    candidate = reach & (out == BG)
    edge = _edge_mask(h, w)
    near_struct = _dilate8(structural)
    near_unreach = _dilate4(~reach)
    return candidate & (edge | near_struct | near_unreach)


def _paint_output(structural, reach, marker_rc):
    out = _base_output(structural, marker_rc)
    out[_fill_mask_variant(structural, reach, marker_rc)] = FILL
    return out


def _variant_signal(structural, reach, marker_rc):
    old_fill = _fill_mask_original(structural, reach, marker_rc)
    new_fill = _fill_mask_variant(structural, reach, marker_rc)
    removed = old_fill & ~new_fill
    edge_removed = removed & _edge_mask(*structural.shape)
    diag_removed = removed & _dilate8(structural) & ~_dilate4(~reach)
    return {
        "old_fill": old_fill,
        "new_fill": new_fill,
        "removed": removed,
        "removed_count": int(removed.sum()),
        "edge_removed_count": int(edge_removed.sum()),
        "diag_removed_count": int(diag_removed.sum()),
    }


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

    signal = _variant_signal(structural, reach, marker_rc)
    if signal["removed_count"] < MIN_SIGNAL_TOTAL:
        return None
    if signal["edge_removed_count"] < MIN_EDGE_SIGNAL:
        return None
    if signal["diag_removed_count"] < MIN_DIAG_SIGNAL:
        return None

    out = _paint_output(structural, reach, marker_rc)
    fill_mask = out == FILL
    if int(fill_mask.sum()) < MIN_FILL_CELLS:
        return None
    if int((reach & (out == BG)).sum()) < MIN_INTERIOR_BG:
        return None

    n_decor = rng.randint(*NUM_DECOR_RANGE)
    n_visible = rng.randint(0, n_decor)
    n_hidden = n_decor - n_visible

    visible_allowed = reach & (out == BG) & ~signal["old_fill"]
    hidden_allowed = (~reach) & (~structural)
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
    raise RuntimeError("failed to generate valid input for 8f3a5a89")
