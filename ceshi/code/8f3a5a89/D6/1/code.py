# Variant of 8f3a5a89
# Dimension: D6 — 装饰墙处理规则 (transform)
# Change: 可达区域外的装饰墙擦除为背景色，可达区域内的装饰墙保留为墙色
#   -> 装饰墙变色而非擦除：不可达侧的装饰墙改为绿色标记，而不是消失
#
# Design analysis
# ===============
# variant_spec:
#   focus_dimension: D6 only. Structural-wall detection, reachable-region flood-fill,
#     marker semantics, and orange inner-border painting all stay unchanged.
#   visible_change:
#     - Input still uses a single wall color (blue=1) for both structural walls and decorations.
#     - Output still keeps structural walls blue and still paints the reachable-region inner border
#       orange=7.
#     - The only changed cells are blue decoration pixels that sit on the unreachable side of the
#       structural barrier: old task changed them to background 8, this variant recolors them to
#       green 3 so they remain visible as "flagged hidden decorations".
#   concrete_trace:
#     - Suppose a 2x2 blue decor block sits behind a border-touching blue barrier.
#     - Original flow: first flood cannot reach that pocket -> decor is classified as non-structural
#       and unreachable -> output writes those four cells to bg 8.
#     - Variant flow: the same classification is reused, but the write target becomes green 3.
#     - Difference: the unreachable side now contains an explicit recolored residue instead of empty
#       background, while reachable blue decorations and structural blue walls are unchanged.
# complexity_calibration:
#   - Only one extra output color is introduced, and it has a single role with repeated support
#     across examples: unreachable decorations.
#   - Input colors are unchanged, so the main inference burden stays near the original task; the new
#     reasoning step is a single conditional recolor based on reachability side.
#   - Generator forces at least one hidden decoration patch per sample, and hidden patches have area
#     >= 2 so the D6 change is visible instead of degenerating into the original task.
# implicit_assumptions_review:
#   - Marker detection still assumes a unique non-background corner pixel; generator keeps the
#     magenta marker as the only singleton corner color.
#   - Wall-color inference still assumes the most frequent non-bg, non-marker color is the wall
#     color. This remains valid because the input never contains the green recolor.
#   - Structural-wall recovery still depends on "border-touching and adjacent to reach0"; unchanged.
#   - The former unconditional overwrite to background for unreachable decorations is the only line
#     whose old invariant changed. It now writes green, which is safe because those cells are
#     guaranteed to be non-structural and outside the reachable set.
# placement_legality:
#   - Hidden decorations are placed only in cells that are unreachable when only structural walls are
#     blocked, so the solver will classify them as hidden even though they are wall-colored in input.
#   - Hidden decorations are prevented from touching the outer border to avoid being visually
#     confusable with structural border-connected walls.
#   - Visible decorations are placed only on reachable background cells that are not already orange
#     border cells in the target output.
# observability_and_cleanliness:
#   - Every generated sample includes at least one hidden decoration patch, so green recoloring is
#     always present in the output.
#   - Hidden patches use the same geometry family as the original decorations and stay small, keeping
#     the scene readable without adding new object types.
#   - Green appears only behind structural barriers, which provides a strong spatial cue for the
#     recolor rule.
# main_assumptions_enforced_by_generator:
#   - There is at least one structural wall component and each structural component touches a border
#     and is 4-adjacent to the marker-reachable region.
#   - The orange border remains a single 4-connected band.
#   - At least one hidden decoration patch exists, and its cells become green under the new solver.

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


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
    Paint the reachable region's inner border and recolor hidden wall decorations.

    Input structure:
    - The dominant background color fills most cells, while blue wall pixels form both true barriers
      and small decorative fragments.
    - A unique magenta pixel is anchored on one corner and marks the flood-fill start side.
    - Border-touching blue wall components can act as structural separators; non-structural blue
      pieces are decorations that may lie either on the reachable side or behind the barrier.

    Rule:
    - Flood from the marker corner, first treating all blue pixels as blocked to identify which
      border-touching blue components are genuine structural walls.
    - Flood again while allowing passage through non-structural blue decorations, so only the true
      structural walls split reachable from unreachable space.
    - Recolor blue decorations on the unreachable side to green, keep reachable-side decorations
      blue, and paint orange on reachable background cells that form the 1-cell inner boundary.

    Key insight: The puzzle still hinges on separating structural blue walls from decorative blue
    fragments, but hidden decorations are now marked by a color change instead of being erased.
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

    hidden_col = 3
    for r in range(h):
        for c in range(w):
            if int(out[r, c]) == wall_col and (r, c) not in walls and (r, c) not in reach:
                out[r, c] = hidden_col

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

    return out


BG = 8
WALL = 1
MARKER = 6
FILL = 7
HIDDEN_DECOR = 3

GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (10, 20)

NUM_STRUCTS_RANGE = (1, 4)
STRUCT_SHAPE_RETRIES = 8
RECT_MAX_DIM_DIV = 3
TRI_MAX_SIZE = 6
TRI_MIN_DIM_DIV = 3

REACH_RATIO_RANGE = (0.38, 0.87)

NUM_VISIBLE_DECOR_RANGE = (0, 2)
NUM_HIDDEN_DECOR_RANGE = (1, 1)
VISIBLE_DECOR_AREA_RANGE = (1, 4)
HIDDEN_DECOR_AREA_RANGE = (2, 4)
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


def _paint_output(structural, reach, marker_rc):
    out = np.full(structural.shape, BG, dtype=int)
    out[structural] = WALL
    out[marker_rc] = MARKER

    candidate = reach & (out == BG)

    edge = np.zeros_like(candidate)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True

    near_struct = _dilate8(structural)
    near_unreach = _dilate4(~reach)

    fill = candidate & (edge | near_struct | near_unreach)
    out[fill] = FILL
    return out


def _place_decor(allowed_mask, count, rng, area_range):
    h, w = allowed_mask.shape
    placed = np.zeros((h, w), dtype=bool)
    if count == 0:
        return placed
    for _ in range(count):
        area = rng.randint(*area_range)
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

    out = _paint_output(structural, reach, marker_rc)

    fill_mask = out == FILL
    if not fill_mask.any() or len(components_mask(fill_mask, connectivity=4)) != 1:
        return None

    n_visible = rng.randint(*NUM_VISIBLE_DECOR_RANGE)
    n_hidden = rng.randint(*NUM_HIDDEN_DECOR_RANGE)

    visible_allowed = reach & (out == BG)
    hidden_allowed = (~reach) & (~structural)
    hidden_allowed[0, :] = False
    hidden_allowed[-1, :] = False
    hidden_allowed[:, 0] = False
    hidden_allowed[:, -1] = False

    visible_placed = _place_decor(visible_allowed, n_visible, rng, VISIBLE_DECOR_AREA_RANGE)
    if visible_placed is None:
        return None
    hidden_placed = _place_decor(hidden_allowed, n_hidden, rng, HIDDEN_DECOR_AREA_RANGE)
    if hidden_placed is None or not hidden_placed.any():
        return None

    out_final = out.copy()
    out_final[visible_placed] = WALL
    out_final[hidden_placed] = HIDDEN_DECOR

    inp = out_final.copy()
    inp[inp == FILL] = BG
    inp[inp == HIDDEN_DECOR] = WALL

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
