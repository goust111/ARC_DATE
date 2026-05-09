# Variant of 8f3a5a89
# Dimension: D13 — 边界涂色的邻接条件 (trajectory)
# Change: 三种条件的并集：(1) 位于网格边缘 (2) 8-邻接结构墙 (3) 4-邻接不可达像素... → 距离>=2的外环涂色

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


# -----------------------------------------------------------------------------
# Design analysis for the D13 variant
#
# source_task_status:
#   task: cb2d8a2c
#   solver_source: unavailable in the allowed workspace
#   usable_reference: official I/O json + annotation only
#   consequence: the distance-constraint baseline is implemented directly here
#     from the variant description instead of by copying a source solver pattern.
#
# a_concretization_and_trace:
#   original_behavior:
#     - Reachable background cells on the immediate inside of the boundary turn
#       orange. "Immediate" meant any of:
#         1. the cell itself lies on a grid edge
#         2. the cell is 8-neighbor-adjacent to a structural wall
#         3. the cell is 4-neighbor-adjacent to a non-reachable cell
#   new_behavior:
#     - Keep the same reachable region and the same structural/decoration split.
#     - First compute that original immediate boundary layer, but do not color it.
#     - Then color the next 4-connected layer inward from it.
#     - Result: the orange frame floats one pixel away from walls/outer edges,
#       leaving a visible background gap.
#   concrete_trace:
#     - If a reachable corridor starts at the left grid edge, the old rule colors
#       column 0. The new rule leaves column 0 as bg and colors column 1.
#     - If a reachable cell touches a wall diagonally/orthogonally enough to be
#       part of the old frontier, the new rule leaves that frontier cell bg and
#       colors its inward 4-neighbor layer instead.
#
# b_complexity_calibration:
#   - The task still has only one marker, one structural wall color, and one fill
#     color, so object recognition remains stable.
#   - The distance rule is visible because the output preserves a blank one-cell
#     moat between walls/edges and the orange frame.
#   - The generator rejects thin or fragmented layouts where the shifted frame
#     disappears or breaks into many pieces.
#
# c_old_implicit_assumptions_that_break:
#   - The original solver could color any reachable bg boundary cell immediately.
#     That assumption no longer holds; those cells must stay bg.
#   - The original generator allowed visible decorations anywhere in remaining bg.
#     Under the new rule, a visible decoration placed inside the one-cell gap or
#     on the shifted frame can hide the D13 effect or perturb the offset layer.
#     Therefore visible decorations are limited to deeper interior cells only.
#   - The original success condition only needed a one-cell border. The variant
#     additionally needs at least one inward layer beyond that border.
#
# d_placement_legality:
#   - Structural walls remain border-touching 4-connected components.
#   - Hidden decorations stay off the border so the solver never mistakes them
#     for structural walls.
#   - Visible decorations are placed only on cells whose inward distance from the
#     old frontier is at least 2, so they cannot overlap the moat or shifted ring.
#
# e_visual_observability:
#   - A human can compare input/output and see that orange is no longer flush with
#     the blue walls or the outer frame; a bg band is always left in between.
#   - The generator requires a non-empty shifted ring and rejects degenerate cases
#     where the new output would visually collapse to the old one.
#
# f_visual_cleanliness:
#   - The shifted ring is required to be a single 4-connected component.
#   - Decorations remain sparse and interior, so the ring and the gap remain legible.
#
# g_color_inference:
#   - No new colors or color remappings are introduced. The only behavioral change
#     is which reachable bg cells become orange.
#
# new_main_assumptions_checked_against_generator:
#   - The solver assumes there exists at least one old-frontier bg cell and at
#     least one inward layer reachable from that frontier; the generator verifies
#     the shifted ring is non-empty.
#   - The solver assumes visible decorations do not occupy the shifted ring; the
#     generator enforces placement only on deeper cells and then re-validates by
#     checking main(input) == intended_output.
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


def _compute_frontier_and_offset_fill(
    out: np.ndarray,
    reach: set[tuple[int, int]],
    walls: set[tuple[int, int]],
    bg: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the old immediate frontier, the shifted fill ring, and frontier distance."""
    h, w = out.shape
    reach_mask = np.zeros((h, w), dtype=bool)
    if reach:
        rr, cc = zip(*reach)
        reach_mask[rr, cc] = True

    candidate = reach_mask & (out == bg)
    frontier = np.zeros((h, w), dtype=bool)
    for r, c in reach:
        if not candidate[r, c]:
            continue
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            frontier[r, c] = True
            continue
        if any((nr, nc) in walls for nr, nc in neighbors8(r, c)):
            frontier[r, c] = True
            continue
        if any((nr, nc) not in reach for nr, nc in neighbors4(r, c)):
            frontier[r, c] = True

    dist = np.full((h, w), -1, dtype=int)
    q = deque()
    for r, c in np.argwhere(frontier):
        r = int(r)
        c = int(c)
        dist[r, c] = 0
        q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and candidate[nr, nc] and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))

    fill = candidate & (dist == 1)
    return frontier, fill, dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint a floating inner frame around the region reachable from a corner marker pixel.

    Input structure:
    - bg is the dominant color; blue pixels contain both true border-forming walls and
      smaller decorative fragments.
    - A unique magenta single pixel sits on one corner and marks the flood-fill start.
    - Border-touching wall components next to the marker-side open area are structural;
      non-structural wall fragments are decorations.

    Rule:
    - Flood-fill from the marker corner through background, treating every blue pixel as
      blocked, to find the marker-side open area.
    - Reclassify as structural only those blue components that touch the grid border and
      touch that first reachable area; other blue pixels are decorations.
    - Flood again while allowing traversal through non-structural blue decorations, erase
      decorations hidden behind structural walls, then locate the old immediate inner
      boundary and color the next 4-connected layer inward in orange.

    Key insight:
    - The real job is unchanged wall classification; the D13 variant only shifts the
      painted border one cell inward from the original boundary layer, leaving a blank moat.
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

    for r in range(h):
        for c in range(w):
            if int(out[r, c]) == wall_col and (r, c) not in walls and (r, c) not in reach:
                out[r, c] = bg

    _, fill, _ = _compute_frontier_and_offset_fill(out, reach, walls, bg)
    out[fill] = 7
    return out


# --------------- generator (D13 distance-offset variant) ---------------

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

GEN_ATTEMPTS = 700


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
    h, w = structural.shape
    out = np.full((h, w), BG, dtype=int)
    out[structural] = WALL
    out[marker_rc] = MARKER

    reach_set = set((int(r), int(c)) for r, c in np.argwhere(reach))
    wall_set = set((int(r), int(c)) for r, c in np.argwhere(structural))
    frontier, fill, dist = _compute_frontier_and_offset_fill(out, reach_set, wall_set, BG)
    out[fill] = FILL
    return out, frontier, fill, dist


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

    structural = _place_structural_walls(h, w, marker_rc, rng)
    if structural is None or structural[marker_rc]:
        return None

    reach = _flood_mask(marker_rc, structural, h, w)
    reach_frac = reach.mean()
    if not (REACH_RATIO_RANGE[0] <= reach_frac <= REACH_RATIO_RANGE[1]):
        return None

    if not _validate_structural(structural, reach, h, w):
        return None

    out, frontier, fill, dist = _paint_output(structural, reach, marker_rc)
    fill_mask = out == FILL
    if not fill_mask.any() or len(components_mask(fill_mask, connectivity=4)) != 1:
        return None

    frontier_size = int(frontier.sum())
    fill_size = int(fill.sum())
    if frontier_size < 4 or fill_size < 4:
        return None

    deeper_interior = dist >= 2
    if not deeper_interior.any():
        return None

    n_decor = rng.randint(*NUM_DECOR_RANGE)
    n_visible = rng.randint(0, n_decor)
    n_hidden = n_decor - n_visible

    visible_allowed = deeper_interior & (out == BG)
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

    solved = main(inp)
    if not np.array_equal(solved, out_final):
        return None

    old_style_fill = frontier & (inp == BG)
    if not old_style_fill.any():
        return None
    if np.array_equal(old_style_fill, solved == FILL):
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
    raise RuntimeError("failed to generate valid input for 8f3a5a89 D13 variant")
