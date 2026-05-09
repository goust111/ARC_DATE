# Variant of 8f3a5a89
# Dimension: D13 — 边界涂色的邻接条件 (trajectory)
# Change: 三种条件的并集：(1) 位于网格边缘 (2) 8-邻接结构墙 (3) 4-邻接不可达像素... → 仅邻接结构墙的像素涂色
#
# Design analysis
# a) Concrete behavior trace:
#    - The input still contains azure background, blue wall pixels, and one unique
#      magenta marker in the bottom-left corner. Border-touching blue components
#      that touch the marker-reachable side remain the true barriers; interior blue
#      blobs remain decorations.
#    - The original task painted orange on all reachable background cells lying on
#      the reachable-region boundary, which usually colored long stretches of the
#      left/bottom grid edge. In this variant, orange is only the 8-neighborhood
#      halo of structural walls, so open border corridors stay azure unless they
#      also touch a wall.
#    - Example: if a vertical wall at column 4 splits the grid, the old output
#      colored the left edge corridor and the cells beside the wall; the new output
#      colors only the cells beside the wall. The hidden side decorations are still
#      erased exactly as before.
# b) Complexity calibration:
#    - The barrier-generation logic is unchanged so samples still show 1-4 distinct
#      border-attached structures and optional decorations.
#    - To make D13 visible, the generator rejects layouts where the new output would
#      look too similar to the old one. It requires several reachable edge cells
#      that the old rule would color but the new rule leaves as background.
# c) Implicit-assumption audit:
#    - Solver lines that identify marker color, wall color, structural walls, and
#      hidden decorations do not rely on the old border-paint rule and stay valid.
#    - The old generator relied on edge cells being turned orange, which implicitly
#      prevented visible decorations from being placed on the border. Under D13,
#      those cells can remain background, so visible decorations must now be banned
#      from the border or the solver would misread them as structural walls.
#    - The old "single fill ring" invariant no longer holds in general because
#      isolated walls can produce separate orange halos. The generator therefore
#      validates only that the halo is non-empty and visually meaningful.
# d) Placement legality:
#    - Structural components remain non-overlapping, border-touching, and adjacent
#      to the reachable side so the solver can re-identify them.
#    - Hidden decorations remain off-border and outside reach; visible decorations
#      remain inside reach and now also off-border, so they cannot masquerade as
#      structural walls.
# e) Visual observability:
#    - Across examples, the strongest cue is that the reachable outer border often
#      stays azure while orange hugs blue structures. This is easy to see in a few
#      demonstrations and does not depend on invisible metadata such as color ids.
# f) Visual cleanliness:
#    - Only four colors appear in the puzzle. Orange is limited to a one-cell halo
#      next to true walls, avoiding heavy overlap with decorations.
# g) Color-change inference:
#    - No new color mapping is introduced. The only changed phenomenon is where the
#      existing orange color appears, and its placement is explained by adjacency to
#      blue structural walls.
# h) New solver assumptions and generator guarantees:
#    - `main()` assumes at least one reachable background cell is 8-neighbor to a
#      structural wall, otherwise the output would have no orange. The generator
#      guarantees this by requiring a non-empty halo.
#    - `main()` assumes border-touching non-structural walls do not exist. The
#      generator now explicitly enforces that for both hidden and visible decor.

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


def _edge_mask(h: int, w: int) -> np.ndarray:
    edge = np.zeros((h, w), dtype=bool)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True
    return edge


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


def _candidate_fill_mask(structural: np.ndarray, reach: np.ndarray, marker_rc: tuple[int, int]) -> np.ndarray:
    candidate = reach & (~structural)
    candidate[marker_rc] = False
    return candidate


def _variant_fill_mask(structural: np.ndarray, reach: np.ndarray, marker_rc: tuple[int, int]) -> np.ndarray:
    candidate = _candidate_fill_mask(structural, reach, marker_rc)
    near_struct = _dilate8(structural)
    return candidate & near_struct


def _original_fill_mask(structural: np.ndarray, reach: np.ndarray, marker_rc: tuple[int, int]) -> np.ndarray:
    candidate = _candidate_fill_mask(structural, reach, marker_rc)
    near_struct = _dilate8(structural)
    near_unreach = _dilate4(~reach)
    return candidate & (_edge_mask(*structural.shape) | near_struct | near_unreach)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint only the wall-adjacent halo inside the region reached from a corner marker.

    Input structure:
    - The most frequent color is the background, while blue pixels act either as true
      border-connected barriers or as decorative fragments.
    - A unique single magenta pixel sits on one corner and marks the flood-fill start.
    - Structural walls are exactly the blue components that both touch the frame and
      meet the marker-side reachable area; other blue pieces are decorations.

    Rule:
    - Flood from the marker, first treating every blue pixel as blocked to identify
      which border-touching blue components are real walls.
    - Flood again while allowing traversal through non-structural blue decorations,
      then erase decoration pixels that remain hidden behind the structural side.
    - Paint orange only on reachable background cells that are 8-neighbor to a
      structural wall, producing a wall-hugging halo instead of a full boundary ring.

    Key insight: The orange pattern is driven solely by proximity to the true
    structural wall set; reachable edges and other reachable-boundary cells are not
    colored unless a structural wall is beside them.
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

    wall_mask = np.zeros((h, w), dtype=bool)
    for r, c in walls:
        wall_mask[r, c] = True
    reach_mask = np.zeros((h, w), dtype=bool)
    for r, c in reach:
        reach_mask[r, c] = True

    fill_mask = _variant_fill_mask(wall_mask, reach_mask, (mr, mc))
    out[fill_mask & (out == bg)] = 7
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

GEN_ATTEMPTS = 500
MIN_FILL_PIXELS = 4
MIN_OLD_ONLY_EDGE_PIXELS = 4


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
    out[_variant_fill_mask(structural, reach, marker_rc)] = FILL
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


def _variant_gap_metrics(structural, reach, marker_rc):
    new_fill = _variant_fill_mask(structural, reach, marker_rc)
    old_fill = _original_fill_mask(structural, reach, marker_rc)
    old_only = old_fill & ~new_fill
    edge_only = old_only & _edge_mask(*structural.shape)
    return {
        "new_fill": int(new_fill.sum()),
        "old_only": int(old_only.sum()),
        "edge_only": int(edge_only.sum()),
    }


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

    metrics = _variant_gap_metrics(structural, reach, marker_rc)
    if metrics["new_fill"] < MIN_FILL_PIXELS or metrics["edge_only"] < MIN_OLD_ONLY_EDGE_PIXELS:
        return None

    out = _paint_output(structural, reach, marker_rc)

    n_decor = rng.randint(*NUM_DECOR_RANGE)
    n_visible = rng.randint(0, n_decor)
    n_hidden = n_decor - n_visible

    visible_allowed = reach & (out == BG)
    visible_allowed[_edge_mask(h, w)] = False

    hidden_allowed = (~reach) & (~structural)
    hidden_allowed[_edge_mask(h, w)] = False

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
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 8f3a5a89 D13 variant")
