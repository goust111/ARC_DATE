# Variant of 8f3a5a89
# Dimension: D8 — 洪填连通性 (path)
# Change: 4-连通洪填（只通过上下左右邻居扩展），8-邻接用于判断结构墙附近的边界像素... → 8-连通洪填

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location

"""
Design analysis
===============

variant:
  source_task: none
  preserved_from_original:
    - colors stay bg=8, wall=1, marker=6, fill=7
    - the marker is still the unique single-pixel non-bg corner marker at bottom-left
    - structural walls are still recovered as 4-connected wall-color components that touch a border
    - the orange border test still uses 8-nearness to structural walls and 4-nearness to unreachable cells
  changed_core:
    - both marker-side flood-fills switch from 4-neighbor expansion to 8-neighbor expansion
    - diagonal gaps between separate wall components no longer block reachability

concrete_behavior:
  input:
    - a border-touching blue layout still partitions the board
    - one mandatory structural motif is now a "leaky corner":
      a top-attached vertical wall stops one cell above a right-attached horizontal wall,
      so the two walls miss each other by a single diagonal step
    - an additional orthogonal L-wall creates a genuinely sealed pocket, keeping a visible unreachable side
    - optional small blue decoration patches may appear in reachable background or in the sealed pocket
  output:
    - reachable bg cells get orange on the inner rim
    - reachable now includes the region beyond the diagonal gap, so orange can extend into an upper-right area
      that the old 4-connected rule would have left untouched
    - hidden decorations in the sealed pocket are erased back to bg

behavior_trace:
  toy_example:
    grid: 8x8
    leaky_walls:
      - vertical at col 3, rows 0..3
      - horizontal at row 4, cols 4..7
    start: (7, 0)
    old_4_conn:
      - from (4,3) the flood cannot step to (3,4) because they touch only diagonally
      - the entire upper-right block stays unreachable
    new_8_conn:
      - the flood can step diagonally across that corner gap
      - the upper-right block becomes reachable and receives orange border cells in the output
    observable_difference:
      - same input, but new output paints a substantial top/right-side region that old logic leaves azure

complexity_calibration:
  - only the path-connectivity dimension changes; wall components remain simple and visually legible
  - the generator forces a non-trivial diagonal leak, so the variant does not collapse back to the original task
  - one sealed pocket remains, preventing the board from becoming "almost fully reachable" and preserving the
    original hidden-decoration/erasure behavior

implicit_assumption_review:
  main_related:
    - old _flood assumed cardinal expansion; this is invalid under D8 and must change to neighbors8
    - structural wall extraction used the first flood only to determine which border-touching wall components
      face the marker side; this logic can remain 4-adjacent because the dimension change concerns travel,
      not wall-component identity
    - border painting originally mixed 8-near-wall and 4-near-unreachable tests; these remain unchanged
      because D8 changes reachability, not the definition of the painted rim
    - any generator sample that lacks a diagonal-only passage would silently behave like the original task,
      so generation must explicitly test against the old 4-connected solver and reject weak samples

placement_legality:
  - structural pieces are placed by formula, so they never overlap the marker and never go out of bounds
  - the sealed pocket wall is kept strictly left/above the leaky corner wall, avoiding accidental merges
  - decoration masks exclude the border and the 4-neighborhood of structural walls, so decoration components
    cannot be misread as extra structural barriers by the solver

visual_observability:
  - the diagonal seep is readable from a few examples because the leak always occurs at a visible one-cell
    corner gap and opens a sizable region on the far side
  - the generator enforces both reach-area gain and output-pixel differences versus the old 4-connected solver
  - colors are unchanged and no hidden numeric ordering is used

visual_cleanliness:
  - only three semantic layers coexist in one place: blue walls, azure background, orange border
  - decorations are small and separated from structural walls, so the leaky corner remains easy to spot
"""


# ---- Palette (task colors) ----
BG = 8
WALL = 1
MARKER = 6
FILL = 7

# ---- Canvas size ----
GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (10, 20)

# ---- Reachable region gate ----
REACH_RATIO_RANGE = (0.38, 0.87)

# ---- Decorations ----
NUM_VISIBLE_DECOR_RANGE = (0, 2)
NUM_HIDDEN_DECOR_RANGE = (0, 1)
VISIBLE_DECOR_AREA_RANGE = (1, 4)
HIDDEN_DECOR_AREA_RANGE = (1, 2)
DECOR_BORDER = 1
DECOR_PADDING = 1

# ---- Variant-signal gate ----
MIN_DIAG_LEAK_AREA = 8
MIN_OUTPUT_DIFF = 8

# ---- Outer retry budget ----
GEN_ATTEMPTS = 300


def _flood(start: tuple[int, int], passable, h: int, w: int, connectivity: int) -> set[tuple[int, int]]:
    neigh = neighbors8 if connectivity == 8 else neighbors4
    q = deque([start])
    vis = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in neigh(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and passable(nr, nc):
                vis.add((nr, nc))
                q.append((nr, nc))
    return vis


def _solve(grid: np.ndarray, connectivity: int) -> np.ndarray:
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
        connectivity=connectivity,
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
        connectivity=connectivity,
    )

    for r in range(h):
        for c in range(w):
            if int(out[r, c]) == wall_col and (r, c) not in walls and (r, c) not in reach:
                out[r, c] = bg

    for r, c in reach:
        if int(out[r, c]) != bg:
            continue
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            out[r, c] = FILL
            continue
        if any((nr, nc) in walls for nr, nc in neighbors8(r, c)):
            out[r, c] = FILL
            continue
        if any((nr, nc) not in reach for nr, nc in neighbors4(r, c)):
            out[r, c] = FILL

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the marker-side interior and paint its inner rim, with diagonal travel now counting as connected.

    Input structure:
    - The most common color is the background, and blue wall-colored pixels divide the board.
    - A unique single magenta pixel on a corner marks the flood-fill start.
    - Border-touching blue components are candidate structural walls, while smaller blue patches can be decorations.

    Rule:
    - Flood-fill from the corner marker using 8-neighbor movement, so the reachable side may slip through diagonal gaps.
    - Treat border-touching wall components that face that marker-side reach as the real barriers; erase unreachable
      decoration walls hidden behind them.
    - Paint orange on every reachable background cell that lies on the grid edge, touches a structural wall in the
      8-neighborhood, or sits 4-neighbor-adjacent to an unreachable cell.

    Key insight: The only real rule change is the travel connectivity: diagonal one-cell cracks are now passable, so
    walls must close those corners too if they want to block the marker-side region.
    """
    return _solve(grid, connectivity=8)


def _flood_mask(start, blocked, h, w, connectivity):
    reach = np.zeros((h, w), dtype=bool)
    reach[start] = True
    neigh = neighbors8 if connectivity == 8 else neighbors4
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in neigh(r, c):
            if 0 <= nr < h and 0 <= nc < w and not reach[nr, nc] and not blocked[nr, nc]:
                reach[nr, nc] = True
                q.append((nr, nc))
    return reach


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


def _paint_output(structural: np.ndarray, reach: np.ndarray, marker_rc: tuple[int, int]) -> np.ndarray:
    h, w = structural.shape
    out = np.full((h, w), BG, dtype=int)
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


def _validate_structural(structural: np.ndarray, reach: np.ndarray, h: int, w: int) -> bool:
    reach_nbr = _dilate4(reach)
    for comp_arr in components_mask(structural, connectivity=4):
        cmask = np.zeros((h, w), dtype=bool)
        cmask[comp_arr[:, 0], comp_arr[:, 1]] = True
        if not _touches_border(cmask):
            return False
        if not (cmask & reach_nbr).any():
            return False
    return True


def _place_decor(allowed_mask: np.ndarray, count: int, rng, area_range: tuple[int, int]) -> np.ndarray | None:
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


def _build_structural_layout(h: int, w: int, rng) -> np.ndarray:
    structural = np.zeros((h, w), dtype=bool)

    gap_r = rng.randint(max(4, h // 3), h - 4)
    gap_c = rng.randint(max(4, w // 3), w - 4)
    pocket_h = rng.randint(3, gap_r - 1)
    pocket_w = rng.randint(3, gap_c - 1)

    structural[: gap_r + 1, gap_c] = True
    structural[gap_r + 1, gap_c + 1 :] = True

    structural[: pocket_h + 1, pocket_w] = True
    structural[pocket_h, : pocket_w + 1] = True

    return structural


def _gen_attempt():
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    marker_rc = (h - 1, 0)

    structural = _build_structural_layout(h, w, rng)
    if structural[marker_rc]:
        return None

    reach8 = _flood_mask(marker_rc, structural, h, w, connectivity=8)
    reach4 = _flood_mask(marker_rc, structural, h, w, connectivity=4)
    leak_gain = reach8 & ~reach4

    if not reach8[0, w - 1] or reach4[0, w - 1]:
        return None
    if int(leak_gain.sum()) < MIN_DIAG_LEAK_AREA:
        return None

    reach_frac = float(reach8.mean())
    if not (REACH_RATIO_RANGE[0] <= reach_frac <= REACH_RATIO_RANGE[1]):
        return None

    if not _validate_structural(structural, reach8, h, w):
        return None

    out = _paint_output(structural, reach8, marker_rc)
    fill_mask = out == FILL
    if not fill_mask.any() or len(components_mask(fill_mask, connectivity=8)) != 1:
        return None

    structural_buffer = _dilate4(structural)
    visible_allowed = reach8 & (out == BG) & ~structural_buffer
    hidden_allowed = (~reach8) & (~structural) & ~structural_buffer
    hidden_allowed[0, :] = hidden_allowed[-1, :] = False
    hidden_allowed[:, 0] = hidden_allowed[:, -1] = False

    n_visible = rng.randint(*NUM_VISIBLE_DECOR_RANGE)
    n_hidden = rng.randint(0, NUM_HIDDEN_DECOR_RANGE[1] if hidden_allowed.any() else 0)

    visible_placed = _place_decor(visible_allowed, n_visible, rng, VISIBLE_DECOR_AREA_RANGE)
    if visible_placed is None:
        return None
    hidden_placed = _place_decor(hidden_allowed, n_hidden, rng, HIDDEN_DECOR_AREA_RANGE)
    if hidden_placed is None:
        return None

    out_final = out.copy()
    out_final[visible_placed] = WALL

    inp = out_final.copy()
    inp[inp == FILL] = BG
    inp[hidden_placed] = WALL

    if not np.array_equal(main(inp), out_final):
        return None

    old_out = _solve(inp, connectivity=4)
    if int(np.count_nonzero(old_out != out_final)) < MIN_OUTPUT_DIFF:
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
    raise RuntimeError("failed to generate valid input for 8f3a5a89 D8 variant")
