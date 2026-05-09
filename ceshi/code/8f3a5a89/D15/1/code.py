# Variant of 8f3a5a89
# Dimension: D15 — 标记像素唯一性识别 (src)
# Change: 标记像素 = 在角落位置、非背景色、网格中只出现1次的像素... → 标记通过颜色唯一性识别（不限位置）

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


# -----------------------------------------------------------------------------
# Design analysis for the D15 variant
#
# source_task_baseline:
# - The requested source task is b99e7126, but within the allowed local files
#   only its evaluation JSON is available; no solver code is present to read.
#   The usable baseline extracted from those samples is the intended mechanism:
#   the special object is located by scanning for a globally unique color rather
#   than by relying on a fixed corner position.
#
# concretization_and_trace:
# - Original task: the magenta marker is forced to the bottom-left corner, so
#   solver logic can inspect only the 4 corners and stop once it finds a
#   non-background color whose total count is 1.
# - Variant: the marker is still a single pixel of a unique color, but it can
#   appear anywhere inside the marker-reachable region. The visible input now
#   contains a lone magenta pixel floating in the interior instead of being
#   attached to a corner.
# - Small execution trace:
#   original input fragment:
#     marker at (h-1, 0) -> corner scan finds it immediately -> flood starts at BL
#   variant input fragment:
#     marker at (7, 5), no special corner pixels -> corner scan would fail ->
#     new solver must count colors, find the only non-bg color with frequency 1,
#     then locate that pixel and start flood-fill there.
# - The rest of the wall/barrier logic is intentionally unchanged. Once the
#   start cell is identified, reachability, structural-wall detection, hidden
#   decoration removal, and orange inner-border painting behave exactly as in
#   the original task.
#
# complexity_calibration:
# - The change is visually obvious because the marker is no longer in any
#   corner. A human must scan the whole grid to find the lone magenta pixel.
# - The rule is still learnable from a few examples because there is only one
#   candidate unique color and the downstream output remains the same type of
#   region-bordering operation as the base task.
# - To avoid over-complication, the palette stays the same: bg=8, wall=1,
#   marker=6, fill=7.
#
# implicit_assumption_audit_for_main:
# - Old code assumption: "the only relevant unique pixel is on a corner".
#   This breaks immediately once the marker moves inward.
# - Old code loop over corners depended on fixed placement and would raise an
#   assertion on valid variant inputs. Therefore the loop is replaced by:
#   1. count all colors,
#   2. find the unique non-bg color,
#   3. locate its single occurrence anywhere in the grid.
# - Wall-color inference still depends on there being at least one additional
#   non-bg color besides the marker; the generator must still place wall pixels.
# - Flood start assumes the unique-color pixel lies in the passable region and
#   is not itself a wall. The generator guarantees this by placing the marker on
#   a reachable interior background cell.
#
# placement_legality:
# - Structural walls remain border-touching, 4-connected, and non-overlapping.
# - The marker is chosen only after the reachable region is known, from cells
#   that are reachable background interior: not on the grid edge, not adjacent
#   to structural walls, and not adjacent to unreachable space. This avoids
#   overlaps and prevents the marker from punching a hole into the orange border.
# - Hidden wall decorations remain non-border-touching so solver wall
#   re-identification cannot confuse them with structural walls.
#
# visual_observability_and_tidiness:
# - Marker positions are restricted to interior non-fill cells, so the magenta
#   pixel is cleanly separated from orange border pixels and blue walls.
# - The variant is not allowed to degenerate back into the original cue:
#   candidate marker cells are never on corners, and in practice never on any
#   edge because they are selected from the interior-only pool.
#
# color_change_inference:
# - No new color mapping is introduced. The only color-based inference is
#   "which non-background color occurs exactly once?" That is directly visible.
#
# new_main_assumptions_and_generator_guarantees:
# - main() now assumes exactly one non-background color has global count 1.
#   generate_input() enforces this by using only one marker pixel of color 6 and
#   keeping all wall/decor pixels color 1.
# - main() assumes the unique-color pixel is passable and belongs to the target
#   reachable region. generate_input() enforces this by placing it only on a
#   reachable interior background cell after structural walls are finalized.
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint an inner border around the region reachable from a uniquely colored marker pixel.

    Input structure:
    - The background is the most common color, while blue pixels form wall components that
      partition the board into reachable and sealed-off zones.
    - A single marker pixel of a globally unique non-background color appears somewhere in
      the reachable region; it is not constrained to a corner.
    - Some blue components are true border-linked barriers, while others are decorative
      fragments that should not survive if they sit behind those barriers.

    Rule:
    - Find the lone unique-color pixel anywhere in the grid and flood-fill outward through
      background cells to determine the marker's reachable side of the layout.
    - Treat only border-touching blue components adjacent to that initial reachable area as
      structural walls; hidden blue decorations behind them are erased.
    - Paint orange on reachable background cells that lie on the region's inner boundary:
      grid edge, next to a structural wall, or next to an unreachable cell.

    Key insight: The marker is no longer defined by position. Its color frequency isolates
    the correct flood-fill seed, after which the original barrier-vs-decoration logic applies.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    counts = Counter(int(v) for v in g.ravel())

    unique_cols = [col for col, n in counts.items() if col != bg and n == 1]
    assert len(unique_cols) == 1, "expected exactly one unique non-background marker color"
    marker_col = unique_cols[0]

    marker_locs = np.argwhere(g == marker_col)
    assert len(marker_locs) == 1, "marker color must occupy exactly one cell"
    mr, mc = (int(marker_locs[0, 0]), int(marker_locs[0, 1]))

    other_cols = [col for col in counts.keys() if col not in (bg, marker_col)]
    assert other_cols, "no non-background wall color found"
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
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            out[r, c] = fill_col
            continue
        if any((nr, nc) in walls for nr, nc in neighbors8(r, c)):
            out[r, c] = fill_col
            continue
        if any((nr, nc) not in reach for nr, nc in neighbors4(r, c)):
            out[r, c] = fill_col

    return out


# ---- Palette (task colors) ----
BG = 8
WALL = 1
MARKER = 6
FILL = 7

# ---- Canvas size ----
GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (10, 20)

# ---- Structural walls ----
NUM_STRUCTS_RANGE = (1, 4)
STRUCT_SHAPE_RETRIES = 8
RECT_MAX_DIM_DIV = 3
TRI_MAX_SIZE = 6
TRI_MIN_DIM_DIV = 3

# ---- Reachable region gate ----
REACH_RATIO_RANGE = (0.38, 0.87)

# ---- Decorations ----
NUM_DECOR_RANGE = (0, 3)
DECOR_AREA_RANGE = (1, 4)
DECOR_BORDER = 1
DECOR_PADDING = 1

# ---- Outer retry budget ----
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


def _compute_fill_mask(structural: np.ndarray, reach: np.ndarray) -> np.ndarray:
    candidate = reach & (~structural)

    edge = np.zeros_like(candidate)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True

    near_struct = _dilate8(structural)
    near_unreach = _dilate4(~reach)
    return candidate & (edge | near_struct | near_unreach)


def _place_structural_walls(h, w, rng):
    num = rng.randint(*NUM_STRUCTS_RANGE)
    structural = np.zeros((h, w), dtype=bool)
    placed = 0
    for _ in range(num):
        for _try in range(STRUCT_SHAPE_RETRIES):
            cells = rng.choice(SHAPE_FACTORIES)(h, w, rng)
            if not cells:
                continue
            if any(structural[r, c] for r, c in cells):
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


def _marker_candidates(reach: np.ndarray, structural: np.ndarray) -> np.ndarray:
    fill_mask = _compute_fill_mask(structural, reach)
    return reach & (~fill_mask)


def _paint_output(structural, reach, marker_rc):
    out = np.full(structural.shape, BG, dtype=int)
    out[structural] = WALL
    out[_compute_fill_mask(structural, reach)] = FILL
    out[marker_rc] = MARKER
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


def _gen_attempt():
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)

    structural = _place_structural_walls(h, w, rng)
    if structural is None or structural.all():
        return None

    free_cells = np.argwhere(~structural)
    if len(free_cells) == 0:
        return None

    seed_r, seed_c = map(int, free_cells[rng.randrange(len(free_cells))])
    reach = _flood_mask((seed_r, seed_c), structural, h, w)
    reach_frac = reach.mean()
    if not (REACH_RATIO_RANGE[0] <= reach_frac <= REACH_RATIO_RANGE[1]):
        return None

    if not _validate_structural(structural, reach, h, w):
        return None

    marker_pool = _marker_candidates(reach, structural)
    if not marker_pool.any():
        return None

    marker_choices = np.argwhere(marker_pool)
    mr, mc = map(int, marker_choices[rng.randrange(len(marker_choices))])
    marker_rc = (mr, mc)

    out = _paint_output(structural, reach, marker_rc)

    fill_mask = out == FILL
    if not fill_mask.any() or len(components_mask(fill_mask, connectivity=4)) != 1:
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
    raise RuntimeError("failed to generate valid input for 8f3a5a89 D15 variant")
