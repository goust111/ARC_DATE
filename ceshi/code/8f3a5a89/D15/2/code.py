# Variant of 8f3a5a89
# Dimension: D15 — 标记像素唯一性识别 (src)
# Change: 标记像素 = 在角落位置、非背景色、网格中只出现1次的像素... → 多个同色像素作为标记组

from __future__ import annotations

from collections import Counter, deque
from itertools import combinations
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


# Design analysis for the D15 variant
# -----------------------------------
# a) Concrete behavior and execution trace
#    - Visual form: the marker is now a border-only group of 2..4 same-colored pixels.
#      They may sit on different edges and, critically, in different wall-separated
#      open pockets. Structural walls remain blue edge-touching components; background
#      is azure; output still paints orange on the inner boundary of the region that is
#      reachable from the marker side.
#    - Small trace:
#      Original: one magenta pixel at bottom-left -> single-source flood reaches only
#      that pocket -> orange border is painted only around that one reachable basin.
#      Variant: three magenta pixels on three border segments -> multi-source flood
#      starts from all three cells -> the reachable set is the union of the three open
#      pockets -> orange appears around every selected pocket and blue decorations on
#      pockets with no marker remain erasable.
#    - The changed step is the reach computation. Input construction must therefore
#      create several border-touching free-space components and assign markers to at
#      least two of them, otherwise the new rule would collapse back to the old one.
# b) Complexity calibration
#    - The variant is kept narrow: only the marker identification and flood source
#      multiplicity change. Colors, wall semantics, decoration semantics, and border
#      painting all stay the same.
#    - To make the change visible, the generator requires at least two distinct
#      marker-hosting free-space components. This avoids the trivial case where extra
#      markers live inside one already-connected region and have no effect.
#    - Marker count is capped at 4 so the signal stays readable rather than noisy.
# c) Implicit-assumption audit of the original main()
#    - Original corner scan assumed exactly one useful marker pixel and that its color
#      appears once globally. That fails immediately when the marker repeats.
#    - Original _flood accepted one start coordinate. The variant needs an iterable of
#      seeds; otherwise only one marker would be honored and the generator would lie.
#    - Original wall-color inference excluded one marker color by identity, which still
#      works once the whole marker group shares one color; the solver now identifies the
#      marker color as the only non-background color whose every pixel lies on the grid
#      border and whose count is at least two.
#    - Original fill painting skipped non-background cells, so keeping several markers is
#      safe; no extra overwrite guard is needed beyond placing markers before fill.
# d) Placement legality
#    - Marker cells are sampled from border cells of selected free-space components, so
#      they never overlap walls and are always passable seeds.
#    - Hidden decorations stay away from the border so they cannot be misread as
#      structural walls. Decoration placement still uses padded random_free_location,
#      which also prevents overlap and accidental merges.
#    - Structural validation is tightened to the selected multi-source reach: every wall
#      component must touch a border and be adjacent to at least one selected reachable
#      region, matching the solver's structural-wall test.
# e) Visual observability
#    - A human can now see several same-colored border markers and corresponding orange
#      borders around several separated pockets. This is a visible rule change from the
#      original single corner dot.
#    - The generator prefers non-corner border cells for markers when possible so the
#      samples do not look like the original task with accidental duplicates.
# f) Visual cleanliness
#    - Each marker is exactly one cell; the number of markers is small; wall/decor color
#      reuse is unchanged. This keeps the grid legible even when multiple pockets are
#      active at once.
# g) Color inference
#    - No new color mapping is introduced. The new inference cue is purely geometric:
#      "the repeated border-only color marks all sources".


def _flood(starts, passable, h: int, w: int) -> set[tuple[int, int]]:
    seeds = [(int(r), int(c)) for r, c in starts if 0 <= r < h and 0 <= c < w and passable(r, c)]
    assert seeds, "no valid flood-fill seeds found"
    q = deque(seeds)
    vis = set(seeds)
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and passable(nr, nc):
                vis.add((nr, nc))
                q.append((nr, nc))
    return vis


def _on_border(r: int, c: int, h: int, w: int) -> bool:
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _is_corner(r: int, c: int, h: int, w: int) -> bool:
    return (r, c) in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint an inner border around every region claimed by a shared border marker group.

    Input structure:
    - The background is the dominant color, while blue wall components still split the grid into pockets.
    - The start cue is a repeated non-background color whose pixels all lie on the outer border; these cells form one marker group.
    - Wall-colored pixels can be either structural barriers or decorative fragments, just as in the original task.

    Rule:
    - Detect the border-only marker color, then launch one flood-fill from all of its pixels at once.
    - Re-identify the structural blue walls as the border-touching wall components adjacent to that multi-source reachable union.
    - Delete wall-colored decorations on the unreachable side and paint orange on reachable background cells that lie on the grid edge, touch a structural wall diagonally/orthogonally, or border an unreached cell.

    Key insight:
    - The important change is not the wall logic but the source logic: all repeated border markers with the same color act together, so the target region is the union of several wall-separated pockets rather than the reach from one special corner pixel.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    counts = Counter(int(v) for v in g.ravel())

    marker_candidates: list[tuple[int, list[tuple[int, int]]]] = []
    for col in counts:
        if col == bg or counts[col] < 2:
            continue
        coords = [(int(r), int(c)) for r, c in np.argwhere(g == col)]
        if all(_on_border(r, c, h, w) for r, c in coords):
            marker_candidates.append((int(col), coords))
    assert len(marker_candidates) == 1, "expected exactly one repeated border-only marker color"

    marker_col, markers = marker_candidates[0]

    other_cols = [c for c in counts.keys() if c not in (bg, marker_col)]
    assert other_cols, "no non-background non-marker colors found; cannot determine wall color"
    wall_col = max(other_cols, key=lambda c: counts[c])

    reach0 = _flood(
        markers,
        lambda r, c: int(g[r, c]) in (bg, marker_col),
        h,
        w,
    )

    walls: set[tuple[int, int]] = set()
    for comp_arr in components_mask(g == wall_col, connectivity=4):
        comp = [(int(r), int(c)) for r, c in comp_arr]
        touches_border = any(_on_border(r, c, h, w) for r, c in comp)
        if not touches_border:
            continue
        adjacent_to_reach0 = any(any((nr, nc) in reach0 for nr, nc in neighbors4(r, c)) for r, c in comp)
        if adjacent_to_reach0:
            walls |= set(comp)

    reach = _flood(
        markers,
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
        if _on_border(r, c, h, w):
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

# ---- Markers ----
NUM_MARKERS_RANGE = (2, 4)

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


def _flood_mask(starts, blocked, h, w):
    reach = np.zeros((h, w), dtype=bool)
    seeds = [(int(r), int(c)) for r, c in starts if 0 <= r < h and 0 <= c < w and not blocked[r, c]]
    if not seeds:
        return reach
    q = deque(seeds)
    for r, c in seeds:
        reach[r, c] = True
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


def _paint_output(structural, reach, markers):
    h, w = structural.shape
    out = np.full((h, w), BG, dtype=int)
    out[structural] = WALL
    for r, c in markers:
        out[r, c] = MARKER

    candidate = reach & (out == BG)

    edge = np.zeros_like(candidate)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True

    near_struct = _dilate8(structural)
    near_unreach = _dilate4(~reach)

    fill = candidate & (edge | near_struct | near_unreach)
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
            placed[r0:r0 + ph, c0:c0 + pw] = True
            break
        else:
            return None
    return placed


def _border_components(mask):
    h, w = mask.shape
    infos = []
    for comp_arr in components_mask(mask, connectivity=4):
        coords = [(int(r), int(c)) for r, c in comp_arr]
        border_cells = [(r, c) for r, c in coords if _on_border(r, c, h, w)]
        if not border_cells:
            continue
        cmask = np.zeros((h, w), dtype=bool)
        cmask[comp_arr[:, 0], comp_arr[:, 1]] = True
        infos.append({"mask": cmask, "border_cells": border_cells, "size": len(coords)})
    return infos


def _choose_marker_cells(comp_infos, idxs, h, w, rng):
    markers = []
    for idx in idxs:
        border_cells = comp_infos[idx]["border_cells"]
        preferred = [(r, c) for r, c in border_cells if not _is_corner(r, c, h, w)]
        pool = preferred if preferred else border_cells
        markers.append(tuple(rng.choice(pool)))
    return markers


def _select_marker_layout(structural, rng):
    h, w = structural.shape
    comp_infos = _border_components(~structural)
    if len(comp_infos) < NUM_MARKERS_RANGE[0]:
        return None

    candidate_layouts = []
    max_markers = min(NUM_MARKERS_RANGE[1], len(comp_infos))
    for k in range(NUM_MARKERS_RANGE[0], max_markers + 1):
        for idxs in combinations(range(len(comp_infos)), k):
            reach = np.zeros((h, w), dtype=bool)
            for idx in idxs:
                reach |= comp_infos[idx]["mask"]
            reach_frac = reach.mean()
            if not (REACH_RATIO_RANGE[0] <= reach_frac <= REACH_RATIO_RANGE[1]):
                continue
            if not _validate_structural(structural, reach, h, w):
                continue
            markers = _choose_marker_cells(comp_infos, idxs, h, w, rng)
            candidate_layouts.append((markers, reach))

    if not candidate_layouts:
        return None
    return rng.choice(candidate_layouts)


def _gen_attempt():
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)

    structural = _place_structural_walls(h, w, rng)
    if structural is None:
        return None

    layout = _select_marker_layout(structural, rng)
    if layout is None:
        return None
    markers, reach = layout

    out = _paint_output(structural, reach, markers)
    fill_mask = out == FILL
    if not fill_mask.any():
        return None

    visible_allowed = reach & (out == BG)
    hidden_allowed = (~reach) & (~structural)
    hidden_allowed[0, :] = hidden_allowed[-1, :] = False
    hidden_allowed[:, 0] = hidden_allowed[:, -1] = False

    n_decor = rng.randint(*NUM_DECOR_RANGE)
    if not visible_allowed.any():
        n_visible = 0
    else:
        n_visible = rng.randint(0, n_decor)
    n_hidden = n_decor - n_visible
    if n_hidden > 0 and not hidden_allowed.any():
        n_hidden = 0
        n_visible = n_decor
    if n_visible > 0 and not visible_allowed.any():
        n_visible = 0
        n_hidden = n_decor

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
