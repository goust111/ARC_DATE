# Variant of 8f3a5a89
# Dimension: D3 — 源对象（标记像素）位置 (src)
# Change: 单个品红像素固定在底左角 (H-1, 0)，始终为BL角... → 多个标记像素（多源洪填）

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


# ---------------------------------------------------------------------------
# Design analysis
# ---------------------------------------------------------------------------
# variant_spec:
#   grid: 10-20 rows x 10-20 cols
#   colors:
#     bg: 8
#     wall: 1
#     markers: 2 / 4 / 6
#     overlap_fill: 7
#   sources:
#     count: 2 or 3 unique single-pixel markers
#     placement: border cells only; at least two distinct border positions
#   region_rule:
#     Each marker flood-fills independently through bg cells, all marker cells,
#     and non-structural wall decorations. Structural walls remain blue blockers.
#     A reachable bg boundary cell is colored with that marker's own color.
#     If multiple markers claim the same boundary cell, that cell becomes orange.
#
# behavior_trace:
#   original:
#     1. Find the unique corner marker.
#     2. Flood from that single start.
#     3. Identify border-touching blue components adjacent to that reach as walls.
#     4. Paint one orange inner border.
#   variant:
#     1. Find every unique single-pixel non-blue marker on the border.
#     2. Flood independently from each marker.
#     3. Use the union of the first-pass reaches to identify structural walls.
#     4. Paint per-source inner borders in marker colors; merge overlaps to orange.
#   concrete example:
#     If red and magenta markers sit in the same open component, their reaches are
#     identical, so every boundary cell of that component is claimed twice and
#     therefore turns orange. If a yellow marker sits in another component, its
#     boundary stays yellow. This makes both "independent border" and "overlap
#     merge" visible in one sample.
#
# complexity_calibration:
#   - The wall logic, decoration logic, and inner-border criterion stay unchanged.
#   - The only new moving parts are: multiple border markers, per-source claims,
#     and a single fixed merge color. This keeps the task legible.
#   - Marker colors double as their fill colors, so the human-visible mapping is
#     simple: "this marker paints its own outline"; orange is the only special case.
#
# hidden_assumption_review:
#   - Original code assumes exactly one corner marker. That fails here, so marker
#     detection now scans the whole border for unique singleton colors.
#   - Original wall detection uses one reach0. With several markers, using only one
#     would miss walls that matter to other sources, so the solver uses the union.
#   - Original input reconstruction can erase fill by color equality (orange only).
#     Here fill can equal marker colors, so the generator tracks fill positions by
#     mask, not by color id.
#   - Independent floods on the same obstacle map can only be disjoint or identical.
#     The generator therefore intentionally mixes "all separate" and "shared pair"
#     layouts so the overlap branch is actually exercised.
#
# placement_legality:
#   - Markers are chosen only from open border cells and kept at least 3 Manhattan
#     steps apart when possible, so singleton markers stay visually separate.
#   - Visible decor is restricted to union-reachable non-boundary bg cells.
#   - Hidden decor is restricted to non-border cells outside every source reach,
#     preventing it from being reclassified as structural.
#
# observability_and_cleanliness:
#   - 2-3 markers are always present, so the variant never collapses back to the
#     original single-source behavior.
#   - The generator prefers layouts with either multiple distinct source colors or
#     a shared-pair overlap plus a separate source, so sample outputs visibly differ
#     from the original orange-only task.


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
    Paint source-specific inner borders around regions reached from several border markers.

    Input structure:
    - bg is the dominant color; blue components still act as candidate walls or decorations.
    - There are 2-3 distinct single-pixel markers on border cells, each using its own color.
    - Wall-colored components that touch the border may be structural; interior blue pieces remain decorations.

    Rule:
    - Flood-fill independently from each marker, using the union of the first-pass reaches to decide which blue components are structural blockers.
    - Erase non-structural blue decorations that stay unreachable from every marker.
    - On reachable background boundary cells, paint each source's inner border in that marker's color; if multiple sources claim the same boundary cell, paint it orange.

    Key insight: The wall-identification step is still global, but the border-painting step is source-wise: shared reachable components produce overlapping claims that collapse to one merge color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

    counts = Counter(int(v) for v in g.ravel())
    non_bg = [c for c in counts if c != bg]
    assert non_bg, "no non-background colors found"

    wall_col = max(non_bg, key=lambda c: counts[c])

    markers: list[tuple[int, int, int]] = []
    seen_colors: set[int] = set()
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            v = int(g[r, c])
            if v in (bg, wall_col):
                continue
            if counts.get(v, 0) != 1:
                continue
            if v in seen_colors:
                continue
            markers.append((r, c, v))
            seen_colors.add(v)

    assert len(markers) >= 2, "expected at least two unique border markers"
    marker_cols = {col for _, _, col in markers}

    passable0 = lambda r, c: int(g[r, c]) == bg or int(g[r, c]) in marker_cols
    reach0_by_marker = [
        _flood((mr, mc), passable0, h, w)
        for mr, mc, _ in markers
    ]
    reach0_union = set().union(*reach0_by_marker)

    walls: set[tuple[int, int]] = set()
    for comp_arr in components_mask(g == wall_col, connectivity=4):
        comp = [(int(r), int(c)) for r, c in comp_arr]
        touches_border = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)
        if not touches_border:
            continue
        adjacent_to_reach0 = any(
            any((nr, nc) in reach0_union for nr, nc in neighbors4(r, c))
            for r, c in comp
        )
        if adjacent_to_reach0:
            walls |= set(comp)

    def passable(r: int, c: int) -> bool:
        v = int(g[r, c])
        return v == bg or v in marker_cols or (v == wall_col and (r, c) not in walls)

    reaches: list[tuple[int, set[tuple[int, int]]]] = []
    union_reach: set[tuple[int, int]] = set()
    for mr, mc, marker_col in markers:
        reach = _flood((mr, mc), passable, h, w)
        reaches.append((marker_col, reach))
        union_reach |= reach

    for r in range(h):
        for c in range(w):
            if int(out[r, c]) == wall_col and (r, c) not in walls and (r, c) not in union_reach:
                out[r, c] = bg

    claims: dict[tuple[int, int], list[int]] = {}
    for fill_col, reach in reaches:
        for r, c in reach:
            if int(out[r, c]) != bg:
                continue
            is_boundary = False
            if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                is_boundary = True
            elif any((nr, nc) in walls for nr, nc in neighbors8(r, c)):
                is_boundary = True
            elif any((nr, nc) not in reach for nr, nc in neighbors4(r, c)):
                is_boundary = True
            if is_boundary:
                claims.setdefault((r, c), []).append(fill_col)

    overlap_fill = 7
    for (r, c), cols in claims.items():
        if len(cols) == 1:
            out[r, c] = cols[0]
        else:
            out[r, c] = overlap_fill

    return out


# --------------- generator ---------------

BG = 8
WALL = 1
MARKER_COLORS = (2, 4, 6)
OVERLAP_FILL = 7

GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (10, 20)

NUM_STRUCTS_RANGE = (1, 4)
STRUCT_SHAPE_RETRIES = 8
RECT_MAX_DIM_DIV = 3
TRI_MAX_SIZE = 6
TRI_MIN_DIM_DIV = 3

UNION_REACH_RATIO_RANGE = (0.38, 0.90)
MIN_COMPONENT_AREA = 8
MIN_SOURCE_REACH_RATIO = 0.08

NUM_DECOR_RANGE = (0, 3)
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
    m = _dilate4(mask)
    m[1:, 1:] |= mask[:-1, :-1]
    m[1:, :-1] |= mask[:-1, 1:]
    m[:-1, 1:] |= mask[1:, :-1]
    m[:-1, :-1] |= mask[1:, 1:]
    return m


def _place_structural_walls(h, w, forbidden_cells, rng):
    num = rng.randint(*NUM_STRUCTS_RANGE)
    structural = np.zeros((h, w), dtype=bool)
    placed = 0
    forbidden = set(forbidden_cells)
    for _ in range(num):
        for _try in range(STRUCT_SHAPE_RETRIES):
            cells = rng.choice(SHAPE_FACTORIES)(h, w, rng)
            if not cells:
                continue
            if any((r, c) in forbidden or structural[r, c] for r, c in cells):
                continue
            for r, c in cells:
                structural[r, c] = True
            placed += 1
            break
    return structural if placed > 0 else None


def _open_components_with_border(structural):
    h, w = structural.shape
    comps = []
    for comp_arr in components_mask(~structural, connectivity=4):
        coords = [(int(r), int(c)) for r, c in comp_arr]
        border_cells = [(r, c) for r, c in coords if r == 0 or c == 0 or r == h - 1 or c == w - 1]
        if not border_cells:
            continue
        comps.append(
            {
                "coords": coords,
                "area": len(coords),
                "border_cells": border_cells,
            }
        )
    return comps


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _pick_positions(cells, k, rng, min_dist=3):
    cells = list(cells)
    rng.shuffle(cells)
    chosen = []
    for cell in cells:
        if all(_manhattan(cell, prev) >= min_dist for prev in chosen):
            chosen.append(cell)
            if len(chosen) == k:
                return chosen
    if len(cells) >= k:
        return cells[:k]
    return None


def _choose_markers(structural, rng):
    comps = [
        comp for comp in _open_components_with_border(structural)
        if comp["area"] >= MIN_COMPONENT_AREA
    ]
    if not comps:
        return None

    multi_border = [comp for comp in comps if len(comp["border_cells"]) >= 2]
    separate_pool = [comp for comp in comps if len(comp["border_cells"]) >= 1]

    options = []
    if multi_border and len(separate_pool) >= 2:
        options.append("shared_pair")
    if len(separate_pool) >= 3:
        options.append("three_separate")
    if len(separate_pool) >= 2:
        options.append("two_separate")
    if not options:
        return None

    weights = {
        "shared_pair": 5,
        "three_separate": 3,
        "two_separate": 1,
    }
    mode = rng.choices(options, weights=[weights[o] for o in options], k=1)[0]

    markers = []
    if mode == "two_separate":
        chosen_comps = rng.sample(separate_pool, 2)
        positions = []
        for comp in chosen_comps:
            pos = _pick_positions(comp["border_cells"], 1, rng)
            if pos is None:
                return None
            positions.extend(pos)
    elif mode == "three_separate":
        chosen_comps = rng.sample(separate_pool, 3)
        positions = []
        for comp in chosen_comps:
            pos = _pick_positions(comp["border_cells"], 1, rng)
            if pos is None:
                return None
            positions.extend(pos)
    elif mode == "shared_pair":
        shared_comp = rng.choice(multi_border)
        other_choices = [comp for comp in separate_pool if comp is not shared_comp]
        if not other_choices:
            return None
        positions = _pick_positions(shared_comp["border_cells"], 2, rng)
        if positions is None:
            return None
        other_pos = _pick_positions(rng.choice(other_choices)["border_cells"], 1, rng)
        if other_pos is None:
            return None
        positions.extend(other_pos)
    rng.shuffle(positions)
    colors = rng.sample(list(MARKER_COLORS), len(positions))
    for (r, c), col in zip(positions, colors):
        markers.append((r, c, col))
    return markers


def _validate_structural(structural, reach_union0, h, w):
    reach_nbr = _dilate4(reach_union0)
    for comp_arr in components_mask(structural, connectivity=4):
        cmask = np.zeros((h, w), dtype=bool)
        cmask[comp_arr[:, 0], comp_arr[:, 1]] = True
        if not _touches_border(cmask):
            return False
        if not (cmask & reach_nbr).any():
            return False
    return True


def _paint_output(structural, reaches, markers):
    h, w = structural.shape
    out = np.full((h, w), BG, dtype=int)
    out[structural] = WALL
    for r, c, col in markers:
        out[r, c] = col

    claims = np.zeros((h, w), dtype=int)
    owner = np.full((h, w), -1, dtype=int)
    edge = np.zeros((h, w), dtype=bool)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True
    near_struct = _dilate8(structural)

    for idx, reach in enumerate(reaches):
        candidate = reach & (out == BG)
        near_unreach = _dilate4(~reach)
        boundary = candidate & (edge | near_struct | near_unreach)
        claims[boundary] += 1
        owner[boundary] = idx

    painted = claims > 0
    overlap = claims > 1
    for idx, (_, _, col) in enumerate(markers):
        out[(owner == idx) & (claims == 1)] = col
    out[overlap] = OVERLAP_FILL
    return out, painted, overlap


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


def _marker_reaches(markers, structural, h, w):
    blocked = structural.astype(bool)
    return [_flood_mask((r, c), blocked, h, w) for r, c, _ in markers]


def _gen_attempt():
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)

    structural = _place_structural_walls(h, w, forbidden_cells=set(), rng=rng)
    if structural is None:
        return None

    markers = _choose_markers(structural, rng)
    if markers is None:
        return None
    marker_cells = {(r, c) for r, c, _ in markers}
    if any(structural[r, c] for r, c in marker_cells):
        return None

    reaches0 = _marker_reaches(markers, structural, h, w)
    reach_union0 = np.any(np.stack(reaches0, axis=0), axis=0)
    union_ratio = float(reach_union0.mean())
    if not (UNION_REACH_RATIO_RANGE[0] <= union_ratio <= UNION_REACH_RATIO_RANGE[1]):
        return None

    if not all(float(reach.mean()) >= MIN_SOURCE_REACH_RATIO for reach in reaches0):
        return None

    if not _validate_structural(structural, reach_union0, h, w):
        return None

    out, painted_mask, overlap_mask = _paint_output(structural, reaches0, markers)
    union_reach = reach_union0.copy()

    visible_color_count = sum(int(((out == col) & painted_mask).any()) for col in MARKER_COLORS)
    if len(markers) < 2:
        return None
    if overlap_mask.any() and not (len(markers) == 3 and visible_color_count >= 1):
        return None
    if not overlap_mask.any() and visible_color_count < min(len(markers), 2):
        return None
    if painted_mask.sum() < max(h, w) // 2:
        return None

    n_decor = rng.randint(*NUM_DECOR_RANGE)
    n_visible = rng.randint(0, n_decor)
    n_hidden = n_decor - n_visible

    visible_allowed = union_reach & (out == BG)
    hidden_allowed = (~union_reach) & (~structural)
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
    inp[painted_mask] = BG
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
    raise RuntimeError("failed to generate valid input for 8f3a5a89 D3 variant")
