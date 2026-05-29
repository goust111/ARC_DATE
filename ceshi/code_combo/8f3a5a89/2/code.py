# Combo variant of 8f3a5a89 (index 2)
# combo_spec: D3/2+D14/2
# Name: 多源洪填+标记同色边界

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


# ---------------------------------------------------------------------------
# Design analysis
# ---------------------------------------------------------------------------
# combo_spec_summary:
#   - Keep the original wall-vs-decoration interpretation of blue pixels.
#   - Replace the single bottom-left marker with 2-3 singleton markers placed on
#     border cells, each using a distinct non-blue color.
#   - Replace the fixed orange fill with source-colored boundary painting:
#     every marker paints reachable boundary cells in its own color.
#   - When several markers can paint the same boundary cell, the winner is the
#     first flood to arrive there by 4-neighbor distance; equal-distance ties are
#     broken by a deterministic border-order scan of the markers.
#
# a) Concreteization and behavior trace:
#   - Original on a 12x12 board:
#     1. Find one corner marker.
#     2. Flood from that one start through azure cells.
#     3. Use the first reach to decide which blue border-touching components are
#        structural walls.
#     4. Paint one orange inner border around the reachable side.
#   - Combo on a 12x12 board:
#     1. Find 2-3 unique singleton markers on the border, for example red on the
#        top edge, yellow on the right edge, magenta on the bottom edge.
#     2. Each marker gets its own flood through azure cells, all markers, and
#        non-structural blue decorations.
#     3. The union of the first-pass floods still identifies the real blue walls.
#     4. Every marker computes the same kind of inner boundary as the base task,
#        but the painted color is the marker's own color.
#     5. If red and yellow sit in the same open chamber, some boundary cells are
#        reachable from both. Those cells do not turn orange; instead they are
#        assigned to whichever marker reaches them first in BFS distance.
#   - Difference from the base task:
#     the solver must now parse multiple starts, keep per-source reach logic, and
#     resolve overlaps by arrival priority rather than by a single global fill color.
#
# b) Complexity calibration:
#   - Marker colors are restricted to a small repeated set {2,4,6}. This keeps the
#     source->boundary color mapping obvious from a few examples.
#   - The generator always creates at least one shared open component containing
#     two markers, so the "overlap then first-arrival split" behavior is visible.
#   - A third marker is added only when another open component is available, which
#     shows both shared-component and separate-component behavior without cluttering
#     every sample.
#
# c) Implicit assumption audit:
#   - The original code assumed exactly one corner marker. That no longer holds, so
#     marker detection must scan the full border for unique singleton colors.
#   - The original wall test used one reach0. With multiple sources that would miss
#     walls seen only from another source, so the combo uses the union of reach0s.
#   - The original fill write was a constant color 7. That fails here because the
#     output color depends on which source reaches each boundary cell first.
#   - The original generator could erase output fill by color equality. That is not
#     safe anymore because painted boundary colors equal marker colors, so the combo
#     generator tracks the painted boundary by mask instead.
#   - New tie-break dependence is limited to equal-distance cases. The generator
#     rejects layouts with too many tied boundary cells so most color splits are
#     explained by visible proximity, not by hidden ordering.
#
# d) Placement legality:
#   - Markers are sampled only from non-wall border cells of open components.
#   - Shared markers are kept several steps apart on the same border-connected open
#     component so they stay visually distinct and both get nontrivial territory.
#   - Visible decorations are placed only on untouched interior reachable cells;
#     hidden decorations are restricted to unreachable non-border cells so they
#     cannot be mistaken for structural walls.
#
# e) Visual observability:
#   - Humans can infer the new rule from repeated evidence:
#     each boundary segment uses one of the marker colors, and samples always show
#     at least one region where two markers compete over the same enclosure.
#   - No invisible color ordering such as "smaller color id wins" is used. The main
#     explanatory cue is geometric arrival from the markers.
#
# f) Visual cleanliness:
#   - The palette remains sparse: azure background, blue walls, 2-3 marker colors.
#   - Boundary paint stays one cell thick, and visible decorations are kept away
#     from the painted ring, so the region structure remains easy to parse.
#
# g) Change inferability:
#   - The color rule is structural rather than arbitrary: a marker's own color
#     reappears on the boundary it controls. This repeated source-target coupling
#     makes the color behavior learnable from only a few examples.
#
# 9) New main() assumptions and generator guarantees:
#   - main() assumes 2-3 border singleton markers with distinct colors. The
#     generator always emits exactly that.
#   - main() assumes wall color is the most frequent non-background color. The
#     generator uses only blue for walls/decorations, so this remains stable.
#   - main() assumes border-order tie-breaks are only a small correction on top of
#     distance-based ownership. The generator rejects tie-heavy layouts.
#   - main() assumes every marker has a visible painted boundary contribution. The
#     generator validates a minimum painted count per marker before accepting a grid.


BG = 8
WALL = 1
MARKER_COLORS = (2, 4, 6)

GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (10, 20)

NUM_STRUCTS_RANGE = (1, 4)
STRUCT_SHAPE_RETRIES = 8
RECT_MAX_DIM_DIV = 3
TRI_MAX_SIZE = 6
TRI_MIN_DIM_DIV = 3

UNION_REACH_RATIO_RANGE = (0.38, 0.90)
MIN_COMPONENT_AREA = 8
MIN_SHARED_BORDER_CELLS = 6
MIN_MARKER_GAP = 4
MIN_SOURCE_REACH_RATIO = 0.08
MIN_BOUNDARY_PER_MARKER = 4
MAX_TIED_BOUNDARY_RATIO = 0.12

NUM_DECOR_RANGE = (0, 3)
DECOR_AREA_RANGE = (1, 4)
DECOR_BORDER = 1
DECOR_PADDING = 1

GEN_ATTEMPTS = 700


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


def _distance_map(start: tuple[int, int], passable, h: int, w: int) -> np.ndarray:
    dist = np.full((h, w), -1, dtype=int)
    sr, sc = start
    if not passable(sr, sc):
        return dist
    dist[sr, sc] = 0
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and dist[nr, nc] < 0 and passable(nr, nc):
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def _border_cells_clockwise(h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(w):
        cells.append((0, c))
    for r in range(1, h - 1):
        cells.append((r, w - 1))
    for c in range(w - 1, -1, -1):
        cells.append((h - 1, c))
    for r in range(h - 2, 0, -1):
        cells.append((r, 0))
    return cells


def _marker_order(markers: list[tuple[int, int, int]], h: int, w: int) -> list[tuple[int, int, int]]:
    border_rank = {cell: idx for idx, cell in enumerate(_border_cells_clockwise(h, w))}
    return sorted(markers, key=lambda m: border_rank[(m[0], m[1])])


def _select_owner(
    cell: tuple[int, int],
    dist_maps: list[np.ndarray],
) -> tuple[int | None, bool]:
    r, c = cell
    best_idx: int | None = None
    best_dist: int | None = None
    tie = False
    for idx, dist in enumerate(dist_maps):
        d = int(dist[r, c])
        if d < 0:
            continue
        if best_dist is None or d < best_dist:
            best_dist = d
            best_idx = idx
            tie = False
        elif d == best_dist:
            tie = True
    return best_idx, tie


def _find_markers(
    g: np.ndarray,
    bg: int,
    wall_col: int,
    counts: Counter[int],
) -> list[tuple[int, int, int]]:
    h, w = g.shape
    markers: list[tuple[int, int, int]] = []
    seen_colors: set[int] = set()
    for r, c in _border_cells_clockwise(h, w):
        v = int(g[r, c])
        if v in (bg, wall_col):
            continue
        if counts.get(v, 0) != 1:
            continue
        if v in seen_colors:
            continue
        markers.append((r, c, v))
        seen_colors.add(v)
    return markers


def _collect_solver_state(
    g: np.ndarray,
) -> tuple[
    int,
    int,
    int,
    list[tuple[int, int, int]],
    set[tuple[int, int]],
    list[tuple[int, set[tuple[int, int]]]],
    set[tuple[int, int]],
    list[np.ndarray],
]:
    h, w = g.shape
    bg = get_bg(g)
    counts = Counter(int(v) for v in g.ravel())
    non_bg = [c for c in counts if c != bg]
    assert non_bg, "no non-background colors found"
    wall_col = max(non_bg, key=lambda c: counts[c])

    markers = _find_markers(g, bg, wall_col, counts)
    assert 2 <= len(markers) <= 3, "expected 2-3 unique singleton border markers"
    markers = _marker_order(markers, h, w)
    marker_cols = {col for _, _, col in markers}

    passable0 = lambda r, c: int(g[r, c]) == bg or int(g[r, c]) in marker_cols
    reach0_by_marker = [_flood((mr, mc), passable0, h, w) for mr, mc, _ in markers]
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
    dist_maps: list[np.ndarray] = []
    for mr, mc, marker_col in markers:
        reach = _flood((mr, mc), passable, h, w)
        reaches.append((marker_col, reach))
        union_reach |= reach
        dist_maps.append(_distance_map((mr, mc), passable, h, w))

    return bg, wall_col, h, markers, walls, reaches, union_reach, dist_maps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint source-colored inner borders from several border markers with first-arrival overlap resolution.

    Input structure:
    - The dominant color is the background, blue still denotes candidate walls or decorations,
      and there are 2-3 distinct singleton markers placed on border cells.
    - Blue components touching the border may be true structural barriers, while interior blue
      fragments remain decorations that can survive or be erased depending on reachability.
    - Marker colors are meaningful instructions: each marker later paints the boundary of the
      region it can reach.

    Rule:
    - Flood independently from every marker while treating blue as blocked, then use the union
      of those first-pass reaches to identify the border-touching blue components that are truly
      structural.
    - Flood again from each marker while allowing traversal through non-structural blue
      decorations, and erase decorations that remain unreachable from all markers.
    - For every reachable background cell that lies on an inner boundary, paint it in the color
      of the marker that reaches that cell first by BFS distance; if several markers tie, keep
      the earliest marker in the fixed border-scan order.

    Key insight: Wall detection is still global, but coloring is source-specific. The same open
    chamber can be reachable from multiple markers, so the final boundary is a distance-based
    partition of one shared enclosure rather than a single uniformly colored ring.
    """
    g = np.asarray(grid, dtype=int)
    bg, wall_col, h, markers, walls, reaches, union_reach, dist_maps = _collect_solver_state(g)
    w = g.shape[1]
    out = g.copy()

    for r in range(h):
        for c in range(w):
            if int(out[r, c]) == wall_col and (r, c) not in walls and (r, c) not in union_reach:
                out[r, c] = bg

    claims: set[tuple[int, int]] = set()
    for _, reach in reaches:
        for r, c in reach:
            if int(out[r, c]) != bg:
                continue
            if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                claims.add((r, c))
                continue
            if any((nr, nc) in walls for nr, nc in neighbors8(r, c)):
                claims.add((r, c))
                continue
            if any((nr, nc) not in reach for nr, nc in neighbors4(r, c)):
                claims.add((r, c))

    for cell in claims:
        owner_idx, _ = _select_owner(cell, dist_maps)
        assert owner_idx is not None, "claimed boundary cell is unreachable from every marker"
        r, c = cell
        out[r, c] = markers[owner_idx][2]

    return out


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


def _distance_mask(start, blocked, h, w):
    dist = np.full((h, w), -1, dtype=int)
    dist[start] = 0
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and dist[nr, nc] < 0 and not blocked[nr, nc]:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


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


def _pick_positions(cells, k, rng, min_dist=MIN_MARKER_GAP):
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
    h, w = structural.shape
    comps = [comp for comp in _open_components_with_border(structural) if comp["area"] >= MIN_COMPONENT_AREA]
    if not comps:
        return None

    shared_pool = [comp for comp in comps if len(comp["border_cells"]) >= MIN_SHARED_BORDER_CELLS]
    if not shared_pool:
        return None

    mode = "shared_pair"
    shared_comp = rng.choice(shared_pool)
    shared_positions = _pick_positions(shared_comp["border_cells"], 2, rng)
    if shared_positions is None:
        return None

    positions = list(shared_positions)
    other_comps = [comp for comp in comps if comp is not shared_comp]
    if other_comps and rng.random() < 0.7:
        other_comp = rng.choice(other_comps)
        extra = _pick_positions(other_comp["border_cells"], 1, rng, min_dist=1)
        if extra is None:
            return None
        positions.extend(extra)
        mode = "paired_plus_single"

    colors = rng.sample(list(MARKER_COLORS), len(positions))
    markers = [(r, c, col) for (r, c), col in zip(positions, colors)]
    markers = _marker_order(markers, h, w)

    # Guarantee that the intended overlap really exists.
    if mode in {"shared_pair", "paired_plus_single"}:
        marker_cells = [(r, c) for r, c, _ in markers]
        comp_lookup = {}
        for idx, comp in enumerate(comps):
            for cell in comp["coords"]:
                comp_lookup[cell] = idx
        first_two_ids = {comp_lookup[marker_cells[0]], comp_lookup[marker_cells[1]]}
        if len(first_two_ids) != 1:
            shared_id = comp_lookup[shared_positions[0]]
            shared_markers = [m for m in markers if comp_lookup[(m[0], m[1])] == shared_id]
            if len(shared_markers) < 2:
                return None

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


def _marker_reaches(markers, structural, h, w):
    blocked = structural.astype(bool)
    return [_flood_mask((r, c), blocked, h, w) for r, c, _ in markers]


def _marker_distances(markers, structural, h, w):
    blocked = structural.astype(bool)
    return [_distance_mask((r, c), blocked, h, w) for r, c, _ in markers]


def _boundary_mask(reach, structural):
    h, w = structural.shape
    edge = np.zeros((h, w), dtype=bool)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True
    near_struct = _dilate8(structural)
    near_unreach = _dilate4(~reach)
    return reach & (edge | near_struct | near_unreach) & (~structural)


def _paint_output(structural, reaches, dist_maps, markers):
    h, w = structural.shape
    out = np.full((h, w), BG, dtype=int)
    out[structural] = WALL
    for r, c, col in markers:
        out[r, c] = col

    painted_mask = np.zeros((h, w), dtype=bool)
    tie_mask = np.zeros((h, w), dtype=bool)
    for reach in reaches:
        boundary = _boundary_mask(reach, structural) & (out == BG)
        painted_mask |= boundary

    for r, c in np.argwhere(painted_mask):
        rr, cc = int(r), int(c)
        owner_idx, tie = _select_owner((rr, cc), dist_maps)
        if owner_idx is None:
            return None, None, None, None
        out[rr, cc] = markers[owner_idx][2]
        tie_mask[rr, cc] = tie

    color_counts = {
        col: int(((out == col) & painted_mask).sum())
        for _, _, col in markers
    }
    return out, painted_mask, tie_mask, color_counts


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


def _shared_pair_count(reaches):
    count = 0
    for i in range(len(reaches)):
        for j in range(i + 1, len(reaches)):
            if np.array_equal(reaches[i], reaches[j]):
                count += 1
    return count


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

    reaches = _marker_reaches(markers, structural, h, w)
    reach_union = np.any(np.stack(reaches, axis=0), axis=0)
    union_ratio = float(reach_union.mean())
    if not (UNION_REACH_RATIO_RANGE[0] <= union_ratio <= UNION_REACH_RATIO_RANGE[1]):
        return None

    if not all(float(reach.mean()) >= MIN_SOURCE_REACH_RATIO for reach in reaches):
        return None

    if _shared_pair_count(reaches) == 0:
        return None

    if not _validate_structural(structural, reach_union, h, w):
        return None

    dist_maps = _marker_distances(markers, structural, h, w)
    out, painted_mask, tie_mask, color_counts = _paint_output(structural, reaches, dist_maps, markers)
    if out is None:
        return None

    if painted_mask.sum() < max(h, w) // 2:
        return None

    if any(color_counts[col] < MIN_BOUNDARY_PER_MARKER for _, _, col in markers):
        return None

    tie_ratio = float(tie_mask.sum()) / float(max(1, painted_mask.sum()))
    if tie_ratio > MAX_TIED_BOUNDARY_RATIO:
        return None

    n_decor = rng.randint(*NUM_DECOR_RANGE)
    n_visible = rng.randint(0, n_decor)
    n_hidden = n_decor - n_visible

    visible_allowed = reach_union & (out == BG)
    hidden_allowed = (~reach_union) & (~structural)
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
    raise RuntimeError("failed to generate valid input for 8f3a5a89 combo D3/2+D14/2")
