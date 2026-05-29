# Combo variant of a47bf94d (index 17)
# combo_spec: D3/1+D14/2
# Name: 间歇轨迹+管道段颜色交替

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    - The base token logic remains the same: a solid 3x3 source becomes a diamond
#      in place and emits an X to the far endpoint; an X source stays and emits a
#      diamond unless that far diamond is already present.
#    - The wire system is no longer uniformly cyan. Instead, ordinary pipe spans
#      alternate between colors 8 and 9. Grey 5-cells mark crossover rows and act
#      as segment boundaries.
#    - A token's travel trail is also no longer globally periodic. While tracing a
#      route, the k-th pipe segment paints every k-th eligible cell with the token
#      color. Segment 1 paints densely, segment 2 paints every other cell, segment
#      3 every third cell, and so on.
#    - In a concrete route:
#         top endpoint -> [8 8 8] -> 5 -> [9 9 9 9] -> 5 -> [8 8 8 8 8]
#      a transmitted token recolors:
#         segment 1: all eligible cells,
#         segment 2: cells 0,2,...,
#         segment 3: cells 0,3,...
#      so the dash spacing and the alternating structural color both advance when
#      the path crosses a new grey crossover row.
#
# b) Complexity calibration:
#    - Only two ordinary pipe colors are used, so the visual language stays small.
#      The growing period is tied to segment order rather than to hidden color ids.
#    - Generator instances require at least two crossover rows, so routes expose at
#      least three segment indices inside one example. This gives repeated evidence
#      for the "after each crossing, both color parity and dash period advance"
#      rule without exploding the number of cases.
#
# c) Hidden assumptions audited:
#    - The original solver treated color 9 as a special rectangular pass-through.
#      That assumption conflicts with using 9 as a normal alternating pipe color,
#      so this combo removes embedded 9-rectangles entirely.
#    - The original trace only needed the terminal endpoint. This combo needs the
#      ordered path, because the output depends on how many distinct crossover rows
#      have been passed before each visible wire cell.
#    - Shared intersection cells are dangerous recolor targets because multiple
#      logical routes touch them. Therefore grey 5-cells remain structural-only and
#      trajectory painting is restricted to degree<=2 cells whose input color is 8
#      or 9.
#    - Segment counting now assumes that each logical route meets crossover rows in
#      monotone order from one side of the grid to the other. The generator keeps
#      the original downward routing construction, so this remains true.
#
# d) Placement legality:
#    - Token colors stay in {1,2,3,4,6,7}, disjoint from structural colors 5/8/9,
#      so token detection remains unambiguous.
#    - Channel spacing and token offsets follow the original bounds, so 3x3 tokens
#      do not overlap each other or the wire network.
#    - Crossover rows are spaced apart and the generator rejects cases where the
#      visible segment lengths are too short to show the first few trajectory
#      periods.
#
# e) Visual observability:
#    - A human comparing input and output can see two linked regularities:
#      successive pipe spans alternate 8/9/8/... between grey crossover rows, and
#      the token-colored dashes on those spans get sparser in the same order.
#    - Because every route passes through the same ordered set of crossover rows,
#      the pattern repeats across 4-5 channels within one example instead of being
#      a one-off coincidence.
#
# f) Visual cleanliness:
#    - Grey boundary rows are left unpainted by trajectories, which keeps the
#      crossover structure legible and prevents multi-token overwrite noise at
#      shared intersections.
#    - Ordinary colored spans carry the dashed trajectories; there is no second
#      independent overlay on already-special grey cells.
#
# g) Inferability of change:
#    - No hidden color mapping is introduced. The trail always uses the traveling
#      token's own color.
#    - The only inference burden is relating segment order to dash spacing, and the
#      alternating 8/9 structural encoding provides the repeated cue for that order.

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]

SEGMENT_COLORS = (8, 9)
CROSS_COLOR = 5
STRUCTURE_COLORS = {CROSS_COLOR, *SEGMENT_COLORS}


def _segment_color(segment_idx: int) -> int:
    return SEGMENT_COLORS[(segment_idx - 1) % 2]


def _find_solid_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for maximal 3x3 solid blocks."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            col = int(sub[0, 0])
            if col == bg or np.any(sub != col):
                continue
            maximal = True
            for rr in range(r0, r0 + 3):
                for cc in range(c0, c0 + 3):
                    for nr, nc in neighbors4(rr, cc):
                        if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                            if int(g[nr, nc]) == col:
                                maximal = False
                                break
                    if not maximal:
                        break
                if not maximal:
                    break
            if not maximal:
                continue
            found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _find_x_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for isolated 3x3 X patterns."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}

    mask = np.zeros((3, 3), dtype=bool)
    for rr, cc in [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]:
        mask[rr, cc] = True

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            col = int(sub[0, 0])
            if col == bg:
                continue
            if np.any(sub[mask] != col) or np.any(sub[~mask] != bg):
                continue
            isolated = True
            for dr0, dc0 in _X_OFF:
                rr, cc = r0 + 1 + dr0, c0 + 1 + dc0
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                        if int(g[nr, nc]) == col:
                            isolated = False
                            break
                if not isolated:
                    break
            if not isolated:
                continue
            found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _find_diamond_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for isolated 3x3 diamond patterns."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}

    mask = np.zeros((3, 3), dtype=bool)
    for rr, cc in [(0, 1), (1, 0), (1, 2), (2, 1)]:
        mask[rr, cc] = True

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            col = int(sub[0, 1])
            if col == bg:
                continue
            if np.any(sub[mask] != col) or np.any(sub[~mask] != bg):
                continue
            isolated = True
            for dr0, dc0 in _DIAMOND_OFF:
                rr, cc = r0 + 1 + dr0, c0 + 1 + dc0
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                        if int(g[nr, nc]) == col:
                            isolated = False
                            break
                if not isolated:
                    break
            if not isolated:
                continue
            found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=4)]


def _trace_wire(
    comp: set[tuple[int, int]],
    start: tuple[int, int],
    dir0: tuple[int, int],
    h: int,
    w: int,
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]:
    """Follow a wire greedily and return (terminal_cell, exit_direction, visited_path)."""
    cur, d, prev = start, dir0, None
    path = [start]
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 4):
        if (cur, d) in visited:
            assert False, "wire trace looped"
        visited.add((cur, d))

        r, c = cur
        opts = [((nr, nc), nr - r, nc - c) for nr, nc in neighbors4(r, c) if (nr, nc) in comp]
        if not opts:
            return cur, d, path

        rev = (-d[0], -d[1])
        nonrev = [o for o in opts if (o[1], o[2]) != rev]
        opts2 = nonrev if nonrev else opts

        straight = [o for o in opts2 if (o[1], o[2]) == d]
        if straight:
            p, dr, dc = straight[0]
        else:
            perp = [o for o in opts2 if o[1] * d[0] + o[2] * d[1] == 0]
            if perp:
                p, dr, dc = sorted(perp, key=lambda x: (x[1], x[2]))[0]
            else:
                p, dr, dc = opts2[0]

        if p == prev:
            return cur, d, path

        prev = cur
        cur = p
        d = (dr, dc)
        path.append(cur)

    assert False, "wire trace exceeded step limit"


def _degree_map(comp: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    deg = {p: 0 for p in comp}
    for r, c in comp:
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in comp:
                deg[(r, c)] += 1
    return deg


def _segment_cells_from_path(
    g: np.ndarray,
    path: list[tuple[int, int]],
    deg: dict[tuple[int, int], int],
) -> dict[int, list[tuple[int, int]]]:
    """Bucket eligible visible wire cells by route-segment index."""
    seg_cells: dict[int, list[tuple[int, int]]] = {}
    current_segment = 1
    seen_boundary_rows: set[int] = set()
    seen_cells: set[tuple[int, int]] = set()
    for r, c in path:
        val = int(g[r, c])
        if val == CROSS_COLOR:
            if r not in seen_boundary_rows:
                seen_boundary_rows.add(r)
                current_segment += 1
            continue
        if val not in SEGMENT_COLORS:
            continue
        cell = (r, c)
        if cell in seen_cells:
            continue
        seen_cells.add(cell)
        if deg.get(cell, 0) > 2:
            continue
        seg_cells.setdefault(current_segment, []).append(cell)
    return seg_cells


def _paint_trajectory(
    out: np.ndarray,
    g: np.ndarray,
    path: list[tuple[int, int]],
    deg: dict[tuple[int, int], int],
    col: int,
) -> dict[int, int]:
    """Paint every k-th cell on the k-th visible route segment."""
    seg_cells = _segment_cells_from_path(g, path, deg)
    for seg_idx, cells in seg_cells.items():
        for idx, (r, c) in enumerate(cells):
            if idx % seg_idx == 0:
                out[r, c] = col
    return {seg_idx: len(cells) for seg_idx, cells in seg_cells.items()}


def _iter_routes(
    g: np.ndarray,
    solids: dict[tuple[int, int], tuple[int, tuple[int, int]]],
    xs: dict[tuple[int, int], tuple[int, tuple[int, int]]],
) -> list[dict[str, object]]:
    """Enumerate source-token routes with the traced path and destination anchor."""
    h, w = g.shape
    bg = get_bg(g)
    diamonds = _find_diamond_3x3(g, bg)
    track_mask = g != bg
    for (_, _), (_, (r0, c0)) in solids.items():
        track_mask[r0 : r0 + 3, c0 : c0 + 3] = False
    for cr, cc in xs.keys():
        for dr, dc in _X_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                track_mask[rr, cc2] = False
    for (_, _), (_, (r0, c0)) in diamonds.items():
        track_mask[r0 : r0 + 3, c0 : c0 + 3] = False

    routes: list[dict[str, object]] = []
    for pts in _components(track_mask):
        comp = set(pts)
        if len(comp) < 4:
            continue
        comp_colors = {int(g[r, c]) for r, c in comp}
        deg = _degree_map(comp)
        endpoints = [p for p in comp if deg[p] <= 1]
        if not endpoints:
            continue

        for ep in endpoints:
            er, ec = ep
            for odr, odc in _N4:
                cen = (er + 2 * odr, ec + 2 * odc)
                if not (0 <= cen[0] < h and 0 <= cen[1] < w):
                    continue

                if cen in solids:
                    col, _ = solids[cen]
                    shape = "solid"
                elif cen in xs:
                    col, _ = xs[cen]
                    shape = "x"
                else:
                    continue

                if col in comp_colors:
                    continue

                dir_in = (-odr, -odc)
                end, dd, path = _trace_wire(comp, ep, dir_in, h, w)
                dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                if shape == "x" and dest in diamonds and diamonds[dest][0] == col:
                    continue
                routes.append(
                    {
                        "color": col,
                        "shape": shape,
                        "dest": dest,
                        "path": path,
                        "deg": deg,
                    }
                )
    return routes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route complementary tokens and reveal a segment-indexed dashed travel trail.

    Input structure:
    - Background surrounds a single orthogonal pipe network whose ordinary spans
      alternate between colors 8 and 9; grey 5-cells mark crossover rows.
    - 3x3 source tokens sit two cells away from pipe endpoints. They are either
      solid blocks or X-patterns, and some far endpoints may already hold a
      resting diamond of the same color.

    Rule:
    - Each solid source becomes a diamond in place and sends an X to the far
      endpoint. Each X source stays and sends a diamond, unless that color already
      appears on the connected pipe component via a pre-placed far diamond.
    - While the token travels, only visible 8/9 pipe cells are trail candidates.
      On the first route segment every candidate is recolored to the token color;
      after the first grey crossover row the period becomes 2, after the next it
      becomes 3, and so on.

    Key insight: Grey crossover rows partition each route into ordered segments;
    the pipe alternates 8/9 between those segments, and the token-colored dash
    spacing on segment k is exactly k.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)
    routes = _iter_routes(g, solids, xs)

    out = g.copy()

    for (_, _), (col, (r0, c0)) in solids.items():
        out[r0 : r0 + 3, c0 : c0 + 3] = bg
        cr, cc = r0 + 1, c0 + 1
        for dr, dc in _DIAMOND_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                out[rr, cc2] = col

    for route in routes:
        col = int(route["color"])
        dest = route["dest"]
        path = route["path"]
        deg = route["deg"]
        if route["shape"] == "solid":
            offsets = _X_OFF
        else:
            offsets = _DIAMOND_OFF
        for dr, dc in offsets:
            rr, cc2 = dest[0] + dr, dest[1] + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                out[rr, cc2] = col
        _paint_trajectory(out, g, path, deg, col)

    return out


# --------------- generator (alternating_segments + coupled_trajectory) ---------------

_DIAMOND = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]

CHANNEL_COUNTS = [4, 5]
CHANNEL_SPACING = [4, 5]
LEFT_MARGIN_RANGE = (2, 3)
BYPASS_COL_OFFSET = 2
TOKEN_ROW_OPTIONS = [2, 3]
TOKEN_COLORS_POOL = [1, 2, 3, 4, 6, 7]
GRID_H_RANGE = (20, 26)
GRID_W_MIN = 22
GRID_W_MAX = 26
EP_TOKEN_GAP = 2

X_INPUT_COUNT_WEIGHTS = [70, 30]
FAR_DIAMOND_PROB = 0.2

MIN_CROSSOVERS = 2
MIN_SEGMENTS_WITH_TWO_DASHES = 3
GEN_MAX_RETRIES = 400


def _decompose_to_crossovers(perm: list[int]) -> list[tuple[int, int]]:
    """Decompose a permutation into sequential crossover moves using the bypass."""
    n = len(perm)
    visited = [False] * n
    crossovers = []
    for start in range(n):
        if visited[start] or perm[start] == start:
            visited[start] = True
            continue
        cycle = []
        i = start
        while not visited[i]:
            visited[i] = True
            cycle.append(i)
            i = perm[i]
        if len(cycle) < 2:
            continue
        crossovers.append((cycle[0], -1))
        for k in range(len(cycle) - 1, 0, -1):
            crossovers.append((cycle[k], cycle[(k + 1) % len(cycle)]))
        crossovers.append((-1, cycle[1]))
    return crossovers


def _stamp_token(grid: np.ndarray, center_r: int, center_c: int, color: int, offsets: list[tuple[int, int]]) -> None:
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = color


def _build_wire_network(
    grid: np.ndarray,
    n: int,
    ch_cols: list[int],
    bypass_col: int,
    top_ep_row: int,
    bot_ep_row: int,
    crossovers: list[tuple[int, int]],
    xover_rows: list[int],
) -> tuple[list[int | None], int | None]:
    """Draw the alternating-color wire network onto grid."""
    slot_wire: list[int | None] = list(range(n))
    bypass_wire: int | None = None

    current_segment = 1
    col_seg_start: dict[int, int] = {}
    initial_color = _segment_color(current_segment)
    for j in range(n):
        col_seg_start[ch_cols[j]] = top_ep_row
        grid[top_ep_row, ch_cols[j]] = initial_color

    for xover_idx, (src_slot, dst_slot) in enumerate(crossovers):
        xrow = xover_rows[xover_idx]

        if src_slot == -1:
            src_col, dst_col = bypass_col, ch_cols[dst_slot]
        elif dst_slot == -1:
            src_col, dst_col = ch_cols[src_slot], bypass_col
        else:
            src_col, dst_col = ch_cols[src_slot], ch_cols[dst_slot]

        active_cols = {ch_cols[j] for j in range(n) if slot_wire[j] is not None}
        if bypass_wire is not None:
            active_cols.add(bypass_col)

        span_color = _segment_color(current_segment)
        for c in active_cols:
            if c in col_seg_start:
                start = col_seg_start[c] + 1
                if start <= xrow - 1:
                    grid[start:xrow, c] = span_color

        next_color = _segment_color(current_segment + 1)
        c_min, c_max = min(src_col, dst_col), max(src_col, dst_col)
        grid[xrow, c_min : c_max + 1] = next_color
        for c in active_cols:
            grid[xrow, c] = CROSS_COLOR

        new_seg_start = {c: xrow for c in active_cols if c != src_col}
        new_seg_start[dst_col] = xrow
        col_seg_start = new_seg_start

        if src_slot == -1:
            slot_wire[dst_slot] = bypass_wire
            bypass_wire = None
        elif dst_slot == -1:
            bypass_wire = slot_wire[src_slot]
            slot_wire[src_slot] = None
        else:
            slot_wire[dst_slot] = slot_wire[src_slot]
            slot_wire[src_slot] = None

        current_segment += 1

    final_color = _segment_color(current_segment)
    for j in range(n):
        if slot_wire[j] is not None:
            c = ch_cols[j]
            if c in col_seg_start:
                start = col_seg_start[c] + 1
                if start <= bot_ep_row:
                    grid[start : bot_ep_row + 1, c] = final_color

    return slot_wire, bypass_wire


def _derive_input(
    grid: np.ndarray,
    n: int,
    ch_cols: list[int],
    perm: list[int],
    token_colors: list[int],
    top_token_row: int,
    bot_token_row: int,
) -> np.ndarray:
    """Derive the input grid from the wire-only grid by placing source tokens."""
    inp = grid.copy()
    n_x = random.choices([1, 2], weights=X_INPUT_COUNT_WEIGHTS)[0]
    x_channels = set(random.sample(range(n), min(n_x, n - 1)))
    far_diamond_channels: set[int] = set()
    if len(x_channels) >= 2 and random.random() < FAR_DIAMOND_PROB:
        far_diamond_channels.add(random.choice(sorted(x_channels)))

    for i in range(n):
        col = token_colors[i]
        top_center = (top_token_row, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[perm[i]])

        if i in x_channels:
            if random.choice([True, False]):
                near, far = top_center, bot_center
            else:
                near, far = bot_center, top_center
            _stamp_token(inp, near[0], near[1], col, _X)
            if i in far_diamond_channels:
                _stamp_token(inp, far[0], far[1], col, _DIAMOND)
        else:
            side = random.choice([top_center, bot_center])
            _stamp_token(inp, side[0], side[1], col, _SOLID)
    return inp


def _analyze_fused_behavior(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    """Measure whether the generated sample visibly uses the combo behavior."""
    bg = get_bg(inp)
    solids = _find_solid_3x3(inp, bg)
    xs = _find_x_3x3(inp, bg)
    routes = _iter_routes(inp, solids, xs)

    max_segment_index = 0
    routes_with_three_strides = 0
    painted_wire_cells = 0
    has_both_structural_colors = int(all((inp == col).any() for col in SEGMENT_COLORS))

    for route in routes:
        col = int(route["color"])
        seg_cells = _segment_cells_from_path(inp, route["path"], route["deg"])
        if seg_cells:
            max_segment_index = max(max_segment_index, max(seg_cells))
        visible_three = True
        for seg_idx in range(1, MIN_SEGMENTS_WITH_TWO_DASHES + 1):
            cells = seg_cells.get(seg_idx, [])
            if len(cells) < seg_idx + 1:
                visible_three = False
                break
        if visible_three:
            routes_with_three_strides += 1

        for cells in seg_cells.values():
            for r, c in cells:
                if int(out[r, c]) == col:
                    painted_wire_cells += 1

    return {
        "route_count": len(routes),
        "solid_source_count": len(solids),
        "x_source_count": len(xs),
        "active_x_route_count": sum(int(route["shape"] == "x") for route in routes),
        "max_segment_index": max_segment_index,
        "routes_with_three_strides": routes_with_three_strides,
        "painted_wire_cells": painted_wire_cells,
        "has_both_structural_colors": has_both_structural_colors,
    }


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(CHANNEL_COUNTS)
    spacing = random.choice(CHANNEL_SPACING)

    left_margin = random.randint(*LEFT_MARGIN_RANGE)
    first_col = left_margin + 1
    ch_cols = [first_col + i * spacing for i in range(n)]
    bypass_col = ch_cols[-1] + BYPASS_COL_OFFSET

    w = max(bypass_col + 3, GRID_W_MIN)
    if w > GRID_W_MAX:
        return None

    top_token_row = random.choice(TOKEN_ROW_OPTIONS)
    top_ep_row = top_token_row + EP_TOKEN_GAP
    token_colors = random.sample(TOKEN_COLORS_POOL, n)

    perm = list(range(n))
    while True:
        random.shuffle(perm)
        crossovers = _decompose_to_crossovers(perm)
        if perm != list(range(n)) and len(crossovers) >= MIN_CROSSOVERS:
            break

    n_xovers = len(crossovers)
    min_wire_rows = n_xovers * 2 + 2
    h_min = top_ep_row + min_wire_rows + top_token_row + 3
    if h_min > GRID_H_RANGE[1]:
        return None
    h = random.randint(max(GRID_H_RANGE[0], h_min), GRID_H_RANGE[1])

    bot_token_row = h - top_token_row - 1
    bot_ep_row = bot_token_row - EP_TOKEN_GAP
    if bot_ep_row - top_ep_row < min_wire_rows:
        return None

    avail = bot_ep_row - top_ep_row - 2
    step = max(2, avail // (n_xovers + 1))
    xover_rows = [top_ep_row + 1 + step * (s + 1) for s in range(n_xovers)]
    if xover_rows[-1] >= bot_ep_row:
        xover_rows = [top_ep_row + 2 + s * 2 for s in range(n_xovers)]
    if xover_rows[-1] >= bot_ep_row:
        return None

    grid = np.zeros((h, w), dtype=int)
    slot_wire, bypass_wire = _build_wire_network(
        grid, n, ch_cols, bypass_col, top_ep_row, bot_ep_row, crossovers, xover_rows
    )

    inv_perm = [0] * n
    for i in range(n):
        inv_perm[perm[i]] = i
    if any(slot_wire[j] != inv_perm[j] for j in range(n)) or bypass_wire is not None:
        return None

    inp = _derive_input(grid, n, ch_cols, perm, token_colors, top_token_row, bot_token_row)
    try:
        out = main(inp)
    except Exception:
        return None

    metrics = _analyze_fused_behavior(inp, out)
    if metrics["solid_source_count"] < 1 or metrics["x_source_count"] < 1:
        return None
    if metrics["active_x_route_count"] < 1:
        return None
    if metrics["route_count"] < 3:
        return None
    if metrics["has_both_structural_colors"] != 1:
        return None
    if metrics["max_segment_index"] < MIN_SEGMENTS_WITH_TWO_DASHES:
        return None
    if metrics["routes_with_three_strides"] < 1:
        return None
    if metrics["painted_wire_cells"] < n * 5:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
