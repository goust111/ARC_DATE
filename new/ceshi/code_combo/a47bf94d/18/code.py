# Combo variant of a47bf94d (index 18)
# combo_spec: D6/3+D17/2
# Name: 单像素源+最小化token

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior shift:
#    Original a47bf94d uses 3x3 solids/X-shapes placed two cells outside wire
#    endpoints. A solid keeps a local remainder and emits a complementary 3x3
#    shape to the far end. In this variant the source token shrinks to a single
#    colored pixel at the same "center" location, so there is no local shape
#    decomposition at all. Example: if a red source pixel sits above a top
#    endpoint, the solver traces that wire to the matching bottom endpoint. If
#    the route touches no maroon junction rectangle, it creates a red diamond at
#    the far center; if it passes through one maroon rectangle, it creates a red
#    plus-cross there instead. The source pixel disappears.
# b) Complexity calibration:
#    The output shape has only two states: diamond versus plus-cross. This keeps
#    the rule learnable, while the wire permutation and the presence/absence of
#    a junction on different routes still provide visible variety. Every
#    generated grid is forced to contain both route types so the new control
#    variable is observable.
# c) Implicit assumptions that changed:
#    - The old solver assumed tokens occupied recognisable 3x3 footprints.
#      That no longer holds, so token detection must look for single colored
#      pixels exactly two cells outside endpoints.
#    - The old solver removed token cells from the track mask shape-by-shape.
#      Here the track mask is simply the wire colors {5,8,9}; all other colors
#      are sources and never part of the track.
#    - The old output logic depended on token type (solid vs X). The new logic
#      depends on route metadata (junction count), so tracing must return that
#      count together with the far endpoint.
# d) Placement legality:
#    Source pixels stay two cells outside endpoints, preserving the original 3x3
#    clear zone at the destination. Junction rectangles are only embedded into
#    final vertical destination segments below the last crossover, so they do
#    not overlap tokens, endpoints, or other junctions.
# e) Visual observability:
#    A human can compare two channels: same kind of colored source pixel, same
#    routing idea, but one path visibly enters a maroon rectangle and yields a
#    plus, while another bypasses all maroon rectangles and yields a diamond.
#    No hidden ordering by color id is used.
# f) Visual cleanliness:
#    The generator keeps all sources on the top side and all generated shapes on
#    the bottom side. This avoids mixing source pixels with output shapes in the
#    same area. Junction rectangles are width-3 vertical blocks on isolated
#    columns, so they remain easy to distinguish from wires.
# g) Inference support:
#    The only color transformation is identity: the source pixel's color is
#    preserved in the far-end shape. The human only has to infer location and
#    shape, not any arbitrary recoloring.

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_WIRE_COLORS = {5, 8, 9}
_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_PLUS_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DIR2FACE = {(1, 0): "N", (-1, 0): "S", (0, 1): "W", (0, -1): "E"}
_FACE2DIR = {"S": (1, 0), "N": (-1, 0), "E": (0, 1), "W": (0, -1)}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=4)]


def _find_9rects(g: np.ndarray) -> list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]]:
    """Find maroon rectangles and the wire-adjacent cells on each exposed face."""
    mask9 = g == 9
    if not mask9.any():
        return []

    h, w = g.shape
    rects = []
    for comp9 in components_mask(mask9, connectivity=4):
        cells = set((int(r), int(c)) for r, c in comp9)
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        faces: dict[str, list[tuple[int, int]]] = {}
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in cells or not (0 <= nr < h and 0 <= nc < w):
                    continue
                if int(g[nr, nc]) not in _WIRE_COLORS - {9}:
                    continue
                face = "N" if nr < r0 else "S" if nr > r1 else "W" if nc < c0 else "E"
                faces.setdefault(face, []).append((nr, nc))
        rects.append((cells, faces))
    return rects


def _trace_wire(
    comp: set[tuple[int, int]],
    start: tuple[int, int],
    dir0: tuple[int, int],
    h: int,
    w: int,
    rects_9: list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]],
) -> tuple[tuple[int, int], tuple[int, int], int]:
    """
    Follow one logical wire strand through a connected wire component.

    Returns:
    - the terminal endpoint cell
    - the outgoing direction at that endpoint
    - how many color-9 junction rectangles were traversed
    """
    cur, d, prev = start, dir0, None
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    junction_count = 0

    for _ in range(h * w * 4):
        state = (cur, d)
        if state in visited:
            raise AssertionError("wire trace looped unexpectedly")
        visited.add(state)

        jumped = False
        for cells9, faces9 in rects_9:
            if cur not in cells9:
                continue
            junction_count += 1
            entry_face = _DIR2FACE[d]
            opp = _OPP[entry_face]
            if opp in faces9:
                exit_face = opp
            else:
                others = [f for f in faces9 if f != entry_face]
                if len(others) != 1:
                    raise AssertionError("invalid 9-rect routing")
                exit_face = others[0]
            pts = faces9[exit_face]
            er, ec = cur
            if entry_face in ("N", "S"):
                exit_cell = min(pts, key=lambda p: abs(p[1] - ec))
            else:
                exit_cell = min(pts, key=lambda p: abs(p[0] - er))
            cur, d, prev = exit_cell, _FACE2DIR[exit_face], None
            jumped = True
            break
        if jumped:
            continue

        r, c = cur
        opts = [((nr, nc), nr - r, nc - c) for nr, nc in neighbors4(r, c) if (nr, nc) in comp]
        if not opts:
            return cur, d, junction_count

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
            return cur, d, junction_count

        prev = cur
        cur = p
        d = (dr, dc)

    raise AssertionError("wire trace exceeded step limit")


def _stamp_shape(grid: np.ndarray, center: tuple[int, int], color: int, offsets: list[tuple[int, int]]) -> None:
    h, w = grid.shape
    cr, cc = center
    for dr, dc in offsets:
        rr, cc2 = cr + dr, cc + dc
        if 0 <= rr < h and 0 <= cc2 < w:
            grid[rr, cc2] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Send single-pixel source tokens through the wire network and materialize a
    fixed receiving shape at the far endpoint.

    Input structure:
    - Background cells are empty.
    - Wire networks are made of colors 8 and 5, with optional color-9
      rectangles acting as visible junction blocks.
    - Each source token is exactly one colored pixel placed two cells outside a
      wire endpoint; no 3x3 source shape exists anymore.

    Rule:
    - Remove every source pixel from its original side.
    - Trace the adjacent wire strand to the opposite endpoint.
    - If the traced route touches no color-9 junction block, draw a diamond at
      the far token center; otherwise draw a plus-cross there.

    Key insight:
    - The color is carried unchanged by the wire; only the route metadata
      matters. Here the decisive metadata is whether the path passes through a
      maroon junction rectangle before reaching the other end.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    track_mask = np.isin(g, list(_WIRE_COLORS))
    comps = _components(track_mask)
    rects_9 = _find_9rects(g)

    out = g.copy()
    source_mask = (g != bg) & ~track_mask
    out[source_mask] = bg

    processed_sources: set[tuple[int, int]] = set()

    for pts in comps:
        comp = set(pts)
        if len(comp) < 4:
            continue

        deg = {p: 0 for p in comp}
        for r, c in comp:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in comp:
                    deg[(r, c)] += 1
        endpoints = [p for p in comp if deg[p] <= 1]

        for ep in endpoints:
            er, ec = ep
            for odr, odc in _N4:
                sr, sc = er + 2 * odr, ec + 2 * odc
                mr, mc = er + odr, ec + odc
                if not (0 <= sr < h and 0 <= sc < w and 0 <= mr < h and 0 <= mc < w):
                    continue
                if (sr, sc) in processed_sources:
                    continue
                color = int(g[sr, sc])
                if color == bg or color in _WIRE_COLORS:
                    continue
                if int(g[mr, mc]) != bg:
                    continue

                processed_sources.add((sr, sc))
                end, dd, junction_count = _trace_wire(comp, ep, (-odr, -odc), h, w, rects_9)
                dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                offsets = _DIAMOND_OFF if junction_count == 0 else _PLUS_OFF
                _stamp_shape(out, dest, color, offsets)

    return out


# --------------- generator ---------------

CHANNEL_COUNTS = [4, 5]
TOKEN_COLORS_POOL = [1, 2, 3, 4, 6, 7]
TOKEN_ROW_OPTIONS = [2, 3]
LEFT_MARGIN_RANGE = (2, 3)
GRID_H_RANGE = (22, 26)
GRID_W_MIN = 22
GRID_W_MAX = 26
EP_TOKEN_GAP = 2
BYPASS_COL_OFFSET = 2
GEN_MAX_RETRIES = 200


def _decompose_to_crossovers(perm: list[int]) -> list[tuple[int, int]]:
    """Decompose a permutation into crossover moves using one bypass column."""
    n = len(perm)
    visited = [False] * n
    crossovers: list[tuple[int, int]] = []
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
    """Draw the connected wire network and return the slot assignment at the bottom."""
    slot_wire: list[int | None] = list(range(n))
    bypass_wire: int | None = None

    col_seg_start = {}
    for j in range(n):
        col_seg_start[ch_cols[j]] = top_ep_row
        grid[top_ep_row, ch_cols[j]] = 8

    for xrow, (src_slot, dst_slot) in zip(xover_rows, crossovers):
        if src_slot == -1:
            src_col, dst_col = bypass_col, ch_cols[dst_slot]
        elif dst_slot == -1:
            src_col, dst_col = ch_cols[src_slot], bypass_col
        else:
            src_col, dst_col = ch_cols[src_slot], ch_cols[dst_slot]

        active_cols = {ch_cols[j] for j in range(n) if slot_wire[j] is not None}
        if bypass_wire is not None:
            active_cols.add(bypass_col)

        for c in active_cols:
            if c in col_seg_start:
                start = col_seg_start[c] + 1
                if start <= xrow - 1:
                    grid[start:xrow, c] = 8

        c_min, c_max = min(src_col, dst_col), max(src_col, dst_col)
        grid[xrow, c_min : c_max + 1] = 8
        for c in active_cols:
            grid[xrow, c] = 8

        for c in active_cols:
            if c == src_col or c == dst_col:
                continue
            if c_min < c < c_max:
                if c - 1 >= c_min:
                    grid[xrow, c - 1] = 5
                if c + 1 <= c_max:
                    grid[xrow, c + 1] = 5

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

    for j in range(n):
        if slot_wire[j] is not None:
            c = ch_cols[j]
            if c in col_seg_start:
                start = col_seg_start[c] + 1
                if start <= bot_ep_row:
                    grid[start : bot_ep_row + 1, c] = 8

    return slot_wire, bypass_wire


def _place_bottom_junction(
    grid: np.ndarray,
    col: int,
    band_top: int,
    band_bottom: int,
) -> tuple[int, int, int, int] | None:
    """Embed a width-3 maroon rectangle into a final vertical destination segment."""
    h, w = grid.shape
    if col - 1 < 0 or col + 1 >= w or band_top > band_bottom:
        return None

    max_height = min(4, band_bottom - band_top + 1)
    if max_height < 2:
        return None

    rect_h = random.randint(2, max_height)
    r0_max = band_bottom - rect_h + 1
    if r0_max < band_top:
        return None
    r0 = random.randint(band_top, r0_max)
    r1 = r0 + rect_h - 1
    c0, c1 = col - 1, col + 1

    if not np.all(grid[r0 : r1 + 1, col] == 8):
        return None
    if grid[r0 - 1, col] != 8 or grid[r1 + 1, col] != 8:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all((sub == 0) | (sub == 8)):
        return None

    grid[r0 : r1 + 1, c0 : c1 + 1] = 9
    return r0, c0, r1, c1


def _stamp_output_shape(output: np.ndarray, center: tuple[int, int], color: int, has_junction: bool) -> None:
    offsets = _PLUS_OFF if has_junction else _DIAMOND_OFF
    _stamp_shape(output, center, color, offsets)


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(CHANNEL_COUNTS)
    spacing = 4 if n == 5 else random.choice([4, 5])

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

    while True:
        perm = list(range(n))
        random.shuffle(perm)
        if perm != list(range(n)):
            break

    crossovers = _decompose_to_crossovers(perm)
    if not crossovers:
        return None

    min_wire_rows = len(crossovers) * 2 + 6
    h_min = top_ep_row + min_wire_rows + top_token_row + 3
    if h_min > GRID_H_RANGE[1]:
        return None
    h = random.randint(max(GRID_H_RANGE[0], h_min), GRID_H_RANGE[1])

    bot_token_row = h - top_token_row - 1
    bot_ep_row = bot_token_row - EP_TOKEN_GAP
    if bot_ep_row - top_ep_row < min_wire_rows:
        return None

    avail = bot_ep_row - top_ep_row - 2
    step = max(2, avail // (len(crossovers) + 1))
    xover_rows = [top_ep_row + 1 + step * (idx + 1) for idx in range(len(crossovers))]
    if xover_rows[-1] >= bot_ep_row - 2:
        xover_rows = [top_ep_row + 2 + idx * 2 for idx in range(len(crossovers))]
    if xover_rows[-1] >= bot_ep_row - 2:
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

    junction_slots = set(random.sample(range(n), random.randint(1, n - 1)))
    band_top = xover_rows[-1] + 2
    band_bottom = bot_ep_row - 2
    if band_top > band_bottom:
        return None

    for slot in junction_slots:
        if _place_bottom_junction(grid, ch_cols[slot], band_top, band_bottom) is None:
            return None

    inp = grid.copy()
    for i in range(n):
        inp[top_token_row, ch_cols[i]] = token_colors[i]

    expected = grid.copy()
    for i in range(n):
        dest_slot = perm[i]
        center = (bot_token_row, ch_cols[dest_slot])
        _stamp_output_shape(expected, center, token_colors[i], dest_slot in junction_slots)

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    plus_count = 0
    diamond_count = 0
    for i in range(n):
        dest_slot = perm[i]
        if dest_slot in junction_slots:
            plus_count += 1
        else:
            diamond_count += 1
    if plus_count == 0 or diamond_count == 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
