# Variant of a47bf94d
# Dimension: D7 — 传输目标与停止条件 (dst)
# Change: 管道对端端点（pipeline far endpoint）——token的互补形状传输到同一管道的另一端点，距端点2格处... → 堆叠式放置

import random
import numpy as np

from arc_utils import components_mask, get_bg, neighbors4


# Design analysis for the D7 stacking variant
# -------------------------------------------
# a) Pixel-level behavior
#    The original task writes the transmitted complementary token directly onto the far endpoint
#    anchor `end + 2*dir`. That is harmless only when the destination anchor is empty or already
#    contains the same expected result. This variant makes the collision visible: when a new token
#    reaches an occupied anchor, it must be placed in the next side slot instead of overwriting.
#    In this task family the generator keeps top/bottom endpoints, so the outgoing direction at the
#    top is upward and at the bottom is downward. We therefore stack clockwise relative to that
#    outward direction: top arrivals extend to the right, bottom arrivals extend to the left.
#    The stack step is 3 cells between centers, which keeps 3x3 X/diamond patterns disjoint.
#
#    Concrete trace:
#    - Old rule: top has a solid block, bottom has an X of the same color on the connected endpoint.
#      The solid becomes a top diamond and sends an X to the bottom; the bottom X stays and sends a
#      diamond to the top. Both transmitted shapes are written onto already occupied anchors, so the
#      later write just overwrites the existing identical shape and nothing new is visible.
#    - New rule: the same input now produces two top diamonds (base anchor, then one slot to the
#      right) and two bottom Xs (base anchor, then one slot to the left). The wire logic is the
#      same; only destination conflict handling changes.
#
# b) Complexity calibration
#    To make stacking observable without exploding the state space, the generator uses only three
#    input regimes per channel:
#    - solid-only: top solid, bottom empty
#    - X-with-prefill: top diamond already present, bottom X sends another diamond into it
#    - double: top solid and bottom X, causing stacking on both sides
#    This keeps the combinatorics small while producing repeated evidence for the new destination rule.
#
# c) Hidden assumptions broken by D7
#    The original `main()` depended on "destination is effectively single-use":
#    - `out[rr, cc2] = col` inside the transmit loops assumed empty or harmless overwrite.
#    - No explicit ordering was needed because repeated writes at the same anchor collapsed to one
#      visible shape anyway.
#    - Generator spacing 4-5 was enough because only one 3x3 output shape occupied each endpoint.
#    In the variant these assumptions fail. The solver now searches for the first empty stack slot,
#    and the generator widens channel spacing to 6 (with 3-4 channels) so stacked 3x3 shapes remain
#    visually separate and never overlap neighboring channels.
#
# d) Placement legality
#    Every output endpoint must have room for the base token and, in stacked cases, one extra token
#    three columns to the side. The generator therefore:
#    - uses channel spacing 6
#    - keeps a left margin of 3-4 so bottom-left stacks stay in bounds
#    - sizes the canvas wide enough for the rightmost top stack
#    - validates every sample by running `main(input)` and comparing with the constructed output
#
# e) Visual observability
#    The new behavior is visible because duplicated same-color same-shape groups appear only at
#    occupied destinations. The stacking direction is tied to endpoint side, not color ids or scan
#    order. Training examples repeatedly show top duplicates extending rightward and bottom
#    duplicates extending leftward, which is inferable from the pixels alone.
#
# f) Visual cleanliness
#    Only one extra token per endpoint is generated. That avoids tangled piles while still making
#    the "do not overwrite; place beside" rule obvious.
#
# g) Color inference
#    Colors are unchanged. The variant does not add any color remapping; color still identifies the
#    channel/token identity, and stacking only changes multiplicity and placement.


_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID_OFF = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]
_STACK_STEP = 3


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
            if maximal:
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
            if isolated:
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
            center_col = int(sub[0, 1])
            if center_col == bg:
                continue
            if np.any(sub[mask] != center_col) or np.any(sub[~mask] != bg):
                continue
            isolated = True
            for dr0, dc0 in _DIAMOND_OFF:
                rr, cc = r0 + 1 + dr0, c0 + 1 + dc0
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                        if int(g[nr, nc]) == center_col:
                            isolated = False
                            break
                if not isolated:
                    break
            if isolated:
                found[(r0 + 1, c0 + 1)] = (center_col, (r0, c0))
    return found


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=4)]


def _find_9rects(g: np.ndarray) -> list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]]:
    """Find color-9 rectangular blocks and their wire connections per face."""
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
                if (nr, nc) not in cells and 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) in (5, 8):
                    face = "N" if nr < r0 else "S" if nr > r1 else "W" if nc < c0 else "E"
                    faces.setdefault(face, []).append((nr, nc))
        rects.append((cells, faces))
    return rects


_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DIR2FACE = {(1, 0): "N", (-1, 0): "S", (0, 1): "W", (0, -1): "E"}
_FACE2DIR = {"S": (1, 0), "N": (-1, 0), "E": (0, 1), "W": (0, -1)}


def _trace_wire(
    comp: set[tuple[int, int]],
    start: tuple[int, int],
    dir0: tuple[int, int],
    h: int,
    w: int,
    rects_9: list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Greedy wire following with color-9 rectangle pass-through."""
    cur, d, prev = start, dir0, None
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 4):
        if (cur, d) in visited:
            raise AssertionError("wire trace looped — should not happen with 9-rect pass-through")
        visited.add((cur, d))

        jumped = False
        for cells9, faces9 in rects_9:
            if cur in cells9:
                entry_face = _DIR2FACE[d]
                opp = _OPP[entry_face]
                if opp in faces9:
                    exit_face = opp
                else:
                    others = [f for f in faces9 if f != entry_face]
                    assert len(others) == 1
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
            return cur, d

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
            return cur, d

        prev = cur
        cur = p
        d = (dr, dc)

    raise AssertionError("wire trace exceeded step limit")


def _shape_cells(center: tuple[int, int], offsets: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr, cc = center
    return [(cr + dr, cc + dc) for dr, dc in offsets]


def _stack_dir(outward: tuple[int, int]) -> tuple[int, int]:
    """Clockwise side-step from the endpoint's outward direction."""
    dr, dc = outward
    return (dc, -dr)


def _place_stacked_shape(
    out: np.ndarray,
    bg: int,
    base_center: tuple[int, int],
    outward: tuple[int, int],
    offsets: list[tuple[int, int]],
    color: int,
) -> tuple[int, int] | None:
    """Place on the first empty stack slot; slot 0 is the original anchor."""
    h, w = out.shape
    sdr, sdc = _stack_dir(outward)
    for k in range(max(h, w) + 1):
        center = (base_center[0] + k * _STACK_STEP * sdr, base_center[1] + k * _STACK_STEP * sdc)
        cells = _shape_cells(center, offsets)
        if not all(0 <= r < h and 0 <= c < w for r, c in cells):
            continue
        if all(int(out[r, c]) == bg for r, c in cells):
            for r, c in cells:
                out[r, c] = color
            return center
    return None


def _stamp_token(grid: np.ndarray, center_r: int, center_c: int, color: int, offsets: list[tuple[int, int]]) -> None:
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route complementary token shapes through the wire network, stacking them beside occupied targets.

    Input structure:
    - Background cells are empty; non-background cells form the wire network plus nearby 3x3 tokens.
    - Tokens are solid 3x3 blocks, X patterns, and sometimes pre-placed diamonds already sitting at an endpoint anchor.
    - Each token center is still two cells away from a degree-1 wire endpoint, exactly as in the base task.

    Rule:
    - Every solid block is replaced in place by a diamond.
    - Every solid emits an X to the far endpoint, and every X emits a diamond to the far endpoint.
    - If the far endpoint anchor is occupied, the arriving shape is not overwritten there; instead it is placed in the first empty side slot of that endpoint's stack.

    Key insight: The wire tracing is unchanged; only the destination write changes from a blind overwrite
    to endpoint-local stacking, with top endpoints growing rightward and bottom endpoints leftward.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)
    diamonds = _find_diamond_3x3(g, bg)

    track_mask = g != bg
    for _center, (_col, (r0, c0)) in solids.items():
        track_mask[r0 : r0 + 3, c0 : c0 + 3] = False
    for cr, cc in xs.keys():
        for dr, dc in _X_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                track_mask[rr, cc2] = False
    for cr, cc in diamonds.keys():
        for dr, dc in _DIAMOND_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                track_mask[rr, cc2] = False

    comps = _components(track_mask)
    rects_9 = _find_9rects(g)
    out = g.copy()

    for (cr, cc), (col, (r0, c0)) in solids.items():
        out[r0 : r0 + 3, c0 : c0 + 3] = bg
        for dr, dc in _DIAMOND_OFF:
            out[cr + dr, cc + dc] = col

    deliveries: list[tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]], int, tuple[int, int]]] = []
    for pts in comps:
        comp = set(pts)
        if len(comp) < 4:
            continue

        comp_colors = {int(g[r, c]) for r, c in comp}
        deg = {p: 0 for p in comp}
        for r, c in comp:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in comp:
                    deg[(r, c)] += 1
        endpoints = sorted(p for p in comp if deg[p] <= 1)
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
                    if col in comp_colors:
                        continue
                    end, dd = _trace_wire(comp, ep, (-odr, -odc), h, w, rects_9)
                    deliveries.append((cen, (end[0] + 2 * dd[0], end[1] + 2 * dd[1]), _X_OFF, col, dd))
                elif cen in xs:
                    col, _ = xs[cen]
                    if col in comp_colors:
                        continue
                    end, dd = _trace_wire(comp, ep, (-odr, -odc), h, w, rects_9)
                    deliveries.append((cen, (end[0] + 2 * dd[0], end[1] + 2 * dd[1]), _DIAMOND_OFF, col, dd))

    for _src, dest, offsets, col, outward in sorted(deliveries, key=lambda item: item[0]):
        _place_stacked_shape(out, bg, dest, outward, offsets, col)

    return out


# --------------- generator (stacked-endpoint variant) ---------------

CHANNEL_COUNTS = [3, 4]
CHANNEL_SPACING = [6]
LEFT_MARGIN_RANGE = (3, 4)
BYPASS_COL_OFFSET = 2
TOKEN_ROW_OPTIONS = [3]
TOKEN_COLORS_POOL = [1, 2, 3, 4, 6, 7]
GRID_H_RANGE = (20, 26)
GRID_W_MIN = 27
GRID_W_MAX = 29
EP_TOKEN_GAP = 2

RECT_COUNT_WEIGHTS = [12, 40, 30, 18]
RECT_WIRE_COUNT_WEIGHTS = [50, 38, 12]
RECT_PARALLEL_PROB = 0.75
RECT_H_RANGE = (2, 5)
RECT_MAX_WIDTH = 8
RECT_RETRY_FACTOR = 5
RECT_MARGIN_GAP = 1

GEN_MAX_RETRIES = 300

MODE_SOLID = "solid"
MODE_X_STACK = "x_stack"
MODE_DOUBLE = "double"


def _decompose_to_crossovers(perm: list[int]) -> list[tuple[int, int]]:
    """Decompose a permutation into sequential crossover moves using a bypass column."""
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


def _try_place_9rect(
    grid: np.ndarray,
    ch_cols: list[int],
    xover_rows: list[int],
    top_ep_row: int,
    bot_ep_row: int,
    n_channels: int,
    placed_rects: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    """Attempt to place one color-9 rectangle over 1-3 existing wires."""
    h, w = grid.shape
    target_wires = random.choices([1, 2, 3], weights=RECT_WIRE_COUNT_WEIGHTS)[0]

    if target_wires >= 2 and random.random() < RECT_PARALLEL_PROB and n_channels >= 2:
        mode = "parallel"
    elif target_wires >= 2 and xover_rows:
        mode = "crossing"
    else:
        mode = "single"

    if mode == "parallel":
        j0 = random.randint(0, n_channels - 2)
        c_lo, c_hi = ch_cols[j0], ch_cols[j0 + 1]
        margin_l, margin_r = random.randint(0, 1), random.randint(0, 1)
    else:
        j0 = random.randint(0, n_channels - 1)
        c_lo = c_hi = ch_cols[j0]
        margin_l, margin_r = random.randint(0, 2), random.randint(0, 2)
        if margin_l + margin_r < 1:
            if random.random() < 0.5:
                margin_l = 1
            else:
                margin_r = 1

    rect_c0, rect_c1 = c_lo - margin_l, c_hi + margin_r
    if rect_c1 - rect_c0 + 1 > RECT_MAX_WIDTH or rect_c0 < 0 or rect_c1 >= w:
        return None

    max_h = min(RECT_H_RANGE[1], bot_ep_row - top_ep_row - 4)
    if max_h < RECT_H_RANGE[0]:
        return None
    rect_h = random.randint(RECT_H_RANGE[0], max_h)
    if mode == "crossing" and xover_rows:
        xr = random.choice(xover_rows)
        r_start = xr - random.randint(0, rect_h - 1)
    else:
        r_start = random.randint(top_ep_row + 2, bot_ep_row - rect_h - 1)
    rect_r0, rect_r1 = r_start, r_start + rect_h - 1

    wire_cols = sorted(c for c in ch_cols if rect_c0 <= c <= rect_c1)
    if not wire_cols:
        return None
    for c in wire_cols:
        if not np.all(grid[rect_r0 : rect_r1 + 1, c] == 8):
            return None

    sub = grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1]
    if not np.all((sub == 0) | (sub == 5) | (sub == 8)):
        return None

    for pr0, pc0, pr1, pc1 in placed_rects:
        if rect_r0 <= pr1 + RECT_MARGIN_GAP and pr0 <= rect_r1 + RECT_MARGIN_GAP and rect_c0 <= pc1 + RECT_MARGIN_GAP and pc0 <= rect_c1 + RECT_MARGIN_GAP:
            return None

    if rect_r0 <= top_ep_row or rect_r1 >= bot_ep_row:
        return None
    for c in wire_cols:
        if grid[rect_r0 - 1, c] != 8 or grid[rect_r1 + 1, c] != 8:
            return None

    saved = grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1].copy()
    grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1] = 9

    n_cols, s_cols, w_rows, e_rows = set(), set(), set(), set()
    for c in range(rect_c0, rect_c1 + 1):
        if rect_r0 - 1 >= 0 and grid[rect_r0 - 1, c] in (5, 8):
            n_cols.add(c)
        if rect_r1 + 1 < h and grid[rect_r1 + 1, c] in (5, 8):
            s_cols.add(c)
    for r in range(rect_r0, rect_r1 + 1):
        if rect_c0 - 1 >= 0 and grid[r, rect_c0 - 1] in (5, 8):
            w_rows.add(r)
        if rect_c1 + 1 < w and grid[r, rect_c1 + 1] in (5, 8):
            e_rows.add(r)
    total_wires = len(n_cols & s_cols) + len(w_rows & e_rows)

    if total_wires < 1 or total_wires > 3:
        grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1] = saved
        return None
    return (rect_r0, rect_c0, rect_r1, rect_c1)


def _build_wire_network(
    grid: np.ndarray,
    n_channels: int,
    ch_cols: list[int],
    bypass_col: int,
    top_ep_row: int,
    bot_ep_row: int,
    crossovers: list[tuple[int, int]],
    xover_rows: list[int],
) -> tuple[list[int | None], int | None]:
    """Draw the wire network; return bottom-slot source mapping and bypass state."""
    slot_wire: list[int | None] = list(range(n_channels))
    bypass_wire: int | None = None

    col_seg_start = {}
    for j in range(n_channels):
        col_seg_start[ch_cols[j]] = top_ep_row
        grid[top_ep_row, ch_cols[j]] = 8

    for xover_idx, (src_slot, dst_slot) in enumerate(crossovers):
        xrow = xover_rows[xover_idx]
        if src_slot == -1:
            src_col, dst_col = bypass_col, ch_cols[dst_slot]
        elif dst_slot == -1:
            src_col, dst_col = ch_cols[src_slot], bypass_col
        else:
            src_col, dst_col = ch_cols[src_slot], ch_cols[dst_slot]

        active_cols = {ch_cols[j] for j in range(n_channels) if slot_wire[j] is not None}
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

    for j in range(n_channels):
        if slot_wire[j] is not None:
            c = ch_cols[j]
            if c in col_seg_start:
                start = col_seg_start[c] + 1
                if start <= bot_ep_row:
                    grid[start : bot_ep_row + 1, c] = 8

    return slot_wire, bypass_wire


def _choose_modes(n_channels: int) -> dict[int, str]:
    """Ensure each sample contains visible stacking without overloading the grid."""
    order = list(range(n_channels))
    random.shuffle(order)
    n_double = 1 if n_channels == 3 else random.choice([1, 2])
    doubles = set(order[:n_double])
    remaining = order[n_double:]
    n_x_stack = 0 if not remaining else random.choice([0, 1])
    x_stacks = set(remaining[:n_x_stack])

    modes = {}
    for i in range(n_channels):
        if i in doubles:
            modes[i] = MODE_DOUBLE
        elif i in x_stacks:
            modes[i] = MODE_X_STACK
        else:
            modes[i] = MODE_SOLID
    return modes


def _build_expected_output(
    grid: np.ndarray,
    n_channels: int,
    ch_cols: list[int],
    perm: list[int],
    token_colors: list[int],
    top_token_row: int,
    bot_token_row: int,
    modes: dict[int, str],
) -> np.ndarray:
    out = grid.copy()
    bg = 0
    for i in range(n_channels):
        top_center = (top_token_row, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[perm[i]])
        color = token_colors[i]
        _stamp_token(out, top_center[0], top_center[1], color, _DIAMOND_OFF)
        _stamp_token(out, bot_center[0], bot_center[1], color, _X_OFF)

    for i in range(n_channels):
        top_center = (top_token_row, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[perm[i]])
        color = token_colors[i]
        mode = modes[i]
        if mode in (MODE_X_STACK, MODE_DOUBLE):
            _place_stacked_shape(out, bg, top_center, (-1, 0), _DIAMOND_OFF, color)
        if mode == MODE_DOUBLE:
            _place_stacked_shape(out, bg, bot_center, (1, 0), _X_OFF, color)
    return out


def _derive_input(
    grid: np.ndarray,
    n_channels: int,
    ch_cols: list[int],
    perm: list[int],
    token_colors: list[int],
    top_token_row: int,
    bot_token_row: int,
    modes: dict[int, str],
) -> np.ndarray:
    inp = grid.copy()
    for i in range(n_channels):
        top_center = (top_token_row, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[perm[i]])
        color = token_colors[i]
        mode = modes[i]
        if mode in (MODE_SOLID, MODE_DOUBLE):
            _stamp_token(inp, top_center[0], top_center[1], color, _SOLID_OFF)
        if mode in (MODE_X_STACK, MODE_DOUBLE):
            _stamp_token(inp, bot_center[0], bot_center[1], color, _X_OFF)
        if mode == MODE_X_STACK:
            _stamp_token(inp, top_center[0], top_center[1], color, _DIAMOND_OFF)
    return inp


def _gen_attempt() -> np.ndarray | None:
    n_channels = random.choice(CHANNEL_COUNTS)
    spacing = random.choice(CHANNEL_SPACING)

    left_margin = random.randint(*LEFT_MARGIN_RANGE)
    first_col = left_margin + 1
    ch_cols = [first_col + i * spacing for i in range(n_channels)]
    bypass_col = ch_cols[-1] + BYPASS_COL_OFFSET

    needed_w = max(bypass_col + 3, ch_cols[-1] + (_STACK_STEP + 2), GRID_W_MIN)
    if needed_w > GRID_W_MAX:
        return None
    w = random.randint(needed_w, GRID_W_MAX)

    top_token_row = random.choice(TOKEN_ROW_OPTIONS)
    top_ep_row = top_token_row + EP_TOKEN_GAP
    token_colors = random.sample(TOKEN_COLORS_POOL, n_channels)
    modes = _choose_modes(n_channels)

    while True:
        perm = list(range(n_channels))
        random.shuffle(perm)
        if perm != list(range(n_channels)):
            break

    crossovers = _decompose_to_crossovers(perm)
    if not crossovers:
        return None
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
    xover_rows = [top_ep_row + 1 + step * (k + 1) for k in range(n_xovers)]
    if xover_rows[-1] >= bot_ep_row:
        xover_rows = [top_ep_row + 2 + k * 2 for k in range(n_xovers)]
    if xover_rows[-1] >= bot_ep_row:
        return None

    grid = np.zeros((h, w), dtype=int)
    slot_wire, bypass_wire = _build_wire_network(
        grid, n_channels, ch_cols, bypass_col, top_ep_row, bot_ep_row, crossovers, xover_rows
    )

    inv_perm = [0] * n_channels
    for i in range(n_channels):
        inv_perm[perm[i]] = i
    if any(slot_wire[j] != inv_perm[j] for j in range(n_channels)) or bypass_wire is not None:
        return None

    n_rects = random.choices(range(4), weights=RECT_COUNT_WEIGHTS)[0]
    placed_rects: list[tuple[int, int, int, int]] = []
    for _ in range(n_rects * RECT_RETRY_FACTOR):
        if len(placed_rects) >= n_rects:
            break
        result = _try_place_9rect(grid, ch_cols, xover_rows, top_ep_row, bot_ep_row, n_channels, placed_rects)
        if result is not None:
            placed_rects.append(result)

    expected = _build_expected_output(
        grid, n_channels, ch_cols, perm, token_colors, top_token_row, bot_token_row, modes
    )
    inp = _derive_input(
        grid, n_channels, ch_cols, perm, token_colors, top_token_row, bot_token_row, modes
    )
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
