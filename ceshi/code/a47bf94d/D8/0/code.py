# Variant of a47bf94d
# Dimension: D8 — 9号矩形块的角色 (obstacle)
# Change: 直通连接节点——9号颜色的矩形块（2x2到5x4）嵌入管道中，管道从入口面直线穿过到对面出口面（仅当单入单出在相邻面时才... → 信号变换节点

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

# Design analysis for the D8 variant
# ----------------------------------
# a) Concrete behavior:
#    - Keep the same visible vocabulary: tokens are still 3x3 solid blocks or 3x3 X-patterns,
#      wires are still 8/5 networks, and maroon 9-regions are still embedded rectangles.
#    - The new observable effect is on the transmitted far-end token only. A maroon rectangle
#      now acts like a signal inverter: every time a routed token passes through one 9-rectangle,
#      the far-end emission swaps between X and diamond.
#    - Small trace example:
#      original: solid source -> local diamond, far X.
#      variant through one 9-node: solid source -> local diamond, far diamond.
#      original: X source -> local X, far diamond.
#      variant through one 9-node: X source -> local X, far X.
#      The local behavior stays familiar; only the transmitted signal changes.
# b) Complexity calibration:
#    - The transform has only two states: untouched vs inverted. This is enough to be visible but
#      not too combinatorial.
#    - The solver supports multiple 9-nodes by toggling parity, but the generator restricts each
#      channel path to at most one encountered 9-rectangle so the examples show a clean binary rule.
#    - Each generated grid must contain both transformed and untransformed channels, so the effect
#      is contrastive within a single example instead of relying on cross-example memorization.
# c) Hidden assumptions audited:
#    - `_trace_wire()` previously assumed 9-rectangles only changed geometry. In the variant the
#      same geometry rule still holds, but shape emission can no longer be hard-coded at the callsite.
#    - The original routing loop unconditionally wrote X for solids and diamond for X tokens. That
#      becomes wrong once far-shape depends on path history, so the callsite now reads a transform
#      parity returned by `_trace_wire()`.
#    - The in-place rewrite for solid tokens remains safe because the variant does not change local
#      stabilization; only the far-end token is conditional.
# d) Placement legality:
#    - Tokens remain centered exactly two cells away from degree-1 endpoints, so they do not overlap
#      the wire network or the 9-rectangles.
#    - Rectangles are still placed by replacing existing wire/crossing cells only, never extending
#      beyond the route corridor. The generator additionally rejects layouts where a channel would
#      pass through more than one 9-rectangle, because that would obscure the intended binary cue.
# e) Visual observability:
#    - The rule is visible without using hidden ids: channels that cross a maroon rectangle produce
#      the same rest-shape on both ends; channels that avoid maroon keep the original complementary
#      behavior. This can be inferred directly from geometry and shape, not from palette order.
# f) Visual cleanliness:
#    - The generator keeps the original sparse top/bottom token rows and rejects "all changed" or
#      "none changed" layouts. That avoids cluttered all-rect or no-rect cases.
# g) Color-change inferability:
#    - No recoloring is introduced. Token color identity is preserved exactly, so the only new rule
#      the solver must learn is the shape inversion caused by 9-rectangles.
# h) New solver assumptions guaranteed by the generator:
#    - Each logical channel still owns exactly one source token and one destination slot.
#    - Every path touches at most one transforming 9-rectangle, and at least one path touches one.
#    - The generated inputs continue to satisfy the original isolation assumptions for 3x3 solids
#      and X patterns.

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]

_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DIR2FACE = {(1, 0): "N", (-1, 0): "S", (0, 1): "W", (0, -1): "E"}
_FACE2DIR = {"S": (1, 0), "N": (-1, 0), "E": (0, 1), "W": (0, -1)}


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


def _find_diamond_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], int]:
    """Return {center: color} for isolated 3x3 diamonds (4 cardinals, empty center/corners)."""
    h, w = g.shape
    found: dict[tuple[int, int], int] = {}

    mask = np.zeros((3, 3), dtype=bool)
    for rr, cc in [(0, 1), (1, 0), (1, 2), (2, 1)]:
        mask[rr, cc] = True

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            cols = {int(sub[rr, cc]) for rr, cc in [(0, 1), (1, 0), (1, 2), (2, 1)]}
            if len(cols) != 1:
                continue
            col = cols.pop()
            if col == bg:
                continue
            if np.any(sub[mask] != col) or np.any(sub[~mask] != bg):
                continue
            found[(r0 + 1, c0 + 1)] = col
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


def _trace_wire(
    comp: set[tuple[int, int]],
    start: tuple[int, int],
    dir0: tuple[int, int],
    h: int,
    w: int,
    rects_9: list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]],
) -> tuple[tuple[int, int], tuple[int, int], int]:
    """
    Greedy wire following with 9-rect traversal.

    The geometry is unchanged from the base task, but every visited 9-rectangle
    contributes one signal inversion toggle for the emitted far-end token.
    """
    cur, d, prev = start, dir0, None
    flips = 0
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 4):
        if (cur, d) in visited:
            assert False, "wire trace looped — should not happen with 9-rect traversal"
        visited.add((cur, d))

        jumped = False
        for cells9, faces9 in rects_9:
            if cur in cells9:
                flips ^= 1
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
            return cur, d, flips

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
            return cur, d, flips

        prev = cur
        cur = p
        d = (dr, dc)

    assert False, "wire trace exceeded step limit"


def _remote_offsets(source_kind: str, flips: int) -> list[tuple[int, int]]:
    """Return the far-end token shape after routing a source token through `flips` inverters."""
    if source_kind == "solid":
        return _DIAMOND_OFF if flips % 2 else _X_OFF
    return _X_OFF if flips % 2 else _DIAMOND_OFF


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transmit colored tokens through wire networks while maroon rectangles invert the outbound motif.

    Input structure:
    - Background cells surround one or more wire components made of colors 8 and 5, with optional
      embedded color-9 rectangles and 3x3 token patterns placed two cells away from endpoints.

    Rule:
    - A solid 3x3 token still relaxes locally into a diamond, and an X token still stays as an X.
    - The remote token sent along the wire is no longer fixed: each traversed 9-rectangle swaps the
      far-end emission between X and diamond once, so odd/even counts determine the final shape.

    Key insight:
    - Solve exactly as in the base routing task, but attach one extra bit to the traced signal:
      the parity of encountered maroon rectangles controls whether the transmitted complement flips.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)

    track_mask = g != bg
    for (_, _), (_, (r0, c0)) in solids.items():
        track_mask[r0 : r0 + 3, c0 : c0 + 3] = False
    for cr, cc in xs.keys():
        for dr, dc in _X_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                track_mask[rr, cc2] = False

    comps = _components(track_mask)
    rects_9 = _find_9rects(g)
    out = g.copy()

    for (_, _), (col, (r0, c0)) in solids.items():
        out[r0 : r0 + 3, c0 : c0 + 3] = bg
        center = (r0 + 1, c0 + 1)
        for dr, dc in _DIAMOND_OFF:
            rr, cc2 = center[0] + dr, center[1] + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                out[rr, cc2] = col

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
                    if col in comp_colors:
                        continue
                    end, dd, flips = _trace_wire(comp, ep, (-odr, -odc), h, w, rects_9)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    for dr, dc in _remote_offsets("solid", flips):
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col

                elif cen in xs:
                    col, _ = xs[cen]
                    if col in comp_colors:
                        continue
                    end, dd, flips = _trace_wire(comp, ep, (-odr, -odc), h, w, rects_9)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    for dr, dc in _remote_offsets("x", flips):
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col

    return out


# --------------- generator (signal_inverting_9rect) ---------------

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

RECT_COUNT_WEIGHTS = [55, 30, 15]               # weights for 1, 2, 3 rects
RECT_WIRE_COUNT_WEIGHTS = [50, 38, 12]
RECT_PARALLEL_PROB = 0.75
RECT_H_RANGE = (2, 5)
RECT_MAX_WIDTH = 7
RECT_RETRY_FACTOR = 8
RECT_MARGIN_GAP = 1

X_INPUT_COUNT_WEIGHTS = [60, 30, 10]
FAR_DIAMOND_PROB = 0.33

GEN_MAX_RETRIES = 400


def _decompose_to_crossovers(perm: list[int]) -> list[tuple[int, int]]:
    """Decompose a permutation into sequential crossover moves using the bypass column."""
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


def _stamp_token(grid: np.ndarray, center_r: int, center_c: int, color: int, offsets: list[tuple[int, int]]) -> None:
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = color


def _try_place_9rect(
    grid: np.ndarray,
    ch_cols: list[int],
    xover_rows: list[int],
    top_ep_row: int,
    bot_ep_row: int,
    n_channels: int,
    placed_rects: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    """Attempt to place one color-9 rectangular block. Returns (r0, c0, r1, c1) or None."""
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

    rect_h = random.randint(RECT_H_RANGE[0], min(RECT_H_RANGE[1], bot_ep_row - top_ep_row - 4))
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
        if (
            rect_r0 <= pr1 + RECT_MARGIN_GAP
            and pr0 <= rect_r1 + RECT_MARGIN_GAP
            and rect_c0 <= pc1 + RECT_MARGIN_GAP
            and pc0 <= rect_c1 + RECT_MARGIN_GAP
        ):
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
    """Draw the wire network onto grid. Returns (slot_wire, bypass_wire) after all crossovers."""
    slot_wire: list[int | None] = list(range(n_channels))
    bypass_wire: int | None = None

    col_seg_start: dict[int, int] = {}
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


def _analyze_channels(
    grid: np.ndarray,
    ch_cols: list[int],
    top_ep_row: int,
    bot_ep_row: int,
) -> dict[int, dict[str, int]]:
    """Trace each top endpoint to recover its destination column and inverter count."""
    rects_9 = _find_9rects(grid)
    bg = get_bg(grid)
    solids = _find_solid_3x3(grid, bg)
    xs = _find_x_3x3(grid, bg)
    diamonds = _find_diamond_3x3(grid, bg)

    track_mask = grid != bg
    for (_, _), (_, (r0, c0)) in solids.items():
        track_mask[r0 : r0 + 3, c0 : c0 + 3] = False
    for cr, cc in xs.keys():
        for dr, dc in _X_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < grid.shape[0] and 0 <= cc2 < grid.shape[1]:
                track_mask[rr, cc2] = False
    for cr, cc in diamonds.keys():
        for dr, dc in _DIAMOND_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < grid.shape[0] and 0 <= cc2 < grid.shape[1]:
                track_mask[rr, cc2] = False

    comps = [set(comp) for comp in _components(track_mask)]
    comp_by_cell = {}
    for comp in comps:
        for cell in comp:
            comp_by_cell[cell] = comp

    info: dict[int, dict[str, int]] = {}
    for i, c in enumerate(ch_cols):
        ep = (top_ep_row, c)
        if ep not in comp_by_cell:
            raise AssertionError("top endpoint missing from traced component")
        end, dd, flips = _trace_wire(comp_by_cell[ep], ep, (1, 0), grid.shape[0], grid.shape[1], rects_9)
        if end[0] != bot_ep_row or dd != (1, 0) or end[1] not in ch_cols:
            raise AssertionError("wire analysis did not terminate at a bottom endpoint")
        info[i] = {"dest_slot": ch_cols.index(end[1]), "flips": flips}
    return info


def _choose_token_plan(n_channels: int, transformed_slots: list[int]) -> list[dict[str, object]]:
    """Choose source side/kind while ensuring a transformed solid example is present."""
    n_x = random.choices([0, 1, 2], weights=X_INPUT_COUNT_WEIGHTS)[0]
    x_channels = set(random.sample(range(n_channels), min(n_x, n_channels)))
    transformed_x = [slot for slot in transformed_slots if slot in x_channels]
    if transformed_slots and len(transformed_x) == len(transformed_slots):
        x_channels.remove(random.choice(transformed_x))

    plan = []
    for i in range(n_channels):
        plan.append(
            {
                "channel": i,
                "kind": "x" if i in x_channels else "solid",
                "source_side": random.choice(["top", "bottom"]),
            }
        )
    return plan


def _derive_input(
    grid: np.ndarray,
    ch_cols: list[int],
    perm: list[int],
    token_colors: list[int],
    top_token_row: int,
    bot_token_row: int,
    channel_info: dict[int, dict[str, int]],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Derive an input grid and token plan from the routed wire layout."""
    inp = grid.copy()
    transformed_slots = [i for i, info in channel_info.items() if info["flips"] % 2 == 1]
    plan = _choose_token_plan(len(ch_cols), transformed_slots)

    for item in plan:
        i = int(item["channel"])
        color = token_colors[i]
        top_center = (top_token_row, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[perm[i]])
        near = top_center if item["source_side"] == "top" else bot_center
        far = bot_center if item["source_side"] == "top" else top_center

        if item["kind"] == "x":
            _stamp_token(inp, near[0], near[1], color, _X)
            if channel_info[i]["flips"] % 2 == 0 and random.random() < FAR_DIAMOND_PROB:
                _stamp_token(inp, far[0], far[1], color, _DIAMOND)
        else:
            _stamp_token(inp, near[0], near[1], color, _SOLID)

    return inp, plan


def _build_expected_output(
    grid: np.ndarray,
    ch_cols: list[int],
    perm: list[int],
    token_colors: list[int],
    top_token_row: int,
    bot_token_row: int,
    plan: list[dict[str, object]],
    channel_info: dict[int, dict[str, int]],
) -> np.ndarray:
    out = grid.copy()
    for item in plan:
        i = int(item["channel"])
        color = token_colors[i]
        top_center = (top_token_row, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[perm[i]])
        near = top_center if item["source_side"] == "top" else bot_center
        far = bot_center if item["source_side"] == "top" else top_center

        if item["kind"] == "solid":
            _stamp_token(out, near[0], near[1], color, _DIAMOND)
        else:
            _stamp_token(out, near[0], near[1], color, _X)
        _stamp_token(out, far[0], far[1], color, _remote_offsets(str(item["kind"]), channel_info[i]["flips"]))
    return out


def _gen_attempt() -> np.ndarray | None:
    n_channels = random.choice(CHANNEL_COUNTS)
    spacing = random.choice(CHANNEL_SPACING)

    left_margin = random.randint(*LEFT_MARGIN_RANGE)
    first_col = left_margin + 1
    ch_cols = [first_col + i * spacing for i in range(n_channels)]
    bypass_col = ch_cols[-1] + BYPASS_COL_OFFSET

    width = max(bypass_col + 3, GRID_W_MIN)
    if width > GRID_W_MAX:
        return None

    top_token_row = random.choice(TOKEN_ROW_OPTIONS)
    top_ep_row = top_token_row + EP_TOKEN_GAP
    token_colors = random.sample(TOKEN_COLORS_POOL, n_channels)

    while True:
        perm = list(range(n_channels))
        random.shuffle(perm)
        if perm != list(range(n_channels)):
            break

    crossovers = _decompose_to_crossovers(perm)
    if not crossovers:
        return None

    min_wire_rows = len(crossovers) * 2 + 2
    height_min = top_ep_row + min_wire_rows + top_token_row + 3
    if height_min > GRID_H_RANGE[1]:
        return None
    height = random.randint(max(GRID_H_RANGE[0], height_min), GRID_H_RANGE[1])

    bot_token_row = height - top_token_row - 1
    bot_ep_row = bot_token_row - EP_TOKEN_GAP
    if bot_ep_row - top_ep_row < min_wire_rows:
        return None

    avail = bot_ep_row - top_ep_row - 2
    step = max(2, avail // (len(crossovers) + 1))
    xover_rows = [top_ep_row + 1 + step * (s + 1) for s in range(len(crossovers))]
    if xover_rows[-1] >= bot_ep_row:
        xover_rows = [top_ep_row + 2 + s * 2 for s in range(len(crossovers))]
    if xover_rows[-1] >= bot_ep_row:
        return None

    grid = np.zeros((height, width), dtype=int)
    slot_wire, bypass_wire = _build_wire_network(
        grid, n_channels, ch_cols, bypass_col, top_ep_row, bot_ep_row, crossovers, xover_rows
    )

    inv_perm = [0] * n_channels
    for i in range(n_channels):
        inv_perm[perm[i]] = i
    if any(slot_wire[j] != inv_perm[j] for j in range(n_channels)) or bypass_wire is not None:
        return None

    target_rects = random.choices([1, 2, 3], weights=RECT_COUNT_WEIGHTS)[0]
    placed_rects: list[tuple[int, int, int, int]] = []
    for _ in range(target_rects * RECT_RETRY_FACTOR):
        if len(placed_rects) >= target_rects:
            break
        result = _try_place_9rect(grid, ch_cols, xover_rows, top_ep_row, bot_ep_row, n_channels, placed_rects)
        if result is not None:
            placed_rects.append(result)
    if not placed_rects:
        return None

    try:
        channel_info = _analyze_channels(grid, ch_cols, top_ep_row, bot_ep_row)
    except Exception:
        return None

    traced_perm = [channel_info[i]["dest_slot"] for i in range(n_channels)]
    if traced_perm != perm:
        return None

    flip_counts = [channel_info[i]["flips"] for i in range(n_channels)]
    transformed = [i for i, count in enumerate(flip_counts) if count % 2 == 1]
    if any(count > 1 for count in flip_counts):
        return None
    if not transformed or len(transformed) == n_channels:
        return None

    inp, plan = _derive_input(grid, ch_cols, perm, token_colors, top_token_row, bot_token_row, channel_info)
    output = _build_expected_output(
        grid, ch_cols, perm, token_colors, top_token_row, bot_token_row, plan, channel_info
    )
    try:
        if np.array_equal(main(inp), output):
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
