# Variant of a47bf94d
# Dimension: D4 — token形状分解方式 (transform)
# Change: 分解（decomposition）——实心3x3块被分解为十字形（中心+4个正交邻居，留在原位）和四角星形（4个对角角，... → 镜像对称分解
#
# Design analysis for this variant
# a) Concrete behavior:
#    - Keep the original task's channel/wire layout: 4-5 vertical channels, tokens sit two cells above or below
#      degree-1 wire endpoints, and wires still realize a non-identity permutation.
#    - Replace the old rotational decomposition with a mirror decomposition specialized to the existing top/bottom
#      layout. A solid 3x3 block splits across the horizontal midline into two overlapping 2x3 slabs:
#        upper half = top two rows of the 3x3
#        lower half = bottom two rows of the 3x3
#      These two halves are horizontal mirror images and overlap on the middle row, so their union is again the
#      full 3x3 block.
#    - A source solid above the network becomes an upper half in place and emits the lower half to the routed
#      destination below. A source solid below the network becomes a lower half in place and emits the upper half.
#      Existing half-tokens are already "at rest" and only emit the complementary half.
#    - Example trace on one channel:
#        original logic: top solid -> top diamond in place, bottom X at routed column
#        variant logic:  top solid -> top upper-half in place, bottom lower-half at routed column
#      The wire traversal and permutation logic are unchanged; only the token decomposition/writeback changes.
# b) Complexity calibration:
#    - We keep exactly three token states per color: solid, upper-half, lower-half.
#    - The half shapes are visually distinct from the original diamond/X pair and from each other, while remaining
#      simple enough to infer from a few examples.
#    - We do not introduce per-color or per-channel hidden orientation choices; the split is always horizontal,
#      which avoids ambiguity for a symmetric solid.
# c) Old assumptions that no longer hold:
#    - The old solver assumed every solid could be converted immediately to the same in-place shape (diamond).
#      That breaks here because the in-place result depends on whether the solid is above or below the wire.
#      Therefore solids are now rewritten when encountered from their endpoint, not by a global unconditional pass.
#    - The old solver only detected X-pattern partial tokens. That no longer works, so we explicitly detect upper
#      and lower 2x3 slabs and remove their cells from the track mask.
#    - The old generator's target output always used top diamonds and bottom X's. That is replaced with top
#      upper-halves and bottom lower-halves.
# d) Placement legality:
#    - Tokens still occupy a 3x3 neighborhood centered exactly two cells away from a degree-1 endpoint.
#    - Upper/lower halves stay within that 3x3 footprint, so the original margins and grid bounds remain valid.
#    - The generator only places upper halves on the top side and lower halves on the bottom side, matching the
#      solver's source-side assumption and preventing ambiguous half orientation.
# e) Visual observability:
#    - The change is directly visible: outputs contain 2x3 mirror slabs instead of a 4-cell diamond / 5-cell X.
#    - Because some solids originate on top and some on bottom, both upper and lower halves appear as sources
#      across samples, making the "one half stays, the mirrored half travels" rule observable.
# f) Visual cleanliness:
#    - The variant reuses the original spacing and channel geometry, so token footprints remain separated.
#    - Using only one mirror axis avoids mixing too many asymmetric token families in the same local area.
# g) Color inference:
#    - No color remapping is introduced. Colors still identify channels only, so the new rule is purely shape-based.
# h) Variant verification target:
#    - Generated outputs should contain only upper/lower halves plus the original wire colors, and should differ
#      from the original solver's diamond/X outputs on essentially every nontrivial sample.
# i) New solver assumptions to audit:
#    - Any solid token processed by main() must be attached vertically (above/below) to an endpoint. The generator
#      guarantees this by keeping the original top/bottom layout.
#    - Any half-token must already match its side: upper on top or lower on bottom. The generator enforces that.

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_UPPER_OFF = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1)]
_LOWER_OFF = [(0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
_SOLID_OFF = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]


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


def _find_half_3x3(
    g: np.ndarray,
    bg: int,
    mask: np.ndarray,
) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return isolated 3x3 mirror-half tokens matching the given mask."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            cols = {int(v) for v in sub[mask].flat}
            if len(cols) != 1:
                continue
            col = next(iter(cols))
            if col == bg or np.any(sub[~mask] != bg):
                continue
            isolated = True
            for rr, cc in zip(*np.where(mask)):
                gr, gc = r0 + int(rr), c0 + int(cc)
                for nr, nc in neighbors4(gr, gc):
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


def _find_9rects(g: np.ndarray) -> list[tuple[set, dict]]:
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
        faces: dict[str, list] = {}
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
    rects_9: list[tuple[set, dict]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Greedy wire following with 9-rect pass-through.

    At a color-9 rectangle: exit from opposite face (aligned with entry column/row).
    If no opposite exit, assert exactly one other exit face and use that.
    """
    cur, d, prev = start, dir0, None
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 4):
        if (cur, d) in visited:
            assert False, "wire trace looped — should not happen with 9-rect pass-through"
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

    assert False, "wire trace exceeded step limit"


def _stamp_token(grid: np.ndarray, center_r: int, center_c: int, color: int, offsets: list[tuple[int, int]]) -> None:
    for dr, dc in offsets:
        rr, cc = center_r + dr, center_c + dc
        if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
            grid[rr, cc] = color


def _clear_3x3(grid: np.ndarray, center: tuple[int, int], bg: int) -> None:
    cr, cc = center
    grid[cr - 1 : cr + 2, cc - 1 : cc + 2] = bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route colored 3x3 tokens through the wire network, but split solids into mirror halves instead of radial parts.

    Input structure:
    - Background cells are empty. Non-background cells form one or more orthogonal wire components plus 3x3 tokens
      placed exactly two cells outside degree-1 endpoints. Tokens are either full solid 3x3 blocks, upper 2x3 slabs,
      or lower 2x3 slabs.

    Rule:
    - A solid token decomposes along the horizontal midline. If it is above the wire, its upper half stays in place
      and its lower half is transmitted; if it is below the wire, the lower half stays and the upper half is sent.
    - A half-token already represents the stationary part at its side, so it remains unchanged and sends only its
      complementary half to the routed endpoint on the opposite side.

    Key insight:
    - The wire geometry and permutation logic are unchanged from the base task; only the token family changes from a
      rotational decomposition to a horizontal mirror decomposition, so solving reduces to endpoint matching plus
      choosing upper-versus-lower writeback from the token's side.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    upper_mask = np.zeros((3, 3), dtype=bool)
    upper_mask[0:2, :] = True
    lower_mask = np.zeros((3, 3), dtype=bool)
    lower_mask[1:3, :] = True

    solids = _find_solid_3x3(g, bg)
    uppers = _find_half_3x3(g, bg, upper_mask)
    lowers = _find_half_3x3(g, bg, lower_mask)

    track_mask = g != bg
    for _, (_, (r0, c0)) in solids.items():
        track_mask[r0 : r0 + 3, c0 : c0 + 3] = False
    for _, (_, (r0, c0)) in uppers.items():
        track_mask[r0 : r0 + 2, c0 : c0 + 3] = False
    for _, (_, (r0, c0)) in lowers.items():
        track_mask[r0 + 1 : r0 + 3, c0 : c0 + 3] = False

    comps = _components(track_mask)
    rects_9 = _find_9rects(g)
    out = g.copy()

    rewritten_solids: set[tuple[int, int]] = set()
    routed_tokens: set[tuple[str, tuple[int, int]]] = set()

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
                if int(g[cen]) == bg:
                    continue

                emit_offsets = None
                solid_key = None
                token_key = None

                if cen in solids:
                    col, _ = solids[cen]
                    if col in comp_colors or odc != 0:
                        continue
                    solid_key = cen
                    token_key = ("solid", cen)
                    if odr == -1:
                        keep_offsets, emit_offsets = _UPPER_OFF, _LOWER_OFF
                    elif odr == 1:
                        keep_offsets, emit_offsets = _LOWER_OFF, _UPPER_OFF
                    else:
                        continue
                    if solid_key not in rewritten_solids:
                        _clear_3x3(out, cen, bg)
                        _stamp_token(out, cen[0], cen[1], col, keep_offsets)
                        rewritten_solids.add(solid_key)

                elif cen in uppers:
                    col, _ = uppers[cen]
                    if col in comp_colors or odr != -1 or odc != 0:
                        continue
                    token_key = ("upper", cen)
                    emit_offsets = _LOWER_OFF

                elif cen in lowers:
                    col, _ = lowers[cen]
                    if col in comp_colors or odr != 1 or odc != 0:
                        continue
                    token_key = ("lower", cen)
                    emit_offsets = _UPPER_OFF

                else:
                    continue

                if token_key in routed_tokens:
                    continue

                dir_in = (-odr, -odc)
                end, dd = _trace_wire(comp, ep, dir_in, h, w, rects_9)
                dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                _stamp_token(out, dest[0], dest[1], col, emit_offsets)
                routed_tokens.add(token_key)

    return out


# --------------- generator (mirror_half_tokens) ---------------

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

RECT_COUNT_WEIGHTS = [10, 40, 30, 20]
RECT_WIRE_COUNT_WEIGHTS = [50, 38, 12]
RECT_PARALLEL_PROB = 0.75
RECT_H_RANGE = (2, 5)
RECT_MAX_WIDTH = 7
RECT_RETRY_FACTOR = 5
RECT_MARGIN_GAP = 1

HALF_INPUT_COUNT_WEIGHTS = [60, 30, 10]
FAR_HALF_PROB = 0.33

GEN_MAX_RETRIES = 300


def _decompose_to_crossovers(perm):
    """Decompose permutation into sequential crossover moves using bypass for cycles."""
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


def _try_place_9rect(grid, ch_cols, xover_rows, top_ep_row, bot_ep_row, n, placed_rects):
    """Attempt to place one color-9 rectangular block. Returns (r0, c0, r1, c1) or None."""
    h, w = grid.shape
    target_wires = random.choices([1, 2, 3], weights=RECT_WIRE_COUNT_WEIGHTS)[0]

    if target_wires >= 2 and random.random() < RECT_PARALLEL_PROB and n >= 2:
        mode = "parallel"
    elif target_wires >= 2 and xover_rows:
        mode = "crossing"
    else:
        mode = "single"

    if mode == "parallel":
        j0 = random.randint(0, n - 2)
        c_lo, c_hi = ch_cols[j0], ch_cols[j0 + 1]
        margin_l, margin_r = random.randint(0, 1), random.randint(0, 1)
    else:
        j0 = random.randint(0, n - 1)
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

    gap = RECT_MARGIN_GAP
    for pr0, pc0, pr1, pc1 in placed_rects:
        if rect_r0 <= pr1 + gap and pr0 <= rect_r1 + gap and rect_c0 <= pc1 + gap and pc0 <= rect_c1 + gap:
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


def _build_wire_network(grid, n, ch_cols, bypass_col, top_ep_row, bot_ep_row, crossovers, xover_rows):
    """Draw the wire network onto grid. Returns (slot_wire, bypass_wire) after all crossovers."""
    slot_wire = list(range(n))
    bypass_wire = None

    col_seg_start = {}
    for j in range(n):
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


def _derive_input(grid, n, ch_cols, perm, token_colors, top_token_row, bot_token_row):
    """Derive the input grid from the wire-only grid by placing solid or half tokens."""
    inp = grid.copy()
    n_half = random.choices([0, 1, 2], weights=HALF_INPUT_COUNT_WEIGHTS)[0]
    half_channels = set(random.sample(range(n), min(n_half, n)))

    for i in range(n):
        col = token_colors[i]
        top_center = (top_token_row, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[perm[i]])

        if i in half_channels:
            if random.choice([True, False]):
                near, near_offsets = top_center, _UPPER_OFF
                far, far_offsets = bot_center, _LOWER_OFF
            else:
                near, near_offsets = bot_center, _LOWER_OFF
                far, far_offsets = top_center, _UPPER_OFF
            _stamp_token(inp, near[0], near[1], col, near_offsets)
            if random.random() < FAR_HALF_PROB:
                _stamp_token(inp, far[0], far[1], col, far_offsets)
        else:
            side = random.choice([top_center, bot_center])
            _stamp_token(inp, side[0], side[1], col, _SOLID_OFF)

    return inp


def _gen_attempt():
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

    while True:
        perm = list(range(n))
        random.shuffle(perm)
        if perm != list(range(n)):
            break

    crossovers = _decompose_to_crossovers(perm)
    n_xovers = len(crossovers)
    if n_xovers == 0:
        return None

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

    n_rects = random.choices(range(4), weights=RECT_COUNT_WEIGHTS)[0]
    placed_rects = []
    for _ in range(n_rects * RECT_RETRY_FACTOR):
        if len(placed_rects) >= n_rects:
            break
        result = _try_place_9rect(grid, ch_cols, xover_rows, top_ep_row, bot_ep_row, n, placed_rects)
        if result is not None:
            placed_rects.append(result)

    output = grid.copy()
    for i in range(n):
        _stamp_token(output, top_token_row, ch_cols[i], token_colors[i], _UPPER_OFF)
        _stamp_token(output, bot_token_row, ch_cols[perm[i]], token_colors[i], _LOWER_OFF)

    inp = _derive_input(grid, n, ch_cols, perm, token_colors, top_token_row, bot_token_row)
    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input():
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
