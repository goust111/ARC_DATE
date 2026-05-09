# Combo variant of a47bf94d (index 29)
# combo_spec: D4/2+D6/0+D9/0
# Name: 镜像分割+L/T形token+双形状系统
#
# Design analysis for this combo variant
# a) Concrete behavior and trace:
#    - Keep the original task's wire-permutation backbone: 4-5 vertical channels, top/bottom endpoint anchors,
#      color-8 wires with color-5 crossing brackets, and optional color-9 pass-through rectangles.
#    - Replace the old solid/X/diamond family with a 2-shape source system:
#        source L(top)    = 5-cell NW-corner L in a 3x3 box
#        source L(bottom) = its 180-degree mate, a SE-corner L
#        source T(top)    = 5-cell north-facing T
#        source T(bottom) = its 180-degree mate, a south-facing T
#    - Replace the old radial decomposition with a horizontal mirror split plus shape flip:
#        L -> T
#          top L    becomes a top 3-cell bar locally and sends the bottom 3-cell bar to the far endpoint
#          bottom L becomes a bottom 3-cell bar locally and sends the top 3-cell bar upward
#        T -> L
#          top T    becomes an upper 3-cell left-corner locally and sends the lower left-corner downward
#          bottom T becomes the lower left-corner locally and sends the upper left-corner upward
#    - Example:
#        old logic: top solid(color a) -> top diamond(a) + routed bottom X(a)
#        new logic: top L(color a)     -> top upper-T-half(a) + routed bottom lower-T-half(a)
#        old logic: bottom X(color b)  -> bottom X(b) + routed top diamond(b)
#        new logic: bottom T(color b)  -> bottom lower-L-half(b) + routed top upper-L-half(b)
#      The route tracing itself is unchanged; only the endpoint-adjacent token family and writeback change.
# b) Complexity calibration:
#    - Only two source kinds remain, L and T, so the D9 two-stage cycle is visible: L↔T.
#    - Mirror splitting is still obvious because every local/remote pair is an upper-vs-lower complementary pair.
#    - We intentionally keep just one L-half orientation (left corner) for T outputs; this avoids too many hidden
#      orientation cases while still making L-outputs visually distinct from T-outputs.
#    - Each generated sample is forced to contain both L and T sources, and usually both top- and bottom-side
#      sources, so the shape flip and mirror-half rule are repeated within single grids.
# c) Implicit assumptions that changed:
#    - The original solver assumed fixed 3x3 solid/X detectors and token-cell subtraction from track masks.
#      That no longer holds, so the solver now detects four exact L/T source templates and treats only {5,8,9}
#      as wires. Optional far-end rest tokens are preserved automatically instead of being interpreted as sources.
#    - The original in-place rewrite was unconditional for solids. Here the local rewrite depends on both source
#      kind and which side of the network the source is on, so source records carry `kind` and `side`.
#    - Endpoint scanning used to accept any token two cells away. Here it additionally requires side consistency:
#      a `top` source must be two rows above an endpoint; a `bottom` source must be two rows below one.
# d) Placement legality:
#    - All full sources and all resting halves stay inside the same isolated 3x3 anchor footprint used by the
#      original task, exactly two cells from degree-1 endpoints, so the original spacing and margins remain valid.
#    - The generator places at most one object per anchor, uses a permutation for bottom anchors, and only pre-fills
#      the far anchor with the exact complementary half expected by the solver.
# e) Visual observability:
#    - Humans can see that large 5-cell L/T sources shrink to smaller local halves, while matching complementary
#      halves appear at routed partner anchors.
#    - The transformation is not color-coded; colors only identify pairings along the wire permutation, while shape
#      determines whether the outputs are bar-halves (T family) or corner-halves (L family).
# f) Visual cleanliness:
#    - Token colors remain disjoint from wire colors {5,8,9}, and all token shapes are confined to clean 3x3 boxes
#      separated from wires by a one-cell gap.
#    - Using only horizontal mirror pairs keeps top/bottom outputs easy to compare even when routes cross.
# g) Change inferability:
#    - No color remapping is introduced. The only nontrivial inference is shape-based:
#      L sources produce T halves, T sources produce L halves, and the local-vs-remote halves are upper/lower
#      complements determined by whether the source sits above or below the network.
# h) Fusion-specific non-degeneracy target:
#    - Generated examples should routinely contain both `L -> T` and `T -> L` channels; otherwise the combo would
#      collapse to a single-shape mirror variant and stop exhibiting the intended dual-shape behavior.
# i) New solver assumptions to audit against the generator:
#    - Every full source is one of the four exact templates below.
#    - A source anchor is vertical-only relative to its endpoint (`odc == 0`).
#    - Any optional far-end token is already the exact complementary resting half and never overlaps another source.

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_WIRE_COLORS = (5, 8, 9)

_SRC_L_TOP = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (1, -1)]
_SRC_L_BOTTOM = [(-1, 1), (0, 1), (1, -1), (1, 0), (1, 1)]
_SRC_T_TOP = [(-1, -1), (-1, 0), (-1, 1), (0, 0), (1, 0)]
_SRC_T_BOTTOM = [(-1, 0), (0, 0), (1, -1), (1, 0), (1, 1)]

_T_UPPER = [(-1, -1), (-1, 0), (-1, 1)]
_T_LOWER = [(1, -1), (1, 0), (1, 1)]
_L_UPPER = [(-1, -1), (-1, 0), (0, -1)]
_L_LOWER = [(0, -1), (1, -1), (1, 0)]

SOURCE_PATTERNS = {
    ("L", "top"): _SRC_L_TOP,
    ("L", "bottom"): _SRC_L_BOTTOM,
    ("T", "top"): _SRC_T_TOP,
    ("T", "bottom"): _SRC_T_BOTTOM,
}

REST_PATTERNS = {
    ("T", "upper"): _T_UPPER,
    ("T", "lower"): _T_LOWER,
    ("L", "upper"): _L_UPPER,
    ("L", "lower"): _L_LOWER,
}

SOURCE_TO_RESTS = {
    ("L", "top"): (("T", "upper"), ("T", "lower")),
    ("L", "bottom"): (("T", "lower"), ("T", "upper")),
    ("T", "top"): (("L", "upper"), ("L", "lower")),
    ("T", "bottom"): (("L", "lower"), ("L", "upper")),
}


def _make_mask(offsets: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((3, 3), dtype=bool)
    for dr, dc in offsets:
        mask[dr + 1, dc + 1] = True
    return mask


SOURCE_MASKS = {key: _make_mask(offsets) for key, offsets in SOURCE_PATTERNS.items()}


def _find_source_tokens(g: np.ndarray, bg: int) -> dict[tuple[int, int], dict]:
    """Return endpoint-adjacent full L/T sources keyed by 3x3 center."""
    h, w = g.shape
    found: dict[tuple[int, int], dict] = {}

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            non_bg = {int(sub[rr, cc]) for rr in range(3) for cc in range(3) if int(sub[rr, cc]) != bg}
            if len(non_bg) != 1:
                continue
            col = next(iter(non_bg))
            if col in _WIRE_COLORS:
                continue

            for key, mask in SOURCE_MASKS.items():
                if np.any(sub[mask] != col) or np.any(sub[~mask] != bg):
                    continue

                isolated = True
                for dr, dc in SOURCE_PATTERNS[key]:
                    rr, cc = r0 + 1 + dr, c0 + 1 + dc
                    for nr, nc in neighbors4(rr, cc):
                        if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                            if int(g[nr, nc]) == col:
                                isolated = False
                                break
                    if not isolated:
                        break
                if not isolated:
                    continue

                center = (r0 + 1, c0 + 1)
                found[center] = {
                    "color": col,
                    "bbox": (r0, c0),
                    "kind": key[0],
                    "side": key[1],
                }
                break
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

    At a color-9 rectangle: exit from the opposite face when available, otherwise use the unique other face.
    """
    cur, d, prev = start, dir0, None
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 4):
        if (cur, d) in visited:
            raise AssertionError("wire trace looped")
        visited.add((cur, d))

        jumped = False
        for cells9, faces9 in rects_9:
            if cur in cells9:
                entry_face = _DIR2FACE[d]
                opp = _OPP[entry_face]
                if opp in faces9:
                    exit_face = opp
                else:
                    others = [face for face in faces9 if face != entry_face]
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
        nonrev = [opt for opt in opts if (opt[1], opt[2]) != rev]
        opts2 = nonrev if nonrev else opts

        straight = [opt for opt in opts2 if (opt[1], opt[2]) == d]
        if straight:
            p, dr, dc = straight[0]
        else:
            perp = [opt for opt in opts2 if opt[1] * d[0] + opt[2] * d[1] == 0]
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


def _stamp_offsets(grid: np.ndarray, center_r: int, center_c: int, color: int, offsets: list[tuple[int, int]]) -> None:
    for dr, dc in offsets:
        rr, cc = center_r + dr, center_c + dc
        if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
            grid[rr, cc] = color


def _clear_3x3(grid: np.ndarray, center: tuple[int, int], bg: int) -> None:
    cr, cc = center
    grid[cr - 1 : cr + 2, cc - 1 : cc + 2] = bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route full L/T sources through the original wire network, but mirror-split them into complementary halves of the opposite shape family.

    Input structure:
    - Background cells are empty. Non-background cells are either wire pieces (8 with 5-marked crossings and optional 9-rect pass-through blocks) or colored 3x3 tokens placed exactly two cells above or below degree-1 wire endpoints.
    - A full source token is one of four exact templates: top/bottom L or top/bottom T. Some far anchors may already contain the matching resting half that would be produced after routing.

    Rule:
    - Every full L source is replaced by a local T-half and emits the complementary T-half to the routed opposite endpoint.
    - Every full T source is replaced by a local L-half and emits the complementary L-half to the routed opposite endpoint.
    - The local-versus-remote half is determined by side: sources above the wire keep the upper half and send the lower half; sources below the wire do the opposite.

    Key insight:
    - The network semantics are unchanged from the base task; the new work is local shape recognition plus the two-step family flip `L -> T` and `T -> L`. Once the source kind and side are known, the in-place half and routed half are fixed.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    sources = _find_source_tokens(g, bg)
    track_mask = np.isin(g, _WIRE_COLORS)
    comps = _components(track_mask)
    rects_9 = _find_9rects(g)

    out = g.copy()
    for center, info in sources.items():
        local_key, _ = SOURCE_TO_RESTS[(info["kind"], info["side"])]
        _clear_3x3(out, center, bg)
        _stamp_offsets(out, center[0], center[1], info["color"], REST_PATTERNS[local_key])

    routed: set[tuple[int, int]] = set()
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
        if not endpoints:
            continue

        for ep in endpoints:
            er, ec = ep
            for odr, odc in _N4:
                cen = (er + 2 * odr, ec + 2 * odc)
                info = sources.get(cen)
                if info is None:
                    continue
                if odc != 0:
                    continue
                expected_odr = -1 if info["side"] == "top" else 1
                if odr != expected_odr or cen in routed:
                    continue

                _, remote_key = SOURCE_TO_RESTS[(info["kind"], info["side"])]
                dir_in = (-odr, -odc)
                end, dd = _trace_wire(comp, ep, dir_in, h, w, rects_9)
                dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                _clear_3x3(out, dest, bg)
                _stamp_offsets(out, dest[0], dest[1], info["color"], REST_PATTERNS[remote_key])
                routed.add(cen)

    return out


# --------------- generator ---------------

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

SOURCE_KIND_WEIGHTS = [50, 50]
FAR_HALF_PROB = 0.35
GEN_MAX_RETRIES = 300


def _decompose_to_crossovers(perm: list[int]) -> list[tuple[int, int]]:
    """Decompose permutation into sequential crossover moves using a bypass channel."""
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
    n: int,
    placed_rects: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    """Attempt to place one color-9 rectangular block."""
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
    n: int,
    ch_cols: list[int],
    bypass_col: int,
    top_ep_row: int,
    bot_ep_row: int,
    crossovers: list[tuple[int, int]],
    xover_rows: list[int],
) -> tuple[list[int | None], int | None]:
    """Draw the wire network onto grid. Returns slot wiring after all crossovers."""
    slot_wire: list[int | None] = list(range(n))
    bypass_wire: int | None = None

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


def _choose_specs(n: int) -> list[dict]:
    # Keep every sample diagnostically rich: with 4-5 channels we can always expose
    # all four `(shape, side)` branches so humans can infer the full fused rule.
    combos = [("L", "top"), ("L", "bottom"), ("T", "top"), ("T", "bottom")]
    random.shuffle(combos)

    while len(combos) < n:
        extra_kind = random.choices(["L", "T"], weights=SOURCE_KIND_WEIGHTS, k=1)[0]
        extra_side = random.choice(["top", "bottom"])
        combos.append((extra_kind, extra_side))

    return [{"kind": kind, "side": side} for kind, side in combos[:n]]


def _build_io(
    grid: np.ndarray,
    n: int,
    ch_cols: list[int],
    perm: list[int],
    token_colors: list[int],
    top_token_row: int,
    bot_token_row: int,
) -> tuple[np.ndarray, np.ndarray]:
    inp = grid.copy()
    out = grid.copy()
    specs = _choose_specs(n)

    for i in range(n):
        col = token_colors[i]
        top_center = (top_token_row, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[perm[i]])
        spec = specs[i]
        src_key = (spec["kind"], spec["side"])
        local_key, remote_key = SOURCE_TO_RESTS[src_key]

        if spec["side"] == "top":
            src_center, dst_center = top_center, bot_center
        else:
            src_center, dst_center = bot_center, top_center

        _stamp_offsets(inp, src_center[0], src_center[1], col, SOURCE_PATTERNS[src_key])
        if random.random() < FAR_HALF_PROB:
            _stamp_offsets(inp, dst_center[0], dst_center[1], col, REST_PATTERNS[remote_key])

        _stamp_offsets(out, src_center[0], src_center[1], col, REST_PATTERNS[local_key])
        _stamp_offsets(out, dst_center[0], dst_center[1], col, REST_PATTERNS[remote_key])

    return inp, out


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
    slot_wire, bypass_wire = _build_wire_network(grid, n, ch_cols, bypass_col, top_ep_row, bot_ep_row, crossovers, xover_rows)

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

    inp, out = _build_io(grid, n, ch_cols, perm, token_colors, top_token_row, bot_token_row)
    try:
        if np.array_equal(main(inp), out):
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
