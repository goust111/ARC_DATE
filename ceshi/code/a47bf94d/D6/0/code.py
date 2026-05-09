# Variant of a47bf94d
# Dimension: D6 — 源token形状类型 (src)
# Change: 三种token形状——实心3x3块（输入中主要形式）、X形（对角+中心，5像素）、菱形（4正交邻居，4像素）；每种颜色最... → L形/T形token

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

# Design analysis for the D6 variant
# a) Concrete behavior:
#    Keep the original wire-permutation backbone: color-8/5/9 networks still connect opposite endpoints,
#    and the token anchor remains exactly two cells away from an endpoint. The change is local to the source
#    token family. Inputs now use 3x3-centered 5-cell L or T shapes. Each source token is orientation-bearing:
#    L is one full side-row plus one adjacent full side-column; T is one full side-row/column plus a 2-cell stem.
#    The output is no longer diamond/X. Instead the source token splits into a small "rest" fragment left at the
#    source anchor and a 180-degree counterpart routed to the far anchor.
#    Example: an L with its elbow at the south-west corner used to be impossible in the original task because the
#    solver only understood "solid 3x3" and "X". Under the old logic the endpoint scan would look two cells out,
#    fail to find a solid/X center, and transmit nothing. In the variant the same source is recognized as L(SW),
#    replaced in-place by the 3-cell SW corner, and the opposite 3-cell NE corner is stamped at the remote anchor.
# b) Complexity calibration:
#    There are only two source families and four rotations each, so the freedom stays learnable. Both families are
#    visually distinct and repeated within each grid. The routed fragment is always the 180-degree mate of the local
#    fragment, so the branch structure is small enough for 2-4 examples.
# c) Old implicit assumptions that break:
#    - `_find_solid_3x3` / `_find_x_3x3` assumed every active source token is a fixed 3x3 template. That no longer
#      holds, so the solver must scan for eight new templates.
#    - `track_mask = g != bg minus token cells` relied on enumerating source-cell footprints. That becomes fragile
#      with optional far-end resting fragments; the variant instead treats only colors {5,8,9} as track.
#    - "solid becomes diamond" hard-coded a single local rewrite. Now the local rewrite depends on both token family
#      and orientation, so the source token record must carry its local and remote fragment offsets.
# d) Placement legality:
#    Tokens still live in isolated 3x3 boxes centered on the original anchors, so the existing 4-5 column spacing
#    and endpoint gap remain valid. The generator guarantees one unique source per color, optional remote fragment
#    only at the opposite anchor, and no contact between fragments and wires.
# e) Visual observability:
#    Humans can see two changes at once: the large 5-cell source disappears, and a smaller 3-cell bar/corner pair
#    appears across the linked endpoints. Because L and T yield different resting fragments, the solver must first
#    identify shape family before inferring the routed partner.
# f) Visual cleanliness:
#    Every token or fragment occupies its own 3x3 footprint, separated from the wire by a one-cell gap. Wire marks
#    keep their original colors, while token colors remain in {1,2,3,4,6,7}, so object classes stay separable.
# g) Color reasoning:
#    Colors are preserved exactly. The variant uses shape/orientation, not color id, to decide the split.

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_WIRE_COLORS = (5, 8, 9)


def _rot180(offsets):
    return [(-dr, -dc) for dr, dc in offsets]


SOURCE_PATTERNS = {
    ("L", "NW"): [(-1, -1), (-1, 0), (-1, 1), (0, -1), (1, -1)],
    ("L", "NE"): [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)],
    ("L", "SW"): [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)],
    ("L", "SE"): [(-1, 1), (0, 1), (1, -1), (1, 0), (1, 1)],
    ("T", "N"): [(-1, -1), (-1, 0), (-1, 1), (0, 0), (1, 0)],
    ("T", "E"): [(-1, 1), (0, -1), (0, 0), (0, 1), (1, 1)],
    ("T", "S"): [(-1, 0), (0, 0), (1, -1), (1, 0), (1, 1)],
    ("T", "W"): [(-1, -1), (0, -1), (0, 0), (0, 1), (1, -1)],
}

LOCAL_RESTS = {
    ("L", "NW"): [(-1, -1), (-1, 0), (0, -1)],
    ("L", "NE"): [(-1, 0), (-1, 1), (0, 1)],
    ("L", "SW"): [(0, -1), (1, -1), (1, 0)],
    ("L", "SE"): [(0, 1), (1, 0), (1, 1)],
    ("T", "N"): [(-1, -1), (-1, 0), (-1, 1)],
    ("T", "E"): [(-1, 1), (0, 1), (1, 1)],
    ("T", "S"): [(1, -1), (1, 0), (1, 1)],
    ("T", "W"): [(-1, -1), (0, -1), (1, -1)],
}

REMOTE_RESTS = {key: _rot180(val) for key, val in LOCAL_RESTS.items()}


def _make_mask(offsets):
    mask = np.zeros((3, 3), dtype=bool)
    for dr, dc in offsets:
        mask[dr + 1, dc + 1] = True
    return mask


SOURCE_MASKS = {key: _make_mask(offsets) for key, offsets in SOURCE_PATTERNS.items()}


def _find_source_tokens(g: np.ndarray, bg: int) -> dict[tuple[int, int], dict]:
    """Return source token records keyed by 3x3 center."""
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
                    "orientation": key[1],
                    "local_offsets": LOCAL_RESTS[key],
                    "remote_offsets": REMOTE_RESTS[key],
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


def _stamp_offsets(grid: np.ndarray, center_r: int, center_c: int, color: int, offsets) -> None:
    for dr, dc in offsets:
        rr, cc = center_r + dr, center_c + dc
        if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
            grid[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route orientation-bearing L/T source tokens through the same wire networks and split each one into two smaller fragments.

    Input structure:
    - Background cells are empty. Non-background cells are either wire pieces (8 with 5-marked crossings and optional 9-rect pass-through blocks) or colored source/rest tokens placed two cells away from wire endpoints.
    - A source token is a 5-cell L or T inside a 3x3 footprint; some far endpoints may already contain the matching 3-cell resting fragment.

    Rule:
    - Replace every source token by its local resting fragment at the same anchor: L leaves a 3-cell corner at its elbow, T leaves a 3-cell bar on its crossbar side.
    - Send the 180-degree counterpart of that resting fragment through the connected wire component and stamp it at the opposite endpoint anchor.

    Key insight:
    - The wire logic is unchanged; only source-shape recognition changes. Solve by identifying the L/T orientation at each endpoint-adjacent 3x3 anchor, then map it to a fixed local fragment plus its 180-degree mate at the far anchor.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    sources = _find_source_tokens(g, bg)
    track_mask = np.isin(g, _WIRE_COLORS)
    comps = _components(track_mask)
    rects_9 = _find_9rects(g)

    out = g.copy()

    for (cr, cc), info in sources.items():
        r0, c0 = info["bbox"]
        out[r0 : r0 + 3, c0 : c0 + 3] = bg
        _stamp_offsets(out, cr, cc, info["color"], info["local_offsets"])

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

                dir_in = (-odr, -odc)
                end, dd = _trace_wire(comp, ep, dir_in, h, w, rects_9)
                dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                _stamp_offsets(out, dest[0], dest[1], info["color"], info["remote_offsets"])

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

SOURCE_KIND_WEIGHTS = [55, 45]
FAR_FRAGMENT_PROB = 0.35
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


def _choose_specs(n):
    kinds = random.choices(["L", "T"], weights=SOURCE_KIND_WEIGHTS, k=n)
    if len(set(kinds)) == 1:
        flip_idx = random.randrange(n)
        kinds[flip_idx] = "T" if kinds[flip_idx] == "L" else "L"

    sides = random.choices(["top", "bottom"], k=n)
    if len(set(sides)) == 1:
        sides[random.randrange(n)] = "bottom" if sides[0] == "top" else "top"

    specs = []
    for kind, side in zip(kinds, sides):
        ori = random.choice(["NW", "NE", "SW", "SE"]) if kind == "L" else random.choice(["N", "E", "S", "W"])
        specs.append({"kind": kind, "side": side, "orientation": ori})
    return specs


def _build_io(grid, n, ch_cols, perm, token_colors, top_token_row, bot_token_row):
    inp = grid.copy()
    out = grid.copy()
    specs = _choose_specs(n)

    for i in range(n):
        col = token_colors[i]
        top_center = (top_token_row, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[perm[i]])
        spec = specs[i]
        key = (spec["kind"], spec["orientation"])

        if spec["side"] == "top":
            src_center, dst_center = top_center, bot_center
        else:
            src_center, dst_center = bot_center, top_center

        _stamp_offsets(inp, src_center[0], src_center[1], col, SOURCE_PATTERNS[key])
        if random.random() < FAR_FRAGMENT_PROB:
            _stamp_offsets(inp, dst_center[0], dst_center[1], col, REMOTE_RESTS[key])

        _stamp_offsets(out, src_center[0], src_center[1], col, LOCAL_RESTS[key])
        _stamp_offsets(out, dst_center[0], dst_center[1], col, REMOTE_RESTS[key])

    return inp, out


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


def generate_input():
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
