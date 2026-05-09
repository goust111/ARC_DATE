# Combo variant of a47bf94d (index 7)
# combo_spec: D4/3+D6/0
# Name: L形分解+L/T形源token
#
# Design analysis
# a) Concrete behavior trace:
#    - The original task looked two cells away from each wire endpoint for either a solid 3x3 block or an X,
#      then rewrote that token into diamond/X complements across the routed endpoint pair.
#    - In this combo, the wire / crossing / color-9 rectangle logic stays intact, but the endpoint-adjacent
#      source token family changes. Active sources are now:
#        * a 6-cell thick L inside a 3x3 box, with one of four corner orientations;
#        * a canonical 5-cell T shape inside a 3x3 box, with one of four cardinal orientations.
#    - The transform also changes. Instead of diamond/X, every active source resolves into the same asymmetric
#      complementary L-pair borrowed from D4/3:
#        * a 5-cell heavy L fragment;
#        * a 4-cell complementary heavy L fragment.
#      For L-sources, the 5-cell fragment stays at the source and the 4-cell fragment is routed away.
#      For T-sources, this polarity flips: the 4-cell fragment stays locally and the 5-cell fragment travels.
#    - Example: a top-side L(NW) source used to be ignored by the original solver, because it is neither a solid
#      3x3 nor an X. Here it is recognized, rewritten in place to the NW 5-cell heavy-L, then its routed partner
#      stamps the rotated 4-cell complement at the far endpoint.
#    - Example: a bottom-side T(E) source is similarly recognized first by family and orientation; it leaves the
#      4-cell local fragment on the source side and sends the 5-cell complement to the wire-matched top endpoint.
# b) Complexity calibration:
#    - There are only two source families with four rotations each. That is enough visible diversity to make the
#      combo distinct from the original task, while keeping the case count learnable from a few demonstrations.
#    - The same two output fragment types repeat throughout every grid. The key branch is therefore small and
#      observable: "identify source family, then decide which L-half stays and which half is transmitted."
#    - Generator-side mixing is forced so most samples contain both L and T sources; otherwise the task would
#      collapse into a single-family special case and the fusion would be visually under-supported.
# c) Hidden assumptions audited against the original main():
#    - `_find_solid_3x3` / `_find_x_3x3` are no longer valid because active tokens are not fixed solid/X masks.
#      The solver now scans eight explicit source masks instead.
#    - The original track extraction subtracted token cells from `g != bg`. With optional pre-placed passive
#      fragments and denser custom shapes, that becomes fragile. The combo treats only colors {5,8,9} as track,
#      which matches the generator and keeps routing independent of token footprint details.
#    - The original local rewrite was unconditional and shape-invariant. That assumption breaks here: the local
#      fragment depends on source family, so each detected source record carries both its local and remote masks.
#    - Endpoint probing still assumes the active source center is exactly two cells away from a degree-1 wire cell.
#      The generator preserves that geometry exactly, so the original routing skeleton remains applicable.
# d) Placement legality:
#    - All source and fragment templates fit inside the same 3x3 anchor box used by the original task, so the
#      existing channel spacing and endpoint gap remain sufficient.
#    - Token colors still come from {1,2,3,4,6,7}, disjoint from wire colors {5,8,9}. Optional far-end passive
#      fragments are only stamped at the matched opposite anchor, so there are no stray same-color distractors.
#    - The generator uses the original output-first permutation builder and validates every sample with `main()`,
#      guaranteeing that the routed endpoints, 9-rect pass-through, and fragment placement are jointly legal.
# e) Visual observability:
#    - The difference from the base task is visible at a glance: inputs contain L/T sources instead of solid
#      squares or Xs, and outputs contain two irregular L fragments rather than diamond/X complements.
#    - The family-dependent polarity is also visible: some sources leave the larger 5-cell L locally, while others
#      leave the smaller 4-cell L locally. Repetition across channels gives enough evidence to infer that the split
#      depends on whether the source looked like an L or a T.
# f) Visual cleanliness:
#    - Token footprints stay one cell away from wires, exactly as in the source task. Wires keep using only 5/8/9
#      while tokens keep their own colors, so the scene remains readable even with crossings and 9-rect junctions.
#    - Passive far fragments are not used as routing signals; they are merely optional partial completions, which
#      avoids overlay logic and keeps the final grids uncluttered.
# g) Color and inference:
#    - Colors are preserved one-to-one from source to both output fragments. No hidden color ordering is used.
#    - The only latent variable the solver must infer is shape family/orientation, both of which are directly
#      visible in the 3x3 source footprint. This keeps the combo structurally inferable from 2-4 examples.

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_WIRE_COLORS = (5, 8, 9)


def _rot90(offsets):
    return [(-dc, dr) for dr, dc in offsets]


def _rotate_times(offsets, k):
    out = list(offsets)
    for _ in range(k % 4):
        out = _rot90(out)
    return out


def _make_mask(offsets):
    mask = np.zeros((3, 3), dtype=bool)
    for dr, dc in offsets:
        mask[dr + 1, dc + 1] = True
    return mask


def _stamp_offsets(grid: np.ndarray, center_r: int, center_c: int, color: int, offsets) -> None:
    for dr, dc in offsets:
        rr, cc = center_r + dr, center_c + dc
        if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
            grid[rr, cc] = color


_STAY_L_BASE = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0)]
_TRAVEL_L_BASE = [(0, 1), (1, -1), (1, 0), (1, 1)]
_L_SOURCE_BASE = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (1, -1), (1, 0)]
_T_SOURCE_BASE = [(-1, -1), (-1, 0), (-1, 1), (0, 0), (1, 0)]

_CORNER_ORI = ["NW", "NE", "SE", "SW"]
_CARD_ORI = ["N", "E", "S", "W"]

FRAGMENT_STAY = {ori: _rotate_times(_STAY_L_BASE, idx) for idx, ori in enumerate(_CORNER_ORI)}
FRAGMENT_TRAVEL = {ori: _rotate_times(_TRAVEL_L_BASE, idx) for idx, ori in enumerate(_CORNER_ORI)}
L_SOURCE_PATTERNS = {ori: _rotate_times(_L_SOURCE_BASE, idx) for idx, ori in enumerate(_CORNER_ORI)}
T_SOURCE_PATTERNS = {ori: _rotate_times(_T_SOURCE_BASE, idx) for idx, ori in enumerate(_CARD_ORI)}

SOURCE_PATTERNS = {
    ("L", ori): L_SOURCE_PATTERNS[ori] for ori in _CORNER_ORI
} | {
    ("T", ori): T_SOURCE_PATTERNS[ori] for ori in _CARD_ORI
}
SOURCE_MASKS = {key: _make_mask(offsets) for key, offsets in SOURCE_PATTERNS.items()}

SOURCE_RECORDS = {}
for ori in _CORNER_ORI:
    SOURCE_RECORDS[("L", ori)] = {
        "local_offsets": FRAGMENT_STAY[ori],
        "remote_offsets": FRAGMENT_TRAVEL[ori],
    }
for idx, ori in enumerate(_CARD_ORI):
    corner = _CORNER_ORI[idx]
    SOURCE_RECORDS[("T", ori)] = {
        "local_offsets": FRAGMENT_TRAVEL[corner],
        "remote_offsets": FRAGMENT_STAY[corner],
    }


def _find_source_tokens(g: np.ndarray, bg: int) -> dict[tuple[int, int], dict]:
    """Return active source token records keyed by 3x3 center."""
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
                center = (r0 + 1, c0 + 1)
                for dr, dc in SOURCE_PATTERNS[key]:
                    rr, cc = center[0] + dr, center[1] + dc
                    for nr, nc in neighbors4(rr, cc):
                        if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                            if int(g[nr, nc]) == col:
                                isolated = False
                                break
                    if not isolated:
                        break
                if not isolated:
                    continue

                record = SOURCE_RECORDS[key]
                found[center] = {
                    "color": col,
                    "bbox": (r0, c0),
                    "kind": key[0],
                    "orientation": key[1],
                    "local_offsets": record["local_offsets"],
                    "remote_offsets": record["remote_offsets"],
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
    rects_9: list[tuple[set, dict]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Greedy wire following with 9-rect pass-through.

    At a color-9 rectangle: exit from the opposite face when possible.
    If no opposite exit exists, use the unique other connected face.
    """
    cur, d, prev = start, dir0, None
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 4):
        if (cur, d) in visited:
            assert False, "wire trace looped"
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route L/T-shaped source tokens through the original wire permutation, but resolve each source into a family-dependent pair of complementary L-fragments.

    Input structure:
    - Background cells are 0. Non-background cells are either wire pieces (8 with 5-marked crossings and optional 9-rect pass-through blocks) or endpoint-adjacent source/fragment tokens in ordinary token colors.
    - Every active source occupies one 3x3 anchor two cells away from a wire endpoint and is either a 6-cell thick L or a 5-cell T. Some opposite anchors may already contain the fragment that should appear there in the final state.

    Rule:
    - Replace each L-source in place with the 5-cell heavy-L half and transmit its 4-cell complementary heavy-L half to the endpoint reached by following that wire component.
    - Replace each T-source in place with the 4-cell heavy-L half and transmit the 5-cell complementary heavy-L half instead.

    Key insight:
    - The routing mechanism is unchanged; only endpoint token semantics change. Solve by first recognizing whether each source anchor contains an L-source or a T-source, then apply the corresponding asymmetric L-split while keeping the original wire permutation and 9-rect traversal rules.
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
    """Decompose permutation into sequential crossover moves using a bypass for cycles."""
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


def _try_place_9rect(grid, ch_cols, xover_rows, top_ep_row, bot_ep_row, n_channels, placed_rects):
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


def _build_wire_network(grid, n_channels, ch_cols, bypass_col, top_ep_row, bot_ep_row, crossovers, xover_rows):
    """Draw the wire network and return (slot_wire, bypass_wire) after all crossovers."""
    slot_wire = list(range(n_channels))
    bypass_wire = None

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


def _choose_specs(n_channels):
    kinds = random.choices(["L", "T"], weights=SOURCE_KIND_WEIGHTS, k=n_channels)
    if len(set(kinds)) == 1 and n_channels >= 2:
        flip_idx = random.randrange(n_channels)
        kinds[flip_idx] = "T" if kinds[flip_idx] == "L" else "L"

    sides = random.choices(["top", "bottom"], k=n_channels)
    if len(set(sides)) == 1 and n_channels >= 2:
        sides[random.randrange(n_channels)] = "bottom" if sides[0] == "top" else "top"

    specs = []
    for kind, side in zip(kinds, sides):
        if kind == "L":
            ori = random.choice(_CORNER_ORI)
        else:
            ori = random.choice(_CARD_ORI)
        specs.append({"kind": kind, "side": side, "orientation": ori})
    return specs


def _build_io(grid, n_channels, ch_cols, perm, token_colors, top_token_row, bot_token_row):
    inp = grid.copy()
    out = grid.copy()
    specs = _choose_specs(n_channels)

    for i in range(n_channels):
        col = token_colors[i]
        top_center = (top_token_row, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[perm[i]])
        spec = specs[i]
        key = (spec["kind"], spec["orientation"])
        pattern = SOURCE_PATTERNS[key]
        local_offsets = SOURCE_RECORDS[key]["local_offsets"]
        remote_offsets = SOURCE_RECORDS[key]["remote_offsets"]

        if spec["side"] == "top":
            src_center, dst_center = top_center, bot_center
        else:
            src_center, dst_center = bot_center, top_center

        _stamp_offsets(inp, src_center[0], src_center[1], col, pattern)
        if random.random() < FAR_FRAGMENT_PROB:
            _stamp_offsets(inp, dst_center[0], dst_center[1], col, remote_offsets)

        _stamp_offsets(out, src_center[0], src_center[1], col, local_offsets)
        _stamp_offsets(out, dst_center[0], dst_center[1], col, remote_offsets)

    return inp, out


def _gen_attempt():
    n_channels = random.choice(CHANNEL_COUNTS)
    spacing = random.choice(CHANNEL_SPACING)

    left_margin = random.randint(*LEFT_MARGIN_RANGE)
    first_col = left_margin + 1
    ch_cols = [first_col + i * spacing for i in range(n_channels)]
    bypass_col = ch_cols[-1] + BYPASS_COL_OFFSET

    w = max(bypass_col + 3, GRID_W_MIN)
    if w > GRID_W_MAX:
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
        grid, n_channels, ch_cols, bypass_col, top_ep_row, bot_ep_row, crossovers, xover_rows
    )

    inv_perm = [0] * n_channels
    for i in range(n_channels):
        inv_perm[perm[i]] = i
    if any(slot_wire[j] != inv_perm[j] for j in range(n_channels)) or bypass_wire is not None:
        return None

    n_rects = random.choices(range(4), weights=RECT_COUNT_WEIGHTS)[0]
    placed_rects = []
    for _ in range(n_rects * RECT_RETRY_FACTOR):
        if len(placed_rects) >= n_rects:
            break
        result = _try_place_9rect(grid, ch_cols, xover_rows, top_ep_row, bot_ep_row, n_channels, placed_rects)
        if result is not None:
            placed_rects.append(result)

    inp, out = _build_io(grid, n_channels, ch_cols, perm, token_colors, top_token_row, bot_token_row)
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
