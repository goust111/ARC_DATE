# Variant of a47bf94d
# Dimension: D5 — 形状补全规则 (transform)
# Change: 形状补全（shape_completion）——如果管道一端已存在十字形或四角星形，则在另一端放置互补形状（四角星对应十字，十字对应四角星），颜色与已有形状相同 → 镜像翻转补全

# Design analysis for the D5 variant
# ----------------------------------
# a) Concrete behavior:
#    - Keep the original wire language: 8 = wire, 5 = crossing brackets, 9 = filled pass-through rectangle.
#    - Replace the symmetric diamond/X pair with an asymmetric complementary pair inside the same 3x3 window:
#        ANCHOR = top row + left column (a 5-cell "corner" shape)
#        CORE   = the remaining 2x2 block in the lower-right of that 3x3
#      A solid 3x3 block still decomposes, but now it becomes ANCHOR in place and emits CORE to the far endpoint.
#    - The D5 change is applied only to the emitted complementary shape: before stamping CORE at the far end,
#      mirror it according to the network's global orientation.
#        vertical network (top/bottom endpoints)   -> vertical flip of CORE inside its 3x3
#        horizontal network (left/right endpoints) -> horizontal flip of CORE inside its 3x3
#    - Small trace example:
#      Original logic: solid at one endpoint -> near side diamond, far side X.
#      New logic: solid at one endpoint -> near side ANCHOR, far side mirrored CORE.
#      If the same wire layout is rotated from top-bottom to left-right, the emitted 2x2 block lands in a
#      different corner of the far 3x3 box; this is the visible effect of the new rule.
#
# b) Complexity calibration:
#    - The mirror branch has only two cases (vertical-layout vs horizontal-layout), which is small enough for ARC.
#    - The complementary pair is intentionally asymmetric; otherwise the mirror step would be invisible.
#    - The generator still repeats the same rule over 4-5 channels per sample, so humans get multiple supports.
#
# c) Hidden-assumption review of main():
#    - The original solver only needed to detect symmetric solid/X tokens. Here we must also mask pre-placed
#      mirrored CORE tokens out of the track mask, or the wire component would be corrupted.
#    - The original far-end placement could blindly stamp X/diamond because those were orientation-invariant.
#      Here the emitted shape depends on the whole component orientation, so that orientation must be inferred
#      before placement.
#    - Using source->destination displacement to infer orientation would be unsafe because permutation routing
#      can add large sideways offsets. Instead, layout is inferred from endpoint rows/cols of the wire component.
#
# d) Placement legality:
#    - Token centers remain exactly two cells outside degree-1 wire endpoints, matching the old solver contract.
#    - The chosen 3x3 shapes stay within bounds for both top/bottom and left/right layouts because centers remain
#      at least two cells away from the frame.
#    - The generator preserves the original non-overlap guarantees by reusing the original wire spacing.
#
# e) Visual observability:
#    - With symmetric X/diamond, "mirror" would be indistinguishable from the old task. The asymmetric pair fixes that.
#    - The generator emits both vertical and horizontal layouts, so both mirror branches appear across samples.
#
# f) Visual cleanliness:
#    - Only one token occupies each endpoint window; wires still live in the interior corridor.
#    - The far-end mirrored CORE is a compact 2x2 block, so it does not visually blend with the ANCHOR corner.
#
# g) Color inference:
#    - Colors are unchanged: one stable color per channel, sampled from the old palette. No ordering or remapping
#      depends on color ids.
#
# h) Quantitative behavior target:
#    - The generator should produce both layout orientations and at least one emitted mirrored CORE per sample.
#      Comparing the new output with a "no mirror" baseline should show nonzero changed cells in most samples.
#
# i) New solver assumptions that the generator must satisfy:
#    - Every connected wire component used for routing has endpoints concentrated on exactly two rows or two cols.
#    - Existing non-solid tokens in the input are only ANCHOR or already-correct mirrored CORE windows.
#    - Token colors never equal wire colors {5,8,9}.

import random
import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_SOLID = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]
_ANCHOR = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (1, -1)]
_CORE = [(0, 0), (0, 1), (1, 0), (1, 1)]


def _mirror_offsets(offsets: list[tuple[int, int]], layout: str) -> list[tuple[int, int]]:
    if layout == "V":
        return sorted([(-dr, dc) for dr, dc in offsets])
    return sorted([(dr, -dc) for dr, dc in offsets])


_CORE_V = _mirror_offsets(_CORE, "V")
_CORE_H = _mirror_offsets(_CORE, "H")


def _make_mask(offsets: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((3, 3), dtype=bool)
    for dr, dc in offsets:
        mask[dr + 1, dc + 1] = True
    return mask


_ANCHOR_MASK = _make_mask(_ANCHOR)
_CORE_V_MASK = _make_mask(_CORE_V)
_CORE_H_MASK = _make_mask(_CORE_H)


def _find_solid_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for maximal 3x3 solid blocks."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0:r0 + 3, c0:c0 + 3]
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


def _find_shape_3x3(
    g: np.ndarray,
    bg: int,
    offsets: list[tuple[int, int]],
    mask: np.ndarray,
) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return exact isolated 3x3 matches for a sparse token mask."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0:r0 + 3, c0:c0 + 3]
            vals = sub[mask]
            col = int(vals[0])
            if col == bg or np.any(vals != col) or np.any(sub[~mask] != bg):
                continue
            isolated = True
            for dr, dc in offsets:
                rr, cc = r0 + 1 + dr, c0 + 1 + dc
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


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=4)]


def _find_9rects(g: np.ndarray) -> list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]]:
    """Find color-9 rectangles and their wire contacts by face."""
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
    """
    Greedy wire following with 9-rect pass-through.

    At a color-9 rectangle: exit from the opposite face if present; otherwise take the
    unique non-entry face. This matches the original routing semantics.
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


def _infer_layout(endpoints: list[tuple[int, int]]) -> str:
    rows = sorted({r for r, _ in endpoints})
    cols = sorted({c for _, c in endpoints})
    if len(rows) <= 2 and len(cols) > 2:
        return "V"
    if len(cols) <= 2 and len(rows) > 2:
        return "H"
    row_span = rows[-1] - rows[0] if rows else 0
    col_span = cols[-1] - cols[0] if cols else 0
    return "V" if row_span >= col_span else "H"


def _stamp_token(grid: np.ndarray, center: tuple[int, int], color: int, offsets: list[tuple[int, int]]) -> None:
    cr, cc = center
    for dr, dc in offsets:
        grid[cr + dr, cc + dc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route colored endpoint tokens through the wire network, using mirror-aware completion at the far end.

    Input structure:
    - Background cells are blank.
    - Non-background cells form orthogonal wires plus endpoint-adjacent 3x3 tokens.
    - Tokens are either solid 3x3 blocks, ANCHOR corner shapes, or already-completed mirrored CORE blocks.

    Rule:
    - Every solid 3x3 token decomposes in place into the fixed ANCHOR corner shape.
    - A solid token or an existing ANCHOR token emits the complementary CORE block to the opposite endpoint.
    - Before placing that complementary CORE block, mirror it according to the whole wire component:
      top-bottom components use a vertical flip; left-right components use a horizontal flip.

    Key insight:
    - The wire tracing logic is unchanged from the original task; only the transmitted half of the decomposition
      is now orientation-dependent, so the component's global endpoint arrangement determines which mirrored corner
      appears at the far end.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    solids = _find_solid_3x3(g, bg)
    anchors = _find_shape_3x3(g, bg, _ANCHOR, _ANCHOR_MASK)
    comp_v = _find_shape_3x3(g, bg, _CORE_V, _CORE_V_MASK)
    comp_h = _find_shape_3x3(g, bg, _CORE_H, _CORE_H_MASK)

    track_mask = g != bg
    for _, (_, (r0, c0)) in solids.items():
        track_mask[r0:r0 + 3, c0:c0 + 3] = False
    for token_map, offsets in ((anchors, _ANCHOR), (comp_v, _CORE_V), (comp_h, _CORE_H)):
        for (cr, cc) in token_map.keys():
            for dr, dc in offsets:
                rr, cc2 = cr + dr, cc + dc
                if 0 <= rr < h and 0 <= cc2 < w:
                    track_mask[rr, cc2] = False

    comps = _components(track_mask)
    rects_9 = _find_9rects(g)
    out = g.copy()

    for (cr, cc), (col, (r0, c0)) in solids.items():
        out[r0:r0 + 3, c0:c0 + 3] = bg
        _stamp_token(out, (cr, cc), col, _ANCHOR)

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

        layout = _infer_layout(endpoints)
        emitted = _CORE_V if layout == "V" else _CORE_H

        for ep in endpoints:
            er, ec = ep
            for odr, odc in _N4:
                cen = (er + 2 * odr, ec + 2 * odc)
                if not (0 <= cen[0] < h and 0 <= cen[1] < w):
                    continue

                if cen in solids:
                    col, _ = solids[cen]
                elif cen in anchors:
                    col, _ = anchors[cen]
                else:
                    continue

                if col in comp_colors:
                    continue

                dir_in = (-odr, -odc)
                end, dd = _trace_wire(comp, ep, dir_in, h, w, rects_9)
                dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                _stamp_token(out, dest, col, emitted)

    return out


# --------------- generator ---------------

# Wire layout
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

# 9-rect placement
RECT_COUNT_WEIGHTS = [10, 40, 30, 20]
RECT_WIRE_COUNT_WEIGHTS = [50, 38, 12]
RECT_PARALLEL_PROB = 0.75
RECT_H_RANGE = (2, 5)
RECT_MAX_WIDTH = 7
RECT_RETRY_FACTOR = 5
RECT_MARGIN_GAP = 1

# Variant-specific input derivation
LAYOUT_WEIGHTS = {"V": 1, "H": 1}
ANCHOR_INPUT_COUNT_WEIGHTS = [55, 35, 10]
FAR_CORE_PROB = 0.35

# Retry budget
GEN_MAX_RETRIES = 300


def _decompose_to_crossovers(perm: list[int]) -> list[tuple[int, int]]:
    """Decompose a permutation into sequential crossover moves using the bypass wire."""
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
    """Attempt to place one color-9 rectangle into the wire field."""
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

    if rect_c0 < 0 or rect_c1 >= w or rect_c1 - rect_c0 + 1 > RECT_MAX_WIDTH:
        return None

    rect_h_max = min(RECT_H_RANGE[1], bot_ep_row - top_ep_row - 4)
    if rect_h_max < RECT_H_RANGE[0]:
        return None
    rect_h = random.randint(RECT_H_RANGE[0], rect_h_max)
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
        if not np.all(grid[rect_r0:rect_r1 + 1, c] == 8):
            return None

    sub = grid[rect_r0:rect_r1 + 1, rect_c0:rect_c1 + 1]
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

    saved = grid[rect_r0:rect_r1 + 1, rect_c0:rect_c1 + 1].copy()
    grid[rect_r0:rect_r1 + 1, rect_c0:rect_c1 + 1] = 9

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
        grid[rect_r0:rect_r1 + 1, rect_c0:rect_c1 + 1] = saved
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
    """Draw the wire routing and return the final logical wire assignment."""
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
        grid[xrow, c_min:c_max + 1] = 8

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
                    grid[start:bot_ep_row + 1, c] = 8

    return slot_wire, bypass_wire


def _build_canonical_wire(min_height: int) -> dict[str, object] | None:
    """Build the original top-bottom wire layout; horizontal samples are derived by transposition later."""
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
    n_xovers = len(crossovers)

    min_wire_rows = n_xovers * 2 + 2
    h_floor = max(min_height, GRID_H_RANGE[0])
    h_min = top_ep_row + min_wire_rows + top_token_row + 3
    if max(h_floor, h_min) > GRID_H_RANGE[1]:
        return None
    height = random.randint(max(h_floor, h_min), GRID_H_RANGE[1])

    bot_token_row = height - top_token_row - 1
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

    grid = np.zeros((height, width), dtype=int)
    slot_wire, bypass_wire = _build_wire_network(
        grid,
        n_channels,
        ch_cols,
        bypass_col,
        top_ep_row,
        bot_ep_row,
        crossovers,
        xover_rows,
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

    return {
        "grid": grid,
        "n_channels": n_channels,
        "ch_cols": ch_cols,
        "perm": perm,
        "token_colors": token_colors,
        "top_token_row": top_token_row,
        "bot_token_row": bot_token_row,
    }


def _physical_centers(meta: dict[str, object], layout: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]], np.ndarray]:
    """Return the two endpoint center lists for one sample layout plus the wire-only grid."""
    grid = np.array(meta["grid"], copy=True)
    ch_cols = list(meta["ch_cols"])
    top_token_row = int(meta["top_token_row"])
    bot_token_row = int(meta["bot_token_row"])

    top_centers = [(top_token_row, c) for c in ch_cols]
    bot_centers = [(bot_token_row, c) for c in ch_cols]

    if layout == "V":
        return top_centers, bot_centers, grid

    transposed = grid.T.copy()
    left_centers = [(c, top_token_row) for c in ch_cols]
    right_centers = [(c, bot_token_row) for c in ch_cols]
    return left_centers, right_centers, transposed


def _channel_endpoints(
    side_a: list[tuple[int, int]],
    side_b: list[tuple[int, int]],
    perm: list[int],
    side_choice: int,
    i: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if side_choice == 0:
        return side_a[i], side_b[perm[i]]
    return side_b[perm[i]], side_a[i]


def _gen_attempt() -> np.ndarray | None:
    layout = random.choices(list(LAYOUT_WEIGHTS.keys()), weights=list(LAYOUT_WEIGHTS.values()))[0]
    min_height = GRID_H_RANGE[0] if layout == "V" else GRID_W_MIN
    meta = _build_canonical_wire(min_height)
    if meta is None:
        return None

    perm = list(meta["perm"])
    token_colors = list(meta["token_colors"])
    n_channels = int(meta["n_channels"])
    side_a, side_b, wire_grid = _physical_centers(meta, layout)
    emitted = _CORE_V if layout == "V" else _CORE_H

    source_sides = [random.choice([0, 1]) for _ in range(n_channels)]
    output = wire_grid.copy()
    for i in range(n_channels):
        source, dest = _channel_endpoints(side_a, side_b, perm, source_sides[i], i)
        col = token_colors[i]
        _stamp_token(output, source, col, _ANCHOR)
        _stamp_token(output, dest, col, emitted)

    inp = wire_grid.copy()
    n_anchor = random.choices([0, 1, 2], weights=ANCHOR_INPUT_COUNT_WEIGHTS)[0]
    anchor_channels = set(random.sample(range(n_channels), min(n_anchor, n_channels)))
    for i in range(n_channels):
        source, dest = _channel_endpoints(side_a, side_b, perm, source_sides[i], i)
        col = token_colors[i]
        if i in anchor_channels:
            _stamp_token(inp, source, col, _ANCHOR)
            if random.random() < FAR_CORE_PROB:
                _stamp_token(inp, dest, col, emitted)
        else:
            _stamp_token(inp, source, col, _SOLID)

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
