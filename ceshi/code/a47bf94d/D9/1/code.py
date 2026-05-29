# Variant of a47bf94d
# Dimension: D9 — 形状循环关系 (transform)
# Change: 三形状循环: solid → (留十字, 发X) ; X → (留X, 发diamond) ; diamond → (已完成)
#         → 四形状循环: solid → X → diamond → thin-cross → solid

import random
import sys

import numpy as np

sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import components_mask, get_bg, neighbors4


# Design analysis for the D9 variant
# a) Concretization / trace:
#    Keep the original wire layout, endpoint spacing, and color-9 pass-through blocks.
#    Only the token transform changes. Tokens are now exact 3x3 templates:
#    - solid: all 9 cells
#    - x: diagonals + center
#    - diamond: 4 cardinal neighbors, no center
#    - thin_cross: center + 4 cardinal neighbors
#    Each input token sits two cells outside a wire endpoint as before. The source token
#    transforms locally by two steps in the cycle, while the far endpoint receives the
#    one-step successor. Example: a top-side X token of color 3 becomes a top thin_cross
#    locally and paints a bottom diamond at the routed partner endpoint.
#    This preserves the original solid -> (local diamond, remote X) behavior while
#    extending the same table to two additional stages.
# b) Complexity calibration:
#    Four templates remain small, symmetric, and easy to compare across examples.
#    The new thin_cross differs from diamond by the center cell only, so the generator
#    must produce repeated occurrences of both; otherwise the new state would be too subtle.
#    The generator therefore forces at least one channel to start in {diamond, thin_cross}
#    and also requires at least two distinct source shapes in each puzzle.
# c) Hidden assumptions review:
#    - Original detection only handled solid and X. Here every token lookup must be
#      template-based; otherwise diamond/thin_cross inputs would be mistaken for wire noise.
#    - Original track removal blanked whole 3x3 solids and only X cells. Now the track mask
#      must subtract the exact occupied cells of any recognized template.
#    - Original local rewrite only cleared solids. Now every source token may shrink or grow,
#      so the whole 3x3 token box is cleared before stamping the local result.
#    - Original remote painting assumed only X/diamond outputs. Now it must dispatch by table.
# d) Placement legality:
#    Channel spacing, endpoint gap, and token rows are unchanged from the base task, so any
#    3x3 template still fits without overlap or border issues. Output-side solids are safe
#    because token centers already had room for a full 3x3 in the original task.
# e) Visual observability:
#    A solver can see the new rule from a few examples because the same color may now start
#    as diamond or thin_cross and produce remote thin_cross or solid, cases impossible in the
#    base task. The cycle is visible by repeated color-shape correspondences, not by color ids.
# f) Visual cleanliness:
#    The wire language (8/5/9) is untouched, and each endpoint region still contains at most
#    one 3x3 token. No extra overlays are introduced on the same endpoint.
# g) Color inferability:
#    Colors still act only as identity labels transported by the wire permutation; no color
#    remapping or ordering is introduced.
# h) Source-task note:
#    The prompt lists no separate source_task, so the original task code is the only baseline.
# i) New solver assumptions to satisfy in generation:
#    - every token matches exactly one of the four templates
#    - at most one source token is adjacent to a given endpoint
#    - far endpoints start empty, so remote stamping cannot conflict with another source token
#    - token colors stay outside {5, 8, 9}, preserving wire/token separation


_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_SOLID_OFF = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_THIN_CROSS_OFF = [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)]

_TOKEN_OFFSETS = {
    "solid": _SOLID_OFF,
    "x": _X_OFF,
    "diamond": _DIAMOND_OFF,
    "thin_cross": _THIN_CROSS_OFF,
}
_TOKEN_SHAPES = tuple(_TOKEN_OFFSETS.keys())
_NEXT_SHAPE = {
    "solid": "x",
    "x": "diamond",
    "diamond": "thin_cross",
    "thin_cross": "solid",
}
_LOCAL_SHAPE = {shape: _NEXT_SHAPE[_NEXT_SHAPE[shape]] for shape in _TOKEN_SHAPES}


def _find_pattern_3x3(
    g: np.ndarray, bg: int, offsets: list[tuple[int, int]]
) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for isolated exact 3x3 templates."""
    h, w = g.shape
    mask = np.zeros((3, 3), dtype=bool)
    cells_3x3 = []
    for dr, dc in offsets:
        rr, cc = 1 + dr, 1 + dc
        mask[rr, cc] = True
        cells_3x3.append((rr, cc))

    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            first_rr, first_cc = cells_3x3[0]
            col = int(sub[first_rr, first_cc])
            if col == bg:
                continue
            if any(int(sub[rr, cc]) != col for rr, cc in cells_3x3):
                continue
            if np.any(sub[~mask] != bg):
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
            if not isolated:
                continue

            found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _find_tokens(g: np.ndarray, bg: int) -> dict[tuple[int, int], dict[str, object]]:
    """Detect all isolated token templates near endpoints."""
    tokens: dict[tuple[int, int], dict[str, object]] = {}
    for shape in _TOKEN_SHAPES:
        for center, (color, (r0, c0)) in _find_pattern_3x3(g, bg, _TOKEN_OFFSETS[shape]).items():
            if center in tokens:
                raise AssertionError("token templates overlapped unexpectedly")
            tokens[center] = {"shape": shape, "color": color, "r0": r0, "c0": c0}
    return tokens


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
    """
    Greedy wire following with 9-rect pass-through.

    At a color-9 rectangle: exit from opposite face (aligned with entry column/row).
    If no opposite exit, assert exactly one other exit face and use that.
    """
    cur, d, prev = start, dir0, None
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 4):
        if (cur, d) in visited:
            raise AssertionError("wire trace looped - should not happen with 9-rect pass-through")
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


def _stamp_token(grid: np.ndarray, center_r: int, center_c: int, color: int, offsets: list[tuple[int, int]]) -> None:
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route endpoint tokens through the wire network while advancing them around a four-shape cycle.

    Input structure:
    - Background cells are empty. Non-background cells form the original wire system (8, 5, 9)
      plus isolated 3x3 tokens placed two cells outside wire endpoints.
    - Each token is exactly one of four templates: solid, X, diamond, or thin-cross.

    Rule:
    - A source token rewrites itself in place to the shape two steps ahead in the cycle
      solid -> X -> diamond -> thin-cross -> solid.
    - The same source token also sends its one-step successor to the routed partner endpoint
      of the same wire component.

    Key insight:
    - The geometry of the wires is unchanged from the base task; the only new deduction is the
      longer cycle table. Read the token shape, advance it by one step for the far end and by
      two steps for the local rewrite, then let the wire permutation decide where the color lands.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    tokens = _find_tokens(g, bg)

    track_mask = g != bg
    for (cr, cc), info in tokens.items():
        for dr, dc in _TOKEN_OFFSETS[info["shape"]]:
            rr, cc2 = cr + dr, cc + dc
            track_mask[rr, cc2] = False

    comps = _components(track_mask)
    rects_9 = _find_9rects(g)

    out = g.copy()

    for (cr, cc), info in tokens.items():
        color = int(info["color"])
        r0 = int(info["r0"])
        c0 = int(info["c0"])
        out[r0 : r0 + 3, c0 : c0 + 3] = bg
        _stamp_token(out, cr, cc, color, _TOKEN_OFFSETS[_LOCAL_SHAPE[info["shape"]]])

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
                if cen not in tokens:
                    continue

                info = tokens[cen]
                color = int(info["color"])
                if color in comp_colors:
                    continue

                dir_in = (-odr, -odc)
                end, dd = _trace_wire(comp, ep, dir_in, h, w, rects_9)
                dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                _stamp_token(out, dest[0], dest[1], color, _TOKEN_OFFSETS[_NEXT_SHAPE[info["shape"]]])

    return out


# --------------- generator (four_shape_cycle) ---------------

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

INPUT_SHAPE_WEIGHTS = [24, 28, 24, 24]
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


def _choose_input_shapes(n: int) -> list[str]:
    """Ensure each puzzle exhibits the new cycle beyond the original solid/X cases."""
    for _ in range(100):
        shapes = random.choices(_TOKEN_SHAPES, weights=INPUT_SHAPE_WEIGHTS, k=n)
        shapes[random.randrange(n)] = random.choice(["diamond", "thin_cross"])
        if len(set(shapes)) >= 2 and any(shape in ("diamond", "thin_cross") for shape in shapes):
            return shapes
    raise RuntimeError("failed to choose diverse input shapes")


def _derive_input_output(grid, n, ch_cols, perm, token_colors, top_token_row, bot_token_row):
    inp = grid.copy()
    out = grid.copy()
    input_shapes = _choose_input_shapes(n)

    for i in range(n):
        color = token_colors[i]
        top_center = (top_token_row, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[perm[i]])
        source_top = bool(random.getrandbits(1))
        near = top_center if source_top else bot_center
        far = bot_center if source_top else top_center

        shape = input_shapes[i]
        _stamp_token(inp, near[0], near[1], color, _TOKEN_OFFSETS[shape])
        _stamp_token(out, near[0], near[1], color, _TOKEN_OFFSETS[_LOCAL_SHAPE[shape]])
        _stamp_token(out, far[0], far[1], color, _TOKEN_OFFSETS[_NEXT_SHAPE[shape]])

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

    inp, expected = _derive_input_output(grid, n, ch_cols, perm, token_colors, top_token_row, bot_token_row)
    try:
        if np.array_equal(main(inp), expected):
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
