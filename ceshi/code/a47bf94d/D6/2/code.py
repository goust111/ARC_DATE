# Variant of a47bf94d
# Dimension: D6 — 源token形状类型 (src)
# Change: 三种token形状——实心3x3块（输入中主要形式）、X形（对角+中心，5像素）、菱形（4正交邻居，4像素）；每种颜色最... → 多色复合源块
#
# Design analysis
# ----------------
# Source-task note:
# - The prompt asks to read the solver of `b5ca7ac4` first.
# - In this workspace there is no local solver file for `b5ca7ac4`; only
#   `/home/lixinyu/snap/new/ARC-data/ARC-AGI-2/data/evaluation/b5ca7ac4.json`
#   exists. The implementation therefore uses only the stated
#   "color_grouping" idea: a single visual object carries multiple colors that
#   stay grouped with different subparts of the same shape.
#
# Variant spec:
#   grid: 20-26 rows x 22-26 cols
#   palette:
#     bg: 0
#     wire: 8 (main), 5 (crossing markers), 9 (filled pass-through rectangles)
#     token_colors: chosen from {1,2,3,4,6,7}
#   layout:
#     channels: 4-5 vertical channels with spacing 4-5
#     source side: all composite sources of a grid sit on one side (top or bottom)
#     far side: opposite side receives routed X outputs through a non-identity permutation
#   source token:
#     shape: full 3x3 block, fully filled
#     internal coloring:
#       - orthogonal 4 cells use one color D
#       - corners + center use another color X
#       - D != X
#     visual effect:
#       - the input already shows the exact two subparts that will be separated
#   output:
#     - at each source center, keep only the orthogonal 4-cell diamond in color D
#     - at the opposite endpoint, draw the 5-cell X in color X
#   generator:
#     - output-first, same wire/9-rect machinery as the base task
#     - ordered color pairs are sampled per channel; colors may repeat across
#       channels but each pair is distinct, so the grid truly carries double-color tracking
#
# a) Concrete behavior trace:
# - Original channel: a solid 3x3 color C near one endpoint becomes a local diamond
#   of color C and emits a far X of color C.
# - New channel: a full 3x3 source uses two colors. Example:
#     X-mask cells = red, orthogonal cells = yellow.
#   The solver clears that 3x3, leaves a yellow diamond at the same center, traces
#   the wire to the opposite endpoint, and stamps a red X there.
# - Difference: color tracking no longer collapses to one channel color. The input
#   itself encodes which output shape gets which color.
#
# b) Complexity calibration:
# - The split is still between exactly two fixed masks (diamond vs X), so the task
#   is richer than the original but not combinatorially loose.
# - Using a single global source side per grid keeps the scene readable; using
#   several channels per grid provides repeated evidence for the same rule.
#
# c) Implicit-assumption audit against the base solver:
# - `_find_solid_3x3` relied on all 9 cells sharing one color. That is false now,
#   so source detection must be replaced with a mask-aware 2-color detector.
# - The old in-place rewrite assigned one color to both local and remote outputs.
#   That would silently destroy the variant, so local and routed colors are stored
#   separately as `(diamond_color, x_color)`.
# - Track extraction used token-cell removal. That assumption still holds, but now
#   the whole 3x3 composite block must be removed from the wire mask.
# - Endpoint matching still assumes the token center is exactly two cells outward
#   from a degree-1 wire endpoint. The generator preserves that geometry.
#
# d) Placement legality:
# - Composite tokens occupy the same 3x3 footprint as the base solid token, so the
#   original margin and endpoint spacing remain valid.
# - The generator places all sources on one side only, preventing source/output
#   overlap and ensuring the far X never lands on another source block.
# - Ordered color pairs exclude wire colors and require distinct sub-colors.
#
# e) Visual observability:
# - The new behavior is visible directly in the input: every active 3x3 source is
#   already two-colored, and its two color groups match the two output shapes.
# - No invisible ordering by color id is used; the only cue is the visible 3x3 mask.
#
# f) Visual cleanliness:
# - Keeping one source side per grid avoids a checkerboard of source/output roles.
# - The two masks partition the 3x3 exactly, so there is no ambiguous overlap.
#
# g) Color-rule inferability:
# - Each grid contains 4-5 channels, so the same split rule repeats multiple times.
# - Colors may repeat across channels, but the rule depends only on visible mask
#   membership, not on any specific color identity.
#
# h) Post-change assumptions checked:
# - `main()` assumes every routed source is a full 3x3 with the fixed two-mask color
#   pattern. `generate_input()` constructs exactly that.
# - `main()` assumes each track component endpoint is adjacent to at most one source
#   center at distance 2. The generator keeps the original one-token-per-endpoint layout.

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID_OFF = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]


def _find_composite_3x3(
    g: np.ndarray, bg: int
) -> dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]]:
    """Return {center: ((diamond_color, x_color), (r0,c0))} for full 3x3 two-color sources."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]] = {}
    x_mask = np.zeros((3, 3), dtype=bool)
    for rr, cc in [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]:
        x_mask[rr, cc] = True

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            if np.any(sub == bg):
                continue
            x_col = int(sub[0, 0])
            d_col = int(sub[0, 1])
            if x_col == d_col:
                continue
            if np.any(sub[x_mask] != x_col) or np.any(sub[~x_mask] != d_col):
                continue
            found[(r0 + 1, c0 + 1)] = ((d_col, x_col), (r0, c0))
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each two-color 3x3 source block into a local diamond and a routed X while preserving both colors.

    Input structure:
    - bg = background. Non-bg cells form wire networks plus full 3x3 composite source blocks near wire endpoints.
    - A valid source block is fully filled: the orthogonal 4 cells share one token color, and the corners plus center share a second token color.

    Rule:
    - Every composite source block is cleared and replaced in-place by the orthogonal 4-cell diamond using its orthogonal color.
    - The X-mask color of that source is sent through the attached wire component to the opposite endpoint, where it appears as a monochrome X.

    Key insight:
    - The source already contains the two output parts: orthogonal cells tell you the local diamond color, and the X-mask cells tell you the routed far-end color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    sources = _find_composite_3x3(g, bg)

    track_mask = g != bg
    for _, (_, (r0, c0)) in sources.items():
        track_mask[r0 : r0 + 3, c0 : c0 + 3] = False

    comps = _components(track_mask)
    rects_9 = _find_9rects(g)

    out = g.copy()

    for (cr, cc), ((diamond_col, _x_col), (r0, c0)) in sources.items():
        out[r0 : r0 + 3, c0 : c0 + 3] = bg
        for dr, dc in _DIAMOND_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                out[rr, cc2] = diamond_col

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
                if cen not in sources:
                    continue
                x_col = sources[cen][0][1]
                dir_in = (-odr, -odc)
                end, dd = _trace_wire(comp, ep, dir_in, h, w, rects_9)
                dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                for dr, dc in _X_OFF:
                    rr, cc2 = dest[0] + dr, dest[1] + dc
                    if 0 <= rr < h and 0 <= cc2 < w:
                        out[rr, cc2] = x_col

    return out


# --------------- generator (two_color_composite_sources) ---------------

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


def _stamp_cells(grid, center_r, center_c, color, offsets):
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = color


def _stamp_composite_source(grid, center_r, center_c, diamond_color, x_color):
    _stamp_cells(grid, center_r, center_c, x_color, _X_OFF)
    _stamp_cells(grid, center_r, center_c, diamond_color, _DIAMOND_OFF)


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


def _sample_color_pairs(n: int) -> list[tuple[int, int]]:
    pairs = [(d, x) for d in TOKEN_COLORS_POOL for x in TOKEN_COLORS_POOL if d != x]
    while True:
        chosen = random.sample(pairs, n)
        diamond_cols = {d for d, _ in chosen}
        x_cols = {x for _, x in chosen}
        all_cols = {c for pair in chosen for c in pair}
        if len(diamond_cols) >= 2 and len(x_cols) >= 2 and len(all_cols) >= min(4, n):
            return chosen


def _channel_centers(i, perm, ch_cols, top_token_row, bot_token_row, source_on_top):
    top_center = (top_token_row, ch_cols[i])
    bot_center = (bot_token_row, ch_cols[perm[i]])
    if source_on_top:
        return top_center, bot_center
    return bot_center, top_center


def _derive_input(grid, perm, ch_cols, color_pairs, top_token_row, bot_token_row, source_on_top):
    inp = grid.copy()
    for i, (diamond_col, x_col) in enumerate(color_pairs):
        src_center, _ = _channel_centers(i, perm, ch_cols, top_token_row, bot_token_row, source_on_top)
        _stamp_composite_source(inp, src_center[0], src_center[1], diamond_col, x_col)
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

    source_on_top = random.choice([True, False])
    color_pairs = _sample_color_pairs(n)

    output = grid.copy()
    for i, (diamond_col, x_col) in enumerate(color_pairs):
        src_center, dst_center = _channel_centers(i, perm, ch_cols, top_token_row, bot_token_row, source_on_top)
        _stamp_cells(output, src_center[0], src_center[1], diamond_col, _DIAMOND_OFF)
        _stamp_cells(output, dst_center[0], dst_center[1], x_col, _X_OFF)

    inp = _derive_input(grid, perm, ch_cols, color_pairs, top_token_row, bot_token_row, source_on_top)
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
