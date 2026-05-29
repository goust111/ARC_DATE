# Variant of a47bf94d
# Dimension: D5 — 形状补全规则 (transform)
# Change: 形状补全（shape_completion）——如果管道一端已存在十字形或四角星形，则在另一端放置互补形状（四角星对应十字，十字对应四角星），颜色与已有形状相同 → 颜色变换补全

import itertools
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID_OFF = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]


# Design analysis for the D5 recolor variant:
# - Concrete behavior: the spatial rule stays the same as the base task. A solid 3x3 still collapses
#   into an in-place diamond, and an X still remains in place. The only changed step is the remote
#   complementary shape: its color is offset by how many maroon 9-rectangles the routed wire passes.
# - Small trace: in the original task, a top solid of color 2 routed through two 9-rectangles would
#   produce a bottom X of color 2. In this variant, that exact path instead produces a bottom X of
#   color 4. Likewise, a bottom X of color 1 routed through one 9-rectangle produces a top diamond
#   of color 2 rather than color 1.
# - Complexity calibration: to make the recolor visible without depending on hidden color ids, the
#   generator fixes four channels and uses distinct base colors {1,2,3,4}. It also forces the four
#   paths to realize counts {0,0,1,2} in some order, so every sample contains unchanged, +1, and +2
#   examples. This avoids the degenerate case where all paths see the same count.
# - Implicit assumptions audited:
#   1. The original track extraction removed only solids and X tokens. That was harmless when far-end
#      diamonds were usually absent, but it breaks here because optional pre-placed diamonds would be
#      mistaken for wire branches. The solver therefore detects isolated diamonds and removes them
#      from the wire mask.
#   2. The original remote write used the source color unconditionally. In this variant that would
#      erase the entire D5 change, so the transmitted color is computed from the traced 9-rect count.
#   3. The original generator could randomly put solids or Xs on either side and rely on oracle
#      rejection. Here the generator uses side choices that intentionally exercise the recolor rule
#      while keeping the output easy to read.
# - Placement legality: the generator places three non-overlapping single-wire 9-rectangles only in
#   the guaranteed clear vertical bands above the first crossover or below the last crossover. Those
#   segments are long enough, inside bounds, and belong to predictable channels, so the resulting
#   path counts are controlled and the solver sees valid endpoint geometry.
# - Visual observability and cleanliness: each sample shows three count classes (0/1/2), the 9-rects
#   are large enough to count by eye, and remote recolors happen on isolated 3x3 tokens rather than
#   on the dense wire body. This keeps the grid readable.
# - Color-change inferability: because base colors are the consecutive visible set {1,2,3,4} and the
#   forced counts are 0/1/2, the observed recolors form a simple ladder within that same set rather
#   than wrapping into wire colors or relying on arbitrary hidden ordering.


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
            if not isolated:
                continue
            found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _find_diamond_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for isolated 3x3 diamonds."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}

    mask = np.zeros((3, 3), dtype=bool)
    for rr, cc in [(0, 1), (1, 0), (1, 2), (2, 1)]:
        mask[rr, cc] = True

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            vals = {int(sub[rr, cc]) for rr, cc in [(0, 1), (1, 0), (1, 2), (2, 1)]}
            if len(vals) != 1:
                continue
            col = vals.pop()
            if col == bg:
                continue
            if np.any(sub[mask] != col) or np.any(sub[~mask] != bg):
                continue
            isolated = True
            cr, cc = r0 + 1, c0 + 1
            for dr0, dc0 in _DIAMOND_OFF:
                rr, cc2 = cr + dr0, cc + dc0
                for nr, nc in neighbors4(rr, cc2):
                    if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                        if int(g[nr, nc]) == col:
                            isolated = False
                            break
                if not isolated:
                    break
            if not isolated:
                continue
            found[(cr, cc)] = (col, (r0, c0))
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
) -> tuple[tuple[int, int], tuple[int, int], int]:
    """
    Greedy wire following with 9-rect pass-through.

    Returns (end_cell, outgoing_direction, rect_pass_count).
    """
    cur, d, prev = start, dir0, None
    rect_passes = 0
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
                rect_passes += 1
                jumped = True
                break
        if jumped:
            continue

        r, c = cur
        opts = [((nr, nc), nr - r, nc - c) for nr, nc in neighbors4(r, c) if (nr, nc) in comp]
        if not opts:
            return cur, d, rect_passes

        rev = (-d[0], -d[1])
        nonrev = [o for o in opts if (o[1], o[2]) != rev]
        if not nonrev:
            return cur, d, rect_passes
        opts2 = nonrev

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
            return cur, d, rect_passes

        prev = cur
        cur = p
        d = (dr, dc)

    raise AssertionError("wire trace exceeded step limit")


def _stamp_shape(grid: np.ndarray, center: tuple[int, int], color: int, offsets: list[tuple[int, int]]) -> None:
    cr, cc = center
    for dr, dc in offsets:
        grid[cr + dr, cc + dc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route 3x3 tokens across the wire network; the remote complement is recolored by 9-rect count.

    Input structure:
    - Background cells surround one orthogonal wire network made of 8/5/9, with 3x3 solid blocks,
      X patterns, and occasionally already-resting diamonds sitting two cells away from endpoints.

    Rule:
    - Every solid 3x3 token collapses in place into a diamond of the same color.
    - Any solid or X token sends its complementary shape to the far endpoint, but the transmitted
      color is increased by the number of maroon 9-rectangles that path passes through.

    Key insight: Shape logic is unchanged; only the remote token recolors, and the recolor amount is
    exactly the count of 9-rect pass-through events encountered while tracing that endpoint's wire.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)
    diamonds = _find_diamond_3x3(g, bg)

    track_mask = g != bg
    for _, (_, (r0, c0)) in solids.items():
        track_mask[r0 : r0 + 3, c0 : c0 + 3] = False
    for (cr, cc) in xs.keys():
        for dr, dc in _X_OFF:
            track_mask[cr + dr, cc + dc] = False
    for (cr, cc) in diamonds.keys():
        for dr, dc in _DIAMOND_OFF:
            track_mask[cr + dr, cc + dc] = False

    comps = _components(track_mask)
    rects_9 = _find_9rects(g)
    out = g.copy()

    for (cr, cc), (col, (r0, c0)) in solids.items():
        out[r0 : r0 + 3, c0 : c0 + 3] = bg
        _stamp_shape(out, (cr, cc), col, _DIAMOND_OFF)

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
                    end, dd, rect_passes = _trace_wire(comp, ep, (-odr, -odc), h, w, rects_9)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    shifted = col + rect_passes
                    for dr, dc in _X_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = shifted

                elif cen in xs:
                    col, _ = xs[cen]
                    if col in comp_colors:
                        continue
                    end, dd, rect_passes = _trace_wire(comp, ep, (-odr, -odc), h, w, rects_9)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    shifted = col + rect_passes
                    for dr, dc in _DIAMOND_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = shifted

    return out


# --- Generator configuration ---
CHANNEL_COUNTS = [4]
CHANNEL_SPACING = [4, 5]
LEFT_MARGIN_RANGE = (2, 3)
BYPASS_COL_OFFSET = 2
TOKEN_ROW_OPTIONS = [2, 3]
GRID_H_RANGE = (20, 26)
GRID_W_MIN = 22
GRID_W_MAX = 26
EP_TOKEN_GAP = 2

BASE_TOKEN_COLORS = [1, 2, 3, 4]
X_INPUT_COUNT_WEIGHTS = [35, 45, 20]  # 0,1,2 X-input channels
FAR_DIAMOND_PROB = 0.35

RECT_H_RANGE = (2, 4)
RECT_MARGIN_GAP = 1
GEN_MAX_RETRIES = 300


def _decompose_to_crossovers(perm: list[int]) -> list[tuple[int, int]]:
    """Decompose a permutation into sequential crossover moves using the bypass column."""
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
    """Draw the wire network onto grid. Returns (slot_wire, bypass_wire) after all crossovers."""
    slot_wire: list[int | None] = list(range(n))
    bypass_wire: int | None = None

    col_seg_start: dict[int, int] = {}
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


def _place_vertical_rect(
    grid: np.ndarray,
    col: int,
    row_lo: int,
    row_hi: int,
    placed_rects: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    """Place one non-overlapping single-wire 9-rectangle on a vertical wire segment."""
    h, w = grid.shape
    if row_hi - row_lo + 1 < RECT_H_RANGE[0]:
        return None

    attempts = 20
    for _ in range(attempts):
        rect_h = random.randint(RECT_H_RANGE[0], min(RECT_H_RANGE[1], row_hi - row_lo + 1))
        r0 = random.randint(row_lo, row_hi - rect_h + 1)
        r1 = r0 + rect_h - 1
        c0, c1 = col - 1, col + 1
        if c0 < 0 or c1 >= w:
            continue
        if r0 <= 0 or r1 >= h - 1:
            continue
        if grid[r0 - 1, col] != 8 or grid[r1 + 1, col] != 8:
            continue
        if not np.all(grid[r0 : r1 + 1, col] == 8):
            continue

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all((sub == 0) | (sub == 5) | (sub == 8)):
            continue

        gap = RECT_MARGIN_GAP
        clash = False
        for pr0, pc0, pr1, pc1 in placed_rects:
            if r0 <= pr1 + gap and pr0 <= r1 + gap and c0 <= pc1 + gap and pc0 <= c1 + gap:
                clash = True
                break
        if clash:
            continue

        grid[r0 : r1 + 1, c0 : c1 + 1] = 9
        rect = (r0, c0, r1, c1)
        placed_rects.append(rect)
        return rect
    return None


def _trace_channel_info(
    grid: np.ndarray,
    ch_cols: list[int],
    top_ep_row: int,
) -> tuple[list[int], list[int]]:
    """Trace from each top endpoint and return (destination columns, rect counts)."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    rects_9 = _find_9rects(g)
    comps = _components(g != 0)

    dest_cols = []
    rect_counts = []
    for c in ch_cols:
        start = (top_ep_row, c)
        comp = None
        for pts in comps:
            s = set(pts)
            if start in s:
                comp = s
                break
        assert comp is not None
        end, _dd, rect_passes = _trace_wire(comp, start, (1, 0), h, w, rects_9)
        dest_cols.append(end[1])
        rect_counts.append(rect_passes)
    return dest_cols, rect_counts


def _assign_base_colors(path_counts: list[int]) -> list[int] | None:
    """Assign distinct base colors 1..4 so recolored outputs stay within 1..4."""
    perms = list(itertools.permutations(BASE_TOKEN_COLORS))
    random.shuffle(perms)
    for perm in perms:
        if all(perm[i] + path_counts[i] <= 4 for i in range(len(path_counts))):
            return list(perm)
    return None


def _derive_input_and_output(
    grid: np.ndarray,
    ch_cols: list[int],
    perm: list[int],
    base_colors: list[int],
    path_counts: list[int],
    top_token_row: int,
    bot_token_row: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a consistent input/output pair for the recolor variant."""
    n = len(ch_cols)
    inp = grid.copy()
    out = grid.copy()

    n_x = random.choices([0, 1, 2], weights=X_INPUT_COUNT_WEIGHTS)[0]
    x_channels = set(random.sample(range(n), min(n_x, n)))

    for i in range(n):
        base = base_colors[i]
        shifted = base + path_counts[i]
        top_center = (top_token_row, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[perm[i]])

        if i in x_channels:
            _stamp_shape(inp, bot_center, base, _X_OFF)
            if random.random() < FAR_DIAMOND_PROB:
                _stamp_shape(inp, top_center, shifted, _DIAMOND_OFF)

            _stamp_shape(out, bot_center, base, _X_OFF)
            _stamp_shape(out, top_center, shifted, _DIAMOND_OFF)
        else:
            _stamp_shape(inp, top_center, base, _SOLID_OFF)
            _stamp_shape(out, top_center, base, _DIAMOND_OFF)
            _stamp_shape(out, bot_center, shifted, _X_OFF)

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

    while True:
        perm = list(range(n))
        random.shuffle(perm)
        if perm != list(range(n)):
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
    xover_rows = [top_ep_row + 1 + step * (s + 1) for s in range(n_xovers)]
    if xover_rows[-1] >= bot_ep_row:
        xover_rows = [top_ep_row + 2 + s * 2 for s in range(n_xovers)]
    if xover_rows[-1] >= bot_ep_row:
        return None

    top_band = (top_ep_row + 1, xover_rows[0] - 1)
    bot_band = (xover_rows[-1] + 1, bot_ep_row - 1)
    if top_band[1] - top_band[0] + 1 < RECT_H_RANGE[0]:
        return None
    if bot_band[1] - bot_band[0] + 1 < RECT_H_RANGE[0]:
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

    double_src, single_src = random.sample(range(n), 2)
    placed_rects: list[tuple[int, int, int, int]] = []
    if _place_vertical_rect(grid, ch_cols[double_src], top_band[0], top_band[1], placed_rects) is None:
        return None
    if _place_vertical_rect(grid, ch_cols[perm[double_src]], bot_band[0], bot_band[1], placed_rects) is None:
        return None
    if _place_vertical_rect(grid, ch_cols[single_src], top_band[0], top_band[1], placed_rects) is None:
        return None

    traced_dest_cols, traced_counts = _trace_channel_info(grid, ch_cols, top_ep_row)
    if traced_dest_cols != [ch_cols[p] for p in perm]:
        return None

    expected_counts = [0] * n
    expected_counts[double_src] = 2
    expected_counts[single_src] = 1
    if traced_counts != expected_counts:
        return None

    base_colors = _assign_base_colors(expected_counts)
    if base_colors is None:
        return None

    inp, expected = _derive_input_and_output(
        grid, ch_cols, perm, base_colors, expected_counts, top_token_row, bot_token_row
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
