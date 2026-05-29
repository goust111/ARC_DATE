# Variant of a47bf94d
# Dimension: D11 — 通道数量与空间布局 (meta)
# Change: 4-5个通道，均匀列间距4-5，顶部和底部放置token行，管道竖直为主加横向交叉段... → 不等间距通道

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

# Design analysis
# variant:
#   name: a47bf94d D11 uneven-spacing channels
#   core_change:
#     original: channel centers sat on uniformly spaced columns, so the top and bottom token rows
#       looked like a regular comb.
#     new: channel centers still form one left-to-right ordered bundle, but the gaps are now
#       strictly increasing. For 4 channels the gaps are [3,4,5] or [4,5,6]; for 5 channels the
#       gaps are [3,4,5,6]. The bypass column stays to the right of the last channel.
#   concrete_behavior:
#     input:
#       - tokens still sit exactly two cells outside degree-1 wire endpoints.
#       - top and bottom token rows remain horizontal, but token centers are no longer equally spaced.
#       - the wire bundle still routes colors through 8-wires, 5 crossing markers, and optional 9-rects.
#     output:
#       - every source solid becomes an in-place diamond.
#       - the far endpoint still receives the complementary pattern (solid -> X, X -> diamond).
#       - the only visual meta-change is spatial: columns must be read from actual endpoints, not from
#         a constant inter-column rhythm.
#   behavior_trace:
#     original_example:
#       endpoint columns [3,7,11,15] let a reader infer "one token every 4 columns".
#     variant_example:
#       endpoint columns [3,6,10,15,21] have gaps [3,4,5,6]. The same routing rule applies, but a
#       reader cannot extrapolate the next token from a constant step; they must follow the wire/endpoints.
#     solver_impact:
#       - token detection must stay local to each endpoint instead of assuming regular channel positions.
#       - generator must guarantee enough horizontal room for the growing gaps and for crossover markers.
#   complexity_calibration:
#     - only one spacing family is used (strictly increasing gaps), so the visual rule is stable across
#       2-4 examples and not hidden behind too many cases.
#     - minimum gap is 3, which preserves a visible blank column between neighboring 3x3 token footprints
#       and leaves room for crossing markers.
#   implicit_assumption_audit:
#     - _trace_wire assumes local orthogonal connectivity only; unchanged and still valid.
#     - main() assumes token centers are two cells away from endpoints in a cardinal direction; preserved.
#     - main() must not infer channel identity from column order or constant spacing; this variant keeps the
#       solver endpoint-driven so the old "equal stride" assumption is not reintroduced.
#     - generator code that formerly built ch_cols via first_col + i * spacing is no longer valid and must
#       be replaced everywhere by an explicit irregular column list.
#   placement_legality:
#     - width is expanded to allow the widest 5-channel layout.
#     - 9-rect placement adapts its lateral margins to the actual span between adjacent channels so wider
#       gaps do not make rectangles impossible.
#     - tokens remain on the same top/bottom rows, still two rows away from endpoints, and never overlap.
#   visual_observability:
#     - the uneven spacing is directly visible in both input and output because each example contains 4-5
#       repeated channel centers on the same two token rows.
#     - no invisible attribute such as color id is used to choose the spacing pattern.
#   cleanliness:
#     - the bundle remains monotone left-to-right and mostly vertical, so the grid stays legible despite
#       the widened right side.
#   color_reasoning:
#     - colors keep their original role only; there is no new color remapping.
#   new_main_assumptions_checked_against_generator:
#     - each live wire component must still have endpoints whose outward side has room for a 3x3 token
#       center two cells away; generator enforces that with the same token rows/margins as before.
#     - crossing rows must stay separated by at least one row of vertical travel; generator keeps the
#       original spacing logic for xover_rows.

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]


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


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=4)]


def _find_9rects(g: np.ndarray) -> list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]]:
    """Find color-9 rectangles and the wire cells touching each face."""
    mask9 = g == 9
    if not mask9.any():
        return []

    h, w = g.shape
    rects = []
    for comp9 in components_mask(mask9, connectivity=4):
        cells = {(int(r), int(c)) for r, c in comp9}
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
    Greedily follow one wire component, allowing 9-rect pass-through.

    At a color-9 rectangle, continue straight through the opposite face when available.
    If the entry face has no opposite exit, use the sole other face.
    """
    cur, d, prev = start, dir0, None
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 4):
        if (cur, d) in visited:
            raise AssertionError("wire trace looped unexpectedly")
        visited.add((cur, d))

        jumped = False
        for cells9, faces9 in rects_9:
            if cur not in cells9:
                continue
            entry_face = _DIR2FACE[d]
            opp = _OPP[entry_face]
            if opp in faces9:
                exit_face = opp
            else:
                others = [f for f in faces9 if f != entry_face]
                if len(others) != 1:
                    raise AssertionError("ambiguous 9-rect exit")
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route endpoint tokens through an unevenly spaced wire bundle and stamp their complementary forms.

    Input structure:
    - The background surrounds one orthogonal wire network made of colors 8, 5, and optional 9-rectangles.
    - Colored 3x3 tokens still sit two cells beyond wire endpoints, but the channel columns on the top and
      bottom token rows are not equally spaced anymore.

    Rule:
    - Any solid 3x3 token collapses in place to a diamond (the four cardinal cells around its center).
    - Any token adjacent to an endpoint sends its complementary pattern to the wire's opposite endpoint:
      solid emits an X, and X emits a diamond.
    - Column rhythm is irrelevant; only the actual endpoint geometry and wire connectivity determine where
      the transmitted token lands.

    Key insight:
    - Do not infer channel positions from a constant step size. Detect tokens locally from each endpoint,
      trace the connected wire to the far endpoint, and stamp the complementary 3x3 footprint there.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)

    track_mask = g != bg
    for _, (_, (r0, c0)) in solids.items():
        track_mask[r0 : r0 + 3, c0 : c0 + 3] = False
    for cr, cc in xs:
        for dr, dc in _X_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                track_mask[rr, cc2] = False

    comps = _components(track_mask)
    rects_9 = _find_9rects(g)
    out = g.copy()

    for (cr, cc), (col, (r0, c0)) in solids.items():
        out[r0 : r0 + 3, c0 : c0 + 3] = bg
        for dr, dc in _DIAMOND_OFF:
            rr, cc2 = cr + dr, cc + dc
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
                    dir_in = (-odr, -odc)
                    end, dd = _trace_wire(comp, ep, dir_in, h, w, rects_9)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    for dr, dc in _X_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col

                elif cen in xs:
                    col, _ = xs[cen]
                    if col in comp_colors:
                        continue
                    dir_in = (-odr, -odc)
                    end, dd = _trace_wire(comp, ep, dir_in, h, w, rects_9)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    for dr, dc in _DIAMOND_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col

    return out


# --------------- generator ---------------

_DIAMOND = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]

CHANNEL_COUNTS = [4, 5]
LEFT_MARGIN_RANGE = (2, 3)
BYPASS_COL_OFFSET = 2
TOKEN_ROW_OPTIONS = [2, 3]
TOKEN_COLORS_POOL = [1, 2, 3, 4, 6, 7]
GRID_H_RANGE = (20, 26)
GRID_W_MIN = 24
GRID_W_MAX = 28
EP_TOKEN_GAP = 2

RECT_COUNT_WEIGHTS = [10, 40, 30, 20]
RECT_WIRE_COUNT_WEIGHTS = [50, 38, 12]
RECT_PARALLEL_PROB = 0.75
RECT_H_RANGE = (2, 5)
RECT_MAX_WIDTH = 8
RECT_RETRY_FACTOR = 5
RECT_MARGIN_GAP = 1

X_INPUT_COUNT_WEIGHTS = [60, 30, 10]
FAR_DIAMOND_PROB = 0.33

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


def _stamp_token(grid: np.ndarray, center_r: int, center_c: int, color: int, offsets: list[tuple[int, int]]) -> None:
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = color


def _build_channel_columns(n: int) -> tuple[list[int], list[int], int, int]:
    """
    Build an uneven left-to-right channel layout with strictly increasing gaps.

    Returns (channel_columns, gaps, bypass_col, grid_width).
    """
    left_margin = random.randint(*LEFT_MARGIN_RANGE)
    first_col = left_margin + 1
    gap_start = 3 if n == 5 else random.choice([3, 4])
    gaps = [gap_start + i for i in range(n - 1)]
    ch_cols = [first_col]
    for gap in gaps:
        ch_cols.append(ch_cols[-1] + gap)
    bypass_col = ch_cols[-1] + BYPASS_COL_OFFSET
    width = max(bypass_col + 3, GRID_W_MIN)
    return ch_cols, gaps, bypass_col, width


def _choose_rect_span(mode: str, ch_cols: list[int], n: int) -> tuple[int, int] | None:
    """Choose a rectangle column span that respects the current uneven channel spacing."""
    if mode == "parallel":
        j0 = random.randint(0, n - 2)
        c_lo, c_hi = ch_cols[j0], ch_cols[j0 + 1]
        base_width = c_hi - c_lo + 1
        max_extra = RECT_MAX_WIDTH - base_width
        if max_extra < 0:
            return None
        extra = random.randint(0, max_extra)
        margin_l = random.randint(0, extra)
        margin_r = extra - margin_l
    else:
        j0 = random.randint(0, n - 1)
        c_lo = c_hi = ch_cols[j0]
        extra = random.randint(1, min(4, RECT_MAX_WIDTH - 1))
        margin_l = random.randint(0, extra)
        margin_r = extra - margin_l
    return c_lo - margin_l, c_hi + margin_r


def _try_place_9rect(
    grid: np.ndarray,
    ch_cols: list[int],
    xover_rows: list[int],
    top_ep_row: int,
    bot_ep_row: int,
    n: int,
    placed_rects: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    """Attempt to place one color-9 rectangle that still behaves as wire pass-through."""
    h, w = grid.shape
    target_wires = random.choices([1, 2, 3], weights=RECT_WIRE_COUNT_WEIGHTS)[0]

    if target_wires >= 2 and random.random() < RECT_PARALLEL_PROB and n >= 2:
        mode = "parallel"
    elif target_wires >= 2 and xover_rows:
        mode = "crossing"
    else:
        mode = "single"

    span = _choose_rect_span("parallel" if mode == "parallel" else "single", ch_cols, n)
    if span is None:
        return None
    rect_c0, rect_c1 = span
    if rect_c0 < 0 or rect_c1 >= w or rect_c1 - rect_c0 + 1 > RECT_MAX_WIDTH:
        return None

    max_rect_h = min(RECT_H_RANGE[1], bot_ep_row - top_ep_row - 4)
    if max_rect_h < RECT_H_RANGE[0]:
        return None
    rect_h = random.randint(RECT_H_RANGE[0], max_rect_h)
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
    return rect_r0, rect_c0, rect_r1, rect_c1


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
    """Draw the wire network and return channel occupancy after all crossover moves."""
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
            if c not in col_seg_start:
                continue
            start = col_seg_start[c] + 1
            if start <= xrow - 1:
                grid[start:xrow, c] = 8

        c_min, c_max = min(src_col, dst_col), max(src_col, dst_col)
        grid[xrow, c_min : c_max + 1] = 8

        for c in active_cols:
            grid[xrow, c] = 8

        for c in active_cols:
            if c in (src_col, dst_col):
                continue
            if c_min < c < c_max:
                grid[xrow, c - 1] = 5
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
        if slot_wire[j] is None:
            continue
        c = ch_cols[j]
        if c not in col_seg_start:
            continue
        start = col_seg_start[c] + 1
        if start <= bot_ep_row:
            grid[start : bot_ep_row + 1, c] = 8

    return slot_wire, bypass_wire


def _derive_input(
    grid: np.ndarray,
    n: int,
    ch_cols: list[int],
    perm: list[int],
    token_colors: list[int],
    top_token_row: int,
    bot_token_row: int,
) -> np.ndarray:
    """Place the source-side solid/X tokens onto the wire-only grid."""
    inp = grid.copy()
    n_x = random.choices([0, 1, 2], weights=X_INPUT_COUNT_WEIGHTS)[0]
    x_channels = set(random.sample(range(n), min(n_x, n)))

    for i in range(n):
        col = token_colors[i]
        top_center = (top_token_row, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[perm[i]])

        if i in x_channels:
            if random.choice([True, False]):
                near, far = top_center, bot_center
            else:
                near, far = bot_center, top_center
            _stamp_token(inp, near[0], near[1], col, _X)
            if random.random() < FAR_DIAMOND_PROB:
                _stamp_token(inp, far[0], far[1], col, _DIAMOND)
        else:
            side = random.choice([top_center, bot_center])
            _stamp_token(inp, side[0], side[1], col, _SOLID)
    return inp


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(CHANNEL_COUNTS)
    ch_cols, gaps, bypass_col, w = _build_channel_columns(n)
    if len(set(gaps)) != len(gaps) or any(gaps[i] >= gaps[i + 1] for i in range(len(gaps) - 1)):
        return None
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

    output = grid.copy()
    for i in range(n):
        _stamp_token(output, top_token_row, ch_cols[i], token_colors[i], _DIAMOND)
        _stamp_token(output, bot_token_row, ch_cols[perm[i]], token_colors[i], _X)

    inp = _derive_input(grid, n, ch_cols, perm, token_colors, top_token_row, bot_token_row)
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
