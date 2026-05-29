# Variant of a47bf94d
# Dimension: D7 — 传输目标与停止条件 (dst)
# Change: 管道对端端点（pipeline far endpoint）——token的互补形状传输到同一管道的另一端点，距端点2格处... → 距离编码停止

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

# Design analysis
# ---------------
# concrete_behavior:
#   kept_from_source:
#   - token vocabulary is unchanged: solid 3x3, X, and diamond.
#   - the wire network still consists of 8-cells, 5-marked crossings, and optional 9-rect pass-through blocks.
#   - source tokens still sit two cells outside a degree-1 main endpoint.
#   changed_for_D7:
#   - the complementary token no longer goes to the far main endpoint.
#   - each active source endpoint now has a prebuilt 1-cell side spur attached to the main path.
#   - the solver counts colored pixels in the source token (solid=9, X=5), walks exactly that many wire steps from
#     the source endpoint, and then uses the side spur attached at that stop cell as the delivery dock.
#   - the emitted token is placed exactly as in the original endpoint rule: its center is 2 cells beyond that spur endpoint.
# behavior_trace:
# - original: a top solid at channel i walked all the way through the permutation network and emitted an X at the opposite
#   main endpoint.
# - variant: the same solid still becomes a diamond in place, but its 9 colored cells encode "walk 9 wire cells from the
#   source endpoint"; the X appears at the spur dock attached there instead of at the far end.
# - difference: the routing path still matters, but the stopping target is now an intermediate distance-coded dock.
# complexity_calibration:
# - only two travel distances are used, 5 and 9, because they are directly countable from the visible source tokens and
#   repeat multiple times per sample.
# - every generated sample contains both X and solid inputs so the two distance classes can be compared inside one grid.
# - permutations are restricted to modest crossover counts so the network remains readable and the 5-vs-9 stop behavior
#   is visually trackable.
# implicit_assumptions_audited:
# - the original solver assumed "trace until far endpoint, then place token 2 cells beyond it". That no longer holds.
# - endpoint scanning itself remains valid, but once a nearby token is found we must stop after a bounded prefix walk and
#   then locate the unique spur endpoint adjacent to that stop cell.
# - branch cells must be degree-1 leaves; otherwise endpoint detection and docking become ambiguous.
# generator_legality:
# - spur endpoints are only placed where the stop cell lies on a straight path segment, so the main walker never diverts
#   into the spur while counting steps.
# - 9-rects are forbidden from overlapping any used 9-step prefix, stop cell, spur, or dock token area, keeping the
#   distance count visually legible and preserving the planned route.
# observability_and_cleanliness:
# - the new behavior is visible because emitted tokens sit on short side docks instead of far endpoints.
# - using actual token pixel counts (5 and 9) avoids hidden color/id logic.
# - far-end preplaced diamonds are removed; otherwise they would blur whether the token truly stopped early.

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID_OFF = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]

_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DIR2FACE = {(1, 0): "N", (-1, 0): "S", (0, 1): "W", (0, -1): "E"}
_FACE2DIR = {"S": (1, 0), "N": (-1, 0), "E": (0, 1), "W": (0, -1)}

_KIND_TO_STEPS = {"solid": len(_SOLID_OFF), "x": len(_X_OFF)}
_KIND_TO_SOURCE = {"solid": _SOLID_OFF, "x": _X_OFF}
_KIND_TO_EMITTED = {"solid": _X_OFF, "x": _DIAMOND_OFF}


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
            if not maximal:
                continue
            found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _find_x_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for isolated 3x3 'X' patterns."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}

    mask = np.zeros((3, 3), dtype=bool)
    for rr, cc in [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]:
        mask[rr, cc] = True

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0:r0 + 3, c0:c0 + 3]
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


def _step_wire_once(
    comp: set[tuple[int, int]],
    cur: tuple[int, int],
    d: tuple[int, int],
    prev: tuple[int, int] | None,
    h: int,
    w: int,
    rects_9: list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int] | None]:
    """Advance one abstract wire step, honoring straight-through traversal inside color-9 rectangles."""
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
            return exit_cell, _FACE2DIR[exit_face], None

    r, c = cur
    opts = [((nr, nc), nr - r, nc - c) for nr, nc in neighbors4(r, c) if (nr, nc) in comp]
    if not opts:
        return cur, d, prev

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
        return cur, d, prev

    return p, (dr, dc), cur


def _walk_wire_cells(
    comp: set[tuple[int, int]],
    start: tuple[int, int],
    dir0: tuple[int, int],
    steps: int,
    h: int,
    w: int,
    rects_9: list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]],
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    path = [start]
    cur, d, prev = start, dir0, None
    for _ in range(steps):
        nxt, d2, prev2 = _step_wire_once(comp, cur, d, prev, h, w, rects_9)
        assert nxt != cur, "wire shorter than required distance code"
        path.append(nxt)
        cur, d, prev = nxt, d2, prev2
    return path, d


def _stamp_token(grid: np.ndarray, center_r: int, center_c: int, color: int, offsets: list[tuple[int, int]]) -> None:
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = color


def _deg_map(comp: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    deg = {p: 0 for p in comp}
    for r, c in comp:
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in comp:
                deg[(r, c)] += 1
    return deg


def _find_branch_endpoint(
    comp: set[tuple[int, int]],
    deg: dict[tuple[int, int], int],
    stop: tuple[int, int],
    d: tuple[int, int],
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """Return the unique degree-1 side spur attached perpendicular to the stop cell."""
    perps = [(-d[1], d[0]), (d[1], -d[0])]
    found = []
    for dr, dc in perps:
        p = (stop[0] + dr, stop[1] + dc)
        if p in comp and deg.get(p, 0) <= 1:
            found.append((p, (dr, dc)))
    if len(found) == 1:
        return found[0]
    return None, None


def _trace_wire(
    comp: set[tuple[int, int]],
    start: tuple[int, int],
    dir0: tuple[int, int],
    h: int,
    w: int,
    rects_9: list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    cur, d = start, dir0
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 4):
        if (cur, d) in visited:
            assert False, "wire trace looped"
        visited.add((cur, d))
        nxt, d2, _prev2 = _step_wire_once(comp, cur, d, None, h, w, rects_9)
        if nxt == cur:
            return cur, d
        cur, d = nxt, d2
    assert False, "wire trace exceeded step limit"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transmit colored tokens along wire networks, but stop at a distance-coded side dock instead of the far endpoint.

    Input structure:
    - Background cells surround one orthogonal wire network made of 8-cells, 5 crossing markers, and optional 9-rect pass-through blocks.
    - Colored 3x3 tokens (solid blocks or X patterns) sit two cells outside selected main wire endpoints, and one-cell side spurs may appear along the route as empty delivery docks.

    Rule:
    - Every solid 3x3 token collapses in place into a diamond.
    - Each active token emits its complementary shape, but the destination is chosen by distance: count the token's colored cells (solid=9, X=5), walk that many wire steps from the source endpoint, then place the emitted shape at the side spur attached there.

    Key insight:
    - The wire still determines the route, but the stop is no longer "the far end"; it is the intermediate dock located at a path-length encoded by the visible source token size.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)

    track_mask = g != bg
    for (_cr, _cc), (_col, (r0, c0)) in solids.items():
        track_mask[r0:r0 + 3, c0:c0 + 3] = False
    for cr, cc in xs.keys():
        for dr, dc in _X_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                track_mask[rr, cc2] = False

    comps = _components(track_mask)
    rects_9 = _find_9rects(g)

    out = g.copy()

    for (cr, cc), (col, (r0, c0)) in solids.items():
        out[r0:r0 + 3, c0:c0 + 3] = bg
        for dr, dc in _DIAMOND_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                out[rr, cc2] = col

    for pts in comps:
        comp = set(pts)
        if len(comp) < 4:
            continue

        comp_colors = {int(g[r, c]) for r, c in comp}
        deg = _deg_map(comp)
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
                    path, d = _walk_wire_cells(comp, ep, dir_in, _KIND_TO_STEPS["solid"], h, w, rects_9)
                    dock_ep, dock_dir = _find_branch_endpoint(comp, deg, path[-1], d)
                    if dock_ep is None or dock_dir is None:
                        continue
                    dest = (dock_ep[0] + 2 * dock_dir[0], dock_ep[1] + 2 * dock_dir[1])
                    for dr, dc in _X_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col

                elif cen in xs:
                    col, _ = xs[cen]
                    if col in comp_colors:
                        continue
                    dir_in = (-odr, -odc)
                    path, d = _walk_wire_cells(comp, ep, dir_in, _KIND_TO_STEPS["x"], h, w, rects_9)
                    dock_ep, dock_dir = _find_branch_endpoint(comp, deg, path[-1], d)
                    if dock_ep is None or dock_dir is None:
                        continue
                    dest = (dock_ep[0] + 2 * dock_dir[0], dock_ep[1] + 2 * dock_dir[1])
                    for dr, dc in _DIAMOND_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col

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
GRID_W_MAX = 30
EP_TOKEN_GAP = 2

RECT_COUNT_WEIGHTS = [20, 45, 25, 10]
RECT_WIRE_COUNT_WEIGHTS = [50, 38, 12]
RECT_PARALLEL_PROB = 0.75
RECT_H_RANGE = (2, 5)
RECT_MAX_WIDTH = 7
RECT_RETRY_FACTOR = 5
RECT_MARGIN_GAP = 1

GEN_MAX_RETRIES = 800
MAX_CROSSOVERS = 3


def _decompose_to_crossovers(perm: list[int]) -> list[tuple[int, int]]:
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


def _try_place_9rect(
    grid: np.ndarray,
    ch_cols: list[int],
    xover_rows: list[int],
    top_ep_row: int,
    bot_ep_row: int,
    n_channels: int,
    placed_rects: list[tuple[int, int, int, int]],
    forbidden: set[tuple[int, int]],
) -> tuple[int, int, int, int] | None:
    """Attempt to place one color-9 rectangular block away from reserved routing/dock cells."""
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
        if not np.all(grid[rect_r0:rect_r1 + 1, c] == 8):
            return None

    sub = grid[rect_r0:rect_r1 + 1, rect_c0:rect_c1 + 1]
    if not np.all((sub == 0) | (sub == 5) | (sub == 8)):
        return None

    for rr in range(rect_r0, rect_r1 + 1):
        for cc in range(rect_c0, rect_c1 + 1):
            if (rr, cc) in forbidden:
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


def _choose_perm(n_channels: int) -> tuple[list[int], list[tuple[int, int]]] | None:
    for _ in range(150):
        perm = list(range(n_channels))
        random.shuffle(perm)
        if perm == list(range(n_channels)):
            continue
        crossovers = _decompose_to_crossovers(perm)
        if 1 <= len(crossovers) <= MAX_CROSSOVERS:
            return perm, crossovers
    return None


def _branch_candidates(d: tuple[int, int]) -> list[tuple[int, int]]:
    perps = [(-d[1], d[0]), (d[1], -d[0])]
    random.shuffle(perps)
    return perps


def _source_choices(
    i: int,
    perm: list[int],
    ch_cols: list[int],
    top_token_row: int,
    top_ep_row: int,
    bot_token_row: int,
    bot_ep_row: int,
) -> list[tuple[str, tuple[int, int], tuple[int, int], tuple[int, int]]]:
    choices = [
        ("top", (top_ep_row, ch_cols[i]), (top_token_row, ch_cols[i]), (1, 0)),
        ("bottom", (bot_ep_row, ch_cols[perm[i]]), (bot_token_row, ch_cols[perm[i]]), (-1, 0)),
    ]
    # Alternate the preferred source side so docks are not all forced onto the same half first.
    if i % 2 == 1:
        choices.reverse()
    return choices


def _pick_branch_plan(
    grid: np.ndarray,
    comp: set[tuple[int, int]],
    stop: tuple[int, int],
    d: tuple[int, int],
    emitted_offsets: list[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], set[tuple[int, int]]] | None:
    h, w = grid.shape
    prev_cell = (stop[0] - d[0], stop[1] - d[1])
    next_cell = (stop[0] + d[0], stop[1] + d[1])
    if prev_cell not in comp or next_cell not in comp:
        return None

    for br in _branch_candidates(d):
        ep = (stop[0] + br[0], stop[1] + br[1])
        cen = (ep[0] + 2 * br[0], ep[1] + 2 * br[1])
        if not (0 <= ep[0] < h and 0 <= ep[1] < w and 0 <= cen[0] < h and 0 <= cen[1] < w):
            continue
        if int(grid[ep]) != 0 or ep in reserved:
            continue

        nbs = [(ep[0] + dr, ep[1] + dc) for dr, dc in _N4 if 0 <= ep[0] + dr < h and 0 <= ep[1] + dc < w]
        occupied_nbs = [p for p in nbs if p in comp or p in reserved]
        if occupied_nbs != [stop]:
            continue

        token_cells = {(cen[0] + dr, cen[1] + dc) for dr, dc in emitted_offsets}
        if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in token_cells):
            continue
        if any((rr, cc) in reserved for rr, cc in token_cells):
            continue
        if any(int(grid[rr, cc]) != 0 for rr, cc in token_cells):
            continue

        footprint = {ep, cen, stop}
        footprint |= token_cells
        return ep, br, cen, footprint
    return None


def _gen_attempt() -> np.ndarray | None:
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

    perm_and_cross = _choose_perm(n_channels)
    if perm_and_cross is None:
        return None
    perm, crossovers = perm_and_cross
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

    base = np.zeros((h, w), dtype=int)
    slot_wire, bypass_wire = _build_wire_network(base, n_channels, ch_cols, bypass_col, top_ep_row, bot_ep_row, crossovers, xover_rows)

    inv_perm = [0] * n_channels
    for i in range(n_channels):
        inv_perm[perm[i]] = i
    if any(slot_wire[j] != inv_perm[j] for j in range(n_channels)) or bypass_wire is not None:
        return None

    comp = set((int(r), int(c)) for r, c in np.argwhere(base != 0))

    n_x = random.choice([1, 2] if n_channels >= 4 else [1])
    x_channels = set(random.sample(range(n_channels), n_x))
    plans = []
    reserved: set[tuple[int, int]] = set()
    forbidden: set[tuple[int, int]] = set()

    # Place longer-distance solid channels first; they have fewer viable dock positions.
    channel_order = sorted(
        range(n_channels),
        key=lambda i: (_KIND_TO_STEPS["x" if i in x_channels else "solid"], -i),
        reverse=True,
    )

    for i in channel_order:
        kind = "x" if i in x_channels else "solid"
        emitted_offsets = _KIND_TO_EMITTED[kind]
        selected = None

        for _src_side, src_ep, src_center, dir_in in _source_choices(
            i,
            perm,
            ch_cols,
            top_token_row,
            top_ep_row,
            bot_token_row,
            bot_ep_row,
        ):
            path, d = _walk_wire_cells(comp, src_ep, dir_in, _KIND_TO_STEPS[kind], h, w, [])
            branch = _pick_branch_plan(base, comp, path[-1], d, emitted_offsets, reserved)
            if branch is not None:
                selected = (src_center, path, branch)
                break

        if selected is None:
            return None
        src_center, path, branch = selected
        forbidden.update(path)
        branch_ep, branch_dir, dest_center, footprint = branch
        reserved |= footprint
        forbidden |= footprint
        plans.append(
            {
                "color": token_colors[i],
                "kind": kind,
                "src_center": src_center,
                "branch_ep": branch_ep,
                "branch_dir": branch_dir,
                "dest_center": dest_center,
            }
        )

    wire_grid = base.copy()
    n_rects = random.choices(range(4), weights=RECT_COUNT_WEIGHTS)[0]
    placed_rects: list[tuple[int, int, int, int]] = []
    for _ in range(n_rects * RECT_RETRY_FACTOR):
        if len(placed_rects) >= n_rects:
            break
        result = _try_place_9rect(
            wire_grid,
            ch_cols,
            xover_rows,
            top_ep_row,
            bot_ep_row,
            n_channels,
            placed_rects,
            forbidden,
        )
        if result is not None:
            placed_rects.append(result)

    for plan in plans:
        br, bc = plan["branch_ep"]
        if wire_grid[br, bc] != 0:
            return None
        wire_grid[br, bc] = 8

    inp = wire_grid.copy()
    out = wire_grid.copy()

    for plan in plans:
        color = plan["color"]
        kind = plan["kind"]
        src_r, src_c = plan["src_center"]
        dst_r, dst_c = plan["dest_center"]

        _stamp_token(inp, src_r, src_c, color, _KIND_TO_SOURCE[kind])

        if kind == "solid":
            _stamp_token(out, src_r, src_c, color, _DIAMOND_OFF)
            _stamp_token(out, dst_r, dst_c, color, _X_OFF)
        else:
            _stamp_token(out, src_r, src_c, color, _X_OFF)
            _stamp_token(out, dst_r, dst_c, color, _DIAMOND_OFF)

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
