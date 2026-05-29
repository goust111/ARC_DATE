# Variant of a47bf94d
# Dimension: D13 — 多步处理顺序 (multi_step)
# Change: 单步变换——每个token独立执行一次分解和传输操作，各通道之间没有顺序依赖... → 按管道长度排序处理

"""
Design analysis for the D13 variant.

a) Concrete behavior and trace
- Keep the original visual vocabulary: 8/5/9 wires, solid 3x3 tokens, X tokens, and diamonds.
- The new visible effect is created by making one pair of channel centers only 2 columns apart.
  Two transmitted X-shapes aimed at that close pair overlap in exactly two cells.
- Old logic: both emissions are independent writes, so the overlap cells are effectively order-free.
- New logic: collect all endpoint tokens first, sort them by traced wire length, then execute them
  one by one. Emitted cells only fill current background, so the shorter route claims the shared
  overlap cells and the longer route can only paint its remaining unique cells.

b) Complexity calibration
- The only new freedom is the processing order induced by route length.
- Conflict geometry is fixed to one close destination pair, which keeps the pattern readable.
- Token colors remain unique per channel so overlap ownership is visually traceable.

c) Hidden-assumption audit
- Original solver painted every emitted shape unconditionally; that was safe because generated
  destinations did not overlap. In this variant that would erase the whole point of sequencing, so
  destination painting becomes background-only.
- Original solver converted all solids before routing anything. That relied on commutativity.
  Here each token must be fully executed in sorted order because earlier emissions may occupy cells
  that later tokens want.
- Original generator assumed uniform 4-5 spacing, hence far-end 3x3 zones never overlapped.
  This variant intentionally breaks that by inserting one 2-column gap.

d) Placement legality
- Input tokens are only placed on one side per instance, and source channels are chosen so their
  3x3 footprints never overlap even though one pair of channel centers is close.
- The close pair is used as a destination pair, not a simultaneous source pair.
- The existing oracle `main(generate_input())` is still used to validate each sample.

e) Visual observability
- The ordering key is visible: shorter pipes have fewer bends / detours.
- The effect is visible: in the overlap zone, the shorter token owns the shared cells while the
  longer token appears truncated only at those cells.
- Samples are rejected unless there is at least one overlap pair with unequal route lengths.

f) Visual cleanliness
- Only one overlap hotspot is forced per grid, so the result stays legible.
- Other channels remain spaced far enough apart to preserve the original neat wire layout.

g) Color inference
- No hidden numeric property of colors is used. Colors only identify which token reached which
  cells first. The ordering is driven exclusively by pipe geometry.
"""

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]

_DIAMOND = _DIAMOND_OFF
_X = _X_OFF
_SOLID = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
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
            if maximal:
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
    Follow a wire greedily and return (terminal_cell, outward_direction, route_length).

    Route length is counted in traversed cell-to-cell steps; a jump across a 9-rectangle uses
    Manhattan distance to the chosen exit wire cell so ordering still reflects visible detours.
    """
    cur, d, prev = start, dir0, None
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    steps = 0
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
                    if len(others) != 1:
                        raise AssertionError("ambiguous 9-rectangle exit")
                    exit_face = others[0]
                pts = faces9[exit_face]
                er, ec = cur
                if entry_face in ("N", "S"):
                    exit_cell = min(pts, key=lambda p: abs(p[1] - ec))
                else:
                    exit_cell = min(pts, key=lambda p: abs(p[0] - er))
                steps += abs(exit_cell[0] - cur[0]) + abs(exit_cell[1] - cur[1])
                cur, d, prev = exit_cell, _FACE2DIR[exit_face], None
                jumped = True
                break
        if jumped:
            continue

        r, c = cur
        opts = [((nr, nc), nr - r, nc - c) for nr, nc in neighbors4(r, c) if (nr, nc) in comp]
        if not opts:
            return cur, d, steps

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
            return cur, d, steps

        prev = cur
        cur = p
        d = (dr, dc)
        steps += 1

    raise AssertionError("wire trace exceeded step limit")


def _shape_cells(center: tuple[int, int], offsets: list[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    cells = set()
    for dr, dc in offsets:
        rr, cc = center[0] + dr, center[1] + dc
        if 0 <= rr < h and 0 <= cc < w:
            cells.add((rr, cc))
    return cells


def _stamp_token(grid: np.ndarray, center_r: int, center_c: int, color: int, offsets: list[tuple[int, int]]) -> None:
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = color


def _stamp_if_bg(
    grid: np.ndarray,
    center: tuple[int, int],
    color: int,
    offsets: list[tuple[int, int]],
    bg: int,
) -> None:
    h, w = grid.shape
    for dr, dc in offsets:
        rr, cc = center[0] + dr, center[1] + dc
        if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == bg:
            grid[rr, cc] = color


def _build_track_mask(g: np.ndarray, bg: int, solids: dict, xs: dict) -> np.ndarray:
    track_mask = g != bg
    for _center, (_col, (r0, c0)) in solids.items():
        track_mask[r0 : r0 + 3, c0 : c0 + 3] = False
    for cr, cc in xs.keys():
        for dr, dc in _X_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < g.shape[0] and 0 <= cc2 < g.shape[1]:
                track_mask[rr, cc2] = False
    return track_mask


def _collect_tokens(
    g: np.ndarray,
    solids: dict[tuple[int, int], tuple[int, tuple[int, int]]],
    xs: dict[tuple[int, int], tuple[int, tuple[int, int]]],
    comps: list[list[tuple[int, int]]],
    rects_9: list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]],
) -> list[dict]:
    h, w = g.shape
    tokens: list[dict] = []
    seen: set[tuple[tuple[int, int], str]] = set()

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
        endpoints = sorted(p for p in comp if deg[p] <= 1)
        if not endpoints:
            continue

        for ep in endpoints:
            er, ec = ep
            for odr, odc in _N4:
                cen = (er + 2 * odr, ec + 2 * odc)
                if not (0 <= cen[0] < h and 0 <= cen[1] < w):
                    continue

                if cen in solids:
                    col, (r0, c0) = solids[cen]
                    kind = "solid"
                elif cen in xs:
                    col, (r0, c0) = xs[cen]
                    kind = "x"
                else:
                    continue

                if col in comp_colors or (cen, kind) in seen:
                    continue

                dir_in = (-odr, -odc)
                end, dd, route_len = _trace_wire(comp, ep, dir_in, h, w, rects_9)
                dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                tokens.append(
                    {
                        "kind": kind,
                        "color": col,
                        "src_center": cen,
                        "src_bbox": (r0, c0),
                        "dest_center": dest,
                        "route_len": route_len,
                        "emit_offsets": _X_OFF if kind == "solid" else _DIAMOND_OFF,
                    }
                )
                seen.add((cen, kind))

    tokens.sort(key=lambda t: (t["route_len"], t["src_center"][0], t["src_center"][1], 0 if t["kind"] == "solid" else 1))
    return tokens


def _route_info_for_vertical_layout(
    grid: np.ndarray,
    ch_cols: list[int],
    top_ep_row: int,
    rects_9: list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]],
) -> dict[int, dict]:
    h, w = grid.shape
    comps = _components(grid != 0)
    comp = set(max(comps, key=len))
    info: dict[int, dict] = {}
    for i, c in enumerate(ch_cols):
        end, dd, route_len = _trace_wire(comp, (top_ep_row, c), (1, 0), h, w, rects_9)
        info[i] = {"route_len": route_len, "dest_center": (end[0] + 2 * dd[0], end[1] + 2 * dd[1])}
    return info


def _analyze_variant_case(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)
    track_mask = _build_track_mask(g, bg, solids, xs)
    comps = _components(track_mask)
    rects_9 = _find_9rects(g)
    tokens = _collect_tokens(g, solids, xs, comps, rects_9)
    conflict_pairs = 0
    ordered_conflicts = 0
    max_overlap = 0
    for i in range(len(tokens)):
        cells_i = _shape_cells(tokens[i]["dest_center"], tokens[i]["emit_offsets"], g.shape)
        for j in range(i + 1, len(tokens)):
            cells_j = _shape_cells(tokens[j]["dest_center"], tokens[j]["emit_offsets"], g.shape)
            ov = cells_i & cells_j
            if ov:
                conflict_pairs += 1
                max_overlap = max(max_overlap, len(ov))
                if tokens[i]["route_len"] != tokens[j]["route_len"]:
                    ordered_conflicts += 1
    return {
        "token_count": len(tokens),
        "conflict_pairs": conflict_pairs,
        "ordered_conflicts": ordered_conflicts,
        "max_overlap": max_overlap,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route endpoint tokens through the wire network, but execute them in travel-time order.

    Input structure:
    - Background cells are 0. Non-background cells form cyan wire networks with 5/9 routing
      markers, plus endpoint tokens shaped as solid 3x3 blocks or X-patterns.

    Rule:
    - First detect every endpoint token and compute the length of its pipe path to the far end.
    - Then process tokens from shorter path to longer path. A solid token turns into a local
      diamond and emits an X; an X token stays in place and emits a diamond. Emitted cells only
      fill background, so earlier arrivals can block later ones where targets overlap.

    Key insight:
    - The original per-token transformation is no longer commutative. The decisive information is
      which token reaches a contested destination first, and that is determined by wire length.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)
    track_mask = _build_track_mask(g, bg, solids, xs)
    comps = _components(track_mask)
    rects_9 = _find_9rects(g)
    tokens = _collect_tokens(g, solids, xs, comps, rects_9)

    out = g.copy()
    for token in tokens:
        if token["kind"] == "solid":
            r0, c0 = token["src_bbox"]
            out[r0 : r0 + 3, c0 : c0 + 3] = bg
            _stamp_token(out, token["src_center"][0], token["src_center"][1], token["color"], _DIAMOND_OFF)
        _stamp_if_bg(out, token["dest_center"], token["color"], token["emit_offsets"], bg)
    return out


# --------------- generator (length_order_conflict) ---------------

CHANNEL_COUNTS = [4, 5]
NORMAL_GAP_OPTIONS = [4, 5]
CONFLICT_GAP = 2
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

EXTRA_X_PROB = 0.25
GEN_MAX_RETRIES = 500


def _decompose_to_crossovers(perm: list[int]) -> list[tuple[int, int]]:
    """Decompose a permutation into sequential crossover moves using one bypass column."""
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


def _choose_channel_layout(n: int) -> tuple[list[int], int, int, int] | None:
    tries = 40
    interior_choices = list(range(1, n - 2)) if n >= 5 else [1]
    for _ in range(tries):
        gaps = [random.choice(NORMAL_GAP_OPTIONS) for _ in range(n - 1)]
        conflict_pair = random.choice(interior_choices)
        gaps[conflict_pair] = CONFLICT_GAP
        first_col = random.randint(*LEFT_MARGIN_RANGE) + 1
        ch_cols = [first_col]
        for gap in gaps:
            ch_cols.append(ch_cols[-1] + gap)
        bypass_col = ch_cols[-1] + BYPASS_COL_OFFSET
        width = max(bypass_col + 3, GRID_W_MIN)
        if width <= GRID_W_MAX:
            return ch_cols, bypass_col, conflict_pair, width
    return None


def _try_place_9rect(
    grid: np.ndarray,
    ch_cols: list[int],
    xover_rows: list[int],
    top_ep_row: int,
    bot_ep_row: int,
    n: int,
    placed_rects: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
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


def _nonoverlapping_source_channels(channels: set[int], ch_cols: list[int]) -> bool:
    cols = sorted(ch_cols[i] for i in channels)
    return all(b - a >= 3 for a, b in zip(cols, cols[1:]))


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(CHANNEL_COUNTS)
    layout = _choose_channel_layout(n)
    if layout is None:
        return None
    ch_cols, bypass_col, conflict_pair, width = layout

    top_token_row = random.choice(TOKEN_ROW_OPTIONS)
    top_ep_row = top_token_row + EP_TOKEN_GAP
    token_colors = random.sample(TOKEN_COLORS_POOL, n)

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
    height = random.randint(max(GRID_H_RANGE[0], h_min), GRID_H_RANGE[1])
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
    slot_wire, bypass_wire = _build_wire_network(grid, n, ch_cols, bypass_col, top_ep_row, bot_ep_row, crossovers, xover_rows)

    inv_perm = [0] * n
    for i in range(n):
        inv_perm[perm[i]] = i
    if any(slot_wire[j] != inv_perm[j] for j in range(n)) or bypass_wire is not None:
        return None

    n_rects = random.choices(range(4), weights=RECT_COUNT_WEIGHTS)[0]
    placed_rects: list[tuple[int, int, int, int]] = []
    for _ in range(n_rects * RECT_RETRY_FACTOR):
        if len(placed_rects) >= n_rects:
            break
        rect = _try_place_9rect(grid, ch_cols, xover_rows, top_ep_row, bot_ep_row, n, placed_rects)
        if rect is not None:
            placed_rects.append(rect)

    rects_9 = _find_9rects(grid)
    top_info = _route_info_for_vertical_layout(grid, ch_cols, top_ep_row, rects_9)

    dst_a, dst_b = conflict_pair, conflict_pair + 1
    src_a, src_b = inv_perm[dst_a], inv_perm[dst_b]
    if src_a == src_b or not _nonoverlapping_source_channels({src_a, src_b}, ch_cols):
        return None
    info_a, info_b = top_info[src_a], top_info[src_b]
    if info_a["dest_center"] != (bot_token_row, ch_cols[dst_a]) or info_b["dest_center"] != (bot_token_row, ch_cols[dst_b]):
        return None
    if info_a["route_len"] == info_b["route_len"]:
        return None

    top_channels = {src_a, src_b}
    extras = [i for i in range(n) if i not in top_channels]
    random.shuffle(extras)
    extra_target = random.randint(1, max(1, min(2, n - 2)))
    for i in extras:
        dest_col = ch_cols[perm[i]]
        if abs(dest_col - ch_cols[dst_a]) < 3 or abs(dest_col - ch_cols[dst_b]) < 3:
            continue
        trial = set(top_channels)
        trial.add(i)
        if _nonoverlapping_source_channels(trial, ch_cols):
            top_channels = trial
        if len(top_channels) >= 2 + extra_target:
            break
    if not _nonoverlapping_source_channels(top_channels, ch_cols):
        return None

    x_channel = None
    x_candidates = [i for i in top_channels if i not in (src_a, src_b)]
    if x_candidates and random.random() < EXTRA_X_PROB:
        x_channel = random.choice(x_candidates)

    inp = grid.copy()
    for i in sorted(top_channels):
        offsets = _X if i == x_channel else _SOLID
        _stamp_token(inp, top_token_row, ch_cols[i], token_colors[i], offsets)

    out = main(inp)
    overlap = _shape_cells((bot_token_row, ch_cols[dst_a]), _X_OFF, inp.shape) & _shape_cells((bot_token_row, ch_cols[dst_b]), _X_OFF, inp.shape)
    if len(overlap) < 1:
        return None

    shorter = src_a if info_a["route_len"] < info_b["route_len"] else src_b
    longer = src_b if shorter == src_a else src_a
    shorter_color = token_colors[shorter]
    longer_color = token_colors[longer]
    if not overlap or any(int(out[r, c]) != shorter_color for r, c in overlap):
        return None
    longer_unique = _shape_cells((bot_token_row, ch_cols[perm[longer]]), _X_OFF, inp.shape) - overlap
    if not any(int(out[r, c]) == longer_color for r, c in longer_unique):
        return None

    metrics = _analyze_variant_case(inp)
    if metrics["ordered_conflicts"] < 1 or metrics["max_overlap"] < 1:
        return None

    if random.random() < 0.5:
        inp = np.flipud(inp)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        try:
            result = _gen_attempt()
        except Exception:
            result = None
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
