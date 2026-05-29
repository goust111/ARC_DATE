# Variant of a47bf94d
# Dimension: D8 — 9号矩形块的角色 (obstacle)
# Change: 直通连接节点——9号颜色的矩形块（2x2到5x4）嵌入管道中，管道从入口面直线穿过到对面出口面（仅当单入单出在相邻面时才转弯）→ 阻断/过滤节点

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

# Design analysis for the D8 variant
# a) Concrete behavior:
#    The visible objects stay the same: colored 3x3 solids or X-tokens sit two cells away from
#    wire endpoints, teal/grey wires route colors across the grid, and maroon rectangles remain
#    embedded inside those wires. The change is only the role of the maroon rectangle.
#    Original behavior at a routed token:
#    - a token reaches a 9-rectangle and is always forwarded through it, so the far endpoint
#      receives the complementary shape.
#    Variant behavior:
#    - the token still starts from the same endpoint and follows the same wire geometry, but the
#      first odd-area 9-rectangle on that route absorbs it.
#    - even-area 9-rectangles still forward the token using the original straight/turn rule.
#    Example: a solid token on the top of one channel becomes a local diamond either way.
#    If its route contains a 2x3 maroon rectangle (area 6), the bottom endpoint still receives
#    an X. If the route instead contains a 3x3 maroon rectangle (area 9), the bottom endpoint
#    stays empty. That is the only semantic difference.
# b) Complexity calibration:
#    The variant must be visually inferable from 2-4 examples, so the generator cannot allow
#    "no 9-rect" or "all channels behave the same" grids to dominate. Each generated sample is
#    forced to include both an odd rectangle that blocks at least one route and an even rectangle
#    that allows at least one route to pass through. This keeps the rule compact: only one binary
#    predicate (area parity) controls the new behavior.
# c) Hidden assumptions audit:
#    - Original _trace_wire assumed every entered 9-rectangle must yield an exit face. That is no
#      longer true, so tracing now returns a blocked flag.
#    - Original main unconditionally drew the far complementary token after tracing. In the variant
#      that must only happen when tracing is not blocked.
#    - Original generator sometimes pre-placed the far diamond for X-input channels. That is still
#      harmless for passable routes, but it would fake a successful transmission on blocked routes,
#      so blocked X-channels are never given a pre-placed far diamond.
# d) Placement legality:
#    Rectangles still obey the old geometric constraints: axis-aligned, non-overlapping, embedded
#    inside existing wire cells, and separated by a 1-cell margin. After placement, the generator
#    traces every channel route through the wire component to confirm the expected endpoints and to
#    label which rectangles each channel actually traverses.
# e) Visual observability:
#    The rule depends only on visible width/height, not on hidden ids or ordering. A human can
#    inspect maroon rectangles and compare examples where a 2x2 / 2x3 rectangle still lets the far
#    complement appear while a 3x3 / 3x1 rectangle leaves the far side blank.
# f) Visual neatness:
#    No extra markers are added. The only additional effect is absence/presence of the far emitted
#    token, which occurs in the already reserved 3x3 endpoint zones, so the grids stay readable.
# g) Color inference:
#    Colors keep their original identity-preserving role. The new rule does not map one color to
#    another, avoiding any need for hidden color semantics.

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]

_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DIR2FACE = {(1, 0): "N", (-1, 0): "S", (0, 1): "W", (0, -1): "E"}
_FACE2DIR = {"S": (1, 0), "N": (-1, 0), "E": (0, 1), "W": (0, -1)}


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


def _find_9rects(g: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    """Find color-9 rectangles, their face connections, and area parity."""
    mask9 = g == 9
    if not mask9.any():
        return [], {}

    h, w = g.shape
    rects: list[dict] = []
    cell_to_rect: dict[tuple[int, int], int] = {}
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
        rect = {
            "cells": cells,
            "faces": faces,
            "bbox": (r0, c0, r1, c1),
            "area": (r1 - r0 + 1) * (c1 - c0 + 1),
        }
        rect["passable"] = rect["area"] % 2 == 0
        idx = len(rects)
        rects.append(rect)
        for cell in cells:
            cell_to_rect[cell] = idx
    return rects, cell_to_rect


def _rect_exit(rect: dict, cur: tuple[int, int], d: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return the exit cell and direction for a passable 9-rectangle."""
    entry_face = _DIR2FACE[d]
    opp = _OPP[entry_face]
    faces = rect["faces"]
    if opp in faces:
        exit_face = opp
    else:
        others = [face for face in faces if face != entry_face]
        assert len(others) == 1
        exit_face = others[0]

    pts = faces[exit_face]
    er, ec = cur
    if entry_face in ("N", "S"):
        exit_cell = min(pts, key=lambda p: abs(p[1] - ec))
    else:
        exit_cell = min(pts, key=lambda p: abs(p[0] - er))
    return exit_cell, _FACE2DIR[exit_face]


def _trace_wire(
    comp: set[tuple[int, int]],
    start: tuple[int, int],
    dir0: tuple[int, int],
    h: int,
    w: int,
    rects_9: list[dict],
    rect_lookup: dict[tuple[int, int], int],
    *,
    block_on_odd: bool,
) -> tuple[tuple[int, int], tuple[int, int], bool, list[int]]:
    """
    Greedily follow one wire path, optionally stopping when an odd-area 9-rect is entered.

    Returns (end_cell, final_dir, blocked, rect_ids_hit_in_order).
    """
    cur, d, prev = start, dir0, None
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    hit_rects: list[int] = []
    seen_rects: set[int] = set()

    for _ in range(h * w * 4):
        if (cur, d) in visited:
            assert False, "wire trace looped — should not happen"
        visited.add((cur, d))

        rect_id = rect_lookup.get(cur)
        if rect_id is not None:
            if rect_id not in seen_rects:
                seen_rects.add(rect_id)
                hit_rects.append(rect_id)
            rect = rects_9[rect_id]
            if block_on_odd and not rect["passable"]:
                return cur, d, True, hit_rects
            cur, d = _rect_exit(rect, cur, d)
            prev = None
            continue

        r, c = cur
        opts = [((nr, nc), nr - r, nc - c) for nr, nc in neighbors4(r, c) if (nr, nc) in comp]
        if not opts:
            return cur, d, False, hit_rects

        rev = (-d[0], -d[1])
        nonrev = [opt for opt in opts if (opt[1], opt[2]) != rev]
        opts2 = nonrev if nonrev else opts

        straight = [opt for opt in opts2 if (opt[1], opt[2]) == d]
        if straight:
            nxt, dr, dc = straight[0]
        else:
            perp = [opt for opt in opts2 if opt[1] * d[0] + opt[2] * d[1] == 0]
            if perp:
                nxt, dr, dc = sorted(perp, key=lambda x: (x[1], x[2]))[0]
            else:
                nxt, dr, dc = opts2[0]

        if nxt == prev:
            return cur, d, False, hit_rects

        prev = cur
        cur = nxt
        d = (dr, dc)

    assert False, "wire trace exceeded step limit"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route endpoint tokens across the wire network while maroon rectangles act as parity gates.

    Input structure:
    - Background cells are 0. Non-background cells form one orthogonal wire network plus 3x3
      colored tokens near degree-1 endpoints. Wire cells use 8/5, and filled color-9 rectangles
      are embedded inside the wires.

    Rule:
    - A solid 3x3 token still collapses in place to a diamond.
    - Each endpoint token then tries to send its complementary shape to the far endpoint.
    - When the route enters a color-9 rectangle, the rectangle's visible area decides the outcome:
      even area lets the token continue through the rectangle using the original wire geometry,
      while odd area blocks the transmission and no far complementary token is created.

    Key insight:
    - The maroon rectangle is no longer just a connector. Its width-by-height parity is the gate,
      so successful transmissions are exactly the routes that avoid odd-area rectangles.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)

    track_mask = g != bg
    for _center, (_col, (r0, c0)) in solids.items():
        track_mask[r0 : r0 + 3, c0 : c0 + 3] = False
    for cr, cc in xs.keys():
        for dr, dc in _X_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                track_mask[rr, cc2] = False

    comps = _components(track_mask)
    rects_9, rect_lookup = _find_9rects(g)

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
                    end, dd, blocked, _ = _trace_wire(
                        comp, ep, dir_in, h, w, rects_9, rect_lookup, block_on_odd=True
                    )
                    if blocked:
                        continue
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
                    end, dd, blocked, _ = _trace_wire(
                        comp, ep, dir_in, h, w, rects_9, rect_lookup, block_on_odd=True
                    )
                    if blocked:
                        continue
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    for dr, dc in _DIAMOND_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col

    return out


# --- Token shape offsets (relative to 3x3 center) ---
_DIAMOND = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]

# --- Layout configuration ---
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

# --- 9-rect placement ---
RECT_COUNT_CHOICES = [2, 3]
RECT_COUNT_WEIGHTS = [65, 35]
RECT_WIRE_COUNT_WEIGHTS = [50, 38, 12]
RECT_PARALLEL_PROB = 0.75
RECT_H_RANGE = (2, 5)
RECT_MAX_WIDTH = 7
RECT_RETRY_FACTOR = 9
RECT_MARGIN_GAP = 1

# --- Input derivation ---
X_INPUT_COUNT_WEIGHTS = [60, 30, 10]
FAR_DIAMOND_PROB = 0.33

# --- Generation retry ---
GEN_MAX_RETRIES = 500


def _decompose_to_crossovers(perm: list[int]) -> list[tuple[int, int]]:
    """Decompose permutation into sequential crossover moves using the bypass column."""
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


def _try_place_9rect(
    grid: np.ndarray,
    ch_cols: list[int],
    xover_rows: list[int],
    top_ep_row: int,
    bot_ep_row: int,
    n_channels: int,
    placed_rects: list[tuple[int, int, int, int]],
    required_parity: int | None,
) -> tuple[int, int, int, int] | None:
    """Attempt to place one rectangular 9-block with an optional area parity target."""
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
    rect_w = rect_c1 - rect_c0 + 1
    if rect_w > RECT_MAX_WIDTH or rect_c0 < 0 or rect_c1 >= w:
        return None

    max_h = min(RECT_H_RANGE[1], bot_ep_row - top_ep_row - 4)
    if max_h < RECT_H_RANGE[0]:
        return None

    heights = list(range(RECT_H_RANGE[0], max_h + 1))
    if required_parity is not None:
        heights = [rh for rh in heights if (rh * rect_w) % 2 == required_parity]
        if not heights:
            return None
    rect_h = random.choice(heights)

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
    straight_ok = n_cols == s_cols and w_rows == e_rows
    turn_ok = (
        len(n_cols) + len(s_cols) + len(w_rows) + len(e_rows) == 2
        and sum(bool(face) for face in (n_cols, s_cols, w_rows, e_rows)) == 2
        and not (n_cols and s_cols)
        and not (w_rows and e_rows)
    )
    total_wires = len(n_cols & s_cols) + len(w_rows & e_rows) + (1 if turn_ok else 0)
    if not (straight_ok or turn_ok) or total_wires < 1 or total_wires > 3:
        grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1] = saved
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
    """Draw the wire network and return the final wire assignment."""
    slot_wire: list[int | None] = list(range(n_channels))
    bypass_wire: int | None = None

    col_seg_start: dict[int, int] = {}
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


def _channel_rect_hits(
    grid: np.ndarray,
    ch_cols: list[int],
    perm: list[int],
    top_ep_row: int,
    bot_ep_row: int,
) -> list[list[int]] | None:
    """Trace each top-to-bottom channel through the wire-only grid and record touched 9-rect ids."""
    h, w = grid.shape
    rects_9, rect_lookup = _find_9rects(grid)
    comp = set((int(r), int(c)) for r, c in np.argwhere(grid != 0))
    hits_per_channel: list[list[int]] = []
    for i, col in enumerate(ch_cols):
        end, _dd, blocked, hits = _trace_wire(
            comp, (top_ep_row, col), (1, 0), h, w, rects_9, rect_lookup, block_on_odd=False
        )
        if blocked:
            return None
        if end != (bot_ep_row, ch_cols[perm[i]]):
            return None
        hits_per_channel.append(hits)
    return hits_per_channel


def _derive_input(
    grid: np.ndarray,
    ch_cols: list[int],
    perm: list[int],
    token_colors: list[int],
    top_token_row: int,
    bot_token_row: int,
    clear_paths: list[bool],
) -> tuple[np.ndarray, list[dict]]:
    """Place tokens while avoiding fake pass-through evidence on blocked X channels."""
    inp = grid.copy()
    channel_specs: list[dict] = []
    n_channels = len(ch_cols)
    n_x = random.choices([0, 1, 2], weights=X_INPUT_COUNT_WEIGHTS)[0]
    x_channels = set(random.sample(range(n_channels), min(n_x, n_channels)))

    for i in range(n_channels):
        color = token_colors[i]
        top_center = (top_token_row, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[perm[i]])
        source_side = random.choice(["top", "bottom"])
        source = top_center if source_side == "top" else bot_center
        far = bot_center if source_side == "top" else top_center
        kind = "x" if i in x_channels else "solid"

        if kind == "x":
            _stamp_token(inp, source[0], source[1], color, _X)
            if clear_paths[i] and random.random() < FAR_DIAMOND_PROB:
                _stamp_token(inp, far[0], far[1], color, _DIAMOND)
        else:
            _stamp_token(inp, source[0], source[1], color, _SOLID)

        channel_specs.append(
            {
                "color": color,
                "kind": kind,
                "source": source,
                "far": far,
                "clear": clear_paths[i],
            }
        )
    return inp, channel_specs


def _build_expected_output(grid: np.ndarray, channel_specs: list[dict], bg: int = 0) -> np.ndarray:
    """Independent oracle for the parity-gate behavior used by the generator."""
    out = grid.copy()
    for spec in channel_specs:
        color = spec["color"]
        sr, sc = spec["source"]
        fr, fc = spec["far"]

        if spec["kind"] == "solid":
            out[sr - 1 : sr + 2, sc - 1 : sc + 2] = bg
            _stamp_token(out, sr, sc, color, _DIAMOND)
            if spec["clear"]:
                _stamp_token(out, fr, fc, color, _X)
        else:
            if spec["clear"]:
                _stamp_token(out, fr, fc, color, _DIAMOND)
    return out


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

    while True:
        perm = list(range(n_channels))
        random.shuffle(perm)
        if perm != list(range(n_channels)):
            break

    crossovers = _decompose_to_crossovers(perm)
    if not crossovers:
        return None

    min_wire_rows = len(crossovers) * 2 + 2
    h_min = top_ep_row + min_wire_rows + top_token_row + 3
    if h_min > GRID_H_RANGE[1]:
        return None
    h = random.randint(max(GRID_H_RANGE[0], h_min), GRID_H_RANGE[1])

    bot_token_row = h - top_token_row - 1
    bot_ep_row = bot_token_row - EP_TOKEN_GAP
    if bot_ep_row - top_ep_row < min_wire_rows:
        return None

    avail = bot_ep_row - top_ep_row - 2
    step = max(2, avail // (len(crossovers) + 1))
    xover_rows = [top_ep_row + 1 + step * (idx + 1) for idx in range(len(crossovers))]
    if xover_rows[-1] >= bot_ep_row:
        xover_rows = [top_ep_row + 2 + idx * 2 for idx in range(len(crossovers))]
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

    n_rects = random.choices(RECT_COUNT_CHOICES, weights=RECT_COUNT_WEIGHTS)[0]
    placed_rects: list[tuple[int, int, int, int]] = []
    required_parities = [0, 1] + [None] * max(0, n_rects - 2)
    random.shuffle(required_parities)
    for parity in required_parities:
        placed = None
        for _ in range(RECT_RETRY_FACTOR):
            placed = _try_place_9rect(
                grid, ch_cols, xover_rows, top_ep_row, bot_ep_row, n_channels, placed_rects, parity
            )
            if placed is not None:
                placed_rects.append(placed)
                break
        if placed is None:
            return None

    rects_9, _lookup = _find_9rects(grid)
    if not any(rect["passable"] for rect in rects_9) or not any(not rect["passable"] for rect in rects_9):
        return None

    channel_hits = _channel_rect_hits(grid, ch_cols, perm, top_ep_row, bot_ep_row)
    if channel_hits is None:
        return None

    clear_paths: list[bool] = []
    blocked_via_odd = 0
    passed_via_even = 0
    for hits in channel_hits:
        seen_even = any(rects_9[idx]["passable"] for idx in hits)
        seen_odd = any(not rects_9[idx]["passable"] for idx in hits)
        clear = not seen_odd
        clear_paths.append(clear)
        if seen_odd:
            blocked_via_odd += 1
        if clear and seen_even:
            passed_via_even += 1

    if blocked_via_odd < 1 or passed_via_even < 1:
        return None

    inp, channel_specs = _derive_input(
        grid, ch_cols, perm, token_colors, top_token_row, bot_token_row, clear_paths
    )
    expected = _build_expected_output(inp, channel_specs, bg=0)

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
