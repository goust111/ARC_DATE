# Variant of a47bf94d
# Dimension: D3 — 传输轨迹可视化 (trajectory)
# Change: 无轨迹涂色——管道网络在输入和输出中保持不变，不因token传输而涂色或改变... → 颜色渐变轨迹

# Design analysis for this variant
# - Concrete behavior:
#   The original task already has unambiguous source/destination endpoints and a long routed
#   transport path, so the D3 change is applied only to the output-side rendering of that path.
#   Input wires remain plain structural colors {5,8,9}. After solving the usual shape logic,
#   every genuinely transmitted token also paints its route on the wire network. The painted
#   route uses a fixed visible sequence [source token color -> 5 -> 9 -> destination token color].
#   Because this task preserves token color during transport, source and destination endpoint
#   colors are usually the same; the new information is therefore carried by the middle colors
#   and by the direction implied by which endpoint contained the input token.
#   Example: if a red solid at the top sends an X to the bottom, the old output kept the whole
#   pipe in {5,8,9}; the new output keeps the same topology but recolors the ordered route cells
#   from red near the source, through grey then maroon, back to red near the arrival end.
# - Where the pipeline changes:
#   Original flow:
#   input token detection -> route tracing -> place complementary token -> leave wires unchanged.
#   Variant flow:
#   input token detection -> route tracing -> place complementary token -> recolor ordered route
#   cells with the fixed gradient, while leaving ambiguous shared junction cells structural.
# - Complexity calibration:
#   The gradient uses only two intermediate colors (5 and 9). That is enough to make motion
#   visible on long wires, but still small enough that a human can infer a single repeated rule
#   across 2-4 examples. More bands would make the mapping harder to infer; fewer bands would
#   look too close to the original task.
# - Hidden assumptions audited from the original solver:
#   1. `track_mask = g != bg` still works because the input wire network is not pre-colored by
#      trajectories; only the output is colored.
#   2. `_find_9rects` may keep assuming structural wire colors around rectangles are in {5,8},
#      because rectangles are parsed only from the input grid.
#   3. `out = g.copy()` is no longer enough: in the original task leaving untouched wire colors
#      was harmless, but here the route must be painted explicitly after tracing.
#   4. Shared cells are now dangerous. At crossing centers or inside color-9 rectangles, two
#      logical routes can touch the same structural region. Blindly recoloring every visited cell
#      would create overwrite conflicts that did not matter before. This variant therefore only
#      paints route cells that are visually attributable to one path: non-9 cells whose local
#      degree inside the component is at most 2. Crossing centers and wide 9-block interiors stay
#      structural.
#   5. `comp_colors` still serves the old purpose of suppressing transport when a far-end diamond
#      is already connected, but that branch would hide the new D3 behavior. The generator is
#      adjusted to stop pre-placing far diamonds so every sampled transport remains visible.
# - Placement legality:
#   The generator keeps the original channel spacing, token offsets, endpoint gaps, rectangle
#   placement checks, and permutation routing. The only sampling change is that solid inputs are
#   placed on the top side and X inputs on the bottom side, matching the original output layout
#   and avoiding rejected random orientations. This preserves non-overlap and keeps every route
#   compatible with the solver's endpoint search.
# - Visual observability:
#   The topology, crossings, and 9-rectangles remain as in the source task, so the examples still
#   teach the same routing rule. The new feature is visible because each successful transport adds
#   a repeated multi-color trail on the exact route. The generator rejects samples with too few
#   painted trajectory cells or without both intermediate colors appearing.
# - Color-change inferability:
#   No hidden numeric ordering of color ids is used. The band sequence is fixed and directly
#   visible. Each example contains several channels, so the same middle colors repeat on multiple
#   routes in one grid, giving enough evidence that the change depends on travel order rather than
#   on token identity.
# - Main's new implicit assumptions and generator guarantees:
#   1. Token colors must stay outside {5,8,9}; the generator still draws from {1,2,3,4,6,7}.
#   2. Ambiguous shared cells must be rare and localized to crossings / 9-rectangles; the original
#      routing construction already guarantees that.
#   3. Each active route needs enough simple wire cells for the gradient to be visible; the
#      generator enforces this through a post-check on painted trajectory statistics.

import numpy as np
import random

from arc_utils import components_mask, get_bg, neighbors4

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-neigh around center
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]  # diagonals + center
_TRAJ_MID_PALETTE = [5, 9]
_WIRE_STRUCT_COLORS = {5, 8, 9}


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


_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DIR2FACE = {(1, 0): "N", (-1, 0): "S", (0, 1): "W", (0, -1): "E"}
_FACE2DIR = {"S": (1, 0), "N": (-1, 0), "E": (0, 1), "W": (0, -1)}


def _trace_wire_path(
    comp: set[tuple[int, int]],
    start: tuple[int, int],
    dir0: tuple[int, int],
    h: int,
    w: int,
    rects_9: list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]],
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    """
    Greedy wire following with 9-rect pass-through, returning the ordered path.

    The returned path contains the visible wire cells encountered in order. Wide color-9
    blocks are still traversed logically, but only their entry cell and the exterior exit
    connector appear in the ordered path; the interiors are kept structural in the output.
    """
    cur, d, prev = start, dir0, None
    path = [start]
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 4):
        state = (cur, d)
        if state in visited:
            raise AssertionError("wire trace looped — should not happen with 9-rect pass-through")
        visited.add(state)

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
                if path[-1] != cur:
                    path.append(cur)
                jumped = True
                break
        if jumped:
            continue

        r, c = cur
        opts = [((nr, nc), nr - r, nc - c) for nr, nc in neighbors4(r, c) if (nr, nc) in comp]
        if not opts:
            return path, cur, d

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
            return path, cur, d

        prev = cur
        cur = p
        d = (dr, dc)
        path.append(cur)

    raise AssertionError("wire trace exceeded step limit")


def _paint_trajectory(
    out: np.ndarray,
    g: np.ndarray,
    path: list[tuple[int, int]],
    deg: dict[tuple[int, int], int],
    color: int,
) -> dict[str, object]:
    """Paint a route gradient on the unambiguous path cells."""
    paintable = [p for p in path if int(g[p]) != 9 and deg.get(p, 0) <= 2]
    palette = [color, *_TRAJ_MID_PALETTE, color]
    band_hits = {mid: 0 for mid in _TRAJ_MID_PALETTE}

    if not paintable:
        return {"count": 0, "band_hits": band_hits}

    n = len(paintable)
    last_band = len(palette) - 1
    for idx, (r, c) in enumerate(paintable):
        band = 0 if n == 1 else int(idx * last_band / (n - 1))
        col = palette[band]
        out[r, c] = col
        if col in band_hits:
            band_hits[col] += 1
    return {"count": n, "band_hits": band_hits}


def _solve(grid: np.ndarray, *, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)

    track_mask = g != bg
    for (_cr, _cc), (_col, (r0, c0)) in solids.items():
        track_mask[r0 : r0 + 3, c0 : c0 + 3] = False
    for cr, cc in xs.keys():
        for dr, dc in _X_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                track_mask[rr, cc2] = False

    comps = _components(track_mask)
    rects_9 = _find_9rects(g)

    out = g.copy()
    route_stats = []

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
                    path, end, dd = _trace_wire_path(comp, ep, dir_in, h, w, rects_9)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    for dr, dc in _X_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col
                    paint_meta = _paint_trajectory(out, g, path, deg, col)
                    route_stats.append({"kind": "solid", "color": col, **paint_meta})

                elif cen in xs:
                    col, _ = xs[cen]
                    if col in comp_colors:
                        continue
                    dir_in = (-odr, -odc)
                    path, end, dd = _trace_wire_path(comp, ep, dir_in, h, w, rects_9)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    for dr, dc in _DIAMOND_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col
                    paint_meta = _paint_trajectory(out, g, path, deg, col)
                    route_stats.append({"kind": "x", "color": col, **paint_meta})

    if not return_meta:
        return out

    band_totals = {mid: 0 for mid in _TRAJ_MID_PALETTE}
    for stat in route_stats:
        for mid in _TRAJ_MID_PALETTE:
            band_totals[mid] += int(stat["band_hits"][mid])
    meta = {
        "route_count": len(route_stats),
        "trajectory_cells": sum(int(stat["count"]) for stat in route_stats),
        "band_totals": band_totals,
        "route_lengths": [int(stat["count"]) for stat in route_stats],
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route endpoint tokens through the wire network, while also visualizing each actual transport path.

    Input structure:
    - Background cells are empty. Non-background cells form the wire network (colors 5/8/9) and
      endpoint tokens: either a solid 3x3 block or an X-pattern placed two cells away from a wire tip.

    Rule:
    - A solid 3x3 token becomes an in-place diamond, and it sends an X of the same color to the
      opposite endpoint. An X-token stays an X and sends an in-place diamond of the same color.
    - In addition, every route that is genuinely used gets painted on the output wire network with
      a fixed ordered gradient: source token color -> 5 -> 9 -> destination token color. Shared
      structural junction cells (notably 9-block interiors and high-degree crossing centers) keep
      their original wire colors instead of receiving an ambiguous path color.

    Key insight:
    - The routing and shape logic stay the same as the source task; the new thing to notice is that
      the transported token now leaves a directional color trail along the exact pipe it followed.
    """
    return _solve(grid)


# --------------- generator (trajectory_gradient) ---------------

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
RECT_COUNT_WEIGHTS = [10, 40, 30, 20]
RECT_WIRE_COUNT_WEIGHTS = [50, 38, 12]
RECT_PARALLEL_PROB = 0.75
RECT_H_RANGE = (2, 5)
RECT_MAX_WIDTH = 7
RECT_RETRY_FACTOR = 5
RECT_MARGIN_GAP = 1

# --- Input derivation ---
X_INPUT_COUNT_WEIGHTS = [60, 30, 10]
FAR_DIAMOND_PROB = 0.0

# --- Generation retry / visibility guard ---
GEN_MAX_RETRIES = 300
MIN_TRAJECTORY_CELLS = 18


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


def _stamp_token(grid, center_r, center_c, color, offsets):
    """Write a token pattern (diamond/X/solid) onto grid at given center."""
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = color


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
    """Draw the wire network onto grid. Returns (slot_wire, bypass_wire) after all crossovers."""
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


def _derive_input(grid, n_channels, ch_cols, perm, token_colors, top_token_row, bot_token_row):
    """Derive the input grid from the wire-only grid by placing visible source tokens."""
    inp = grid.copy()
    n_x = random.choices([0, 1, 2], weights=X_INPUT_COUNT_WEIGHTS)[0]
    x_channels = set(random.sample(range(n_channels), min(n_x, n_channels)))

    for i in range(n_channels):
        col = token_colors[i]
        top_center = (top_token_row, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[perm[i]])

        if i in x_channels:
            _stamp_token(inp, bot_center[0], bot_center[1], col, _X)
            if random.random() < FAR_DIAMOND_PROB:
                _stamp_token(inp, top_center[0], top_center[1], col, _DIAMOND)
        else:
            _stamp_token(inp, top_center[0], top_center[1], col, _SOLID)
    return inp


def _trajectory_visible(meta: dict[str, object], n_channels: int) -> bool:
    if int(meta["route_count"]) != n_channels:
        return False
    if int(meta["trajectory_cells"]) < MIN_TRAJECTORY_CELLS:
        return False
    band_totals = meta["band_totals"]
    if any(int(band_totals[mid]) == 0 for mid in _TRAJ_MID_PALETTE):
        return False
    if min(meta["route_lengths"], default=0) < 4:
        return False
    return True


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

    inp = _derive_input(grid, n_channels, ch_cols, perm, token_colors, top_token_row, bot_token_row)
    try:
        _out, meta = _solve(inp, return_meta=True)
    except Exception:
        return None
    if not _trajectory_visible(meta, n_channels):
        return None
    return inp


def generate_input():
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
