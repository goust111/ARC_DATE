# Variant of a47bf94d
# Dimension: D6 — 源token形状类型 (src)
# Change: 三种token形状——实心3x3块（输入中主要形式）、X形（对角+中心，5像素）、菱形（4正交邻居，4像素）；每种颜色最多出现一个实心或X形 → 5x5实心块

from pathlib import Path
import random
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SOLVER_DIR))

from arc_utils import components_mask, get_bg, neighbors4

# Design analysis for the D6 variant:
# a) Concrete behavior:
# - The source token is no longer a 3x3 solid. It is a 5x5 solid block whose
#   center sits 4 cells away from a wire endpoint.
# - The 5x5 solid splits into two complementary 5x5-scale shapes:
#   1. a large cross: full middle row + full middle column (9 cells)
#   2. a corner-star: the remaining sixteen cells, forming four 2x2 corner blocks
# - Local behavior mirrors the original task: a solid token collapses in place to
#   the resting shape (large cross), while the missing complement (corner-star)
#   is routed to the far endpoint. A corner-star input stays as-is and emits the
#   large cross to the far endpoint. A pre-placed large cross is already stable.
# - Example on one channel:
#   original: solid 3x3 at top -> top diamond, bottom X
#   variant: solid 5x5 at top -> top large cross, bottom corner-star
#   The wire-routing logic is unchanged; only the token geometry and spacing grow.
# b) Complexity calibration:
# - A 5x5 solid is visually much larger than the original 3x3 and cannot be
#   confused with the wire colors.
# - The two complementary results are still few and repeated across channels:
#   every sample shows several large crosses and several corner-stars, which is
#   enough for humans to infer the decomposition.
# - The corner-star is disconnected under 4-connectivity, while the cross is
#   connected; this asymmetry makes the output visibly different from the source.
# c) Implicit assumptions that changed:
# - The old solver assumes centers are 2 cells from endpoints. That fails here,
#   because the 5x5 shapes extend 2 cells from center. The new solver uses a
#   center-to-endpoint offset of 4.
# - The old track-mask logic could ignore passive diamonds because they were
#   disconnected 1-cell pieces. The new passive cross is a connected 9-cell
#   object, so it must be explicitly removed from the wire mask.
# - Shape detection can no longer rely on a colored top-left cell: the cross has
#   background in its corners. Generic 5x5 mask matching is required.
# d) Placement legality:
# - Channel spacing is increased to 6-7 columns so 5x5 tokens do not overlap.
# - Grid height/width and token rows are enlarged so the bigger tokens stay in
#   bounds and leave room for the existing crossover network.
# - The 4-cell endpoint gap keeps passive large crosses detached from wires.
# e) Visual observability:
# - The change is visible immediately: inputs contain multiple 5x5 solids instead
#   of 3x3 squares, and outputs contain large crosses / four-corner clusters.
# - No hidden color ordering or color remapping is used; colors only identify
#   channel correspondences through the wire routing already present in the task.
# f) Visual cleanliness:
# - Tokens remain confined to top/bottom bands; wires and 9-rect junctions stay
#   in the middle band. This avoids clutter despite the larger token footprint.
# g) Generator guarantees for new solver assumptions:
# - Every token center is aligned exactly 4 cells from one degree-1 endpoint.
# - Passive large crosses are isolated from the wire mask.
# - Distinct token colors are preserved, so endpoint-to-token matching remains
#   unambiguous when the solver scans outward from component endpoints.

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_TOKEN_EP_GAP = 4
_TOKEN_RADIUS = 2
_TOKEN_SIZE = 2 * _TOKEN_RADIUS + 1

_CROSS5_OFF = [(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3) if dr == 0 or dc == 0]
_CORNER5_OFF = [(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3) if dr != 0 and dc != 0]
_SOLID5_OFF = [(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3)]


def _make_mask(offsets: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((_TOKEN_SIZE, _TOKEN_SIZE), dtype=bool)
    for dr, dc in offsets:
        mask[dr + _TOKEN_RADIUS, dc + _TOKEN_RADIUS] = True
    return mask


_CROSS5_MASK = _make_mask(_CROSS5_OFF)
_CORNER5_MASK = _make_mask(_CORNER5_OFF)


def _find_solid_5x5(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for maximal 5x5 solid blocks."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for r0 in range(h - 4):
        for c0 in range(w - 4):
            sub = g[r0 : r0 + 5, c0 : c0 + 5]
            col = int(sub[0, 0])
            if col == bg or np.any(sub != col):
                continue
            maximal = True
            for rr in range(r0, r0 + 5):
                for cc in range(c0, c0 + 5):
                    for nr, nc in neighbors4(rr, cc):
                        if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 5 and c0 <= nc < c0 + 5):
                            if int(g[nr, nc]) == col:
                                maximal = False
                                break
                    if not maximal:
                        break
                if not maximal:
                    break
            if not maximal:
                continue
            found[(r0 + 2, c0 + 2)] = (col, (r0, c0))
    return found


def _find_masked_5x5(
    g: np.ndarray, bg: int, mask: np.ndarray
) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for isolated 5x5 masked patterns."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    mask_coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    off_coords = [(r - _TOKEN_RADIUS, c - _TOKEN_RADIUS) for r, c in mask_coords]

    for r0 in range(h - 4):
        for c0 in range(w - 4):
            sub = g[r0 : r0 + 5, c0 : c0 + 5]
            sr, sc = mask_coords[0]
            col = int(sub[sr, sc])
            if col == bg:
                continue
            if np.any(sub[mask] != col) or np.any(sub[~mask] != bg):
                continue

            isolated = True
            cr, cc = r0 + _TOKEN_RADIUS, c0 + _TOKEN_RADIUS
            for dr, dc in off_coords:
                rr, cc2 = cr + dr, cc + dc
                for nr, nc in neighbors4(rr, cc2):
                    if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 5 and c0 <= nc < c0 + 5):
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


def _stamp_token(grid: np.ndarray, center_r: int, center_c: int, color: int, offsets: list[tuple[int, int]]) -> None:
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = color


def _erase_offsets(mask: np.ndarray, center: tuple[int, int], offsets: list[tuple[int, int]]) -> None:
    h, w = mask.shape
    cr, cc = center
    for dr, dc in offsets:
        rr, cc2 = cr + dr, cc + dc
        if 0 <= rr < h and 0 <= cc2 < w:
            mask[rr, cc2] = False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transmit enlarged colored tokens along wire networks, splitting each 5×5 source into complementary large shapes.

    Input structure:
    - The background is the dominant color. Non-background cells are either wire elements (8/5/9) or 5×5-scale tokens near wire endpoints: a solid 5×5 block, a corner-star made of four 2×2 corner chunks, or an already-settled large cross.

    Rule:
    - Every solid 5×5 block is replaced in place by the large cross (middle row + middle column), and its complementary corner-star is sent to the far endpoint of the same wire component.
    - Every corner-star token stays where it is and sends the large cross to the opposite endpoint. Pre-placed large crosses are stable and remain unchanged.

    Key insight:
    - Inside a 5×5 square, the solid token decomposes exactly into two complementary pieces: a 9-cell cross and sixteen corner cells. The wire network only decides where the missing complementary piece should appear.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    solids = _find_solid_5x5(g, bg)
    corners = _find_masked_5x5(g, bg, _CORNER5_MASK)
    crosses = _find_masked_5x5(g, bg, _CROSS5_MASK)

    track_mask = g != bg
    for (_, _), (_, (r0, c0)) in solids.items():
        track_mask[r0 : r0 + 5, c0 : c0 + 5] = False
    for center in corners:
        _erase_offsets(track_mask, center, _CORNER5_OFF)
    for center in crosses:
        _erase_offsets(track_mask, center, _CROSS5_OFF)

    comps = _components(track_mask)
    rects_9 = _find_9rects(g)
    out = g.copy()

    for (cr, cc), (col, (r0, c0)) in solids.items():
        out[r0 : r0 + 5, c0 : c0 + 5] = bg
        _stamp_token(out, cr, cc, col, _CROSS5_OFF)

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
                cen = (er + _TOKEN_EP_GAP * odr, ec + _TOKEN_EP_GAP * odc)
                if not (0 <= cen[0] < h and 0 <= cen[1] < w):
                    continue

                if cen in solids:
                    col, _ = solids[cen]
                    if col in comp_colors:
                        continue
                    dir_in = (-odr, -odc)
                    end, dd = _trace_wire(comp, ep, dir_in, h, w, rects_9)
                    dest = (end[0] + _TOKEN_EP_GAP * dd[0], end[1] + _TOKEN_EP_GAP * dd[1])
                    _stamp_token(out, dest[0], dest[1], col, _CORNER5_OFF)

                elif cen in corners:
                    col, _ = corners[cen]
                    if col in comp_colors:
                        continue
                    dir_in = (-odr, -odc)
                    end, dd = _trace_wire(comp, ep, dir_in, h, w, rects_9)
                    dest = (end[0] + _TOKEN_EP_GAP * dd[0], end[1] + _TOKEN_EP_GAP * dd[1])
                    _stamp_token(out, dest[0], dest[1], col, _CROSS5_OFF)

    return out


# --------------- generator ---------------

CHANNEL_COUNTS = [4, 5]
CHANNEL_SPACING = [6, 7]
LEFT_MARGIN_RANGE = (2, 4)
BYPASS_COL_OFFSET = 3
TOKEN_ROW_OPTIONS = [4, 5]
TOKEN_COLORS_POOL = [1, 2, 3, 4, 6, 7]
GRID_H_RANGE = (29, 38)
GRID_W_MIN = 33
GRID_W_MAX = 42

RECT_COUNT_WEIGHTS = [10, 40, 30, 20]
RECT_WIRE_COUNT_WEIGHTS = [50, 38, 12]
RECT_PARALLEL_PROB = 0.75
RECT_H_RANGE = (2, 6)
RECT_MAX_WIDTH = 10
RECT_RETRY_FACTOR = 5
RECT_MARGIN_GAP = 1

CORNER_INPUT_COUNT_WEIGHTS = [60, 30, 10]
FAR_CROSS_PROB = 0.33

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
    """Derive the input grid from the wire-only grid by placing large tokens."""
    inp = grid.copy()
    n_corner = random.choices([0, 1, 2], weights=CORNER_INPUT_COUNT_WEIGHTS)[0]
    corner_channels = set(random.sample(range(n_channels), min(n_corner, n_channels)))

    for i in range(n_channels):
        col = token_colors[i]
        top_center = (top_token_row, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[perm[i]])

        if i in corner_channels:
            if random.choice([True, False]):
                near, far = top_center, bot_center
            else:
                near, far = bot_center, top_center
            _stamp_token(inp, near[0], near[1], col, _CORNER5_OFF)
            if random.random() < FAR_CROSS_PROB:
                _stamp_token(inp, far[0], far[1], col, _CROSS5_OFF)
        else:
            side = random.choice([top_center, bot_center])
            _stamp_token(inp, side[0], side[1], col, _SOLID5_OFF)
    return inp


def _gen_attempt():
    n_channels = random.choice(CHANNEL_COUNTS)
    spacing = random.choice(CHANNEL_SPACING)

    left_margin = random.randint(*LEFT_MARGIN_RANGE)
    first_col = left_margin + _TOKEN_RADIUS
    ch_cols = [first_col + i * spacing for i in range(n_channels)]
    bypass_col = ch_cols[-1] + BYPASS_COL_OFFSET

    w = max(bypass_col + 4, GRID_W_MIN)
    if w > GRID_W_MAX:
        return None

    top_token_row = random.choice(TOKEN_ROW_OPTIONS)
    top_ep_row = top_token_row + _TOKEN_EP_GAP
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
    h_min = min_wire_rows + 2 * top_token_row + 2 * _TOKEN_EP_GAP + 1
    if h_min > GRID_H_RANGE[1]:
        return None
    h = random.randint(max(GRID_H_RANGE[0], h_min), GRID_H_RANGE[1])

    bot_token_row = h - top_token_row - 1
    bot_ep_row = bot_token_row - _TOKEN_EP_GAP
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

    output = grid.copy()
    for i in range(n_channels):
        _stamp_token(output, top_token_row, ch_cols[i], token_colors[i], _CROSS5_OFF)
        _stamp_token(output, bot_token_row, ch_cols[perm[i]], token_colors[i], _CORNER5_OFF)

    inp = _derive_input(grid, n_channels, ch_cols, perm, token_colors, top_token_row, bot_token_row)
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
