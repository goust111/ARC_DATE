# Variant of a47bf94d
# Dimension: D12 — 输入中token的初始状态 (src)
# Change: 混合输入——大部分通道以实心3x3块作为输入token，0-2个通道以X形作为输入token；X形通道的远端约67%留空... → 双端部分形状输入

# Design analysis for this D12 variant
# -----------------------------------
# spec_variant:
#   preserved:
#     - grid size, channel count, spacing, wire colors, crossing markers, and 9-rect pass-through behavior
#     - one color per routed channel, with a non-identity permutation implemented by the wire network
#     - output remains the same stable complementary pair per routed color:
#       top endpoint = full diamond/cross-without-center, bottom routed endpoint = full X/star-with-center
#   changed_core:
#     - the input no longer places a single "source token" and leaves the far side blank or already-resting
#     - instead every routed pair is visible at both ends in the input:
#       top endpoint gets an incomplete diamond (exactly 3 of the 4 cardinal arms)
#       bottom routed endpoint gets an incomplete X (center + exactly 3 diagonal corners)
#     - the solver must complete both partial shapes into the full complementary pair
#
#   behavior_trace:
#     - original local flow for one color:
#       input might contain a solid 3x3 at top, nothing at bottom; solver converts top solid -> diamond and routes
#       an X to the far endpoint
#     - new local flow for one color:
#       input already shows two incomplete resting states, e.g. top has a 3-arm diamond missing its east arm and
#       the routed bottom side has an X missing its lower-right corner
#       solver follows the same wire pairing, then fills only the missing arm/corner so the pair becomes
#       full diamond + full X
#     - difference:
#       the old rule had one informative end and one blank/optional end; the new rule gives both ends partial
#       information, so the reasoning target is "finish the paired endpoints" rather than "transport to an empty end"
#
#   complexity_calibration:
#     - partial-diamond and partial-X each use a small controlled family:
#       4 possible missing arms for diamond, 4 possible missing corners for X
#     - this is visually obvious across 2-4 examples and does not explode into too many subcases
#     - both partial shapes remain easy to distinguish from wires because token colors stay in {1,2,3,4,6,7}
#       while wires use {5,8,9}
#
#   implicit_assumption_review:
#     - original code assumed endpoint-adjacent tokens were either full solids or full Xs found by global detectors
#       and that only the solid side needed in-place rewriting
#     - that breaks here because incomplete shapes would not be detected, and the output would miss completions
#     - new detector logic therefore scans for partial diamonds and partial Xs, and track extraction removes only
#       the occupied token cells that are actually present in the input
#     - original routing loop was safe because many endpoints had no token nearby; in the variant every endpoint does,
#       so pair de-duplication is required to avoid processing both directions twice
#
#   placement_legality:
#     - generator keeps the original endpoint gap of 2 cells, so a 3x3 token never touches the wire cells
#     - top/bottom token centers are unchanged, therefore partial completions cannot overlap wire segments or 9-rects
#     - each endpoint receives exactly one nearby token center, ensuring the solver's endpoint lookup is unambiguous
#
#   observability_and_cleanliness:
#     - every channel visibly has two active ends in the input, unlike the original mostly single-ended state
#     - each partial differs from its output by one clearly missing cell, so the completion rule is easy to see
#     - the grid stays tidy because the only added complexity is missing cells inside the existing 3x3 token zones
#
#   color_inference:
#     - no color remapping is introduced
#     - the same token color appears on both routed ends of a channel, so the human-visible structure is
#       "same-colored partial pair connected by the wire permutation"
#
#   new_solver_assumptions_to_enforce_in_generator:
#     - each endpoint-adjacent token is either a 3-arm diamond or a 4-cell X with center present
#     - token colors remain isolated from same-colored neighbors outside their own 3x3
#     - the routed opposite endpoint exists and carries the same color, so paired completion is well-defined

from pathlib import Path
import random
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[5] / "solvers"))
from arc_utils import components_mask, get_bg, neighbors4

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_X_CORNERS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _inside_3x3(r0: int, c0: int, r: int, c: int) -> bool:
    return r0 <= r < r0 + 3 and c0 <= c < c0 + 3


def _isolated_positions(
    g: np.ndarray,
    bg: int,
    r0: int,
    c0: int,
    color: int,
    offsets: list[tuple[int, int]],
) -> bool:
    h, w = g.shape
    for dr, dc in offsets:
        rr, cc = r0 + 1 + dr, c0 + 1 + dc
        for nr, nc in neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and not _inside_3x3(r0, c0, nr, nc):
                if int(g[nr, nc]) == color:
                    return False
    return True


def _find_partial_diamonds(
    g: np.ndarray,
    bg: int,
) -> dict[tuple[int, int], tuple[int, list[tuple[int, int]]]]:
    """Return {center: (color, present_offsets)} for isolated partial/full diamonds."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, list[tuple[int, int]]]] = {}
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            if int(sub[1, 1]) != bg:
                continue
            if any(int(sub[rr, cc]) != bg for rr, cc in [(0, 0), (0, 2), (2, 0), (2, 2)]):
                continue

            vals = [int(sub[1 + dr, 1 + dc]) for dr, dc in _DIAMOND_OFF]
            nonbg = [v for v in vals if v != bg]
            if len(nonbg) not in (3, 4):
                continue
            color = nonbg[0]
            if any(v not in (bg, color) for v in vals) or any(v != color for v in nonbg):
                continue

            present = [off for off, v in zip(_DIAMOND_OFF, vals) if v == color]
            if not _isolated_positions(g, bg, r0, c0, color, present):
                continue
            found[(r0 + 1, c0 + 1)] = (color, present)
    return found


def _find_partial_xs(
    g: np.ndarray,
    bg: int,
) -> dict[tuple[int, int], tuple[int, list[tuple[int, int]]]]:
    """Return {center: (color, present_offsets)} for isolated partial/full X patterns."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, list[tuple[int, int]]]] = {}

    x_mask = np.zeros((3, 3), dtype=bool)
    for rr, cc in [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]:
        x_mask[rr, cc] = True

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            center_color = int(sub[1, 1])
            if center_color == bg:
                continue
            if np.any(sub[~x_mask] != bg):
                continue

            vals = [int(sub[1 + dr, 1 + dc]) for dr, dc in _X_OFF]
            nonbg = [v for v in vals if v != bg]
            if len(nonbg) not in (4, 5):
                continue
            if any(v not in (bg, center_color) for v in vals):
                continue
            if any(v != center_color for v in nonbg):
                continue

            corner_count = sum(int(sub[1 + dr, 1 + dc]) == center_color for dr, dc in _X_CORNERS)
            if corner_count not in (3, 4):
                continue

            present = [off for off, v in zip(_X_OFF, vals) if v == center_color]
            if not _isolated_positions(g, bg, r0, c0, center_color, present):
                continue
            found[(r0 + 1, c0 + 1)] = (center_color, present)
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


def _endpoint_token(
    ep: tuple[int, int],
    diamonds: dict[tuple[int, int], tuple[int, list[tuple[int, int]]]],
    xs: dict[tuple[int, int], tuple[int, list[tuple[int, int]]]],
) -> tuple[tuple[int, int], str, int, tuple[int, int]] | None:
    hits = []
    er, ec = ep
    for odr, odc in _N4:
        cen = (er + 2 * odr, ec + 2 * odc)
        if cen in diamonds:
            hits.append((cen, "diamond", diamonds[cen][0], (-odr, -odc)))
        if cen in xs:
            hits.append((cen, "x", xs[cen][0], (-odr, -odc)))
    if not hits:
        return None
    assert len(hits) == 1, "each endpoint should have exactly one nearby partial token"
    return hits[0]


def _stamp_shape(out: np.ndarray, center: tuple[int, int], color: int, kind: str) -> None:
    cr, cc = center
    offsets = _DIAMOND_OFF if kind == "diamond" else _X_OFF
    for dr, dc in offsets:
        out[cr + dr, cc + dc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete paired partial endpoint tokens into full complementary shapes across each wire route.

    Input structure:
    - Background cells stay empty. Non-background cells form the wire network plus two endpoint-adjacent 3x3 fragments per routed channel.
    - The top-side fragment is a partial diamond (3 of 4 cardinal arms, no center); the routed opposite side is a partial X (center + 3 diagonals).

    Rule:
    - Follow each wire from one endpoint to its routed partner and identify the partial token at both ends.
    - Fill the missing arm of every diamond fragment and the missing corner of every X fragment, keeping colors and wire layout unchanged.

    Key insight:
    - Nothing new is transported onto a blank side anymore: both ends already show incomplete members of the final diamond/X pair.
    - The wire network is used to pair the two partial endpoints of the same channel so both resting shapes can be completed consistently.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    diamonds = _find_partial_diamonds(g, bg)
    xs = _find_partial_xs(g, bg)

    track_mask = g != bg
    for (cr, cc), (_, offsets) in diamonds.items():
        for dr, dc in offsets:
            track_mask[cr + dr, cc + dc] = False
    for (cr, cc), (_, offsets) in xs.items():
        for dr, dc in offsets:
            track_mask[cr + dr, cc + dc] = False

    comps = _components(track_mask)
    rects_9 = _find_9rects(g)
    out = g.copy()

    processed_pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()

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
            start_info = _endpoint_token(ep, diamonds, xs)
            if start_info is None:
                continue
            center, kind, color, dir_in = start_info
            end, dd = _trace_wire(comp, ep, dir_in, h, w, rects_9)
            pair_key = tuple(sorted((ep, end)))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)

            dest_center = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
            dest_info = None
            if dest_center in diamonds:
                dest_info = ("diamond", diamonds[dest_center][0])
            elif dest_center in xs:
                dest_info = ("x", xs[dest_center][0])

            _stamp_shape(out, center, color, kind)
            if dest_info is None:
                dest_kind = "x" if kind == "diamond" else "diamond"
                dest_color = color
            else:
                dest_kind, dest_color = dest_info
                if dest_color != color:
                    dest_color = color
            _stamp_shape(out, dest_center, dest_color, dest_kind)

    return out


# --------------- generator (dual_partial_endpoints) ---------------

_DIAMOND = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]

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


def _paint_offsets(grid: np.ndarray, center_r: int, center_c: int, color: int, offsets) -> None:
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = color


def _sample_partial_diamond() -> list[tuple[int, int]]:
    missing = random.choice(_DIAMOND)
    return [off for off in _DIAMOND if off != missing]


def _sample_partial_x() -> list[tuple[int, int]]:
    missing = random.choice(_X_CORNERS)
    return [off for off in _X if off != missing]


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


def _derive_input(grid, n, ch_cols, perm, token_colors, top_token_row, bot_token_row):
    """Place a partial diamond at the top end and a partial X at the routed bottom end of every channel."""
    inp = grid.copy()
    for i in range(n):
        color = token_colors[i]
        top_center = (top_token_row, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[perm[i]])
        _paint_offsets(inp, top_center[0], top_center[1], color, _sample_partial_diamond())
        _paint_offsets(inp, bot_center[0], bot_center[1], color, _sample_partial_x())
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

    output = grid.copy()
    for i in range(n):
        _paint_offsets(output, top_token_row, ch_cols[i], token_colors[i], _DIAMOND)
        _paint_offsets(output, bot_token_row, ch_cols[perm[i]], token_colors[i], _X)

    inp = _derive_input(grid, n, ch_cols, perm, token_colors, top_token_row, bot_token_row)
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
