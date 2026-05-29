# Variant of a47bf94d
# Dimension: D8 — 9号矩形块的角色 (obstacle)
# Change: 直通连接节点——9号颜色的矩形块（2x2到5x4）嵌入管道中，管道从入口面直线穿过到对面出口面（仅当单入单出在相邻面时才... → 延迟旋转节点

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

# Design analysis for the D8 variant
# a) Concrete behavior
#    - The original task routes endpoint tokens through one wire component and only changes
#      the token shape at the destination. The D8 change only affects that transmitted shape,
#      so the wire routing logic should remain intact.
#    - Rotation was invisible before because both transmitted shapes (diamond and X) are
#      quarter-turn symmetric. To make D8 observable, the token pair is replaced by a 5-cell
#      "emit" shape and its 4-cell complementary "rest" shape. They still partition a 3x3
#      solid block, so the old solid->split intuition is preserved.
#    - Canonical rest shape B0 = {N,W,C,S}; canonical emitted shape A0 = the other five cells
#      {NW,NE,E,SW,SE}. A0/B0 are both non-symmetric under 90-degree rotation.
#    - Example trace: if a solid source sends A0 through a wide 9-rectangle, the path still
#      ends at the same far endpoint as before, but the placed far token is A1 (A0 rotated CW)
#      instead of A0. A tall rectangle would produce A3 (CCW). This is the exact D8 delta.
# b) Complexity calibration
#    - Only four orientations exist, and they are reused across all samples. This is enough to
#      make the rule visible without adding an open-ended orientation system.
#    - The generator still uses 4-5 channels and 1-3 embedded rectangles, so there is repeated
#      support inside each sample rather than a one-off exceptional case.
# c) Hidden assumptions from the original main()
#    - Original `_trace_wire` assumed 9-blocks only affected exit face. That remains true for
#      geometry, but now the function must also accumulate a rotation counter.
#    - Original track extraction ignored pre-placed destination diamonds because they were split
#      into isolated cells and could not mimic a wire. The new 4-cell rest token is connected,
#      so it must be removed from the track mask explicitly.
#    - Original source X tokens had no orientation state. The new emitted token does, so the
#      solver must preserve its source orientation and rotate the complementary destination
#      shape relative to that orientation.
# d) Placement legality
#    - Tokens still live in 3x3 boxes centered two cells outside degree-1 endpoints, so they
#      do not overlap wires or each other under the original spacing constraints.
#    - The generator forbids square 9-rectangles because D8 only defines width>height and
#      height>width cases. It also rejects layouts where every channel gets net rotation 0.
# e) Visual observability
#    - The new shapes are monochrome and orientation is encoded purely by visible occupancy, not
#      by color ids or hidden ordering. Humans can compare the aspect ratio of a 9-rectangle to
#      the turned destination token directly.
# f) Visual cleanliness
#    - Only three token types appear near endpoints: solid source, 5-cell emitted source, and
#      4-cell completed/rest state. They are all 3x3-local and clearly separated from the wire.
# g) Color inference
#    - Colors are unchanged by the transform. Every channel keeps one token color, so D8 adds a
#      spatial/orientation rule without introducing a color remapping puzzle.

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_SOLID_OFF = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]
_REST_BASE = [(-1, 0), (0, -1), (0, 0), (1, 0)]  # canonical 4-cell complement left behind
_EMIT_BASE = [(-1, -1), (-1, 1), (0, 1), (1, -1), (1, 1)]  # complement sent through wire


def _rotate_offset(dr: int, dc: int, rot: int) -> tuple[int, int]:
    rot %= 4
    for _ in range(rot):
        dr, dc = dc, -dr
    return dr, dc


def _rotate_offsets(offsets: list[tuple[int, int]], rot: int) -> list[tuple[int, int]]:
    return [_rotate_offset(dr, dc, rot) for dr, dc in offsets]


def _mask_from_offsets(offsets: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((3, 3), dtype=bool)
    for dr, dc in offsets:
        mask[1 + dr, 1 + dc] = True
    return mask


_REST_VARIANTS = [_rotate_offsets(_REST_BASE, rot) for rot in range(4)]
_EMIT_VARIANTS = [_rotate_offsets(_EMIT_BASE, rot) for rot in range(4)]
_REST_MASKS = [_mask_from_offsets(offsets) for offsets in _REST_VARIANTS]
_EMIT_MASKS = [_mask_from_offsets(offsets) for offsets in _EMIT_VARIANTS]


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


def _is_isolated_pattern(
    g: np.ndarray,
    r0: int,
    c0: int,
    col: int,
    offsets: list[tuple[int, int]],
) -> bool:
    h, w = g.shape
    for dr, dc in offsets:
        rr, cc = r0 + 1 + dr, c0 + 1 + dc
        for nr, nc in neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                if int(g[nr, nc]) == col:
                    return False
    return True


def _find_oriented_3x3(
    g: np.ndarray,
    bg: int,
    masks: list[np.ndarray],
    variants: list[list[tuple[int, int]]],
) -> dict[tuple[int, int], tuple[int, int, tuple[int, int]]]:
    """Return {center: (color, rotation, (r0,c0))} for isolated oriented 3x3 patterns."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, int, tuple[int, int]]] = {}
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            for rot, mask in enumerate(masks):
                vals = sub[mask]
                if vals.size == 0:
                    continue
                col = int(vals[0])
                if col == bg or np.any(vals != col) or np.any(sub[~mask] != bg):
                    continue
                if not _is_isolated_pattern(g, r0, c0, col, variants[rot]):
                    continue
                found[(r0 + 1, c0 + 1)] = (col, rot, (r0, c0))
                break
    return found


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=4)]


def _rect_turn(r0: int, r1: int, c0: int, c1: int) -> int:
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if w > h:
        return 1
    if h > w:
        return -1
    return 0


def _find_9rects(g: np.ndarray) -> list[dict[str, object]]:
    """Find color-9 rectangular blocks and their wire connections per face."""
    mask9 = g == 9
    if not mask9.any():
        return []
    h, w = g.shape
    rects: list[dict[str, object]] = []
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
        rects.append(
            {
                "cells": cells,
                "faces": faces,
                "turn": _rect_turn(r0, r1, c0, c1),
            }
        )
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
    rects_9: list[dict[str, object]],
) -> tuple[tuple[int, int], tuple[int, int], int]:
    """
    Greedy wire following with 9-rect pass-through and accumulated token rotation.

    Geometry is unchanged by D8: wires still exit opposite faces when possible, or take the
    unique adjacent-face continuation in the original corner case. The extra return value is
    the net clockwise quarter-turn count modulo 4.
    """
    cur, d, prev = start, dir0, None
    net_rot = 0
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 4):
        state = (cur, d)
        if state in visited:
            raise AssertionError("wire trace looped — should not happen with 9-rect routing")
        visited.add(state)

        jumped = False
        for rect in rects_9:
            cells9 = rect["cells"]
            if cur not in cells9:
                continue
            faces9 = rect["faces"]
            entry_face = _DIR2FACE[d]
            opp = _OPP[entry_face]
            if opp in faces9:
                exit_face = opp
            else:
                others = [face for face in faces9 if face != entry_face]
                if len(others) != 1:
                    raise AssertionError("ambiguous 9-rect exit face")
                exit_face = others[0]
            pts = faces9[exit_face]
            er, ec = cur
            if entry_face in ("N", "S"):
                exit_cell = min(pts, key=lambda p: abs(p[1] - ec))
            else:
                exit_cell = min(pts, key=lambda p: abs(p[0] - er))
            net_rot = (net_rot + int(rect["turn"])) % 4
            cur, d, prev = exit_cell, _FACE2DIR[exit_face], None
            jumped = True
            break
        if jumped:
            continue

        r, c = cur
        opts = [((nr, nc), nr - r, nc - c) for nr, nc in neighbors4(r, c) if (nr, nc) in comp]
        if not opts:
            return cur, d, net_rot

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
            return cur, d, net_rot

        prev = cur
        cur = p
        d = (dr, dc)

    raise AssertionError("wire trace exceeded step limit")


def _stamp_token(
    grid: np.ndarray,
    center_r: int,
    center_c: int,
    color: int,
    offsets: list[tuple[int, int]],
    rot: int = 0,
) -> None:
    for dr, dc in _rotate_offsets(offsets, rot):
        grid[center_r + dr, center_c + dc] = color


def _remove_token_cells(
    track_mask: np.ndarray,
    items: dict[tuple[int, int], tuple[int, int, tuple[int, int]]] | dict[tuple[int, int], tuple[int, tuple[int, int]]],
    base_variants: list[list[tuple[int, int]]] | None,
) -> None:
    for center, data in items.items():
        if base_variants is None:
            _, (r0, c0) = data  # type: ignore[misc]
            track_mask[r0 : r0 + 3, c0 : c0 + 3] = False
            continue
        _, rot, _ = data  # type: ignore[misc]
        cr, cc = center
        for dr, dc in base_variants[rot]:
            track_mask[cr + dr, cc + dc] = False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route endpoint tokens through wire networks while 9-rectangles accumulate quarter-turns.

    Input structure:
    - Background is the dominant color. Non-background cells form one or more orthogonal wire
      components (8 with 5 as straight-through crossing markers), embedded 9-rectangles, and
      3x3 endpoint tokens.
    - Source tokens are either solid 3x3 blocks or 5-cell oriented emit-shapes. Some far
      endpoints may already contain the 4-cell rest-shape that would be transmitted there.

    Rule:
    - A solid block becomes the canonical 4-cell rest-shape in place and sends its 5-cell
      complement to the opposite endpoint.
    - A 5-cell emit token stays in place and sends the complementary 4-cell rest-shape.
    - Each time a transmitted complement passes through a 9-rectangle, rotate it by 90 degrees:
      clockwise if the rectangle is wider than tall, counterclockwise if taller than wide.

    Key insight:
    - The wire geometry still only chooses which two endpoints are paired; the new information is
      the net quarter-turn accumulated from the aspect ratios of the 9-rectangles on that route.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    solids = _find_solid_3x3(g, bg)
    emits = _find_oriented_3x3(g, bg, _EMIT_MASKS, _EMIT_VARIANTS)
    rests = _find_oriented_3x3(g, bg, _REST_MASKS, _REST_VARIANTS)

    track_mask = g != bg
    _remove_token_cells(track_mask, solids, None)
    _remove_token_cells(track_mask, emits, _EMIT_VARIANTS)
    _remove_token_cells(track_mask, rests, _REST_VARIANTS)

    comps = _components(track_mask)
    rects_9 = _find_9rects(g)
    out = g.copy()

    for (cr, cc), (col, (r0, c0)) in solids.items():
        out[r0 : r0 + 3, c0 : c0 + 3] = bg
        _stamp_token(out, cr, cc, col, _REST_BASE, rot=0)

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
                if not (0 <= cen[0] < h and 0 <= cen[1] < w):
                    continue

                if cen in solids:
                    col, _ = solids[cen]
                    base_rot = 0
                    dest_offsets = _EMIT_BASE
                elif cen in emits:
                    col, base_rot, _ = emits[cen]
                    dest_offsets = _REST_BASE
                else:
                    continue

                dir_in = (-odr, -odc)
                end, dd, net_rot = _trace_wire(comp, ep, dir_in, h, w, rects_9)
                dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                _stamp_token(out, dest[0], dest[1], col, dest_offsets, rot=(base_rot + net_rot) % 4)

    return out


# --------------- generator (delayed_rotation_emitpair) ---------------

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

RECT_COUNT_WEIGHTS = [45, 35, 20]               # weights for 1, 2, 3 rects
RECT_WIRE_COUNT_WEIGHTS = [50, 38, 12]
RECT_PARALLEL_PROB = 0.75
RECT_H_RANGE = (2, 5)
RECT_MAX_WIDTH = 7
RECT_RETRY_FACTOR = 8
RECT_MARGIN_GAP = 1

EMIT_INPUT_COUNT_WEIGHTS = [60, 30, 10]         # weights for 0, 1, 2 emit-shape inputs
FAR_REST_PROB = 0.33

GEN_MAX_RETRIES = 400


def _decompose_to_crossovers(perm: list[int]) -> list[tuple[int, int]]:
    """Decompose permutation into sequential crossover moves using bypass for cycles."""
    n = len(perm)
    visited = [False] * n
    crossovers: list[tuple[int, int]] = []
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
    n: int,
    placed_rects: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    """Attempt to place one non-square color-9 rectangle."""
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
    rect_w = rect_c1 - rect_c0 + 1
    if rect_w > RECT_MAX_WIDTH or rect_c0 < 0 or rect_c1 >= w:
        return None

    rect_h_max = min(RECT_H_RANGE[1], bot_ep_row - top_ep_row - 4)
    if rect_h_max < RECT_H_RANGE[0]:
        return None
    rect_h = random.randint(RECT_H_RANGE[0], rect_h_max)
    if rect_h == rect_w:
        return None

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
        if (
            rect_r0 <= pr1 + RECT_MARGIN_GAP
            and pr0 <= rect_r1 + RECT_MARGIN_GAP
            and rect_c0 <= pc1 + RECT_MARGIN_GAP
            and pc0 <= rect_c1 + RECT_MARGIN_GAP
        ):
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


def _compute_channel_routes(
    grid: np.ndarray,
    ch_cols: list[int],
    top_ep_row: int,
    bot_ep_row: int,
) -> tuple[list[int], list[int]]:
    """Return bottom-slot permutation and net rotations for each top channel."""
    comps = [set(comp) for comp in _components(grid != 0)]
    rects_9 = _find_9rects(grid)
    col_to_slot = {c: i for i, c in enumerate(ch_cols)}
    perm: list[int] = []
    rots: list[int] = []
    for c in ch_cols:
        ep = (top_ep_row, c)
        comp = next((comp for comp in comps if ep in comp), None)
        if comp is None:
            raise AssertionError("top endpoint missing from wire component")
        end, d, rot = _trace_wire(comp, ep, (1, 0), grid.shape[0], grid.shape[1], rects_9)
        if end[0] != bot_ep_row or d != (1, 0) or end[1] not in col_to_slot:
            raise AssertionError("channel route does not terminate at a bottom endpoint")
        perm.append(col_to_slot[end[1]])
        rots.append(rot % 4)
    return perm, rots


def _derive_input_output(
    grid: np.ndarray,
    n: int,
    ch_cols: list[int],
    perm: list[int],
    rots: list[int],
    token_colors: list[int],
    top_token_row: int,
    bot_token_row: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a random valid input and its expected output for the rotated-token variant."""
    inp = grid.copy()
    out = grid.copy()
    n_emit = random.choices([0, 1, 2], weights=EMIT_INPUT_COUNT_WEIGHTS)[0]
    emit_channels = set(random.sample(range(n), min(n_emit, n)))

    for i in range(n):
        col = token_colors[i]
        top_center = (top_token_row, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[perm[i]])
        net_rot = rots[i] % 4

        src_is_top = random.choice([True, False])
        src_center = top_center if src_is_top else bot_center
        dst_center = bot_center if src_is_top else top_center

        if i in emit_channels:
            src_rot = random.randrange(4)
            dst_rot = (src_rot + net_rot) % 4
            _stamp_token(inp, src_center[0], src_center[1], col, _EMIT_BASE, src_rot)
            if random.random() < FAR_REST_PROB:
                _stamp_token(inp, dst_center[0], dst_center[1], col, _REST_BASE, dst_rot)
            _stamp_token(out, src_center[0], src_center[1], col, _EMIT_BASE, src_rot)
            _stamp_token(out, dst_center[0], dst_center[1], col, _REST_BASE, dst_rot)
        else:
            _stamp_token(inp, src_center[0], src_center[1], col, _SOLID_OFF, 0)
            _stamp_token(out, src_center[0], src_center[1], col, _REST_BASE, 0)
            _stamp_token(out, dst_center[0], dst_center[1], col, _EMIT_BASE, net_rot)

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
    token_colors = random.sample(TOKEN_COLORS_POOL, n)

    while True:
        perm = list(range(n))
        random.shuffle(perm)
        if perm != list(range(n)):
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
    xover_rows = [top_ep_row + 1 + step * (s + 1) for s in range(len(crossovers))]
    if xover_rows[-1] >= bot_ep_row:
        xover_rows = [top_ep_row + 2 + s * 2 for s in range(len(crossovers))]
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

    target_rects = random.choices([1, 2, 3], weights=RECT_COUNT_WEIGHTS)[0]
    placed_rects: list[tuple[int, int, int, int]] = []
    for _ in range(target_rects * RECT_RETRY_FACTOR):
        if len(placed_rects) >= target_rects:
            break
        result = _try_place_9rect(grid, ch_cols, xover_rows, top_ep_row, bot_ep_row, n, placed_rects)
        if result is not None:
            placed_rects.append(result)
    if not placed_rects:
        return None

    try:
        routed_perm, rots = _compute_channel_routes(grid, ch_cols, top_ep_row, bot_ep_row)
    except Exception:
        return None
    if routed_perm != perm:
        return None
    if all(rot % 4 == 0 for rot in rots):
        return None

    inp, out = _derive_input_output(grid, n, ch_cols, perm, rots, token_colors, top_token_row, bot_token_row)
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
