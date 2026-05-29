# Combo variant of a47bf94d (index 9)
# combo_spec: D5/3+D12/0
# Name: 擦除源+全实心初始状态

"""
Design analysis for the combo variant

a) Concretization and behavior trace
- The wire geometry stays the same as the original task: 4-5 channels, one routed orthogonal
  network, optional 9-rect pass-through blocks, and token centers exactly two cells outside
  degree-1 wire endpoints.
- The original solver had two visible token stages: a solid 3x3 source kept a local 4-cell
  diamond while sending an X to the far end. This combo removes both the source remnant and the
  mixed-shape input family.
- Example: a blue 3x3 block above top endpoint i routes to bottom endpoint perm[i]. Original
  output left a blue diamond at the top and an X at the bottom. Combo output erases the entire
  top 3x3 and creates a full blue 3x3 only at the bottom destination.
- Because D12/0 forces every input token to be solid, there is a single observable source type in
  all examples. The puzzle becomes “whole block transport” rather than “shape splitting”.

b) Complexity calibration
- The routing burden remains intact: non-identity permutations, crossovers marked by 5, and 9
  rectangles still have to be interpreted. So the variant is simpler than the original only in the
  token rule, not in the path-following rule.
- To avoid over-simplifying the visuals, the generator still places sources on both top and bottom
  sides across a grid. That makes the erase-and-reappear rule visible in multiple contexts.
- Restricting inputs to solid sources reduces branchiness without making samples degenerate, because
  several colors still move through different routes in each instance.

c) Implicit assumption audit
- Original branch “solid becomes local diamond” is invalid and must be removed; source pads now
  become plain background.
- Original branch “emit complementary X/diamond” is invalid and must be replaced by one uniform
  destination action: stamp a solid 3x3 block.
- Original support for X-source detection is no longer required for this combo, because the data
  generator must never create X or diamond inputs. The solver can therefore use the stronger
  assumption that every token it needs to act on is a maximal solid 3x3.
- The old track-mask subtraction still works, but only solid 3x3 pads need to be removed from the
  wire mask.

d) Placement legality
- Each channel still owns two disjoint 3x3 pads, one above and one below the network, with a
  one-cell gap to the endpoint. This guarantees no token/wire overlap.
- The generator places exactly one solid source per color on exactly one side, leaving the far pad
  empty in the input. The output places exactly one solid block on the opposite pad.
- Channel columns are unique and bottom destinations use a permutation, so no two transported
  blocks overlap.

e) Visual observability
- Across 2-4 examples, a human can see that every colored 3x3 block disappears from its original
  side and reappears intact at the routed far side. No partial remnant ever remains behind.
- Because every example uses only solid inputs, the learner does not need to infer hidden
  sub-rules for X or diamond sources.

f) Visual cleanliness
- Output contains only one token shape, the full 3x3 block, which makes the transformation much
  cleaner than the original mixed diamond/X endpoints.
- Wire colors 5/8/9 remain spatially separated from token colors, so the transported objects are
  easy to parse.

g) Change inferability
- There is no recoloring. Each token keeps its own color, and the wire topology is the only cue for
  where that color goes.
- The crucial combo-specific cue is quantitative and visible: every source loses all 9 colored
  cells, and every destination gains all 9 colored cells. That differs sharply from the original
  4-cell/5-cell split behavior.

Review of new solver assumptions
- `_find_solid_3x3` assumes all actionable tokens are maximal solid 3x3 blocks; the generator
  enforces solid-only inputs and keeps destination pads empty in the input.
- Endpoint matching assumes every source center is exactly two cells away from a degree-1 endpoint;
  the generator reuses the original endpoint spacing.
- Destination stamping assumes the far 3x3 pad is empty in the input and unique per channel; the
  generator guarantees that by placing one source per color on one side only.
"""

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_SOLID_OFF = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]

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


def _find_solid_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0, c0))} for maximal solid 3x3 blocks."""
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
                        inside = r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3
                        if 0 <= nr < h and 0 <= nc < w and not inside and int(g[nr, nc]) == col:
                            maximal = False
                            break
                    if not maximal:
                        break
                if not maximal:
                    break
            if maximal:
                found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=4)]


def _find_9rects(g: np.ndarray) -> list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]]:
    """Find color-9 rectangles and record wire contacts by face."""
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
                if (nr, nc) in cells or not (0 <= nr < h and 0 <= nc < w):
                    continue
                if int(g[nr, nc]) in (5, 8):
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
    Greedy wire following with 9-rect pass-through.

    Inside a color-9 rectangle, continue through the opposite face when possible; otherwise use the
    only other legal exit face.
    """
    cur, d, prev = start, dir0, None
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 4):
        if (cur, d) in visited:
            raise AssertionError("wire trace looped")
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
                others = [face for face in faces9 if face != entry_face]
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


def _stamp_solid(out: np.ndarray, center: tuple[int, int], color: int) -> None:
    cr, cc = center
    h, w = out.shape
    for dr, dc in _SOLID_OFF:
        rr, cc2 = cr + dr, cc + dc
        if 0 <= rr < h and 0 <= cc2 < w:
            out[rr, cc2] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Send every solid endpoint token through its wire and rebuild it only at the far end.

    Input structure:
    - Background cells fill the canvas; non-background cells form one routed wire network using
      colors 8, 5, and optional rectangular 9 blocks.
    - Every actionable token is a solid 3x3 block whose center sits exactly two cells outside a
      degree-1 wire endpoint. In this combo, generated inputs contain no X or diamond tokens.

    Rule:
    - Detect each solid 3x3 source block near an endpoint.
    - Erase the full 3x3 source footprint back to background.
    - Trace the wire to the opposite endpoint and stamp a full solid 3x3 block of the same color at
      the destination token center.

    Key insight:
    - The wires still define the endpoint pairing, but the old split-into-halves behavior is gone.
      Every source block moves as one intact 3x3 object, so source pads vanish and only far-end
      pads become occupied.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    solids = _find_solid_3x3(g, bg)

    track_mask = g != bg
    for _, (_, (r0, c0)) in solids.items():
        track_mask[r0:r0 + 3, c0:c0 + 3] = False

    comps = _components(track_mask)
    rects_9 = _find_9rects(g)

    out = g.copy()
    for _, (_, (r0, c0)) in solids.items():
        out[r0:r0 + 3, c0:c0 + 3] = bg

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
                if cen not in solids:
                    continue
                col, _ = solids[cen]
                dir_in = (-odr, -odc)
                end, dd = _trace_wire(comp, ep, dir_in, h, w, rects_9)
                dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                _stamp_solid(out, dest, col)

    return out


def _decompose_to_crossovers(perm: list[int]) -> list[tuple[int, int]]:
    """Decompose a permutation into crossover moves using the bypass column."""
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
) -> tuple[int, int, int, int] | None:
    """Attempt to place one color-9 rectangular block."""
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

    gap = RECT_MARGIN_GAP
    for pr0, pc0, pr1, pc1 in placed_rects:
        rows_overlap = rect_r0 <= pr1 + gap and pr0 <= rect_r1 + gap
        cols_overlap = rect_c0 <= pc1 + gap and pc0 <= rect_c1 + gap
        if rows_overlap and cols_overlap:
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
    return rect_r0, rect_c0, rect_r1, rect_c1


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
    """Draw the wire network onto grid. Returns wire ownership by bottom slot and bypass."""
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
            if c not in col_seg_start:
                continue
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
        if slot_wire[j] is None:
            continue
        c = ch_cols[j]
        if c in col_seg_start:
            start = col_seg_start[c] + 1
            if start <= bot_ep_row:
                grid[start:bot_ep_row + 1, c] = 8

    return slot_wire, bypass_wire


def _choose_source_sides(n_channels: int) -> list[bool]:
    """Return per-channel source side; True = top, False = bottom, with both sides represented."""
    n_top = random.randint(1, n_channels - 1)
    top_ids = set(random.sample(range(n_channels), n_top))
    sides = [i in top_ids for i in range(n_channels)]
    return sides


def _choose_layout_params(n_channels: int) -> tuple[int, int] | None:
    """Sample a channel spacing and left margin that fit the current channel count."""
    valid: list[tuple[int, int]] = []
    for spacing in CHANNEL_SPACING:
        for left_margin in range(LEFT_MARGIN_RANGE[0], LEFT_MARGIN_RANGE[1] + 1):
            first_col = left_margin + 1
            last_col = first_col + (n_channels - 1) * spacing
            bypass_col = last_col + BYPASS_COL_OFFSET
            width = max(bypass_col + 3, GRID_W_MIN)
            if width <= GRID_W_MAX:
                valid.append((spacing, left_margin))
    if not valid:
        return None
    return random.choice(valid)


def _derive_input_and_output(
    grid: np.ndarray,
    n_channels: int,
    ch_cols: list[int],
    perm: list[int],
    token_colors: list[int],
    top_token_row: int,
    bot_token_row: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Place one solid source token per channel and build the expected transported output."""
    inp = grid.copy()
    out = grid.copy()
    source_on_top = _choose_source_sides(n_channels)

    top_sources = 0
    bottom_sources = 0
    for i in range(n_channels):
        col = token_colors[i]
        top_center = (top_token_row, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[perm[i]])

        if source_on_top[i]:
            src, dst = top_center, bot_center
            top_sources += 1
        else:
            src, dst = bot_center, top_center
            bottom_sources += 1

        _stamp_token(inp, src[0], src[1], col, _SOLID_OFF)
        _stamp_token(out, dst[0], dst[1], col, _SOLID_OFF)

    stats = {
        "top_sources": top_sources,
        "bottom_sources": bottom_sources,
        "solid_sources": n_channels,
        "input_token_cells": 9 * n_channels,
        "output_token_cells": 9 * n_channels,
    }
    return inp, out, stats


def _gen_attempt(return_meta: bool = False):
    n_channels = random.choice(CHANNEL_COUNTS)
    layout = _choose_layout_params(n_channels)
    if layout is None:
        return None
    spacing, left_margin = layout
    first_col = left_margin + 1
    ch_cols = [first_col + i * spacing for i in range(n_channels)]
    bypass_col = ch_cols[-1] + BYPASS_COL_OFFSET

    width = max(bypass_col + 3, GRID_W_MIN)
    if width > GRID_W_MAX:
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
    slot_wire, bypass_wire = _build_wire_network(
        grid, n_channels, ch_cols, bypass_col, top_ep_row, bot_ep_row, crossovers, xover_rows
    )

    inv_perm = [0] * n_channels
    for i in range(n_channels):
        inv_perm[perm[i]] = i
    if any(slot_wire[j] != inv_perm[j] for j in range(n_channels)) or bypass_wire is not None:
        return None

    n_rects = random.choices(range(4), weights=RECT_COUNT_WEIGHTS)[0]
    placed_rects: list[tuple[int, int, int, int]] = []
    for _ in range(n_rects * RECT_RETRY_FACTOR):
        if len(placed_rects) >= n_rects:
            break
        result = _try_place_9rect(grid, ch_cols, xover_rows, top_ep_row, bot_ep_row, n_channels, placed_rects)
        if result is not None:
            placed_rects.append(result)

    inp, expected, stats = _derive_input_and_output(
        grid, n_channels, ch_cols, perm, token_colors, top_token_row, bot_token_row
    )
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    if return_meta:
        return {
            "input": inp,
            "expected": expected,
            "perm": perm,
            "n_channels": n_channels,
            "n_rects": len(placed_rects),
            **stats,
        }
    return inp


def generate_input():
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt(return_meta=False)
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
