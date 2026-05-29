# Variant of a47bf94d
# Dimension: D5 — 形状补全规则 (transform)
# Change: 形状补全（shape_completion）——如果管道一端已存在十字形或四角星形，则在另一端放置互补形状（四角星对应十... → 擦除源+完整传送

"""
Design analysis for the D5 variant

a) Concretization and behavior trace
- The original puzzle has one orthogonal wire network connecting top endpoints to bottom endpoints.
  A token center sits exactly 2 cells away from an endpoint, so each token occupies a clean 3x3
  pad separated from the wire by a 1-cell gap.
- In the original task, a solid 3x3 source token at the top is split into two visible remnants:
  a 4-cell diamond stays at the source side and an X is drawn at the far endpoint. X-input channels
  similarly keep the X at the source and only place the complementary diamond at the far end.
- In this variant, the token is transported as a whole. Example:
  source = solid 3x3 of color 2 above top endpoint i, wire routes to bottom endpoint perm[i].
  Original output: top keeps a diamond of color 2, bottom gets an X of color 2.
  Variant output: the entire top 3x3 disappears to background, and the bottom destination becomes
  a full solid 3x3 of color 2. No half-shape remains anywhere.
- This changes the token-processing stage only: detect source token -> erase source footprint ->
  trace the wire -> stamp a full 3x3 block at the far token center.

b) Complexity calibration
- The routing complexity stays intact: 4-5 channels, non-identity permutations, crossovers with
  5-markers, and embedded 9-rect pass-through blocks remain unchanged.
- The token rule is simplified to a single visible transport rule in the generated data so that
  2-4 ARC examples are enough to infer it. The solver still recognizes X-sources for robustness,
  but the generator uses solid sources only, avoiding a hidden branch that might appear in test
  without appearing in train.
- This is not too simple visually: several colored 3x3 blocks vanish from mixed top/bottom source
  positions and reappear at routed far endpoints, so the permutation and the “erase then move”
  behavior remain both observable.

c) Old implicit assumptions that must be removed
- Old code path “solid tokens become diamonds in place” relied on the invariant that a source
  always leaves a local remnant. This is false now, so the source footprint must be erased fully.
- Old routing branch “solid emits X” and “X emits diamond” assumed the far side always receives a
  complementary half-shape. This is false now; both branches must emit the same destination shape:
  a complete solid 3x3.
- The track-mask logic that subtracts token cells from the wire remains valid, because the physical
  placement of tokens relative to endpoints is unchanged.

d) Placement legality
- Generator keeps the original safe geometry: top token row is 2 or 3, bottom token row is
  H-top_token_row-1, and endpoint rows are 2 cells inward from the token centers.
- Every source token is placed on exactly one endpoint pad per color, never on both ends.
- Destination pads are left empty in the input, so the transported full 3x3 never overlaps another
  token. Distinct channel colors stay distinct and channel destinations remain disjoint.
- Wire rectangles of color 9 are still inserted only inside already-drawn wires and must preserve
  legal pass-through structure.

e) Visual observability
- The new rule is easy to see: source 3x3 blocks disappear entirely, while full 3x3 blocks appear
  at the opposite wire ends. This is visibly different from the original half-at-source / half-at-
  destination behavior.
- No color-based hidden ordering is introduced. Color is preserved exactly along the wire.

f) Visual cleanliness
- Tokens remain one cell away from wires, so wire colors (5, 8, 9) do not touch token colors.
- Output has only one token shape, the solid 3x3, which makes the “whole block teleports” effect
  visually crisp rather than cluttered.

g) Color inference
- There is no recoloring rule. Each transported block keeps its own source color, and the wire
  topology provides the structure needed to infer where that color should end up.

Review of new solver assumptions
- Each wire component still has degree-1 endpoints; generator constructs the network that way.
- A source token center must be exactly 2 cells outward from an endpoint in a cardinal direction;
  generator guarantees this by reusing the original layout.
- A destination 3x3 area is empty in the input; generator guarantees one source per color and no
  pre-placed far-end token, so stamping the transported solid does not destroy valid content.
"""

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID_OFF = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]


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


def _find_x_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for isolated 3x3 X patterns."""
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
                    inside = r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3
                    if 0 <= nr < h and 0 <= nc < w and not inside and int(g[nr, nc]) == col:
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
    """Find color-9 rectangular blocks and their wire connections per face."""
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

    At a color-9 rectangle: exit from the opposite face when present; otherwise use the only
    remaining legal exit face.
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


def _erase_x(out: np.ndarray, center: tuple[int, int], bg: int) -> None:
    cr, cc = center
    h, w = out.shape
    for dr, dc in _X_OFF:
        rr, cc2 = cr + dr, cc + dc
        if 0 <= rr < h and 0 <= cc2 < w:
            out[rr, cc2] = bg


def _stamp_solid(out: np.ndarray, center: tuple[int, int], color: int) -> None:
    cr, cc = center
    h, w = out.shape
    for dr, dc in _SOLID_OFF:
        rr, cc2 = cr + dr, cc + dc
        if 0 <= rr < h and 0 <= cc2 < w:
            out[rr, cc2] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each endpoint token through its wire path and rebuild it as a full 3x3 block at the far end.

    Input structure:
    - Background cells are 0-like majority color; non-background cells form one routed wire network
      with colors 8, 5, and optional rectangular 9 blocks.
    - Colored tokens sit near wire endpoints with their centers exactly two cells outside an endpoint.
      A source token can be a solid 3x3 block or an isolated 3x3 X pattern.

    Rule:
    - Detect each source token adjacent to a wire endpoint.
    - Erase the source token completely, leaving only background at its original location.
    - Follow the wire to the opposite endpoint and draw a full solid 3x3 block of the same color at
      the destination token center.

    Key insight:
    - The wire geometry still determines the endpoint pairing, but the token is no longer split into
      complementary halves. The whole object disappears from the source side and reappears intact at
      the far side.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)

    track_mask = g != bg
    for _, (_, (r0, c0)) in solids.items():
        track_mask[r0:r0 + 3, c0:c0 + 3] = False
    for cr, cc in xs:
        for dr, dc in _X_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                track_mask[rr, cc2] = False

    comps = _components(track_mask)
    rects_9 = _find_9rects(g)

    out = g.copy()

    for _, (_, (r0, c0)) in solids.items():
        out[r0:r0 + 3, c0:c0 + 3] = bg
    for center in xs:
        _erase_x(out, center, bg)

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
                elif cen in xs:
                    col, _ = xs[cen]
                else:
                    continue

                dir_in = (-odr, -odc)
                end, dd = _trace_wire(comp, ep, dir_in, h, w, rects_9)
                dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                _stamp_solid(out, dest, col)

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
    """Return per-channel source side; True = top, False = bottom. Force both sides to appear."""
    sides = [bool(random.getrandbits(1)) for _ in range(n_channels)]
    if all(sides):
        sides[random.randrange(n_channels)] = False
    elif not any(sides):
        sides[random.randrange(n_channels)] = True
    return sides


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

    stats = {"top_sources": top_sources, "bottom_sources": bottom_sources}
    return inp, out, stats


def _gen_attempt(return_meta: bool = False):
    n_channels = random.choice(CHANNEL_COUNTS)
    spacing = random.choice(CHANNEL_SPACING)

    left_margin = random.randint(*LEFT_MARGIN_RANGE)
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
