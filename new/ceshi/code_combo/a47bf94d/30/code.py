# Combo variant of a47bf94d (index 30)
# combo_spec: D11/0+D15/0
# Name: 左右布局+裁剪输出

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

# Design analysis for the fused variant:
# - Behavior trace:
#   In the base task, each channel runs top-to-bottom, with a token center sitting
#   exactly 2 cells outside a degree-1 endpoint. A solid token turns into a diamond
#   in place and emits an X to the far endpoint; an X token emits a diamond.
#   Here the same local token/wire rule is preserved, but the whole construction is
#   rotated so channels run left-to-right. The solver therefore should not hardcode
#   "top" or "bottom"; it should only rely on endpoint geometry. After all routing
#   and token conversion, the output is cropped to the tight bounding box of all
#   non-background cells, so the returned grid keeps only the pipe band and results.
# - Complexity calibration:
#   The sample still uses 4-5 channels, a non-identity permutation, crossings, and
#   optional color-9 pass-through blocks. This keeps the task visually distinct from
#   the original while avoiding extra rule branches. The crop is a single global
#   post-process, so the human-visible change is strong but still learnable.
# - Hidden assumptions reviewed:
#   `_find_solid_3x3` and `_find_x_3x3` assume tokens are isolated 3x3 motifs; the
#   generator preserves that. Endpoint matching assumes a token center is 2 cells in
#   a cardinal direction from a degree-1 wire endpoint; the rotated generator keeps
#   exactly that spacing. `_trace_wire` assumes each 9-rectangle exposes valid faces
#   and wires do not branch ambiguously; the canonical generator already guarantees
#   this, and transposition preserves it. The new crop assumes there are no stray
#   non-bg marks outside the intended pipe region, which the generator enforces.
# - Placement legality and cleanliness:
#   The generator first builds a legal top/bottom canonical instance, verifies it
#   against the solver, and only then transposes it into the left/right variant.
#   Transposition preserves non-overlap, endpoint distances, and rectangle geometry.
#   To make the cropped result visibly wide, the canonical instance is generated
#   taller than it is wide; after transposition the retained pipe band becomes a
#   wide rectangle with left/right tokens at its extremes.
# - Observability / inferability:
#   Across examples, tokens clearly appear on the left and right sides, wires run
#   mostly horizontally, and the output border hugs the structure on all four sides.
#   Those repeated cues are enough for a human to infer both the layout change and
#   the "crop to the active area" rule from a few samples.

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID_OFF = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]


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


def _find_x_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0, c0))} for isolated 3x3 X patterns."""
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
    """Find color-9 rectangles and the wire-touch cells on each face."""
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
                if (nr, nc) in cells:
                    continue
                if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) in (5, 8):
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

    At a color-9 rectangle, exit from the opposite face when possible.
    If opposite is absent, exactly one other face must be available.
    """
    cur, d, prev = start, dir0, None
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 4):
        state = (cur, d)
        if state in visited:
            raise AssertionError("wire trace looped")
        visited.add(state)

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
            return cur, d
        prev = cur
        cur = p
        d = (dr, dc)

    raise AssertionError("wire trace exceeded step limit")


def _crop_non_bg(g: np.ndarray, bg: int) -> np.ndarray:
    pts = np.argwhere(g != bg)
    if pts.size == 0:
        return g.copy()
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return g[int(r0):int(r1) + 1, int(c0):int(c1) + 1].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route token complements through a sideways pipe network, then crop to the active band.

    Input structure:
    - Background cells surround a left/right wire system made of colors 8, 5, and optional 9-rectangles.
    - Colored 3x3 tokens sit two cells outside degree-1 wire endpoints on the left or right edge of each channel.

    Rule:
    - Every solid 3x3 token is reduced in place to a diamond.
    - Any token at an endpoint sends its complementary pattern to the opposite endpoint through the wire network.
    - After all transmissions, discard all outer blank rows and columns and return only the tight bounding box containing wires and resulting tokens.

    Key insight:
    - The token logic is still local and direction-agnostic: identify endpoints, trace the matching wire component, place the complementary shape two cells beyond the far endpoint, then crop the final non-background footprint.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)

    track_mask = g != bg
    for _center, (_col, (r0, c0)) in solids.items():
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
                    end, dd = _trace_wire(comp, ep, (-odr, -odc), h, w, rects_9)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    for dr, dc in _X_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col

                elif cen in xs:
                    col, _ = xs[cen]
                    if col in comp_colors:
                        continue
                    end, dd = _trace_wire(comp, ep, (-odr, -odc), h, w, rects_9)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    for dr, dc in _DIAMOND_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col

    return _crop_non_bg(out, bg)


# Canonical generator constants. We build a tall top/bottom instance first and then
# transpose it into the left/right combo so the pipe band becomes wide after crop.
CHANNEL_COUNTS = [4, 5]
CHANNEL_SPACING = [4, 5]
LEFT_MARGIN_RANGE = (2, 3)
BYPASS_COL_OFFSET = 2
TOKEN_ROW_OPTIONS = [2, 3]
TOKEN_COLORS_POOL = [1, 2, 3, 4, 6, 7]
CANON_GRID_H_RANGE = (28, 36)
CANON_GRID_W_MIN = 24
CANON_GRID_W_MAX = 29
EP_TOKEN_GAP = 2

RECT_COUNT_WEIGHTS = [10, 40, 30, 20]
RECT_WIRE_COUNT_WEIGHTS = [50, 38, 12]
RECT_PARALLEL_PROB = 0.75
RECT_H_RANGE = (2, 5)
RECT_MAX_WIDTH = 7
RECT_RETRY_FACTOR = 5
RECT_MARGIN_GAP = 1

# Keep both token-interpretation branches visible in almost every task instance:
# each sample still has mostly solid tokens, but now also includes 1-2 X tokens
# so humans can infer the X -> diamond transmission rule from 2-4 examples.
X_INPUT_COUNT_WEIGHTS = [0, 65, 35]
FAR_DIAMOND_PROB = 0.33
GEN_MAX_RETRIES = 400


def _decompose_to_crossovers(perm: list[int]) -> list[tuple[int, int]]:
    """Decompose a permutation into crossover moves; -1 denotes the bypass column."""
    n = len(perm)
    visited = [False] * n
    crossovers = []
    for start in range(n):
        if visited[start] or perm[start] == start:
            visited[start] = True
            continue
        cycle = []
        cur = start
        while not visited[cur]:
            visited[cur] = True
            cycle.append(cur)
            cur = perm[cur]
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
    """Attempt to place one legal color-9 block into the canonical vertical network."""
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
        margin_l = random.randint(0, 1)
        margin_r = random.randint(0, 1)
    else:
        j0 = random.randint(0, n_channels - 1)
        c_lo = c_hi = ch_cols[j0]
        margin_l = random.randint(0, 2)
        margin_r = random.randint(0, 2)
        if margin_l + margin_r < 1:
            if random.random() < 0.5:
                margin_l = 1
            else:
                margin_r = 1

    rect_c0 = c_lo - margin_l
    rect_c1 = c_hi + margin_r
    if rect_c1 - rect_c0 + 1 > RECT_MAX_WIDTH or rect_c0 < 0 or rect_c1 >= w:
        return None

    max_rect_h = min(RECT_H_RANGE[1], bot_ep_row - top_ep_row - 4)
    if max_rect_h < RECT_H_RANGE[0]:
        return None
    rect_h = random.randint(RECT_H_RANGE[0], max_rect_h)
    if mode == "crossing" and xover_rows:
        xr = random.choice(xover_rows)
        rect_r0 = xr - random.randint(0, rect_h - 1)
    else:
        rect_r0 = random.randint(top_ep_row + 2, bot_ep_row - rect_h - 1)
    rect_r1 = rect_r0 + rect_h - 1

    wire_cols = sorted(c for c in ch_cols if rect_c0 <= c <= rect_c1)
    if not wire_cols:
        return None
    for c in wire_cols:
        if not np.all(grid[rect_r0:rect_r1 + 1, c] == 8):
            return None

    sub = grid[rect_r0:rect_r1 + 1, rect_c0:rect_c1 + 1]
    if not np.all((sub == 0) | (sub == 5) | (sub == 8)):
        return None

    for pr0, pc0, pr1, pc1 in placed_rects:
        overlap_rows = rect_r0 <= pr1 + RECT_MARGIN_GAP and pr0 <= rect_r1 + RECT_MARGIN_GAP
        overlap_cols = rect_c0 <= pc1 + RECT_MARGIN_GAP and pc0 <= rect_c1 + RECT_MARGIN_GAP
        if overlap_rows and overlap_cols:
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
    """Draw the canonical vertical wire network implementing the permutation."""
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
        if slot_wire[j] is None:
            continue
        c = ch_cols[j]
        if c in col_seg_start:
            start = col_seg_start[c] + 1
            if start <= bot_ep_row:
                grid[start:bot_ep_row + 1, c] = 8

    return slot_wire, bypass_wire


def _derive_input(
    grid: np.ndarray,
    n_channels: int,
    ch_cols: list[int],
    perm: list[int],
    token_colors: list[int],
    top_token_row: int,
    bot_token_row: int,
) -> np.ndarray:
    """Place input tokens on the canonical top/bottom layout."""
    inp = grid.copy()
    n_x = random.choices([0, 1, 2], weights=X_INPUT_COUNT_WEIGHTS)[0]
    x_channels = set(random.sample(range(n_channels), min(n_x, n_channels)))

    for i in range(n_channels):
        col = token_colors[i]
        top_center = (top_token_row, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[perm[i]])
        if i in x_channels:
            if random.choice([True, False]):
                near, far = top_center, bot_center
            else:
                near, far = bot_center, top_center
            _stamp_token(inp, near[0], near[1], col, _X_OFF)
            if random.random() < FAR_DIAMOND_PROB:
                _stamp_token(inp, far[0], far[1], col, _DIAMOND_OFF)
        else:
            _stamp_token(inp, *(random.choice([top_center, bot_center])), col, _SOLID_OFF)
    return inp


def _gen_attempt() -> np.ndarray | None:
    n_channels = random.choice(CHANNEL_COUNTS)
    spacing = random.choice(CHANNEL_SPACING)

    left_margin = random.randint(*LEFT_MARGIN_RANGE)
    first_col = left_margin + 1
    ch_cols = [first_col + i * spacing for i in range(n_channels)]
    bypass_col = ch_cols[-1] + BYPASS_COL_OFFSET

    canon_w = max(bypass_col + 3, CANON_GRID_W_MIN)
    if canon_w > CANON_GRID_W_MAX:
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
    if h_min > CANON_GRID_H_RANGE[1]:
        return None
    canon_h = random.randint(max(CANON_GRID_H_RANGE[0], h_min), CANON_GRID_H_RANGE[1])

    bot_token_row = canon_h - top_token_row - 1
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

    grid = np.zeros((canon_h, canon_w), dtype=int)
    slot_wire, bypass_wire = _build_wire_network(
        grid,
        n_channels,
        ch_cols,
        bypass_col,
        top_ep_row,
        bot_ep_row,
        crossovers,
        xover_rows,
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

    output = grid.copy()
    for i in range(n_channels):
        _stamp_token(output, top_token_row, ch_cols[i], token_colors[i], _DIAMOND_OFF)
        _stamp_token(output, bot_token_row, ch_cols[perm[i]], token_colors[i], _X_OFF)

    inp = _derive_input(grid, n_channels, ch_cols, perm, token_colors, top_token_row, bot_token_row)

    transposed_inp = inp.T.copy()
    transposed_out = _crop_non_bg(output.T.copy(), 0)
    out_h, out_w = transposed_out.shape
    if out_w <= out_h:
        return None
    if transposed_out.shape == transposed_inp.shape:
        return None
    try:
        if np.array_equal(main(transposed_inp), transposed_out):
            return transposed_inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
