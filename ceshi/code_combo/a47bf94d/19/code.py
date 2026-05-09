# Combo variant of a47bf94d (index 19)
# combo_spec: D7/0+D11/0
# Name: 距离编码停止+左右布局

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

# Design analysis for the fused variant:
# - Concrete behavior: keep the original token vocabulary (solid 3x3, X, diamond) but transpose the
#   wiring layout so channels run left-to-right. A source token still sits two cells outside a wire
#   endpoint. The new stop rule no longer targets the far endpoint: the emitted complementary token is
#   centered on the k-th wire cell reached from the source endpoint, where k is the visible colored-cell
#   count of the source token (solid=9, X=5).
# - Complexity calibration: only two travel distances appear, so the rule is visible and repeated across
#   samples instead of exploding into many arbitrary lengths. Each generated grid contains both solids and
#   X tokens, so both distances are supported by repeated evidence.
# - Broken implicit assumptions from the original solver: "the answer always sits outside the opposite
#   endpoint" and "tokens are mainly arranged top/bottom" are no longer valid. The new solver therefore
#   matches tokens to any endpoint at manhattan offset 2, traces the wire, and stops early on the traced
#   path instead of using the far endpoint.
# - Placement legality: the generator still uses the original permutation-via-crossovers construction,
#   but builds it in a transposed frame. Row spacing is kept at 4-5 so 3x3 emitted tokens cannot collide
#   across channels. Source tokens remain two cells off the endpoint, so they never overlap the wire.
# - Visual observability: left/right source placement makes horizontal transport immediately visible. A
#   human can count 5 colored cells in an X source versus 9 in a solid source and compare that with the
#   shorter/longer travel distance of the emitted complement.
# - Visual cleanliness: the generator omits optional color-9 rectangles. The solver still supports them,
#   but leaving them out keeps early wire-step counting legible and avoids hidden path length inside large
#   blocks.
# - New solver assumptions reviewed against generation: every colored token is isolated, every source is
#   exactly two cells from one endpoint, path length from any chosen source side exceeds 9, and at least
#   one source appears on each of the left and right sides. The generator enforces all of those.

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
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


def _find_9rects(g: np.ndarray) -> list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]]:
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
) -> list[tuple[int, int]]:
    """Follow one wire from an endpoint and return visited cells from source endpoint to destination endpoint."""
    cur, d, prev = start, dir0, None
    path = [start]
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
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
                        raise AssertionError("ambiguous 9-rect routing")
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
            return path

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
            return path

        prev = cur
        cur = p
        d = (dr, dc)
        path.append(cur)

    raise AssertionError("wire trace exceeded step limit")


def _stamp(out: np.ndarray, center: tuple[int, int], color: int, offsets: list[tuple[int, int]]) -> None:
    h, w = out.shape
    cr, cc = center
    for dr, dc in offsets:
        rr, cc2 = cr + dr, cc + dc
        if 0 <= rr < h and 0 <= cc2 < w:
            out[rr, cc2] = color


def _travel_distance(kind: str) -> int:
    return 9 if kind == "solid" else 5


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each source token in place, then send its complement partway along the connected pipe.

    Input structure:
    - Background cells are 0. Non-background cells form orthogonal wire networks (8 with optional 5/9 features)
      plus isolated 3x3 source tokens positioned exactly two cells outside wire endpoints.
    - The layout is left/right oriented in the generated data: rows act as channels and the pipes run mainly
      horizontally, although the tracing logic is orientation-agnostic.

    Rule:
    - A solid 3x3 source turns into a diamond at the same location and emits an X of the same color.
    - An X source stays as an X and emits a diamond of the same color.
    - The emitted complement does not go to the far endpoint. Instead it stops on the traced wire after a
      number of steps equal to the source token's colored-cell count: 9 for solid, 5 for X.

    Key insight:
    - The wire still tells you which path to follow, but the stopping point is encoded by the source shape
      itself. Counting visible pixels in the source token gives the travel distance.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)

    track_mask = g != bg
    for _, (_, (r0, c0)) in solids.items():
        track_mask[r0 : r0 + 3, c0 : c0 + 3] = False
    for cr, cc in xs.keys():
        for dr, dc in _X_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                track_mask[rr, cc2] = False

    comps = _components(track_mask)
    rects_9 = _find_9rects(g)

    out = g.copy()

    for (cr, cc), (col, (r0, c0)) in solids.items():
        out[r0 : r0 + 3, c0 : c0 + 3] = bg
        _stamp(out, (cr, cc), col, _DIAMOND_OFF)

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
                center = (er + 2 * odr, ec + 2 * odc)
                if not (0 <= center[0] < h and 0 <= center[1] < w):
                    continue

                if center in solids:
                    col, _ = solids[center]
                    path = _trace_wire_path(comp, ep, (-odr, -odc), h, w, rects_9)
                    dest = path[min(_travel_distance("solid"), len(path) - 1)]
                    _stamp(out, dest, col, _X_OFF)
                elif center in xs:
                    col, _ = xs[center]
                    path = _trace_wire_path(comp, ep, (-odr, -odc), h, w, rects_9)
                    dest = path[min(_travel_distance("x"), len(path) - 1)]
                    _stamp(out, dest, col, _DIAMOND_OFF)

    return out


CHANNEL_COUNTS = [4, 5]
CHANNEL_SPACING = [4, 5]
LEFT_MARGIN_RANGE = (2, 3)
BYPASS_COL_OFFSET = 2
TOKEN_ROW_OPTIONS = [2, 3]
TOKEN_COLORS_POOL = [1, 2, 3, 4, 6, 7]
FINAL_H_RANGE = (20, 26)
FINAL_W_RANGE = (24, 30)
EP_TOKEN_GAP = 2
GEN_MAX_RETRIES = 300


def _decompose_to_crossovers(perm: list[int]) -> list[tuple[int, int]]:
    """Decompose a permutation into sequential crossover moves using one bypass lane."""
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

    for j in range(n):
        if slot_wire[j] is not None:
            c = ch_cols[j]
            if c in col_seg_start:
                start = col_seg_start[c] + 1
                if start <= bot_ep_row:
                    grid[start:bot_ep_row + 1, c] = 8

    return slot_wire, bypass_wire


def _stamp_token(grid: np.ndarray, center_r: int, center_c: int, color: int, offsets: list[tuple[int, int]]) -> None:
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = color


def _derive_input_base(
    grid: np.ndarray,
    n: int,
    ch_cols: list[int],
    perm: list[int],
    token_colors: list[int],
    top_token_row: int,
    bot_token_row: int,
) -> np.ndarray:
    inp = grid.copy()
    n_x = random.choice([1, 2])
    x_channels = set(random.sample(range(n), min(n_x, n)))

    sides = [random.choice(["top", "bottom"]) for _ in range(n)]
    if all(side == "top" for side in sides):
        sides[random.randrange(n)] = "bottom"
    if all(side == "bottom" for side in sides):
        sides[random.randrange(n)] = "top"

    for i in range(n):
        center_top = (top_token_row, ch_cols[i])
        center_bot = (bot_token_row, ch_cols[perm[i]])
        center = center_top if sides[i] == "top" else center_bot
        offsets = _X_OFF if i in x_channels else _SOLID_OFF
        _stamp_token(inp, center[0], center[1], token_colors[i], offsets)
    return inp


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(CHANNEL_COUNTS)
    spacing = random.choice(CHANNEL_SPACING)

    final_h = random.randint(*FINAL_H_RANGE)
    base_w = final_h

    left_margin = random.randint(*LEFT_MARGIN_RANGE)
    first_col = left_margin + 1
    ch_cols = [first_col + i * spacing for i in range(n)]
    bypass_col = ch_cols[-1] + BYPASS_COL_OFFSET

    if bypass_col + 2 >= base_w:
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

    min_base_h = top_ep_row + (n_xovers * 2 + 2) + top_token_row + 3
    if min_base_h > FINAL_W_RANGE[1]:
        return None
    base_h = random.randint(max(FINAL_W_RANGE[0], min_base_h), FINAL_W_RANGE[1])

    bot_token_row = base_h - top_token_row - 1
    bot_ep_row = bot_token_row - EP_TOKEN_GAP
    if bot_ep_row - top_ep_row < n_xovers * 2 + 2:
        return None

    avail = bot_ep_row - top_ep_row - 2
    step = max(2, avail // (n_xovers + 1))
    xover_rows = [top_ep_row + 1 + step * (s + 1) for s in range(n_xovers)]
    if xover_rows[-1] >= bot_ep_row:
        xover_rows = [top_ep_row + 2 + s * 2 for s in range(n_xovers)]
    if xover_rows[-1] >= bot_ep_row:
        return None

    base_grid = np.zeros((base_h, base_w), dtype=int)
    slot_wire, bypass_wire = _build_wire_network(
        base_grid, n, ch_cols, bypass_col, top_ep_row, bot_ep_row, crossovers, xover_rows
    )

    inv_perm = [0] * n
    for i in range(n):
        inv_perm[perm[i]] = i
    if any(slot_wire[j] != inv_perm[j] for j in range(n)) or bypass_wire is not None:
        return None

    base_input = _derive_input_base(base_grid, n, ch_cols, perm, token_colors, top_token_row, bot_token_row)
    final_input = base_input.T

    try:
        out = main(final_input)
    except Exception:
        return None

    if out.shape != final_input.shape:
        return None

    # Reject degenerate cases where all emitted distances would look identical.
    bg = get_bg(final_input)
    solids = _find_solid_3x3(final_input, bg)
    xs = _find_x_3x3(final_input, bg)
    if not solids or not xs:
        return None

    return final_input


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
