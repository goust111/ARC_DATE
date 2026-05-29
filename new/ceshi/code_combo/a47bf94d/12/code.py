# Combo variant of a47bf94d (index 12)
# combo_spec: D10/1+D13/0
# Name: 条件置换+链式反应

from __future__ import annotations

import random
import sys

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import components_mask, get_bg, neighbors4


# Design notes for the fused variant:
# - Concrete behavior: each input color starts as one top-layer solid 3x3 token. It first travels
#   through the upper wire band using a solid-specific permutation P_s, arrives at the middle layer
#   as an X token, then immediately continues through the lower wire band using an X-specific
#   permutation P_x, and finally lands as a solid token at the bottom layer.
# - Example trace for channel i:
#   top solid at col i -> top diamond at col i;
#   send X to middle col P_s(i) -> keep that X in place;
#   send solid to bottom col P_x(P_s(i))) -> stop there.
# - Why this is visually learnable: every sample repeats the same three-stage motif for 4-5 colors,
#   so the solver can compare many same-structure traces inside one grid. The middle layer is no
#   longer empty; it visibly records the first hop as X, making the chain reaction observable.
# - Complexity calibration: only two token types control routing (solid vs X), and each sample uses
#   both because every top solid turns into a middle X before the second hop. We keep permutations
#   short (single transpositions) so wires stay readable while still differing across the two bands.
# - Assumptions audited against main():
#   token centers are exactly 2 cells from wire endpoints; each relay center touches at most two
#   bands (one above, one below); after excluding the incoming endpoint, a relay has at most one
#   outgoing endpoint; no two colors share a center because both permutations are bijections.
#   generate_input() enforces all of these constraints.
# - Placement legality: token rows are separated from endpoints by 2 rows, bands are disjoint in
#   row range, and width/spacing keep all 3x3 tokens in-bounds.

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_DIAMOND = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]

_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DIR2FACE = {(1, 0): "N", (-1, 0): "S", (0, 1): "W", (0, -1): "E"}
_FACE2DIR = {"S": (1, 0), "N": (-1, 0), "E": (0, 1), "W": (0, -1)}

# --- Layout choices for the fused generator ---
CHANNEL_COUNTS = [4, 5]
TOKEN_COLORS_POOL = [1, 2, 3, 4, 6, 7]
SPACING = 4
TOP_TOKEN_ROW = 2
MID_TOKEN_ROW = 12
BOT_TOKEN_ROW = 23
TOP_EP_ROW = TOP_TOKEN_ROW + 2
MID_IN_EP_ROW = MID_TOKEN_ROW - 2
MID_OUT_EP_ROW = MID_TOKEN_ROW + 2
BOT_IN_EP_ROW = BOT_TOKEN_ROW - 2
GRID_H = 26
GRID_W = 25
BYPASS_COL_OFFSET = 2
GEN_MAX_RETRIES = 300


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
            sub = g[r0:r0 + 3, c0:c0 + 3]
            col = int(sub[0, 0])
            if col == bg:
                continue
            if np.any(sub[mask] != col) or np.any(sub[~mask] != bg):
                continue
            isolated = True
            for dr0, dc0 in _X:
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


def _trace_wire(
    comp: set[tuple[int, int]],
    start: tuple[int, int],
    dir0: tuple[int, int],
    h: int,
    w: int,
    rects_9: list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Follow one wire component from one endpoint to the other."""
    cur, d, prev = start, dir0, None
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
                        raise AssertionError("bad 9-rect routing")
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
            p, dr, dc = sorted(perp if perp else opts2, key=lambda x: (x[1], x[2]))[0]

        if p == prev:
            return cur, d
        prev = cur
        cur = p
        d = (dr, dc)

    raise AssertionError("wire trace exceeded step limit")


def _stamp_token(grid: np.ndarray, center_r: int, center_c: int, color: int, offsets: list[tuple[int, int]]) -> None:
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = color


def _clear_token_box(grid: np.ndarray, center: tuple[int, int], bg: int) -> None:
    r, c = center
    grid[r - 1:r + 2, c - 1:c + 2] = bg


def _write_shape(grid: np.ndarray, center: tuple[int, int], color: int, shape: str, bg: int) -> None:
    _clear_token_box(grid, center, bg)
    if shape == "diamond":
        _stamp_token(grid, center[0], center[1], color, _DIAMOND)
    elif shape == "x":
        _stamp_token(grid, center[0], center[1], color, _X)
    elif shape == "solid":
        _stamp_token(grid, center[0], center[1], color, _SOLID)
    else:
        raise ValueError(shape)


def _decompose_to_crossovers(perm: list[int]) -> list[tuple[int, int]]:
    """Decompose a permutation into sequential crossover moves using a bypass column."""
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
    """Draw one permutation band onto grid."""
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


def _choose_cross_rows(top_ep_row: int, bot_ep_row: int, num_cross: int) -> list[int]:
    avail = list(range(top_ep_row + 1, bot_ep_row))
    if num_cross > len(avail):
        raise ValueError("not enough rows for crossovers")
    if num_cross == 0:
        return []
    if num_cross == 1:
        return [avail[len(avail) // 2]]
    idxs = [round(i * (len(avail) - 1) / (num_cross - 1)) for i in range(num_cross)]
    return [avail[i] for i in idxs]


def _candidate_permutations(n: int) -> list[list[int]]:
    perms = []
    for i in range(n):
        for j in range(i + 1, n):
            perm = list(range(n))
            perm[i], perm[j] = perm[j], perm[i]
            perms.append(perm)
    return perms


def _extract_perm_from_band(g: np.ndarray, top_ep_row: int, bot_ep_row: int, ch_cols: list[int]) -> list[int]:
    """Recover the forward source->destination permutation from one generated wire band."""
    n = len(ch_cols)
    w = g.shape[1]
    wire_vals = {5, 8, 9}
    seen_cols = {
        c
        for r in range(top_ep_row, bot_ep_row + 1)
        for c in range(w)
        if int(g[r, c]) in wire_vals
    }
    extra_cols = sorted(c for c in seen_cols if c not in ch_cols)
    bypass_col = max(extra_cols) if extra_cols else None

    slot_wire: list[int | None] = list(range(n))
    bypass_wire: int | None = None

    for r in range(top_ep_row + 1, bot_ep_row):
        horiz_cols = []
        for c in range(w):
            if int(g[r, c]) not in wire_vals:
                continue
            left = c - 1 >= 0 and int(g[r, c - 1]) in wire_vals
            right = c + 1 < w and int(g[r, c + 1]) in wire_vals
            if left or right:
                horiz_cols.append(c)
        if not horiz_cols:
            continue

        src_candidates = []
        dst_candidates = []
        for c in horiz_cols:
            above = int(g[r - 1, c]) in wire_vals
            below = int(g[r + 1, c]) in wire_vals
            if above and not below:
                src_candidates.append(c)
            if below and not above:
                dst_candidates.append(c)
        if len(src_candidates) == 1 and len(dst_candidates) == 1:
            src_col, dst_col = src_candidates[0], dst_candidates[0]
        else:
            src_col, dst_col = min(horiz_cols), max(horiz_cols)
        src_slot = -1 if src_col == bypass_col else ch_cols.index(src_col)
        dst_slot = -1 if dst_col == bypass_col else ch_cols.index(dst_col)

        if src_slot == -1:
            slot_wire[dst_slot] = bypass_wire
            bypass_wire = None
        elif dst_slot == -1:
            bypass_wire = slot_wire[src_slot]
            slot_wire[src_slot] = None
        else:
            slot_wire[dst_slot] = slot_wire[src_slot]
            slot_wire[src_slot] = None

    if bypass_wire is not None or any(src is None for src in slot_wire):
        raise AssertionError("failed to decode band permutation")

    forward = [0] * n
    for dest_idx, src_idx in enumerate(slot_wire):
        forward[int(src_idx)] = dest_idx
    return forward


def _band_exists(g: np.ndarray, top_ep_row: int, bot_ep_row: int, ch_cols: list[int]) -> bool:
    wire_vals = {5, 8, 9}
    return any(int(g[top_ep_row, c]) in wire_vals or int(g[bot_ep_row, c]) in wire_vals for c in ch_cols)


def _endpoint_cols(g: np.ndarray, row: int) -> list[int]:
    return [c for c in range(g.shape[1]) if int(g[row, c]) == 8]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Relay each colored token through two stacked pipe bands, switching the routing rule after the first hop.

    Input structure:
    - Background is the most common color. The grid contains two separated wire bands built from 8/5/9 cells.
    - Colored 3x3 tokens sit 2 cells away from wire endpoints; generated inputs start with top-layer solid blocks.

    Rule:
    - A solid token fires through the adjacent band using the solid-routing permutation, turns into a diamond in place,
      and creates an X token at the far endpoint.
    - If that far endpoint also connects to the lower band, the new X token immediately fires again using the X-routing
      permutation, remains an X in place, and creates a solid block at the next endpoint. The chain stops when no new
      outgoing band remains.

    Key insight:
    - The same color is routed twice because its type changes after the first hop: solid uses the first permutation,
      then the relay X uses a different permutation in the next band.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    solids = _find_solid_3x3(g, bg)
    x_tokens = _find_x_3x3(g, bg)
    out = g.copy()
    top_tokens = sorted(
        [(center, col) for center, (col, _box) in solids.items() if center[0] == TOP_TOKEN_ROW],
        key=lambda x: x[0][1],
    )
    relay_tokens: dict[int, int] = {
        center[1]: color
        for center, (color, _box) in x_tokens.items()
        if center[0] == MID_TOKEN_ROW
    }
    if top_tokens:
        ch_cols = [center[1] for center, _col in top_tokens]
    elif relay_tokens:
        ch_cols = _endpoint_cols(g, MID_OUT_EP_ROW)
    else:
        return out

    perm_s = _extract_perm_from_band(g, TOP_EP_ROW, MID_IN_EP_ROW, ch_cols) if top_tokens else None
    has_lower_band = _band_exists(g, MID_OUT_EP_ROW, BOT_IN_EP_ROW, ch_cols)
    perm_x = _extract_perm_from_band(g, MID_OUT_EP_ROW, BOT_IN_EP_ROW, ch_cols) if has_lower_band else None
    relay_tokens = {col: color for col, color in relay_tokens.items() if col in ch_cols}

    for src_idx, (center, color) in enumerate(top_tokens):
        _write_shape(out, center, color, "diamond", bg)
        mid_idx = perm_s[src_idx]
        relay_tokens[ch_cols[mid_idx]] = color

    for mid_col, color in relay_tokens.items():
        _write_shape(out, (MID_TOKEN_ROW, mid_col), color, "x", bg)
        if perm_x is None:
            continue
        mid_idx = ch_cols.index(mid_col)
        bot_idx = perm_x[mid_idx]
        _write_shape(out, (BOT_TOKEN_ROW, ch_cols[bot_idx]), color, "solid", bg)

    return out


def _pick_permutations(n: int) -> tuple[list[int], list[int]]:
    candidates = _candidate_permutations(n)
    random.shuffle(candidates)
    for perm_s in candidates:
        for perm_x in candidates:
            if perm_x == perm_s:
                continue
            after1 = perm_s
            after2 = [perm_x[perm_s[i]] for i in range(n)]
            moved_second = sum(after2[i] != after1[i] for i in range(n))
            if moved_second >= 2:
                return perm_s, perm_x
    raise RuntimeError("failed to choose permutations")


def _build_expected_output(
    grid: np.ndarray,
    ch_cols: list[int],
    perm_s: list[int],
    perm_x: list[int],
    token_colors: list[int],
) -> np.ndarray:
    out = grid.copy()
    bg = int(get_bg(grid))
    for i, col in enumerate(token_colors):
        _write_shape(out, (TOP_TOKEN_ROW, ch_cols[i]), col, "diamond", bg)
        mid_idx = perm_s[i]
        bot_idx = perm_x[mid_idx]
        _write_shape(out, (MID_TOKEN_ROW, ch_cols[mid_idx]), col, "x", bg)
        _write_shape(out, (BOT_TOKEN_ROW, ch_cols[bot_idx]), col, "solid", bg)
    return out


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(CHANNEL_COUNTS)
    first_col = 4 if n == 5 else random.choice([4, 5, 6])
    ch_cols = [first_col + i * SPACING for i in range(n)]
    if ch_cols[-1] + 2 >= GRID_W:
        return None
    bypass_col = ch_cols[-1] + BYPASS_COL_OFFSET

    perm_s, perm_x = _pick_permutations(n)
    cross_s = _decompose_to_crossovers(perm_s)
    cross_x = _decompose_to_crossovers(perm_x)
    if len(cross_s) > 5 or len(cross_x) > 3:
        return None

    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    rows_s = _choose_cross_rows(TOP_EP_ROW, MID_IN_EP_ROW, len(cross_s))
    rows_x = _choose_cross_rows(MID_OUT_EP_ROW, BOT_IN_EP_ROW, len(cross_x))
    slot_s, bypass_s = _build_wire_network(grid, n, ch_cols, bypass_col, TOP_EP_ROW, MID_IN_EP_ROW, cross_s, rows_s)
    slot_x, bypass_x = _build_wire_network(grid, n, ch_cols, bypass_col, MID_OUT_EP_ROW, BOT_IN_EP_ROW, cross_x, rows_x)

    inv_s = [0] * n
    inv_x = [0] * n
    for i in range(n):
        inv_s[perm_s[i]] = i
        inv_x[perm_x[i]] = i
    if any(slot_s[j] != inv_s[j] for j in range(n)) or bypass_s is not None:
        return None
    if any(slot_x[j] != inv_x[j] for j in range(n)) or bypass_x is not None:
        return None

    token_colors = random.sample(TOKEN_COLORS_POOL, n)
    inp = grid.copy()
    for i, col in enumerate(token_colors):
        _write_shape(inp, (TOP_TOKEN_ROW, ch_cols[i]), col, "solid", 0)

    expected = _build_expected_output(grid, ch_cols, perm_s, perm_x, token_colors)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
