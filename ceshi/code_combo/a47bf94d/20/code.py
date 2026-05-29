# Combo variant of a47bf94d (index 20)
# combo_spec: D9/1+D12/2
# Name: 四形状循环+双端部分token

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new")
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, get_bg, neighbors4


# Design analysis for the D9/1 + D12/2 combo
# a) Concrete behavior and trace:
#    - Keep the original task's spatial backbone: 4-5 vertical channels, one top token center and one bottom token
#      center per channel, each center exactly two cells outside a degree-1 wire endpoint, plus the original
#      orthogonal wire permutation and optional maroon 9-rect junctions.
#    - Replace the old three-step token family with a four-step cycle over fixed 3x3 monochrome shapes:
#        0. solid      = all 9 cells
#        1. X          = diagonals + center
#        2. diamond    = four corners
#        3. thin-cross = four cardinal arms, no center
#      The channel stage is now "top shows shape k, routed bottom shows shape k+1 (mod 4)".
#    - Input no longer has a single active source token. Every channel shows two incomplete tokens:
#        partial(solid)      = plus-with-center (5 cells)
#        partial(X)          = main diagonal triplet
#        partial(diamond)    = top two corners
#        partial(thin-cross) = vertical arm pair
#      Example trace for one routed channel:
#        old task: top solid, bottom empty  -> output top diamond, bottom X
#        combo:    top plus-with-center, bottom diagonal-triplet -> output top full solid, bottom full X
#      Another channel may simultaneously show:
#        top vertical arm pair, bottom plus-with-center -> output top thin-cross, bottom solid
#      so the dataset visibly contains a four-state cycle rather than the original solid/X-only source mix.
# b) Complexity calibration:
#    - Four full shapes are enough to make D9/1 visible but still tractable. They all live in the same 3x3 box,
#      so the scene stays comparable to the source task instead of becoming a different puzzle family.
#    - The partial vocabulary is deliberately small: only four masks repeated across channels. Each mask appears
#      in multiple roles across samples, which gives repetition support without exploding the number of cases.
#    - The partial vocabulary stays sparse and asymmetric, so each fragment is visibly incomplete while still needing
#      the routed opposite end to reveal which consecutive pair in the 4-cycle is present.
# c) Implicit assumptions from the original solver that no longer hold:
#    - Original main() assumed tokens could be found globally as isolated solid or X patterns, then converted with
#      a fixed source-side rule. That breaks because now every endpoint already contains a partial token and there
#      are four possible stages. The new main() reads token neighborhoods only relative to wire endpoints.
#    - Original track extraction used "all non-bg minus token cells". That is fragile once both ends always contain
#      partial tokens. Here wire extraction is explicit: only colors 5/8/9 belong to the network.
#    - Original routing logic only needed one endpoint token to determine what to emit. The combo instead must read
#      both routed endpoints, match the observed pair to one of the four consecutive stage pairs, then complete both.
# d) Placement legality:
#    - All full and partial masks stay within the same 3x3 footprint as the source task, so the original margins,
#      token rows, and endpoint offset remain valid.
#    - Channel spacing remains 4-5 columns, which keeps neighboring 3x3 token boxes disjoint.
#    - Generator writes tokens only after the wire network is fixed, and token colors are drawn from {1,2,3,4,6,7},
#      disjoint from wire colors 5/8/9, so tokens cannot merge into the wire component.
# e) Visual observability:
#    - Across examples a human sees two stable changes at once: both ends already contain colored fragments, and the
#      outputs use four visibly different 3x3 completions instead of the old diamond/X destination pair.
#    - Because each sample uses at least three stages and the 5-channel case repeats one stage, the four-cycle is
#      supported by repeated evidence rather than a one-off hidden label.
# f) Visual cleanliness:
#    - No extra markers are added. The same sparse top/bottom token bands and the same wire colors are reused.
#    - Partial masks were chosen to be low-density and centered, so even with two visible fragments per channel the
#      token areas remain legible and do not visually collide with the wire endpoints.
# g) Change inferability:
#    - No color remapping is introduced. Colors still identify channels only; all new logic is shape-based.
#    - The rule does not depend on numeric color order or any invisible property. Stage is exposed by repeated shape
#      pairs and by the fact that routed bottom tokens are permuted through the visible wire network.
# h) New main() assumptions audited against the generator:
#    - Each degree-1 endpoint has exactly one token center two cells outward in a cardinal direction; generator keeps
#      the original top/bottom placement, so the lookup is unique.
#    - Every observed 3x3 token neighborhood contains exactly one token color and matches one of the four partial
#      masks; generator constructs only those masks.
#    - For each top endpoint, tracing the wire reaches one routed bottom endpoint whose opposite 3x3 neighborhood
#      uses the same token color and the next partial in the cycle; output-first generation enforces this.


_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
WIRE_COLORS = (5, 8, 9)

SHAPE_NAMES = ("solid", "x", "diamond", "thin_cross")

_SOLID_MASK = np.array(
    [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ],
    dtype=bool,
)
_X_MASK = np.array(
    [
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
    ],
    dtype=bool,
)
_DIAMOND_MASK = np.array(
    [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1],
    ],
    dtype=bool,
)
_THIN_CROSS_MASK = np.array(
    [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ],
    dtype=bool,
)

_PARTIAL_SOLID_MASK = np.array(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ],
    dtype=bool,
)
_PARTIAL_X_MASK = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ],
    dtype=bool,
)
_PARTIAL_DIAMOND_MASK = np.array(
    [
        [1, 0, 1],
        [0, 0, 0],
        [0, 0, 0],
    ],
    dtype=bool,
)
_PARTIAL_THIN_CROSS_MASK = np.array(
    [
        [0, 1, 0],
        [0, 0, 0],
        [0, 1, 0],
    ],
    dtype=bool,
)

FULL_MASKS = [
    _SOLID_MASK,
    _X_MASK,
    _DIAMOND_MASK,
    _THIN_CROSS_MASK,
]
PARTIAL_MASKS = [
    _PARTIAL_SOLID_MASK,
    _PARTIAL_X_MASK,
    _PARTIAL_DIAMOND_MASK,
    _PARTIAL_THIN_CROSS_MASK,
]

_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DIR2FACE = {(1, 0): "N", (-1, 0): "S", (0, 1): "W", (0, -1): "E"}
_FACE2DIR = {"S": (1, 0), "N": (-1, 0), "E": (0, 1), "W": (0, -1)}


def _mask_offsets(mask: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r) - 1, int(c) - 1) for r, c in np.argwhere(mask)]


FULL_OFFSETS = [_mask_offsets(mask) for mask in FULL_MASKS]
PARTIAL_OFFSETS = [_mask_offsets(mask) for mask in PARTIAL_MASKS]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=4)]


def _find_9rects(g: np.ndarray) -> list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]]:
    """Find color-9 rectangles and record which wire cells touch each face."""
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


def _trace_wire(
    comp: set[tuple[int, int]],
    start: tuple[int, int],
    dir0: tuple[int, int],
    h: int,
    w: int,
    rects_9: list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Follow a wire component greedily, respecting crossing markers and 9-rect pass-through.

    When a trace enters a 9-rectangle, it exits through the opposite face when present; otherwise the unique
    alternative face is used, matching the source task's "straight or forced turn" rule.
    """
    cur, d, prev = start, dir0, None
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 4):
        if (cur, d) in visited:
            raise AssertionError("wire trace looped unexpectedly")
        visited.add((cur, d))

        jumped = False
        for cells9, faces9 in rects_9:
            if cur in cells9:
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
            return cur, d

        prev = cur
        cur = nxt
        d = (dr, dc)

    raise AssertionError("wire trace exceeded step limit")


def _stamp_mask(grid: np.ndarray, center: tuple[int, int], color: int, mask: np.ndarray) -> None:
    cr, cc = center
    for dr, dc in _mask_offsets(mask):
        rr, cc2 = cr + dr, cc + dc
        if 0 <= rr < grid.shape[0] and 0 <= cc2 < grid.shape[1]:
            grid[rr, cc2] = color


def _clear_3x3(grid: np.ndarray, center: tuple[int, int], bg: int) -> None:
    cr, cc = center
    grid[cr - 1 : cr + 2, cc - 1 : cc + 2] = bg


def _extract_token_mask(g: np.ndarray, center: tuple[int, int], bg: int) -> tuple[int, np.ndarray]:
    cr, cc = center
    sub = g[cr - 1 : cr + 2, cc - 1 : cc + 2]
    token_colors = {int(v) for v in sub.flatten() if int(v) != bg and int(v) not in WIRE_COLORS}
    if len(token_colors) != 1:
        raise AssertionError(f"expected exactly one token color near {center}, got {sorted(token_colors)}")
    color = next(iter(token_colors))
    mask = sub == color
    return color, mask


def _find_token_center_for_endpoint(
    g: np.ndarray,
    endpoint: tuple[int, int],
    bg: int,
) -> tuple[tuple[int, int], int, np.ndarray, tuple[int, int]]:
    h, w = g.shape
    candidates = []
    er, ec = endpoint
    for odr, odc in _N4:
        center = (er + 2 * odr, ec + 2 * odc)
        if not (1 <= center[0] < h - 1 and 1 <= center[1] < w - 1):
            continue
        sub = g[center[0] - 1 : center[0] + 2, center[1] - 1 : center[1] + 2]
        token_colors = {int(v) for v in sub.flatten() if int(v) != bg and int(v) not in WIRE_COLORS}
        if not token_colors:
            continue
        if len(token_colors) != 1:
            continue
        color = next(iter(token_colors))
        mask = sub == color
        if any(np.array_equal(mask, pm) for pm in PARTIAL_MASKS):
            candidates.append((center, color, mask, (odr, odc)))
    if len(candidates) != 1:
        raise AssertionError(f"endpoint {endpoint} has {len(candidates)} token-center candidates")
    return candidates[0]


def _infer_stage(top_mask: np.ndarray, bot_mask: np.ndarray) -> int:
    matches = [
        stage
        for stage in range(4)
        if np.array_equal(top_mask, PARTIAL_MASKS[stage])
        and np.array_equal(bot_mask, PARTIAL_MASKS[(stage + 1) % 4])
    ]
    if len(matches) != 1:
        raise AssertionError("failed to infer unique four-stage pair")
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete paired partial tokens on both ends of each wire-routed channel using a four-shape cycle.

    Input structure:
    - Background is the most frequent color. Wire networks use only colors 5, 8, and 9. Each degree-1 endpoint has
      a token center exactly two cells outward in one cardinal direction, and every channel shows partial same-color
      3x3 token fragments on both the top side and the routed bottom side.

    Rule:
    - The full-shape cycle is solid -> X -> diamond -> thin-cross -> solid.
    - For a routed channel, the observed pair of partial masks identifies one consecutive stage pair:
      top = partial(shape_k), bottom = partial(shape_{k+1}).
    - Replace each partial with its completed full shape, keeping the top as shape_k and the routed bottom as
      shape_{k+1}. Wires themselves do not change.

    Key insight:
    - A single endpoint no longer tells the whole story. The solver must follow the wire to the opposite endpoint and
      read the two partial shapes together; only their ordered pair reveals which link of the four-step cycle is active.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    wire_mask = np.isin(g, WIRE_COLORS)
    comps = _components(wire_mask)
    rects_9 = _find_9rects(g)

    out = g.copy()

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

        ep_info = {ep: _find_token_center_for_endpoint(g, ep, bg) for ep in endpoints}

        for ep in endpoints:
            top_center, top_color, top_mask, outward = ep_info[ep]
            if outward[0] != -1:
                continue

            dir_in = (-outward[0], -outward[1])
            end, dd = _trace_wire(comp, ep, dir_in, h, w, rects_9)
            bot_center = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
            bot_color, bot_mask = _extract_token_mask(g, bot_center, bg)
            if top_color != bot_color:
                raise AssertionError("routed token colors do not match")

            stage = _infer_stage(top_mask, bot_mask)
            _clear_3x3(out, top_center, bg)
            _clear_3x3(out, bot_center, bg)
            _stamp_mask(out, top_center, top_color, FULL_MASKS[stage])
            _stamp_mask(out, bot_center, bot_color, FULL_MASKS[(stage + 1) % 4])

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

GEN_MAX_RETRIES = 400


def _decompose_to_crossovers(perm: list[int]) -> list[tuple[int, int]]:
    """Decompose a permutation into sequential crossover moves using the bypass column for cycles."""
    n = len(perm)
    visited = [False] * n
    crossovers = []
    for start in range(n):
        if visited[start] or perm[start] == start:
            visited[start] = True
            continue
        cycle = []
        idx = start
        while not visited[idx]:
            visited[idx] = True
            cycle.append(idx)
            idx = perm[idx]
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
    """Attempt to replace a wire segment block with a maroon pass-through rectangle."""
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
    """Draw the original-style wire permutation network."""
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


def _choose_stage_list(n: int) -> list[int]:
    base = [0, 1, 2, 3]
    if n == 4:
        stages = base[:]
    else:
        stages = base + [random.choice(base)]
    random.shuffle(stages)
    return stages


def _derive_input(
    grid: np.ndarray,
    n: int,
    ch_cols: list[int],
    perm: list[int],
    token_colors: list[int],
    stages: list[int],
    top_token_row: int,
    bot_token_row: int,
) -> np.ndarray:
    """Place a partial token at both routed ends of every channel."""
    inp = grid.copy()
    for i in range(n):
        color = token_colors[i]
        stage = stages[i]
        top_center = (top_token_row, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[perm[i]])
        _stamp_mask(inp, top_center, color, PARTIAL_MASKS[stage])
        _stamp_mask(inp, bot_center, color, PARTIAL_MASKS[(stage + 1) % 4])
    return inp


def _build_output(
    grid: np.ndarray,
    n: int,
    ch_cols: list[int],
    perm: list[int],
    token_colors: list[int],
    stages: list[int],
    top_token_row: int,
    bot_token_row: int,
) -> np.ndarray:
    out = grid.copy()
    for i in range(n):
        color = token_colors[i]
        stage = stages[i]
        _stamp_mask(out, (top_token_row, ch_cols[i]), color, FULL_MASKS[stage])
        _stamp_mask(out, (bot_token_row, ch_cols[perm[i]]), color, FULL_MASKS[(stage + 1) % 4])
    return out


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
    stages = _choose_stage_list(n)

    while True:
        perm = list(range(n))
        random.shuffle(perm)
        if perm != list(range(n)):
            break

    crossovers = _decompose_to_crossovers(perm)
    if not crossovers:
        return None

    n_xovers = len(crossovers)
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
    placed_rects: list[tuple[int, int, int, int]] = []
    for _ in range(n_rects * RECT_RETRY_FACTOR):
        if len(placed_rects) >= n_rects:
            break
        rect = _try_place_9rect(grid, ch_cols, xover_rows, top_ep_row, bot_ep_row, n, placed_rects)
        if rect is not None:
            placed_rects.append(rect)

    output = _build_output(grid, n, ch_cols, perm, token_colors, stages, top_token_row, bot_token_row)
    inp = _derive_input(grid, n, ch_cols, perm, token_colors, stages, top_token_row, bot_token_row)

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate valid input after {GEN_MAX_RETRIES} attempts")
