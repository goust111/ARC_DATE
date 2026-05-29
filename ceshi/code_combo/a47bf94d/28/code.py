# Combo variant of a47bf94d (index 28)
# combo_spec: D5/0+D12/2+D13/3
# Name: 叠加合并+双端部分token+双向传输

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import components_mask, get_bg, neighbors4
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[4] / "solvers"))
    from arc_utils import components_mask, get_bg, neighbors4


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - Preserve the original scaffold: 4-5 vertical channels, top and bottom token centers exactly two cells
#      outside degree-1 wire endpoints, orthogonal 8-wire routing with 5 crossing markers and optional 9-rect
#      pass-through blocks.
#    - Replace the original "one active sender + one empty/partial far side" state with a fully bidirectional pair:
#      every routed channel starts with two same-colored incomplete tokens, one diamond (4 cardinals) and one X
#      (diagonals + center).
#    - Both endpoints act at once. The diamond endpoint sends the missing X-part to the far side, and the X endpoint
#      sends the missing diamond-part back. Because both endpoints already contain the complementary stationary half,
#      each arrival overlays onto an existing same-colored fragment and both 3x3 windows become solid blocks.
#    - Example on one channel:
#      original task: top solid, bottom empty -> output top diamond, bottom X.
#      this combo:    top diamond, bottom X  -> output top solid, bottom solid.
#      Or the orientation may be flipped:
#      input top X, bottom diamond          -> output top solid, bottom solid.
#      The wire still matters because it tells which bottom token pairs with which top token.
# b) Complexity calibration:
#    - The token vocabulary is intentionally small: only diamond and X appear in input, and solid appears in output.
#      That keeps the shape logic learnable while still making the fusion visible: now both ends change on every
#      active channel, not just one end.
#    - Each sample mixes both orientations (some channels top=diamond, some top=X), so the solver cannot simplify the
#      task to a fixed "top behaves one way, bottom behaves another way" rule.
#    - Keeping same-color paired endpoints avoids introducing invisible in-pipe interaction states; the simultaneity is
#      exposed directly by the fact that both endpoints complete on the same channel.
# c) Implicit assumptions from the original solver that no longer hold:
#    - Original main() assumed a source token could be found globally as a solid or X and that the far endpoint was
#      mostly empty or at most had one passive receiver. Here every endpoint already contains a token fragment.
#    - Original track extraction used "non-background minus detected token cells". With dual-end fragments that is
#      brittle, so the new solver treats only colors 5/8/9 as wire and finds token neighborhoods relative to degree-1
#      endpoints.
#    - Original routing rewrote only the source side locally and only wrote one far-side complement. That assumption is
#      broken by simultaneous two-way transport: after pairing endpoints, both 3x3 neighborhoods must be rewritten.
# d) Placement legality:
#    - Diamond, X, and solid all fit inside the original 3x3 token footprint, so the source task's spacing and margins
#      remain valid.
#    - Channel spacing stays 4-5 columns; routed bottom centers are a permutation of top columns, so 3x3 token windows
#      remain disjoint.
#    - Token colors come from {1,2,3,4,6,7}, disjoint from wire colors 5/8/9, so token fragments cannot merge into the
#      wire component.
# e) Visual observability:
#    - Across 2-4 examples, a human can see that every input channel already has two partial same-colored shapes at the
#      ends of one wire path, and every output turns both ends into filled 3x3 blocks.
#    - Mixing top-diamond/bottom-X and top-X/bottom-diamond channels inside the same sample makes the "mutual completion"
#      rule visible without needing hidden case labels.
# f) Visual cleanliness:
#    - No new symbols are added beyond the original wire colors and the original token family. The output simply replaces
#      sparse endpoint fragments with clean solid 3x3 blocks.
#    - There is no same-cell multi-color overlap: the merge is conceptual, but the rendered result is a single monochrome
#      solid at each endpoint.
# g) Change inferability:
#    - No color remapping is introduced. Color only identifies which two endpoints belong together through the routed
#      channel.
#    - The new rule is shape-based and symmetric. The same two partial shapes always combine into the same solid block,
#      so a small number of examples is enough to infer it.
# h) New main() assumptions audited against the generator:
#    - Each degree-1 endpoint has exactly one token center two cells outward in a cardinal direction; generator keeps the
#      original top/bottom geometry so this lookup is unique.
#    - Every token neighborhood contains exactly one non-wire, non-background color and matches either the diamond mask
#      or the X mask; generator constructs only those two inputs.
#    - For each top endpoint, tracing the wire reaches one bottom endpoint with the same token color and the complementary
#      partial shape; output-first generation enforces that one-to-one pairing.


_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
WIRE_COLORS = (5, 8, 9)

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID_OFF = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

_DIAMOND_MASK = np.array(
    [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
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
    Follow a wire component greedily, respecting crossing markers and 9-rect pass-through.

    When a trace enters a 9-rectangle, it exits through the opposite face when available; otherwise it uses the
    unique other face, matching the source task's pass-through behavior.
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


def _stamp_token(
    grid: np.ndarray,
    center_r: int,
    center_c: int,
    color: int,
    offsets: list[tuple[int, int]],
) -> None:
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = color


def _clear_3x3(grid: np.ndarray, center: tuple[int, int], bg: int) -> None:
    cr, cc = center
    grid[cr - 1 : cr + 2, cc - 1 : cc + 2] = bg


def _extract_token_shape(g: np.ndarray, center: tuple[int, int], bg: int) -> tuple[int, str]:
    cr, cc = center
    sub = g[cr - 1 : cr + 2, cc - 1 : cc + 2]
    token_colors = {int(v) for v in sub.flatten() if int(v) != bg and int(v) not in WIRE_COLORS}
    if len(token_colors) != 1:
        raise AssertionError(f"expected exactly one token color near {center}, got {sorted(token_colors)}")
    color = next(iter(token_colors))
    mask = sub == color
    if np.array_equal(mask, _DIAMOND_MASK):
        return color, "diamond"
    if np.array_equal(mask, _X_MASK):
        return color, "x"
    raise AssertionError(f"unrecognized token shape near {center}")


def _find_token_center_for_endpoint(
    g: np.ndarray,
    endpoint: tuple[int, int],
    bg: int,
) -> tuple[tuple[int, int], int, str, tuple[int, int]]:
    h, w = g.shape
    er, ec = endpoint
    candidates = []
    for odr, odc in _N4:
        center = (er + 2 * odr, ec + 2 * odc)
        if not (1 <= center[0] < h - 1 and 1 <= center[1] < w - 1):
            continue
        try:
            color, shape = _extract_token_shape(g, center, bg)
        except AssertionError:
            continue
        candidates.append((center, color, shape, (odr, odc)))
    if len(candidates) != 1:
        raise AssertionError(f"endpoint {endpoint} has {len(candidates)} token-center candidates")
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete both ends of every routed channel by mutually exchanging complementary token halves.

    Input structure:
    - Background is the most common color. Wire cells use only colors 5, 8, and 9.
    - Each degree-1 wire endpoint has a 3x3 token center exactly two cells outward in one cardinal direction.
    - Every channel already shows same-colored partial tokens on both ends: one endpoint has a diamond and the routed
      opposite endpoint has an X.

    Rule:
    - Diamond and X are complementary halves of a solid 3x3 block.
    - Both endpoints transmit simultaneously through the wire: the diamond endpoint sends the missing X-part, and the
      X endpoint sends the missing diamond-part.
    - At each far endpoint, the arriving complement is overlaid onto the already-present partial, so both endpoint
      windows become solid 3x3 blocks. The wire network itself is unchanged.

    Key insight:
    - No endpoint is merely a passive receiver anymore. The solver must pair the two routed endpoints, verify that they
      hold complementary same-colored partials, and then complete both sides of the channel rather than only one side.
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
            src_center, src_color, src_shape, outward = ep_info[ep]
            if outward != (-1, 0):
                continue

            dir_in = (1, 0)
            end, dd = _trace_wire(comp, ep, dir_in, h, w, rects_9)
            dst_center = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
            dst_color, dst_shape = _extract_token_shape(g, dst_center, bg)

            if src_color != dst_color:
                raise AssertionError("routed token colors do not match")
            if {src_shape, dst_shape} != {"diamond", "x"}:
                raise AssertionError("each channel must contain one diamond and one X")

            _clear_3x3(out, src_center, bg)
            _clear_3x3(out, dst_center, bg)
            _stamp_token(out, src_center[0], src_center[1], src_color, _SOLID_OFF)
            _stamp_token(out, dst_center[0], dst_center[1], dst_color, _SOLID_OFF)

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
    """Decompose a permutation into crossover moves using the bypass column for cycles."""
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


def _build_input_output(
    grid: np.ndarray,
    n: int,
    ch_cols: list[int],
    perm: list[int],
    token_colors: list[int],
    top_token_row: int,
    bot_token_row: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Place complementary partial pairs on both ends and solid outputs on both ends."""
    inp = grid.copy()
    out = grid.copy()

    n_top_x = random.randint(1, n - 1)
    top_x_channels = set(random.sample(range(n), n_top_x))

    for i in range(n):
        col = token_colors[i]
        top_center = (top_token_row, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[perm[i]])

        if i in top_x_channels:
            _stamp_token(inp, top_center[0], top_center[1], col, _X_OFF)
            _stamp_token(inp, bot_center[0], bot_center[1], col, _DIAMOND_OFF)
        else:
            _stamp_token(inp, top_center[0], top_center[1], col, _DIAMOND_OFF)
            _stamp_token(inp, bot_center[0], bot_center[1], col, _X_OFF)

        _stamp_token(out, top_center[0], top_center[1], col, _SOLID_OFF)
        _stamp_token(out, bot_center[0], bot_center[1], col, _SOLID_OFF)

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
    xover_rows = [top_ep_row + 1 + step * (idx + 1) for idx in range(len(crossovers))]
    if xover_rows[-1] >= bot_ep_row:
        xover_rows = [top_ep_row + 2 + idx * 2 for idx in range(len(crossovers))]
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

    inp, out = _build_input_output(grid, n, ch_cols, perm, token_colors, top_token_row, bot_token_row)
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
