# Variant of a47bf94d
# Dimension: D13 — 多步处理顺序 (multi_step)
# Change: 单步变换——每个token独立执行一次分解和传输操作，各通道之间没有顺序依赖... → 链式反应传播

from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

# Design analysis
# ===============
# variant:
#   summary: >
#     Keep the original wire vocabulary (8/5/9) and the same token shapes, but split the
#     network into two stacked wire layers. The middle token row is both the destination of
#     layer 1 and the source of layer 2, so a transmitted shape can trigger a second send.
#   concrete_behavior:
#     - Input tokens appear only on the top token row.
#     - A top solid sends X into layer 1 and leaves a diamond behind.
#     - A top X sends a diamond into layer 1 and stays X.
#     - When the received shape lands on the middle row, it does not stop if another port is
#       attached there: X forwards diamond, diamond forwards solid, and a forwarded solid
#       would again decompose into a diamond before sending X onward.
#     - Bottom-row tokens are terminal because they have only one port.
#   trace_example:
#     - Original: solid(top) -> X(bottom), done.
#     - Variant: solid(top) -> X(mid) -> diamond(bottom).
#     - Variant with X source: X(top) -> diamond(mid) -> solid(bottom).
#   complexity_calibration:
#     - Use exactly 3 token rows and 2 wire layers, so every example visibly contains a
#       nontrivial second hop but does not become visually cluttered.
#     - Each wire layer keeps the original crossing/bypass style, but only uses one adjacent
#       swap permutation, which limits the required height.
#   implicit_assumptions_audited:
#     - The original solver assumed every relevant token is adjacent to exactly one endpoint.
#       Here, middle-row slots may touch two different endpoints, so the solver is rewritten
#       around port identities rather than center positions alone.
#     - The original routing loop could emit directly from a source endpoint to the far end
#       because destinations were always terminal. That is no longer safe; the new solver must
#       remember the arrival port and continue through the other port only.
#     - Solid->diamond is now tied to "sending" rather than "existing". A solid that arrives
#       at a terminal slot stays solid, but a solid at a nonterminal slot decomposes and emits.
#   placement_legality:
#     - Token centers are always two cells away from their endpoint, matching the original.
#     - Stage rows are spaced so the 3x3 token windows never overlap the wire layers.
#     - Middle-row slots receive one port from the upper layer and one from the lower layer.
#     - Wire layers are vertically separated, so their 9-rectangles cannot merge accidentally.
#   visual_observability:
#     - The second hop is directly visible because every output has an occupied middle row.
#     - Shape changes remain purely geometric; color identity is preserved throughout.
#   cleanliness:
#     - Only three token bands are used, all aligned to the same channel columns.
#     - 9-rectangles remain optional decoration on the wires, not on the token rows.

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_SOLID_OFF = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_STATE_TO_OFFSETS = {
    "solid": _SOLID_OFF,
    "x": _X_OFF,
    "diamond": _DIAMOND_OFF,
}

_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DIR2FACE = {(1, 0): "N", (-1, 0): "S", (0, 1): "W", (0, -1): "E"}
_FACE2DIR = {"S": (1, 0), "N": (-1, 0), "E": (0, 1), "W": (0, -1)}


def _find_solid_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for maximal solid 3x3 blocks."""
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


def _find_pattern_3x3(
    g: np.ndarray,
    bg: int,
    offsets: list[tuple[int, int]],
    sample_offset: tuple[int, int],
) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return isolated 3x3 pattern instances as {center: (color, (r0,c0))}."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    mask = np.zeros((3, 3), dtype=bool)
    for dr, dc in offsets:
        mask[dr + 1, dc + 1] = True

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            col = int(sub[sample_offset[0] + 1, sample_offset[1] + 1])
            if col == bg:
                continue
            if np.any(sub[mask] != col) or np.any(sub[~mask] != bg):
                continue
            isolated = True
            for dr, dc in offsets:
                rr, cc = r0 + 1 + dr, c0 + 1 + dc
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


def _find_x_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    return _find_pattern_3x3(g, bg, _X_OFF, (-1, -1))


def _find_diamond_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    return _find_pattern_3x3(g, bg, _DIAMOND_OFF, (-1, 0))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=4)]


def _find_9rects(g: np.ndarray) -> list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]]:
    """Find color-9 rectangles and record adjacent wire cells on each face."""
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
    """
    Follow one wire path greedily, including straight-through traversal inside 9-rectangles.

    Returns:
        (terminal_endpoint, outward_direction_at_terminal)
    """
    cur, d, prev = start, dir0, None
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 4):
        key = (cur, d)
        if key in visited:
            raise AssertionError("wire trace looped unexpectedly")
        visited.add(key)

        jumped = False
        for cells9, faces9 in rects_9:
            if cur in cells9:
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


def _clear_window(out: np.ndarray, center: tuple[int, int], bg: int) -> None:
    cr, cc = center
    out[cr - 1 : cr + 2, cc - 1 : cc + 2] = bg


def _stamp_state(out: np.ndarray, center: tuple[int, int], color: int, state: str) -> None:
    cr, cc = center
    for dr, dc in _STATE_TO_OFFSETS[state]:
        out[cr + dr, cc + dc] = color


def _advance_state(state: str) -> str:
    return {"solid": "x", "x": "diamond", "diamond": "solid"}[state]


def _display_state(state: str, has_next_port: bool) -> str:
    if state == "solid" and has_next_port:
        return "diamond"
    return state


def _build_port_graph(
    g: np.ndarray,
) -> tuple[
    dict[tuple[tuple[int, int], tuple[int, int]], tuple[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]]],
    dict[tuple[int, int], list[tuple[tuple[int, int], tuple[int, int]]]],
]:
    """Build directed port transitions from the wire geometry."""
    h, w = g.shape
    rects_9 = _find_9rects(g)
    comps = _components(np.isin(g, (5, 8, 9)))

    port_to_edge: dict[
        tuple[tuple[int, int], tuple[int, int]],
        tuple[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]],
    ] = {}
    ports_by_center: dict[tuple[int, int], list[tuple[tuple[int, int], tuple[int, int]]]] = defaultdict(list)

    for pts in comps:
        comp = set(pts)
        if len(comp) < 2:
            continue

        deg = {p: 0 for p in comp}
        for r, c in comp:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in comp:
                    deg[(r, c)] += 1
        endpoints = [p for p in comp if deg[p] <= 1]
        for ep in endpoints:
            nbrs = [(nr, nc) for nr, nc in neighbors4(ep[0], ep[1]) if (nr, nc) in comp]
            if len(nbrs) != 1:
                continue
            outward = (ep[0] - nbrs[0][0], ep[1] - nbrs[0][1])
            center = (ep[0] + 2 * outward[0], ep[1] + 2 * outward[1])
            if not (1 <= center[0] < h - 1 and 1 <= center[1] < w - 1):
                continue

            dir_in = (-outward[0], -outward[1])
            end, dd = _trace_wire(comp, ep, dir_in, h, w, rects_9)
            dest_center = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
            if not (1 <= dest_center[0] < h - 1 and 1 <= dest_center[1] < w - 1):
                continue

            start_port = (center, outward)
            dest_port = (dest_center, dd)
            port_to_edge[start_port] = (dest_center, dest_port)
            if start_port not in ports_by_center[center]:
                ports_by_center[center].append(start_port)

    return port_to_edge, dict(ports_by_center)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate token shapes through stacked wire layers, allowing a received token to fire again.

    Input structure:
    - Background cells surround cyan wire tracks (8 with 5/9 routing markers) and 3x3 color tokens.
    - Tokens sit two cells away from wire endpoints; middle-row slots may touch two different endpoints.

    Rule:
    - A token processes locally when it has an unused outgoing port: solid becomes a diamond and sends X,
      X stays X and sends a diamond, diamond stays diamond and sends a solid.
    - When a transmitted shape lands on a slot that has another connected port, it continues through that
      next wire layer; terminal slots simply display the arrived shape.

    Key insight:
    - The task is no longer independent one-hop routing. You must follow port-to-port arrivals and keep
      advancing the token shape along the chain until no further port remains.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)
    diamonds = _find_diamond_3x3(g, bg)
    port_to_edge, ports_by_center = _build_port_graph(g)

    initial_tokens: dict[tuple[int, int], tuple[str, int]] = {}
    for center, (color, _box) in solids.items():
        if color not in (5, 8, 9):
            initial_tokens[center] = ("solid", int(color))
    for center, (color, _box) in xs.items():
        if color not in (5, 8, 9):
            initial_tokens[center] = ("x", int(color))
    for center, (color, _box) in diamonds.items():
        if color not in (5, 8, 9):
            initial_tokens[center] = ("diamond", int(color))

    out = g.copy()
    queue: deque[tuple[tuple[int, int], str, int, tuple[tuple[int, int], tuple[int, int]] | None]] = deque()
    for center, (state, color) in initial_tokens.items():
        queue.append((center, state, color, None))

    seen: set[tuple[tuple[int, int], str, int, tuple[tuple[int, int], tuple[int, int]] | None]] = set()
    step_limit = g.size * 8
    steps = 0

    while queue:
        center, state, color, arrival_port = queue.popleft()
        key = (center, state, color, arrival_port)
        if key in seen:
            continue
        seen.add(key)

        ports = ports_by_center.get(center, [])
        next_ports = [port for port in ports if port != arrival_port]
        assert len(next_ports) <= 1, "generator guarantees at most one onward port"
        has_next = len(next_ports) == 1

        _clear_window(out, center, bg)
        _stamp_state(out, center, color, _display_state(state, has_next))

        if has_next:
            dest_center, dest_arrival = port_to_edge[next_ports[0]]
            queue.append((dest_center, _advance_state(state), color, dest_arrival))

        steps += 1
        if steps > step_limit:
            raise AssertionError("token propagation exceeded step limit")

    return out


# --------------- generator (stacked_chain_reaction) ---------------

CHANNEL_COUNTS = [4]
CHANNEL_SPACING = [4, 5]
LEFT_MARGIN_RANGE = (2, 3)
BYPASS_COL_OFFSET = 2
TOKEN_ROW_OPTIONS = [2, 3]
TOKEN_COLORS_POOL = [1, 2, 3, 4, 6, 7]
EP_TOKEN_GAP = 2
LAYER_SPANS = [8, 9]
GRID_W_MIN = 22
GRID_W_MAX = 28
RECT_COUNT_WEIGHTS = [50, 35, 15]  # 0,1,2 rects per layer
RECT_WIRE_COUNT_WEIGHTS = [50, 38, 12]
RECT_PARALLEL_PROB = 0.75
RECT_H_RANGE = (2, 4)
RECT_MAX_WIDTH = 7
RECT_RETRY_FACTOR = 5
RECT_MARGIN_GAP = 1
X_INPUT_COUNT_WEIGHTS = [70, 30]  # 1 or 2 X sources
GEN_MAX_RETRIES = 400


def _decompose_to_crossovers(perm: list[int]) -> list[tuple[int, int]]:
    """Decompose a permutation into bypass-assisted crossover moves."""
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


def _sample_adjacent_swap_perm(n: int) -> list[int]:
    perm = list(range(n))
    j = random.randint(0, n - 2)
    perm[j], perm[j + 1] = perm[j + 1], perm[j]
    return perm


def _stamp_token(grid: np.ndarray, center_r: int, center_c: int, color: int, offsets: list[tuple[int, int]]) -> None:
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = color


def _try_place_9rect(
    grid: np.ndarray,
    ch_cols: list[int],
    xover_rows: list[int],
    top_ep_row: int,
    bot_ep_row: int,
    n: int,
    placed_rects: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    """Attempt to place one embedded 9-rectangle within a single wire layer."""
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
            margin_l = 1
    rect_c0, rect_c1 = c_lo - margin_l, c_hi + margin_r
    if rect_c0 < 0 or rect_c1 >= w or rect_c1 - rect_c0 + 1 > RECT_MAX_WIDTH:
        return None

    max_h = min(RECT_H_RANGE[1], bot_ep_row - top_ep_row - 4)
    if max_h < RECT_H_RANGE[0]:
        return None
    rect_h = random.randint(RECT_H_RANGE[0], max_h)
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
        if grid[rect_r0 - 1, c] in (5, 8):
            n_cols.add(c)
        if grid[rect_r1 + 1, c] in (5, 8):
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
    """Draw one wire layer using the original bypass-and-crossing style."""
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


def _apply_output_state(grid: np.ndarray, center: tuple[int, int], color: int, incoming_state: str, has_next: bool) -> None:
    state = _display_state(incoming_state, has_next)
    _stamp_token(grid, center[0], center[1], color, _STATE_TO_OFFSETS[state])


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(CHANNEL_COUNTS)
    spacing = random.choice(CHANNEL_SPACING)
    left_margin = random.randint(*LEFT_MARGIN_RANGE)
    first_col = left_margin + 1
    ch_cols = [first_col + i * spacing for i in range(n)]
    bypass_col = ch_cols[-1] + BYPASS_COL_OFFSET
    width = max(GRID_W_MIN, bypass_col + 3)
    if width > GRID_W_MAX:
        return None

    top_row = random.choice(TOKEN_ROW_OPTIONS)
    stage_rows = [top_row]
    for span in (random.choice(LAYER_SPANS), random.choice(LAYER_SPANS)):
        stage_rows.append(stage_rows[-1] + span + 2 * EP_TOKEN_GAP)
    height = stage_rows[-1] + 3

    token_colors = random.sample(TOKEN_COLORS_POOL, n)
    perms = [_sample_adjacent_swap_perm(n), _sample_adjacent_swap_perm(n)]

    grid = np.zeros((height, width), dtype=int)

    for layer_idx, perm in enumerate(perms):
        top_ep_row = stage_rows[layer_idx] + EP_TOKEN_GAP
        bot_ep_row = stage_rows[layer_idx + 1] - EP_TOKEN_GAP
        crossovers = _decompose_to_crossovers(perm)
        if len(crossovers) != 3:
            return None
        xover_rows = [top_ep_row + 2, top_ep_row + 4, top_ep_row + 6]
        if xover_rows[-1] >= bot_ep_row:
            return None

        slot_wire, bypass_wire = _build_wire_network(
            grid, n, ch_cols, bypass_col, top_ep_row, bot_ep_row, crossovers, xover_rows
        )
        inv_perm = [0] * n
        for i in range(n):
            inv_perm[perm[i]] = i
        if bypass_wire is not None or any(slot_wire[j] != inv_perm[j] for j in range(n)):
            return None

        target_rects = random.choices([0, 1, 2], weights=RECT_COUNT_WEIGHTS)[0]
        placed_rects: list[tuple[int, int, int, int]] = []
        for _ in range(target_rects * RECT_RETRY_FACTOR):
            if len(placed_rects) >= target_rects:
                break
            result = _try_place_9rect(grid, ch_cols, xover_rows, top_ep_row, bot_ep_row, n, placed_rects)
            if result is not None:
                placed_rects.append(result)

    input_grid = grid.copy()
    expected = grid.copy()

    n_x = random.choices([1, 2], weights=X_INPUT_COUNT_WEIGHTS)[0]
    x_channels = set(random.sample(range(n), min(n_x, n)))

    for i in range(n):
        color = token_colors[i]
        start_state = "x" if i in x_channels else "solid"
        top_center = (stage_rows[0], ch_cols[i])
        mid_idx = perms[0][i]
        mid_center = (stage_rows[1], ch_cols[mid_idx])
        bot_idx = perms[1][mid_idx]
        bot_center = (stage_rows[2], ch_cols[bot_idx])

        _stamp_token(input_grid, top_center[0], top_center[1], color, _STATE_TO_OFFSETS[start_state])

        _apply_output_state(expected, top_center, color, start_state, True)
        mid_incoming = _advance_state(start_state)
        _apply_output_state(expected, mid_center, color, mid_incoming, True)
        bot_incoming = _advance_state(mid_incoming)
        _apply_output_state(expected, bot_center, color, bot_incoming, False)

    try:
        actual = main(input_grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
