# Combo variant of cbebaa4b (index 2)
# combo_spec: D1/3+D8/0
# Name: 多色连接器 + 颜色编码优先级

# Design analysis for this combo variant
# -------------------------------------
# concretization and behavior trace:
# - Keep the original output-first jigsaw template: one fixed yellow anchor plus 5-6 scattered pieces.
# - Connector pixels now use exactly three visible colors {red=2, blue=1, green=3}. Same-color overlap is
#   still the only legal connection, but color also carries an ordering role.
# - The ordering role is local to each multi-port piece: when a piece contains several connector colors,
#   its highest-priority visible connector (red before blue before green) is the side that must be used to
#   attach that piece into the current assembly; lower-priority colors on that same piece are deferred ports
#   that only become usable after the piece has already entered the structure.
# - Small trace:
#   original: a red-red match is enough; every red port on a piece is interchangeable.
#   combo: a red-blue piece may geometrically fit a blue target as well, but that is still illegal because
#   the piece must enter through red first; blue is only an outgoing socket after placement.
#
# complexity calibration:
# - Restrict connector colors to three frequent, highly salient ARC colors: red, blue, green.
# - Body colors exclude connector colors and yellow, so the dual color semantics stay readable.
# - The generator guarantees repeated evidence: at least one red-blue piece and at least one blue-green
#   piece exist in every sample, so examples visibly demonstrate both same-color matching and local priority.
# - Piece counts stay within the original range, and the scaffold stays small and readable: one mixed-color
#   chain plus colored side leaves around the anchor.
#
# implicit assumption audit against the original solver:
# - _extract_pieces can no longer look only for arr == 2; it must collect all connector colors and keep them.
# - _try_place originally tolerated any same-color overlap. In the combo, same-color body overlap is still
#   illegal; only same-color connector overlap is valid.
# - The original red-count state keyed only by coordinate. That loses essential information now because
#   (r,c,red) and (r,c,blue) are different constraints. The state therefore stores both color and count.
# - The original DFS assumed every connector on a piece was equivalent. That fails here: a red-blue piece is
#   not allowed to join through its blue side before its red side has been consumed.
# - The original solver treated every connector on a piece as equivalent. That fails here because a red-blue
#   piece may legally match blue geometrically while still being semantically wrong: blue is its outgoing
#   connector, not its entry connector.
#
# placement legality:
# - Generation remains output-first. Pieces are first assembled in local coordinates, then scattered.
# - Only connector-on-connector overlaps of identical color are legal during assembled rendering.
# - The assembled scaffold is fixed and legible: the anchor exposes red, blue, and green targets, and a
#   rightward red->blue->green chain plus colored side leaves guarantee a valid priority-respecting assembly.
#
# visual observability and cleanliness:
# - Red, blue, and green are all visible in both input and output, while body colors are drawn from
#   {grey=5, pink=6, orange=7, teal=8, maroon=9}. This keeps connectors easy to isolate by eye.
# - Decorative bars remain in the body color, so connector colors are concentrated on the matching interfaces.
# - The scatter step preserves generous gaps, preventing mixed-color ports from visually colliding across pieces.
#
# inferability check:
# - Across samples, humans can see that same-color ports meet, but on mixed-color pieces the red side is the
#   plugging side while blue/green extend outward as deferred sockets.
# - Because every sample includes repeated mixed-color pieces and direct blue/green anchor leaves, the learner
#   can compare "entry colors" against "still exposed colors" inside one scene instead of guessing globally.
#
# new solver assumptions and matching generator guarantees:
# - Assumption: connector colors are only {2,1,3}; generator guarantees this.
# - Assumption: non-connector pixels never use connector colors; generator enforces BODY_COLORS.
# - Assumption: a piece enters through its minimum-priority connector color. Generator builds explicit
#   red->blue and blue->green mixed pieces, and solver asserts this at search time.
# - Assumption: the D8 signal must create real forbidden placements, not just recolor the ports. Generator
#   checks that each accepted input has at least one geometrically legal lower-priority entry move.

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


CONNECTOR_COLORS = (2, 1, 3)  # red, blue, green
CONNECTOR_SET = set(CONNECTOR_COLORS)
BODY_COLORS = [5, 6, 7, 8, 9]
PRIORITY = {2: 0, 1: 1, 3: 2}
ENTRY_COLOR_BY_STAGE = {1: 2, 2: 1, 3: 3}


def _is_connector(arr: np.ndarray) -> np.ndarray:
    return np.isin(arr, CONNECTOR_COLORS)


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        port_cells = np.argwhere(_is_connector(arr))
        ports = [(int(r), int(c), int(arr[r, c])) for r, c in port_cells]
        pieces.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "ports": ports,
                "shape": (int(arr.shape[0]), int(arr.shape[1])),
                "area": int(np.sum(arr != bg)),
                "entry_priority": min(PRIORITY[int(arr[r, c])] for r, c in port_cells)
                if len(port_cells)
                else 99,
            }
        )
    return pieces


def _try_place(
    canvas: np.ndarray, arr: np.ndarray, top: int, left: int, bg: int
) -> np.ndarray | None:
    h, w = canvas.shape
    ph, pw = arr.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return None

    sub = canvas[top : top + ph, left : left + pw]
    mask = arr != bg
    if not np.any(mask):
        return None

    arr_conn = _is_connector(arr)
    sub_conn = _is_connector(sub)
    overlap_ok = arr_conn & sub_conn & (arr == sub)
    conflict = mask & (sub != bg) & ~overlap_ok
    if np.any(conflict):
        return None

    out = canvas.copy()
    out_sub = out[top : top + ph, left : left + pw]
    out_sub[mask] = arr[mask]
    out[top : top + ph, left : left + pw] = out_sub
    return out


def _solve(
    grid: np.ndarray,
    *,
    enforce_phase: bool,
    enforce_entry: bool,
    max_solutions: int = 2,
) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must have at least one colored piece"

    anchors = [p for p in pieces if np.any(p["arr"] == 4)]
    assert len(anchors) == 1, "input must have exactly one yellow (4) anchor piece"
    anchor = anchors[0]
    rest = [p for p in pieces if p is not anchor]

    canvas = np.full((h, w), bg, dtype=int)
    ar0, ac0, _, _ = anchor["bbox"]
    placed = _try_place(canvas, anchor["arr"], ar0, ac0, bg)
    assert placed is not None, "anchor piece must fit on canvas at its original position"

    total_ports = int(sum(len(p["ports"]) for p in pieces))
    assert total_ports % 2 == 0, "total connector ports must be even for complete pairing"
    target_unique_ports = total_ports // 2

    port_state: dict[tuple[int, int], tuple[int, int]] = {}
    for rr, cc, col in anchor["ports"]:
        key = (ar0 + rr, ac0 + cc)
        prev = port_state.get(key)
        if prev is None:
            port_state[key] = (col, 1)
        else:
            prev_col, prev_count = prev
            assert prev_col == col, "anchor cannot stack different connector colors"
            port_state[key] = (col, prev_count + 1)

    solutions: list[np.ndarray] = []

    def dfs(
        cur: np.ndarray,
        remaining: list[dict],
        state: dict[tuple[int, int], tuple[int, int]],
    ) -> None:
        if len(solutions) >= max_solutions:
            return
        if not remaining:
            if len(state) == target_unique_ports and all(count == 2 for _col, count in state.values()):
                solutions.append(cur)
            return
        if len(state) > target_unique_ports:
            return

        unpaired = [(r, c, col) for (r, c), (col, count) in state.items() if count == 1]
        if not unpaired:
            return

        active_priority = min(PRIORITY[col] for _r, _c, col in unpaired) if enforce_phase else None
        targets = (
            [(r, c, col) for r, c, col in unpaired if PRIORITY[col] == active_priority]
            if enforce_phase
            else unpaired
        )
        open_once = {(r, c) for (r, c), (_col, count) in state.items() if count == 1}

        best_moves = None
        for tr, tc, tcol in targets:
            moves = []
            for idx, p in enumerate(remaining):
                if enforce_phase and enforce_entry and p["entry_priority"] != active_priority:
                    continue
                ph, pw = p["shape"]
                for pr, pc, pcol in p["ports"]:
                    if pcol != tcol:
                        continue
                    if enforce_entry and PRIORITY[pcol] != p["entry_priority"]:
                        continue

                    top = tr - pr
                    left = tc - pc
                    if top < 0 or left < 0 or top + ph > h or left + pw > w:
                        continue

                    bad = False
                    overlap = 0
                    for rr, cc, col in p["ports"]:
                        gr, gc = top + rr, left + cc
                        prev = state.get((gr, gc))
                        if prev is None:
                            continue
                        prev_col, prev_count = prev
                        if prev_col != col or prev_count >= 2:
                            bad = True
                            break
                        if (gr, gc) in open_once:
                            overlap += 1
                    if bad:
                        continue
                    moves.append((idx, top, left, overlap))
            if not moves:
                continue
            if best_moves is None or len(moves) < len(best_moves):
                best_moves = moves
                if len(best_moves) == 1:
                    break

        if best_moves is None:
            return

        dedup: dict[tuple[int, int, int], int] = {}
        for idx, top, left, ov in best_moves:
            key = (idx, top, left)
            dedup[key] = max(dedup.get(key, 0), ov)
        moves2 = [(idx, top, left, ov) for (idx, top, left), ov in dedup.items()]
        moves2.sort(key=lambda x: (-x[3], -remaining[x[0]]["area"], x[1], x[2], x[0]))

        for idx, top, left, _ov in moves2:
            p = remaining[idx]
            next_remaining = remaining[:idx] + remaining[idx + 1 :]

            next_state = dict(state)
            bad = False
            for rr, cc, col in p["ports"]:
                key = (top + rr, left + cc)
                prev = next_state.get(key)
                if prev is None:
                    next_state[key] = (col, 1)
                    continue
                prev_col, prev_count = prev
                if prev_col != col or prev_count >= 2:
                    bad = True
                    break
                next_state[key] = (col, prev_count + 1)
            if bad:
                continue

            nxt = _try_place(cur, p["arr"], top, left, bg)
            if nxt is None:
                continue
            dfs(nxt, next_remaining, next_state)
            if len(solutions) >= max_solutions:
                return

    dfs(placed, rest, port_state)
    return solutions


def _color_only_solution_count(grid: np.ndarray, cap: int = 2) -> int:
    return len(_solve(grid, enforce_phase=False, enforce_entry=False, max_solutions=cap))


def _wrong_entry_move_count(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = 0
    pieces = _extract_pieces(g, bg)
    anchors = [p for p in pieces if np.any(p["arr"] == 4)]
    assert len(anchors) == 1
    anchor = anchors[0]
    rest = [p for p in pieces if p is not anchor]

    canvas = np.full((h, w), bg, dtype=int)
    ar0, ac0, _, _ = anchor["bbox"]
    placed = _try_place(canvas, anchor["arr"], ar0, ac0, bg)
    assert placed is not None

    state: dict[tuple[int, int], tuple[int, int]] = {}
    for rr, cc, col in anchor["ports"]:
        key = (ar0 + rr, ac0 + cc)
        prev = state.get(key)
        if prev is None:
            state[key] = (col, 1)
        else:
            prev_col, prev_count = prev
            assert prev_col == col
            state[key] = (col, prev_count + 1)

    open_ports = [(r, c, col) for (r, c), (col, count) in state.items() if count == 1]
    illegal_moves = set()
    for idx, p in enumerate(rest):
        ph, pw = p["shape"]
        for tr, tc, tcol in open_ports:
            for pr, pc, pcol in p["ports"]:
                if pcol != tcol or PRIORITY[pcol] == p["entry_priority"]:
                    continue
                top = tr - pr
                left = tc - pc
                if top < 0 or left < 0 or top + ph > h or left + pw > w:
                    continue
                bad = False
                for rr, cc, col in p["ports"]:
                    gr, gc = top + rr, left + cc
                    prev = state.get((gr, gc))
                    if prev is None:
                        continue
                    prev_col, prev_count = prev
                    if prev_col != col or prev_count >= 2:
                        bad = True
                        break
                if bad:
                    continue
                if _try_place(placed, p["arr"], top, left, bg) is None:
                    continue
                illegal_moves.add((idx, top, left, pcol))
    return len(illegal_moves)


def _mixed_piece_count(grid: np.ndarray) -> int:
    pieces = _extract_pieces(np.asarray(grid, dtype=int), 0)
    count = 0
    for p in pieces:
        if np.any(p["arr"] == 4):
            continue
        if len({col for _r, _c, col in p["ports"]}) >= 2:
            count += 1
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the scattered sprites by respecting both connector-color matching and connector-color priority.

    Input structure:
    - The background is black (0), one sprite contains the fixed yellow (4) anchor block, and the other
      sprites are scattered elsewhere.
    - Connector pixels sit on sprite boundaries, but they can be red (2), blue (1), or green (3).
    - Some pieces carry more than one connector color, so color is part of each piece's internal role.

    Rule:
    - Translate non-anchor sprites without rotation onto the growing assembly.
    - Overlap is legal only when both cells are connector pixels of the same color.
    - If a piece has several connector colors, it must enter the assembly through its highest-priority one:
      red before blue before green. Lower-priority colors on that piece are deferred connectors for later use.

    Key insight:
    - Connector color has dual meaning: it tells you which ports may touch, and on mixed-color pieces it also
      tells you which side is allowed to plug into the structure. The search therefore has to track open
      coordinates, their colors, and which connector on each piece is its legal entry side.
    """
    solutions = _solve(
        np.asarray(grid, dtype=int),
        enforce_phase=False,
        enforce_entry=True,
        max_solutions=2,
    )
    assert len(solutions) == 1, "assembly must have exactly one priority-respecting solution"
    return solutions[0]


# --------------- generator (o5x) ---------------

# Canvas
GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
MAX_ASSEMBLY_DIM = 24

# Pieces
PIECE_COUNT_RANGE = (5, 6)
PIECE_H_RANGE = (4, 7)
PIECE_W_RANGE = (4, 8)
PIECE_DIM_CANDIDATES = 12
PIECE_CAP = 2
BAR_SKIP_PROB = 0.35

# Anchor
ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]

# Scatter
SCATTER_PADDING = 2

# Generation
GEN_RETRY_LIMIT = 500

# Geometry
SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}


def _new_node(kind, h, w, color, stage, top=0, left=0, cap=PIECE_CAP):
    return {
        "kind": kind,
        "shape": (int(h), int(w)),
        "color": int(color),
        "stage": int(stage),
        "top": int(top),
        "left": int(left),
        "cap": int(cap),
        "sides": set(),
        "ports": {},
        "conn_colors": {},
    }


def _ensure_port(node, side):
    if side in node["ports"]:
        return
    h, w = node["shape"]
    span = h if side in ("left", "right") else w
    node["ports"][side] = random.randint(1, span - 3)


def _port_cells(node, side):
    h, w = node["shape"]
    k = node["ports"][side]
    if side == "top":
        return [(0, k), (0, k + 1)]
    if side == "bottom":
        return [(h - 1, k), (h - 1, k + 1)]
    if side == "left":
        return [(k, 0), (k + 1, 0)]
    return [(k, w - 1), (k + 1, w - 1)]


def _global_port(node, side):
    tr, tc = node["top"], node["left"]
    return [(tr + r, tc + c) for r, c in _port_cells(node, side)]


def _bbox_of(node):
    h, w = node["shape"]
    return (node["top"], node["left"], node["top"] + h - 1, node["left"] + w - 1)


def _rects_overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _union_bbox(nodes):
    boxes = [_bbox_of(n) for n in nodes]
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _draw_path(arr, start, end, color, order):
    r, c = start
    er, ec = end
    axes = (
        [(r, er, c, True), (c, ec, er, False)]
        if order == 0
        else [(c, ec, r, False), (r, er, ec, True)]
    )
    for a_start, a_end, fixed, is_row in axes:
        step = 1 if a_end >= a_start else -1
        for v in range(a_start, a_end + step, step):
            rr, cc = (v, fixed) if is_row else (fixed, v)
            if arr[rr, cc] not in CONNECTOR_SET:
                arr[rr, cc] = color


def _paint_bar(arr, side, color, used):
    h, w = arr.shape
    is_horizontal = side in ("top", "bottom")
    fixed = 0 if side in ("top", "left") else (h - 1 if side == "bottom" else w - 1)
    span = w if is_horizontal else h
    lo_conflict = "left" if is_horizontal else "top"
    hi_conflict = "right" if is_horizontal else "bottom"
    for i in range(span):
        if (i == 0 and lo_conflict in used) or (i == span - 1 and hi_conflict in used):
            continue
        if is_horizontal:
            arr[fixed, i] = color
        else:
            arr[i, fixed] = color


def _bar_endpoint(shape, side, hub):
    h, w = shape
    hr, hc = hub
    if side in ("top", "bottom"):
        row = 0 if side == "top" else h - 1
        return (row, min(max(hc, 1), w - 2))
    col = 0 if side == "left" else w - 1
    return (min(max(hr, 1), h - 2), col)


def _make_piece(node):
    h, w = node["shape"]
    color = node["color"]
    used = set(node["sides"])
    arr = np.zeros((h, w), dtype=int)
    hub = (random.randint(1, h - 2), random.randint(1, w - 2))

    bar_sides = [s for s in SIDES if s not in used]
    if len(bar_sides) > 1 and random.random() < BAR_SKIP_PROB:
        bar_sides.remove(random.choice(bar_sides))
    for side in bar_sides:
        _paint_bar(arr, side, color, used)

    terminals = []
    for side in used:
        dr, dc = INWARD[side]
        connector_color = node["conn_colors"][side]
        for r, c in _port_cells(node, side):
            arr[r, c] = connector_color
            rr, cc = r + dr, c + dc
            arr[rr, cc] = color
            terminals.append((rr, cc))

    arr[hub] = color
    order = random.randint(0, 1)
    for t in terminals:
        _draw_path(arr, t, hub, color, order)
    for side in bar_sides:
        _draw_path(arr, hub, _bar_endpoint((h, w), side, hub), color, 1 - order)

    if len(components_mask(arr != 0, connectivity=4)) != 1:
        return None
    if int(np.sum(_is_connector(arr))) != 2 * len(used):
        return None
    return arr


def _make_anchor(node):
    bh, bw = node["block"]
    h, w = node["shape"]
    arr = np.zeros((h, w), dtype=int)
    arr[1 : 1 + bh, 1 : 1 + bw] = 4
    for side in node["sides"]:
        connector_color = node["conn_colors"][side]
        for r, c in _port_cells(node, side):
            arr[r, c] = connector_color
    return arr


def _candidate_fits(nodes, parent_i, child):
    cb = _bbox_of(child)
    if not _rects_overlap(cb, _bbox_of(nodes[parent_i])):
        return False
    for i, other in enumerate(nodes):
        if i == parent_i:
            continue
        if _rects_overlap(cb, _bbox_of(other)):
            return False
    r0, c0, r1, c1 = _union_bbox(nodes + [child])
    return (r1 - r0 + 1) <= MAX_ASSEMBLY_DIM and (c1 - c0 + 1) <= MAX_ASSEMBLY_DIM


def _grow_tree():
    n_pieces = random.randint(*PIECE_COUNT_RANGE)
    bh, bw = random.choice(ANCHOR_BLOCK_CHOICES)
    anchor = _new_node("anchor", bh + 2, bw + 2, 4, stage=0, cap=4)
    anchor["block"] = (bh, bw)
    nodes = [anchor]

    def add_child(parent_i, parent_side, child, color):
        parent = nodes[parent_i]
        incoming = OPPOSITE[parent_side]
        _ensure_port(parent, parent_side)
        _ensure_port(child, incoming)
        pr, pc = _global_port(parent, parent_side)[0]
        cr, cc = _port_cells(child, incoming)[0]
        child["top"] = pr - cr
        child["left"] = pc - cc
        child["sides"].add(incoming)
        if not _candidate_fits(nodes, parent_i, child):
            return None
        parent["sides"].add(parent_side)
        parent["conn_colors"][parent_side] = color
        child["conn_colors"][incoming] = color
        nodes.append(child)
        return len(nodes) - 1

    a = _new_node(
        "piece",
        random.randint(*PIECE_H_RANGE),
        random.randint(*PIECE_W_RANGE),
        random.choice(BODY_COLORS),
        stage=1,
        cap=PIECE_CAP,
    )
    ia = add_child(0, "right", a, 2)
    if ia is None:
        return None

    b = _new_node(
        "piece",
        random.randint(*PIECE_H_RANGE),
        random.randint(*PIECE_W_RANGE),
        random.choice(BODY_COLORS),
        stage=2,
        cap=PIECE_CAP,
    )
    ib = add_child(ia, "right", b, 1)
    if ib is None:
        return None

    c = _new_node(
        "piece",
        random.randint(*PIECE_H_RANGE),
        random.randint(*PIECE_W_RANGE),
        random.choice(BODY_COLORS),
        stage=3,
        cap=1,
    )
    if add_child(ib, "right", c, 3) is None:
        return None

    d = _new_node(
        "piece",
        random.randint(*PIECE_H_RANGE),
        random.randint(*PIECE_W_RANGE),
        random.choice(BODY_COLORS),
        stage=2,
        cap=1,
    )
    if add_child(0, "left", d, 1) is None:
        return None

    e = _new_node(
        "piece",
        random.randint(*PIECE_H_RANGE),
        random.randint(*PIECE_W_RANGE),
        random.choice(BODY_COLORS),
        stage=3,
        cap=1,
    )
    if add_child(0, "top", e, 3) is None:
        return None

    if n_pieces == 6:
        f = _new_node(
            "piece",
            random.randint(*PIECE_H_RANGE),
            random.randint(*PIECE_W_RANGE),
            random.choice(BODY_COLORS),
            stage=1,
            cap=1,
        )
        if add_child(0, "bottom", f, 2) is None:
            return None

    return nodes


def _place_node_on_canvas(out, node, sr, sc):
    arr = node["arr"]
    top, left = node["top"] + sr, node["left"] + sc
    node["pos"] = (top, left)
    port_counts = {}
    mask = arr != 0
    rows, cols = np.nonzero(mask)
    g_rows, g_cols = rows + top, cols + left
    vals = arr[rows, cols]
    cur_vals = out[g_rows, g_cols]

    val_conn = np.isin(vals, CONNECTOR_COLORS)
    cur_conn = np.isin(cur_vals, CONNECTOR_COLORS)
    conflicts = (cur_vals != 0) & ~(val_conn & cur_conn & (vals == cur_vals))
    if np.any(conflicts):
        return None

    for gr, gc, col in zip(g_rows[val_conn], g_cols[val_conn], vals[val_conn]):
        key = (int(gr), int(gc))
        if key in port_counts:
            prev_col, prev_count = port_counts[key]
            if prev_col != int(col):
                return None
            port_counts[key] = (prev_col, prev_count + 1)
        else:
            port_counts[key] = (int(col), 1)

    paint_mask = cur_vals == 0
    out[g_rows[paint_mask], g_cols[paint_mask]] = vals[paint_mask]
    return port_counts


def _assemble(nodes, h, w):
    for node in nodes:
        node["arr"] = _make_anchor(node) if node["kind"] == "anchor" else _make_piece(node)
        if node["arr"] is None:
            return None

    r0, c0, r1, c1 = _union_bbox(nodes)
    if (r1 - r0 + 1) > h or (c1 - c0 + 1) > w:
        return None
    sr = random.randint(-r0, h - 1 - r1)
    sc = random.randint(-c0, w - 1 - c1)

    out = np.zeros((h, w), dtype=int)
    all_port_counts = {}
    for node in nodes:
        rc = _place_node_on_canvas(out, node, sr, sc)
        if rc is None:
            return None
        for key, (col, count) in rc.items():
            prev = all_port_counts.get(key)
            if prev is None:
                all_port_counts[key] = (col, count)
                continue
            prev_col, prev_count = prev
            if prev_col != col:
                return None
            all_port_counts[key] = (col, prev_count + count)

    total_ports = sum(int(np.sum(_is_connector(n["arr"]))) for n in nodes)
    if len(all_port_counts) * 2 != total_ports:
        return None
    if any(count != 2 for _col, count in all_port_counts.values()):
        return None
    if len(components_mask(out != 0, connectivity=4)) != 1:
        return None
    if len({int(v[0]) for v in all_port_counts.values()}) < 3:
        return None
    return out


def _scatter(nodes, h, w):
    inp = np.zeros((h, w), dtype=int)
    anchor = nodes[0]
    paste(inp, anchor["arr"], *anchor["pos"], transparent=0)
    rest = nodes[1:]
    random.shuffle(rest)
    for node in rest:
        try:
            r, c = random_free_location(
                inp,
                node["arr"],
                transparent=0,
                padding=SCATTER_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inp, node["arr"], r, c, transparent=0)
    return inp


def _gen_attempt():
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    nodes = _grow_tree()
    if nodes is None:
        return None
    out = _assemble(nodes, h, w)
    if out is None:
        return None
    inp = _scatter(nodes, h, w)
    if inp is None:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    if _mixed_piece_count(inp) < 2:
        return None
    if _wrong_entry_move_count(inp) == 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cbebaa4b combo input")
