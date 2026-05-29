# Variant of cbebaa4b
# Dimension: D1 — 连接器匹配条件 (match_cond)
# Change: 精灵的红色像素与目标（中央黄色方块或已连接精灵）的红色像素对齐，形成连接... → 多色连接器匹配

# Design analysis for this D1 variant
# -----------------------------------
# concretization:
# - Keep the original scene template: one fixed yellow anchor plus 4-6 scattered sprites.
# - The only semantic change is at connector pixels. Instead of every connector being red(2),
#   each tree edge is assigned one connector color from a small visible palette {2,3,8}.
# - A piece with degree 2 can therefore show two different connector colors on different sides.
# - In the output, an overlap is legal only when both overlapping cells are connector pixels of
#   the same color. Red may connect to red, green to green, teal to teal; cross-color overlap is illegal.
# - Small trace:
#   original: anchor has red ports on top/right; a piece with one red side may attach to either if geometry fits.
#   variant: anchor top may be green while right is teal; the piece side colored green may only align to
#   the green target, even if the teal target is geometrically compatible. The search state must therefore
#   track both coordinate and connector color.
#
# complexity calibration:
# - Restrict connector colors to exactly three visible colors so the rule is learnable from 2-4 examples.
# - Each generated instance uses at least two connector colors, and every used color appears on at least one edge.
# - Body colors exclude connector colors, so connector pixels remain visually salient and are not confused
#   with decorative paths.
#
# implicit assumption audit against the original solver:
# - _extract_pieces originally used arr == 2. That assumes every port is red; now it must collect any cell
#   in CONNECTOR_COLORS and preserve the per-port color.
# - _try_place originally allowed overlap when arr == sub, which was harmless only because piece body colors
#   were unique and only red-red overlap was intended. With repeated body colors now allowed, same-color body
#   overlap must still be rejected; only same-color connector overlap is legal.
# - main() originally stored red_counts keyed only by coordinate. That loses information in the variant,
#   because an unpaired green port at (r,c) is a different constraint from an unpaired teal port at (r,c).
#   The new state stores {coord: (connector_color, count)}.
# - DFS move generation originally tried every red port against every unpaired red coordinate. The variant
#   must filter candidate ports by matching connector color before even considering geometry.
#
# placement legality:
# - Generator still builds the assembled structure first and scatters afterward.
# - Child placement in local coordinates is unchanged: a child is translated so one side's connector cells
#   coincide with the parent's connector cells.
# - During assembly validation, only same-color connector overlaps are accepted; all other non-bg overlaps
#   are rejected. This keeps the output compatible with the solver's legality test.
#
# visual observability and cleanliness:
# - Connector colors are limited to {2,3,8}; anchor stays yellow(4); body colors come from {1,5,6,7,9}.
# - This separation keeps connector pixels easy to spot and ensures the D1 change is visible in both input
#   and output without adding new symbolic markings or clutter.
# - Decorative bars and internal paths remain in the piece body color, so the multi-color signal stays
#   concentrated exactly on the matching interfaces.
#
# new solver assumptions and generator guarantees:
# - Assumption: connector colors are drawn only from CONNECTOR_COLORS. Generator guarantees this.
# - Assumption: non-connector body pixels never use connector colors. Generator guarantees this by drawing
#   body pixels from BODY_COLORS only.
# - Assumption: every legal assembled connector coordinate is occupied by exactly two same-color ports.
#   Generator verifies this in _assemble(), and main() asserts uniqueness of the full solution.

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


CONNECTOR_COLORS = (2, 3, 8)
CONNECTOR_SET = set(CONNECTOR_COLORS)
BODY_COLORS = [1, 5, 6, 7, 9]


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered sprites around a fixed yellow anchor by matching connector colors as well as positions.

    Input structure:
    - bg = black (0). Several colored sprites are scattered on the grid.
    - Exactly one anchor sprite contains a yellow (4) rectangle and stays fixed.
    - Connector pixels lie on sprite edges, but they can now be red (2), green (3), or teal (8).

    Rule:
    - Translate the non-anchor sprites one at a time onto the growing assembly.
    - A move is legal only if overlapping pixels are connector pixels of the same color.
    - Every connector pixel must end up paired exactly once with another connector of the same color, producing a single completed structure.

    Key insight: Connector pixels still behave like jigsaw pegs, but now each peg also carries a color label, so the search must track both the location of open ports and their colors.
    """
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
            assert prev_col == col, "anchor cannot place different connector colors on one cell"
            port_state[key] = (col, prev_count + 1)

    solutions: list[np.ndarray] = []

    def dfs(
        cur: np.ndarray,
        remaining: list[dict],
        state: dict[tuple[int, int], tuple[int, int]],
    ) -> None:
        if len(solutions) >= 2:
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

        best_moves = None
        for tr, tc, tcol in unpaired:
            moves = []
            for idx, p in enumerate(remaining):
                ph, pw = p["shape"]
                for pr, pc, pcol in p["ports"]:
                    if pcol != tcol:
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
                        if prev_count == 1:
                            overlap += 1
                    if bad:
                        continue
                    moves.append((idx, top, left, overlap))
            if not moves:
                return
            if best_moves is None or len(moves) < len(best_moves):
                best_moves = moves
                if len(best_moves) == 1:
                    break

        assert best_moves is not None, "there must be at least one feasible connector move"

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
            if len(solutions) >= 2:
                return

    dfs(placed, rest, port_state)
    assert len(solutions) == 1, "jigsaw assembly must have exactly one valid solution"
    return solutions[0]


# --------------- generator (o5x) ---------------

# Canvas
GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
MAX_ASSEMBLY_DIM = 24

# Pieces
PIECE_COUNT_RANGE = (4, 6)
PIECE_H_RANGE = (4, 7)
PIECE_W_RANGE = (4, 8)
PIECE_DIM_CANDIDATES = 12
PIECE_CAP = 2
BAR_SKIP_PROB = 0.35

# Anchor
ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]
ANCHOR_CAP_CHOICES = [1, 2]

# Scatter
SCATTER_PADDING = 2

# Generation
GEN_RETRY_LIMIT = 300

# Side geometry lookups
SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}


def _new_node(kind, h, w, color, top=0, left=0, cap=PIECE_CAP):
    return {
        "kind": kind,
        "shape": (int(h), int(w)),
        "color": int(color),
        "top": int(top),
        "left": int(left),
        "cap": int(cap),
        "sides": set(),
        "ports": {},
        "conn_colors": {},
    }


def _ensure_port(node, side):
    """Lazily assign a random port offset for *side* if not already set."""
    if side in node["ports"]:
        return
    h, w = node["shape"]
    span = h if side in ("left", "right") else w
    node["ports"][side] = random.randint(1, span - 3)


def _port_cells(node, side):
    """Return the two local (r, c) cells that form the connector on *side*."""
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
    """Draw an L-shaped path from *start* to *end*, preserving connector pixels."""
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
    """Paint a decorative bar along *side*, skipping corners claimed by *used* sides."""
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
    """Render a piece sprite array with colored connectors and decorative bars."""
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
    """Check child overlaps only its parent and total assembly stays within bounds."""
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


def _assign_edge_colors(nodes, edges):
    active_count = random.randint(2, min(len(CONNECTOR_COLORS), len(edges)))
    active_colors = random.sample(CONNECTOR_COLORS, active_count)
    edge_colors = active_colors[:] + [
        random.choice(active_colors) for _ in range(len(edges) - active_count)
    ]
    random.shuffle(edge_colors)
    for (parent_i, parent_side, child_i, child_side), color in zip(edges, edge_colors):
        nodes[parent_i]["conn_colors"][parent_side] = color
        nodes[child_i]["conn_colors"][child_side] = color


def _grow_tree():
    """Build a tree of connected nodes (1 anchor + N pieces) in local coordinates."""
    bh, bw = random.choice(ANCHOR_BLOCK_CHOICES)
    anchor = _new_node("anchor", bh + 2, bw + 2, 4, cap=random.choice(ANCHOR_CAP_CHOICES))
    anchor["block"] = (bh, bw)
    nodes = [anchor]
    edges = []
    n_pieces = random.randint(*PIECE_COUNT_RANGE)

    for _ in range(n_pieces):
        placed = False
        dims = [
            (random.randint(*PIECE_H_RANGE), random.randint(*PIECE_W_RANGE))
            for _ in range(PIECE_DIM_CANDIDATES)
        ]
        random.shuffle(dims)
        for ph, pw in dims:
            parents = [j for j, n in enumerate(nodes) if len(n["sides"]) < n["cap"]]
            random.shuffle(parents)
            for parent_i in parents:
                parent = nodes[parent_i]
                sides = [s for s in SIDES if s not in parent["sides"]]
                random.shuffle(sides)
                for side in sides:
                    child = _new_node("piece", ph, pw, random.choice(BODY_COLORS))
                    incoming = OPPOSITE[side]
                    _ensure_port(parent, side)
                    _ensure_port(child, incoming)
                    pr, pc = _global_port(parent, side)[0]
                    cr, cc = _port_cells(child, incoming)[0]
                    child["top"] = pr - cr
                    child["left"] = pc - cc
                    child["sides"].add(incoming)
                    if not _candidate_fits(nodes, parent_i, child):
                        continue
                    parent["sides"].add(side)
                    nodes.append(child)
                    child_i = len(nodes) - 1
                    edges.append((parent_i, side, child_i, incoming))
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None

    _assign_edge_colors(nodes, edges)
    return nodes


def _place_node_on_canvas(out, node, sr, sc):
    """Stamp a single node's sprite onto *out*, tracking same-color connector overlaps."""
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
    """Render all nodes, shift into grid, and validate the assembled output."""
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
    if len(set(int(v[0]) for v in all_port_counts.values())) < 2:
        return None
    return out


def _scatter(nodes, h, w):
    """Place anchor at its assembled position, then scatter remaining pieces."""
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
    if np.array_equal(solved, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cbebaa4b input")
