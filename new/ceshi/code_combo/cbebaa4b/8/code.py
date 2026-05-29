# Combo variant of cbebaa4b (index 8)
# combo_spec: D3/2+D13/0
# Name: 线段锚点 + 仅单方向连接

from __future__ import annotations

from pathlib import Path
import sys
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    - Source task: keep the unique yellow anchor fixed, then translate the other
#      sprites so red connector pixels overlap in pairs until one connected output
#      is assembled.
#    - D3/2 changes the anchor from a compact yellow block into a visible straight
#      yellow segment. D13/0 then removes 2D freedom by forcing every connector to
#      live on one axis only.
#    - Fused pixel form used here:
#      - horizontal case: the anchor is a 1x5..7 yellow line, with one 2-cell red
#        vertical port on its left end and one on its right end.
#      - vertical case: the anchor is a 5..7x1 yellow line, with one 2-cell red
#        horizontal port on its top end and one on its bottom end.
#      - every non-anchor piece has red ports only on the same axis sides as the
#        anchor: left/right for the horizontal case, top/bottom for the vertical case.
#    - Concrete trace:
#      - Original behavior on a sample with a central yellow block: a first piece
#        may dock from above, another from the left, then later pieces can branch
#        again in multiple directions.
#      - Fused behavior on a sample with a horizontal yellow line: one chain grows
#        leftward from the line's left end and another chain grows rightward from
#        the line's right end. No piece ever introduces a top/bottom red port, so
#        the assembly can lengthen only along that horizontal axis.
#      - Visible difference: the completed output is a band or corridor centered on
#        the yellow line, not a radial cluster around a yellow block.
#
# b) Complexity calibration
#    - To keep the change visually obvious, the generator fixes anchor degree = 2
#      and splits 4-6 pieces into two chains so both ends of the line are active.
#      This avoids degenerate samples where only one end grows.
#    - To avoid ambiguity, each non-anchor piece has only 1 or 2 ports (2 or 4 red
#      cells total), always on opposite sides of the chosen axis. That preserves the
#      original red-pairing logic while preventing hidden turns or perpendicular docks.
#    - Decorative bars are still allowed on the perpendicular sides so pieces remain
#      visually distinct, but they never create extra red ports.
#
# c) Hidden-assumption audit of the original main()
#    - `_extract_pieces(...)` still holds because the scattered input is still a set
#      of disjoint 4-connected non-black components.
#    - `anchors = [p for p in pieces if np.any(p["arr"] == 4)]` is too weak now.
#      The solver must validate that the yellow cells form a straight line and that
#      the four red cells are on the line's two ends, not on its flanks.
#    - The DFS still assumes translation-only motion and exact red pairing. Those
#      assumptions remain correct after fusion.
#    - The source `_try_place()` allowed equal non-red overlaps via `(arr == sub)`.
#      That was harmless before because colors were typically distinct, but for this
#      fusion the physical rule is stricter: only red-on-red overlap is legal. The
#      solver and generator are both updated to enforce that.
#    - New assumption introduced here: every non-anchor piece exposes red ports only
#      on the anchor axis. `main()` checks that with `_piece_axis_info(...)`, and the
#      generator constructs every piece to satisfy it.
#
# d) Placement legality
#    - Generation stays output-first: build a valid assembled target first, then
#      scatter all non-anchor sprites away from the fixed anchor.
#    - In the assembled output, a child may overlap its parent only at paired red
#      cells; all other occupied/occupied collisions are rejected.
#    - Child candidates are rejected if they would leave the local canvas union,
#      overlap a non-parent bbox, or make the total assembled bbox exceed the grid.
#
# e) Visual observability
#    - Across examples, a human sees one consistent anchor family: a thin yellow
#      segment, never a yellow block.
#    - The important fused cue is also visible without hidden metadata: all red
#      connector pairs on the ordinary pieces lie on the same axis as the yellow
#      line, and the finished structure extends from the line's two ends.
#    - No decision depends on color ordering, numeric ids, or hidden generation-time
#      labels; the rule is carried by shape, port placement, and preserved anchor
#      position.
#
# f) Visual cleanliness
#    - Because only red cells may overlap in the output, the assembled structure
#      remains legible even when the chains get dense.
#    - Restricting growth to one axis reduces central clutter compared with the D3/2
#      single variant that grows from the flanks of the line.
#
# g) Change inferability
#    - There is no color remapping puzzle here. Yellow still means anchor and red
#      still means connector.
#    - The inferable new rule is geometric: identify the yellow line, notice that
#      the red ports are on its ends, then notice that all other sprites can only
#      dock along that same axis.
#
# h) Behavior metrics used after implementation
#    - Metric 1: axis purity. After excluding the anchor, 100% of scattered sprites
#      must have red ports only on the anchor axis.
#    - Metric 2: two-ended growth. In the solved output, there must be non-anchor
#      pixels strictly beyond both ends of the yellow line along the chosen axis.
#    - Metric 3: anchor orientation mix. Generated samples should include both the
#      horizontal and vertical line-anchor cases over repeated trials.
#
# i) New solver assumptions and generator guarantees
#    - `main()` assumes there is exactly one valid line-anchor. `_make_anchor()` plus
#      `_line_anchor_info(...)` guarantee and verify that.
#    - `main()` assumes every non-anchor piece has 2 or 4 red cells confined to one
#      axis. `_make_piece()` and `_piece_axis_info(...)` enforce and verify that.
#    - `main()` assumes the final assembly is unique. `_gen_attempt()` checks this by
#      requiring `main(input) == assembled_output` before accepting a sample.


SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}
AXIS_SIDES = {"horizontal": ("left", "right"), "vertical": ("top", "bottom")}


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        reds = np.argwhere(arr == 2)
        pieces.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "reds": [(int(r), int(c)) for r, c in reds],
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

    conflict = mask & (sub != bg) & ~((arr == 2) & (sub == 2))
    if np.any(conflict):
        return None

    out = canvas.copy()
    out_sub = out[top : top + ph, left : left + pw]
    out_sub[mask] = arr[mask]
    out[top : top + ph, left : left + pw] = out_sub
    return out


def _line_anchor_info(arr: np.ndarray) -> dict | None:
    yellow = np.argwhere(arr == 4)
    if len(yellow) < 5:
        return None

    r0, c0, r1, c1 = bbox_coords(yellow)
    if (r0 == r1) == (c0 == c1):
        return None
    if np.any(arr[r0 : r1 + 1, c0 : c1 + 1] != 4):
        return None
    if int(np.sum(arr == 4)) != (r1 - r0 + 1) * (c1 - c0 + 1):
        return None

    reds = np.argwhere(arr == 2)
    if len(reds) != 4:
        return None

    if r0 == r1:
        cols = sorted({int(c) for _, c in reds})
        if cols != [c0 - 1, c1 + 1]:
            return None
        for cc in cols:
            rows = sorted(int(r) for r, c in reds if int(c) == cc)
            if rows != [r0, r0 + 1]:
                return None
        return {"axis": "horizontal", "yellow_bbox": (r0, c0, r1, c1)}

    rows = sorted({int(r) for r, _ in reds})
    if rows != [r0 - 1, r1 + 1]:
        return None
    for rr in rows:
        cols = sorted(int(c) for r, c in reds if int(r) == rr)
        if cols != [c0, c0 + 1]:
            return None
    return {"axis": "vertical", "yellow_bbox": (r0, c0, r1, c1)}


def _piece_axis_info(arr: np.ndarray, expected_axis: str | None = None) -> dict | None:
    reds = np.argwhere(arr == 2)
    if len(reds) not in (2, 4):
        return None

    h, w = arr.shape
    groups: dict[str, list[tuple[int, int]]] = {}
    for rr, cc in reds:
        sides = []
        if int(rr) == 0:
            sides.append("top")
        if int(rr) == h - 1:
            sides.append("bottom")
        if int(cc) == 0:
            sides.append("left")
        if int(cc) == w - 1:
            sides.append("right")
        if len(sides) != 1:
            return None
        groups.setdefault(sides[0], []).append((int(rr), int(cc)))

    used_sides = []
    for side, cells in groups.items():
        if len(cells) != 2:
            return None
        rows = sorted(r for r, _ in cells)
        cols = sorted(c for _, c in cells)
        if side in ("top", "bottom"):
            if rows[0] != rows[1] or cols[1] != cols[0] + 1:
                return None
        else:
            if cols[0] != cols[1] or rows[1] != rows[0] + 1:
                return None
        used_sides.append(side)

    if expected_axis is None:
        if set(used_sides).issubset({"left", "right"}):
            axis = "horizontal"
        elif set(used_sides).issubset({"top", "bottom"}):
            axis = "vertical"
        else:
            return None
    else:
        axis = expected_axis
        if not set(used_sides).issubset(set(AXIS_SIDES[axis])):
            return None

    return {"axis": axis, "used_sides": tuple(sorted(used_sides)), "port_count": len(reds)}


def _anchor_global_yellow_bbox(anchor: dict) -> tuple[int, int, int, int]:
    br0, bc0, _, _ = anchor["bbox"]
    yr0, yc0, yr1, yc1 = anchor["anchor_info"]["yellow_bbox"]
    return br0 + yr0, bc0 + yc0, br0 + yr1, bc0 + yc1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered connector sprites around a fixed yellow line-anchor whose ports only open along one axis.

    Input structure:
    - bg = black (0). Several disconnected colored sprites are scattered on the grid.
    - Exactly one sprite contains yellow (4) cells arranged as a straight horizontal or vertical line, with one red (2) connector pair at each end of that line.
    - Every other sprite has either one or two 2-cell red connector pairs, and all of those pairs lie on the same axis as the yellow line.

    Rule:
    - Keep the yellow line-anchor at its input position.
    - Translate each remaining sprite, without rotation or reflection, so its red connector pixels overlap currently unpaired red pixels on the growing assembly.
    - Only red-on-red overlap is legal, and every original red pixel must end up paired exactly once to form the completed connected structure.

    Key insight: The red-port pairing mechanism is unchanged, but the topology is no longer radial. The yellow seed is a line with ports on its two ends, so valid growth is constrained to two opposite directions along one shared axis.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must have at least one colored piece"

    anchors = []
    for piece in pieces:
        info = _line_anchor_info(piece["arr"])
        if info is not None:
            piece["anchor_info"] = info
            anchors.append(piece)
    assert len(anchors) == 1, "input must have exactly one yellow line-anchor piece"
    anchor = anchors[0]
    axis = anchor["anchor_info"]["axis"]
    rest = [p for p in pieces if p is not anchor]

    for piece in rest:
        info = _piece_axis_info(piece["arr"], axis)
        assert info is not None, "all non-anchor pieces must use only the anchor axis for red ports"
        piece["axis_info"] = info

    canvas = np.full((h, w), bg, dtype=int)
    ar0, ac0, _, _ = anchor["bbox"]
    placed = _try_place(canvas, anchor["arr"], ar0, ac0, bg)
    assert placed is not None, "anchor piece must fit on canvas at its original position"

    total_ports = int(sum(len(p["reds"]) for p in pieces))
    assert total_ports % 2 == 0, "total red connector ports must be even for complete pairing"
    target_unique_red = total_ports // 2

    red_counts: dict[tuple[int, int], int] = {}
    for rr, cc in anchor["reds"]:
        key = (ar0 + rr, ac0 + cc)
        red_counts[key] = red_counts.get(key, 0) + 1

    solutions: list[np.ndarray] = []

    def dfs(cur: np.ndarray, remaining: list[dict], rc: dict[tuple[int, int], int]) -> None:
        if len(solutions) >= 2:
            return

        if not remaining:
            if len(rc) == target_unique_red and all(v == 2 for v in rc.values()):
                solutions.append(cur)
            return

        if len(rc) > target_unique_red:
            return

        unpaired = [k for k, v in rc.items() if v == 1]
        if not unpaired:
            return

        rc1 = {k for k, v in rc.items() if v == 1}

        best_moves = None
        for tr, tc in unpaired:
            moves = []
            for idx, piece in enumerate(remaining):
                ph, pw = piece["shape"]
                for pr, pc in piece["reds"]:
                    top = tr - pr
                    left = tc - pc
                    if top < 0 or left < 0 or top + ph > h or left + pw > w:
                        continue
                    bad = False
                    overlap = 0
                    for rr, cc in piece["reds"]:
                        gr, gc = top + rr, left + cc
                        prev = rc.get((gr, gc), 0)
                        if prev >= 2:
                            bad = True
                            break
                        if (gr, gc) in rc1:
                            overlap += 1
                    if not bad:
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
            piece = remaining[idx]
            next_remaining = remaining[:idx] + remaining[idx + 1 :]
            next_rc = dict(rc)
            for rr, cc in piece["reds"]:
                gr, gc = top + rr, left + cc
                next_rc[(gr, gc)] = next_rc.get((gr, gc), 0) + 1

            nxt = _try_place(cur, piece["arr"], top, left, bg)
            if nxt is None:
                continue
            dfs(nxt, next_remaining, next_rc)
            if len(solutions) >= 2:
                return

    dfs(placed, rest, red_counts)
    assert len(solutions) == 1, "jigsaw assembly must have exactly one valid solution"
    return solutions[0]


# --------------- generator ---------------

# Canvas
GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
MAX_ASSEMBLY_DIM = 24

# Pieces
PIECE_COUNT_RANGE = (4, 6)
PIECE_H_RANGE = (4, 7)
PIECE_W_RANGE = (4, 8)
PIECE_DIM_CANDIDATES = 12
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]
BAR_SKIP_PROB = 0.35

# Anchor
ANCHOR_ORIENTATIONS = ("horizontal", "vertical")
ANCHOR_LINE_LENGTHS = (5, 6, 7)

# Scatter
SCATTER_PADDING = 2

# Generation
GEN_RETRY_LIMIT = 400


def _new_node(kind, h, w, color, axis, top=0, left=0):
    return {
        "kind": kind,
        "shape": (int(h), int(w)),
        "color": int(color),
        "axis": axis,
        "top": int(top),
        "left": int(left),
        "sides": set(),
        "ports": {},
    }


def _ensure_port(node, side):
    if node["kind"] == "anchor" or side in node["ports"]:
        return
    h, w = node["shape"]
    if side in ("left", "right"):
        lo, hi = 1, h - 3
    else:
        lo, hi = 1, w - 3
    node["ports"][side] = (lo + hi) // 2


def _port_cells(node, side):
    if node["kind"] == "anchor":
        h, w = node["shape"]
        if node["axis"] == "horizontal":
            if side == "left":
                return [(1, 0), (2, 0)]
            return [(1, w - 1), (2, w - 1)]
        if side == "top":
            return [(0, 1), (0, 2)]
        return [(h - 1, 1), (h - 1, 2)]

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
    boxes = [_bbox_of(node) for node in nodes]
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
            if arr[rr, cc] != 2:
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


def _decorative_sides(node) -> list[str]:
    axis_sides = set(AXIS_SIDES[node["axis"]])
    used = set(node["sides"])
    bars = [side for side in SIDES if side not in axis_sides and side not in used]
    free_axis = [side for side in AXIS_SIDES[node["axis"]] if side not in used]
    if len(used) == 1 and free_axis:
        bars.append(free_axis[0])
    return bars


def _make_piece(node):
    h, w = node["shape"]
    color = node["color"]
    used = set(node["sides"])
    arr = np.zeros((h, w), dtype=int)
    hub = (min(max(h // 2, 1), h - 2), min(max(w // 2, 1), w - 2))

    bar_sides = _decorative_sides(node)
    if len(bar_sides) > 1 and random.random() < BAR_SKIP_PROB:
        bar_sides.remove(random.choice(bar_sides))
    for side in bar_sides:
        _paint_bar(arr, side, color, used)

    terminals = []
    for side in used:
        _ensure_port(node, side)
        dr, dc = INWARD[side]
        for r, c in _port_cells(node, side):
            arr[r, c] = 2
            rr, cc = r + dr, c + dc
            arr[rr, cc] = color
            terminals.append((rr, cc))

    arr[hub] = color
    order = random.randint(0, 1)
    for terminal in terminals:
        _draw_path(arr, terminal, hub, color, order)
    for side in bar_sides:
        _draw_path(arr, hub, _bar_endpoint((h, w), side, hub), color, 1 - order)

    if len(components_mask(arr != 0, connectivity=4)) != 1:
        return None
    if _piece_axis_info(arr, node["axis"]) is None:
        return None
    return arr


def _make_anchor(node):
    h, w = node["shape"]
    arr = np.zeros((h, w), dtype=int)
    line_len = node["line_len"]
    if node["axis"] == "horizontal":
        arr[1, 1 : 1 + line_len] = 4
    else:
        arr[1 : 1 + line_len, 1] = 4
    for side in AXIS_SIDES[node["axis"]]:
        for r, c in _port_cells(node, side):
            arr[r, c] = 2
    if len(components_mask(arr != 0, connectivity=4)) != 1:
        return None
    if _line_anchor_info(arr) is None:
        return None
    return arr


def _candidate_fits(nodes, parent_i, child):
    child_box = _bbox_of(child)
    if not _rects_overlap(child_box, _bbox_of(nodes[parent_i])):
        return False
    for idx, other in enumerate(nodes):
        if idx == parent_i:
            continue
        if _rects_overlap(child_box, _bbox_of(other)):
            return False
    r0, c0, r1, c1 = _union_bbox(nodes + [child])
    return (r1 - r0 + 1) <= MAX_ASSEMBLY_DIM and (c1 - c0 + 1) <= MAX_ASSEMBLY_DIM


def _attach_child(nodes, parent_i, color, parent_side, child_side_set):
    dims = [
        (random.randint(*PIECE_H_RANGE), random.randint(*PIECE_W_RANGE))
        for _ in range(PIECE_DIM_CANDIDATES)
    ]
    random.shuffle(dims)
    axis = nodes[0]["axis"]
    for ph, pw in dims:
        child = _new_node("piece", ph, pw, color, axis)
        child["sides"].update(child_side_set)
        incoming = OPPOSITE[parent_side]
        assert incoming in child["sides"], "child must expose the incoming axis side"
        parent = nodes[parent_i]
        parent["sides"].add(parent_side)
        _ensure_port(parent, parent_side)
        _ensure_port(child, incoming)
        pr, pc = _global_port(parent, parent_side)[0]
        cr, cc = _port_cells(child, incoming)[0]
        child["top"] = pr - cr
        child["left"] = pc - cc
        if not _candidate_fits(nodes, parent_i, child):
            continue
        nodes.append(child)
        return len(nodes) - 1
    return None


def _grow_tree():
    axis = random.choice(ANCHOR_ORIENTATIONS)
    line_len = random.choice(ANCHOR_LINE_LENGTHS)
    ah, aw = (4, line_len + 2) if axis == "horizontal" else (line_len + 2, 4)
    anchor = _new_node("anchor", ah, aw, 4, axis)
    anchor["line_len"] = line_len
    nodes = [anchor]

    n_pieces = random.randint(*PIECE_COUNT_RANGE)
    colors = random.sample(PIECE_COLORS, n_pieces)
    left_branch_len = n_pieces // 2
    right_branch_len = n_pieces - left_branch_len
    branch_sides = AXIS_SIDES[axis]
    branch_lengths = [left_branch_len, right_branch_len]
    if random.random() < 0.5:
        branch_lengths.reverse()

    color_idx = 0
    for branch_side, branch_len in zip(branch_sides, branch_lengths):
        parent_i = 0
        outgoing_side = branch_side
        incoming_side = OPPOSITE[branch_side]
        for step in range(branch_len):
            child_sides = {incoming_side}
            if step < branch_len - 1:
                child_sides.add(outgoing_side)
            idx = _attach_child(nodes, parent_i, colors[color_idx], branch_side, child_sides)
            if idx is None:
                return None
            parent_i = idx
            color_idx += 1
    return nodes


def _place_node_on_canvas(out, node, sr, sc):
    arr = node["arr"]
    top, left = node["top"] + sr, node["left"] + sc
    node["pos"] = (top, left)
    red_counts = {}
    mask = arr != 0
    rows, cols = np.nonzero(mask)
    g_rows, g_cols = rows + top, cols + left
    vals = arr[rows, cols]
    cur_vals = out[g_rows, g_cols]
    conflicts = (cur_vals != 0) & ~((cur_vals == 2) & (vals == 2))
    if np.any(conflicts):
        return None
    red_mask = vals == 2
    for gr, gc in zip(g_rows[red_mask], g_cols[red_mask]):
        key = (int(gr), int(gc))
        red_counts[key] = red_counts.get(key, 0) + 1
    paint_mask = cur_vals == 0
    out[g_rows[paint_mask], g_cols[paint_mask]] = vals[paint_mask]
    return red_counts


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
    all_red_counts = {}
    for node in nodes:
        rc = _place_node_on_canvas(out, node, sr, sc)
        if rc is None:
            return None
        for key, value in rc.items():
            all_red_counts[key] = all_red_counts.get(key, 0) + value

    total_ports = sum(int(np.sum(node["arr"] == 2)) for node in nodes)
    if len(all_red_counts) * 2 != total_ports:
        return None
    if any(value != 2 for value in all_red_counts.values()):
        return None
    if len(components_mask(out != 0, connectivity=4)) != 1:
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


def _fusion_metrics(inp: np.ndarray) -> dict:
    pieces = _extract_pieces(inp, 0)
    anchors = []
    for piece in pieces:
        info = _line_anchor_info(piece["arr"])
        if info is not None:
            piece["anchor_info"] = info
            anchors.append(piece)
    if len(anchors) != 1:
        raise ValueError("metric computation requires exactly one anchor")
    anchor = anchors[0]
    axis = anchor["anchor_info"]["axis"]
    rest = [p for p in pieces if p is not anchor]
    axis_pure = sum(_piece_axis_info(p["arr"], axis) is not None for p in rest) / max(len(rest), 1)

    solved = main(inp)
    yr0, yc0, yr1, yc1 = _anchor_global_yellow_bbox(anchor)
    if axis == "horizontal":
        extends_neg = bool(np.any(solved[:, : max(yc0 - 1, 0)] != 0))
        extends_pos = bool(np.any(solved[:, yc1 + 2 :] != 0))
    else:
        extends_neg = bool(np.any(solved[: max(yr0 - 1, 0), :] != 0))
        extends_pos = bool(np.any(solved[yr1 + 2 :, :] != 0))
    return {
        "axis": axis,
        "axis_pure": float(axis_pure),
        "extends_both_ends": bool(extends_neg and extends_pos),
    }


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
        metrics = _fusion_metrics(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    if metrics["axis_pure"] != 1.0 or not metrics["extends_both_ends"]:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cbebaa4b combo input")
