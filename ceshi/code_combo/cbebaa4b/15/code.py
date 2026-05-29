# Combo variant of cbebaa4b (index 15)
# combo_spec: D7/0+D15/1
# Name: 由外向内组装 + 精灵沿边缘排列

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste


# -----------------------------------------------------------------------------
# Design analysis for the combo
#
# a) Concrete behavior trace
# - Original task: pieces are randomly scattered, so input position is irrelevant.
#   The solver can ignore geometry and backtrack only on red-connector matches.
# - This combo changes two linked stages together:
#   1. Input layout: every non-anchor sprite is flush with exactly one grid edge.
#      The six sprites occupy the four sides with counts [1, 1, 2, 2]. The doubled
#      sides are opposite each other, so the eye sees two deep "arms" and two
#      shallow "arms".
#   2. Assembly order: within a doubled side, the border sprite farther from the
#      center (usually nearer a corner) is the outer sprite, and the nearer one is
#      the inner sprite. The mapping therefore contracts from edge to center.
# - Small example: if the doubled axis is top/bottom, then top has two pieces on
#   row 0. The one nearer the corner becomes the top leaf; the one nearer the top
#   midpoint becomes the top inner segment attached next to the anchor. The same
#   pattern repeats on the bottom side.
#
# b) Complexity calibration
# - A fully generic tree plus edge-order constraints would create many fragile
#   cases. To keep the pattern visible and learnable, generation uses a controlled
#   arm layout: all four sides are used, and exactly one opposite pair has depth 2.
# - This still leaves enough freedom in sizes, colors, ports, and exact edge
#   positions to avoid repetition, while keeping the side/depth pattern obvious in
#   2-4 examples.
#
# c) Implicit assumptions broken from the original solver
# - Original assumption: input scatter is arbitrary, so spatial position contains
#   no task signal. Broken here: border side and border distance are now essential.
# - Original assumption: piece order can be discovered solely by connector
#   feasibility. Broken here: opposite inner segments can be connector-compatible
#   on either opposite side, so the input border side disambiguates them.
# - Original assumption: generic DFS can choose any remaining piece next. Broken
#   here: doubled sides are interpreted in descending input distance order.
#
# d) Placement legality
# - The generator constructs the final assembly first, validates overlap/pairing,
#   then redistributes sprites onto edges while keeping every piece in-bounds,
#   non-overlapping, and away from corners enough to touch exactly one border.
#
# e) Visual observability
# - Humans can see two things directly: every sprite starts on a border, and the
#   doubled sides show a "farther near corner / nearer near midpoint" ordering.
#   The output then places those same sprites as outer vs inner arm segments.
#
# f) Visual cleanliness
# - Only one anchor sits near the center; the input uses one border per piece; the
#   output uses four orthogonal arms. This keeps regions separated and readable.
#
# g) New solver assumptions and generator guarantees
# - Exactly six non-anchor pieces exist.
# - Each piece touches exactly one border in the input.
# - Border-side counts are [1, 1, 2, 2], and the doubled sides are opposite.
# - On a doubled side, the farther input sprite has one connector side, and the
#   nearer input sprite has two connector sides. The generator enforces this.
# -----------------------------------------------------------------------------


SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}


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
    conflict = mask & (sub != bg) & ~(((arr == 2) & (sub == 2)) | (arr == sub))
    if np.any(conflict):
        return None
    out = canvas.copy()
    out_sub = out[top : top + ph, left : left + pw]
    out_sub[mask] = arr[mask]
    out[top : top + ph, left : left + pw] = out_sub
    return out


def _connector_sides(arr: np.ndarray) -> set[str]:
    h, w = arr.shape
    out = set()
    if np.any(arr[0, :] == 2):
        out.add("top")
    if np.any(arr[h - 1, :] == 2):
        out.add("bottom")
    if np.any(arr[:, 0] == 2):
        out.add("left")
    if np.any(arr[:, w - 1] == 2):
        out.add("right")
    return out


def _side_reds(arr: np.ndarray, side: str) -> list[tuple[int, int]]:
    h, w = arr.shape
    if side == "top":
        pts = [(0, c) for c in range(w) if arr[0, c] == 2]
    elif side == "bottom":
        pts = [(h - 1, c) for c in range(w) if arr[h - 1, c] == 2]
    elif side == "left":
        pts = [(r, 0) for r in range(h) if arr[r, 0] == 2]
    else:
        pts = [(r, w - 1) for r in range(h) if arr[r, w - 1] == 2]
    pts.sort()
    return pts


def _touching_side(piece: dict, h: int, w: int) -> str:
    r0, c0, r1, c1 = piece["bbox"]
    sides = []
    if r0 == 0:
        sides.append("top")
    if r1 == h - 1:
        sides.append("bottom")
    if c0 == 0:
        sides.append("left")
    if c1 == w - 1:
        sides.append("right")
    assert len(sides) == 1, "each piece must touch exactly one border"
    return sides[0]


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = bbox
    return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)


def _center_distance(piece: dict, anchor_center: tuple[float, float]) -> float:
    pr, pc = _bbox_center(piece["bbox"])
    ar, ac = anchor_center
    return (pr - ar) ** 2 + (pc - ac) ** 2


def _global_side_reds(arr: np.ndarray, top: int, left: int, side: str) -> list[tuple[int, int]]:
    return [(top + r, left + c) for r, c in _side_reds(arr, side)]


def _attach_by_side(
    canvas: np.ndarray,
    piece: dict,
    target_reds: list[tuple[int, int]],
    side_on_piece: str,
    bg: int,
) -> tuple[np.ndarray, tuple[int, int]] | None:
    local_reds = _side_reds(piece["arr"], side_on_piece)
    if len(local_reds) != 2 or len(target_reds) != 2:
        return None
    top = target_reds[0][0] - local_reds[0][0]
    left = target_reds[0][1] - local_reds[0][1]
    nxt = _try_place(canvas, piece["arr"], top, left, bg)
    if nxt is None:
        return None
    if _global_side_reds(piece["arr"], top, left, side_on_piece) != target_reds:
        return None
    return nxt, (int(top), int(left))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble edge-lined sprites into a central anchored figure by reading both connector geometry and border distance.

    Input structure:
    - bg = black (0). One yellow anchor sits near the center and stays fixed.
    - Six non-anchor sprites sit flush against the four outer borders; each sprite touches exactly one edge.
    - One opposite pair of edges carries two sprites each, so those sides visually show a farther sprite near a corner and a nearer sprite closer to the midpoint.

    Rule:
    - Group sprites by which border they touch in the input.
    - On doubled borders, sort those two sprites by distance from the yellow anchor: farther border sprite becomes the outer arm segment, nearer border sprite becomes the inner segment.
    - Rebuild the final connected object around the anchor by attaching every side-specific segment with matching red connector pairs.

    Key insight:
    - Red pixels still define legal joins, but they are no longer sufficient by themselves: the input edge side tells you which arm a sprite belongs to, and the farther-versus-nearer ordering on a doubled edge tells you which sprite is outer versus inner.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must contain pieces"

    anchors = [p for p in pieces if np.any(p["arr"] == 4)]
    assert len(anchors) == 1, "there must be exactly one anchor"
    anchor = anchors[0]
    rest = [p for p in pieces if p is not anchor]
    assert len(rest) == 6, "combo variant expects exactly six non-anchor pieces"

    anchor_center = _bbox_center(anchor["bbox"])
    groups: dict[str, list[dict]] = {side: [] for side in SIDES}
    for p in rest:
        p["input_side"] = _touching_side(p, h, w)
        p["dist2"] = _center_distance(p, anchor_center)
        p["connector_sides"] = _connector_sides(p["arr"])
        groups[p["input_side"]].append(p)

    counts = sorted(len(v) for v in groups.values())
    assert counts == [1, 1, 2, 2], "edge groups must have counts [1,1,2,2]"
    deep_sides = [side for side, vals in groups.items() if len(vals) == 2]
    assert len(deep_sides) == 2 and OPPOSITE[deep_sides[0]] == deep_sides[1], "doubled sides must be opposite"

    for side in SIDES:
        group = groups[side]
        if len(group) == 1:
            p = group[0]
            assert p["connector_sides"] == {OPPOSITE[side]}, "single-depth side must use one inward connector"
            p["role"] = "single"
        else:
            group.sort(key=lambda p: (-p["dist2"], p["bbox"]))
            outer, inner = group
            assert outer["connector_sides"] == {OPPOSITE[side]}, "farther border piece must be the outer leaf"
            assert inner["connector_sides"] == {side, OPPOSITE[side]}, "nearer border piece must be the inner segment"
            outer["role"] = "outer"
            inner["role"] = "inner"

    canvas = np.full((h, w), bg, dtype=int)
    ar0, ac0, _, _ = anchor["bbox"]
    placed = _try_place(canvas, anchor["arr"], ar0, ac0, bg)
    assert placed is not None, "anchor must fit at its input position"

    anchor_ports = {side: _global_side_reds(anchor["arr"], ar0, ac0, side) for side in SIDES}
    for side in SIDES:
        assert len(anchor_ports[side]) == 2, "anchor must expose one connector pair on every side"

    red_counts: dict[tuple[int, int], int] = {}
    for rr, cc in anchor["reds"]:
        red_counts[(ar0 + rr, ac0 + cc)] = red_counts.get((ar0 + rr, ac0 + cc), 0) + 1

    def add_reds(piece: dict, top: int, left: int) -> None:
        for rr, cc in piece["reds"]:
            key = (top + rr, left + cc)
            red_counts[key] = red_counts.get(key, 0) + 1

    for side in SIDES:
        group = groups[side]
        inward_side = OPPOSITE[side]
        if len(group) == 1:
            piece = group[0]
            attached = _attach_by_side(placed, piece, anchor_ports[side], inward_side, bg)
            assert attached is not None, "single arm piece must attach to anchor"
            placed, (top, left) = attached
            add_reds(piece, top, left)
            continue

        group.sort(key=lambda p: p["role"])
        inner = next(p for p in group if p["role"] == "inner")
        outer = next(p for p in group if p["role"] == "outer")

        attached_inner = _attach_by_side(placed, inner, anchor_ports[side], inward_side, bg)
        assert attached_inner is not None, "inner segment must attach to anchor"
        placed, (inner_top, inner_left) = attached_inner
        add_reds(inner, inner_top, inner_left)

        outer_target = _global_side_reds(inner["arr"], inner_top, inner_left, side)
        attached_outer = _attach_by_side(placed, outer, outer_target, inward_side, bg)
        assert attached_outer is not None, "outer segment must attach to inner segment"
        placed, (outer_top, outer_left) = attached_outer
        add_reds(outer, outer_top, outer_left)

    assert len(components_mask(placed != 0, connectivity=4)) == 1, "output must be connected"
    assert red_counts and all(v == 2 for v in red_counts.values()), "every red pixel must be paired exactly once"
    return placed


# --------------- generator ---------------

GRID_H_RANGE = (24, 26)
GRID_W_RANGE = (24, 26)
MAX_ASSEMBLY_DIM = 22

PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]
PIECE_H_RANGE = (4, 6)
PIECE_W_RANGE = (4, 7)
BAR_SKIP_PROB = 0.30

ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]
GEN_RETRY_LIMIT = 500
SCATTER_PADDING = 2


def _new_node(kind, h, w, color, top=0, left=0):
    return {
        "kind": kind,
        "shape": (int(h), int(w)),
        "color": int(color),
        "top": int(top),
        "left": int(left),
        "sides": set(),
        "ports": {},
    }


def _ensure_port(node, side):
    if side in node["ports"]:
        return
    h, w = node["shape"]
    span = h if side in ("left", "right") else w
    node["ports"][side] = max(1, min(span - 3, span // 2 - 1))


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
    return [(node["top"] + r, node["left"] + c) for r, c in _port_cells(node, side)]


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
        for r, c in _port_cells(node, side):
            arr[r, c] = 2
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
    if int(np.sum(arr == 2)) != 2 * len(used):
        return None
    return arr


def _make_anchor(node):
    bh, bw = node["block"]
    h, w = node["shape"]
    arr = np.zeros((h, w), dtype=int)
    arr[1 : 1 + bh, 1 : 1 + bw] = 4
    for side in node["sides"]:
        for r, c in _port_cells(node, side):
            arr[r, c] = 2
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


def _place_child(nodes, parent_i, side, child):
    parent = nodes[parent_i]
    incoming = OPPOSITE[side]
    _ensure_port(parent, side)
    _ensure_port(child, incoming)
    pr, pc = _global_port(parent, side)[0]
    cr, cc = _port_cells(child, incoming)[0]
    child["top"] = pr - cr
    child["left"] = pc - cc
    if not _candidate_fits(nodes, parent_i, child):
        return False
    return True


def _random_piece_shape(orientation: str) -> tuple[int, int]:
    if orientation == "vertical":
        return random.randint(4, 5), random.randint(4, 5)
    if orientation == "horizontal":
        return random.randint(4, 5), random.randint(4, 6)
    return random.randint(*PIECE_H_RANGE), random.randint(*PIECE_W_RANGE)


def _build_structure():
    deep_axis = random.choice(["vertical", "horizontal"])
    deep_sides = ("top", "bottom") if deep_axis == "vertical" else ("left", "right")
    order = [*deep_sides, *[s for s in SIDES if s not in deep_sides]]

    bh, bw = random.choice(ANCHOR_BLOCK_CHOICES)
    anchor = _new_node("anchor", bh + 2, bw + 2, 4)
    anchor["block"] = (bh, bw)
    anchor["sides"] = set(SIDES)

    pieces_needed = 6
    colors = random.sample(PIECE_COLORS, pieces_needed)
    color_i = 0
    nodes = [anchor]

    for side in order:
        if side in deep_sides:
            inner_h, inner_w = _random_piece_shape(deep_axis)
            inner = _new_node("piece", inner_h, inner_w, colors[color_i])
            color_i += 1
            inner["sides"] = {side, OPPOSITE[side]}
            inner["home_side"] = side
            inner["depth_role"] = "inner"
            if not _place_child(nodes, 0, side, inner):
                return None
            nodes.append(inner)

            outer_h, outer_w = _random_piece_shape(deep_axis)
            outer = _new_node("piece", outer_h, outer_w, colors[color_i])
            color_i += 1
            outer["sides"] = {OPPOSITE[side]}
            outer["home_side"] = side
            outer["depth_role"] = "outer"
            if not _place_child(nodes, len(nodes) - 1, side, outer):
                return None
            nodes.append(outer)
        else:
            orient = "horizontal" if side in ("left", "right") else "vertical"
            leaf_h, leaf_w = _random_piece_shape(orient)
            leaf = _new_node("piece", leaf_h, leaf_w, colors[color_i])
            color_i += 1
            leaf["sides"] = {OPPOSITE[side]}
            leaf["home_side"] = side
            leaf["depth_role"] = "single"
            if not _place_child(nodes, 0, side, leaf):
                return None
            nodes.append(leaf)

    return nodes, deep_sides


def _place_node_on_canvas(out, node, sr, sc):
    arr = node["arr"]
    top, left = node["top"] + sr, node["left"] + sc
    node["pos"] = (top, left)
    red_counts = {}
    rows, cols = np.nonzero(arr != 0)
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

    anchor = nodes[0]
    desired_top = h // 2 - anchor["shape"][0] // 2
    desired_left = w // 2 - anchor["shape"][1] // 2
    sr_min, sr_max = -r0, h - 1 - r1
    sc_min, sc_max = -c0, w - 1 - c1
    sr = min(max(desired_top, sr_min), sr_max)
    sc = min(max(desired_left, sc_min), sc_max)

    out = np.zeros((h, w), dtype=int)
    red_counts = {}
    for node in nodes:
        rc = _place_node_on_canvas(out, node, sr, sc)
        if rc is None:
            return None
        for k, v in rc.items():
            red_counts[k] = red_counts.get(k, 0) + v

    total_ports = sum(int(np.sum(n["arr"] == 2)) for n in nodes)
    if len(red_counts) * 2 != total_ports:
        return None
    if any(v != 2 for v in red_counts.values()):
        return None
    if len(components_mask(out != 0, connectivity=4)) != 1:
        return None
    return out


def _center_at_position(shape: tuple[int, int], top: int, left: int) -> tuple[float, float]:
    return (top + (shape[0] - 1) / 2.0, left + (shape[1] - 1) / 2.0)


def _edge_candidates(side: str, shape: tuple[int, int], h: int, w: int, anchor_center: tuple[float, float]):
    ph, pw = shape
    out = []
    if side in ("top", "bottom"):
        top = 0 if side == "top" else h - ph
        for left in range(1, w - pw):
            center = _center_at_position(shape, top, left)
            dist2 = (center[0] - anchor_center[0]) ** 2 + (center[1] - anchor_center[1]) ** 2
            out.append((dist2, top, left))
    else:
        left = 0 if side == "left" else w - pw
        for top in range(1, h - ph):
            center = _center_at_position(shape, top, left)
            dist2 = (center[0] - anchor_center[0]) ** 2 + (center[1] - anchor_center[1]) ** 2
            out.append((dist2, top, left))
    out.sort(key=lambda x: (-x[0], x[1], x[2]))
    return out


def _fits_scatter(inp: np.ndarray, arr: np.ndarray, top: int, left: int, padding: int) -> bool:
    h, w = inp.shape
    ph, pw = arr.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    if np.any((inp[top : top + ph, left : left + pw] != 0) & (arr != 0)):
        return False
    r0 = max(0, top - padding)
    c0 = max(0, left - padding)
    r1 = min(h, top + ph + padding)
    c1 = min(w, left + pw + padding)
    region = inp[r0:r1, c0:c1]
    if np.any(region != 0):
        return False
    return True


def _scatter_to_edges(nodes, h, w):
    inp = np.zeros((h, w), dtype=int)
    anchor = nodes[0]
    paste(inp, anchor["arr"], *anchor["pos"], transparent=0)
    anchor_center = _center_at_position(anchor["shape"], *anchor["pos"])

    by_side = {side: [] for side in SIDES}
    for node in nodes[1:]:
        by_side[node["home_side"]].append(node)

    for side, group in by_side.items():
        if len(group) == 2:
            group.sort(key=lambda n: {"outer": 0, "inner": 1}[n["depth_role"]])
        else:
            group.sort(key=lambda n: n["depth_role"])

        for idx, node in enumerate(group):
            cands = _edge_candidates(side, node["arr"].shape, h, w, anchor_center)
            if len(group) == 1:
                cands = sorted(cands, key=lambda x: abs(x[0] - np.median([d for d, _, _ in cands])))
            elif idx == 0:
                pass
            else:
                cands = list(reversed(cands))

            placed = False
            for _dist2, top, left in cands:
                if not _fits_scatter(inp, node["arr"], top, left, SCATTER_PADDING):
                    continue
                center = _center_at_position(node["arr"].shape, top, left)
                side_ok = (
                    center[1] < anchor_center[1] if side == "left" else
                    center[1] > anchor_center[1] if side == "right" else
                    center[0] < anchor_center[0] if side == "top" else
                    center[0] > anchor_center[0]
                )
                if not side_ok:
                    continue
                paste(inp, node["arr"], top, left, transparent=0)
                node["input_pos"] = (top, left)
                placed = True
                break
            if not placed:
                return None
    return inp


def _gen_attempt():
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    built = _build_structure()
    if built is None:
        return None
    nodes, _deep_sides = built
    out = _assemble(nodes, h, w)
    if out is None:
        return None
    inp = _scatter_to_edges(nodes, h, w)
    if inp is None:
        return None
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid combo cbebaa4b input")
