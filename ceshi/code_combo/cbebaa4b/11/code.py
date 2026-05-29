# Combo variant of cbebaa4b (index 11)
# combo_spec: D4/2+D12/1
# Name: 渐变上色 + 颜色编码层级

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis for the fused variant
# a) Concrete behavior:
#    Original: scattered pieces keep their original body colors, then simply translate
#    onto matching red ports and keep those colors in the output.
#    Fused: the input body colors are no longer arbitrary. They must follow one fixed
#    layer palette in attachment order: blue -> green -> orange -> teal -> gray -> pink.
#    After assembly, the same pieces are recolored with a second ordered palette:
#    maroon -> pink -> gray -> blue -> teal -> green. Example with three pieces:
#    input colors [blue, green, orange] imply "attach first/second/third"; output colors
#    [maroon, pink, gray] confirm that same order after translation.
# b) Complexity calibration:
#    A full tree plus free per-piece colors would hide the new rule inside geometry. To
#    keep the color logic visible in 2-4 examples, the generator builds a bent chain:
#    there is exactly one frontier connector pair at each step, and every sample reuses
#    the same ordered input/output palettes. Geometry still varies, but the color cue is
#    stable and repeated across samples.
# c) New assumptions vs. the original solver:
#    - Non-red overlaps are never legal, even if colors match. The original helper allowed
#      same-color overlap; this fused rule cannot rely on that because colors are semantic.
#    - The non-anchor pieces use the first N colors of INPUT_LAYER_COLORS as a prefix.
#    - During assembly there are exactly two unpaired red cells before each non-final step
#      and zero after the last step. The generator guarantees this by constructing a chain.
# d) Placement legality:
#    Generation remains output-first. Nodes are placed in assembled coordinates first,
#    checked for bbox fit, boundary fit, red-pair counts, and single-component output, then
#    scattered with padding so input pieces do not touch.
# e) Visual observability:
#    Humans can see that input colors are ordered cues, because the same prefix palette is
#    reused across examples and each color appears on a whole sprite rather than tiny marks.
#    The output shows a different ordered palette, so the recoloring is obvious and not
#    confusable with the original task.
# f) Visual cleanliness:
#    Only one layer of semantics is added per stage: input uses one solid body color per
#    piece, output uses one replacement body color per piece, while anchor stays yellow and
#    connectors stay red. No extra overlays are introduced.
# g) Inferability of color changes:
#    The mapping is structured by step index, not by arbitrary sprite identity. Because the
#    chain order is color-coded in the input, the recoloring rule is recoverable from a few
#    examples without hidden numeric conventions.


INPUT_LAYER_COLORS = [1, 3, 7, 8, 5, 6]
OUTPUT_GRADIENT_COLORS = [9, 6, 5, 1, 8, 3]


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        reds = np.argwhere(arr == 2)
        colors = sorted(int(v) for v in np.unique(arr) if v not in (bg, 2))
        pieces.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "reds": [(int(r), int(c)) for r, c in reds],
                "shape": (int(arr.shape[0]), int(arr.shape[1])),
                "area": int(np.sum(arr != bg)),
                "body_colors": colors,
            }
        )
    return pieces


def _try_place(canvas: np.ndarray, arr: np.ndarray, top: int, left: int, bg: int) -> np.ndarray | None:
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


def _input_step(body_color: int) -> int:
    assert body_color in INPUT_LAYER_COLORS, "piece color must be part of the input layer palette"
    return INPUT_LAYER_COLORS.index(body_color)


def _piece_body_color(piece: dict) -> int:
    colors = [c for c in piece["body_colors"] if c != 4]
    assert len(colors) == 1, "each non-anchor piece must have exactly one non-red body color"
    return colors[0]


def _evaluate_placement(
    cur: np.ndarray,
    piece: dict,
    top: int,
    left: int,
    rc: dict[tuple[int, int], int],
    bg: int,
) -> tuple[np.ndarray, dict[tuple[int, int], int]] | None:
    nxt = _try_place(cur, piece["arr"], top, left, bg)
    if nxt is None:
        return None

    frontier = {k for k, v in rc.items() if v == 1}
    overlap_count = 0
    next_rc = dict(rc)
    for rr, cc in piece["reds"]:
        gr, gc = top + rr, left + cc
        prev = next_rc.get((gr, gc), 0)
        if prev >= 2:
            return None
        if (gr, gc) in frontier:
            overlap_count += 1
        next_rc[(gr, gc)] = prev + 1

    if overlap_count != 2:
        return None
    if any(v > 2 for v in next_rc.values()):
        return None
    return nxt, next_rc


def _recolor_solution(base: np.ndarray, placements: list[tuple[dict, int, int, int]]) -> np.ndarray:
    out = base.copy()
    for piece, top, left, step in placements:
        ph, pw = piece["shape"]
        sub = out[top : top + ph, left : left + pw]
        mask = (piece["arr"] != 0) & (piece["arr"] != 2)
        sub[mask] = OUTPUT_GRADIENT_COLORS[step]
        out[top : top + ph, left : left + pw] = sub
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the scattered connector sprites around the fixed yellow hub, then recolor the assembled pieces by attachment order.

    Input structure:
    - The grid background is black (0), with one yellow (4) anchor sprite and several scattered non-anchor sprites.
    - Every sprite has red (2) connector pixels on its boundary.
    - The non-anchor body colors are not arbitrary: they use the first N colors of one ordered cue palette, so body color encodes layer/order.

    Rule:
    - Keep the yellow anchor in place and attach the remaining sprites one at a time by overlapping exactly one frontier red pair at each step.
    - The next piece is chosen by the input layer palette order, from earliest cue color to latest cue color.
    - After the full structure is assembled, keep the red connectors and yellow anchor unchanged, but recolor each attached piece body with the matching color from the output gradient palette.

    Key insight:
    - Color has become part of the logic: the input palette tells you which piece must connect next, and the output palette restates that same temporal order after assembly.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must have at least one colored piece"

    anchors = [p for p in pieces if 4 in p["body_colors"]]
    assert len(anchors) == 1, "input must have exactly one yellow anchor piece"
    anchor = anchors[0]
    rest = [p for p in pieces if p is not anchor]
    assert 1 <= len(rest) <= len(INPUT_LAYER_COLORS), "piece count must fit the ordered layer palette"

    for p in rest:
        p["body_color"] = _piece_body_color(p)
        p["step"] = _input_step(p["body_color"])
    ordered = sorted(rest, key=lambda p: p["step"])
    assert [p["body_color"] for p in ordered] == INPUT_LAYER_COLORS[: len(ordered)], (
        "non-anchor colors must be the ordered layer palette prefix"
    )

    canvas = np.full((h, w), bg, dtype=int)
    ar0, ac0, _, _ = anchor["bbox"]
    placed = _try_place(canvas, anchor["arr"], ar0, ac0, bg)
    assert placed is not None, "anchor piece must fit on canvas at its original position"

    total_ports = int(sum(len(p["reds"]) for p in pieces))
    assert total_ports % 2 == 0, "total red connector count must be even"
    target_unique_red = total_ports // 2

    red_counts: dict[tuple[int, int], int] = {}
    for rr, cc in anchor["reds"]:
        red_counts[(ar0 + rr, ac0 + cc)] = red_counts.get((ar0 + rr, ac0 + cc), 0) + 1

    solutions: list[tuple[np.ndarray, list[tuple[dict, int, int, int]]]] = []

    def dfs(
        idx: int,
        cur: np.ndarray,
        rc: dict[tuple[int, int], int],
        placements: list[tuple[dict, int, int, int]],
    ) -> None:
        if len(solutions) >= 2:
            return

        if idx == len(ordered):
            if len(rc) == target_unique_red and all(v == 2 for v in rc.values()):
                solutions.append((cur, list(placements)))
            return

        frontier = [k for k, v in rc.items() if v == 1]
        assert len(frontier) == 2, "chain variant must expose exactly one frontier pair before each placement"

        piece = ordered[idx]
        candidates: dict[tuple[int, int], tuple[np.ndarray, dict[tuple[int, int], int]]] = {}
        for tr, tc in frontier:
            for pr, pc in piece["reds"]:
                top = tr - pr
                left = tc - pc
                key = (top, left)
                if key in candidates:
                    continue
                attempt = _evaluate_placement(cur, piece, top, left, rc, bg)
                if attempt is None:
                    continue
                nxt, next_rc = attempt
                next_frontier_count = sum(v == 1 for v in next_rc.values())
                expected = 0 if idx == len(ordered) - 1 else 2
                if next_frontier_count != expected:
                    continue
                candidates[key] = (nxt, next_rc)

        assert candidates, "the next layer-coded piece must have at least one valid attachment"
        for top, left in sorted(candidates):
            nxt, next_rc = candidates[(top, left)]
            placements.append((piece, top, left, piece["step"]))
            dfs(idx + 1, nxt, next_rc, placements)
            placements.pop()
            if len(solutions) >= 2:
                return

    dfs(0, placed, red_counts, [])
    assert len(solutions) == 1, "layer-coded jigsaw assembly must have exactly one valid solution"
    assembled, placements = solutions[0]
    return _recolor_solution(assembled, placements)


# --------------- generator (fused chain variant) ---------------

# Canvas
GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
MAX_ASSEMBLY_DIM = 24

# Pieces
PIECE_COUNT_RANGE = (4, 6)
PIECE_H_RANGE = (4, 7)
PIECE_W_RANGE = (4, 8)
PIECE_DIM_CANDIDATES = 14
PIECE_CAP = 2
BAR_SKIP_PROB = 0.35

# Anchor
ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]

# Scatter / generation
SCATTER_PADDING = 2
GEN_RETRY_LIMIT = 400

# Side geometry
SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}


def _new_node(kind, h, w, color, step=-1, top=0, left=0, cap=PIECE_CAP):
    return {
        "kind": kind,
        "shape": (int(h), int(w)),
        "color": int(color),
        "step": int(step),
        "top": int(top),
        "left": int(left),
        "cap": int(cap),
        "sides": set(),
        "ports": {},
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


def _grow_chain():
    bh, bw = random.choice(ANCHOR_BLOCK_CHOICES)
    anchor = _new_node("anchor", bh + 2, bw + 2, 4, cap=1)
    anchor["block"] = (bh, bw)
    nodes = [anchor]
    n_pieces = random.randint(*PIECE_COUNT_RANGE)

    for step in range(n_pieces):
        placed = False
        parent_i = len(nodes) - 1
        parent = nodes[parent_i]
        dims = [
            (random.randint(*PIECE_H_RANGE), random.randint(*PIECE_W_RANGE))
            for _ in range(PIECE_DIM_CANDIDATES)
        ]
        random.shuffle(dims)
        for ph, pw in dims:
            sides = [s for s in SIDES if s not in parent["sides"]]
            random.shuffle(sides)
            for side in sides:
                child = _new_node("piece", ph, pw, INPUT_LAYER_COLORS[step], step=step)
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
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
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


def _apply_output_gradient(assembled: np.ndarray, nodes: list[dict]) -> np.ndarray:
    out = assembled.copy()
    for node in nodes[1:]:
        top, left = node["pos"]
        ph, pw = node["shape"]
        sub = out[top : top + ph, left : left + pw]
        mask = (node["arr"] != 0) & (node["arr"] != 2)
        sub[mask] = OUTPUT_GRADIENT_COLORS[node["step"]]
        out[top : top + ph, left : left + pw] = sub
    return out


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

    assembled = np.zeros((h, w), dtype=int)
    all_red_counts = {}
    for node in nodes:
        rc = _place_node_on_canvas(assembled, node, sr, sc)
        if rc is None:
            return None
        for k, v in rc.items():
            all_red_counts[k] = all_red_counts.get(k, 0) + v

    total_ports = sum(int(np.sum(n["arr"] == 2)) for n in nodes)
    if len(all_red_counts) * 2 != total_ports:
        return None
    if any(v != 2 for v in all_red_counts.values()):
        return None
    if len(components_mask(assembled != 0, connectivity=4)) != 1:
        return None
    recolored = _apply_output_gradient(assembled, nodes)
    if np.array_equal(assembled, recolored):
        return None
    return assembled, recolored


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
    nodes = _grow_chain()
    if nodes is None:
        return None
    assembled_pair = _assemble(nodes, h, w)
    if assembled_pair is None:
        return None
    _assembled_input, out = assembled_pair
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
    raise RuntimeError("failed to generate a valid fused cbebaa4b input")
