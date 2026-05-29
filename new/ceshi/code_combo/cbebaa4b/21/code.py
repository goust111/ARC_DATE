# Combo variant of cbebaa4b (index 21)
# combo_spec: D7/1+D12/1
# Name: 按颜色分组批量连接 + 颜色编码层级

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior trace:
#    - Small example: the anchor exposes four red cells (two connector pairs), the input also
#      contains two blue sprites and two green sprites. In the original task any remaining
#      sprite could be tried next as long as one red pixel overlapped an unpaired red.
#      In the fused task the two blue sprites must be attached first as one color batch, and
#      only after the whole blue layer is in place may the two green sprites attach.
#      The same visible color now serves two roles at once: grouping ("all blue together")
#      and layer order ("blue before green").
# b) Complexity calibration:
#    - The layer palette is intentionally small and fixed: blue(1) -> green(3) -> orange(7).
#      Blue and green always repeat, so the batch rule is visually supported by multiple
#      pieces instead of a single arbitrary recoloring. Orange is optional and capped at one
#      extra layer to keep examples legible.
# c) Assumption audit against the original solver:
#    - Original DFS assumption: any remaining piece may attach to any current unpaired red.
#      Not valid here; only the current batch color may consume the current unpaired set.
#    - Original red-count assumption: newly exposed reds may come from any just-placed piece.
#      Tightened here: after finishing batch k, the number of unpaired reds must equal
#      exactly 2 * (#pieces in batch k+1), because every next-layer piece consumes one
#      two-cell connector pair from the previous layer.
#    - Original overlap helper allowed same-color overlap. That becomes dangerous once many
#      same-color pieces exist, so this variant forbids every non-red overlap.
# d) Placement legality:
#    - The generator builds a rooted layered tree. Each non-anchor piece has exactly one
#      parent in the previous layer and at most one child in the next layer. This guarantees
#      same-color pieces never need to connect to one another, which makes the "simultaneous
#      batch" interpretation physically realizable.
#    - During assembly, candidate placements are rejected on out-of-bounds, non-red overlap,
#      reusing already-paired reds, or trying to consume reds created earlier in the same batch.
#      The scattered input keeps Chebyshev gap >= 2 between components.
# e) Visual observability / neatness / inferability:
#    - Humans only need to infer three fixed visible colors and a repeated depth pattern:
#      blue pieces touch the yellow anchor, green pieces touch blue pieces, orange pieces
#      touch green pieces. The output therefore reads as clean color bands rather than a
#      noisy free-for-all red matching puzzle.


BG = 0
RED = 2
YELLOW = 4
LAYER_COLORS = (1, 3, 7)  # blue -> green -> orange
SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}


def _main_color(arr: np.ndarray) -> int:
    cols = sorted({int(v) for v in arr.flatten()} - {BG, RED})
    assert cols, "piece must contain at least one non-background non-red color"
    if YELLOW in cols:
        assert cols == [YELLOW], "anchor should only use yellow besides red"
        return YELLOW
    assert len(cols) == 1, "non-anchor piece must have exactly one body color"
    return cols[0]


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        reds = np.argwhere(arr == RED)
        pieces.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "reds": [(int(r), int(c)) for r, c in reds],
                "shape": (int(arr.shape[0]), int(arr.shape[1])),
                "area": int(np.sum(arr != bg)),
                "color": _main_color(arr),
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

    # Only red connector pixels may overlap.
    conflict = mask & (sub != bg) & ~((arr == RED) & (sub == RED))
    if np.any(conflict):
        return None

    out = canvas.copy()
    out_sub = out[top : top + ph, left : left + pw]
    out_sub[mask] = arr[mask]
    out[top : top + ph, left : left + pw] = out_sub
    return out


def _add_piece_reds(
    rc: dict[tuple[int, int], int], piece: dict, top: int, left: int
) -> dict[tuple[int, int], int]:
    nxt = dict(rc)
    for rr, cc in piece["reds"]:
        key = (top + rr, left + cc)
        nxt[key] = nxt.get(key, 0) + 1
    return nxt


def _unpaired_reds(rc: dict[tuple[int, int], int]) -> set[tuple[int, int]]:
    return {k for k, v in rc.items() if v == 1}


def _candidate_moves(
    cur: np.ndarray,
    rc: dict[tuple[int, int], int],
    frozen: set[tuple[int, int]],
    piece: dict,
    bg: int,
) -> list[tuple[int, int, np.ndarray]]:
    h, w = cur.shape
    ph, pw = piece["shape"]
    out: dict[tuple[int, int], np.ndarray] = {}
    for tr, tc in frozen:
        for pr, pc in piece["reds"]:
            top = tr - pr
            left = tc - pc
            if top < 0 or left < 0 or top + ph > h or left + pw > w:
                continue

            overlap_frozen = 0
            bad = False
            for rr, cc in piece["reds"]:
                gr, gc = top + rr, left + cc
                prev = rc.get((gr, gc), 0)
                if prev >= 2:
                    bad = True
                    break
                if prev == 1:
                    if (gr, gc) in frozen:
                        overlap_frozen += 1
                    else:
                        bad = True
                        break
            if bad or overlap_frozen != 2:
                continue

            nxt = _try_place(cur, piece["arr"], top, left, bg)
            if nxt is not None:
                out[(top, left)] = nxt

    return [(top, left, out[(top, left)]) for top, left in sorted(out)]


def _solve_batch(
    cur: np.ndarray,
    rc: dict[tuple[int, int], int],
    batch: list[dict],
    next_batch_size: int,
    bg: int,
) -> tuple[np.ndarray, dict[tuple[int, int], int]]:
    frozen = _unpaired_reds(rc)
    assert len(frozen) == 2 * len(batch), "each batch piece must consume one prior connector pair"

    solutions: dict[bytes, tuple[np.ndarray, dict[tuple[int, int], int]]] = {}

    def dfs(cur2: np.ndarray, rc2: dict[tuple[int, int], int], remaining: list[dict]) -> None:
        if len(solutions) >= 2:
            return
        if not remaining:
            unpaired = _unpaired_reds(rc2)
            if any(coord in unpaired for coord in frozen):
                return
            if len(unpaired) != 2 * next_batch_size:
                return
            solutions[cur2.tobytes()] = (cur2, rc2)
            return

        move_sets = []
        for idx, piece in enumerate(remaining):
            moves = _candidate_moves(cur2, rc2, frozen, piece, bg)
            if not moves:
                return
            move_sets.append((len(moves), -piece["area"], idx, moves))
        move_sets.sort()
        _, _, idx, moves = move_sets[0]
        piece = remaining[idx]
        next_remaining = remaining[:idx] + remaining[idx + 1 :]

        for top, left, nxt in moves:
            next_rc = _add_piece_reds(rc2, piece, top, left)
            if any(v > 2 for v in next_rc.values()):
                continue
            dfs(nxt, next_rc, next_remaining)
            if len(solutions) >= 2:
                return

    dfs(cur, rc, batch)
    assert len(solutions) == 1, "each color batch must have exactly one assembled result"
    return next(iter(solutions.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered connector sprites around a fixed yellow anchor, but do it in color-coded layers.

    Input structure:
    - Background is black (0). One component is the yellow (4) anchor; all other components are colored sprites.
    - Every sprite carries red (2) connector pixels on its boundary.
    - Non-anchor sprite colors are restricted to a visible layer code: blue (1), green (3), and sometimes orange (7).

    Rule:
    - Keep the yellow anchor in place.
    - Attach all blue sprites first, all green sprites second, and all orange sprites last.
    - Within one color, each sprite uses exactly one previously exposed red connector pair, and same-color sprites never connect to one another in the same batch.

    Key insight:
    - Color is not decorative here. It simultaneously tells you which sprites form one assembly batch and which layer of the growing structure they belong to, so the exposed red pairs after one batch are reserved for the next color only.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = BG

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must contain at least one piece"

    anchors = [p for p in pieces if p["color"] == YELLOW]
    assert len(anchors) == 1, "input must contain exactly one yellow anchor"
    anchor = anchors[0]
    rest = [p for p in pieces if p is not anchor]

    color_groups = {color: [p for p in rest if p["color"] == color] for color in LAYER_COLORS}
    present = [color for color in LAYER_COLORS if color_groups[color]]
    assert present[:2] == [1, 3], "blue and green layers must both be present"
    assert present == list(LAYER_COLORS[: len(present)]), "layer colors must form a blue->green->orange prefix"
    assert len(color_groups[1]) >= 2 and len(color_groups[3]) >= 2, "blue and green must repeat"

    canvas = np.full((h, w), bg, dtype=int)
    ar0, ac0, _, _ = anchor["bbox"]
    placed = _try_place(canvas, anchor["arr"], ar0, ac0, bg)
    assert placed is not None, "anchor placement must be valid"

    red_counts: dict[tuple[int, int], int] = {}
    for rr, cc in anchor["reds"]:
        red_counts[(ar0 + rr, ac0 + cc)] = red_counts.get((ar0 + rr, ac0 + cc), 0) + 1

    cur = placed
    rc = red_counts
    for i, color in enumerate(present):
        batch = color_groups[color]
        next_batch_size = len(color_groups[present[i + 1]]) if i + 1 < len(present) else 0
        cur, rc = _solve_batch(cur, rc, batch, next_batch_size, bg)

    assert not _unpaired_reds(rc), "final assembly must pair every red connector"
    return cur


# --------------- generator (combo) ---------------

# Canvas
GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
MAX_ASSEMBLY_DIM = 24

# Piece geometry
PIECE_H_RANGE = (4, 7)
PIECE_W_RANGE = (4, 8)
PIECE_DIM_CANDIDATES = 12
BAR_SKIP_PROB = 0.35

# Anchor
ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]

# Scatter
SCATTER_PADDING = 2

# Generation
GEN_RETRY_LIMIT = 400
LAYER_PATTERNS = [
    (2, 2, 0),
    (3, 2, 0),
    (2, 2, 1),
    (3, 2, 1),
]


def _new_node(kind, h, w, color, layer, top=0, left=0, cap=2):
    return {
        "kind": kind,
        "shape": (int(h), int(w)),
        "color": int(color),
        "layer": int(layer),
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
            if arr[rr, cc] != RED:
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
            arr[r, c] = RED
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
    if int(np.sum(arr == RED)) != 2 * len(used):
        return None
    return arr


def _make_anchor(node):
    bh, bw = node["block"]
    h, w = node["shape"]
    arr = np.zeros((h, w), dtype=int)
    arr[1 : 1 + bh, 1 : 1 + bw] = YELLOW
    for side in node["sides"]:
        for r, c in _port_cells(node, side):
            arr[r, c] = RED
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


def _attach_child(nodes, parent_i, color, layer):
    dims = [
        (random.randint(*PIECE_H_RANGE), random.randint(*PIECE_W_RANGE))
        for _ in range(PIECE_DIM_CANDIDATES)
    ]
    random.shuffle(dims)
    parent = nodes[parent_i]
    sides = [s for s in SIDES if s not in parent["sides"]]
    random.shuffle(sides)
    for ph, pw in dims:
        for side in sides:
            child = _new_node("piece", ph, pw, color, layer)
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
            return len(nodes) - 1
    return None


def _grow_layered_tree():
    blue_count, green_count, orange_count = random.choice(LAYER_PATTERNS)
    bh, bw = random.choice(ANCHOR_BLOCK_CHOICES)
    anchor = _new_node("anchor", bh + 2, bw + 2, YELLOW, layer=0, cap=blue_count)
    anchor["block"] = (bh, bw)
    nodes = [anchor]

    blue_indices = []
    for _ in range(blue_count):
        idx = _attach_child(nodes, 0, 1, 1)
        if idx is None:
            return None
        blue_indices.append(idx)

    green_parent_indices = random.sample(blue_indices, green_count)
    green_indices = []
    for parent_i in green_parent_indices:
        idx = _attach_child(nodes, parent_i, 3, 2)
        if idx is None:
            return None
        green_indices.append(idx)

    orange_parent_indices = random.sample(green_indices, orange_count)
    orange_indices = []
    for parent_i in orange_parent_indices:
        idx = _attach_child(nodes, parent_i, 7, 3)
        if idx is None:
            return None
        orange_indices.append(idx)

    meta = {
        "layer_counts": (blue_count, green_count, orange_count),
        "anchor_degree": blue_count,
        "blue_nonleaf": green_count,
        "green_nonleaf": orange_count,
        "orange_count": orange_count,
    }
    return nodes, meta


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
    conflicts = (cur_vals != 0) & ~((cur_vals == RED) & (vals == RED))
    if np.any(conflicts):
        return None
    red_mask = vals == RED
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
        for k, v in rc.items():
            all_red_counts[k] = all_red_counts.get(k, 0) + v

    total_ports = sum(int(np.sum(n["arr"] == RED)) for n in nodes)
    if len(all_red_counts) * 2 != total_ports:
        return None
    if any(v != 2 for v in all_red_counts.values()):
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


def _gen_attempt(with_meta=False):
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grown = _grow_layered_tree()
    if grown is None:
        return None
    nodes, meta = grown

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

    if with_meta:
        return inp, meta
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fused cbebaa4b input")
