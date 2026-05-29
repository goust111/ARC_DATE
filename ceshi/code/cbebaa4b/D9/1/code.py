# Variant of cbebaa4b
# Dimension: D9 — 锚点方块的属性 (meta)
# Change: 中央黄色方块为实心正方形，使用独特颜色（黄色），与所有精灵颜色不同
#   → 锚点颜色与某个精灵相同

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis for the D9 variant
# ----------------------------------
# a) concretization_and_trace:
#    - The anchor remains a solid inner rectangle with red connector pairs on the outer rim.
#      The only D9 change is that the anchor body color is no longer unique: it is chosen from
#      the normal sprite palette and exactly one ordinary sprite reuses that same color.
#    - Input/output behavior is otherwise unchanged: the anchor stays fixed and the scattered
#      sprites translate to pair red ports until one connected assembly is formed.
#    - Critical flow difference:
#      original: `main()` found the anchor via `np.any(arr == 4)`, and `_try_place()` also
#      tolerated same-color overlaps because distinct non-red colors made that harmless.
#      variant: color can no longer identify the anchor, and allowing same-color overlap would
#      incorrectly let the shared-color sprite cover anchor cells. Therefore the solver must
#      detect the anchor by structure and placement must allow only red-on-red overlap.
# b) complexity_calibration:
#    - The change is visually obvious because the old unique yellow cue disappears while one
#      normal sprite now shares the anchor's color.
#    - To avoid overcomplicating the task, only one sprite shares the anchor color; the rest
#      remain distinct. This keeps the color ambiguity real without creating many equivalent
#      same-color candidates.
# c) implicit_assumptions_review:
#    - `_try_place`: original `arr == sub` exception depended on non-red colors being unique.
#      In this variant that assumption is false, so it is removed.
#    - `anchors = [p for p in pieces if np.any(p["arr"] == 4)]`: depended on unique yellow.
#      Replaced with `_is_anchor_piece`, which recognizes the solid rectangular body plus rim ports.
#    - Anchor tie-breaking now uses centrality only as a fallback; generator guarantees exactly
#      one structural anchor candidate, so solver and generator stay aligned.
# d) placement_legality:
#    - Generator still builds the assembled output first, verifies pairwise red overlaps only,
#      and scatters the non-anchor pieces afterward with Chebyshev padding.
#    - Anchor placement is biased near the grid center so the loss of unique color remains easy
#      to perceive, but it is still clamped to legal shifts that keep the whole assembly in-bounds.
# e) visual_observability:
#    - Humans can now identify the anchor from its solid rectangle shape and near-central position,
#      not from a special color. The shared-color sprite makes the D9 change visible in 2-3 samples.
# f) visual_cleanliness:
#    - No new overlays are introduced beyond the original red-port overlaps in the final assembly.
#      Same-color non-red overlaps are explicitly forbidden.
# g) color_inference:
#    - The color rule is simple and repeated: the anchor color is reused by one sprite. No hidden
#      ordering or color-id arithmetic is involved.

ANCHOR_BODY_CHOICES = ((3, 3), (4, 5))


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


def _anchor_center_distance_sq(piece: dict, h: int, w: int) -> int:
    r0, c0, r1, c1 = piece["bbox"]
    pr2 = r0 + r1
    pc2 = c0 + c1
    return int((pr2 - (h - 1)) ** 2 + (pc2 - (w - 1)) ** 2)


def _is_anchor_piece(arr: np.ndarray, bg: int) -> bool:
    body_mask = (arr != bg) & (arr != 2)
    if not np.any(body_mask):
        return False

    body_colors = np.unique(arr[body_mask])
    if len(body_colors) != 1:
        return False
    body_color = int(body_colors[0])

    br0, bc0, br1, bc1 = bbox_coords(np.argwhere(body_mask))
    bh, bw = br1 - br0 + 1, bc1 - bc0 + 1
    if (bh, bw) not in ANCHOR_BODY_CHOICES:
        return False
    if np.any(arr[br0 : br1 + 1, bc0 : bc1 + 1] != body_color):
        return False

    extras = (arr != bg) & ~body_mask
    if not np.any(extras):
        return False

    for r, c in np.argwhere(extras):
        if arr[r, c] != 2:
            return False
        if r < br0 and br0 - r == 1 and bc0 <= c <= bc1:
            continue
        if r > br1 and r - br1 == 1 and bc0 <= c <= bc1:
            continue
        if c < bc0 and bc0 - c == 1 and br0 <= r <= br1:
            continue
        if c > bc1 and c - bc1 == 1 and br0 <= r <= br1:
            continue
        return False
    return True


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered sprites around a fixed solid anchor whose color is no longer unique.

    Input structure:
    - bg = black (0). Several disconnected sprites are scattered on the grid.
    - One sprite is the anchor: a solid inner rectangle with red (2) connector pairs on its rim.
    - At least one ordinary sprite shares the anchor's non-red body color, so color alone does not identify the anchor.

    Rule:
    - The anchor remains at its input position while the other sprites translate into place.
    - A move is valid only when red connector pixels overlap existing unpaired red connector pixels, with no non-red overlap.
    - The final output is the unique complete assembly in which every red connector is paired exactly once.

    Key insight:
    - The anchor must be recognized by structure rather than by a special color: it is the solid rectangular body near the center, while the remaining sprites are routed bars/brackets.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must have at least one colored piece"

    anchor_candidates = [p for p in pieces if _is_anchor_piece(p["arr"], bg)]
    assert anchor_candidates, "input must contain a solid rectangular anchor candidate"
    anchor = min(anchor_candidates, key=lambda p: (_anchor_center_distance_sq(p, h, w), p["bbox"]))
    rest = [p for p in pieces if p is not anchor]

    canvas = np.full((h, w), bg, dtype=int)
    ar0, ac0, _, _ = anchor["bbox"]
    placed = _try_place(canvas, anchor["arr"], ar0, ac0, bg)
    assert placed is not None, "anchor piece must fit on canvas at its original position"

    total_ports = int(sum(len(p["reds"]) for p in pieces))
    assert total_ports % 2 == 0, "total red connector ports must be even for complete pairing"
    target_unique_red = total_ports // 2

    red_counts: dict[tuple[int, int], int] = {}
    for rr, cc in anchor["reds"]:
        red_counts[(ar0 + rr, ac0 + cc)] = red_counts.get((ar0 + rr, ac0 + cc), 0) + 1

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
        for t in unpaired:
            moves = []
            tr, tc = t
            for idx, p in enumerate(remaining):
                ph, pw = p["shape"]
                for pr, pc in p["reds"]:
                    top = tr - pr
                    left = tc - pc
                    if top < 0 or left < 0 or top + ph > h or left + pw > w:
                        continue
                    bad = False
                    overlap = 0
                    for rr, cc in p["reds"]:
                        gr, gc = top + rr, left + cc
                        prev = rc.get((gr, gc), 0)
                        if prev >= 2:
                            bad = True
                            break
                        if (gr, gc) in rc1:
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

            next_rc = dict(rc)
            for rr, cc in p["reds"]:
                gr, gc = top + rr, left + cc
                next_rc[(gr, gc)] = next_rc.get((gr, gc), 0) + 1

            nxt = _try_place(cur, p["arr"], top, left, bg)
            if nxt is None:
                continue
            dfs(nxt, next_remaining, next_rc)
            if len(solutions) >= 2:
                return

    dfs(placed, rest, red_counts)
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
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]
PIECE_CAP = 2
BAR_SKIP_PROB = 0.35

# Anchor
ANCHOR_BLOCK_CHOICES = list(ANCHOR_BODY_CHOICES)
ANCHOR_CAP_CHOICES = [1, 2]
ANCHOR_CENTER_JITTER = 1

# Scatter
SCATTER_PADDING = 2

# Generation
GEN_RETRY_LIMIT = 300

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
    if _is_anchor_piece(arr, 0):
        return None
    return arr


def _make_anchor(node):
    bh, bw = node["block"]
    arr = np.zeros(node["shape"], dtype=int)
    arr[1 : 1 + bh, 1 : 1 + bw] = node["color"]
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


def _choose_colors(n_pieces: int) -> tuple[int, list[int]]:
    anchor_color = random.choice(PIECE_COLORS)
    other_colors = [c for c in PIECE_COLORS if c != anchor_color]
    random.shuffle(other_colors)
    piece_colors = other_colors[:n_pieces]
    shared_idx = random.randrange(n_pieces)
    piece_colors[shared_idx] = anchor_color
    return anchor_color, piece_colors


def _grow_tree():
    bh, bw = random.choice(ANCHOR_BLOCK_CHOICES)
    n_pieces = random.randint(*PIECE_COUNT_RANGE)
    anchor_color, colors = _choose_colors(n_pieces)
    anchor = _new_node("anchor", bh + 2, bw + 2, anchor_color, cap=random.choice(ANCHOR_CAP_CHOICES))
    anchor["block"] = (bh, bw)
    nodes = [anchor]

    for i in range(n_pieces):
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
                    child = _new_node("piece", ph, pw, colors[i])
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


def _choose_shift(nodes, h, w):
    r0, c0, r1, c1 = _union_bbox(nodes)
    sr_lo, sr_hi = -r0, h - 1 - r1
    sc_lo, sc_hi = -c0, w - 1 - c1
    anchor = nodes[0]
    ah, aw = anchor["shape"]
    sr_target = (h - ah) // 2 - anchor["top"]
    sc_target = (w - aw) // 2 - anchor["left"]

    sr_lo2 = max(sr_lo, sr_target - ANCHOR_CENTER_JITTER)
    sr_hi2 = min(sr_hi, sr_target + ANCHOR_CENTER_JITTER)
    sc_lo2 = max(sc_lo, sc_target - ANCHOR_CENTER_JITTER)
    sc_hi2 = min(sc_hi, sc_target + ANCHOR_CENTER_JITTER)

    sr = random.randint(sr_lo2, sr_hi2) if sr_lo2 <= sr_hi2 else min(max(sr_target, sr_lo), sr_hi)
    sc = random.randint(sc_lo2, sc_hi2) if sc_lo2 <= sc_hi2 else min(max(sc_target, sc_lo), sc_hi)
    return sr, sc


def _assemble(nodes, h, w):
    for node in nodes:
        node["arr"] = _make_anchor(node) if node["kind"] == "anchor" else _make_piece(node)
        if node["arr"] is None:
            return None

    r0, c0, r1, c1 = _union_bbox(nodes)
    if (r1 - r0 + 1) > h or (c1 - c0 + 1) > w:
        return None
    sr, sc = _choose_shift(nodes, h, w)

    out = np.zeros((h, w), dtype=int)
    all_red_counts = {}
    for node in nodes:
        rc = _place_node_on_canvas(out, node, sr, sc)
        if rc is None:
            return None
        for key, val in rc.items():
            all_red_counts[key] = all_red_counts.get(key, 0) + val

    total_ports = sum(int(np.sum(n["arr"] == 2)) for n in nodes)
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
                inp, node["arr"], transparent=0, padding=SCATTER_PADDING, padding_connectivity=8
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
    raise RuntimeError("failed to generate a valid cbebaa4b D9 input")
