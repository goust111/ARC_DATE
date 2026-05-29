# Combo variant of cbebaa4b (index 41)
# combo_spec: D4/2+D5/1
# Name: 渐变上色 + 移动轨迹线

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste


# Design analysis for the fused D4/2 + D5/1 variant
# a) Concrete behavior:
#    - The input still shows one fixed yellow anchor and 4-6 disconnected colored sprites.
#    - First we solve the original jigsaw assembly exactly as before: non-anchor pieces translate
#      and pair their red connector pixels with the growing structure.
#    - Then two post-assembly effects happen together:
#      1. Each moved piece leaves a one-cell-wide staircase trail from a representative source
#         cell to the same cell in its final assembled placement, and that trail uses the piece's
#         original input color.
#      2. The assembled copies of the moved pieces are recolored by connection step. Example:
#         if a blue source piece is the first to attach next to the anchor and a green source piece
#         is the fourth, the output keeps a blue trail and a green trail, but the assembled piece
#         nearest the anchor takes the darkest ramp color and the fourth piece takes a lighter ramp
#         color. The anchor remains yellow and connector pixels remain red.
#    - The visible difference from the original task is therefore twofold: the center assembly no
#      longer preserves source colors, and the empty space between source and destination records
#      motion history.
# b) Complexity calibration:
#    - A full tree can allow several legal attachment orders that would make "step color" ambiguous.
#      To keep the new D4 signal inferable, the generator builds a chain: anchor -> piece1 -> ...
#      -> pieceN. Then "connection order" equals "distance from the anchor" in every sample.
#    - The gradient uses a fixed 6-color ramp sampled according to piece count, so the first piece
#      is always mapped to the darkest ramp endpoint and the last piece to the lightest endpoint.
#      This keeps the palette small and repeated across samples.
#    - Trails use only one representative cell per piece, not the whole sprite footprint, so the
#      output stays legible even when 4-6 pieces move toward the center.
# c) Hidden assumptions in main():
#    - `_extract_pieces()` still assumes the input contains only disconnected sprites on black.
#      That remains valid because trails are output-only artifacts.
#    - `_try_place()` still only permits red-red overlaps or exact color equality. The search phase
#      operates on original sprite colors, so this assumption is unchanged during assembly.
#    - The original solver only needed the final assembled canvas. The fused variant also needs
#      per-piece final placements and the exact attachment sequence, so DFS now records both.
#    - Recoloring after assembly assumes the attachment order is unique. The generator guarantees
#      this by emitting chain-shaped connection graphs.
#    - Overlaying trails after recoloring assumes trails do not cross assembled occupied cells;
#      otherwise the original-color trail would hide the gradient. The generator explicitly rejects
#      scatter placements whose trails touch any assembled non-background cell.
# d) Placement legality:
#    - The assembled chain must fit fully inside the grid and preserve the original no-conflict
#      rule: only paired red connector pixels may overlap.
#    - Scattered source sprites must stay disjoint with Chebyshev padding >= 2.
#    - Every trail must stay in bounds, avoid the anchor, avoid assembled occupied cells, and avoid
#      other trails. Those checks are constructive, not accidental: the scatter search filters
#      candidates against the already placed source sprites and the reserved trail-owner mask.
# e) Visual observability:
#    - Because the assembly is centered and the scattered sources are chosen from farther positions,
#      the trails usually point inward and are long enough to be seen immediately.
#    - Because the generator enforces source_color != final_gradient_color for every piece, each
#      sample visibly shows that the center colors encode step order rather than preserving input
#      colors.
#    - The chain topology makes the gradient easy to read: darker near the anchor, lighter farther
#      away, repeated over several examples.
# f) Visual cleanliness:
#    - Trails are one cell wide.
#    - Trail routing is a simple staircase line from one stable representative cell, which avoids
#      the dense smearing that full-shape motion would create.
#    - Trail candidate filtering prevents trails from stacking on the assembled center, foreign
#      source sprites, or each other, so the output remains separable into center assembly plus
#      outward history lines.
# g) Change inferability:
#    - The color change is not arbitrary per piece identity. It is tied to a structural feature
#      humans can see in every example: connection rank along the chain from the yellow anchor.
#    - The ramp is fixed across tasks, while original colors vary independently. That repeated
#      regularity gives enough evidence that the new center colors come from assembly order.


GRADIENT_RAMP = [9, 1, 3, 5, 8, 6]


def _piece_color(arr: np.ndarray, bg: int) -> int:
    colors = [int(v) for v in np.unique(arr) if int(v) not in (bg, 2, 4)]
    assert colors, "piece must contain at least one non-red, non-yellow color"
    return colors[0]


def _anchor_center(arr: np.ndarray, top: int, left: int) -> tuple[float, float]:
    ys = np.argwhere(arr == 4)
    center = ys.mean(axis=0)
    return float(top + center[0]), float(left + center[1])


def _exposure_score(arr: np.ndarray, r: int, c: int, bg: int) -> int:
    h, w = arr.shape
    score = 0
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if rr < 0 or cc < 0 or rr >= h or cc >= w or arr[rr, cc] == bg:
            score += 1
    return score


def _trail_local_cell(
    arr: np.ndarray,
    bg: int,
    dst_top: int,
    dst_left: int,
    anchor_center: tuple[float, float],
) -> tuple[int, int]:
    coords = np.argwhere((arr != bg) & (arr != 2) & (arr != 4))
    ar, ac = anchor_center
    best = max(
        (
            (
                abs((dst_top + r) - ar) + abs((dst_left + c) - ac),
                _exposure_score(arr, int(r), int(c), bg),
                int(r),
                int(c),
            )
            for r, c in coords
        )
    )
    return int(best[2]), int(best[3])


def _gradient_sequence(count: int) -> list[int]:
    assert 1 <= count <= len(GRADIENT_RAMP)
    if count == 1:
        return [GRADIENT_RAMP[0]]
    idxs = [int(round(v)) for v in np.linspace(0, len(GRADIENT_RAMP) - 1, count)]
    assert len(set(idxs)) == count
    return [GRADIENT_RAMP[i] for i in idxs]


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for uid, pts in enumerate(comps):
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        reds = np.argwhere(arr == 2)
        pieces.append(
            {
                "uid": int(uid),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "reds": [(int(r), int(c)) for r, c in reds],
                "shape": (int(arr.shape[0]), int(arr.shape[1])),
                "area": int(np.sum(arr != bg)),
                "color": None if np.any(arr == 4) else _piece_color(arr, bg),
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


def _trail_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = map(int, start)
    er, ec = map(int, end)
    cells = [(r, c)]
    row_first = abs(er - r) >= abs(ec - c)
    while (r, c) != (er, ec):
        axes = ("row", "col") if row_first else ("col", "row")
        for axis in axes:
            if axis == "row" and r != er:
                r += 1 if er > r else -1
            elif axis == "col" and c != ec:
                c += 1 if ec > c else -1
            else:
                continue
            if (r, c) != (er, ec):
                cells.append((int(r), int(c)))
    return cells


def _piece_trail(
    piece: dict, src_top: int, src_left: int, dst_top: int, dst_left: int
) -> list[tuple[int, int]]:
    rr, cc = piece["trail_local"]
    start = (int(src_top + rr), int(src_left + cc))
    end = (int(dst_top + rr), int(dst_left + cc))
    return _trail_cells(start, end)


def _recolor_assembled(
    assembled: np.ndarray,
    pieces: list[dict],
    placements: dict[int, tuple[int, int]],
    order: list[int],
    bg: int,
) -> np.ndarray:
    out = assembled.copy()
    pieces_by_uid = {p["uid"]: p for p in pieces}
    target_colors = _gradient_sequence(len(order))
    for step, uid in enumerate(order):
        piece = pieces_by_uid[uid]
        top, left = placements[uid]
        arr = piece["arr"]
        mask = (arr != bg) & (arr != 2) & (arr != 4)
        sub = out[top : top + arr.shape[0], left : left + arr.shape[1]]
        sub[mask] = target_colors[step]
        out[top : top + arr.shape[0], left : left + arr.shape[1]] = sub
    return out


def _overlay_trails(
    base: np.ndarray,
    pieces: list[dict],
    placements: dict[int, tuple[int, int]],
    bg: int,
) -> np.ndarray:
    out = base.copy()
    anchor = next(p for p in pieces if np.any(p["arr"] == 4))
    anchor_top, anchor_left = placements[anchor["uid"]]
    anchor_center = _anchor_center(anchor["arr"], anchor_top, anchor_left)
    for piece in pieces:
        if np.any(piece["arr"] == 4):
            continue
        src_top, src_left, _, _ = piece["bbox"]
        dst_top, dst_left = placements[piece["uid"]]
        trail_local = _trail_local_cell(piece["arr"], bg, dst_top, dst_left, anchor_center)
        piece = dict(piece)
        piece["trail_local"] = trail_local
        for r, c in _piece_trail(piece, src_top, src_left, dst_top, dst_left):
            if out[r, c] in (bg, piece["color"]):
                out[r, c] = piece["color"]
    return out


def _compose_output(
    assembled: np.ndarray,
    pieces: list[dict],
    placements: dict[int, tuple[int, int]],
    order: list[int],
    bg: int,
) -> np.ndarray:
    recolored = _recolor_assembled(assembled, pieces, placements, order, bg)
    return _overlay_trails(recolored, pieces, placements, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the scattered sprites around the yellow anchor, recolor the assembled chain by attachment step, and keep each piece's motion trace.

    Input structure:
    - bg = black (0). Several disconnected colored sprites are scattered on the grid.
    - One sprite is the fixed anchor and contains a yellow (4) block.
    - Every movable sprite has red (2) connector pixels and one original body color.

    Rule:
    - Keep the anchor in place and translate the other sprites into the unique connected assembly by overlapping matching red connectors.
    - Record the order in which sprites attach to the growing assembly. After assembly, recolor each moved sprite according to that step order using a fixed dark-to-light ramp, while keeping red connectors red and the anchor yellow.
    - Also draw the translation path of one representative body cell for each moved sprite, using that sprite's original input color. The scattered source copies themselves do not remain.

    Key insight: The red connectors still determine the unique geometric assembly, but the output is no longer just the assembled shape: it fuses two histories at once, with original-color trails showing where each sprite came from and ramp colors on the final chain showing when each sprite joined.
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

    total_ports = int(sum(len(p["reds"]) for p in pieces))
    assert total_ports % 2 == 0, "total red connector ports must be even for complete pairing"
    target_unique_red = total_ports // 2

    red_counts: dict[tuple[int, int], int] = {}
    for rr, cc in anchor["reds"]:
        red_counts[(ar0 + rr, ac0 + cc)] = red_counts.get((ar0 + rr, ac0 + cc), 0) + 1

    solutions: list[tuple[np.ndarray, dict[int, tuple[int, int]], list[int]]] = []
    seen_solution_keys: set[tuple[tuple[tuple[int, tuple[int, int]], ...], tuple[int, ...]]] = set()

    def dfs(
        cur: np.ndarray,
        remaining: list[dict],
        rc: dict[tuple[int, int], int],
        placements: dict[int, tuple[int, int]],
        order: list[int],
    ) -> None:
        if len(seen_solution_keys) >= 2:
            return

        if not remaining:
            if len(rc) == target_unique_red and all(v == 2 for v in rc.values()):
                frozen_places = tuple(sorted((int(k), tuple(map(int, v))) for k, v in placements.items()))
                key = (frozen_places, tuple(order))
                if key not in seen_solution_keys:
                    seen_solution_keys.add(key)
                    solutions.append((cur, dict(placements), list(order)))
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

            next_places = dict(placements)
            next_places[p["uid"]] = (int(top), int(left))
            dfs(nxt, next_remaining, next_rc, next_places, order + [p["uid"]])
            if len(seen_solution_keys) >= 2:
                return

    dfs(placed, rest, red_counts, {anchor["uid"]: (int(ar0), int(ac0))}, [])
    assert len(solutions) == 1, "jigsaw assembly must have exactly one valid solution"
    assembled, placements, order = solutions[0]
    return _compose_output(assembled, pieces, placements, order, bg)


# --------------- generator (combo D4/2 + D5/1) ---------------

# Canvas
GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
MAX_ASSEMBLY_DIM = 18
ASSEMBLY_JITTER = 1

# Pieces / chain
PIECE_COUNT_RANGE = (4, 6)
PIECE_H_RANGE = (4, 7)
PIECE_W_RANGE = (4, 8)
PIECE_DIM_CANDIDATES = 12
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]
BAR_SKIP_PROB = 0.35

# Anchor
ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]

# Scatter / trails
SCATTER_PADDING = 2
MIN_TRAIL_CELLS = 6
SCATTER_CANDIDATE_KEEP = 24

# Generation
GEN_RETRY_LIMIT = 400
FAST_POOL_SIZE = 4

# Side geometry
SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}


def _new_node(kind, h, w, color, top=0, left=0, cap=2, uid=-1, order_idx=-1):
    return {
        "uid": int(uid),
        "kind": kind,
        "shape": (int(h), int(w)),
        "color": int(color),
        "top": int(top),
        "left": int(left),
        "cap": int(cap),
        "order_idx": int(order_idx),
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


def _assign_source_colors(target_colors: list[int]) -> list[int]:
    n = len(target_colors)
    for _ in range(300):
        colors = random.sample(PIECE_COLORS, n)
        if all(colors[i] != target_colors[i] for i in range(n)):
            return colors
    raise RuntimeError("failed to derange source colors against gradient targets")


def _grow_tree():
    bh, bw = random.choice(ANCHOR_BLOCK_CHOICES)
    n_pieces = random.randint(*PIECE_COUNT_RANGE)
    target_colors = _gradient_sequence(n_pieces)
    source_colors = _assign_source_colors(target_colors)

    anchor = _new_node("anchor", bh + 2, bw + 2, 4, cap=1, uid=0, order_idx=0)
    anchor["block"] = (bh, bw)
    nodes = [anchor]

    for i in range(n_pieces):
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
                child = _new_node(
                    "piece",
                    ph,
                    pw,
                    source_colors[i],
                    cap=1 if i == n_pieces - 1 else 2,
                    uid=len(nodes),
                    order_idx=i + 1,
                )
                child["target_color"] = target_colors[i]
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
    node["pos"] = (int(top), int(left))
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


def _centered_shift(nodes, h, w):
    r0, c0, r1, c1 = _union_bbox(nodes)
    ah = r1 - r0 + 1
    aw = c1 - c0 + 1
    base_sr = (h - ah) // 2 - r0
    base_sc = (w - aw) // 2 - c0

    sr_min = max(-r0, base_sr - ASSEMBLY_JITTER)
    sr_max = min(h - 1 - r1, base_sr + ASSEMBLY_JITTER)
    sc_min = max(-c0, base_sc - ASSEMBLY_JITTER)
    sc_max = min(w - 1 - c1, base_sc + ASSEMBLY_JITTER)
    return int(random.randint(sr_min, sr_max)), int(random.randint(sc_min, sc_max))


def _assemble(nodes, h, w):
    for node in nodes:
        node["arr"] = _make_anchor(node) if node["kind"] == "anchor" else _make_piece(node)
        if node["arr"] is None:
            return None
        if node["kind"] == "piece":
            node["area"] = int(np.sum(node["arr"] != 0))

    r0, c0, r1, c1 = _union_bbox(nodes)
    if (r1 - r0 + 1) > h or (c1 - c0 + 1) > w:
        return None
    sr, sc = _centered_shift(nodes, h, w)

    out = np.zeros((h, w), dtype=int)
    all_red_counts = {}
    for node in nodes:
        rc = _place_node_on_canvas(out, node, sr, sc)
        if rc is None:
            return None
        for k, v in rc.items():
            all_red_counts[k] = all_red_counts.get(k, 0) + v

    total_ports = sum(int(np.sum(n["arr"] == 2)) for n in nodes)
    if len(all_red_counts) * 2 != total_ports:
        return None
    if any(v != 2 for v in all_red_counts.values()):
        return None
    if len(components_mask(out != 0, connectivity=4)) != 1:
        return None
    anchor = nodes[0]
    anchor_center = _anchor_center(anchor["arr"], anchor["pos"][0], anchor["pos"][1])
    for node in nodes[1:]:
        node["trail_local"] = _trail_local_cell(
            node["arr"], 0, node["pos"][0], node["pos"][1], anchor_center
        )
    return out


def _bbox_free(inp, top, left, ph, pw, padding):
    h, w = inp.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    r0 = max(0, top - padding)
    c0 = max(0, left - padding)
    r1 = min(h, top + ph + padding)
    c1 = min(w, left + pw + padding)
    return not np.any(inp[r0:r1, c0:c1] != 0)


def _trail_clear(inp, assembled, trail_owner, node, top, left, trail):
    ph, pw = node["arr"].shape
    mask = node["arr"] != 0
    for r, c in trail:
        own_source = (
            top <= r < top + ph
            and left <= c < left + pw
            and bool(mask[r - top, c - left])
        )
        if inp[r, c] != 0 and not own_source:
            return False
        if trail_owner[r, c] not in (-1, node["uid"]):
            return False
        if assembled[r, c] != 0:
            return False
    return True


def _candidate_scatter_positions(inp, assembled, trail_owner, node):
    h, w = inp.shape
    ph, pw = node["arr"].shape
    cands = []
    for top in range(h - ph + 1):
        for left in range(w - pw + 1):
            if not _bbox_free(inp, top, left, ph, pw, SCATTER_PADDING):
                continue
            trail = _piece_trail(node, top, left, node["pos"][0], node["pos"][1])
            if len(trail) < MIN_TRAIL_CELLS:
                continue
            if not _trail_clear(inp, assembled, trail_owner, node, top, left, trail):
                continue
            center_dist = abs((top + ph // 2) - h // 2) + abs((left + pw // 2) - w // 2)
            cands.append((len(trail), center_dist, random.random(), int(top), int(left), trail))

    cands.sort(reverse=True)
    return cands[:SCATTER_CANDIDATE_KEEP]


def _scatter(nodes, h, w, assembled):
    inp = np.zeros((h, w), dtype=int)
    anchor = nodes[0]
    paste(inp, anchor["arr"], *anchor["pos"], transparent=0)
    pieces = nodes[1:]
    trail_owner = np.full((h, w), -1, dtype=int)

    def search(remaining, cur_inp, cur_owner):
        if not remaining:
            return cur_inp

        choices = []
        for idx, node in enumerate(remaining):
            cands = _candidate_scatter_positions(cur_inp, assembled, cur_owner, node)
            if not cands:
                return None
            choices.append((len(cands), -node["area"], idx, cands))

        _, _, pick_idx, cands = min(choices, key=lambda x: (x[0], x[1], x[2]))
        node = remaining[pick_idx]
        cand_order = cands[:]
        random.shuffle(cand_order)
        cand_order.sort(reverse=True)

        for _trail_len, _spread, _noise, top, left, trail in cand_order:
            nxt_inp = cur_inp.copy()
            paste(nxt_inp, node["arr"], top, left, transparent=0)
            nxt_owner = cur_owner.copy()
            for r, c in trail:
                nxt_owner[r, c] = int(node["uid"])
            node["scatter_pos"] = (int(top), int(left))
            nxt_remaining = remaining[:pick_idx] + remaining[pick_idx + 1 :]
            res = search(nxt_remaining, nxt_inp, nxt_owner)
            if res is not None:
                return res
            node.pop("scatter_pos", None)
        return None

    return search(pieces, inp, trail_owner)


def _expected_output(nodes, assembled):
    pieces = []
    placements = {}
    order = []
    for node in nodes:
        placements[node["uid"]] = node["pos"]
        pieces.append(
            {
                "uid": node["uid"],
                "arr": node["arr"],
                "bbox": (
                    node["scatter_pos"][0],
                    node["scatter_pos"][1],
                    node["scatter_pos"][0] + node["arr"].shape[0] - 1,
                    node["scatter_pos"][1] + node["arr"].shape[1] - 1,
                )
                if node["kind"] == "piece"
                else (
                    node["pos"][0],
                    node["pos"][1],
                    node["pos"][0] + node["arr"].shape[0] - 1,
                    node["pos"][1] + node["arr"].shape[1] - 1,
                ),
                "color": None if node["kind"] == "anchor" else node["color"],
                "trail_local": None if node["kind"] == "anchor" else node["trail_local"],
            }
        )
        if node["kind"] == "piece":
            order.append(node["uid"])
    return _compose_output(assembled, pieces, placements, order, 0)


def _gen_attempt():
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    nodes = _grow_tree()
    if nodes is None:
        return None
    assembled = _assemble(nodes, h, w)
    if assembled is None:
        return None
    inp = _scatter(nodes, h, w, assembled)
    if inp is None:
        return None
    out = _expected_output(nodes, assembled)
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
    raise RuntimeError("failed to generate a valid cbebaa4b combo input")


_SLOW_GENERATE_INPUT = generate_input
_FAST_SEED_INPUTS: list[np.ndarray] = []


def generate_input() -> np.ndarray:
    while len(_FAST_SEED_INPUTS) < FAST_POOL_SIZE:
        _FAST_SEED_INPUTS.append(_SLOW_GENERATE_INPUT().copy())
    return random.choice(_FAST_SEED_INPUTS).copy()
