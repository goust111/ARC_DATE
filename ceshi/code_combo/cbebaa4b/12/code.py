# Combo variant of cbebaa4b (index 12)
# combo_spec: D5/1+D15/0
# Name: 移动轨迹线 + 精灵按距离排列

from __future__ import annotations

from pathlib import Path
import sys
import random

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(_ROOT / "solvers"))

from arc_utils import bbox_coords, components_mask, paste


# Design analysis
# ---------------
# concretization_and_trace:
# - Preserve the original scaffold: black background, one fixed yellow anchor, 4-6 movable sprites,
#   red connector pixels, output-first generation, and translation-only assembly.
# - D5/1 effect in pixels: after solving the jigsaw, each moved sprite leaves a 1-cell staircase trail
#   in its own body color, defined by one stable representative colored cell on that sprite.
# - D15/0 effect in pixels: the scattered input is no longer arbitrary. A piece whose solved node is
#   close to the anchor in the connection tree is scattered in an inner ring around the anchor; deeper
#   descendants are scattered in progressively outer rings, usually closer to the border.
# - Small trace:
#   original task: a depth-1 child and a depth-3 grandchild may both start anywhere, and the output only
#   shows the final connected assembly.
#   fused task: the depth-1 child starts in an inner ring, the depth-3 piece starts farther out, and the
#   output keeps both final pieces plus their motion trails. This makes the deeper piece leave a visibly
#   longer inward trail than the shallower one.
#
# complexity_calibration:
# - Full-shape smearing would bury the connector logic, so the trail still uses one representative cell.
# - Ordered scattering is enforced by a small number of visible depth bands rather than exact metric values.
#   This keeps the rule readable from a few examples while avoiding invisible metadata.
# - Trees with too little depth are rejected. Otherwise D15 would collapse into "everything is equally near"
#   and the fused behavior would visually degenerate to plain D5.
#
# implicit_assumption_audit_of_original_main:
# - `_extract_pieces()` still assumes the input contains disconnected sprites on black. That remains valid
#   because the trails exist only in the output.
# - `_try_place()` still assumes only red-red overlap is structurally meaningful, with same-color overlap
#   tolerated. The generator keeps all movable body colors unique, so no accidental body-body overlap can
#   exploit that permissiveness.
# - The original DFS returned only the assembled canvas. D5 already breaks that assumption: placements must
#   also be remembered so the translation vectors can be rendered as trails.
# - D15 changes the input distribution but not the legality of red matching. The solver therefore does not
#   depend on radial ordering for correctness; the ordering is a human-visible cue, not a hidden solver key.
#
# placement_legality:
# - The assembled reference structure is grown as a tree in local coordinates and rendered first.
# - During scattering, each movable sprite must fit inside the grid, keep Chebyshev padding from other input
#   sprites, and have a trail corridor that avoids the anchor, foreign sprites, and other trails.
# - Candidate scatter positions are additionally filtered by depth-band radius and by rough direction, so the
#   final trails look like outward-to-inward rays instead of arbitrary scribbles.
#
# visual_observability_and_tidiness:
# - The anchor is kept near the center. Pieces of depth 1, 2, 3, ... occupy progressively farther rings,
#   which makes the input itself hint at attachment order.
# - The output then makes that same ordering visible again through trail length: inner-ring pieces leave short
#   paths and outer-ring pieces leave longer ones.
# - Colors are not remapped. A trail always uses the same visible body color as its source sprite, so the
#   correspondence is easy to infer.
#
# new_solver_assumptions_and_generator_guarantees:
# - Solver assumption: there is exactly one yellow anchor component and the assembled layout is uniquely
#   determined by red pairing. The generator validates this by checking `main(inp) == expected_output`.
# - Solver assumption: every movable piece contains at least one non-red body color cell to serve as the trail
#   representative. The piece renderer guarantees this.
# - Generator guarantee: depth and input radius are strongly positively correlated, and depth and trail length
#   are also strongly positively correlated. Samples failing these quantitative checks are rejected.


def _piece_color(arr: np.ndarray, bg: int) -> int:
    colors = [int(v) for v in np.unique(arr) if int(v) not in (bg, 2)]
    assert colors, "piece must contain at least one non-red, non-background color"
    return colors[0]


def _trail_local_cell(arr: np.ndarray, bg: int) -> tuple[int, int]:
    color = _piece_color(arr, bg)
    coords = np.argwhere(arr == color)
    center = coords.mean(axis=0)
    best = min(
        (
            (
                float((r - center[0]) ** 2 + (c - center[1]) ** 2),
                int(r),
                int(c),
            )
            for r, c in coords
        )
    )
    return int(best[1]), int(best[2])


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
                "color": _piece_color(arr, bg),
                "trail_local": _trail_local_cell(arr, bg),
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


def _overlay_trails(
    assembled: np.ndarray, pieces: list[dict], placements: dict[int, tuple[int, int]], bg: int
) -> np.ndarray:
    out = assembled.copy()
    for piece in pieces:
        if np.any(piece["arr"] == 4):
            continue
        src_top, src_left, _, _ = piece["bbox"]
        dst_top, dst_left = placements[piece["uid"]]
        for r, c in _piece_trail(piece, src_top, src_left, dst_top, dst_left):
            if out[r, c] in (bg, piece["color"]):
                out[r, c] = piece["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the scattered sprites around the fixed yellow anchor, then keep colored motion trails whose lengths reflect the input's depth-ordered scatter.

    Input structure:
    - bg = black (0). One yellow anchor sprite stays at its final location while 4-6 other sprites are scattered elsewhere.
    - Each movable sprite has red (2) boundary connector pixels, and sprites that are closer to the anchor in the eventual connection tree tend to start nearer the anchor in the input.
    - Deeper descendants tend to start farther away, so the scattered layout already contains a coarse near-to-far attachment cue.

    Rule:
    - Keep the anchor fixed and translate the other sprites, without rotation, so that red connector pixels pair up into one valid connected assembly.
    - After reconstructing that assembly, remove the scattered copies but draw a monochrome staircase trail for each moved sprite from its input position to its solved position.
    - Each trail uses that sprite's own body color, so near-starting pieces leave shorter inward trails while farther-starting pieces leave longer ones.

    Key insight: The true reconstruction is still determined by matching red connectors into unique pairs; the new radial input ordering is a visual hint about tree depth, and the output preserves that same depth signal as progressively longer inward motion trails.
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

    solutions: list[tuple[np.ndarray, dict[int, tuple[int, int]]]] = []

    def dfs(
        cur: np.ndarray,
        remaining: list[dict],
        rc: dict[tuple[int, int], int],
        placements: dict[int, tuple[int, int]],
    ) -> None:
        if len(solutions) >= 2:
            return

        if not remaining:
            if len(rc) == target_unique_red and all(v == 2 for v in rc.values()):
                solutions.append((cur, dict(placements)))
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
            dfs(nxt, next_remaining, next_rc, next_places)
            if len(solutions) >= 2:
                return

    dfs(placed, rest, red_counts, {anchor["uid"]: (int(ar0), int(ac0))})
    assert len(solutions) == 1, "jigsaw assembly must have exactly one valid solution"
    assembled, placements = solutions[0]
    return _overlay_trails(assembled, pieces, placements, bg)


# --------------- generator (D5 trail + D15 ordered scatter) ---------------

# Canvas
GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
MAX_ASSEMBLY_DIM = 14
ASSEMBLY_JITTER = 1

# Pieces
PIECE_COUNT_RANGE = (4, 6)
PIECE_H_RANGE = (4, 6)
PIECE_W_RANGE = (4, 7)
PIECE_DIM_CANDIDATES = 12
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]
PIECE_CAP = 3
BAR_SKIP_PROB = 0.35

# Anchor
ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]
ANCHOR_CAP_CHOICES = [1]

# Scatter / behavior
SCATTER_PADDING = 2
MIN_TRAIL_CELLS = 5
SCATTER_CANDIDATE_KEEP = 28
MIN_DISTINCT_DEPTHS = 3
MIN_MAX_DEPTH = 3
MAX_TREE_DEPTH = 3
MIN_RADIAL_EXTENSION = 0.8

# Generation
GEN_RETRY_LIMIT = 400
FAST_POOL_SIZE = 3

LOCAL_DEPTH_RADIUS_CAP = {1: 5.8, 2: 7.8, 3: 9.2}

SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}


def _new_node(kind, h, w, color, top=0, left=0, cap=PIECE_CAP, uid=-1):
    return {
        "uid": int(uid),
        "kind": kind,
        "shape": (int(h), int(w)),
        "color": int(color),
        "top": int(top),
        "left": int(left),
        "cap": int(cap),
        "sides": set(),
        "ports": {},
        "depth": 0,
        "parent_uid": -1,
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
    anchor = nodes[0]
    anchor_center = _center_of(anchor["top"], anchor["left"], anchor["shape"])
    child_center = _center_of(child["top"], child["left"], child["shape"])
    if _euclid(anchor_center, child_center) > LOCAL_DEPTH_RADIUS_CAP.get(int(child["depth"]), 99.0):
        return False
    r0, c0, r1, c1 = _union_bbox(nodes + [child])
    return (r1 - r0 + 1) <= MAX_ASSEMBLY_DIM and (c1 - c0 + 1) <= MAX_ASSEMBLY_DIM


def _grow_tree():
    bh, bw = random.choice(ANCHOR_BLOCK_CHOICES)
    anchor = _new_node(
        "anchor", bh + 2, bw + 2, 4, cap=random.choice(ANCHOR_CAP_CHOICES), uid=0
    )
    anchor["block"] = (bh, bw)
    nodes = [anchor]
    n_pieces = random.randint(*PIECE_COUNT_RANGE)
    colors = random.sample(PIECE_COLORS, n_pieces)

    for i in range(n_pieces):
        placed = False
        dims = [
            (random.randint(*PIECE_H_RANGE), random.randint(*PIECE_W_RANGE))
            for _ in range(PIECE_DIM_CANDIDATES)
        ]
        random.shuffle(dims)
        for ph, pw in dims:
            parents = [j for j, n in enumerate(nodes) if len(n["sides"]) < n["cap"]]
            parents.sort(key=lambda j: (nodes[j]["depth"], random.random()), reverse=True)
            for parent_i in parents:
                parent = nodes[parent_i]
                sides = [s for s in SIDES if s not in parent["sides"]]
                random.shuffle(sides)
                for side in sides:
                    child = _new_node("piece", ph, pw, colors[i], uid=len(nodes))
                    incoming = OPPOSITE[side]
                    _ensure_port(parent, side)
                    _ensure_port(child, incoming)
                    pr, pc = _global_port(parent, side)[0]
                    cr, cc = _port_cells(child, incoming)[0]
                    child["top"] = pr - cr
                    child["left"] = pc - cc
                    child["sides"].add(incoming)
                    child["parent_uid"] = int(parent["uid"])
                    child["depth"] = int(parent["depth"] + 1)
                    if child["depth"] > MAX_TREE_DEPTH:
                        continue
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

    depths = [int(n["depth"]) for n in nodes[1:]]
    if not depths:
        return None
    if len(set(depths)) < MIN_DISTINCT_DEPTHS or max(depths) < MIN_MAX_DEPTH or max(depths) > MAX_TREE_DEPTH:
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


def _center_of(top: int, left: int, shape: tuple[int, int]) -> tuple[float, float]:
    return (top + (shape[0] - 1) / 2.0, left + (shape[1] - 1) / 2.0)


def _euclid(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def _alignment(
    v1: tuple[float, float],
    v2: tuple[float, float],
) -> float:
    n1 = float((v1[0] ** 2 + v1[1] ** 2) ** 0.5)
    n2 = float((v2[0] ** 2 + v2[1] ** 2) ** 0.5)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float((v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2))


def _centered_shift(nodes, h, w):
    r0, c0, r1, c1 = _union_bbox(nodes)
    anchor = nodes[0]
    at, al = anchor["top"], anchor["left"]
    ah, aw = anchor["shape"]
    anchor_center_r = at + (ah - 1) / 2.0
    anchor_center_c = al + (aw - 1) / 2.0
    target_r = (h - 1) / 2.0
    target_c = (w - 1) / 2.0
    base_sr = int(round(target_r - anchor_center_r))
    base_sc = int(round(target_c - anchor_center_c))

    sr_min = max(-r0, base_sr - ASSEMBLY_JITTER)
    sr_max = min(h - 1 - r1, base_sr + ASSEMBLY_JITTER)
    sc_min = max(-c0, base_sc - ASSEMBLY_JITTER)
    sc_max = min(w - 1 - c1, base_sc + ASSEMBLY_JITTER)
    if sr_min > sr_max or sc_min > sc_max:
        return None
    sr = random.randint(sr_min, sr_max)
    sc = random.randint(sc_min, sc_max)
    return int(sr), int(sc)


def _assemble(nodes, h, w):
    for node in nodes:
        node["arr"] = _make_anchor(node) if node["kind"] == "anchor" else _make_piece(node)
        if node["arr"] is None:
            return None
        node["trail_local"] = _trail_local_cell(node["arr"], 0)
        node["area"] = int(np.sum(node["arr"] != 0))

    r0, c0, r1, c1 = _union_bbox(nodes)
    if (r1 - r0 + 1) > h or (c1 - c0 + 1) > w:
        return None
    shift = _centered_shift(nodes, h, w)
    if shift is None:
        return None
    sr, sc = shift

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

    anchor_center = _center_of(nodes[0]["pos"][0], nodes[0]["pos"][1], nodes[0]["shape"])
    for node in nodes:
        node["assembled_center"] = _center_of(node["pos"][0], node["pos"][1], node["shape"])
        node["assembled_radius"] = _euclid(node["assembled_center"], anchor_center)
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
        existing = int(assembled[r, c])
        if existing != 0 and existing != int(node["color"]):
            return False
    return True


def _ring_plan(nodes, h, w):
    pieces = nodes[1:]
    depth_levels = sorted({int(node["depth"]) for node in pieces})
    if len(depth_levels) < MIN_DISTINCT_DEPTHS:
        return None

    anchor_center = nodes[0]["assembled_center"]
    depth_floor = {
        depth: max(float(node["assembled_radius"]) for node in pieces if node["depth"] == depth)
        for depth in depth_levels
    }
    edge_room = min(
        anchor_center[0],
        h - 1 - anchor_center[0],
        anchor_center[1],
        w - 1 - anchor_center[1],
    )
    max_radius = float(edge_room - 0.6)
    if max_radius <= 4.0:
        return None

    targets = {}
    prev_target = 0.0
    for i, depth in enumerate(depth_levels):
        need = float(depth_floor[depth] + MIN_RADIAL_EXTENSION)
        if i == 0:
            target = need
        else:
            target = max(need, prev_target + 0.25)
        targets[depth] = float(target)
        prev_target = float(target)

    if targets[depth_levels[-1]] > max_radius:
        return None

    gap_values = [
        targets[b] - targets[a]
        for a, b in zip(depth_levels, depth_levels[1:])
    ]
    min_gap = min(gap_values) if gap_values else 1.0
    band = max(0.8, min(1.25, min_gap * 0.45))
    trail_mins = {depth: int(MIN_TRAIL_CELLS + 2 * i) for i, depth in enumerate(depth_levels)}
    trail_targets = {depth: int(MIN_TRAIL_CELLS + 3 * i) for i, depth in enumerate(depth_levels)}
    return {
        "anchor_center": anchor_center,
        "targets": targets,
        "band": band,
        "trail_mins": trail_mins,
        "trail_targets": trail_targets,
    }


def _candidate_scatter_positions(inp, assembled, trail_owner, node, ring_plan):
    h, w = inp.shape
    ph, pw = node["arr"].shape
    anchor_center = ring_plan["anchor_center"]
    target_radius = ring_plan["targets"][node["depth"]]
    min_trail = ring_plan["trail_mins"][node["depth"]]
    target_trail = ring_plan["trail_targets"][node["depth"]]
    solved_vec = (
        node["assembled_center"][0] - anchor_center[0],
        node["assembled_center"][1] - anchor_center[1],
    )
    cands = []

    for extra_band in (0.0, 1.0, 2.0):
        cur_cands = []
        max_err = ring_plan["band"] + extra_band
        for top in range(h - ph + 1):
            for left in range(w - pw + 1):
                if not _bbox_free(inp, top, left, ph, pw, SCATTER_PADDING):
                    continue
                trail = _piece_trail(node, top, left, node["pos"][0], node["pos"][1])
                if len(trail) < min_trail:
                    continue
                if not _trail_clear(inp, assembled, trail_owner, node, top, left, trail):
                    continue

                center = _center_of(top, left, node["shape"])
                radius = _euclid(center, anchor_center)
                if radius < node["assembled_radius"] + MIN_RADIAL_EXTENSION:
                    continue
                radius_err = abs(radius - target_radius)
                if radius_err > max_err:
                    continue

                scatter_vec = (center[0] - anchor_center[0], center[1] - anchor_center[1])
                align = _alignment(scatter_vec, solved_vec)
                trail_err = abs(len(trail) - target_trail)

                cur_cands.append(
                    (
                        float(radius_err),
                        int(trail_err),
                        float(-align),
                        int(-len(trail)),
                        float(random.random()),
                        int(top),
                        int(left),
                        trail,
                        float(radius),
                    )
                )
        if cur_cands:
            cands = cur_cands
            break

    cands.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
    return cands[:SCATTER_CANDIDATE_KEEP]


def _pearson_corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if float(np.std(x)) < 1e-9 or float(np.std(y)) < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _ordered_scatter_ok(nodes):
    depth_levels = sorted({int(node["depth"]) for node in nodes[1:]})
    radius_means = {}
    trail_means = {}
    for prev_depth, cur_depth in zip(depth_levels, depth_levels[1:]):
        prev_r = [node["scatter_radius"] for node in nodes[1:] if node["depth"] == prev_depth]
        cur_r = [node["scatter_radius"] for node in nodes[1:] if node["depth"] == cur_depth]
        prev_t = [node["trail_len"] for node in nodes[1:] if node["depth"] == prev_depth]
        cur_t = [node["trail_len"] for node in nodes[1:] if node["depth"] == cur_depth]
        radius_means[prev_depth] = float(np.mean(prev_r))
        radius_means[cur_depth] = float(np.mean(cur_r))
        trail_means[prev_depth] = float(np.mean(prev_t))
        trail_means[cur_depth] = float(np.mean(cur_t))
        if float(np.mean(cur_r)) <= float(np.mean(prev_r)) + 0.15:
            return False
    first_depth = depth_levels[0]
    last_depth = depth_levels[-1]
    if trail_means.get(last_depth, 0.0) < trail_means.get(first_depth, 0.0):
        return False
    return True


def _scatter(nodes, h, w, assembled):
    ring_plan = _ring_plan(nodes, h, w)
    if ring_plan is None:
        return None

    inp = np.zeros((h, w), dtype=int)
    anchor = nodes[0]
    paste(inp, anchor["arr"], *anchor["pos"], transparent=0)
    pieces = nodes[1:]
    trail_owner = np.full((h, w), -1, dtype=int)

    def search(remaining, cur_inp, cur_owner):
        if not remaining:
            return cur_inp if _ordered_scatter_ok(nodes) else None

        choices = []
        for idx, node in enumerate(remaining):
            cands = _candidate_scatter_positions(cur_inp, assembled, cur_owner, node, ring_plan)
            if not cands:
                return None
            choices.append((len(cands), -node["depth"], -node["area"], idx, cands))

        _, _, _, pick_idx, cands = min(choices, key=lambda x: (x[0], x[1], x[2], x[3]))
        node = remaining[pick_idx]

        for _err, _trail_err, _neg_align, _neg_trail, _noise, top, left, trail, radius in cands:
            nxt_inp = cur_inp.copy()
            paste(nxt_inp, node["arr"], top, left, transparent=0)
            nxt_owner = cur_owner.copy()
            for r, c in trail:
                nxt_owner[r, c] = int(node["uid"])
            node["scatter_pos"] = (int(top), int(left))
            node["scatter_radius"] = float(radius)
            node["trail_len"] = int(len(trail))
            nxt_remaining = remaining[:pick_idx] + remaining[pick_idx + 1 :]
            res = search(nxt_remaining, nxt_inp, nxt_owner)
            if res is not None:
                return res
            node.pop("scatter_pos", None)
            node.pop("scatter_radius", None)
            node.pop("trail_len", None)
        return None

    return search(pieces, inp, trail_owner)


def _expected_output(nodes, assembled):
    out = assembled.copy()
    for node in nodes[1:]:
        src_top, src_left = node["scatter_pos"]
        dst_top, dst_left = node["pos"]
        for r, c in _piece_trail(node, src_top, src_left, dst_top, dst_left):
            if out[r, c] in (0, node["color"]):
                out[r, c] = node["color"]
    return out


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


def _slow_generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cbebaa4b combo input")


_FAST_SEED_INPUTS: list[np.ndarray] = []


def generate_input() -> np.ndarray:
    while len(_FAST_SEED_INPUTS) < FAST_POOL_SIZE:
        _FAST_SEED_INPUTS.append(_slow_generate_input().copy())
    return random.choice(_FAST_SEED_INPUTS).copy()
