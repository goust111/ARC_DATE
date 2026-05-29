# Variant of cbebaa4b
# Dimension: D1 — 连接器匹配条件 (match_cond)
# Change: 精灵的红色像素与目标（中央黄色方块或已连接精灵）的红色像素对齐，形成连接
#   → 旋转/翻转后连接器匹配
#
# Design analysis
# ----------------
# concretization_and_trace:
# - The anchor is unchanged: one fixed sprite containing a yellow (4) rectangle and red (2) ports.
# - Each non-anchor sprite is still a single connected mono-color shape with red boundary ports, but now
#   the scattered input may show that sprite in a rotated or mirrored orientation.
# - The output is the completed assembly selected by the solver's port-matching search. Therefore the
#   solver must not merely translate pieces: it must first infer which dihedral transform of each sprite
#   restores a valid connector geometry.
# - Small trace:
#   original logic: a leaf piece whose two red cells sit on its left edge can only be translated so that
#   those two cells land on an exposed assembly port.
#   variant logic: the same scattered piece may arrive rotated 90 degrees, placing those two red cells on
#   its top edge. The solver must test that transformed bbox and transformed red coordinates before trying
#   translation. The resulting output uses the reoriented piece, not the raw scattered orientation.
#
# complexity_calibration:
# - The original generator could accidentally produce symmetric pieces, making rotation/flip invisible.
# - To keep D1 observable, every movable piece gets one asymmetric interior spur and is accepted only if
#   it has several unique dihedral variants. This keeps the visual rule human-visible without adding new
#   colors or unrelated mechanics.
# - The connection graph is biased to a path-shaped tree. This is the smallest topology restriction that
#   keeps the transformed search space from exploding into many equally valid branch permutations; the
#   novel difficulty stays in choosing each piece orientation, not in unrelated graph ambiguity.
# - Colors of movable pieces remain unique, matching the original code path and avoiding accidental same-
#   color overlap ambiguities in `_try_place`.
#
# implicit_assumption_audit_of_original_main:
# - `ph, pw = p["shape"]` assumed every scattered piece has one fixed bbox. In this variant, each piece
#   has multiple candidate transformed bboxes.
# - `for pr, pc in p["reds"]` assumed connector coordinates were fixed in the input orientation. They now
#   move with rotation/flip and must be read from each transformed variant.
# - Move deduplication by `(piece, top, left)` was sufficient before. Now orientation matters, so the key
#   must include the chosen transformed variant.
# - `_try_place(cur, p["arr"], top, left, bg)` assumed the raw extracted piece already had final pose.
#   The variant must place a transformed candidate array instead.
# - Generator scattering previously pasted `node["arr"]` directly, so input and output orientations matched.
#   The variant must scatter a different `node["scatter_arr"]` while keeping an assembled reference layout
#   only as an internal legality scaffold.
#
# placement_legality:
# - Assembly legality is still the original one: only red-on-red overlap is intended, all pieces must stay
#   in-bounds, and every red pixel must be paired exactly once.
# - The generator still works output-first, validates the assembled red-pair counts, then scatters the
#   transformed sprites with padding so extracted input components are clean.
#
# visual_observability_and_tidiness:
# - Every movable piece is visibly asymmetric, so rotated/flipped input sprites look different by eye.
# - The grid is still sparse black background with separated sprites; no extra overlay channels are added.
# - No color mapping is introduced, so there is no hidden color-id logic for the human solver to infer.
#
# new_solver_assumptions_and_generator_guarantees:
# - Each movable sprite has at least one non-identity dihedral transform and usually several; the generator
#   enforces this by rejecting symmetric renders.
# - All movable sprites are scattered in a non-identity orientation, and at least one of them uses a flip,
#   so the D1 change is not silently inactive.
# - The anchor remains untransformed and stays at its output position, preserving the original global frame.
# - The current port-feasibility test may leave a very small tie set of connector-valid assemblies; the
#   solver resolves that tie by a fixed constrained DFS order, and the generator validates against exactly
#   that canonical search outcome.

import os
import sys
import random
import numpy as np

_ROOT = "/home/lixinyu/snap/new"
_SOLVERS = os.path.join(_ROOT, "solvers")
if _SOLVERS not in sys.path:
    sys.path.append(_SOLVERS)

from arc_utils import bbox_coords, components_mask, paste, random_free_location


def _piece_record(arr: np.ndarray, bg: int) -> dict:
    reds = np.argwhere(arr == 2)
    return {
        "arr": arr.copy(),
        "reds": [(int(r), int(c)) for r, c in reds],
        "shape": (int(arr.shape[0]), int(arr.shape[1])),
        "area": int(np.sum(arr != bg)),
    }


def _variant_records(arr: np.ndarray, bg: int) -> list[dict]:
    """Enumerate all unique dihedral variants of a piece."""
    variants = []
    seen = set()
    for k in range(4):
        rot = np.rot90(arr, k)
        for is_flip in (False, True):
            cur = np.fliplr(rot) if is_flip else rot
            key = (tuple(cur.shape), cur.tobytes())
            if key in seen:
                continue
            seen.add(key)
            rec = _piece_record(cur, bg)
            rec["is_flip"] = bool(is_flip)
            rec["rot_k"] = int(k)
            rec["code"] = "id" if (k == 0 and not is_flip) else (f"fr{90 * k}" if is_flip else f"r{90 * k}")
            variants.append(rec)
    return variants


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        piece = _piece_record(arr, bg)
        piece["bbox"] = (int(r0), int(c0), int(r1), int(c1))
        pieces.append(piece)
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered asymmetric sprites around a fixed yellow anchor by matching red ports after reorientation.

    Input structure:
    - bg = black (0). Several colored sprites are scattered across the grid.
    - One anchor sprite contains a yellow (4) block and remains at its final position.
    - Every movable sprite has red (2) connector pixels, and its scattered orientation may already be rotated or flipped.

    Rule:
    - Keep the anchor fixed and attach the other sprites one at a time to exposed red ports.
    - Before translating a sprite, you may rotate it by 90/180/270 degrees or mirror it, then align one of its red pixels with an unpaired red pixel on the growing assembly.
    - A valid final assembly uses every red connector exactly once in a red-on-red pair and forms one completed structure.

    Key insight: The red pixels still define the attachment graph, but now each candidate piece must be tested across its unique dihedral orientations because the input pose is not necessarily the output pose.
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
    for p in rest:
        p["variants"] = _variant_records(p["arr"], bg)

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
                if not any(np.array_equal(cur, prev) for prev in solutions):
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
                for vidx, v in enumerate(p["variants"]):
                    ph, pw = v["shape"]
                    for pr, pc in v["reds"]:
                        top = tr - pr
                        left = tc - pc
                        if top < 0 or left < 0 or top + ph > h or left + pw > w:
                            continue
                        bad = False
                        overlap = 0
                        for rr, cc in v["reds"]:
                            gr, gc = top + rr, left + cc
                            prev = rc.get((gr, gc), 0)
                            if prev >= 2:
                                bad = True
                                break
                            if (gr, gc) in rc1:
                                overlap += 1
                        if bad:
                            continue
                        moves.append((idx, vidx, top, left, overlap))
            if not moves:
                return
            if best_moves is None or len(moves) < len(best_moves):
                best_moves = moves
                if len(best_moves) == 1:
                    break

        assert best_moves is not None, "there must be at least one feasible connector move"

        dedup: dict[tuple[int, int, int, int], int] = {}
        for idx, vidx, top, left, ov in best_moves:
            k = (idx, vidx, top, left)
            dedup[k] = max(dedup.get(k, 0), ov)
        moves2 = [(idx, vidx, top, left, ov) for (idx, vidx, top, left), ov in dedup.items()]
        moves2.sort(key=lambda x: (-x[4], -remaining[x[0]]["area"], x[2], x[3], x[0], x[1]))

        for idx, vidx, top, left, _ov in moves2:
            p = remaining[idx]
            v = p["variants"][vidx]
            next_remaining = remaining[:idx] + remaining[idx + 1 :]

            next_rc = dict(rc)
            for rr, cc in v["reds"]:
                gr, gc = top + rr, left + cc
                next_rc[(gr, gc)] = next_rc.get((gr, gc), 0) + 1

            nxt = _try_place(cur, v["arr"], top, left, bg)
            if nxt is None:
                continue
            dfs(nxt, next_remaining, next_rc)
            if len(solutions) >= 2:
                return

    global LAST_SOLUTION_COUNT
    dfs(placed, rest, red_counts)
    LAST_SOLUTION_COUNT = len(solutions)
    assert solutions, "jigsaw assembly must have at least one valid solution"
    return solutions[0]


# --------------- generator (o5x, D1 variant) ---------------

# ── Tunable configuration ──────────────────────────────────────────────
GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
MAX_ASSEMBLY_DIM = 24

PIECE_COUNT_RANGE = (4, 6)
PIECE_H_RANGE = (4, 7)
PIECE_W_RANGE = (4, 8)
PIECE_DIM_CANDIDATES = 12
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]
PIECE_CAP = 2
BAR_SKIP_PROB = 0.35
MIN_UNIQUE_VARIANTS = 4

ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]
ANCHOR_CAP_CHOICES = [1]

SCATTER_PADDING = 2
GEN_RETRY_LIMIT = 300

SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}
TAG_CORNERS = {
    "nw": lambda h, w: (1, 1),
    "ne": lambda h, w: (1, w - 2),
    "sw": lambda h, w: (h - 2, 1),
    "se": lambda h, w: (h - 2, w - 2),
}

LAST_GEN_INFO: dict = {}
LAST_SOLUTION_COUNT = 0


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
    """Render a thicker asymmetric piece while keeping only red boundary overlaps."""
    h, w = node["shape"]
    color = node["color"]
    used = set(node["sides"])
    arr = np.zeros((h, w), dtype=int)

    core_h = 2 if h <= 5 else 3
    core_w = 2 if w <= 6 else 3
    core_r0 = random.randint(1, h - core_h - 1)
    core_c0 = random.randint(1, w - core_w - 1)
    arr[core_r0 : core_r0 + core_h, core_c0 : core_c0 + core_w] = color
    hub = (core_r0 + core_h // 2, core_c0 + core_w // 2)

    bar_sides = [s for s in SIDES if s not in used]
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

    order = random.randint(0, 1)
    for t in terminals:
        _draw_path(arr, t, hub, color, order)
    for side in bar_sides:
        _draw_path(arr, hub, _bar_endpoint((h, w), side, hub), color, 1 - order)

    corner_names = list(TAG_CORNERS)
    random.shuffle(corner_names)
    chosen = None
    for name in corner_names:
        trial = arr.copy()
        tag = TAG_CORNERS[name](h, w)
        trial[tag] = color
        _draw_path(trial, hub, tag, color, random.randint(0, 1))
        if len(components_mask(trial != 0, connectivity=4)) != 1:
            continue
        if int(np.sum(trial == 2)) != 2 * len(used):
            continue
        if len(_variant_records(trial, 0)) < MIN_UNIQUE_VARIANTS:
            continue
        arr = trial
        chosen = name
        break
    if chosen is None:
        return None

    node["tag_corner"] = chosen
    if len(components_mask(arr != 0, connectivity=4)) != 1:
        return None
    if int(np.sum(arr == 2)) != 2 * len(used):
        return None
    if len(_variant_records(arr, 0)) < MIN_UNIQUE_VARIANTS:
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


def _grow_tree():
    bh, bw = random.choice(ANCHOR_BLOCK_CHOICES)
    anchor = _new_node("anchor", bh + 2, bw + 2, 4, cap=random.choice(ANCHOR_CAP_CHOICES))
    anchor["block"] = (bh, bw)
    nodes = [anchor]
    n_pieces = random.randint(*PIECE_COUNT_RANGE)
    colors = random.sample(PIECE_COLORS, n_pieces)
    tail_i = 0

    for i in range(n_pieces):
        placed = False
        dims = [
            (random.randint(*PIECE_H_RANGE), random.randint(*PIECE_W_RANGE))
            for _ in range(PIECE_DIM_CANDIDATES)
        ]
        random.shuffle(dims)
        for ph, pw in dims:
            parent = nodes[tail_i]
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
                if not _candidate_fits(nodes, tail_i, child):
                    continue
                parent["sides"].add(side)
                nodes.append(child)
                tail_i = len(nodes) - 1
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

    total_ports = sum(int(np.sum(n["arr"] == 2)) for n in nodes)
    if len(all_red_counts) * 2 != total_ports:
        return None
    if any(v != 2 for v in all_red_counts.values()):
        return None
    if len(components_mask(out != 0, connectivity=4)) != 1:
        return None
    return out


def _choose_scatter_variants(nodes):
    piece_infos = []
    for node in nodes[1:]:
        variants = _variant_records(node["arr"], 0)
        non_identity = [v for v in variants if not np.array_equal(v["arr"], node["arr"])]
        flip_choices = [v for v in non_identity if v["is_flip"]]
        if not non_identity:
            return False
        piece_infos.append((node, variants, non_identity, flip_choices))

    if not piece_infos:
        return False

    for node, variants, non_identity, _flip_choices in piece_infos:
        node["all_variants"] = variants
        chosen = random.choice(non_identity)
        node["scatter_arr"] = chosen["arr"]
        node["scatter_variant"] = chosen["code"]
        node["scatter_is_flip"] = bool(chosen["is_flip"])

    flipped = [node for node, _v, _n, flips in piece_infos if node.get("scatter_is_flip")]
    if not flipped:
        candidates = [(node, flips) for node, _v, _n, flips in piece_infos if flips]
        if not candidates:
            return False
        node, flips = random.choice(candidates)
        chosen = random.choice(flips)
        node["scatter_arr"] = chosen["arr"]
        node["scatter_variant"] = chosen["code"]
        node["scatter_is_flip"] = True

    return True


def _scatter(nodes, h, w):
    inp = np.zeros((h, w), dtype=int)
    anchor = nodes[0]
    paste(inp, anchor["arr"], *anchor["pos"], transparent=0)
    rest = nodes[1:]
    random.shuffle(rest)
    for node in rest:
        arr = node["scatter_arr"]
        try:
            r, c = random_free_location(
                inp,
                arr,
                transparent=0,
                padding=SCATTER_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inp, arr, r, c, transparent=0)
    return inp


def _record_last_gen_info(nodes):
    global LAST_GEN_INFO
    piece_nodes = nodes[1:]
    LAST_GEN_INFO = {
        "piece_count": len(piece_nodes),
        "changed_piece_count": int(
            sum(1 for n in piece_nodes if n.get("scatter_variant") not in (None, "id"))
        ),
        "flip_piece_count": int(sum(1 for n in piece_nodes if n.get("scatter_is_flip"))),
        "variant_codes": [n.get("scatter_variant") for n in piece_nodes],
        "unique_variant_counts": [len(n.get("all_variants", [])) for n in piece_nodes],
        "tag_corners": [n.get("tag_corner") for n in piece_nodes],
        "solution_count": int(LAST_SOLUTION_COUNT),
    }


def _gen_attempt():
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    nodes = _grow_tree()
    if nodes is None:
        return None
    assembled = _assemble(nodes, h, w)
    if assembled is None:
        return None
    if not _choose_scatter_variants(nodes):
        return None
    inp = _scatter(nodes, h, w)
    if inp is None:
        return None
    try:
        _ = main(inp)
    except Exception:
        return None
    _record_last_gen_info(nodes)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cbebaa4b D1-rotation/flip input")
