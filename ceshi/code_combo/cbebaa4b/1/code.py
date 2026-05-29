# Combo variant of cbebaa4b (index 1)
# combo_spec: D1/2+D7/2
# Name: 旋转匹配 + 链式传递组装

from __future__ import annotations

from pathlib import Path
import sys
import random

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(_ROOT / "solvers"))

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis
# ---------------
# concretization_and_trace:
# - Preserved scaffold: black background, one fixed anchor containing yellow (4), red (2) boundary
#   connectors, output-first generation, and red-on-red overlap as the only intended join.
# - D1/2 effect in pixels: a scattered movable sprite may appear rotated or mirrored relative to its
#   solved pose, so connector groups and bbox dimensions are no longer fixed by the input crop.
# - D7/2 effect in pixels: the solved topology is a simple relay chain
#   anchor -> piece1 -> piece2 -> ... -> pieceN. The anchor contributes one red pair, every internal
#   piece has two red pairs, and the terminal piece has one red pair.
# - Small trace:
#   original task: after attaching piece A to the anchor, the next step may use any still-unpaired red
#   on the whole growing object, and each piece is only translated.
#   fused variant: after attaching piece A, only A's other connector pair is the legal next target; if
#   piece B arrives rotated 90 degrees or mirrored, the solver must first pick the correct dihedral
#   variant of B and only then translate that variant so one of B's red pairs exactly covers A's tail.
#   The output uses the recovered orientation, not the scattered one.
#
# complexity_calibration:
# - Pure chaining already reduces branching, so to keep the D1 change visible every movable piece is
#   rendered with an asymmetric interior spur and is accepted only if it has several distinct dihedral
#   variants. This makes the orientation mismatch legible to humans.
# - Pure rotation/flip can explode the search space, so chaining limits the active attachment frontier to
#   exactly one 2-cell red group. This keeps search manageable while still requiring orientation inference.
# - Movable piece colors are sampled without repetition. That is not logically required by the rule, but it
#   prevents accidental same-color overlap ambiguities under `_try_place` and keeps the visual parse clean.
#
# implicit_assumption_audit_of_original_main:
# - `p["shape"]` and `p["reds"]` were treated as fixed. After D1/2 they depend on the chosen dihedral
#   transform, so the solver must enumerate variant records with transformed bbox, red cells, and red groups.
# - The original DFS expanded from "any count==1 red coordinate". After D7/2 that is too permissive: the
#   state must remember the unique active tail group, otherwise the search would silently reintroduce tree
#   branching.
# - Original move deduplication by `(piece, top, left)` is insufficient once orientation changes matter.
#   The fused solver deduplicates by `(piece, variant, top, left)`.
# - `_try_place(cur, p["arr"], ...)` originally assumed the extracted sprite already matched the solved pose.
#   That assumption is invalid here; placement must use the selected transformed array instead.
# - The original generator scattered each piece in its solved orientation. That would deactivate D1/2, so the
#   fused generator stores a separate `scatter_arr` and forces every movable piece into a non-identity variant.
#
# placement_legality:
# - The assembled reference layout is grown as a path in local coordinates. Each new node may overlap only its
#   predecessor bbox, never older non-parent nodes, and the union bbox must stay within `MAX_ASSEMBLY_DIM`.
# - When rendering the final assembly, only red-on-red overlaps are permitted and every red cell must end with
#   multiplicity exactly 2.
# - When scattering the input, the anchor stays at its solved position while all other sprites are placed with
#   Chebyshev padding >= 2 so component extraction is stable.
#
# visual_observability_and_tidiness:
# - The solved outputs visibly form one relay chain with exactly two degree-1 ends: the anchor side and the
#   terminal piece. There are no branch hubs, so the D7 change is easy to spot from examples.
# - Every movable piece is asymmetric and every scattered movable piece is shown in a non-identity orientation;
#   at least one piece uses a flip. This makes the D1 change visually active rather than latent.
# - No hidden ordering uses black=0 or color ids. The only cues are visible geometry: yellow anchor, red ports,
#   sprite shape, and chain topology.
#
# new_solver_assumptions_and_generator_guarantees:
# - The solver assumes the anchor has exactly one 2-cell red connector group and each movable piece has one or
#   two such groups. The generator guarantees that by growing a path and placing ports away from corners.
# - The solver assumes there is at most one open tail group after every non-terminal placement. The generator
#   guarantees this because every internal piece has degree 2 and every leaf has degree 1.
# - The solver assumes each movable piece has at least one non-identity dihedral variant, and that at least one
#   movable piece has a flip candidate. The generator rejects samples that do not satisfy this.
# - This fused rule set still leaves a small reversible symmetry in the local chain geometry, so the solver uses
#   a deterministic DFS ordering and returns the first valid assembled grid. The generator validates that this
#   canonical search succeeds and that translation-only chaining cannot solve the same input.


def _sorted_group(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def _find_red_groups(arr: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    groups = []
    for pts in components_mask(arr == 2, connectivity=4):
        coords = _sorted_group([(int(r), int(c)) for r, c in pts])
        groups.append(coords)
    groups.sort()
    return groups


def _piece_record(arr: np.ndarray, bg: int) -> dict:
    reds = np.argwhere(arr == 2)
    return {
        "arr": arr.copy(),
        "reds": [(int(r), int(c)) for r, c in reds],
        "red_groups": _find_red_groups(arr),
        "shape": (int(arr.shape[0]), int(arr.shape[1])),
        "area": int(np.sum(arr != bg)),
    }


def _variant_records(arr: np.ndarray, bg: int) -> list[dict]:
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


def _match_group_translations(
    local_group: tuple[tuple[int, int], ...], target_group: tuple[tuple[int, int], ...]
) -> list[tuple[int, int]]:
    target = set(target_group)
    out: set[tuple[int, int]] = set()
    for lr, lc in local_group:
        for tr, tc in target_group:
            top = int(tr - lr)
            left = int(tc - lc)
            shifted = {(int(r + top), int(c + left)) for r, c in local_group}
            if shifted == target:
                out.add((top, left))
    return sorted(out)


def _abs_group(group: tuple[tuple[int, int], ...], top: int, left: int) -> tuple[tuple[int, int], ...]:
    return _sorted_group([(r + top, c + left) for r, c in group])


def _solve_chain(grid: np.ndarray, allow_variants: bool) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must have at least one colored piece"

    anchors = [p for p in pieces if np.any(p["arr"] == 4)]
    assert len(anchors) == 1, "input must have exactly one yellow (4) anchor piece"
    anchor = anchors[0]
    rest = [p for p in pieces if p is not anchor]

    assert len(anchor["red_groups"]) == 1, "chain combo requires one anchor connector pair"
    for p in rest:
        assert len(p["red_groups"]) in (1, 2), "each movable piece must have one or two connector pairs"
        for grp in p["red_groups"]:
            assert len(grp) == 2, "each connector group must be exactly two red cells"
        p["variants"] = _variant_records(p["arr"], bg) if allow_variants else [_piece_record(p["arr"], bg)]

    canvas = np.full((h, w), bg, dtype=int)
    ar0, ac0, _, _ = anchor["bbox"]
    placed = _try_place(canvas, anchor["arr"], ar0, ac0, bg)
    assert placed is not None, "anchor piece must fit on canvas at its original position"

    total_ports = int(sum(len(p["reds"]) for p in pieces))
    assert total_ports % 2 == 0, "total red connector ports must be even"
    target_unique_red = total_ports // 2

    red_counts: dict[tuple[int, int], int] = {}
    for rr, cc in anchor["reds"]:
        key = (int(ar0 + rr), int(ac0 + cc))
        red_counts[key] = red_counts.get(key, 0) + 1

    active_tail = _abs_group(anchor["red_groups"][0], ar0, ac0)
    solutions: list[np.ndarray] = []

    def dfs(
        cur: np.ndarray,
        remaining: list[dict],
        rc: dict[tuple[int, int], int],
        tail: tuple[tuple[int, int], ...] | None,
    ) -> None:
        if len(solutions) >= 2:
            return

        if not remaining:
            if tail is None and len(rc) == target_unique_red and all(v == 2 for v in rc.values()):
                if not any(np.array_equal(cur, prev) for prev in solutions):
                    solutions.append(cur)
            return

        if tail is None or len(rc) > target_unique_red:
            return

        moves = []
        for idx, p in enumerate(remaining):
            for vidx, v in enumerate(p["variants"]):
                for red_group in v["red_groups"]:
                    for top, left in _match_group_translations(red_group, tail):
                        ph, pw = v["shape"]
                        if top < 0 or left < 0 or top + ph > h or left + pw > w:
                            continue
                        bad = False
                        for rr, cc in v["reds"]:
                            gr, gc = int(top + rr), int(left + cc)
                            if rc.get((gr, gc), 0) >= 2:
                                bad = True
                                break
                        if bad:
                            continue
                        moves.append((idx, vidx, top, left))

        dedup = sorted(
            set(moves),
            key=lambda x: (
                -len(remaining[x[0]]["variants"][x[1]]["red_groups"]),
                -remaining[x[0]]["area"],
                x[2],
                x[3],
                x[0],
                x[1],
            ),
        )
        for idx, vidx, top, left in dedup:
            p = remaining[idx]
            v = p["variants"][vidx]
            nxt = _try_place(cur, v["arr"], top, left, bg)
            if nxt is None:
                continue

            next_rc = dict(rc)
            for rr, cc in v["reds"]:
                gr, gc = int(top + rr), int(left + cc)
                next_rc[(gr, gc)] = next_rc.get((gr, gc), 0) + 1

            unpaired = _sorted_group([k for k, v2 in next_rc.items() if v2 == 1])
            next_remaining = remaining[:idx] + remaining[idx + 1 :]
            if next_remaining:
                if len(unpaired) != 2:
                    continue
                next_tail = unpaired
            else:
                if unpaired:
                    continue
                next_tail = None

            dfs(nxt, next_remaining, next_rc, next_tail)
            if len(solutions) >= 2:
                return

    dfs(placed, rest, red_counts, active_tail)
    return solutions


def _main_chain_no_reorient(grid: np.ndarray) -> np.ndarray:
    solutions = _solve_chain(grid, allow_variants=False)
    assert solutions, "translation-only chain assembly must be solvable"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered sprites into a relay chain by reorienting each piece before it passes the red connection onward.

    Input structure:
    - bg = black (0). One yellow-anchor sprite and several colored movable sprites are scattered apart.
    - The anchor already sits at its final location and contains the yellow (4) block plus one 2-cell red connector pair.
    - Every movable sprite has one or two 2-cell red connector pairs, and its scattered pose may be rotated or mirrored relative to the solved pose.

    Rule:
    - Keep the anchor fixed. The first movable sprite must cover the anchor's red pair.
    - After that, only the red pair left open by the most recently attached sprite may be used next, so assembly advances as a chain rather than branching anywhere on the structure.
    - Before placing any movable sprite, you may rotate it by 90/180/270 degrees or mirror it; choose the orientation that lets one of its red pairs match the current chain tail, and continue until every red cell is paired exactly once.

    Key insight: The state is two-dimensional: you must infer both which sprite comes next in the relay chain and which dihedral orientation makes that sprite's connector pair line up with the single active tail.
    """
    global LAST_SOLUTION_COUNT
    solutions = _solve_chain(grid, allow_variants=True)
    LAST_SOLUTION_COUNT = len(solutions)
    assert solutions, "combo chain assembly must have at least one valid solution"
    return solutions[0]


# --------------- generator (combo D1/2 + D7/2) ---------------

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
ANCHOR_CAP = 1

SCATTER_PADDING = 2
GEN_RETRY_LIMIT = 450

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
    return arr


def _make_anchor(node):
    bh, bw = node["block"]
    arr = np.zeros(node["shape"], dtype=int)
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
    anchor = _new_node("anchor", bh + 2, bw + 2, 4, cap=ANCHOR_CAP)
    anchor["block"] = (bh, bw)
    nodes = [anchor]
    n_pieces = random.randint(*PIECE_COUNT_RANGE)
    colors = random.sample(PIECE_COLORS, n_pieces)

    parent_i = 0
    for i in range(n_pieces):
        parent = nodes[parent_i]
        placed = False
        dims = [
            (random.randint(*PIECE_H_RANGE), random.randint(*PIECE_W_RANGE))
            for _ in range(PIECE_DIM_CANDIDATES)
        ]
        random.shuffle(dims)
        for ph, pw in dims:
            sides = [s for s in SIDES if s not in parent["sides"]]
            random.shuffle(sides)
            for side in sides:
                child = _new_node("piece", ph, pw, colors[i], cap=PIECE_CAP)
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
                parent_i = len(nodes) - 1
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

    if not any(node.get("scatter_is_flip") for node, _v, _n, _f in piece_infos):
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


def _record_last_gen_info(nodes, rotation_required: bool):
    global LAST_GEN_INFO
    piece_nodes = nodes[1:]
    degrees = [len(n["sides"]) for n in nodes]
    LAST_GEN_INFO = {
        "piece_count": len(piece_nodes),
        "changed_piece_count": int(
            sum(1 for n in piece_nodes if n.get("scatter_variant") not in (None, "id"))
        ),
        "flip_piece_count": int(sum(1 for n in piece_nodes if n.get("scatter_is_flip"))),
        "variant_codes": [n.get("scatter_variant") for n in piece_nodes],
        "unique_variant_counts": [len(n.get("all_variants", [])) for n in piece_nodes],
        "tag_corners": [n.get("tag_corner") for n in piece_nodes],
        "chain_degrees": degrees,
        "anchor_degree": degrees[0],
        "terminal_count": int(sum(1 for d in degrees if d == 1)),
        "max_degree": max(degrees),
        "all_non_identity": bool(
            piece_nodes and all(n.get("scatter_variant") not in (None, "id") for n in piece_nodes)
        ),
        "rotation_required": bool(rotation_required),
        "solution_count": int(LAST_SOLUTION_COUNT),
    }


def _gen_attempt():
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    nodes = _grow_chain()
    if nodes is None:
        return None
    out = _assemble(nodes, h, w)
    if out is None:
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

    rotation_required = False
    try:
        _ = _main_chain_no_reorient(inp)
    except Exception:
        rotation_required = True

    _record_last_gen_info(nodes, rotation_required=rotation_required)
    if not LAST_GEN_INFO["rotation_required"]:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cbebaa4b combo D1/2+D7/2 input")
