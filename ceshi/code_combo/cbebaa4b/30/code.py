# Combo variant of cbebaa4b (index 30)
# combo_spec: D1/2+D11/0+D14/1
# Name: 旋转匹配 + 对称性目标 + 极少精灵

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis
# ----------------
# a) concretization_and_trace:
# - The anchor is still the only fixed sprite and still contains the yellow block. The difference is that
#   there are only 2 or 3 movable sprites, each much larger than before.
# - Two large side sprites form a mirrored pair in the final output. When a third sprite exists, it sits on
#   the vertical symmetry axis above or below the anchor and is itself vertically symmetric.
# - Each movable sprite may arrive scattered in a rotated or flipped pose. The solver therefore cannot use
#   the extracted bbox and red-port coordinates as final; it must enumerate dihedral variants first.
# - Small trace:
#   original task: a leaf piece whose red port is on its right edge is translated directly onto a red port
#   of the anchor.
#   fused variant: the same piece may appear rotated so its red port is on the top edge. The solver must
#   test that rotated or flipped candidate, place it by red overlap, and only keep completions whose whole
#   union is vertically symmetric about the anchor.
#
# b) complexity_calibration:
# - Reducing the piece count to 2-3 removes branch explosion, so the pieces are intentionally larger and
#   visually asymmetric. This keeps the rotation/flip rule observable instead of becoming a trivial move.
# - The symmetry family is fixed to vertical reflection through the anchor center column. Mixing symmetry
#   types across examples would be too hard to infer from a few ARC demonstrations.
# - Side pieces share one color inside each mirrored pair so the final bilateral structure is legible by eye.
#   The optional center piece uses a different color when present so the central axis is still visually clean.
#
# c) implicit_assumption_audit_of_original_main:
# - The original `shape` and `reds` fields were treated as immutable per piece. That fails once scattered
#   pieces can be rotated or flipped, so the solver now builds `variants` for every movable piece.
# - The original terminal condition accepted any full red pairing. That no longer holds because the finished
#   canvas must also be vertically symmetric.
# - The original search did not need an axis. Here the anchor width must stay odd so the mirror axis is a
#   cell-centered column derived from the anchor bbox.
# - The original generator could rely on many small pieces to make the task nontrivial. After D11, that
#   assumption fails, so the generator strengthens asymmetry of each large sprite and rejects generators that
#   are solvable without reorientation.
#
# d) placement_legality:
# - Generation is still output-first. The output skeleton is anchored first, then one large left piece is
#   rendered, mirrored to the right, and optionally paired with one centered top/bottom piece.
# - All placements are validated by the same red-overlap legality as the solver: only red-on-red overlap is
#   allowed, everything must stay in-bounds, all red counts must pair to exactly two, and the final output
#   must be one 4-connected component.
#
# e) visual_observability:
# - In the scattered input, the movable sprites are isolated with padding and shown in non-identity poses.
#   The output makes the new rule visible in two ways at once: the finished structure is obviously symmetric,
#   and the few large pieces clearly change orientation relative to the input.
#
# f) visual_tidiness:
# - The final assembly uses one left branch, its mirrored right branch, and optionally one centered branch.
#   This keeps layers shallow and avoids messy multi-object overlap near the anchor.
#
# g) inferability_of_changes:
# - There is no hidden color arithmetic. Color only marks which pixels belong to the same sprite family.
# - The core human-visible clues are structural: red ports define attachment, the finished figure is mirror
#   symmetric, and at least one large piece must be rotated or flipped to fit.
#
# new_solver_assumptions_and_generator_guarantees:
# - The anchor stays fixed and has odd width, so a unique vertical mirror axis exists.
# - Every movable sprite has at least one non-identity dihedral variant, and the generator scatters all of
#   them in non-identity poses.
# - The generator rejects any instance that is solvable under identity-only orientations, so D1 is active.
# - The generator only keeps inputs whose symmetric reassembly is unique under the full rotated/flipped search.


def _piece_record(arr: np.ndarray, bg: int) -> dict:
    reds = np.argwhere(arr == 2)
    return {
        "arr": arr.copy(),
        "reds": [(int(r), int(c)) for r, c in reds],
        "shape": (int(arr.shape[0]), int(arr.shape[1])),
        "area": int(np.sum(arr != bg)),
    }


def _variant_records(arr: np.ndarray, bg: int) -> list[dict]:
    variants = []
    seen = set()
    for rot_k in range(4):
        rot = np.rot90(arr, rot_k)
        for is_flip in (False, True):
            cur = np.fliplr(rot) if is_flip else rot
            key = (tuple(cur.shape), cur.tobytes())
            if key in seen:
                continue
            seen.add(key)
            rec = _piece_record(cur, bg)
            rec["rot_k"] = int(rot_k)
            rec["is_flip"] = bool(is_flip)
            rec["code"] = (
                "id"
                if (rot_k == 0 and not is_flip)
                else (f"fr{90 * rot_k}" if is_flip else f"r{90 * rot_k}")
            )
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


def _anchor_axis2(anchor: dict) -> int:
    _, c0, _, c1 = anchor["bbox"]
    assert (c1 - c0 + 1) % 2 == 1, "anchor width must be odd"
    return int(c0 + c1)


def _partial_vertical_symmetry_ok(canvas: np.ndarray, axis2: int, bg: int) -> bool:
    rows, cols = np.nonzero(canvas != bg)
    h, w = canvas.shape
    for r, c in zip(rows, cols):
        mc = axis2 - int(c)
        if mc < 0 or mc >= w:
            return False
        mv = int(canvas[int(r), mc])
        if mv != bg and mv != int(canvas[int(r), int(c)]):
            return False
    return True


def _is_vertically_symmetric(canvas: np.ndarray, axis2: int, bg: int) -> bool:
    h, w = canvas.shape
    for r in range(h):
        for c in range(w):
            mc = axis2 - c
            v = int(canvas[r, c])
            if mc < 0 or mc >= w:
                if v != bg:
                    return False
                continue
            if v != int(canvas[r, mc]):
                return False
    return True


def _collect_solutions(
    grid: np.ndarray,
    *,
    require_symmetry: bool,
    allow_reorientation: bool,
    max_solutions: int = 2,
) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must have at least one colored piece"

    anchors = [p for p in pieces if np.any(p["arr"] == 4)]
    assert len(anchors) == 1, "input must have exactly one yellow anchor piece"
    anchor = anchors[0]
    rest = [p for p in pieces if p is not anchor]
    for p in rest:
        if allow_reorientation:
            p["variants"] = _variant_records(p["arr"], bg)
        else:
            p["variants"] = [_piece_record(p["arr"], bg)]

    axis2 = _anchor_axis2(anchor)

    canvas = np.full((h, w), bg, dtype=int)
    ar0, ac0, _, _ = anchor["bbox"]
    placed = _try_place(canvas, anchor["arr"], ar0, ac0, bg)
    assert placed is not None, "anchor must fit in-place"
    if require_symmetry:
        assert _partial_vertical_symmetry_ok(placed, axis2, bg), "anchor must define mirror axis"

    total_ports = int(sum(len(p["reds"]) for p in pieces))
    assert total_ports % 2 == 0, "total red connector ports must be even"
    target_unique_red = total_ports // 2

    red_counts: dict[tuple[int, int], int] = {}
    for rr, cc in anchor["reds"]:
        key = (ar0 + rr, ac0 + cc)
        red_counts[key] = red_counts.get(key, 0) + 1

    solutions: list[np.ndarray] = []

    def dfs(cur: np.ndarray, remaining: list[dict], rc: dict[tuple[int, int], int]) -> None:
        if len(solutions) >= max_solutions:
            return

        if not remaining:
            if len(rc) != target_unique_red or any(v != 2 for v in rc.values()):
                return
            if require_symmetry and not _is_vertically_symmetric(cur, axis2, bg):
                return
            if not any(np.array_equal(cur, prev) for prev in solutions):
                solutions.append(cur.copy())
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

        assert best_moves is not None

        dedup: dict[tuple[int, int, int, int], int] = {}
        for idx, vidx, top, left, overlap in best_moves:
            key = (idx, vidx, top, left)
            dedup[key] = max(dedup.get(key, 0), overlap)
        moves2 = [
            (idx, vidx, top, left, overlap)
            for (idx, vidx, top, left), overlap in dedup.items()
        ]
        moves2.sort(key=lambda x: (-x[4], -remaining[x[0]]["area"], x[2], x[3], x[0], x[1]))

        for idx, vidx, top, left, _overlap in moves2:
            p = remaining[idx]
            v = p["variants"][vidx]
            nxt = _try_place(cur, v["arr"], top, left, bg)
            if nxt is None:
                continue
            if require_symmetry and not _partial_vertical_symmetry_ok(nxt, axis2, bg):
                continue

            next_rc = dict(rc)
            for rr, cc in v["reds"]:
                gr, gc = top + rr, left + cc
                next_rc[(gr, gc)] = next_rc.get((gr, gc), 0) + 1

            next_remaining = remaining[:idx] + remaining[idx + 1 :]
            dfs(nxt, next_remaining, next_rc)
            if len(solutions) >= max_solutions:
                return

    dfs(placed, rest, red_counts)
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble a tiny set of large scattered sprites around the fixed yellow anchor, allowing reorientation and enforcing mirror symmetry.

    Input structure:
    - The background is black (0), with one yellow-anchor sprite and only 2 or 3 larger movable sprites.
    - Two movable sprites belong to a left/right mirrored pair in the final picture; an optional third sprite lies on the vertical center line.
    - Every movable sprite has red (2) connector pixels, and the scattered pose may already be rotated or flipped.

    Rule:
    - Keep the anchor fixed, then attach the other sprites by overlapping red connectors as in the original jigsaw task.
    - Before placing a movable sprite, you may rotate it by 90/180/270 degrees and/or mirror it.
    - A completed placement is valid only if every red connector is paired exactly once and the whole assembled figure is vertically symmetric about the anchor center column.

    Key insight:
    - With only a few large pieces, brute-force orientation search is feasible, but red-port matching alone is not enough.
    - The anchor supplies a mirror axis, so the correct orientation choices are exactly the ones that allow a complete red-port assembly whose finished canvas reflects left-to-right.
    """
    sols = _collect_solutions(
        np.asarray(grid, dtype=int),
        require_symmetry=True,
        allow_reorientation=True,
        max_solutions=2,
    )
    assert len(sols) == 1, "assembly must have exactly one rotated/flipped mirror-symmetric solution"
    return sols[0]


# --------------- generator (output-first, few-piece symmetric assembly) ---------------

GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
MAX_ASSEMBLY_DIM = 22

PIECE_COUNT_CHOICES = (2, 3)
SIDE_PIECE_H_RANGE = (6, 9)
SIDE_PIECE_W_RANGE = (7, 10)
CENTER_PIECE_H_RANGE = (6, 9)
CENTER_PIECE_W_CHOICES = (5, 7, 9)
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]
BAR_SKIP_PROB = 0.25
MIN_SIDE_UNIQUE_VARIANTS = 4
MIN_CENTER_UNIQUE_VARIANTS = 3

ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]
SCATTER_PADDING = 2
GEN_RETRY_LIMIT = 500
RENDER_RETRY = 40

SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
MIRROR_SIDE = {"top": "top", "bottom": "bottom", "left": "right", "right": "left"}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}
TAG_CORNERS = {
    "nw": lambda h, w: (1, 1),
    "ne": lambda h, w: (1, w - 2),
    "sw": lambda h, w: (h - 2, 1),
    "se": lambda h, w: (h - 2, w - 2),
}

LAST_GEN_INFO: dict = {}


def _new_node(kind: str, h: int, w: int, color: int, top: int = 0, left: int = 0) -> dict:
    return {
        "kind": kind,
        "shape": (int(h), int(w)),
        "color": int(color),
        "top": int(top),
        "left": int(left),
        "sides": set(),
        "ports": {},
    }


def _ensure_port(node: dict, side: str) -> None:
    if side in node["ports"]:
        return
    h, w = node["shape"]
    span = h if side in ("left", "right") else w
    node["ports"][side] = random.randint(1, span - 3)


def _port_cells(node: dict, side: str) -> list[tuple[int, int]]:
    h, w = node["shape"]
    k = node["ports"][side]
    if side == "top":
        return [(0, k), (0, k + 1)]
    if side == "bottom":
        return [(h - 1, k), (h - 1, k + 1)]
    if side == "left":
        return [(k, 0), (k + 1, 0)]
    return [(k, w - 1), (k + 1, w - 1)]


def _global_port(node: dict, side: str) -> list[tuple[int, int]]:
    return [(node["top"] + r, node["left"] + c) for r, c in _port_cells(node, side)]


def _bbox_of(node: dict) -> tuple[int, int, int, int]:
    h, w = node["shape"]
    return (node["top"], node["left"], node["top"] + h - 1, node["left"] + w - 1)


def _union_bbox(nodes: list[dict]) -> tuple[int, int, int, int]:
    boxes = [_bbox_of(n) for n in nodes]
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _draw_path(arr: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: int, order: int) -> None:
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


def _paint_bar(arr: np.ndarray, side: str, color: int, used: set[str]) -> None:
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


def _bar_endpoint(shape: tuple[int, int], side: str, hub: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    hr, hc = hub
    if side in ("top", "bottom"):
        row = 0 if side == "top" else h - 1
        return (row, min(max(hc, 1), w - 2))
    col = 0 if side == "left" else w - 1
    return (min(max(hr, 1), h - 2), col)


def _make_side_piece(node: dict) -> np.ndarray | None:
    h, w = node["shape"]
    color = node["color"]
    used = set(node["sides"])

    for _ in range(RENDER_RETRY):
        arr = np.zeros((h, w), dtype=int)
        core_h = 2 if h <= 6 else 3
        core_w = 2 if w <= 7 else 3
        core_r0 = random.randint(1, h - core_h - 1)
        core_c0 = random.randint(1, w - core_w - 1)
        arr[core_r0 : core_r0 + core_h, core_c0 : core_c0 + core_w] = color
        hub = (core_r0 + core_h // 2, core_c0 + core_w // 2)

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

        order = random.randint(0, 1)
        for t in terminals:
            _draw_path(arr, t, hub, color, order)
        for side in bar_sides:
            _draw_path(arr, hub, _bar_endpoint((h, w), side, hub), color, 1 - order)

        trial_ok = False
        for name, loc_fn in random.sample(list(TAG_CORNERS.items()), len(TAG_CORNERS)):
            trial = arr.copy()
            tag = loc_fn(h, w)
            trial[tag] = color
            _draw_path(trial, hub, tag, color, random.randint(0, 1))
            if len(components_mask(trial != 0, connectivity=4)) != 1:
                continue
            if int(np.sum(trial == 2)) != 2 * len(used):
                continue
            if len(_variant_records(trial, 0)) < MIN_SIDE_UNIQUE_VARIANTS:
                continue
            node["tag_corner"] = name
            arr = trial
            trial_ok = True
            break

        if not trial_ok:
            continue
        return arr
    return None


def _make_center_piece(node: dict) -> np.ndarray | None:
    h, w = node["shape"]
    color = node["color"]
    axis = w // 2
    side = next(iter(node["sides"]))
    assert w % 2 == 1, "center piece must have odd width"

    for _ in range(RENDER_RETRY):
        arr = np.zeros((h, w), dtype=int)

        if side == "bottom":
            hub_r = random.randint(max(2, h // 2), h - 3)
            accent_row = random.randint(1, hub_r - 1)
            tail_end = random.randint(hub_r + 1, h - 2)
        else:
            hub_r = random.randint(2, min(h - 3, h // 2))
            accent_row = random.randint(hub_r + 1, h - 2)
            tail_end = random.randint(1, hub_r - 1)

        hub = (hub_r, axis)
        arr[hub] = color

        r0, r1 = sorted((hub_r, accent_row))
        arr[r0 : r1 + 1, axis] = color

        span = random.randint(1, max(1, axis - 1))
        left = axis - span
        right = axis + span
        arr[accent_row, left : right + 1] = color

        if left >= 1 and right <= w - 2:
            outer_row = accent_row + (1 if accent_row < hub_r else -1)
            if 0 < outer_row < h - 1:
                arr[outer_row, left] = color
                arr[outer_row, right] = color
                _draw_path(arr, hub, (outer_row, left), color, 0)
                _draw_path(arr, hub, (outer_row, right), color, 1)

        tr0, tr1 = sorted((hub_r, tail_end))
        arr[tr0 : tr1 + 1, axis] = color

        rail_len = random.randint(1, max(1, abs(hub_r - accent_row)))
        if axis - 1 >= 0 and axis + 1 < w:
            if accent_row < hub_r:
                rr0 = max(accent_row, hub_r - rail_len)
                rr1 = hub_r
            else:
                rr0 = hub_r
                rr1 = min(accent_row, hub_r + rail_len)
            arr[rr0 : rr1 + 1, axis - 1] = color
            arr[rr0 : rr1 + 1, axis + 1] = color

        dr, dc = INWARD[side]
        terminals = []
        for r, c in _port_cells(node, side):
            arr[r, c] = 2
            rr, cc = r + dr, c + dc
            arr[rr, cc] = color
            terminals.append((rr, cc))
        for t in terminals:
            _draw_path(arr, t, hub, color, random.randint(0, 1))

        if len(components_mask(arr != 0, connectivity=4)) != 1:
            continue
        if int(np.sum(arr == 2)) != 2:
            continue
        if not _is_vertically_symmetric(arr, 2 * axis, 0):
            continue
        if len(_variant_records(arr, 0)) < MIN_CENTER_UNIQUE_VARIANTS:
            continue
        return arr
    return None


def _make_anchor(node: dict) -> np.ndarray:
    bh, bw = node["block"]
    h, w = node["shape"]
    arr = np.zeros((h, w), dtype=int)
    arr[1 : 1 + bh, 1 : 1 + bw] = 4
    for side in node["sides"]:
        for r, c in _port_cells(node, side):
            arr[r, c] = 2
    return arr


def _mirror_left_coord(left: int, width: int, axis_col: int) -> int:
    return int(2 * axis_col - (left + width - 1))


def _build_layout() -> list[dict] | None:
    piece_count = random.choice(PIECE_COUNT_CHOICES)
    bh, bw = random.choice(ANCHOR_BLOCK_CHOICES)
    anchor = _new_node("anchor", bh + 2, bw + 2, 4, top=0, left=0)
    anchor["block"] = (bh, bw)
    axis_col = anchor["shape"][1] // 2

    side_color = random.choice(PIECE_COLORS)
    center_color_choices = [c for c in PIECE_COLORS if c != side_color]
    center_color = random.choice(center_color_choices) if center_color_choices else side_color

    k_lr = random.randint(1, anchor["shape"][0] - 3)
    anchor["sides"] = {"left", "right"}
    anchor["ports"]["left"] = k_lr
    anchor["ports"]["right"] = k_lr

    side_h = random.randint(*SIDE_PIECE_H_RANGE)
    side_w = random.randint(*SIDE_PIECE_W_RANGE)
    left = _new_node("side", side_h, side_w, side_color)
    left["sides"] = {"right"}
    _ensure_port(left, "right")
    pr, pc = _global_port(anchor, "left")[0]
    cr, cc = _port_cells(left, "right")[0]
    left["top"] = pr - cr
    left["left"] = pc - cc

    nodes = [anchor, left]

    if piece_count == 3:
        attach_side = random.choice(("top", "bottom"))
        piece_side = OPPOSITE[attach_side]
        center_h = random.randint(*CENTER_PIECE_H_RANGE)
        center_w = random.choice(CENTER_PIECE_W_CHOICES)
        center = _new_node("center", center_h, center_w, center_color)
        center["sides"] = {piece_side}
        anchor["sides"].add(attach_side)
        anchor["ports"][attach_side] = anchor["shape"][1] // 2 - 1
        center["ports"][piece_side] = center_w // 2 - 1
        pr2, pc2 = _global_port(anchor, attach_side)[0]
        cr2, cc2 = _port_cells(center, piece_side)[0]
        center["top"] = pr2 - cr2
        center["left"] = pc2 - cc2
        nodes.append(center)

    for node in nodes:
        if node["kind"] == "anchor":
            node["arr"] = _make_anchor(node)
        elif node["kind"] == "side":
            node["arr"] = _make_side_piece(node)
        else:
            node["arr"] = _make_center_piece(node)
        if node["arr"] is None:
            return None

    right = _new_node("side", side_h, side_w, side_color, top=left["top"], left=0)
    right["sides"] = {MIRROR_SIDE[s] for s in left["sides"]}
    right["ports"]["left"] = left["ports"]["right"]
    right["left"] = _mirror_left_coord(left["left"], side_w, axis_col)
    right["arr"] = np.fliplr(left["arr"]).copy()
    nodes.append(right)

    if piece_count == 2:
        return [anchor, left, right]
    return [anchor, left, nodes[2], right]


def _place_node_on_canvas(out: np.ndarray, node: dict, sr: int, sc: int) -> dict | None:
    arr = node["arr"]
    top = node["top"] + sr
    left = node["left"] + sc
    node["pos"] = (top, left)
    red_counts: dict[tuple[int, int], int] = {}

    mask = arr != 0
    rows, cols = np.nonzero(mask)
    g_rows = rows + top
    g_cols = cols + left
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


def _assemble(nodes: list[dict], h: int, w: int) -> np.ndarray | None:
    r0, c0, r1, c1 = _union_bbox(nodes)
    if (r1 - r0 + 1) > MAX_ASSEMBLY_DIM or (c1 - c0 + 1) > MAX_ASSEMBLY_DIM:
        return None
    if (r1 - r0 + 1) > h or (c1 - c0 + 1) > w:
        return None

    sr = random.randint(-r0, h - 1 - r1)
    sc = random.randint(-c0, w - 1 - c1)

    out = np.zeros((h, w), dtype=int)
    all_red_counts: dict[tuple[int, int], int] = {}
    for node in nodes:
        rc = _place_node_on_canvas(out, node, sr, sc)
        if rc is None:
            return None
        for key, val in rc.items():
            all_red_counts[key] = all_red_counts.get(key, 0) + val

    total_ports = sum(int(np.sum(node["arr"] == 2)) for node in nodes)
    if len(all_red_counts) * 2 != total_ports:
        return None
    if any(v != 2 for v in all_red_counts.values()):
        return None
    if len(components_mask(out != 0, connectivity=4)) != 1:
        return None

    anchor = nodes[0]
    axis2 = 2 * (anchor["pos"][1] + anchor["shape"][1] // 2)
    if not _is_vertically_symmetric(out, axis2, 0):
        return None
    return out


def _choose_scatter_variants(nodes: list[dict]) -> bool:
    movable = [n for n in nodes if n["kind"] != "anchor"]
    flip_candidates = []
    for node in movable:
        variants = _variant_records(node["arr"], 0)
        non_identity = [v for v in variants if not np.array_equal(v["arr"], node["arr"])]
        flips = [v for v in non_identity if v["is_flip"]]
        if not non_identity:
            return False
        node["all_variants"] = variants
        node["scatter_info"] = random.choice(non_identity)
        node["scatter_arr"] = node["scatter_info"]["arr"]
        if flips:
            flip_candidates.append((node, flips))

    if not any(node["scatter_info"]["is_flip"] for node in movable):
        if not flip_candidates:
            return False
        node, flips = random.choice(flip_candidates)
        node["scatter_info"] = random.choice(flips)
        node["scatter_arr"] = node["scatter_info"]["arr"]
    return True


def _scatter(nodes: list[dict], h: int, w: int) -> np.ndarray | None:
    inp = np.zeros((h, w), dtype=int)
    anchor = nodes[0]
    paste(inp, anchor["arr"], *anchor["pos"], transparent=0)

    rest = [n for n in nodes if n["kind"] != "anchor"]
    random.shuffle(rest)
    for node in rest:
        try:
            r, c = random_free_location(
                inp,
                node["scatter_arr"],
                transparent=0,
                padding=SCATTER_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inp, node["scatter_arr"], r, c, transparent=0)
    return inp


def _record_last_info(nodes: list[dict], inp: np.ndarray) -> None:
    global LAST_GEN_INFO
    movable = [n for n in nodes if n["kind"] != "anchor"]
    identity_sym = _collect_solutions(inp, require_symmetry=True, allow_reorientation=False, max_solutions=2)
    all_oriented = _collect_solutions(inp, require_symmetry=True, allow_reorientation=True, max_solutions=2)
    no_sym = _collect_solutions(inp, require_symmetry=False, allow_reorientation=True, max_solutions=4)
    axis2 = 2 * (nodes[0]["pos"][1] + nodes[0]["shape"][1] // 2)
    nonsym_count = sum(1 for arr in no_sym if not _is_vertically_symmetric(arr, axis2, 0))
    LAST_GEN_INFO = {
        "piece_count": len(movable),
        "variant_codes": [n["scatter_info"]["code"] for n in movable],
        "flip_count": int(sum(1 for n in movable if n["scatter_info"]["is_flip"])),
        "changed_piece_count": int(sum(1 for n in movable if n["scatter_info"]["code"] != "id")),
        "unique_variant_counts": [len(n["all_variants"]) for n in movable],
        "identity_sym_solution_count": len(identity_sym),
        "full_sym_solution_count": len(all_oriented),
        "nonsymmetric_valid_completion_count": int(nonsym_count),
    }


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    nodes = _build_layout()
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
        identity_sym = _collect_solutions(
            inp, require_symmetry=True, allow_reorientation=False, max_solutions=2
        )
    except Exception:
        return None
    if identity_sym:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    try:
        _record_last_info(nodes, inp)
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cbebaa4b combo input")
