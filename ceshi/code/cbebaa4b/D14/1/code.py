# Variant of cbebaa4b
# Dimension: D14 — 组装后结构的目标形状 (meta)
# Change: 组装后形成不规则的连通结构，无特定目标形状约束... → 组装后必须具有对称性

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis for the D14 variant
# a) Concrete behavior and trace:
#    The original task accepts any complete red-port assembly around the fixed yellow anchor.
#    Here the completed figure must additionally be vertically symmetric about the anchor's
#    center column. In pixel terms, every nonzero cell at (r, c) in the output must have an
#    equal-color mate at (r, mirror(c)) with the anchor center as the mirror axis. The input
#    still shows the same scattered sprites, and the solver still assembles by translating
#    pieces so red connector pairs overlap. The changed step is the legality test for a
#    finished assembly: "all red ports are paired" is no longer sufficient; the resulting
#    union must also be mirror-symmetric.
#    Small trace: originally, if two remaining pieces can both complete the red pairing tree,
#    either completion would be accepted as long as conflicts do not occur. In the variant,
#    after the same red-pair search reaches a complete canvas, only the completion whose left
#    half mirrors its right half about the anchor survives.
# b) Complexity calibration:
#    Mixing point symmetry, vertical symmetry, and horizontal symmetry across examples would be
#    harder to infer from 2-4 ARC pairs. To keep the new rule visible and learnable, the
#    generator fixes one symmetry family: vertical reflection through the anchor center. The
#    puzzle remains nontrivial because the internal piece shapes are still irregular, ports are
#    still matched by backtracking, and only the target assembled arrangement is constrained.
# c) Implicit-assumption audit against the original main():
#    - _extract_pieces still works because the input is still scattered connected sprites.
#    - The anchor stays fixed; this is still required because the symmetry axis is derived from
#      the anchor bbox center column in input coordinates.
#    - total red ports must remain even; unchanged.
#    - The original dfs() assumed every complete red pairing is acceptable. That assumption is
#      false now, so the terminal condition must also check mirror symmetry.
#    - _try_place allows overlap for equal colors, not just red/red. In the original generator
#      this was harmless because pieces were arranged to overlap only at red ports. In the
#      variant, mirrored placement could accidentally create same-color non-red overlap if the
#      geometry were sloppy. The generator therefore still validates the final assembly with an
#      explicit conflict check and only keeps layouts where overlap is red/red only.
# d) Placement legality:
#    The generator builds an output-first assembly: the anchor has paired left/right ports, a
#    left-side branch is grown in local coordinates, and then that branch is mirrored to the
#    right. Candidate left bboxes are required to stay strictly on the left of the symmetry axis;
#    after mirroring, the full assembly is stamped and checked for bounds, conflicts, complete
#    red pairing, 4-connectivity, and symmetry. The scattered input uses the original padding
#    rule so the solver can re-extract each sprite.
# e) Visual observability:
#    The mirrored pairs reuse the same color on both sides, making the bilateral symmetry easy
#    to spot in the output. The pieces themselves are not simple solid rectangles; they keep the
#    bracket/bar/path style from the original task so the change is not a trivial recoloring.
# f) Visual cleanliness:
#    The output is constrained to one left branch plus its reflected right branch around the
#    central yellow anchor. This avoids messy multi-axis overlaps and keeps the mirrored halves
#    easy to parse by eye.
# g) Color inference:
#    No new color mapping rule is introduced. Colors only identify piece pairs and are mirrored
#    as-is, so the new concept to infer is geometric symmetry, not hidden color arithmetic.


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
    assert (c1 - c0 + 1) % 2 == 1, "anchor width must be odd for a cell-centered mirror axis"
    return int(c0 + c1)


def _partial_vertical_symmetry_ok(canvas: np.ndarray, axis2: int, bg: int) -> bool:
    h, w = canvas.shape
    rows, cols = np.nonzero(canvas != bg)
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
    grid: np.ndarray, *, require_symmetry: bool, max_solutions: int = 2
) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must have at least one colored piece"

    anchors = [p for p in pieces if np.any(p["arr"] == 4)]
    assert len(anchors) == 1, "input must have exactly one yellow (4) anchor piece"
    anchor = anchors[0]
    rest = [p for p in pieces if p is not anchor]

    axis2 = _anchor_axis2(anchor)

    canvas = np.full((h, w), bg, dtype=int)
    ar0, ac0, _, _ = anchor["bbox"]
    placed = _try_place(canvas, anchor["arr"], ar0, ac0, bg)
    assert placed is not None, "anchor piece must fit on canvas at its original position"
    if require_symmetry:
        assert _partial_vertical_symmetry_ok(placed, axis2, bg), "anchor must define the mirror axis"

    total_ports = int(sum(len(p["reds"]) for p in pieces))
    assert total_ports % 2 == 0, "total red connector ports must be even for complete pairing"
    target_unique_red = total_ports // 2

    red_counts: dict[tuple[int, int], int] = {}
    for rr, cc in anchor["reds"]:
        red_counts[(ar0 + rr, ac0 + cc)] = red_counts.get((ar0 + rr, ac0 + cc), 0) + 1

    solutions: list[np.ndarray] = []

    def dfs(cur: np.ndarray, remaining: list[dict], rc: dict[tuple[int, int], int]) -> None:
        if len(solutions) >= max_solutions:
            return

        if not remaining:
            if len(rc) != target_unique_red or any(v != 2 for v in rc.values()):
                return
            if require_symmetry and not _is_vertically_symmetric(cur, axis2, bg):
                return
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
            if require_symmetry and not _partial_vertical_symmetry_ok(nxt, axis2, bg):
                continue
            dfs(nxt, next_remaining, next_rc)
            if len(solutions) >= max_solutions:
                return

    dfs(placed, rest, red_counts)
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the scattered sprites around the fixed yellow anchor, but only keep the mirror-symmetric completion.

    Input structure:
    - bg = black (0). Several colored sprites are scattered apart from each other.
    - One anchor sprite contains the yellow (4) block and stays fixed in place.
    - Every sprite still carries red (2) connector pixels on its boundary.

    Rule:
    - Translate the non-anchor sprites so red connector pixels overlap pairwise, exactly as in the original jigsaw task.
    - A completed placement is valid only if every red connector is paired and the full assembled figure is vertically symmetric about the anchor center column.
    - The output is that unique symmetric assembly.

    Key insight:
    - Red pixels still define how pieces can interlock, but they no longer determine legality by themselves.
    - The anchor also defines a mirror axis, and only the assembly whose left half reflects to the right half about that axis is allowed.
    """
    sols = _collect_solutions(np.asarray(grid, dtype=int), require_symmetry=True, max_solutions=2)
    assert len(sols) == 1, "jigsaw assembly must have exactly one mirror-symmetric valid solution"
    return sols[0]


# --------------- generator (output-first symmetric assembly) ---------------

# Canvas
GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
MAX_ASSEMBLY_DIM = 24

# Pieces
PAIR_COUNT_CHOICES = (2, 3)  # total non-anchor piece count is 4 or 6
PIECE_H_RANGE = (4, 7)
PIECE_W_RANGE = (4, 8)
PIECE_DIM_CANDIDATES = 14
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]
PIECE_CAP = 2
BAR_SKIP_PROB = 0.35

# Anchor
ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]

# Scatter
SCATTER_PADDING = 2

# Generation
GEN_RETRY_LIMIT = 400
HALF_GROW_RETRY = 60

SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
MIRROR_SIDE = {"top": "top", "bottom": "bottom", "left": "right", "right": "left"}
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
    axes = [(r, er, c, True), (c, ec, er, False)] if order == 0 else [
        (c, ec, r, False),
        (r, er, ec, True),
    ]
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


def _mirror_left_coord(left: int, width: int, axis_col: int) -> int:
    return int(2 * axis_col - (left + width - 1))


def _mirror_port_offset(node, side):
    h, w = node["shape"]
    k = node["ports"][side]
    span = h if side in ("left", "right") else w
    return int(span - 2 - k)


def _set_mirrored_ports(dst, src):
    for side, k in src["ports"].items():
        ms = MIRROR_SIDE[side]
        if side in ("top", "bottom"):
            dst["ports"][ms] = _mirror_port_offset(src, side)
        else:
            dst["ports"][ms] = k


def _candidate_fits_half(nodes, parent_i, child, axis_col):
    cb = _bbox_of(child)
    if cb[3] >= axis_col:
        return False
    if not _rects_overlap(cb, _bbox_of(nodes[parent_i])):
        return False
    for i, other in enumerate(nodes):
        if i == parent_i:
            continue
        if _rects_overlap(cb, _bbox_of(other)):
            return False
    mirrored_left = _mirror_left_coord(child["left"], child["shape"][1], axis_col)
    if mirrored_left <= axis_col:
        return False
    trial_boxes = [_bbox_of(n) for n in nodes] + [cb]
    mr0, mc0, mr1, mc1 = cb[0], mirrored_left, cb[2], mirrored_left + child["shape"][1] - 1
    full = trial_boxes + [(mr0, mc0, mr1, mc1)]
    r0 = min(b[0] for b in full)
    c0 = min(b[1] for b in full)
    r1 = max(b[2] for b in full)
    c1 = max(b[3] for b in full)
    return (r1 - r0 + 1) <= MAX_ASSEMBLY_DIM and (c1 - c0 + 1) <= MAX_ASSEMBLY_DIM


def _grow_symmetric_skeleton():
    bh, bw = random.choice(ANCHOR_BLOCK_CHOICES)
    anchor = _new_node("anchor", bh + 2, bw + 2, 4, top=0, left=0, cap=2)
    anchor["block"] = (bh, bw)
    axis_col = anchor["shape"][1] // 2
    anchor["sides"] = {"left", "right"}
    k = random.randint(1, anchor["shape"][0] - 3)
    anchor["ports"]["left"] = k
    anchor["ports"]["right"] = k

    pair_count = random.choice(PAIR_COUNT_CHOICES)
    colors = random.sample(PIECE_COLORS, pair_count)
    left_nodes = [anchor]

    for i in range(pair_count):
        placed = False
        dims = [
            (random.randint(*PIECE_H_RANGE), random.randint(*PIECE_W_RANGE))
            for _ in range(PIECE_DIM_CANDIDATES)
        ]
        random.shuffle(dims)
        for ph, pw in dims:
            if i == 0:
                parent_candidates = [(0, "left")]
            else:
                parent_candidates = []
                for parent_i, parent in enumerate(left_nodes[1:], start=1):
                    outward = [s for s in ("left", "top", "bottom") if s not in parent["sides"]]
                    for side in outward:
                        parent_candidates.append((parent_i, side))
                random.shuffle(parent_candidates)

            for parent_i, side in parent_candidates:
                parent = left_nodes[parent_i]
                child = _new_node("piece", ph, pw, colors[i], cap=PIECE_CAP)
                incoming = OPPOSITE[side]
                _ensure_port(parent, side)
                _ensure_port(child, incoming)
                pr, pc = _global_port(parent, side)[0]
                cr, cc = _port_cells(child, incoming)[0]
                child["top"] = pr - cr
                child["left"] = pc - cc
                child["sides"].add(incoming)
                if not _candidate_fits_half(left_nodes, parent_i, child, axis_col):
                    continue
                parent["sides"].add(side)
                left_nodes.append(child)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    return left_nodes


def _materialize_nodes(left_nodes):
    anchor = left_nodes[0]
    axis_col = anchor["shape"][1] // 2

    for node in left_nodes:
        if node["kind"] == "anchor":
            node["arr"] = _make_anchor(node)
        else:
            node["arr"] = _make_piece(node)
        if node["arr"] is None:
            return None

    right_nodes = []
    for left in left_nodes[1:]:
        h, w = left["shape"]
        right = _new_node("piece", h, w, left["color"], top=left["top"], left=0, cap=left["cap"])
        right["left"] = _mirror_left_coord(left["left"], w, axis_col)
        right["sides"] = {MIRROR_SIDE[s] for s in left["sides"]}
        _set_mirrored_ports(right, left)
        right["arr"] = np.fliplr(left["arr"]).copy()
        right_nodes.append(right)

    return [anchor] + left_nodes[1:] + right_nodes


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
        for key, val in rc.items():
            all_red_counts[key] = all_red_counts.get(key, 0) + val

    total_ports = sum(int(np.sum(n["arr"] == 2)) for n in nodes)
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

    skeleton = None
    for _ in range(HALF_GROW_RETRY):
        skeleton = _grow_symmetric_skeleton()
        if skeleton is not None:
            break
    if skeleton is None:
        return None

    nodes = _materialize_nodes(skeleton)
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
    raise RuntimeError("failed to generate a valid symmetric cbebaa4b input")
