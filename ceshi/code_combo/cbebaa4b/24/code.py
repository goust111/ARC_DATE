# Combo variant of cbebaa4b (index 24)
# combo_spec: D2/0+D14/1
# Name: 规则矩形精灵 + 对称性目标

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis for the fused D2/0 + D14/1 variant
# a) Concrete behavior and trace:
#    The original task uses irregular bracket-like sprites, and any complete red-port assembly
#    is acceptable. Here every non-anchor sprite is simplified into one solid rectangular body
#    with a one-cell border ring, and each connector is a two-pixel red protrusion on that ring.
#    Example trace: in the original puzzle a scattered green piece might be an L-shape with two
#    red edge pixels; the solver only needs to discover where those red pixels pair. In the fused
#    puzzle the green piece is instead a clean rectangle whose red pair sticks out from one side,
#    so the local geometry is more predictable. However, when several mirrored rectangles could
#    all complete the red-pairing tree, the final placement is legal only if the finished figure
#    is vertically symmetric about the anchor center column. The behavioral change is therefore:
#    local placement remains "match red ports by translation", while global acceptance becomes
#    "match red ports and satisfy mirror symmetry".
# b) Complexity calibration:
#    Fully regularizing every piece into identical rectangles would make the puzzle too bland and
#    too ambiguous. This generator still varies rectangle sizes, connector sides, and connector
#    offsets, but keeps them within a small family: each body is a solid rectangle of size 2-5 by
#    2-6, and each connector is always a two-pixel side protrusion. Symmetry is fixed to one
#    learnable family, vertical reflection through the anchor center column, rather than mixing
#    multiple symmetry types across examples.
# c) Implicit-assumption audit against the original main():
#    - Component extraction still works because each scattered sprite remains one 4-connected
#      component.
#    - The anchor still stays fixed, but now its odd width is essential because the mirror axis is
#      derived from its center column.
#    - The original code implicitly tolerated arbitrary piece interiors. That assumption is false
#      for this combo: the solver now validates that every sprite has a solid rectangular body with
#      red protrusions only in the one-cell border ring.
#    - The original DFS accepted any complete pairing of red pixels. That assumption is false here,
#      so both terminal validation and intermediate pruning consult vertical symmetry.
#    - _try_place still allows equal-color overlap, but the generator guarantees assembled overlaps
#      happen only on red/red connector cells, so same-color non-red overlap does not become a
#      hidden loophole.
# d) Placement legality:
#    The generator is output-first. It grows one left-side tree of rectangle sprites in local
#    coordinates, mirrors it to the right, then stamps the full assembly into the grid. Candidate
#    left placements must stay strictly left of the axis, may overlap only the parent bbox, must
#    fit once mirrored, and must yield a single connected output with every red coordinate covered
#    exactly twice.
# e) Visual observability:
#    A human can see two changes at once in 2-3 examples: the scattered sprites are visibly clean
#    rectangles with small red tabs, and the solved output is visibly left-right symmetric around
#    the yellow anchor. No invisible numeric ordering is used.
# f) Visual cleanliness:
#    Each sprite uses exactly one body color plus red connectors, and the body occupies a solid
#    interior rectangle with a one-cell margin. That keeps components legible and avoids the noisy
#    path fillings of the original task. The mirrored assembly is restricted to one bilateral tree,
#    so the output stays tidy instead of producing a tangled mass.
# g) Color inference:
#    Colors are not remapped. Each mirrored pair reuses the same body color on both sides, so the
#    new inference burden is geometric: recognize rectangular source sprites and the symmetric
#    target, not hidden color arithmetic.


SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
MIRROR_SIDE = {"top": "top", "bottom": "bottom", "left": "right", "right": "left"}


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        reds = np.argwhere(arr == 2)
        body_colors = sorted(int(v) for v in np.unique(arr[(arr != bg) & (arr != 2)]))
        assert len(body_colors) == 1, "each sprite must have exactly one non-red body color"
        body_color = body_colors[0]
        info = _analyze_rect_sprite(arr, body_color, bg)
        pieces.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "reds": [(int(r), int(c)) for r, c in reds],
                "shape": (int(arr.shape[0]), int(arr.shape[1])),
                "area": int(np.sum(arr != bg)),
                "body_color": body_color,
                "connector_sides": tuple(sorted(info["sides"])),
                "port_offsets": dict(info["ports"]),
                "body_shape": tuple(info["body_shape"]),
            }
        )
    return pieces


def _analyze_rect_sprite(arr: np.ndarray, body_color: int, bg: int) -> dict:
    h, w = arr.shape
    assert h >= 2 and w >= 2, "rectangular sprites need a nontrivial body"

    body = arr == body_color
    coords = np.argwhere(body)
    assert coords.size > 0, "sprite must have a colored body"
    r0, c0, r1, c1 = bbox_coords(coords)
    assert np.all(arr[r0 : r1 + 1, c0 : c1 + 1] == body_color), "body must be a solid rectangle"

    reds = set((int(r), int(c)) for r, c in np.argwhere(arr == 2))
    seen = set()
    ports = {}
    sides = set()

    if r0 > 0:
        for k in range(c1 - c0):
            cells = {(r0 - 1, c0 + k), (r0 - 1, c0 + k + 1)}
            if cells <= reds:
                assert not (cells & seen), "top connector cells cannot overlap"
                seen |= cells
                sides.add("top")
                ports["top"] = int(k)
    if c1 + 1 < w:
        for k in range(r1 - r0):
            cells = {(r0 + k, c1 + 1), (r0 + k + 1, c1 + 1)}
            if cells <= reds:
                assert not (cells & seen), "right connector cells cannot overlap"
                seen |= cells
                sides.add("right")
                ports["right"] = int(k)
    if r1 + 1 < h:
        for k in range(c1 - c0):
            cells = {(r1 + 1, c0 + k), (r1 + 1, c0 + k + 1)}
            if cells <= reds:
                assert not (cells & seen), "bottom connector cells cannot overlap"
                seen |= cells
                sides.add("bottom")
                ports["bottom"] = int(k)
    if c0 > 0:
        for k in range(r1 - r0):
            cells = {(r0 + k, c0 - 1), (r0 + k + 1, c0 - 1)}
            if cells <= reds:
                assert not (cells & seen), "left connector cells cannot overlap"
                seen |= cells
                sides.add("left")
                ports["left"] = int(k)

    assert seen == reds, "red pixels must be exactly two-pixel side protrusions"
    mask = (arr != bg) & (arr != body_color) & (arr != 2)
    assert not np.any(mask), "sprite may only contain body color, red, and background"
    return {"sides": sides, "ports": ports, "body_shape": (r1 - r0 + 1, c1 - c0 + 1)}


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
    rows, cols = np.nonzero(canvas != bg)
    for r, c in zip(rows, cols):
        mc = axis2 - int(c)
        if mc < 0 or mc >= canvas.shape[1]:
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

    anchors = [p for p in pieces if p["body_color"] == 4]
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
    assert total_ports % 2 == 0, "total red connector ports must be even"
    target_unique_red = total_ports // 2

    red_counts: dict[tuple[int, int], int] = {}
    for rr, cc in anchor["reds"]:
        key = (ar0 + rr, ac0 + cc)
        red_counts[key] = red_counts.get(key, 0) + 1

    solutions: list[np.ndarray] = []
    seen_terminals: set[bytes] = set()

    def dfs(cur: np.ndarray, remaining: list[dict], rc: dict[tuple[int, int], int]) -> None:
        if len(solutions) >= max_solutions:
            return
        if len(rc) > target_unique_red:
            return

        if not remaining:
            if len(rc) != target_unique_red or any(v != 2 for v in rc.values()):
                return
            if require_symmetry and not _is_vertically_symmetric(cur, axis2, bg):
                return
            key = cur.tobytes()
            if key in seen_terminals:
                return
            seen_terminals.add(key)
            solutions.append(cur)
            return

        unpaired = [k for k, v in rc.items() if v == 1]
        if not unpaired:
            return
        rc1 = {k for k, v in rc.items() if v == 1}

        best_moves = None
        for tr, tc in unpaired:
            moves = []
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
    Assemble the scattered rectangular sprites around the fixed yellow anchor and keep only the mirror-symmetric completion.

    Input structure:
    - bg = black (0). The grid contains several separated sprites.
    - One sprite is the yellow anchor: a solid inset rectangle with red side tabs, and it stays fixed.
    - Every non-anchor sprite is a solid colored rectangle with a one-cell border ring; any red (2) pixels appear only as two-cell protruding tabs on that ring.

    Rule:
    - Translate the non-anchor sprites without rotation so red tabs overlap pairwise, exactly as connector matches in the original jigsaw task.
    - A placement is valid only if every red tab is paired exactly once and the finished assembly is vertically symmetric about the anchor center column.
    - Output that unique symmetric assembly.

    Key insight:
    - The red tabs still tell you how sprites can lock together locally, but the rectangle regularity makes each piece easier to parse.
    - Local connector matches are not sufficient on their own: among complete pairings, only the assembly whose left half mirrors its right half around the anchor is allowed.
    """
    sols = _collect_solutions(np.asarray(grid, dtype=int), require_symmetry=True, max_solutions=2)
    assert len(sols) == 1, "assembly must have exactly one symmetric valid solution"
    return sols[0]


# --------------- generator (output-first rectangular symmetric assembly) ---------------

GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
MAX_ASSEMBLY_DIM = 24

PAIR_COUNT_CHOICES = (2, 3)
BODY_H_RANGE = (2, 5)
BODY_W_RANGE = (2, 6)
PIECE_DIM_CANDIDATES = 16
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]
PIECE_CAP = 2

ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]
SCATTER_PADDING = 2

GEN_RETRY_LIMIT = 500
HALF_GROW_RETRY = 80


def _new_node(kind, body_h, body_w, color, top=0, left=0, cap=PIECE_CAP):
    return {
        "kind": kind,
        "body_shape": (int(body_h), int(body_w)),
        "shape": (int(body_h) + 2, int(body_w) + 2),
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
    body_h, body_w = node["body_shape"]
    span = body_h if side in ("left", "right") else body_w
    node["ports"][side] = random.randint(0, span - 2)


def _port_cells(node, side):
    h, w = node["shape"]
    body_h, body_w = node["body_shape"]
    k = node["ports"][side]
    if side == "top":
        return [(0, 1 + k), (0, 2 + k)]
    if side == "bottom":
        return [(h - 1, 1 + k), (h - 1, 2 + k)]
    if side == "left":
        return [(1 + k, 0), (2 + k, 0)]
    return [(1 + k, w - 1), (2 + k, w - 1)]


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


def _make_piece(node):
    h, w = node["shape"]
    body_h, body_w = node["body_shape"]
    arr = np.zeros((h, w), dtype=int)
    arr[1 : 1 + body_h, 1 : 1 + body_w] = node["color"]
    for side in node["sides"]:
        for r, c in _port_cells(node, side):
            arr[r, c] = 2
    if len(components_mask(arr != 0, connectivity=4)) != 1:
        return None
    if int(np.sum(arr == 2)) != 2 * len(node["sides"]):
        return None
    _analyze_rect_sprite(arr, node["color"], 0)
    return arr


def _make_anchor(node):
    arr = _make_piece(node)
    if arr is None:
        return None
    _analyze_rect_sprite(arr, 4, 0)
    return arr


def _mirror_left_coord(left: int, width: int, axis_col: int) -> int:
    return int(2 * axis_col - (left + width - 1))


def _mirror_port_offset(node, side):
    body_h, body_w = node["body_shape"]
    k = node["ports"][side]
    if side in ("top", "bottom"):
        return int(body_w - 2 - k)
    return int(k)


def _set_mirrored_ports(dst, src):
    for side, k in src["ports"].items():
        ms = MIRROR_SIDE[side]
        dst["ports"][ms] = _mirror_port_offset(src, side)


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
    body_h, body_w = random.choice(ANCHOR_BLOCK_CHOICES)
    anchor = _new_node("anchor", body_h, body_w, 4, top=0, left=0, cap=2)
    axis_col = anchor["shape"][1] // 2
    anchor["sides"] = {"left", "right"}
    k = random.randint(0, body_h - 2)
    anchor["ports"]["left"] = k
    anchor["ports"]["right"] = k

    pair_count = random.choice(PAIR_COUNT_CHOICES)
    colors = random.sample(PIECE_COLORS, pair_count)
    left_nodes = [anchor]

    for i in range(pair_count):
        placed = False
        dims = [
            (random.randint(*BODY_H_RANGE), random.randint(*BODY_W_RANGE))
            for _ in range(PIECE_DIM_CANDIDATES)
        ]
        random.shuffle(dims)
        for body_h, body_w in dims:
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
                child = _new_node("piece", body_h, body_w, colors[i], cap=PIECE_CAP)
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
        node["arr"] = _make_anchor(node) if node["kind"] == "anchor" else _make_piece(node)
        if node["arr"] is None:
            return None

    right_nodes = []
    for left in left_nodes[1:]:
        body_h, body_w = left["body_shape"]
        right = _new_node("piece", body_h, body_w, left["color"], top=left["top"], left=0, cap=left["cap"])
        right["left"] = _mirror_left_coord(left["left"], left["shape"][1], axis_col)
        right["sides"] = {MIRROR_SIDE[s] for s in left["sides"]}
        _set_mirrored_ports(right, left)
        right["arr"] = np.fliplr(left["arr"]).copy()
        _analyze_rect_sprite(right["arr"], right["color"], 0)
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
        sym_solutions = _collect_solutions(inp, require_symmetry=True, max_solutions=2)
        if len(sym_solutions) != 1:
            return None
        plain_solutions = _collect_solutions(inp, require_symmetry=False, max_solutions=3)
    except Exception:
        return None

    if not np.array_equal(sym_solutions[0], out):
        return None
    if len(plain_solutions) < 2:
        return None
    if not any(not np.array_equal(sol, out) for sol in plain_solutions):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rectangular symmetric cbebaa4b input")
