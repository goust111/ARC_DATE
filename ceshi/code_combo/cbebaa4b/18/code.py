# Combo variant of cbebaa4b (index 18)
# combo_spec: D3/3+D9/0
# Name: 十字形锚点 + 空心框锚点

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new/solvers")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis for the fused D3/3 + D9/0 variant
# a) Concrete behavior and trace:
#    - Original task: the unique yellow anchor is a convex solid block, so the
#      first pieces only need to dock onto ordinary rectangle edges.
#    - Fused task: the unique yellow anchor is either a thick cross or a thick L,
#      and that silhouette contains enclosed black holes. The shape is therefore
#      both non-convex and hollow.
#    - Pixel-level trace:
#      original: yellow square with one exposed red edge pair -> first child aligns
#      to that flat edge -> remaining children grow from the new exposed red pairs.
#      fused: hollow cross/L already exposes multiple branch-tip red pairs -> the
#      first children must land on those tip ends -> later growth still proceeds
#      only by red-on-red overlaps. The assembly engine is unchanged; only the
#      anchor geometry and its visible docking targets become more structured.
# b) Complexity calibration:
#    - The anchor family is limited to two readable categories (`cross`, `el`),
#      with fixed masks and fixed tip-side counts. This keeps recognition human-
#      tractable while making the visual change obvious in every sample.
#    - Cross anchors always use 3 direct tip ports; L anchors always use 2. That
#      forces multiple branch ends to matter without introducing too many degrees
#      of freedom.
#    - Hollowing is encoded by small enclosed cavities inside the yellow mask
#      instead of a free-form outline generator; this makes the "empty frame"
#      property stable and easy to detect.
# c) Hidden assumptions audited from the original main():
#    - The old assumption "anchor contains a solid yellow rectangle" is invalid.
#      `main()` now identifies the anchor by exact occupancy masks rather than by
#      the mere presence of color 4.
#    - The old anchor ports were derivable from rectangle sides. That breaks for
#      non-convex anchors, so anchor masks now carry explicit tip-side geometry
#      extracted from the mask boundary.
#    - The old anchor renderer filled a solid block. That would erase hollowness,
#      so `_make_anchor()` now paints from a fixed boolean mask.
# d) Placement legality:
#    - Output is still generated first, then scattered.
#    - Every node is rendered as a single 4-connected sprite. Non-red overlaps are
#      forbidden, and only red-on-red pairings may coincide.
#    - Candidate placements are bounded by the full bbox and later validated by an
#      exact canvas stamp. This is important because hollow anchors have empty
#      interior cells inside their bbox.
# e) Visual observability:
#    - The fused rule is visible directly from the input: the yellow anchor is not
#      a plain block, but a hollow thick cross or hollow thick L with tip reds.
#    - No invisible ordering by numeric color id, scan order, or hidden metadata
#      is used. A human can infer that pieces must dock to red branch ends.
# f) Visual cleanliness:
#    - Anchor masks are small, regular, and use only colors {0,2,4}. Their holes
#      are enclosed and remain legible in the scattered input.
#    - Piece geometry is unchanged from the original family, so only one visual
#      novelty is added: the anchor becomes both hollow and non-convex.
# g) Change inferability:
#    - No new color remapping is introduced. The novel inference burden is purely
#      geometric: recognize a hollow cross/L anchor and use its branch-tip reds as
#      the first docking sites.
# h) Fusion metric used after implementation:
#    - We track the anchor family distribution (`cross` vs `el`) and the number of
#      enclosed holes in the anchor mask. A valid sample must have a non-convex
#      anchor with at least one enclosed hole; otherwise the fusion has collapsed.
# i) New solver assumptions and generator guarantees:
#    - `main()` assumes the unique yellow component's occupancy exactly matches one
#      of the supported hollow cross/L masks. `generate_input()` enforces this.
#    - `main()` still assumes a unique red-pair assembly exists. The generator
#      re-runs `main()` on each candidate scattered input and accepts only exact
#      matches to the assembled target output.


def _cross_mask() -> np.ndarray:
    mask = np.zeros((6, 6), dtype=bool)
    mask[2:4, :] = True
    mask[:, 2:4] = True
    mask[2, 2] = False
    return mask


def _el_mask(turns: int = 0) -> np.ndarray:
    mask = np.zeros((6, 6), dtype=bool)
    mask[:, :2] = True
    mask[4:, :] = True
    mask[4, 1] = False
    return np.rot90(mask, k=turns)


def _extract_anchor_ports(mask: np.ndarray) -> dict[str, list[tuple[int, int]]]:
    h, w = mask.shape
    ports: dict[str, list[tuple[int, int]]] = {}

    top_cols = np.flatnonzero(mask[0, :])
    if len(top_cols) == 2 and top_cols[1] == top_cols[0] + 1:
        ports["top"] = [(0, int(top_cols[0])), (0, int(top_cols[1]))]

    bottom_cols = np.flatnonzero(mask[h - 1, :])
    if len(bottom_cols) == 2 and bottom_cols[1] == bottom_cols[0] + 1:
        ports["bottom"] = [(h - 1, int(bottom_cols[0])), (h - 1, int(bottom_cols[1]))]

    left_rows = np.flatnonzero(mask[:, 0])
    if len(left_rows) == 2 and left_rows[1] == left_rows[0] + 1:
        ports["left"] = [(int(left_rows[0]), 0), (int(left_rows[1]), 0)]

    right_rows = np.flatnonzero(mask[:, w - 1])
    if len(right_rows) == 2 and right_rows[1] == right_rows[0] + 1:
        ports["right"] = [(int(right_rows[0]), w - 1), (int(right_rows[1]), w - 1)]
    return ports


def _count_enclosed_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    inv = ~mask
    seen = np.zeros_like(inv, dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not inv[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            touches_border = False
            while stack:
                rr, cc = stack.pop()
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and inv[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _anchor_shape_catalog() -> list[dict]:
    out = [{"kind": "cross", "turns": 0, "mask": _cross_mask()}]
    for turns in range(4):
        out.append({"kind": "el", "turns": turns, "mask": _el_mask(turns)})
    for item in out:
        item["ports"] = _extract_anchor_ports(item["mask"])
        item["holes"] = _count_enclosed_holes(item["mask"])
    return out


ANCHOR_SHAPES = _anchor_shape_catalog()
ANCHOR_LOOKUP = {
    (item["mask"].shape, item["mask"].astype(np.uint8).tobytes()): item for item in ANCHOR_SHAPES
}


def _is_anchor_piece(arr: np.ndarray) -> bool:
    if not np.any(arr == 4):
        return False
    if not np.all(np.isin(arr, [0, 2, 4])):
        return False
    occ = arr != 0
    item = ANCHOR_LOOKUP.get((occ.shape, occ.astype(np.uint8).tobytes()))
    return item is not None and int(item["holes"]) >= 1


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered sprites by docking them onto a hollow yellow cross/L anchor through paired red ports.

    Input structure:
    - bg = black (0). Several colored sprites are scattered on the grid.
    - Exactly one sprite is a yellow (4) anchor whose occupied cells form a thick cross or thick L, and that yellow silhouette contains enclosed black hole cells.
    - Red (2) connector pixels lie on sprite boundaries; on the anchor they sit at branch tips rather than on a plain rectangular face.

    Rule:
    - The yellow hollow anchor stays fixed at its input position.
    - Every non-anchor sprite is translated without rotation so that its red pixels overlap exposed red pixels on the current assembly.
    - Only red pixels may overlap, and every red connector must be paired exactly once to produce one final connected structure.

    Key insight:
    - The reconstruction is still governed by red-pixel pairing, but the initial docking targets now come from the tip ends of a hollow non-convex anchor, so the solver must recognize anchor geometry by shape instead of by a solid yellow block.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must have at least one colored piece"

    anchors = [p for p in pieces if _is_anchor_piece(p["arr"])]
    assert len(anchors) == 1, "input must have exactly one hollow cross/L yellow anchor piece"
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
ANCHOR_DIRECT_DEGREE = {"cross": 3, "el": 2}

# Scatter
SCATTER_PADDING = 2

# Generation
GEN_RETRY_LIMIT = 400

# Geometry
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


def _new_anchor_node() -> dict:
    item = random.choice(ANCHOR_SHAPES)
    mask = item["mask"].copy()
    node = _new_node("anchor", mask.shape[0], mask.shape[1], 4, cap=ANCHOR_DIRECT_DEGREE[item["kind"]])
    node["anchor_kind"] = item["kind"]
    node["anchor_turns"] = int(item["turns"])
    node["mask"] = mask
    node["port_cells"] = {k: list(v) for k, v in item["ports"].items()}
    node["hole_count"] = int(item["holes"])
    if item["kind"] == "cross":
        node["direct_sides"] = set(random.sample(list(node["port_cells"]), ANCHOR_DIRECT_DEGREE[item["kind"]]))
    else:
        node["direct_sides"] = set(node["port_cells"])
    return node


def _ensure_port(node, side):
    if side in node.get("port_cells", {}):
        return
    if side in node["ports"]:
        return
    h, w = node["shape"]
    span = h if side in ("left", "right") else w
    node["ports"][side] = random.randint(1, span - 3)


def _port_cells(node, side):
    explicit = node.get("port_cells", {})
    if side in explicit:
        return explicit[side]
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
    mask = node["mask"]
    arr = np.zeros(mask.shape, dtype=int)
    arr[mask] = 4
    for side in node["sides"]:
        for r, c in node["port_cells"][side]:
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


def _parent_available_sides(node):
    if node["kind"] == "anchor":
        return [s for s in node["direct_sides"] if s not in node["sides"]]
    return [s for s in SIDES if s not in node["sides"]]


def _grow_tree():
    anchor = _new_anchor_node()
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
        need_anchor = len(anchor["sides"]) < len(anchor["direct_sides"])
        for ph, pw in dims:
            if need_anchor:
                parents = [0]
            else:
                parents = [j for j, n in enumerate(nodes) if len(n["sides"]) < n["cap"]]
                random.shuffle(parents)
            for parent_i in parents:
                parent = nodes[parent_i]
                sides = _parent_available_sides(parent)
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
    if int(nodes[0].get("hole_count", 0)) < 1:
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
    raise RuntimeError("failed to generate a valid cbebaa4b combo input")
