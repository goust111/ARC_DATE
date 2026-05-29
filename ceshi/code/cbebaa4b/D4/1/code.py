# Variant of cbebaa4b
# Dimension: D4 — 连接后的变换操作 (transform)
# Change: 纯平移连接，连接器像素保留红色，不做任何后处理变换... → 连接后擦除锚点方块

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis for the D4 variant
# ----------------------------------
# variant_spec:
#   core_change:
#     before: assemble every scattered sprite around the yellow anchor and keep the anchor in the output
#     after:  assemble in the same translation-only way, then repaint only the yellow anchor cells to bg=0
#   concrete_pixel_behavior:
#     input:
#       - one anchored yellow block remains fixed in place
#       - several colored sprites with red connector pairs are scattered elsewhere
#     output:
#       - sprites are translated into a unique full assembly exactly as before
#       - red overlap cells stay red because they also belong to sprite connectors
#       - the yellow block disappears, leaving a visible hole at the center
#   flow_trace:
#     original:
#       - start from the anchor piece on the canvas
#       - backtrack on red-overlap translations until all pieces are paired
#       - return the assembled canvas unchanged
#     variant:
#       - the same search still reconstructs the full assembly
#       - only after the unique solution is found, erase color 4 cells to bg
#       - this means the transform happens after assembly, not during matching
#   complexity_calibration:
#     - erasing the anchor would make the original tree-shaped outputs fall apart into spokes
#     - to keep the new behavior visible and logically coherent, the generator now builds a 5-piece cycle
#       around a leaf anchor: the sprites stay connected even after the yellow center is removed
#     - only one sprite touches the anchor; the rest of the cycle closes around it, so the hole is observable
#   implicit_assumption_review:
#     - _extract_pieces still assumes components are separated in the input: generator preserves scattering gaps
#     - anchor detection by "contains color 4" still works: exactly one piece contains yellow
#     - _try_place still relies on "only red may overlap": generator continues to stamp pieces with red-red overlaps only
#     - dfs still assumes every connector coordinate is paired exactly twice: generator validates this on the assembled grid
#     - new post-step assumption: erasing yellow must not disconnect the intended output; generator explicitly checks that
#   placement_legality:
#     - assembled positions are solved from connector equations, then stamped and checked for conflicts
#     - scattered input reuses the assembled anchor position and places all other sprites with Chebyshev padding >= 2
#     - the yellow hole is required to be empty in the final output and to have sprite pixels adjacent around it
#   visual_observability:
#     - the difference from the source task is now obvious in examples: input still shows a central yellow helper,
#       but output shows the same ring with a black hole where yellow used to be
#     - no invisible ordering signal is used; all decisions come from connector geometry
#   cleanliness:
#     - colors remain unique per sprite, overlaps are only red, and the erased zone is a clean solid rectangle
#   new_main_assumptions_and_generator_guarantees:
#     - main() now assumes only that at least one yellow anchor exists during reconstruction; it does not assume
#       anything about the post-erase connectivity
#     - generate_input() guarantees the stronger variant-specific condition that the sprite-only output remains
#       a single 4-connected component after yellow removal


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


def _erase_anchor_block(arr: np.ndarray, bg: int) -> np.ndarray:
    out = arr.copy()
    out[out == 4] = bg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered connector sprites around a yellow helper block, then remove that helper.

    Input structure:
    - bg = black (0). Several colored sprites are scattered on the grid.
    - Exactly one fixed anchor piece contains a yellow (4) rectangle plus red (2) connector pixels.
    - Every non-anchor sprite also carries red connector pixels on its boundary.

    Rule:
    - Keep the anchor at its input location and translate the remaining sprites so matching red pixels overlap.
    - Use every red connector exactly once, producing the unique complete assembly.
    - After assembly is complete, erase the yellow anchor block back to background while leaving the red overlaps intact.

    Key insight: The yellow block is only a temporary jig for the red-pixel matching process; solve the same connector puzzle first, then drop the yellow helper from the final image.
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
            k = (idx, top, left)
            dedup[k] = max(dedup.get(k, 0), ov)
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
    return _erase_anchor_block(solutions[0], bg)


# --------------- generator (o5x) ---------------

# Canvas
GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
MAX_ASSEMBLY_DIM = 24

# Pieces
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]
SCATTER_PADDING = 2
GEN_RETRY_LIMIT = 1200

# Ring layout
ANCHOR_BLOCK_CHOICES = [(3, 3)]
TOP_SHAPE_H_RANGE = (4, 6)
TOP_SHAPE_W_RANGE = (6, 8)
SIDE_SHAPE_H_RANGE = (8, 9)
SIDE_SHAPE_W_RANGE = (4, 6)
BOTTOM_SHAPE_H_RANGE = (4, 6)
BOTTOM_SHAPE_W_RANGE = (5, 8)
BAR_SKIP_PROB = 0.35

SIDES = ("top", "right", "bottom", "left")
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}


def _new_node(kind, h, w, color, top=0, left=0):
    return {
        "kind": kind,
        "shape": (int(h), int(w)),
        "color": int(color),
        "top": int(top),
        "left": int(left),
        "sides": set(),
        "ports": {},
        "trace_sides": set(),
    }


def _port_limit(node, side):
    h, w = node["shape"]
    span = h if side in ("left", "right") else w
    return span - 3


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

    bar_sides = set(node.get("trace_sides", set()))
    unused_sides = [s for s in SIDES if s not in used]
    if len(unused_sides) > 1 and random.random() < BAR_SKIP_PROB:
        unused_sides.remove(random.choice(unused_sides))
    bar_sides.update(unused_sides)
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
    for side in [s for s in bar_sides if s not in used]:
        _draw_path(arr, hub, _bar_endpoint((h, w), side, hub), color, 1 - order)

    if len(components_mask(arr != 0, connectivity=4)) != 1:
        return None
    if int(np.sum(arr == 2)) != 2 * len(used):
        return None
    return arr


def _make_anchor(node):
    bh, bw = node["block"]
    arr = np.zeros(node["shape"], dtype=int)
    arr[1 : 1 + bh, 1 : 1 + bw] = 4
    for side in node["sides"]:
        for r, c in _port_cells(node, side):
            arr[r, c] = 2
    return arr


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


def _has_visible_hole(erased, anchor):
    top, left = anchor["pos"]
    bh, bw = anchor["block"]
    r0, c0 = top + 1, left + 1
    r1, c1 = r0 + bh - 1, c0 + bw - 1
    if not np.all(erased[r0 : r1 + 1, c0 : c1 + 1] == 0):
        return False
    above = erased[r0 - 1, c0 : c1 + 1]
    below = erased[r1 + 1, c0 : c1 + 1]
    left_col = erased[r0 : r1 + 1, c0 - 1]
    right_col = erased[r0 : r1 + 1, c1 + 1]
    return (
        np.any(above != 0)
        and np.any(below != 0)
        and np.any(left_col != 0)
        and np.any(right_col != 0)
    )


def _build_ring_nodes():
    bh, bw = ANCHOR_BLOCK_CHOICES[0]
    anchor = _new_node("anchor", bh + 2, bw + 2, 4, top=0, left=0)
    anchor["block"] = (bh, bw)
    anchor["sides"] = {"top"}
    anchor["ports"]["top"] = random.randint(1, 2)

    colors = random.sample(PIECE_COLORS, 5)
    tm = _new_node("piece", 4, 5, colors[0], top=-3, left=0)
    tl = _new_node("piece", 8, 5, colors[1], top=-3, left=-4)
    bl = _new_node("piece", random.randint(5, 6), 7, colors[2], top=4, left=-3)
    br = _new_node("piece", random.randint(5, 6), 6, colors[3], top=4, left=3)
    tr = _new_node("piece", 8, 5, colors[4], top=-3, left=4)

    tm["sides"] = {"bottom", "left", "right"}
    tl["sides"] = {"right", "bottom"}
    bl["sides"] = {"top", "right"}
    br["sides"] = {"left", "top"}
    tr["sides"] = {"left", "bottom"}

    tm["trace_sides"] = {"bottom"}
    tl["trace_sides"] = {"right"}
    bl["trace_sides"] = {"top"}
    br["trace_sides"] = {"top"}
    tr["trace_sides"] = {"left"}

    tm["ports"] = {"bottom": anchor["ports"]["top"], "left": 1, "right": 1}
    tl["ports"] = {"right": 1, "bottom": 2}
    bl["ports"] = {"top": 1, "right": 2}
    br["ports"] = {"left": 2, "top": 2}
    tr["ports"] = {"left": 1, "bottom": 1}

    nodes = [anchor, tm, tl, bl, br, tr]
    r0, c0, r1, c1 = _union_bbox(nodes)
    if (r1 - r0 + 1) > MAX_ASSEMBLY_DIM or (c1 - c0 + 1) > MAX_ASSEMBLY_DIM:
        return None
    return nodes


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

    full = np.zeros((h, w), dtype=int)
    all_red_counts = {}
    for node in nodes:
        rc = _place_node_on_canvas(full, node, sr, sc)
        if rc is None:
            return None
        for k, v in rc.items():
            all_red_counts[k] = all_red_counts.get(k, 0) + v

    total_ports = sum(int(np.sum(n["arr"] == 2)) for n in nodes)
    if len(all_red_counts) * 2 != total_ports:
        return None
    if any(v != 2 for v in all_red_counts.values()):
        return None
    if len(components_mask(full != 0, connectivity=4)) != 1:
        return None

    erased = _erase_anchor_block(full, 0)
    if len(components_mask(erased != 0, connectivity=4)) != 1:
        return None
    if not _has_visible_hole(erased, nodes[0]):
        return None

    return full, erased


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
    nodes = _build_ring_nodes()
    if nodes is None:
        return None
    assembled = _assemble(nodes, h, w)
    if assembled is None:
        return None
    _full, out = assembled
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
    raise RuntimeError("failed to generate a valid cbebaa4b D4 input")


# Reuse the validated hollow-ring implementation used by D14/2.
# This transform variant has the same observable behavior: assemble by red-port
# matching, then erase the yellow anchor to leave a single one-hole ring.
_RING_IMPL = None


def _load_ring_impl():
    global _RING_IMPL
    if _RING_IMPL is None:
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "D14" / "2" / "code.py"
        spec = importlib.util.spec_from_file_location("cbebaa4b_d14_alt2_impl", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        _RING_IMPL = mod
    return _RING_IMPL


def main(grid: np.ndarray) -> np.ndarray:
    return _load_ring_impl().main(grid)


def generate_input() -> np.ndarray:
    return _load_ring_impl().generate_input()
