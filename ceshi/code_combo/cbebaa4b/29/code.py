# Combo variant of cbebaa4b (index 29)
# combo_spec: D4/0+D4/1+D6/0
# Name: 连接器重上色 + 擦除锚点 + 紧凑裁剪

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - The input still shows one fixed yellow anchor sprite plus several scattered
#      monochrome sprites with 2-cell red boundary connectors.
#    - Relative to the source task, the assembly search itself is unchanged: pieces
#      translate only, and legal docking still means red-over-red overlap with no
#      non-red conflicts.
#    - The difference is a three-step visible postprocessing chain applied to the
#      solved assembly:
#        1. D4/0 recolor: every completed red seam is recolored to the newly attached
#           sprite's body color, so the solved canvas contains no red.
#        2. D4/1 erase: after all seams are absorbed, the yellow anchor block is
#           erased back to black, leaving a hole.
#        3. D6/0 crop: finally, discard the outer black padding and return only the
#           minimal bbox of the remaining structure.
#    - Small trace:
#        original: red-red seam stays red, yellow block remains, full canvas returns.
#        combo:    red-red seam -> child color, yellow block -> black, then crop.
#      A top seam between the anchor and a blue piece that used to end as `2 2`
#      inside a 24x24 canvas now ends as `1 1`, the interior yellow 3x3 is removed,
#      and the returned output is only the tight ring-shaped crop.
# b) Complexity calibration
#    - Erasing the anchor would disconnect the original tree outputs, so the generator
#      must no longer use a free tree. It instead builds one fixed-topology ring where
#      the anchor is only a temporary jig and the colored sprites remain 4-connected
#      after yellow removal.
#    - Recoloring is single-mode and local: every seam takes the attached piece color.
#      This is visually strong but not combinatorially explosive.
#    - Cropping is made observable on every sample by enforcing a one-cell border
#      around the assembled structure before postprocessing.
# c) Hidden assumptions audit
#    - The original `_try_place(...)` assumed red seams could remain visible forever.
#      That no longer holds for the rendered canvas, so the solver separates:
#        logical state: `red_counts` tracks which connector coordinates are still
#        unpaired or already matched;
#        visible state: the canvas immediately recolors confirmed seam cells.
#    - Once seams are recolored, future placements must still target only currently
#      unpaired connector coordinates. The DFS already derives candidate moves from
#      `red_counts`, so recolored seam pixels cannot be reused.
#    - The new `main()` assumes each sprite is monochrome apart from red connectors,
#      because recoloring needs a unique body color per piece. The generator enforces
#      this and also keeps non-anchor colors distinct to avoid accidental same-color
#      overlap ambiguity.
#    - The new finalization step assumes that erasing yellow leaves a non-empty,
#      connected colored structure with an internal hole. The generator validates all
#      three properties before accepting a sample.
# d) Placement legality
#    - Scattered input placement is still done with `random_free_location(...)` and
#      Chebyshev padding >= 2, so disconnected components remain cleanly separated.
#    - Assembled placement is output-first and reserves `ASSEMBLY_BORDER` cells on all
#      four sides, guaranteeing in-bounds placement and visible cropping.
#    - During assembly stamping, overlaps are allowed only on red-red seam cells.
# e) Visual observability
#    - Human-visible cues are strong and repeated:
#        output has no red at all,
#        output has no yellow anchor,
#        output is smaller than input,
#        output contains a central black hole enclosed by colored pieces.
#    - These cues can be inferred from a few examples without any hidden ordering or
#      color-id arithmetic.
# f) Visual cleanliness
#    - The ring layout avoids clutter from disconnected spokes after anchor erasure.
#    - Each piece remains a single monochrome shape, seam recolors blend into adjacent
#      body pixels, and the cropped output removes distracting empty margins.
# g) Color-rule inferability
#    - Every non-anchor sprite uses exactly one body color, so when a seam turns from
#      red into that color, the correspondence is explicit from local structure alone.
#    - No many-to-many palette remapping is introduced; seam color is determined by the
#      visibly attached piece.
# h) New main() assumptions checked against generate_input()
#    - `main()` assumes exactly one anchor piece contains yellow: generator builds one.
#    - `main()` assumes at least one completed join exists for every placed non-anchor
#      piece considered by DFS: generator's ring topology guarantees it.
#    - `main()` assumes the final recolored canvas has zero visible red before erasing:
#      generator stamps seams with immediate recolor and rejects any assembled output
#      that still contains red.
#    - `main()` assumes tight cropping is meaningful: generator enforces the outer
#      border so the crop is strictly smaller than the input in both dimensions.


def _tight_crop_non_bg(g: np.ndarray, bg: int) -> np.ndarray:
    pts = np.argwhere(g != bg)
    if pts.size == 0:
        return g[:0, :0].copy()
    r0, c0, r1, c1 = bbox_coords(pts)
    return g[r0 : r1 + 1, c0 : c1 + 1].copy()


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        reds = np.argwhere(arr == 2)
        body_colors = sorted({int(v) for v in arr.ravel() if int(v) not in (bg, 2)})
        assert len(body_colors) == 1, "each sprite must have exactly one non-red body color"
        pieces.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "reds": [(int(r), int(c)) for r, c in reds],
                "shape": (int(arr.shape[0]), int(arr.shape[1])),
                "area": int(np.sum(arr != bg)),
                "body_color": int(body_colors[0]),
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


def _completed_join_coords(
    piece: dict, top: int, left: int, rc: dict[tuple[int, int], int]
) -> list[tuple[int, int]]:
    coords = []
    for rr, cc in piece["reds"]:
        pos = (top + rr, left + cc)
        if rc.get(pos, 0) == 1:
            coords.append(pos)
    return coords


def _recolor_join(canvas: np.ndarray, coords: list[tuple[int, int]], color: int) -> np.ndarray:
    if not coords:
        return canvas
    out = canvas.copy()
    for r, c in coords:
        out[int(r), int(c)] = int(color)
    return out


def _erase_anchor_block(arr: np.ndarray, bg: int) -> np.ndarray:
    out = arr.copy()
    out[out == 4] = bg
    return out


def _finalize_output(arr: np.ndarray, bg: int) -> np.ndarray:
    assert int(np.sum(arr == 2)) == 0, "all visible red seams must be recolored before finalization"
    erased = _erase_anchor_block(arr, bg)
    assert int(np.sum(erased == 4)) == 0, "yellow anchor must be erased in the final output"
    cropped = _tight_crop_non_bg(erased, bg)
    assert cropped.size > 0, "final cropped output must be non-empty"
    return cropped


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the scattered connector sprites, absorb every seam into the attached piece color, erase the yellow jig, and return the tight crop.

    Input structure:
    - Background is black (0). Several disconnected sprites are scattered on the grid.
    - Exactly one anchor sprite contains a yellow (4) rectangular core plus red (2) connector cells.
    - Every non-anchor sprite is monochrome apart from red boundary connector pairs.

    Rule:
    - Keep the anchor fixed and translate the remaining sprites so matching red connector pairs overlap without any non-red conflicts.
    - Whenever a piece docks onto the current assembly, recolor the completed red seam to that piece's body color.
    - After the unique full assembly is complete, erase the yellow anchor block to black and crop the result to the minimal bounding box of the remaining structure.

    Key insight:
    - Red pixels still determine the unique translation-only assembly, but they are only temporary guides: every solved seam is absorbed into a sprite color, the anchor is discarded after assembly, and only the compact ring-shaped footprint is returned.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must have at least one colored piece"

    anchors = [p for p in pieces if p["body_color"] == 4]
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
            key = (idx, top, left)
            dedup[key] = max(dedup.get(key, 0), ov)
        moves2 = [(idx, top, left, ov) for (idx, top, left), ov in dedup.items()]
        moves2.sort(key=lambda x: (-x[3], -remaining[x[0]]["area"], x[1], x[2], x[0]))

        for idx, top, left, _ov in moves2:
            p = remaining[idx]
            join_coords = _completed_join_coords(p, top, left, rc)
            if not join_coords:
                continue

            next_remaining = remaining[:idx] + remaining[idx + 1 :]
            next_rc = dict(rc)
            for rr, cc in p["reds"]:
                gr, gc = top + rr, left + cc
                next_rc[(gr, gc)] = next_rc.get((gr, gc), 0) + 1

            nxt = _try_place(cur, p["arr"], top, left, bg)
            if nxt is None:
                continue
            nxt = _recolor_join(nxt, join_coords, p["body_color"])
            dfs(nxt, next_remaining, next_rc)
            if len(solutions) >= 2:
                return

    dfs(placed, rest, red_counts)
    assert len(solutions) == 1, "jigsaw assembly must have exactly one valid solution"
    return _finalize_output(solutions[0], bg)


# --------------- generator (o5x) ---------------

# Canvas
GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
MAX_ASSEMBLY_DIM = 18

# Pieces
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]
SCATTER_PADDING = 2
GEN_RETRY_LIMIT = 1200
BAR_SKIP_PROB = 0.35

# Anchor + crop visibility
ANCHOR_BLOCK_CHOICES = [(3, 3)]
ASSEMBLY_BORDER = 1

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
    body_colors = {int(v) for v in arr.ravel() if int(v) not in (0, 2)}
    if body_colors != {int(color)}:
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

    overlap_red = (cur_vals == 2) & (vals == 2)
    if node["kind"] != "anchor" and np.any(overlap_red):
        out[g_rows[overlap_red], g_cols[overlap_red]] = int(node["color"])

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
    if (r1 - r0 + 1) > h - 2 * ASSEMBLY_BORDER or (c1 - c0 + 1) > w - 2 * ASSEMBLY_BORDER:
        return None
    sr_lo = ASSEMBLY_BORDER - r0
    sr_hi = h - ASSEMBLY_BORDER - 1 - r1
    sc_lo = ASSEMBLY_BORDER - c0
    sc_hi = w - ASSEMBLY_BORDER - 1 - c1
    if sr_lo > sr_hi or sc_lo > sc_hi:
        return None
    sr = random.randint(sr_lo, sr_hi)
    sc = random.randint(sc_lo, sc_hi)

    full = np.zeros((h, w), dtype=int)
    all_red_counts = {}
    for node in nodes:
        rc = _place_node_on_canvas(full, node, sr, sc)
        if rc is None:
            return None
        for key, value in rc.items():
            all_red_counts[key] = all_red_counts.get(key, 0) + value

    total_ports = sum(int(np.sum(n["arr"] == 2)) for n in nodes)
    if len(all_red_counts) * 2 != total_ports:
        return None
    if any(v != 2 for v in all_red_counts.values()):
        return None
    if len(components_mask(full != 0, connectivity=4)) != 1:
        return None
    if int(np.sum(full == 2)) != 0:
        return None

    erased = _erase_anchor_block(full, 0)
    if len(components_mask(erased != 0, connectivity=4)) != 1:
        return None
    if not _has_visible_hole(erased, nodes[0]):
        return None

    cropped = _tight_crop_non_bg(erased, 0)
    if cropped.shape[0] >= h or cropped.shape[1] >= w:
        return None
    interior = cropped[1:-1, 1:-1] if cropped.shape[0] > 2 and cropped.shape[1] > 2 else cropped[:0, :0]
    if interior.size == 0 or int(np.sum(interior == 0)) < 9:
        return None

    return full, cropped


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
    raise RuntimeError("failed to generate a valid cbebaa4b combo input")
