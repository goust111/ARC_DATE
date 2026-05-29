# Combo variant of cbebaa4b (index 9)
# combo_spec: D4/0+D6/0
# Name: 连接器重新上色 + 紧凑裁剪

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    - Source task: keep the yellow anchor fixed, translate the scattered sprites so
#      red connector cells overlap in pairs, and return the full input-sized assembly
#      with those paired seam cells still shown in red.
#    - D4/0 changes the visible state right after each successful join: the overlapped
#      red seam cells are no longer left red; they are recolored to the body color of
#      the newly attached non-anchor piece.
#    - D6/0 then changes the framing: after the whole structure is assembled and seams
#      have been absorbed, the solver discards all outer black margin and returns only
#      the minimal non-black bounding box.
#    - Concrete trace:
#        original: yellow anchor red-red + blue piece red-red -> seam stays red-red,
#                  final output is still a 24x24 canvas.
#        combo:    yellow anchor red-red + blue piece red-red -> seam becomes blue-blue,
#                  then after all pieces attach, the result is cropped to the exact
#                  footprint of the connected object, e.g. 14x13.
#    - The two changes interact: recoloring must happen before cropping, otherwise the
#      final cropped output could still expose red seam evidence from the source task.
#
# b) Complexity calibration
#    - The fusion should still be solved by red-port pairing, not by trivial cropping.
#      The generator therefore keeps the original tree-shaped translation-only assembly
#      and merely enforces that the completed object sits with at least a 1-cell black
#      border on every side of the input canvas.
#    - Recoloring uses a single visible rule: every completed seam adopts the child
#      piece's monochrome body color. There is no menu of recolor cases.
#    - Each sample shows repeated evidence for both changes: multiple joins eliminate
#      red, and every output is visibly smaller than the input canvas.
#
# c) Hidden-assumption audit of the original main()
#    - `_extract_pieces(...)` assumed each scattered sprite is a disconnected 4-neighbor
#      component. Scattering still guarantees that.
#    - Anchor detection by the unique yellow-containing component still holds because
#      this combo does not remove or recolor the anchor in the input.
#    - The original code effectively treated visible red pixels and logical unpaired
#      ports as the same thing. That breaks after D4/0: a completed seam is still part
#      of the logical pairing ledger, but it is no longer visibly red on the canvas.
#      The fix is to keep `red_counts` as the authoritative connector ledger and treat
#      the rendered canvas as purely visual state.
#    - The original conflict check allowed equal-color occupied overlaps. That is not a
#      safe invariant once seams can be recolored to ordinary body colors, so the fused
#      solver only allows red-on-red overlap. Every other occupied/occupied collision is
#      illegal.
#    - Cropping introduces a new postcondition: the assembled object must leave some
#      outer black margin in the full canvas, otherwise D6/0 would degenerate to the
#      original framing. The generator enforces this with a border constraint.
#
# d) Placement legality
#    - Generation remains output-first: build a valid assembled target first, record the
#      anchor's absolute position, then scatter the other pieces with Chebyshev padding
#      >= 2 so the input contains separated sprites.
#    - In the assembled target, only connector cells may overlap, and only as red-red
#      pairs before immediate recoloring. `_place_node_on_canvas(...)` enforces that.
#    - The union bbox must fit inside a reduced inner rectangle so that a 1-cell border
#      remains around the full assembly, making the final crop smaller in both axes.
#
# e) Visual observability
#    - A human looking at 2-4 samples can see that output seams are cleaner than input:
#      red ports exist on scattered pieces but disappear in the final assembly.
#    - The crop is also easy to spot because the output frame shrinks exactly to the
#      assembled footprint instead of keeping the anchor's original absolute padding.
#    - Neither change depends on hidden metadata such as numeric color order or object
#      indexing; both are visible pixel-level effects.
#
# f) Visual cleanliness
#    - The fused output has a single connected object with no extra red seam markers and
#      no surrounding empty border, so it is visually simpler than the source task.
#    - Because only one overlap type is permitted and scattering preserves wide gaps,
#      the input remains easy to parse into separate sprites.
#
# g) Change inferability
#    - The color rule is structurally supported: every non-anchor sprite is monochrome
#      except for red ports, so when a seam turns into that sprite's color, the local
#      reason is visible.
#    - The framing rule is also global and repeated: every output is a tight crop of the
#      assembled object, never a partly cropped or selectively padded view.
#
# h) Behavior metrics used after implementation
#    - Metric 1: `output_red_count` should always be 0 even though inputs contain red.
#    - Metric 2: `crop_margin = input_shape - output_shape` should be strictly positive
#      in both dimensions for every generated sample.
#    - Metric 3: compared against the original full-canvas assembly for the same input,
#      the combo output should differ both in shape and in red-pixel count.
#
# i) New assumptions introduced by the fused solver and how the generator guarantees them
#    - `main()` assumes each attached non-anchor piece has exactly one non-red body
#      color, so a completed seam can be recolored deterministically. The piece renderer
#      enforces this.
#    - `main()` assumes every accepted move overlaps at least one currently unpaired red
#      coordinate. `_completed_join_coords(...)` checks that, and DFS rejects moves that
#      do not complete a seam.
#    - `main()` assumes the final cropped output is smaller than the input canvas. The
#      generator enforces an assembly border and validates the resulting crop size.
#    - `main()` assumes the red-pairing assembly is unique. `_gen_attempt()` validates
#      this by accepting a sample only when `main(input)` exactly matches the prepared
#      combo output.


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

    # In the fused variant, visible seam cells may already have been recolored to an
    # ordinary body color, so only fresh red-on-red overlaps are legal.
    conflict = mask & (sub != bg) & ~((arr == 2) & (sub == 2))
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the scattered connector sprites around the fixed yellow anchor, absorb each completed seam into the attached piece color, then return only the tight crop.

    Input structure:
    - Background is black (0). Several disconnected sprites are scattered on the grid.
    - Exactly one sprite is the anchor: it contains a yellow (4) rectangular core and red boundary connectors.
    - Every non-anchor sprite is monochrome apart from red (2) connector pairs on its boundary.

    Rule:
    - Keep the anchor at its input coordinates and translate the other sprites, without rotation, so red connector pairs overlap exactly once.
    - A legal placement may overlap only currently unpaired red cells; all other occupied collisions are forbidden.
    - Whenever a piece docks, the overlapped red seam is immediately recolored to that piece's body color, and after all joins are complete the result is cropped to the minimal non-black bounding box.

    Key insight:
    - Red pixels are still the assembly clues, but they are only temporary docking marks: the logical pairing is tracked separately from the visible canvas, because every finished seam disappears into a body color before the final tight crop is taken.
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
        pos = (ar0 + rr, ac0 + cc)
        red_counts[pos] = red_counts.get(pos, 0) + 1

    solutions: list[np.ndarray] = []

    def dfs(cur: np.ndarray, remaining: list[dict], rc: dict[tuple[int, int], int]) -> None:
        if len(solutions) >= 2:
            return

        if not remaining:
            if len(rc) == target_unique_red and all(v == 2 for v in rc.values()):
                assert int(np.sum(cur == 2)) == 0, "all completed seams must be recolored"
                solutions.append(cur)
            return

        if len(rc) > target_unique_red:
            return

        unpaired = [k for k, v in rc.items() if v == 1]
        if not unpaired:
            return

        rc1 = {k for k, v in rc.items() if v == 1}

        best_moves = None
        for target in unpaired:
            moves = []
            tr, tc = target
            for idx, piece in enumerate(remaining):
                ph, pw = piece["shape"]
                for pr, pc in piece["reds"]:
                    top = tr - pr
                    left = tc - pc
                    if top < 0 or left < 0 or top + ph > h or left + pw > w:
                        continue
                    bad = False
                    overlap = 0
                    for rr, cc in piece["reds"]:
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
        for idx, top, left, overlap in best_moves:
            key = (idx, top, left)
            dedup[key] = max(dedup.get(key, 0), overlap)
        moves2 = [(idx, top, left, overlap) for (idx, top, left), overlap in dedup.items()]
        moves2.sort(key=lambda x: (-x[3], -remaining[x[0]]["area"], x[1], x[2], x[0]))

        for idx, top, left, _overlap in moves2:
            piece = remaining[idx]
            next_remaining = remaining[:idx] + remaining[idx + 1 :]
            join_coords = _completed_join_coords(piece, top, left, rc)
            if not join_coords:
                continue

            next_rc = dict(rc)
            for rr, cc in piece["reds"]:
                gr, gc = top + rr, left + cc
                next_rc[(gr, gc)] = next_rc.get((gr, gc), 0) + 1

            nxt = _try_place(cur, piece["arr"], top, left, bg)
            if nxt is None:
                continue
            nxt = _recolor_join(nxt, join_coords, piece["body_color"])
            dfs(nxt, next_remaining, next_rc)
            if len(solutions) >= 2:
                return

    dfs(placed, rest, red_counts)
    assert len(solutions) == 1, "jigsaw assembly must have exactly one valid solution"
    return _tight_crop_non_bg(solutions[0], bg)


# --------------- generator (o5x) ---------------

# Canvas
GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
MAX_ASSEMBLY_DIM = 20

# Pieces
PIECE_COUNT_RANGE = (4, 6)
PIECE_H_RANGE = (4, 7)
PIECE_W_RANGE = (4, 8)
PIECE_DIM_CANDIDATES = 12
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]
PIECE_CAP = 2
BAR_SKIP_PROB = 0.35

# Anchor
ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]
ANCHOR_CAP_CHOICES = [1, 2]

# Scatter / framing
SCATTER_PADDING = 2
ASSEMBLY_BORDER = 1

# Generation
GEN_RETRY_LIMIT = 300

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


def _grow_tree():
    bh, bw = random.choice(ANCHOR_BLOCK_CHOICES)
    anchor = _new_node("anchor", bh + 2, bw + 2, 4, cap=random.choice(ANCHOR_CAP_CHOICES))
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
            random.shuffle(parents)
            for parent_i in parents:
                parent = nodes[parent_i]
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

    overlap_red = (cur_vals == 2) & (vals == 2)
    if node["kind"] != "anchor" and np.any(overlap_red):
        out[g_rows[overlap_red], g_cols[overlap_red]] = int(node["color"])
    return red_counts


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
    if int(np.sum(out == 2)) != 0:
        return None
    cropped = _tight_crop_non_bg(out, 0)
    if cropped.shape[0] >= h or cropped.shape[1] >= w:
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
    out_cropped = _tight_crop_non_bg(out, 0)
    inp = _scatter(nodes, h, w)
    if inp is None:
        return None
    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(solved, out_cropped):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cbebaa4b input")
