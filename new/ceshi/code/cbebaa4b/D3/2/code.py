# Variant of cbebaa4b
# Dimension: D3 — 目标锚点的结构 (dst)
# Change: 中央黄色方块作为初始锚点，已连接的精灵作为后续连接的扩展锚点... → 锚点为线段而非方块

from pathlib import Path
import sys
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis for the D3 variant
# variant_spec:
#   core_change:
#     old: the fixed yellow anchor was a compact 3x3 or 4x5 block near the center
#     new: the fixed yellow anchor is a 1-cell-thick straight segment, horizontal or vertical
#   concrete_pixel_form:
#     horizontal_anchor:
#       bbox: 3 x (line_len + 2)
#       yellow: one contiguous 1 x line_len row in the middle
#       reds: one 2-cell red port above the line and one 2-cell red port below the line
#     vertical_anchor:
#       bbox: (line_len + 2) x 3
#       yellow: one contiguous line_len x 1 column in the middle
#       reds: one 2-cell red port on the left and one 2-cell red port on the right
#     pieces:
#       unchanged_mechanism: each non-anchor sprite is still a single 4-connected colored component with red ports
#       allowed_motion: translation only; no rotation or reflection
#   behavior_trace:
#     original_flow:
#       - identify the unique yellow component
#       - keep that block fixed
#       - backtrack by overlapping red ports until all ports are paired
#     variant_flow:
#       - identify the unique yellow component whose yellow cells form a straight line, not a rectangle
#       - keep that line-anchor fixed
#       - attach two first-generation sprites on opposite sides of the line, then continue attaching later sprites to already placed sprites
#     visible_difference:
#       - old outputs look like growth around a compact center
#       - new outputs visibly split into two arms seeded from opposite sides of a long axis
#   complexity_calibration:
#     preserved:
#       - same grid size range
#       - same red-port pairing logic
#       - same scattered-input / assembled-output format
#     adjusted_for_visibility:
#       - anchor degree is fixed at 2 so both sides of the line are used
#       - piece count stays 4-6 so at least two later attachments extend beyond the line-anchor
#       - line length is 5-7 so the anchor cannot be mistaken for a square-ish blob
#   implicit_assumption_audit:
#     solver_lines_kept:
#       - _try_place still assumes only red/red overlaps are legal; this remains true
#       - dfs still assumes every red cell must be paired exactly once; this remains true
#     solver_lines_changed:
#       - anchor detection can no longer be "any yellow piece"; it now validates a straight yellow line with two opposite-side red ports
#       - docstring and assertions now describe a line anchor rather than a block anchor
#     generator_dependencies_on_old_shape:
#       - old _make_anchor painted a filled yellow rectangle; replaced with a centered line
#       - old tree builder allowed arbitrary anchor sides; replaced with orientation-specific opposite sides only
#       - old topology could radiate from the anchor; replaced with two chains that begin on the two sides of the line
#   placement_legality:
#     - every assembled child still overlaps only its parent, and only on a 2-pixel red port
#     - the line-anchor keeps a 1-cell margin inside its bbox so ports can sit outside the yellow line without leaving the component
#     - _candidate_fits still prevents non-parent bbox overlap and keeps the full assembly within MAX_ASSEMBLY_DIM
#   visual_observability:
#     - orientation is a visible branch: some samples have a horizontal yellow bar, others a vertical bar
#     - the two red port pairs lie on opposite sides of that bar, so humans can infer that sprites dock onto the bar's flanks
#     - later sprites extend the already attached branches, preserving the original recursive assembly idea
#   visual_cleanliness:
#     - only red cells may overlap in the output, so objects stay readable
#     - the anchor is thin and isolated, reducing central clutter compared with the block version
#   color_reasoning:
#     - no color remapping is introduced
#     - yellow still uniquely marks the fixed anchor, red still uniquely marks docking ports


SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}
ANCHOR_AXIS_SIDES = {"horizontal": ("top", "bottom"), "vertical": ("left", "right")}


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


def _line_anchor_info(arr: np.ndarray) -> dict | None:
    yellow = np.argwhere(arr == 4)
    if len(yellow) < 5:
        return None

    r0, c0, r1, c1 = bbox_coords(yellow)
    if (r0 == r1) == (c0 == c1):
        return None
    if np.any(arr[r0 : r1 + 1, c0 : c1 + 1] != 4):
        return None
    if int(np.sum(arr == 4)) != (r1 - r0 + 1) * (c1 - c0 + 1):
        return None

    reds = np.argwhere(arr == 2)
    if len(reds) != 4:
        return None

    if r0 == r1:
        rows = sorted({int(r) for r, _ in reds})
        if rows != [r0 - 1, r0 + 1]:
            return None
        if np.any((reds[:, 1] < c0) | (reds[:, 1] > c1)):
            return None
        for rr in rows:
            cols = sorted(int(c) for r, c in reds if int(r) == rr)
            if len(cols) != 2 or cols[1] != cols[0] + 1:
                return None
        return {"orientation": "horizontal", "yellow_bbox": (r0, c0, r1, c1)}

    cols = sorted({int(c) for _, c in reds})
    if cols != [c0 - 1, c0 + 1]:
        return None
    if np.any((reds[:, 0] < r0) | (reds[:, 0] > r1)):
        return None
    for cc in cols:
        rows = sorted(int(r) for r, c in reds if int(c) == cc)
        if len(rows) != 2 or rows[1] != rows[0] + 1:
            return None
    return {"orientation": "vertical", "yellow_bbox": (r0, c0, r1, c1)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered sprites onto a fixed yellow line-anchor by pairing red connector pixels.

    Input structure:
    - bg = black (0). Several colored sprites are scattered across the grid.
    - Exactly one sprite contains yellow (4) cells arranged as a straight horizontal or vertical segment.
    - The yellow line has one red (2) connector pair on each of its two broad sides; every other sprite also carries red boundary connectors.

    Rule:
    - The yellow line-anchor remains at its input position while the other sprites translate into place.
    - A sprite attaches when one of its red connector pairs overlaps unpaired red pixels on the current assembly.
    - Assembly finishes only when every original red pixel is matched exactly once, producing one connected structure.

    Key insight: The puzzle is still solved entirely through red-port pairing, but the fixed seed is now an axial line, so valid growth begins from the two flanks of that line instead of radiating from a yellow block.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must have at least one colored piece"

    anchors = []
    for p in pieces:
        info = _line_anchor_info(p["arr"])
        if info is not None:
            p["anchor_info"] = info
            anchors.append(p)
    assert len(anchors) == 1, "input must have exactly one yellow line-anchor piece"
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
        key = (ar0 + rr, ac0 + cc)
        red_counts[key] = red_counts.get(key, 0) + 1

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
                    if not bad:
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
    return solutions[0]


# --------------- generator ---------------

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
ANCHOR_ORIENTATIONS = ("horizontal", "vertical")
ANCHOR_LINE_LENGTHS = (5, 6, 7)
ANCHOR_CAP = 2

# Scatter
SCATTER_PADDING = 2

# Generation
GEN_RETRY_LIMIT = 300


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
        "entry_side": None,
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
    h, w = node["shape"]
    arr = np.zeros((h, w), dtype=int)
    line_len = node["line_len"]
    if node["orientation"] == "horizontal":
        arr[1, 1 : 1 + line_len] = 4
    else:
        arr[1 : 1 + line_len, 1] = 4
    for side in node["sides"]:
        for r, c in _port_cells(node, side):
            arr[r, c] = 2
    if len(components_mask(arr != 0, connectivity=4)) != 1:
        return None
    if _line_anchor_info(arr) is None:
        return None
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


def _parent_side_order(node):
    options = [s for s in SIDES if s not in node["sides"]]
    if not options:
        return []
    if node["kind"] == "anchor":
        preferred = [s for s in node["spawn_sides"] if s not in node["sides"]]
        return preferred
    if node["entry_side"] is not None:
        forward = OPPOSITE[node["entry_side"]]
        if forward in options:
            rest = [s for s in options if s != forward]
            random.shuffle(rest)
            return [forward] + rest
    random.shuffle(options)
    return options


def _attach_child(nodes, parent_i, color, required_side=None):
    dims = [
        (random.randint(*PIECE_H_RANGE), random.randint(*PIECE_W_RANGE))
        for _ in range(PIECE_DIM_CANDIDATES)
    ]
    random.shuffle(dims)
    for ph, pw in dims:
        parent = nodes[parent_i]
        side_order = [required_side] if required_side is not None else _parent_side_order(parent)
        for side in side_order:
            if side in parent["sides"]:
                continue
            child = _new_node("piece", ph, pw, color)
            incoming = OPPOSITE[side]
            _ensure_port(parent, side)
            _ensure_port(child, incoming)
            pr, pc = _global_port(parent, side)[0]
            cr, cc = _port_cells(child, incoming)[0]
            child["top"] = pr - cr
            child["left"] = pc - cc
            child["sides"].add(incoming)
            child["entry_side"] = incoming
            if not _candidate_fits(nodes, parent_i, child):
                continue
            parent["sides"].add(side)
            nodes.append(child)
            return len(nodes) - 1
    return None


def _grow_tree():
    orientation = random.choice(ANCHOR_ORIENTATIONS)
    line_len = random.choice(ANCHOR_LINE_LENGTHS)
    ah, aw = (3, line_len + 2) if orientation == "horizontal" else (line_len + 2, 3)
    anchor = _new_node("anchor", ah, aw, 4, cap=ANCHOR_CAP)
    anchor["orientation"] = orientation
    anchor["line_len"] = line_len
    anchor["spawn_sides"] = list(ANCHOR_AXIS_SIDES[orientation])
    nodes = [anchor]

    n_pieces = random.randint(*PIECE_COUNT_RANGE)
    colors = random.sample(PIECE_COLORS, n_pieces)
    branch_tails = []

    for side, color in zip(anchor["spawn_sides"], colors[:2]):
        idx = _attach_child(nodes, 0, color, required_side=side)
        if idx is None:
            return None
        branch_tails.append(idx)

    for i, color in enumerate(colors[2:]):
        if i < 2:
            branch_order = [i, 1 - i]
        else:
            branch_order = [0, 1]
            random.shuffle(branch_order)
        placed = False
        for branch in branch_order:
            idx = _attach_child(nodes, branch_tails[branch], color)
            if idx is not None:
                branch_tails[branch] = idx
                placed = True
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
    raise RuntimeError("failed to generate a valid cbebaa4b input")
