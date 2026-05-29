# Combo variant of cbebaa4b (index 14)
# combo_spec: D2/1+D9/0
# Name: 功能像素精灵 + 空心框锚点

"""
Design analysis for the combo variant.

combo_spec:
  preserved_core:
    - The task is still an output-first jigsaw: one yellow anchor stays fixed,
      4-6 non-anchor sprites are scattered in the input, and the output is formed
      by translating sprites only.
    - Red pixels are still the only legal overlap cells in the final assembly.
    - The completed output remains a single 4-connected structure.
  a_concretization_and_behavior_trace:
    - Visual form:
      the anchor is no longer a solid yellow block inside a padded bbox. It is a
      hollow yellow rectangular frame whose entire outer perimeter is occupied by
      yellow, except for 2-cell red connector ports cut into some sides.
    - Non-anchor sprites still have red boundary connectors, but now each sprite
      also has exactly one green 2-cell strip immediately behind one connector
      pair. That green-backed connector is the parent-facing side.
    - Small example:
      imagine a piece with red connectors on its left and top edges. In the
      original task either side could be tried first. In this combo, if the left
      connector has two green support pixels behind it, that left side must be the
      side that approaches the anchor/frame or an already-assembled parent piece;
      the top reds must stay exposed for a later child.
    - Anchor change trace:
      originally the anchor was a solid yellow block with ports around it. Now the
      anchor is a ring, so the solver must treat the yellow object as a hollow
      border and pieces must connect to its outside edge, not to an interior mass.
  b_complexity_calibration:
    - Green is globally reused as a functional marker, so same-color overlap is no
      longer allowed. Only red-on-red overlap is legal.
    - The direction cue is constrained to four discrete side categories
      (top/right/bottom/left), keeping the rule visible instead of free-form.
    - The generator rejects trees that would make the cue redundant: every sample
      must contain at least one non-anchor piece with multiple connector sides, so
      the green side actually disambiguates how that piece joins.
    - The generator also requires the hollow anchor to use multiple frame sides, so
      the frame is visually meaningful rather than a cosmetic recolor.
  c_implicit_assumption_audit:
    - Original `_extract_pieces` only needed red coordinates. The combo must also
      decode side-grouped reds and find the unique green-backed incoming side.
    - Original `_try_place` accepted generic same-color overlap. That breaks once
      green is shared by all pieces, so placement now allows only red-over-red.
    - Original DFS treated all connector reds on a piece as symmetric. The combo
      breaks that symmetry: only the marked connector pair may overlap existing
      unpaired reds when a piece is first attached.
    - Original anchor logic only assumed "contains yellow". The combo keeps that,
      but additionally assumes the yellow object is a hollow perimeter frame; the
      generator constructs exactly that shape.
  d_placement_legality:
    - Output is still built first in local coordinates, then shifted into the grid.
    - Each child node is positioned by matching its marked incoming red pair to one
      parent connector pair. Real conflicts are checked cell-by-cell; non-red
      overlaps are forbidden.
    - Scattering still uses random free placement with Chebyshev padding so the
      input shows disconnected sprites with readable whitespace.
  e_visual_observability:
    - Across a few samples, humans can see two stable cues:
      hollow yellow frame = fixed anchor, green-backed red side = approach side.
    - The green markers sit directly behind a specific red pair, so the directional
      meaning is local and repeatable instead of relying on hidden ordering.
  f_cleanliness:
    - The anchor interior is left black, which keeps the ring readable.
    - The green cue is only two cells and stays embedded in the piece body, adding
      information without detached clutter.
  g_inferability_of_change:
    - No hidden use of color IDs is required. Yellow always means the hollow anchor
      frame, red always means overlap connectors, and green always means the unique
      parent-facing connector strip.
  new_main_assumptions_checked_against_generator:
    - Exactly one component contains yellow, and it is a hollow rectangular frame.
    - Every non-anchor piece has exactly one marked 2-cell incoming connector.
    - At least one non-anchor piece has another connector side besides the marked
      one, so the direction cue is behaviorally active.
    - The anchor uses at least two sides, making the frame geometry visible.
"""

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


BG = 0
RED = 2
DIR = 3
YELLOW = 4

SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}


def _side_reds(arr: np.ndarray) -> dict[str, list[tuple[int, int]]]:
    h, w = arr.shape
    out: dict[str, list[tuple[int, int]]] = {}
    top = [(0, c) for c in range(w) if int(arr[0, c]) == RED]
    bottom = [(h - 1, c) for c in range(w) if int(arr[h - 1, c]) == RED]
    left = [(r, 0) for r in range(h) if int(arr[r, 0]) == RED]
    right = [(r, w - 1) for r in range(h) if int(arr[r, w - 1]) == RED]
    if top:
        out["top"] = [(int(r), int(c)) for r, c in top]
    if right:
        out["right"] = [(int(r), int(c)) for r, c in right]
    if bottom:
        out["bottom"] = [(int(r), int(c)) for r, c in bottom]
    if left:
        out["left"] = [(int(r), int(c)) for r, c in left]
    return out


def _decode_marked_side(arr: np.ndarray, side_reds: dict[str, list[tuple[int, int]]]) -> str:
    h, w = arr.shape
    candidates: list[str] = []
    for side, reds in side_reds.items():
        if len(reds) != 2:
            continue
        dr, dc = INWARD[side]
        support = [(r + dr, c + dc) for r, c in reds]
        if all(0 <= rr < h and 0 <= cc < w and int(arr[rr, cc]) == DIR for rr, cc in support):
            candidates.append(side)
    assert len(candidates) == 1, "each non-anchor piece must have exactly one green-backed connector side"
    return candidates[0]


def _is_hollow_anchor(arr: np.ndarray) -> bool:
    h, w = arr.shape
    if h < 3 or w < 3:
        return False
    ring = np.zeros_like(arr, dtype=bool)
    ring[0, :] = True
    ring[h - 1, :] = True
    ring[:, 0] = True
    ring[:, w - 1] = True
    nonzero = arr != BG
    if np.any(nonzero & ~ring):
        return False
    if not np.all(nonzero[ring]):
        return False
    return bool(np.any(arr[1 : h - 1, 1 : w - 1] == BG))


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        reds = [(int(r), int(c)) for r, c in np.argwhere(arr == RED)]
        side_reds = _side_reds(arr)
        is_anchor = bool(np.any(arr == YELLOW))
        marked_side = None if is_anchor else _decode_marked_side(arr, side_reds)
        incoming_reds = tuple(side_reds.get(marked_side, [])) if marked_side is not None else tuple()
        pieces.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "reds": reds,
                "side_reds": side_reds,
                "marked_side": marked_side,
                "incoming_reds": incoming_reds,
                "incoming_reds_set": set(incoming_reds),
                "shape": (int(arr.shape[0]), int(arr.shape[1])),
                "area": int(np.sum(arr != bg)),
                "is_anchor": is_anchor,
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

    conflict = mask & (sub != bg) & ~((arr == RED) & (sub == RED))
    if np.any(conflict):
        return None

    out = canvas.copy()
    out_sub = out[top : top + ph, left : left + pw]
    out_sub[mask] = arr[mask]
    out[top : top + ph, left : left + pw] = out_sub
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered sprites around a fixed hollow yellow frame by reading both red connectors and green direction markers.

    Input structure:
    - bg = black (0). Several disconnected sprites are scattered across the grid.
    - One sprite is a hollow yellow (4) rectangular frame; it stays in place and serves as the anchor.
    - Every non-anchor sprite has red (2) connector pixels plus exactly one green (3) backing strip directly behind one connector pair.

    Rule:
    - Translate pieces only; do not rotate or mirror them.
    - A piece may join the growing assembly only through its green-backed red connector pair.
    - All other red connectors on that piece must remain unpaired until later, and in the final assembly every red pixel is overlapped exactly once by another red pixel.

    Key insight: The hollow yellow ring tells you which object is the fixed anchor, while the green strip breaks connector symmetry by showing which red side is meant to face inward toward the existing structure.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = BG

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must have at least one colored piece"

    anchors = [p for p in pieces if p["is_anchor"]]
    assert len(anchors) == 1, "input must have exactly one yellow anchor piece"
    anchor = anchors[0]
    assert _is_hollow_anchor(anchor["arr"]), "yellow anchor must be a hollow frame"

    rest = [p for p in pieces if not p["is_anchor"]]
    for p in rest:
        assert len(p["incoming_reds"]) == 2, "each non-anchor piece must expose one marked 2-cell incoming connector"

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
        rc1 = set(unpaired)

        best_moves = None
        for tr, tc in unpaired:
            moves = []
            for idx, p in enumerate(remaining):
                ph, pw = p["shape"]
                incoming = p["incoming_reds"]
                incoming_set = p["incoming_reds_set"]
                for pr, pc in incoming:
                    top = tr - pr
                    left = tc - pc
                    if top < 0 or left < 0 or top + ph > h or left + pw > w:
                        continue

                    incoming_global = {(top + rr, left + cc) for rr, cc in incoming}
                    if not incoming_global.issubset(rc1):
                        continue

                    bad = False
                    overlap = 0
                    for rr, cc in p["reds"]:
                        gr, gc = top + rr, left + cc
                        prev = rc.get((gr, gc), 0)
                        if (rr, cc) in incoming_set:
                            if prev != 1:
                                bad = True
                                break
                            overlap += 1
                        elif prev != 0:
                            bad = True
                            break
                    if bad or overlap != len(incoming):
                        continue
                    moves.append((idx, top, left, overlap))

            if not moves:
                return
            if best_moves is None or len(moves) < len(best_moves):
                best_moves = moves
                if len(best_moves) == 1:
                    break

        assert best_moves is not None, "there must be at least one feasible marked-connector move"

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
    assert len(solutions) == 1, "assembly must have exactly one valid solution"
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
PIECE_COLORS = [1, 5, 6, 7, 8, 9]
PIECE_CAP = 2
BAR_SKIP_PROB = 0.35

# Anchor
ANCHOR_HOLE_CHOICES = [(3, 3), (4, 5)]
ANCHOR_CAP_CHOICES = [2, 3]

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
        "incoming": None,
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
            if arr[rr, cc] == BG:
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


def _pick_hub(h: int, w: int) -> tuple[int, int]:
    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    return random.choice(interior)


def _make_piece(node):
    h, w = node["shape"]
    color = node["color"]
    used = set(node["sides"])
    arr = np.zeros((h, w), dtype=int)
    hub = _pick_hub(h, w)

    bar_sides = [s for s in SIDES if s not in used]
    if len(bar_sides) > 1 and random.random() < BAR_SKIP_PROB:
        bar_sides.remove(random.choice(bar_sides))
    for side in bar_sides:
        _paint_bar(arr, side, color, used)

    terminals = []
    for side in used:
        dr, dc = INWARD[side]
        marker_side = side == node["incoming"]
        for r, c in _port_cells(node, side):
            arr[r, c] = RED
            rr, cc = r + dr, c + dc
            arr[rr, cc] = DIR if marker_side else color
            terminals.append((rr, cc))

    if arr[hub] == BG:
        arr[hub] = color
    order = random.randint(0, 1)
    for t in terminals:
        _draw_path(arr, t, hub, color, order)
    for side in bar_sides:
        _draw_path(arr, hub, _bar_endpoint((h, w), side, hub), color, 1 - order)

    if len(components_mask(arr != BG, connectivity=4)) != 1:
        return None
    if int(np.sum(arr == RED)) != 2 * len(used):
        return None
    if int(np.sum(arr == DIR)) != 2:
        return None
    return arr


def _make_anchor(node):
    hole_h, hole_w = node["hole"]
    h, w = node["shape"]
    assert (h, w) == (hole_h + 2, hole_w + 2)
    arr = np.zeros((h, w), dtype=int)
    arr[0, :] = YELLOW
    arr[h - 1, :] = YELLOW
    arr[:, 0] = YELLOW
    arr[:, w - 1] = YELLOW
    for side in node["sides"]:
        for r, c in _port_cells(node, side):
            arr[r, c] = RED
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


def _tree_is_informative(nodes: list[dict]) -> bool:
    if len(nodes[0]["sides"]) < 2:
        return False
    return any(len(node["sides"]) >= 2 for node in nodes[1:])


def _grow_tree():
    hole_h, hole_w = random.choice(ANCHOR_HOLE_CHOICES)
    anchor = _new_node("anchor", hole_h + 2, hole_w + 2, YELLOW, cap=random.choice(ANCHOR_CAP_CHOICES))
    anchor["hole"] = (hole_h, hole_w)
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
                    child["incoming"] = incoming
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
    if not _tree_is_informative(nodes):
        return None
    return nodes


def _place_node_on_canvas(out, node, sr, sc):
    arr = node["arr"]
    top, left = node["top"] + sr, node["left"] + sc
    node["pos"] = (top, left)
    red_counts = {}
    mask = arr != BG
    rows, cols = np.nonzero(mask)
    g_rows, g_cols = rows + top, cols + left
    vals = arr[rows, cols]
    cur_vals = out[g_rows, g_cols]
    conflicts = (cur_vals != BG) & ~((cur_vals == RED) & (vals == RED))
    if np.any(conflicts):
        return None
    red_mask = vals == RED
    for gr, gc in zip(g_rows[red_mask], g_cols[red_mask]):
        key = (int(gr), int(gc))
        red_counts[key] = red_counts.get(key, 0) + 1
    paint_mask = cur_vals == BG
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

    total_ports = sum(int(np.sum(n["arr"] == RED)) for n in nodes)
    if len(all_red_counts) * 2 != total_ports:
        return None
    if any(v != 2 for v in all_red_counts.values()):
        return None
    if len(components_mask(out != BG, connectivity=4)) != 1:
        return None
    if not _is_hollow_anchor(nodes[0]["arr"]):
        return None
    return out


def _scatter(nodes, h, w):
    inp = np.zeros((h, w), dtype=int)
    anchor = nodes[0]
    paste(inp, anchor["arr"], *anchor["pos"], transparent=BG)
    rest = nodes[1:]
    random.shuffle(rest)
    for node in rest:
        try:
            r, c = random_free_location(
                inp,
                node["arr"],
                transparent=BG,
                padding=SCATTER_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inp, node["arr"], r, c, transparent=BG)
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
