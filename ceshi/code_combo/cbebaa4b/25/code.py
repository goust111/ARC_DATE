# Combo variant of cbebaa4b (index 25)
# combo_spec: D5/2+D9/2
# Name: 原位留轮廓框 + 锚点尺寸编码数量

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste


# Design analysis for the fused variant
# a) Concrete behavior trace
#    - Original task: keep the yellow anchor fixed, translate every other sprite until red
#      connector pixels overlap in pairs, and return only the final connected assembly.
#    - Fused task: the assembly step itself is unchanged, but two visible consequences are
#      added to the output.
#      1) Every moved sprite leaves behind a grey hollow bbox frame at its scattered input
#         location.
#      2) The yellow anchor block is now a square whose side length equals the number of
#         movable sprites. A 4x4 anchor means "four sprites must be attached", 5x5 means
#         five, and 6x6 means six.
#    - Small example: if the input has a 5x5 yellow anchor and five scattered sprites, the
#      solver assembles those five sprites exactly as before, then also draws five grey
#      rectangular source frames where those five sprites used to be. The two clues agree:
#      anchor size says five pieces are needed, and the output shows five abandoned frames.
# b) Complexity calibration
#    - The size code uses only three values {4,5,6}. That is visible across a few examples
#      without creating a large arbitrary lookup table.
#    - Grey is reserved for source frames only; movable sprites never use grey as a body
#      color. This keeps the new D5 cue visually crisp.
#    - We keep the original multi-piece tree assembly rather than simplifying to a star, so
#      the puzzle still requires genuine connector reasoning.
# c) Assumption audit against the original solver
#    - Original assumption: the anchor is identified only by "contains yellow". Still true,
#      but the fused solver additionally assumes the yellow region is a square and reads its
#      side length as the required movable-piece count.
#    - Original assumption: output contains only the assembled structure. No longer true; the
#      solver must append extra grey geometry after assembly.
#    - Original DFS assumptions about red-pair overlap, unique solution search, and
#      translation-only motion remain valid because neither D5/2 nor D9/2 changes connector
#      semantics.
# d) Placement legality
#    - Generation stays output-first: first build a valid assembled tree, then scatter the
#      movable sprites to create the input.
#    - Input spacing is enforced on visible sprite pixels with a Chebyshev radius-2 guard,
#      while output cleanliness is enforced separately on the future grey bbox frames: frame
#      rectangles may not overlap one another, and frame border cells may not overlap or
#      4-touch the final assembled structure.
# e) Visual observability
#    - Humans can directly count the side length of the anchor square and compare it with the
#      number of grey source frames in the output. Both cues are geometric and visible.
# f) Visual cleanliness
#    - The generator rejects layouts where a grey frame would overlap or 4-touch the
#      assembled object. This keeps the output readable as "center assembly + detached
#      validation frames".
# g) Inferability
#    - No hidden ordering, color-id arithmetic, or non-visual metadata is used. The only new
#      information is carried by the anchor's visible size and the visible count of hollow
#      frames.
# h) New solver assumptions and generator guarantees
#    - `main()` assumes `yellow_side == number_of_movable_pieces`; `_grow_tree()` constructs
#      exactly that relation.
#    - `main()` assumes all movable pieces should leave one outline each; `_scatter()` stores
#      one disjoint bbox per movable piece and never reuses those locations.
#    - `main()` assumes drawing source frames will not overwrite the assembled object;
#      `_scatter()` explicitly forbids outline/assembly overlap or 4-neighbor contact.


BG = 0
RED = 2
YELLOW = 4
GREY = 5

GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
MAX_ASSEMBLY_DIM = 24

PIECE_COUNT_VALUES = (4, 5, 6)
PIECE_H_RANGE = (4, 7)
PIECE_W_RANGE = (4, 8)
PIECE_DIM_CANDIDATES = 12
PIECE_COLORS = [1, 3, 6, 7, 8, 9]
PIECE_CAP = 2
BAR_SKIP_PROB = 0.35

ANCHOR_CAP_CHOICES = [1, 2]

SCATTER_PADDING = 2
SCATTER_ATTEMPTS = 320
GEN_RETRY_LIMIT = 1000

SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        reds = np.argwhere(arr == RED)
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

    conflict = mask & (sub != bg) & ~(((arr == RED) & (sub == RED)) | (arr == sub))
    if np.any(conflict):
        return None

    out = canvas.copy()
    out_sub = out[top : top + ph, left : left + pw]
    out_sub[mask] = arr[mask]
    out[top : top + ph, left : left + pw] = out_sub
    return out


def _yellow_side(arr: np.ndarray) -> int:
    pts = np.argwhere(arr == YELLOW)
    assert pts.size > 0, "anchor must contain yellow cells"
    r0, c0, r1, c1 = bbox_coords(pts)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h == w, "anchor yellow block must be a square"
    assert np.all(arr[r0 : r1 + 1, c0 : c1 + 1] == YELLOW), "yellow block must be solid"
    return int(h)


def _outline_cells(top: int, left: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    cells = []
    for c in range(left, left + w):
        cells.append((top, c))
        cells.append((top + h - 1, c))
    for r in range(top + 1, top + h - 1):
        cells.append((r, left))
        cells.append((r, left + w - 1))
    # Deduplicate corners while preserving simple iteration order.
    seen = set()
    out = []
    for cell in cells:
        if cell not in seen:
            seen.add(cell)
            out.append(cell)
    return out


def _draw_outline(out: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    shape = (r1 - r0 + 1, c1 - c0 + 1)
    for r, c in _outline_cells(r0, c0, shape):
        assert out[r, c] == BG, "source outlines must stay detached from the assembled structure"
        out[r, c] = color


def _add_source_outlines(assembled: np.ndarray, pieces: list[dict]) -> np.ndarray:
    out = assembled.copy()
    for piece in pieces:
        _draw_outline(out, piece["bbox"], GREY)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the scattered connector sprites around the fixed yellow anchor, then leave grey source frames behind.

    Input structure:
    - Background is black (0), one component contains the yellow anchor, and the other components are movable sprites.
    - Red (2) boundary pixels are connector ports exactly as in the seed task.
    - The yellow anchor now contains a solid square block whose side length equals the number of movable sprites.

    Rule:
    - Keep the anchor in place and translate the remaining sprites so red connector pixels pair by overlap into one connected assembly.
    - Use every movable sprite exactly once; the anchor size says how many there must be.
    - After assembling, draw a grey hollow rectangle at each moved sprite's original bounding box location.

    Key insight: The red pixels still determine how the jigsaw assembles, but the anchor square size and the count of leftover grey source frames are coupled clues about how many sprites must participate.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = BG

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must have at least one colored piece"

    anchors = [p for p in pieces if np.any(p["arr"] == YELLOW)]
    assert len(anchors) == 1, "input must have exactly one yellow anchor piece"
    anchor = anchors[0]
    rest = [p for p in pieces if p is not anchor]

    expected_piece_count = _yellow_side(anchor["arr"])
    assert len(rest) == expected_piece_count, "anchor size must equal the number of movable sprites"

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
        for idx, top, left, ov in best_moves:
            key = (idx, top, left)
            dedup[key] = max(dedup.get(key, 0), ov)
        moves2 = [(idx, top, left, ov) for (idx, top, left), ov in dedup.items()]
        moves2.sort(key=lambda x: (-x[3], -remaining[x[0]]["area"], x[1], x[2], x[0]))

        for idx, top, left, _ov in moves2:
            piece = remaining[idx]
            next_remaining = remaining[:idx] + remaining[idx + 1 :]

            next_rc = dict(rc)
            for rr, cc in piece["reds"]:
                gr, gc = top + rr, left + cc
                next_rc[(gr, gc)] = next_rc.get((gr, gc), 0) + 1

            nxt = _try_place(cur, piece["arr"], top, left, bg)
            if nxt is None:
                continue
            dfs(nxt, next_remaining, next_rc)
            if len(solutions) >= 2:
                return

    dfs(placed, rest, red_counts)
    assert len(solutions) == 1, "jigsaw assembly must have exactly one valid solution"
    return _add_source_outlines(solutions[0], rest)


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
            if arr[rr, cc] != RED:
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
            arr[r, c] = RED
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
    if int(np.sum(arr == RED)) != 2 * len(used):
        return None
    return arr


def _make_anchor(node):
    size = node["block"]
    arr = np.zeros(node["shape"], dtype=int)
    arr[1 : 1 + size, 1 : 1 + size] = YELLOW
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


def _piece_dim_ranges(n_pieces: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if n_pieces == 4:
        return PIECE_H_RANGE, PIECE_W_RANGE
    if n_pieces == 5:
        return (4, 6), (4, 7)
    return (4, 5), (4, 6)


def _grow_tree(n_pieces: int):
    anchor_size = n_pieces
    anchor = _new_node(
        "anchor",
        anchor_size + 2,
        anchor_size + 2,
        YELLOW,
        cap=random.choice(ANCHOR_CAP_CHOICES),
    )
    anchor["block"] = anchor_size
    nodes = [anchor]
    colors = random.sample(PIECE_COLORS, n_pieces)
    h_range, w_range = _piece_dim_ranges(n_pieces)

    for i in range(n_pieces):
        placed = False
        dims = [
            (random.randint(*h_range), random.randint(*w_range))
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
    conflicts = (cur_vals != 0) & ~((cur_vals == RED) & (vals == RED))
    if np.any(conflicts):
        return None
    red_mask = vals == RED
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

    total_ports = sum(int(np.sum(n["arr"] == RED)) for n in nodes)
    if len(all_red_counts) * 2 != total_ports:
        return None
    if any(v != 2 for v in all_red_counts.values()):
        return None
    if len(components_mask(out != 0, connectivity=4)) != 1:
        return None
    return out


def _make_guard_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    guard = mask.copy()
    pts = np.argwhere(mask)
    for r, c in pts:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = int(r) + dr, int(c) + dc
            if 0 <= rr < h and 0 <= cc < w:
                guard[rr, cc] = True
    return guard


def _translated_cells(arr: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    pts = np.argwhere(arr != 0)
    return [(top + int(r), left + int(c)) for r, c in pts]


def _mark_chebyshev_guard(guard: np.ndarray, cells: list[tuple[int, int]], pad: int) -> None:
    h, w = guard.shape
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            rr = r + dr
            if not (0 <= rr < h):
                continue
            for dc in range(-pad, pad + 1):
                cc = c + dc
                if 0 <= cc < w:
                    guard[rr, cc] = True


def _mark_outline_guard(guard: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = guard.shape
    for r, c in cells:
        guard[r, c] = True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                guard[rr, cc] = True


def _scatter(nodes, assembled):
    h, w = assembled.shape
    inp = np.zeros((h, w), dtype=int)
    anchor = nodes[0]
    paste(inp, anchor["arr"], *anchor["pos"], transparent=0)

    input_guard = np.zeros((h, w), dtype=bool)
    anchor_cells = _translated_cells(anchor["arr"], *anchor["pos"])
    _mark_chebyshev_guard(input_guard, anchor_cells, SCATTER_PADDING)

    outline_guard = _make_guard_mask(assembled != 0)
    outline_rects = []
    rest = nodes[1:]
    rest.sort(key=lambda n: n["shape"][0] * n["shape"][1], reverse=True)

    for node in rest:
        ph, pw = node["shape"]
        placed = False
        for _ in range(SCATTER_ATTEMPTS):
            top = random.randint(0, h - ph)
            left = random.randint(0, w - pw)
            rect = (top, left, top + ph - 1, left + pw - 1)
            if any(_rects_overlap(rect, other) for other in outline_rects):
                continue

            cells = _translated_cells(node["arr"], top, left)
            if any(input_guard[r, c] for r, c in cells):
                continue

            outline_cells = _outline_cells(top, left, node["shape"])
            outline_bad = False
            for r, c in outline_cells:
                if outline_guard[r, c]:
                    outline_bad = True
                    break
            if outline_bad:
                continue

            paste(inp, node["arr"], top, left, transparent=0)
            node["scatter_bbox"] = rect
            _mark_chebyshev_guard(input_guard, cells, SCATTER_PADDING)
            _mark_outline_guard(outline_guard, outline_cells)
            outline_rects.append(rect)
            placed = True
            break
        if not placed:
            return None
    return inp


def _expected_output(nodes, assembled):
    pieces = []
    for node in nodes[1:]:
        bbox = node["scatter_bbox"]
        pieces.append({"bbox": bbox})
    return _add_source_outlines(assembled, pieces)


def _gen_attempt(n_pieces: int):
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    nodes = _grow_tree(n_pieces)
    if nodes is None:
        return None
    assembled = _assemble(nodes, h, w)
    if assembled is None:
        return None
    inp = _scatter(nodes, assembled)
    if inp is None:
        return None
    expected = _expected_output(nodes, assembled)
    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(solved, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    order = list(PIECE_COUNT_VALUES)
    random.shuffle(order)
    budget = max(1, GEN_RETRY_LIMIT // len(order))
    for n_pieces in order:
        for _ in range(budget):
            grid = _gen_attempt(n_pieces)
            if grid is not None:
                return grid
    raise RuntimeError("failed to generate a valid cbebaa4b combo D5/2+D9/2 input")
