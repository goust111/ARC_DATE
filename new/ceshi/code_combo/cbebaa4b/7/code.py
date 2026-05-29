# Combo variant of cbebaa4b (index 7)
# combo_spec: D3/1+D12/0
# Name: 无固定锚点 + 所有精灵同色

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    - Source task: one component is visibly special because it contains a yellow block.
#      The solver keeps that anchor fixed, then translates the other sprites so red
#      connector pixels overlap in pairs.
#    - D3/1 removes the yellow anchor and replaces it with a visible size rule: the
#      stationary starting sprite is the unique largest component by filled area.
#    - D12/0 removes color diversity among sprites. After fusion, every sprite body is
#      the same blue color, so color no longer distinguishes the root from the other
#      pieces, and it also cannot distinguish piece identities.
#    - Concrete trace: suppose the input has five scattered blue sprites with red ports.
#      Their filled areas are 34, 18, 16, 14, and 11. The original solver would search
#      for yellow and fail immediately. The fused solver instead selects the 34-cell
#      sprite as the seed, keeps it at its input coordinates, and attaches the other four
#      blue sprites by pairing red ports until all red cells are matched exactly twice.
#    - The critical low-level difference is placement legality. In the source code,
#      equal non-red colors were allowed to overlap via `(arr == sub)`. Once every sprite
#      is blue, that would create false merges, so the fused solver must allow only
#      red-on-red overlap and reject every other occupied/occupied collision.
#
# b) Complexity calibration
#    - The fusion should not collapse into an ambiguous pile of identical pieces. The
#      generator therefore makes the root visibly larger than every other sprite, with a
#      required post-render area margin.
#    - I keep the connection model unchanged: no rotation, no reflection, tree-shaped
#      connector graph, and every edge still matches two red cells. That preserves the
#      original jigsaw reasoning while removing the two strongest visual cues.
#    - Visual variety still comes from bbox dimensions, side usage, connector offsets,
#      and interior path routing, so "same color" does not mean "same shape".
#
# c) Hidden-assumption audit of the original main()
#    - `_extract_pieces(...)` still holds because the scattered input remains a set of
#      disjoint 4-connected non-black components.
#    - `anchors = [p for p in pieces if np.any(p["arr"] == 4)]` is invalid after D3/1
#      and D12/0 because there is no yellow block anywhere.
#    - The original `_try_place()` accepted equal non-red overlaps. That assumption was
#      harmless when colors usually differed, but it is wrong once all bodies share one
#      color, so the collision predicate must become strict.
#    - The DFS itself only assumes one fixed seed, translation-only motion, and paired
#      red connectors. Those assumptions still hold after the seed-selection rule is
#      changed from "contains yellow" to "has unique maximum area".
#
# d) Placement legality
#    - The generator still uses output-first construction: build a valid assembled target
#      first, then scatter all non-root pieces with Chebyshev padding >= 2.
#    - In the assembled output, only red connector cells may overlap. `_place_node_on_canvas`
#      enforces that and rejects all body-color collisions.
#    - Every node is placed fully in-bounds, child candidates may overlap only their
#      parent in local assembly coordinates, and the union bbox must fit within the grid.
#
# e) Visual observability
#    - Across a few examples, a human can see two consistent changes from the source
#      task: there is no yellow anchor at all, and all sprites share one body color.
#    - The stationary start piece is still inferable because it is always the unique
#      largest sprite by filled area. That is a visible property, not hidden metadata.
#    - Red connector placement remains the only overlap cue, so the solution can still be
#      inferred from geometry alone.
#
# f) Visual cleanliness
#    - The assembled output contains only one overlap type: paired red cells. No blue
#      body pixels stack on each other, so the result stays legible even though all
#      sprites are monochrome.
#    - The root remains at its assembled position while the others are scattered away
#      from it, which keeps the input uncluttered.
#
# g) Change inferability
#    - The color rule is deliberately simple and global: every sprite body is blue(1).
#      There is no per-piece recoloring to reverse-engineer.
#    - The start rule is also simple and global: pick the unique largest sprite. Because
#      both rules are visible in every sample, the fusion is inferable from 2-4 examples.
#
# h) Behavior metrics used after implementation
#    - Metric 1: zero yellow frequency. It should be 100% in generated inputs.
#    - Metric 2: body-color set per sample after excluding black and red. It should
#      always be exactly `{1}`.
#    - Metric 3: root area margin = largest component area - second-largest component
#      area. It should stay strictly positive and comfortably above zero.
#
# i) New assumptions introduced by the fused solver and how the generator guarantees them
#    - `main()` assumes exactly one component has strictly maximal filled area.
#      `_assemble()` verifies that after rendering all sprites.
#    - `main()` assumes that largest component is the stationary seed in the input.
#      `_scatter()` keeps `nodes[0]` at its assembled coordinates.
#    - `main()` assumes non-red overlaps are always illegal. `_assemble()` enforces the
#      same rule used by `_try_place()`.
#    - `main()` assumes the red-pairing assembly is unique. `_gen_attempt()` checks this
#      by requiring `main(input) == assembled_output`.


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

    # In the fused variant every body pixel uses the same color, so only red connectors
    # may overlap. All other occupied/occupied collisions are illegal.
    conflict = mask & (sub != bg) & ~((arr == 2) & (sub == 2))
    if np.any(conflict):
        return None

    out = canvas.copy()
    out_sub = out[top : top + ph, left : left + pw]
    out_sub[mask] = arr[mask]
    out[top : top + ph, left : left + pw] = out_sub
    return out


def _select_seed_piece(pieces: list[dict]) -> tuple[dict, list[dict]]:
    max_area = max(p["area"] for p in pieces)
    seeds = [p for p in pieces if p["area"] == max_area]
    assert len(seeds) == 1, "input must have exactly one largest piece"
    seed = seeds[0]
    rest = [p for p in pieces if p is not seed]
    return seed, rest


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble a monochrome set of scattered connector sprites by first inferring the unique largest starting piece.

    Input structure:
    - bg = black (0). Several disconnected sprites are scattered on the grid.
    - There is no yellow anchor block anywhere; every sprite body uses the same non-red color.
    - Red (2) pixels on sprite boundaries are connector ports, and exactly one sprite has the largest filled area.

    Rule:
    - Keep the unique largest sprite at its input position as the starting assembly.
    - Translate each remaining sprite, without rotation, so some of its red ports overlap currently unpaired red ports in the growing structure.
    - Only red-on-red overlap is legal, and every red connector must end up paired exactly once.

    Key insight: Color provides no identity information at all, so the puzzle is solved by two visible geometric cues only: the largest sprite is the seed, and the rest attach by exact red-port pairing with no body-pixel overlap.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must have at least one colored piece"

    seed, rest = _select_seed_piece(pieces)

    canvas = np.full((h, w), bg, dtype=int)
    sr0, sc0, _, _ = seed["bbox"]
    placed = _try_place(canvas, seed["arr"], sr0, sc0, bg)
    assert placed is not None, "seed piece must fit on canvas at its original position"

    total_ports = int(sum(len(p["reds"]) for p in pieces))
    assert total_ports % 2 == 0, "total red connector ports must be even for complete pairing"
    target_unique_red = total_ports // 2

    red_counts: dict[tuple[int, int], int] = {}
    for rr, cc in seed["reds"]:
        red_counts[(sr0 + rr, sc0 + cc)] = red_counts.get((sr0 + rr, sc0 + cc), 0) + 1

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
        for idx, top, left, overlap in best_moves:
            key = (idx, top, left)
            dedup[key] = max(dedup.get(key, 0), overlap)
        moves2 = [(idx, top, left, overlap) for (idx, top, left), overlap in dedup.items()]
        moves2.sort(key=lambda x: (-x[3], -remaining[x[0]]["area"], x[1], x[2], x[0]))

        for idx, top, left, _overlap in moves2:
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
COMMON_PIECE_COLOR = 1
PIECE_CAP = 2
BAR_SKIP_PROB = 0.35

# Root piece
ROOT_H_RANGE = (8, 10)
ROOT_W_RANGE = (8, 10)
ROOT_CAP_CHOICES = [1, 2]
ROOT_BAR_SKIP_PROB = 0.0
ROOT_AREA_MARGIN = 8

# Scatter
SCATTER_PADDING = 2

# Generation
GEN_RETRY_LIMIT = 300

SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}


def _new_node(kind, h, w, color, top=0, left=0, cap=PIECE_CAP, bar_skip_prob=BAR_SKIP_PROB):
    return {
        "kind": kind,
        "shape": (int(h), int(w)),
        "color": int(color),
        "top": int(top),
        "left": int(left),
        "cap": int(cap),
        "bar_skip_prob": float(bar_skip_prob),
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
    if len(bar_sides) > 1 and random.random() < node["bar_skip_prob"]:
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
    for terminal in terminals:
        _draw_path(arr, terminal, hub, color, order)
    for side in bar_sides:
        _draw_path(arr, hub, _bar_endpoint((h, w), side, hub), color, 1 - order)

    if len(components_mask(arr != 0, connectivity=4)) != 1:
        return None
    if int(np.sum(arr == 2)) != 2 * len(used):
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


def _grow_tree():
    n_pieces = random.randint(*PIECE_COUNT_RANGE)
    root = _new_node(
        "root",
        random.randint(*ROOT_H_RANGE),
        random.randint(*ROOT_W_RANGE),
        COMMON_PIECE_COLOR,
        cap=random.choice(ROOT_CAP_CHOICES),
        bar_skip_prob=ROOT_BAR_SKIP_PROB,
    )
    nodes = [root]

    for _ in range(n_pieces):
        placed = False
        dims = [
            (random.randint(*PIECE_H_RANGE), random.randint(*PIECE_W_RANGE))
            for _ in range(PIECE_DIM_CANDIDATES)
        ]
        random.shuffle(dims)
        for ph, pw in dims:
            parents = [j for j, node in enumerate(nodes) if len(node["sides"]) < node["cap"]]
            random.shuffle(parents)
            for parent_i in parents:
                parent = nodes[parent_i]
                sides = [s for s in SIDES if s not in parent["sides"]]
                random.shuffle(sides)
                for side in sides:
                    child = _new_node("piece", ph, pw, COMMON_PIECE_COLOR)
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


def _assembled_areas(nodes):
    return [int(np.sum(node["arr"] != 0)) for node in nodes]


def _assemble(nodes, h, w):
    for node in nodes:
        node["arr"] = _make_piece(node)
        if node["arr"] is None:
            return None

    areas = _assembled_areas(nodes)
    if len(areas) < 2:
        return None
    root_area = areas[0]
    max_other = max(areas[1:])
    if root_area != max(areas) or areas.count(root_area) != 1:
        return None
    if root_area - max_other < ROOT_AREA_MARGIN:
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
        for key, value in rc.items():
            all_red_counts[key] = all_red_counts.get(key, 0) + value

    total_ports = sum(int(np.sum(node["arr"] == 2)) for node in nodes)
    if len(all_red_counts) * 2 != total_ports:
        return None
    if any(value != 2 for value in all_red_counts.values()):
        return None
    if len(components_mask(out != 0, connectivity=4)) != 1:
        return None
    if np.any(out == 4):
        return None
    if set(int(v) for v in np.unique(out) if v not in (0, 2)) != {COMMON_PIECE_COLOR}:
        return None
    return out


def _scatter(nodes, h, w):
    inp = np.zeros((h, w), dtype=int)
    root = nodes[0]
    paste(inp, root["arr"], *root["pos"], transparent=0)
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
