# Variant of cbebaa4b
# Dimension: D16 — 连接器对齐的精确度要求 (meta)
# Change: 红色连接器像素必须在相邻格子上精确对齐（像素级匹配）... → 连接器需要间隔一格对齐（非直接接触）

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis for the D16 variant
# - Concrete behavior:
#   The original task merged pieces by translating a new sprite until one of its red
#   boundary pixels landed exactly on an unpaired red pixel in the current assembly.
#   In this variant, the matching red pixels never overlap. They must face each other
#   across exactly one background cell, so a top-side red expects a bottom-side red
#   two rows above it, a right-side red expects a left-side red two columns to the
#   right, and so on. The output therefore becomes a tidy "near-connected" layout:
#   every sprite remains its own 4-connected component, but the red ports line up in
#   pairs with a single black gap.
# - Behavioral trace:
#   Old logic for a top connector at (10, 7): a child bottom connector was translated
#   so one of its reds also landed at (10, 7), producing red-on-red overlap and a
#   single connected component. New logic for the same connector: the child bottom
#   red must land at (8, 7), leaving (9, 7) black. The solver therefore cannot count
#   overlapping red coordinates anymore; it must track oriented red ports and verify
#   that each one sees exactly one opposite-facing red two cells away.
# - Complexity calibration:
#   The freedom stays close to the source task: pieces still only translate, still use
#   2-cell red ports, and still form a tree. We deliberately keep the number of legal
#   offsets small by deriving each target placement from connector orientation, which
#   prevents the "any distance-2 red could match any other" ambiguity that would make
#   uniqueness fragile. Visually, the variant is easy to notice because the assembled
#   output now has multiple separated components instead of one fused mass.
# - Implicit assumptions that had to change:
#   `_try_place` in the source solver allowed red-on-red overlap; that assumption is
#   invalid now, so placement must reject all overlap and also reject new 4-neighbor
#   contacts, otherwise the generator could accidentally fall back to direct touching.
#   The source DFS tracked a count for each occupied red coordinate; that also breaks
#   because coordinates are no longer shared. The replacement state keeps a map from
#   global red coordinate -> boundary side, then validates each port by looking at its
#   unique expected partner cell and the mandatory background midpoint.
# - Placement legality:
#   Generator-side assembled placement now computes child offsets by shifting the child
#   connector by two cells along the parent's outward normal. Candidate bounding boxes
#   may no longer overlap at all. After rendering, the assembled canvas is accepted
#   only if every port is paired across a 1-cell black gap and the final canvas has
#   exactly one connected component per sprite, which guarantees no accidental touch.
# - Visual observability and cleanliness:
#   No invisible metadata is used. The only new rule is the visible one-cell spacing
#   between red connector pairs, repeated on every edge of the tree. Colors stay in
#   the original palette and keep their old role, so the human-facing change is purely
#   geometric and easy to infer from a few examples.
# - New solver assumptions and generator guarantees:
#   Each red pixel must sit on exactly one side of its local bbox; the generator uses
#   non-corner 2-pixel ports so this holds. Any unpaired red must keep both its middle
#   gap cell and its expected partner cell empty until the matching piece arrives; the
#   DFS prunes states that violate this, and the generator creates outputs that satisfy
#   it by construction.


SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
OUTWARD = {"top": (-1, 0), "bottom": (1, 0), "left": (0, -1), "right": (0, 1)}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}


def _red_side(arr: np.ndarray, r: int, c: int) -> str:
    h, w = arr.shape
    sides = []
    if r == 0:
        sides.append("top")
    if r == h - 1:
        sides.append("bottom")
    if c == 0:
        sides.append("left")
    if c == w - 1:
        sides.append("right")
    assert len(sides) == 1, "red connector must lie on a unique boundary side"
    return sides[0]


def _extract_reds(arr: np.ndarray) -> list[tuple[int, int, str]]:
    reds = []
    for r, c in np.argwhere(arr == 2):
        reds.append((int(r), int(c), _red_side(arr, int(r), int(c))))
    return reds


def _expected_mid(coord: tuple[int, int], side: str) -> tuple[int, int]:
    dr, dc = OUTWARD[side]
    return coord[0] + dr, coord[1] + dc


def _expected_partner(coord: tuple[int, int], side: str) -> tuple[int, int]:
    dr, dc = OUTWARD[side]
    return coord[0] + 2 * dr, coord[1] + 2 * dc


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        pieces.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "reds": _extract_reds(arr),
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
    if np.any(sub[mask] != bg):
        return None

    rows, cols = np.nonzero(mask)
    for rr, cc in zip(rows, cols):
        gr, gc = top + int(rr), left + int(cc)
        for nr, nc in ((gr - 1, gc), (gr + 1, gc), (gr, gc - 1), (gr, gc + 1)):
            if 0 <= nr < h and 0 <= nc < w and canvas[nr, nc] != bg:
                return None

    out = canvas.copy()
    out_sub = out[top : top + ph, left : left + pw]
    out_sub[mask] = arr[mask]
    out[top : top + ph, left : left + pw] = out_sub
    return out


def _evaluate_red_layout(
    canvas: np.ndarray, red_sides: dict[tuple[int, int], str], bg: int
) -> tuple[list[tuple[int, int]], int] | None:
    h, w = canvas.shape
    unpaired = []
    pair_count = 0

    for coord, side in red_sides.items():
        mr, mc = _expected_mid(coord, side)
        pr, pc = _expected_partner(coord, side)
        if not (0 <= mr < h and 0 <= mc < w and 0 <= pr < h and 0 <= pc < w):
            return None
        if canvas[mr, mc] != bg:
            return None

        dest = int(canvas[pr, pc])
        if dest == bg:
            unpaired.append(coord)
            continue
        if dest != 2:
            return None
        if red_sides.get((pr, pc)) != OPPOSITE[side]:
            return None
        if coord < (pr, pc):
            pair_count += 1

    return unpaired, pair_count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered sprites around the fixed yellow anchor using red ports that face each other across a one-cell gap.

    Input structure:
    - bg = black (0). Several colored sprites are scattered apart.
    - Exactly one sprite is the anchor and contains the yellow (4) block.
    - Red (2) connector pixels sit on sprite boundaries and indicate where gap-aligned matches are allowed.

    Rule:
    - The anchor remains where it is; each other sprite is translated without rotation.
    - A sprite attaches when one of its red boundary pixels lands exactly two cells away from an unpaired red on the current assembly, with the middle cell staying black.
    - Every red connector must end up paired with one opposite-facing red, and no sprites may overlap or touch directly.

    Key insight:
    - Red pixels are still the assembly ports, but the matching test is no longer overlap. Each red port has a unique expected partner position two steps outward, so solving means satisfying all of those oriented distance-2 port constraints.
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

    red_sides: dict[tuple[int, int], str] = {}
    for rr, cc, side in anchor["reds"]:
        red_sides[(ar0 + rr, ac0 + cc)] = side

    anchor_status = _evaluate_red_layout(placed, red_sides, bg)
    assert anchor_status is not None, "anchor connectors must point into free space at the start"

    solutions: list[np.ndarray] = []

    def dfs(cur: np.ndarray, remaining: list[dict], cur_red_sides: dict[tuple[int, int], str]) -> None:
        if len(solutions) >= 2:
            return

        status = _evaluate_red_layout(cur, cur_red_sides, bg)
        if status is None:
            return
        unpaired, pair_count = status

        if not remaining:
            if not unpaired and pair_count * 2 == total_ports:
                solutions.append(cur)
            return

        if not unpaired:
            return

        best_moves = None
        for target in unpaired:
            target_side = cur_red_sides[target]
            want_side = OPPOSITE[target_side]
            partner_r, partner_c = _expected_partner(target, target_side)
            moves = []

            for idx, piece in enumerate(remaining):
                for pr, pc, side in piece["reds"]:
                    if side != want_side:
                        continue
                    top = partner_r - pr
                    left = partner_c - pc
                    nxt = _try_place(cur, piece["arr"], top, left, bg)
                    if nxt is None:
                        continue

                    next_red_sides = dict(cur_red_sides)
                    for rr, cc, s2 in piece["reds"]:
                        key = (top + rr, left + cc)
                        if key in next_red_sides:
                            break
                        next_red_sides[key] = s2
                    else:
                        next_status = _evaluate_red_layout(nxt, next_red_sides, bg)
                        if next_status is None:
                            continue
                        next_unpaired, next_pairs = next_status
                        moves.append((idx, top, left, len(next_unpaired), next_pairs))

            if not moves:
                return
            if best_moves is None or len(moves) < len(best_moves):
                best_moves = moves
                if len(best_moves) == 1:
                    break

        assert best_moves is not None, "there must be at least one feasible connector move"

        dedup: dict[tuple[int, int, int], tuple[int, int]] = {}
        for idx, top, left, unp, pairs in best_moves:
            key = (idx, top, left)
            score = (pairs, -unp)
            prev = dedup.get(key)
            if prev is None or score > prev:
                dedup[key] = score

        moves2 = [(idx, top, left, score[0], score[1]) for (idx, top, left), score in dedup.items()]
        moves2.sort(key=lambda x: (-x[3], -x[4], -remaining[x[0]]["area"], x[1], x[2], x[0]))

        for idx, top, left, _pairs, _neg_unp in moves2:
            piece = remaining[idx]
            nxt = _try_place(cur, piece["arr"], top, left, bg)
            if nxt is None:
                continue

            next_red_sides = dict(cur_red_sides)
            for rr, cc, side in piece["reds"]:
                next_red_sides[(top + rr, left + cc)] = side

            dfs(nxt, remaining[:idx] + remaining[idx + 1 :], next_red_sides)
            if len(solutions) >= 2:
                return

    dfs(placed, rest, red_sides)
    assert len(solutions) == 1, "gap-jigsaw assembly must have exactly one valid solution"
    return solutions[0]


# --------------- generator (o5x) ---------------

# ── Tunable configuration ──────────────────────────────────────────────
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
ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]
ANCHOR_CAP_CHOICES = [1, 2]

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


def _candidate_fits(nodes, child):
    cb = _bbox_of(child)
    for other in nodes:
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
                    dr, dc = OUTWARD[side]
                    child["top"] = pr + 2 * dr - cr
                    child["left"] = pc + 2 * dc - cc
                    child["sides"].add(incoming)
                    if not _candidate_fits(nodes, child):
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
    placed = _try_place(out, arr, top, left, 0)
    if placed is None:
        return None
    node["pos"] = (top, left)
    out[:, :] = placed
    return {
        (top + rr, left + cc): side
        for rr, cc, side in node["reds"]
    }


def _assemble(nodes, h, w):
    for node in nodes:
        node["arr"] = _make_anchor(node) if node["kind"] == "anchor" else _make_piece(node)
        if node["arr"] is None:
            return None
        node["reds"] = _extract_reds(node["arr"])

    r0, c0, r1, c1 = _union_bbox(nodes)
    if (r1 - r0 + 1) > h or (c1 - c0 + 1) > w:
        return None
    sr = random.randint(-r0, h - 1 - r1)
    sc = random.randint(-c0, w - 1 - c1)

    out = np.zeros((h, w), dtype=int)
    all_red_sides: dict[tuple[int, int], str] = {}
    for node in nodes:
        rc = _place_node_on_canvas(out, node, sr, sc)
        if rc is None:
            return None
        for k, v in rc.items():
            if k in all_red_sides:
                return None
            all_red_sides[k] = v

    status = _evaluate_red_layout(out, all_red_sides, 0)
    if status is None:
        return None
    unpaired, pair_count = status
    total_ports = sum(len(n["reds"]) for n in nodes)
    if unpaired:
        return None
    if pair_count * 2 != total_ports:
        return None
    if len(components_mask(out != 0, connectivity=4)) != len(nodes):
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
