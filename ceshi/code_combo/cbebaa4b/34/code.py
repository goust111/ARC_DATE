# Combo variant of cbebaa4b (index 34)
# combo_spec: D4/3+D16/2
# Name: 重叠颜色混合 + 间隔一格对齐

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis for the combo variant
# - Concrete fused behavior:
#   The source task glued sprites by making red connectors overlap exactly. D16/2
#   removes that overlap and instead demands a one-cell interval between opposite
#   red ports. D4/3 would normally color an overlap cell with a blend of the two
#   sprite body colors, but the interval rule eliminates the overlap location. The
#   fused rule therefore relocates the blend to the visible interval cell itself:
#   the two reds still end up two cells apart, and the middle cell becomes a mixed
#   bridge color determined by the two sprite body colors.
# - Behavioral trace on a tiny example:
#   Original: an anchor red at (10, 10) and a child red on its bottom edge were
#   translated so both reds landed on (10, 10), leaving a single red overlap cell.
#   Gap-only D16/2: the child red would land at (8, 10), with (9, 10) staying black.
#   Fused combo: the child still lands at (8, 10), but once paired, (9, 10) changes
#   from black to the blend color. The difference is visible in exactly the place
#   where the new spacing rule matters, so the two dimensions reinforce each other
#   instead of canceling out.
# - Complexity calibration:
#   Arbitrary pairwise color tables would be hard to infer from 2-4 ARC examples, so
#   the generator deliberately keeps the body palette tiny. Non-anchor pieces use only
#   blue(1) and green(3), while the anchor is yellow(4). That restricts blends to the
#   small repeated set {yellow+blue -> pink(6), blue+green -> teal(8), yellow+green
#   -> orange(7)}. The produced trees repeatedly reuse the first two cases, which makes
#   the bridge-color rule visible without flooding the grid with many unrelated colors.
# - Implicit assumptions that break from the source solver:
#   The original `_try_place` allowed red-on-red overlap and the DFS counted how many
#   sprites mapped onto each red coordinate. Neither assumption survives. Here red
#   coordinates are always distinct, the middle cell is the only place that changes
#   color after a match, and a partial state is valid only if every unmatched port
#   still sees a black interval cell while every matched port sees the correct mixed
#   bridge. This is why the solver tracks oriented red ports plus their owner colors.
# - Placement legality:
#   As in D16/2, assembled piece sprites may not overlap and may not touch directly;
#   new contact is introduced only by painting bridge cells after a legal distance-2
#   port match has been established. The generator first places bare sprites under
#   those stricter no-touch rules, then paints bridge cells, and finally revalidates
#   every port pair against the same bridge logic used by the solver.
# - Visual observability and cleanliness:
#   The output keeps the source task's central yellow anchor and scattered colored
#   sprites, but the assembled result now has conspicuous one-pixel bridge colors
#   sitting between opposed red pixels. Because only a few bridge colors are used and
#   they appear repeatedly in the same structural role, a human can read them as the
#   "mixed gap fillers" rather than as new independent objects.
# - New solver assumptions and generator guarantees:
#   Each sprite still has a single body color, each red lies on exactly one bbox side,
#   and every valid bridge corresponds to exactly one opposite-facing red pair two
#   cells apart. The generator constructs pieces and connection trees to satisfy those
#   assumptions, while the solver checks them explicitly before accepting a placement.


SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
OUTWARD = {"top": (-1, 0), "bottom": (1, 0), "left": (0, -1), "right": (0, 1)}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}

BODY_BLUE = 1
BODY_GREEN = 3
ANCHOR_YELLOW = 4
MIX_PINK = 6
MIX_ORANGE = 7
MIX_TEAL = 8

MIX_TABLE = {
    frozenset((ANCHOR_YELLOW, BODY_BLUE)): MIX_PINK,
    frozenset((ANCHOR_YELLOW, BODY_GREEN)): MIX_ORANGE,
    frozenset((BODY_BLUE, BODY_GREEN)): MIX_TEAL,
}


def _mix_color(color_a: int, color_b: int) -> int:
    if color_a == color_b:
        return int(color_a)
    key = frozenset((int(color_a), int(color_b)))
    assert key in MIX_TABLE, f"unsupported blend colors: {sorted(key)}"
    return MIX_TABLE[key]


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


def _extract_reds(arr: np.ndarray, body_color: int) -> list[tuple[int, int, str, int]]:
    reds = []
    for r, c in np.argwhere(arr == 2):
        rr, cc = int(r), int(c)
        reds.append((rr, cc, _red_side(arr, rr, cc), int(body_color)))
    return reds


def _piece_color(arr: np.ndarray, bg: int) -> int:
    colors = sorted({int(v) for v in arr.flatten() if v not in (bg, 2)})
    assert len(colors) == 1, "each sprite must have exactly one body color"
    return colors[0]


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
        body_color = _piece_color(arr, bg)
        pieces.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "color": int(body_color),
                "reds": _extract_reds(arr, body_color),
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


def _apply_bridges(
    canvas: np.ndarray,
    red_info: dict[tuple[int, int], tuple[str, int]],
    bg: int,
) -> tuple[np.ndarray, list[tuple[int, int]], int] | None:
    h, w = canvas.shape
    out = canvas.copy()
    unpaired = []
    pair_count = 0
    mid_claims: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]] = {}

    for coord, (side, body_color) in red_info.items():
        mr, mc = _expected_mid(coord, side)
        pr, pc = _expected_partner(coord, side)
        if not (0 <= mr < h and 0 <= mc < w and 0 <= pr < h and 0 <= pc < w):
            return None

        dest = int(canvas[pr, pc])
        mid_val = int(canvas[mr, mc])
        if dest == bg:
            if mid_val != bg:
                return None
            unpaired.append(coord)
            continue
        if dest != 2:
            return None

        partner_info = red_info.get((pr, pc))
        if partner_info is None or partner_info[0] != OPPOSITE[side]:
            return None

        pair_key = tuple(sorted((coord, (pr, pc))))
        prev = mid_claims.get((mr, mc))
        if prev is not None and prev != pair_key:
            return None
        mid_claims[(mr, mc)] = pair_key

        mix = _mix_color(body_color, partner_info[1])
        if mid_val not in (bg, mix):
            return None
        out[mr, mc] = mix

        if coord < (pr, pc):
            pair_count += 1

    return out, unpaired, pair_count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the scattered sprites around the fixed yellow anchor so that matched red ports stay one cell apart and the interval cell becomes a color blend.

    Input structure:
    - bg = black (0). Several sprites are separated on the canvas.
    - Exactly one sprite is the anchor and contains the yellow (4) block.
    - Every sprite has red (2) boundary ports, and each sprite otherwise uses a single body color.

    Rule:
    - Keep the anchor in place and translate the other sprites without rotation.
    - A valid attachment places one red port exactly two cells away from an opposite-facing unpaired red, so there is one intervening cell between them.
    - When such a pair is completed, that intervening cell changes from black to the mixed bridge color determined by the two sprites' body colors; all ports must be paired this way in the final assembly.

    Key insight:
    - The red pixels still define the matching geometry, but success is no longer "red overlaps red". Each red predicts both a partner position two steps outward and a bridge cell one step outward, so solving means satisfying the distance-2 port pairing and the color of every inserted bridge simultaneously.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must have at least one colored piece"

    anchors = [p for p in pieces if p["color"] == ANCHOR_YELLOW]
    assert len(anchors) == 1, "input must have exactly one yellow (4) anchor piece"
    anchor = anchors[0]
    rest = [p for p in pieces if p is not anchor]

    canvas = np.full((h, w), bg, dtype=int)
    ar0, ac0, _, _ = anchor["bbox"]
    placed = _try_place(canvas, anchor["arr"], ar0, ac0, bg)
    assert placed is not None, "anchor piece must fit on canvas at its original position"

    total_ports = int(sum(len(p["reds"]) for p in pieces))
    assert total_ports % 2 == 0, "total red connector ports must be even for complete pairing"

    red_info: dict[tuple[int, int], tuple[str, int]] = {}
    for rr, cc, side, body_color in anchor["reds"]:
        red_info[(ar0 + rr, ac0 + cc)] = (side, body_color)

    anchor_status = _apply_bridges(placed, red_info, bg)
    assert anchor_status is not None, "anchor connectors must point into free space at the start"

    solutions: list[np.ndarray] = []

    def dfs(
        cur: np.ndarray,
        remaining: list[dict],
        cur_red_info: dict[tuple[int, int], tuple[str, int]],
    ) -> None:
        if len(solutions) >= 2:
            return

        status = _apply_bridges(cur, cur_red_info, bg)
        if status is None:
            return
        cur_with_bridges, unpaired, pair_count = status

        if not remaining:
            if not unpaired and pair_count * 2 == total_ports:
                solutions.append(cur_with_bridges)
            return

        if not unpaired:
            return

        best_moves = None
        for target in unpaired:
            target_side, target_color = cur_red_info[target]
            want_side = OPPOSITE[target_side]
            partner_r, partner_c = _expected_partner(target, target_side)
            moves = []

            for idx, piece in enumerate(remaining):
                if frozenset((piece["color"], target_color)) not in MIX_TABLE and piece["color"] != target_color:
                    continue
                for pr, pc, side, _body_color in piece["reds"]:
                    if side != want_side:
                        continue
                    top = partner_r - pr
                    left = partner_c - pc
                    nxt = _try_place(cur_with_bridges, piece["arr"], top, left, bg)
                    if nxt is None:
                        continue

                    next_red_info = dict(cur_red_info)
                    collision = False
                    for rr, cc, s2, body_color in piece["reds"]:
                        key = (top + rr, left + cc)
                        if key in next_red_info:
                            collision = True
                            break
                        next_red_info[key] = (s2, body_color)
                    if collision:
                        continue

                    next_status = _apply_bridges(nxt, next_red_info, bg)
                    if next_status is None:
                        continue
                    _next_canvas, next_unpaired, next_pairs = next_status
                    moves.append((idx, top, left, len(next_unpaired), next_pairs))

            if not moves:
                return
            if best_moves is None or len(moves) < len(best_moves):
                best_moves = moves
                if len(best_moves) == 1:
                    break

        assert best_moves is not None, "there must be at least one feasible bridge move"

        dedup: dict[tuple[int, int, int], tuple[int, int]] = {}
        for idx, top, left, unp, pairs in best_moves:
            key = (idx, top, left)
            score = (pairs, -unp)
            prev = dedup.get(key)
            if prev is None or score > prev:
                dedup[key] = score

        moves2 = [
            (idx, top, left, score[0], score[1])
            for (idx, top, left), score in dedup.items()
        ]
        moves2.sort(
            key=lambda x: (-x[3], -x[4], -remaining[x[0]]["area"], x[1], x[2], x[0])
        )

        for idx, top, left, _pairs, _neg_unp in moves2:
            piece = remaining[idx]
            nxt = _try_place(cur_with_bridges, piece["arr"], top, left, bg)
            if nxt is None:
                continue

            next_red_info = dict(cur_red_info)
            for rr, cc, side, body_color in piece["reds"]:
                next_red_info[(top + rr, left + cc)] = (side, body_color)

            next_status = _apply_bridges(nxt, next_red_info, bg)
            if next_status is None:
                continue
            nxt_with_bridges, _next_unpaired, _next_pairs = next_status

            dfs(
                nxt_with_bridges,
                remaining[:idx] + remaining[idx + 1 :],
                next_red_info,
            )
            if len(solutions) >= 2:
                return

    dfs(placed, rest, red_info)
    assert len(solutions) == 1, "blend-gap assembly must have exactly one valid solution"
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
PIECE_CAP = 2
BAR_SKIP_PROB = 0.35

# Anchor
ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]
ANCHOR_CAP_CHOICES = [1, 2]

# Scatter
SCATTER_PADDING = 2

# Generation
GEN_RETRY_LIMIT = 400


def _new_node(kind, h, w, color, top=0, left=0, cap=PIECE_CAP, depth=0):
    return {
        "kind": kind,
        "shape": (int(h), int(w)),
        "color": int(color),
        "top": int(top),
        "left": int(left),
        "cap": int(cap),
        "depth": int(depth),
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
    arr[1 : 1 + bh, 1 : 1 + bw] = ANCHOR_YELLOW
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


def _child_color(parent_depth: int) -> int:
    return BODY_BLUE if parent_depth % 2 == 0 else BODY_GREEN


def _grow_tree():
    bh, bw = random.choice(ANCHOR_BLOCK_CHOICES)
    anchor = _new_node(
        "anchor",
        bh + 2,
        bw + 2,
        ANCHOR_YELLOW,
        cap=random.choice(ANCHOR_CAP_CHOICES),
        depth=0,
    )
    anchor["block"] = (bh, bw)
    nodes = [anchor]
    n_pieces = random.randint(*PIECE_COUNT_RANGE)

    for _ in range(n_pieces):
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
                    child = _new_node(
                        "piece",
                        ph,
                        pw,
                        _child_color(parent["depth"]),
                        cap=PIECE_CAP,
                        depth=parent["depth"] + 1,
                    )
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
        (top + rr, left + cc): (side, body_color)
        for rr, cc, side, body_color in node["reds"]
    }


def _assemble(nodes, h, w):
    for node in nodes:
        node["arr"] = _make_anchor(node) if node["kind"] == "anchor" else _make_piece(node)
        if node["arr"] is None:
            return None
        node["reds"] = _extract_reds(node["arr"], node["color"])

    r0, c0, r1, c1 = _union_bbox(nodes)
    if (r1 - r0 + 1) > h or (c1 - c0 + 1) > w:
        return None
    sr = random.randint(-r0, h - 1 - r1)
    sc = random.randint(-c0, w - 1 - c1)

    piece_canvas = np.zeros((h, w), dtype=int)
    all_red_info: dict[tuple[int, int], tuple[str, int]] = {}
    for node in nodes:
        rc = _place_node_on_canvas(piece_canvas, node, sr, sc)
        if rc is None:
            return None
        for k, v in rc.items():
            if k in all_red_info:
                return None
            all_red_info[k] = v

    status = _apply_bridges(piece_canvas, all_red_info, 0)
    if status is None:
        return None
    out, unpaired, pair_count = status
    total_ports = sum(len(n["reds"]) for n in nodes)
    if unpaired:
        return None
    if pair_count * 2 != total_ports:
        return None
    if len(components_mask(out != 0, connectivity=4)) != 1:
        return None

    bridge_colors = {MIX_PINK, MIX_TEAL}
    if not bridge_colors.issubset(set(map(int, out.flatten()))):
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
