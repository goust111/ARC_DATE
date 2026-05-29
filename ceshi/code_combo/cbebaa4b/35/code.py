# Combo variant of cbebaa4b (index 35)
# combo_spec: D9/1+D11/2
# Name: 锚点同色精灵 + 干扰精灵

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) concretization_and_trace:
#    - The base task still assembles scattered sprites by translating them until red connector
#      pixels pair up. The fused change modifies two steps at once:
#      1) the anchor body color is no longer unique, and one real sprite shares that color;
#      2) 1-2 extra red-bearing sprites are present but never belong to the final assembly.
#    - Concrete trace:
#      original: `main()` located the anchor with `arr == 4`, then used every remaining component
#      to close all red ports.
#      combo: color no longer identifies the anchor, so `main()` must first detect the solid
#      centered rectangle structurally. After that, DFS must search only for a subset of remaining
#      pieces, because some components are distractors. Example: a centered 5x6 solid block and a
#      side-bracket sprite may both be color 6; only the solid rectangle is the anchor, while an
#      additional distractor might contain three interior red cells that can never legally overlap
#      any exposed connector.
#    - The visible difference from the source task is therefore twofold in every valid sample:
#      the anchor loses its special yellow cue, and some red-marked sprites disappear in output.
# b) complexity_calibration:
#    - To keep the D9 ambiguity real but controlled, exactly one genuine non-anchor sprite shares
#      the anchor color; distractors are sampled from the other colors so "same color as anchor"
#      remains a meaningful clue rather than random noise.
#    - To keep D11 visible without exploding search, there are only 1-2 distractors, and they are
#      generated from a small family of chunky single components with interior red cells. They look
#      sprite-like, not random speckles, but their connector geometry is consistently invalid.
# c) implicit_assumptions_review:
#    - `_try_place()` in the source code allowed same-color overlap via `(arr == sub)`. That relied
#      on all non-red bodies having distinct colors. Once the anchor shares a color with a real
#      sprite, that exception becomes unsound, so only red-on-red overlap remains legal.
#    - Anchor detection by color also breaks. `main()` now uses `_is_anchor_piece()` plus centrality
#      tie-breaking, because the anchor is the only solid rectangular body with one-cell-outside rim
#      red ports.
#    - The source DFS assumed every component participates, so `total_ports` and "remaining must be
#      empty" are no longer valid. The combo solver instead accepts a state when all currently used
#      red coordinates are perfectly paired and ignores unused leftover components as distractors.
#    - The combo also introduces a new solver assumption: every distractor has zero legal first
#      attachment moves against any real piece. The generator enforces that explicitly.
# d) placement_legality:
#    - Real pieces are still created assembled-first, checked for in-bounds placement, and scattered
#      later with Chebyshev padding.
#    - The anchor is shifted near grid center so humans can still distinguish the solid central body
#      from the same-colored ordinary sprite.
#    - Distractors are only accepted if they are single 4-connected components, non-anchor-shaped,
#      and fail every possible red-alignment attempt against every genuine piece on an empty host
#      canvas. This prevents accidental legal attachments.
# e) visual_observability:
#    - Across 2-4 examples, a human can infer the combo rule: the center block is not identified by
#      color anymore, yet it stays fixed; one same-colored routed sprite moves into it; other
#      red-bearing sprites never appear in output.
#    - No hidden ordering or color-id arithmetic is used. The observable cues are shape, location,
#      and whether a component can participate in the red-pairing structure.
# f) visual_cleanliness:
#    - Same-color ambiguity is limited to one genuine sprite, preventing the scene from becoming a
#      pile of indistinguishable bodies.
#    - Distractors remain isolated scattered components and never overlap anything in input. Output
#      remains a single clean connected assembly with only red-on-red overlaps.
# g) change_inference:
#    - The only color-role change is "anchor color is reused once by a real sprite". This is backed
#      by repeated structure: the repeated color always appears on one solid centered block and one
#      ordinary bracket/path sprite.
# h) post-change solver assumptions audit:
#    - `_is_anchor_piece()` assumes the anchor body is a solid 3x3 or 4x5 rectangle with rim reds
#      exactly one cell outside that body. `_make_anchor()` preserves that.
#    - The DFS assumes a next real move must cover at least one currently unpaired red coordinate.
#      Real pieces satisfy this by construction; distractors are rejected unless they satisfy none.
#    - The uniqueness assertion assumes generation filters out alternative valid subsets. `_gen_attempt()`
#      verifies this by running `main()` and checking against the intended assembled output.

ANCHOR_BODY_CHOICES = ((3, 3), (4, 5))


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


def _anchor_center_distance_sq(piece: dict, h: int, w: int) -> int:
    r0, c0, r1, c1 = piece["bbox"]
    pr2 = r0 + r1
    pc2 = c0 + c1
    return int((pr2 - (h - 1)) ** 2 + (pc2 - (w - 1)) ** 2)


def _is_anchor_piece(arr: np.ndarray, bg: int) -> bool:
    body_mask = (arr != bg) & (arr != 2)
    if not np.any(body_mask):
        return False

    body_colors = np.unique(arr[body_mask])
    if len(body_colors) != 1:
        return False
    body_color = int(body_colors[0])

    br0, bc0, br1, bc1 = bbox_coords(np.argwhere(body_mask))
    bh, bw = br1 - br0 + 1, bc1 - bc0 + 1
    if (bh, bw) not in ANCHOR_BODY_CHOICES:
        return False
    if np.any(arr[br0 : br1 + 1, bc0 : bc1 + 1] != body_color):
        return False

    extras = (arr != bg) & ~body_mask
    if not np.any(extras):
        return False

    for r, c in np.argwhere(extras):
        if arr[r, c] != 2:
            return False
        if r < br0 and br0 - r == 1 and bc0 <= c <= bc1:
            continue
        if r > br1 and r - br1 == 1 and bc0 <= c <= bc1:
            continue
        if c < bc0 and bc0 - c == 1 and br0 <= r <= br1:
            continue
        if c > bc1 and c - bc1 == 1 and br0 <= r <= br1:
            continue
        return False
    return True


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

    conflict = mask & (sub != bg) & ~((arr == 2) & (sub == 2))
    if np.any(conflict):
        return None

    out = canvas.copy()
    out_sub = out[top : top + ph, left : left + pw]
    out_sub[mask] = arr[mask]
    out[top : top + ph, left : left + pw] = out_sub
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the valid connector structure around a fixed solid anchor whose color is reused by one real sprite, while ignoring red-bearing decoys.

    Input structure:
    - bg = black (0). Several disconnected colored sprites are scattered on the grid.
    - One component is the anchor: a solid rectangular body with red (2) connector pairs on its rim, placed near the center.
    - Exactly one genuine non-anchor sprite shares the anchor's non-red body color, and 1-2 additional sprites may contain red pixels but are distractors.

    Rule:
    - Keep the anchor at its input position and translate only genuine assembly pieces.
    - A placement is legal only when the moved sprite overlaps existing unpaired red connector pixels and causes no non-red overlap anywhere.
    - Finish once all red pixels used by the growing assembly are paired exactly once; any leftover components are distractors and do not appear in the output.

    Key insight:
    - Color is no longer enough to find the anchor, and red pixels are no longer enough to prove participation. The correct solution comes from identifying the solid central anchor by structure, then selecting the unique subset of sprites whose red ports can close into a full pairing.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must have at least one colored piece"

    anchor_candidates = [p for p in pieces if _is_anchor_piece(p["arr"], bg)]
    assert anchor_candidates, "input must contain a structural anchor candidate"
    anchor = min(anchor_candidates, key=lambda p: (_anchor_center_distance_sq(p, h, w), p["bbox"]))
    rest = [p for p in pieces if p is not anchor]

    canvas = np.full((h, w), bg, dtype=int)
    ar0, ac0, _, _ = anchor["bbox"]
    placed = _try_place(canvas, anchor["arr"], ar0, ac0, bg)
    assert placed is not None, "anchor piece must fit on canvas at its original position"

    red_counts: dict[tuple[int, int], int] = {}
    for rr, cc in anchor["reds"]:
        red_counts[(ar0 + rr, ac0 + cc)] = red_counts.get((ar0 + rr, ac0 + cc), 0) + 1

    solutions: dict[bytes, np.ndarray] = {}

    def _record_solution(cur: np.ndarray, rc: dict[tuple[int, int], int]) -> None:
        if not rc or any(v != 2 for v in rc.values()):
            return
        key = cur.tobytes()
        if key not in solutions:
            solutions[key] = cur.copy()

    def dfs(cur: np.ndarray, remaining: list[dict], rc: dict[tuple[int, int], int]) -> None:
        if len(solutions) >= 2:
            return

        unpaired = [k for k, v in rc.items() if v == 1]
        if not unpaired:
            _record_solution(cur, rc)
            return

        remaining_reds = sum(len(p["reds"]) for p in remaining)
        if remaining_reds < len(unpaired):
            return

        rc1 = {k for k, v in rc.items() if v == 1}
        best_moves = None

        for target in unpaired:
            moves = []
            tr, tc = target
            for idx, p in enumerate(remaining):
                ph, pw = p["shape"]
                for pr, pc in p["reds"]:
                    top = tr - pr
                    left = tc - pc
                    if top < 0 or left < 0 or top + ph > h or left + pw > w:
                        continue
                    overlap = 0
                    bad = False
                    for rr, cc in p["reds"]:
                        gr, gc = top + rr, left + cc
                        prev = rc.get((gr, gc), 0)
                        if prev >= 2:
                            bad = True
                            break
                        if (gr, gc) in rc1:
                            overlap += 1
                    if bad or overlap == 0:
                        continue
                    moves.append((idx, top, left, overlap))
            if not moves:
                continue
            if best_moves is None or len(moves) < len(best_moves):
                best_moves = moves
                if len(best_moves) == 1:
                    break

        if best_moves is None:
            return

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
            bad = False
            for rr, cc in p["reds"]:
                gr, gc = top + rr, left + cc
                next_rc[(gr, gc)] = next_rc.get((gr, gc), 0) + 1
                if next_rc[(gr, gc)] > 2:
                    bad = True
                    break
            if bad:
                continue

            nxt = _try_place(cur, p["arr"], top, left, bg)
            if nxt is None:
                continue
            dfs(nxt, next_remaining, next_rc)
            if len(solutions) >= 2:
                return

    dfs(placed, rest, red_counts)
    assert len(solutions) == 1, "combo assembly must have exactly one valid decoy-filtered solution"
    return next(iter(solutions.values()))


# --------------- generator ---------------

# Canvas
GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
MAX_ASSEMBLY_DIM = 24

# Real pieces
PIECE_COUNT_RANGE = (4, 6)
PIECE_H_RANGE = (4, 7)
PIECE_W_RANGE = (4, 8)
PIECE_DIM_CANDIDATES = 12
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]
PIECE_CAP = 2
BAR_SKIP_PROB = 0.35

# Anchor
ANCHOR_BLOCK_CHOICES = list(ANCHOR_BODY_CHOICES)
ANCHOR_CAP_CHOICES = [1, 2]
ANCHOR_CENTER_JITTER = 1

# Distractors
DISTRACTOR_COUNT_RANGE = (1, 2)
DISTRACTOR_H_RANGE = (4, 6)
DISTRACTOR_W_RANGE = (4, 6)
DISTRACTOR_ATTEMPTS = 40

# Scatter
SCATTER_PADDING = 2

# Generation
GEN_RETRY_LIMIT = 600
GEN_CACHE_LIMIT = 12

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
    if _is_anchor_piece(arr, 0):
        return None
    return arr


def _make_anchor(node):
    bh, bw = node["block"]
    arr = np.zeros(node["shape"], dtype=int)
    arr[1 : 1 + bh, 1 : 1 + bw] = node["color"]
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


def _choose_colors(n_pieces: int) -> tuple[int, list[int]]:
    anchor_color = random.choice(PIECE_COLORS)
    other_colors = [c for c in PIECE_COLORS if c != anchor_color]
    random.shuffle(other_colors)
    piece_colors = other_colors[:n_pieces]
    shared_idx = random.randrange(n_pieces)
    piece_colors[shared_idx] = anchor_color
    return anchor_color, piece_colors


def _grow_tree():
    bh, bw = random.choice(ANCHOR_BLOCK_CHOICES)
    n_pieces = random.randint(*PIECE_COUNT_RANGE)
    anchor_color, colors = _choose_colors(n_pieces)
    anchor = _new_node(
        "anchor",
        bh + 2,
        bw + 2,
        anchor_color,
        cap=random.choice(ANCHOR_CAP_CHOICES),
    )
    anchor["block"] = (bh, bw)
    nodes = [anchor]

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
    return red_counts


def _choose_shift(nodes, h, w):
    r0, c0, r1, c1 = _union_bbox(nodes)
    sr_lo, sr_hi = -r0, h - 1 - r1
    sc_lo, sc_hi = -c0, w - 1 - c1
    anchor = nodes[0]
    ah, aw = anchor["shape"]
    sr_target = (h - ah) // 2 - anchor["top"]
    sc_target = (w - aw) // 2 - anchor["left"]

    sr_lo2 = max(sr_lo, sr_target - ANCHOR_CENTER_JITTER)
    sr_hi2 = min(sr_hi, sr_target + ANCHOR_CENTER_JITTER)
    sc_lo2 = max(sc_lo, sc_target - ANCHOR_CENTER_JITTER)
    sc_hi2 = min(sc_hi, sc_target + ANCHOR_CENTER_JITTER)

    sr = (
        random.randint(sr_lo2, sr_hi2)
        if sr_lo2 <= sr_hi2
        else min(max(sr_target, sr_lo), sr_hi)
    )
    sc = (
        random.randint(sc_lo2, sc_hi2)
        if sc_lo2 <= sc_hi2
        else min(max(sc_target, sc_lo), sc_hi)
    )
    return sr, sc


def _assemble(nodes, h, w):
    for node in nodes:
        node["arr"] = _make_anchor(node) if node["kind"] == "anchor" else _make_piece(node)
        if node["arr"] is None:
            return None

    r0, c0, r1, c1 = _union_bbox(nodes)
    if (r1 - r0 + 1) > h or (c1 - c0 + 1) > w:
        return None
    sr, sc = _choose_shift(nodes, h, w)

    out = np.zeros((h, w), dtype=int)
    all_red_counts = {}
    for node in nodes:
        rc = _place_node_on_canvas(out, node, sr, sc)
        if rc is None:
            return None
        for key, val in rc.items():
            all_red_counts[key] = all_red_counts.get(key, 0) + val

    total_ports = sum(int(np.sum(n["arr"] == 2)) for n in nodes)
    if len(all_red_counts) * 2 != total_ports:
        return None
    if any(v != 2 for v in all_red_counts.values()):
        return None
    if len(components_mask(out != 0, connectivity=4)) != 1:
        return None
    return out


def _make_distractor(shape, color):
    h, w = shape
    arr = np.full((h, w), color, dtype=int)

    if random.random() < 0.6:
        side = random.choice(SIDES)
        if side == "top":
            arr[0, 1:-1] = 0
        elif side == "bottom":
            arr[-1, 1:-1] = 0
        elif side == "left":
            arr[1:-1, 0] = 0
        else:
            arr[1:-1, -1] = 0

    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    random.shuffle(interior)
    n_reds = random.choice([3, 4])
    for r, c in interior[:n_reds]:
        arr[r, c] = 2

    if len(components_mask(arr != 0, connectivity=4)) != 1:
        return None
    if int(np.sum(arr == 2)) != n_reds:
        return None
    if _is_anchor_piece(arr, 0):
        return None
    return arr


def _can_attach_to_host(candidate_arr, host_arr):
    cand_reds = np.argwhere(candidate_arr == 2)
    host_reds = np.argwhere(host_arr == 2)
    ph, pw = candidate_arr.shape
    hh, hw = host_arr.shape
    margin_r = ph + 3
    margin_c = pw + 3
    canvas = np.zeros((hh + 2 * margin_r, hw + 2 * margin_c), dtype=int)
    host_top, host_left = margin_r, margin_c
    paste(canvas, host_arr, host_top, host_left, transparent=0)
    for hr, hc in host_reds:
        for pr, pc in cand_reds:
            top = host_top + int(hr) - int(pr)
            left = host_left + int(hc) - int(pc)
            if _try_place(canvas, candidate_arr, top, left, 0) is not None:
                return True
    return False


def _make_distractors(real_nodes, anchor_color):
    distractors = []
    host_arrays = [node["arr"] for node in real_nodes]
    colors = [c for c in PIECE_COLORS if c != anchor_color]
    n = random.randint(*DISTRACTOR_COUNT_RANGE)
    for _ in range(n):
        made = None
        for _attempt in range(DISTRACTOR_ATTEMPTS):
            shape = (
                random.randint(*DISTRACTOR_H_RANGE),
                random.randint(*DISTRACTOR_W_RANGE),
            )
            color = random.choice(colors)
            arr = _make_distractor(shape, color)
            if arr is None:
                continue
            if any(_can_attach_to_host(arr, host_arr) for host_arr in host_arrays):
                continue
            made = {"arr": arr, "shape": shape, "color": color}
            break
        if made is None:
            return None
        distractors.append(made)
    return distractors


def _scatter(nodes, distractors, h, w):
    inp = np.zeros((h, w), dtype=int)
    anchor = nodes[0]
    paste(inp, anchor["arr"], *anchor["pos"], transparent=0)

    rest = [node["arr"] for node in nodes[1:]] + [d["arr"] for d in distractors]
    random.shuffle(rest)
    for arr in rest:
        try:
            r, c = random_free_location(
                inp,
                arr,
                transparent=0,
                padding=SCATTER_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inp, arr, r, c, transparent=0)
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
    distractors = _make_distractors(nodes, nodes[0]["color"])
    if distractors is None:
        return None
    inp = _scatter(nodes, distractors, h, w)
    if inp is None:
        return None
    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(solved, out):
        return inp
    return None


_VALID_INPUT_CACHE: list[np.ndarray] = []


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            if len(_VALID_INPUT_CACHE) < GEN_CACHE_LIMIT:
                _VALID_INPUT_CACHE.append(grid.copy())
            else:
                _VALID_INPUT_CACHE[random.randrange(len(_VALID_INPUT_CACHE))] = grid.copy()
            return grid
    if _VALID_INPUT_CACHE:
        return random.choice(_VALID_INPUT_CACHE).copy()
    raise RuntimeError("failed to generate a valid cbebaa4b combo input")
