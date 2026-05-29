# Combo variant of cbebaa4b (index 6)
# combo_spec: D3/0+D11/2
# Name: 多锚点 + 干扰精灵

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    The source task had one yellow anchor and every other component had to join that single tree.
#    This combo changes two places at once:
#    - D3/0: there are now 2-3 yellow anchors, so the initial red frontier is split across several
#      disconnected groups.
#    - D11/2: 1-2 extra red-bearing sprites are present in the input but disappear in the output.
#    Small trace: if anchor A and anchor B each expose a red pair, and scattered pieces of colors 3
#    and 7 can legally close those pairs, DFS must choose which anchor-group each real piece belongs
#    to. A distractor may also contain red pixels, but every attempted red alignment conflicts with
#    some occupied non-red cell, so it never enters the final output. Compared with the original
#    task, the solver must infer both "which anchor?" and "real piece or decoy?" simultaneously.
# b) Complexity calibration:
#    To keep the variant visible but tractable, each anchor-group gets only 1-3 real pieces and the
#    total number of real pieces stays at 4-5. Groups reuse one color internally so the grouping is
#    visible to a human, while distractors may borrow those same colors so color alone does not
#    solve the task. Distractors come from a small repeated family: thick bodies with interior red
#    pixels, which makes the decoy pattern observable across examples instead of arbitrary noise.
# c) Hidden assumptions broken from the original solver:
#    - `len(anchors) == 1` no longer holds; all anchors must seed the search.
#    - "all components participate" no longer holds; DFS must accept leftover pieces as distractors.
#    - the original `_try_place` allowed same-color overlap; that becomes unsafe once a group reuses
#      a color, so legal overlap is tightened to red-on-red only.
#    - stopping only when `remaining` is empty is invalid; the correct stopping condition is that
#      every currently placed red coordinate is paired exactly twice.
# d) Placement legality:
#    Generation remains output-first. We first build each anchor-group in local coordinates, then
#    place the finished groups into the global output with 1-cell separation, then scatter every
#    non-anchor sprite and distractor with Chebyshev padding 2. Distractors are rejected unless
#    they fail every possible first attachment against every genuine node, which guarantees they
#    also fail against any partial or complete assembly.
# e) Visual observability:
#    Across 2-4 examples, a human can see multiple yellow anchors staying fixed, multiple detached
#    output assemblies, and extra red-bearing sprites that vanish. Same-color pieces within a group
#    reinforce the anchor assignment, while the disappearing distractors show that not every red
#    sprite participates.
# f) Visual cleanliness:
#    Output groups are assembled without overlap clutter and kept separated. Distractors remain
#    clean single components rather than random pixels, and input scatter keeps enough spacing that
#    anchors, real pieces, and decoys remain individually readable.
# g) Inferability of changes:
#    No hidden numeric ordering or color remapping is used. Color only serves as a repeated visual
#    grouping cue; the actual rule is still spatial red-connector matching plus exclusion of pieces
#    that never legally attach.
# h) New solver assumptions introduced here:
#    - Every genuine group contains exactly one yellow anchor.
#    - Genuine pieces can still be attached by translation only, with no rotation.
#    - Each distractor has zero legal first attachments onto any genuine node.
# i) Generator obligations:
#    The generator must enforce the previous bullet explicitly and must verify every produced input
#    with `main(input) == output`. It also records the chosen anchor count and distractor count so
#    the fusion behavior can be measured after sampling.


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
                "has_yellow": bool(np.any(arr == 4)),
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

    # Same-color overlap is forbidden in the combo variant; only red-on-red may overlap.
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
    Reassemble the valid connector structures around several fixed yellow anchors while ignoring red-bearing decoys.

    Input structure:
    - bg = black (0). Multiple scattered colored sprites appear on the grid.
    - Exactly 2-3 sprites contain yellow (4) anchor blocks, and those anchors stay fixed in place.
    - The remaining sprites include genuine pieces that belong to one anchor-group each, plus 1-2 distractors that also contain red (2) pixels.

    Rule:
    - Start from all yellow anchors at once and repeatedly translate one genuine sprite so that one or more of its red pixels overlap currently unpaired red pixels on one existing group.
    - Only red-on-red overlap is legal; every other colored cell must stay non-overlapping.
    - Stop when every placed red coordinate has been paired exactly once; any still-unused scattered sprites are distractors and do not appear in the output.

    Key insight:
    - The answer is the unique subset-and-grouping of sprites whose red ports can close perfectly around the fixed anchors.
    - Solving requires two coupled inferences: which anchor-group a real sprite belongs to, and which red-bearing sprites must be rejected entirely.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must have at least one colored piece"

    anchors = [p for p in pieces if p["has_yellow"]]
    assert 2 <= len(anchors) <= 3, "input must have 2 or 3 yellow (4) anchor pieces"
    rest = [p for p in pieces if not p["has_yellow"]]
    assert rest, "input must contain non-anchor pieces"

    canvas = np.full((h, w), bg, dtype=int)
    placed = canvas
    red_counts: dict[tuple[int, int], int] = {}
    for anchor in anchors:
        ar0, ac0, _, _ = anchor["bbox"]
        placed2 = _try_place(placed, anchor["arr"], ar0, ac0, bg)
        assert placed2 is not None, "anchor piece must fit on canvas at its original position"
        placed = placed2
        for rr, cc in anchor["reds"]:
            key = (ar0 + rr, ac0 + cc)
            red_counts[key] = red_counts.get(key, 0) + 1

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
            tr, tc = target
            moves = []
            for idx, piece in enumerate(remaining):
                ph, pw = piece["shape"]
                for pr, pc in piece["reds"]:
                    top = tr - pr
                    left = tc - pc
                    if top < 0 or left < 0 or top + ph > h or left + pw > w:
                        continue
                    overlap = 0
                    bad = False
                    for rr, cc in piece["reds"]:
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
        for idx, top, left, overlap in best_moves:
            key = (idx, top, left)
            dedup[key] = max(dedup.get(key, 0), overlap)
        moves2 = [(idx, top, left, overlap) for (idx, top, left), overlap in dedup.items()]
        moves2.sort(key=lambda x: (-x[3], -remaining[x[0]]["area"], x[1], x[2], x[0]))

        for idx, top, left, _overlap in moves2:
            piece = remaining[idx]
            next_rc = dict(rc)
            bad = False
            for rr, cc in piece["reds"]:
                gr, gc = top + rr, left + cc
                next_rc[(gr, gc)] = next_rc.get((gr, gc), 0) + 1
                if next_rc[(gr, gc)] > 2:
                    bad = True
                    break
            if bad:
                continue

            nxt = _try_place(cur, piece["arr"], top, left, bg)
            if nxt is None:
                continue
            next_remaining = remaining[:idx] + remaining[idx + 1 :]
            dfs(nxt, next_remaining, next_rc)
            if len(solutions) >= 2:
                return

    dfs(placed, rest, red_counts)
    assert len(solutions) == 1, "multi-anchor decoy-filtered assembly must have exactly one valid solution"
    return next(iter(solutions.values()))


# --------------- generator ---------------

# Canvas
GRID_H_RANGE = (24, 26)
GRID_W_RANGE = (24, 26)
OUTPUT_GROUP_PADDING = 1
OUTPUT_BORDER = 1

# Group structure
ANCHOR_COUNT_CHOICES = [2, 3]
ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]
ANCHOR_CAP_CHOICES = [1, 2]
GROUP_SIZE_OPTIONS = {
    2: [(2, 2), (2, 3)],
    3: [(1, 1, 2), (1, 2, 2)],
}
GROUP_MAX_DIM = 10

# Pieces
PIECE_H_RANGE = (4, 5)
PIECE_W_RANGE = (4, 6)
PIECE_DIM_CANDIDATES = 10
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]
PIECE_CAP = 2
BAR_SKIP_PROB = 0.35

# Distractors
DISTRACTOR_COUNT_RANGE = (1, 2)
DISTRACTOR_H_RANGE = (4, 5)
DISTRACTOR_W_RANGE = (4, 5)
DISTRACTOR_ATTEMPTS = 50

# Scatter
SCATTER_PADDING = 2

# Generation
GEN_RETRY_LIMIT = 500

# Geometry
SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}

_LAST_GENERATED_META: dict | None = None


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
    boxes = [_bbox_of(node) for node in nodes]
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
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
        for value in range(a_start, a_end + step, step):
            rr, cc = (value, fixed) if is_row else (fixed, value)
            if arr[rr, cc] != 2:
                arr[rr, cc] = color


def _paint_bar(arr, side, color, used):
    h, w = arr.shape
    is_horizontal = side in ("top", "bottom")
    fixed = 0 if side in ("top", "left") else (h - 1 if side == "bottom" else w - 1)
    span = w if is_horizontal else h
    lo_conflict = "left" if is_horizontal else "top"
    hi_conflict = "right" if is_horizontal else "bottom"
    for idx in range(span):
        if (idx == 0 and lo_conflict in used) or (idx == span - 1 and hi_conflict in used):
            continue
        if is_horizontal:
            arr[fixed, idx] = color
        else:
            arr[idx, fixed] = color


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

    bar_sides = [side for side in SIDES if side not in used]
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
    for terminal in terminals:
        _draw_path(arr, terminal, hub, color, order)
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
    child_box = _bbox_of(child)
    if not _rects_overlap(child_box, _bbox_of(nodes[parent_i])):
        return False
    for idx, other in enumerate(nodes):
        if idx == parent_i:
            continue
        if _rects_overlap(child_box, _bbox_of(other)):
            return False
    r0, c0, r1, c1 = _union_bbox(nodes + [child])
    return (r1 - r0 + 1) <= GROUP_MAX_DIM and (c1 - c0 + 1) <= GROUP_MAX_DIM


def _choose_anchor_blocks(anchor_count):
    if anchor_count == 2:
        blocks = list(ANCHOR_BLOCK_CHOICES)
        random.shuffle(blocks)
        return blocks
    blocks = [ANCHOR_BLOCK_CHOICES[0], ANCHOR_BLOCK_CHOICES[1], random.choice(ANCHOR_BLOCK_CHOICES)]
    random.shuffle(blocks)
    return blocks


def _choose_group_sizes(anchor_count):
    sizes = list(random.choice(GROUP_SIZE_OPTIONS[anchor_count]))
    random.shuffle(sizes)
    return sizes


def _grow_group(block, n_pieces, color):
    bh, bw = block
    anchor = _new_node("anchor", bh + 2, bw + 2, 4, cap=random.choice(ANCHOR_CAP_CHOICES))
    anchor["block"] = (bh, bw)
    nodes = [anchor]

    for _ in range(n_pieces):
        placed = False
        dims = [
            (random.randint(*PIECE_H_RANGE), random.randint(*PIECE_W_RANGE))
            for _ in range(PIECE_DIM_CANDIDATES)
        ]
        random.shuffle(dims)
        for ph, pw in dims:
            parents = [idx for idx, node in enumerate(nodes) if len(node["sides"]) < node["cap"]]
            random.shuffle(parents)
            for parent_i in parents:
                parent = nodes[parent_i]
                sides = [side for side in SIDES if side not in parent["sides"]]
                random.shuffle(sides)
                for side in sides:
                    child = _new_node("piece", ph, pw, color)
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


def _normalize_nodes(nodes):
    r0, c0, _, _ = _union_bbox(nodes)
    for node in nodes:
        node["top"] -= r0
        node["left"] -= c0


def _render_group(nodes):
    for node in nodes:
        node["arr"] = _make_anchor(node) if node["kind"] == "anchor" else _make_piece(node)
        if node["arr"] is None:
            return None

    _normalize_nodes(nodes)
    _, _, r1, c1 = _union_bbox(nodes)
    patch = np.zeros((r1 + 1, c1 + 1), dtype=int)
    red_counts = {}
    for node in nodes:
        patch2 = _try_place(patch, node["arr"], node["top"], node["left"], 0)
        if patch2 is None:
            return None
        patch = patch2
        for rr, cc in np.argwhere(node["arr"] == 2):
            key = (node["top"] + int(rr), node["left"] + int(cc))
            red_counts[key] = red_counts.get(key, 0) + 1

    total_ports = sum(int(np.sum(node["arr"] == 2)) for node in nodes)
    if len(red_counts) * 2 != total_ports:
        return None
    if any(value != 2 for value in red_counts.values()):
        return None
    if len(components_mask(patch != 0, connectivity=4)) != 1:
        return None
    return patch


def _assemble_groups(groups, h, w):
    out = np.zeros((h, w), dtype=int)
    ordered = sorted(groups, key=lambda group: group["patch"].shape[0] * group["patch"].shape[1], reverse=True)
    for group in ordered:
        try:
            gr, gc = random_free_location(
                out,
                group["patch"],
                transparent=0,
                border=OUTPUT_BORDER,
                padding=OUTPUT_GROUP_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(out, group["patch"], gr, gc, transparent=0)
        for node in group["nodes"]:
            node["pos"] = (gr + node["top"], gc + node["left"])

    if len(components_mask(out != 0, connectivity=4)) != len(groups):
        return None
    return out


def _make_distractor(shape, color):
    h, w = shape
    arr = np.full((h, w), color, dtype=int)

    if random.random() < 0.65:
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


def _make_distractors(groups):
    host_arrays = [node["arr"] for group in groups for node in group["nodes"]]
    group_colors = [group["color"] for group in groups]
    distractors = []
    n_distractors = random.randint(*DISTRACTOR_COUNT_RANGE)
    for _ in range(n_distractors):
        made = None
        for _attempt in range(DISTRACTOR_ATTEMPTS):
            shape = (
                random.randint(*DISTRACTOR_H_RANGE),
                random.randint(*DISTRACTOR_W_RANGE),
            )
            color = random.choice(group_colors if random.random() < 0.75 else PIECE_COLORS)
            arr = _make_distractor(shape, color)
            if arr is None:
                continue
            if any(_can_attach_to_host(arr, host_arr) for host_arr in host_arrays):
                continue
            made = {"arr": arr, "color": color, "shape": shape}
            break
        if made is None:
            return None
        distractors.append(made)
    return distractors


def _scatter(groups, distractors, h, w):
    inp = np.zeros((h, w), dtype=int)
    for group in groups:
        anchor = group["nodes"][0]
        paste(inp, anchor["arr"], *anchor["pos"], transparent=0)

    rest = [node["arr"] for group in groups for node in group["nodes"][1:]]
    rest.extend(d["arr"] for d in distractors)
    random.shuffle(rest)
    for arr in rest:
        try:
            r, c = random_free_location(
                inp,
                arr,
                transparent=0,
                border=1,
                padding=SCATTER_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inp, arr, r, c, transparent=0)
    return inp


def _sample_fusion_metrics(inp: np.ndarray, meta: dict) -> dict:
    comps_in = components_mask(inp != 0, connectivity=4)
    return {
        "anchor_count": int(meta["anchor_count"]),
        "distractor_count": int(meta["distractor_count"]),
        "input_components": int(len(comps_in)),
        "real_piece_count": int(sum(meta["group_sizes"])),
    }


def _gen_attempt():
    global _LAST_GENERATED_META

    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    anchor_count = random.choice(ANCHOR_COUNT_CHOICES)
    group_sizes = _choose_group_sizes(anchor_count)
    group_colors = random.sample(PIECE_COLORS, anchor_count)
    blocks = _choose_anchor_blocks(anchor_count)

    groups = []
    for block, n_pieces, color in zip(blocks, group_sizes, group_colors):
        nodes = _grow_group(block, n_pieces, color)
        if nodes is None:
            return None
        patch = _render_group(nodes)
        if patch is None:
            return None
        groups.append({"nodes": nodes, "patch": patch, "size": n_pieces, "color": color})

    out = _assemble_groups(groups, h, w)
    if out is None:
        return None

    distractors = _make_distractors(groups)
    if distractors is None:
        return None

    inp = _scatter(groups, distractors, h, w)
    if inp is None:
        return None

    if len(components_mask(out != 0, connectivity=4)) != anchor_count:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    meta = {
        "anchor_count": anchor_count,
        "group_sizes": tuple(group_sizes),
        "distractor_count": len(distractors),
    }
    _LAST_GENERATED_META = _sample_fusion_metrics(inp, meta)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cbebaa4b combo input")
