# Variant of cbebaa4b
# Dimension: D7 — 多步迭代组装策略 (multi_step)
# Change: 步骤1:识别中央黄色方块为锚点；步骤2:找到红色像素可与锚点红色像素对齐的精灵；步骤3:连接精灵；步骤4:重复直到所有精... → 链式传递组装（精灵接力连接）

from __future__ import annotations

from pathlib import Path
import sys
import random

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(_ROOT / "solvers"))

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis for the D7 chaining variant
# -------------------------------------------
# spec:
#   preserved:
#     - black background, one yellow anchor, red connector pixels, pure translations only
#     - output-first generation: build solved assembly first, then scatter pieces for the input
#     - non-red overlap remains illegal; only red-on-red connector overlap is allowed
#   changed_core:
#     - topology is no longer an arbitrary tree around the anchor
#     - the anchor has exactly one connector group (two adjacent red pixels)
#     - pieces form a simple path: anchor -> piece1 -> piece2 -> ... -> pieceN
#     - every non-final piece has two connector groups: one already attached, one still open
#   concrete_behavior_trace:
#     - original: after attaching one sprite, the DFS may choose any unpaired red on the whole
#       assembled object, so the next sprite can branch off a hub near the anchor
#     - variant: after attaching one sprite, only the two red cells belonging to the current
#       chain tail remain open; the next sprite must cover exactly that pair
#     - example: if piece A first overlaps the anchor's two reds, piece B must overlap A's other
#       two reds, never the anchor, because the anchor now has no count==1 red cells left
#   complexity_calibration:
#     - keeping 4-6 pieces preserves multi-step assembly depth; the variation is visible because
#       the solved output is a relay chain with one terminal piece instead of a star/tree
#     - connector groups are restricted to size 2 and piece degree to {1,2}; this keeps the
#       search state compact and visually legible without introducing hidden metadata
#   implicit_assumptions_checked:
#     - the original solver treated every count==1 red coordinate as equivalent expansion front;
#       that assumption breaks here, so the new solver tracks a single active tail group instead
#     - the original generator allowed anchor degree 1 or 2 and arbitrary parents; that would
#       silently reintroduce star layouts, so generation is restricted to a path grown only from
#       the most recently added node
#     - red pixels are extracted as 4-connected groups of length 2; the generator guarantees this
#       by placing ports away from corners, so each connector group stays separable
#   placement_legality:
#     - assembled placement requires each new node bbox to overlap only its predecessor bbox and
#       no earlier non-parent bbox
#     - final scattered input keeps a Chebyshev gap so solver-side component extraction is stable
#   visual_observability:
#     - humans can spot the rule from examples because the solved output always has exactly one
#       anchor-side junction and one free-end piece; intermediate pieces visibly relay the chain
#   cleanliness:
#     - each piece remains a single colored sprite with at most two red connector groups, so the
#       output does not collapse into dense multi-branch overlaps near the anchor


def _sorted_group(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def _find_red_groups(arr: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    groups = []
    for pts in components_mask(arr == 2, connectivity=4):
        coords = _sorted_group([(int(r), int(c)) for r, c in pts])
        groups.append(coords)
    groups.sort()
    return groups


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        red_groups = _find_red_groups(arr)
        pieces.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "reds": [(int(r), int(c)) for r, c in np.argwhere(arr == 2)],
                "red_groups": red_groups,
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


def _match_group_translations(
    local_group: tuple[tuple[int, int], ...], target_group: tuple[tuple[int, int], ...]
) -> list[tuple[int, int]]:
    target = set(target_group)
    out: set[tuple[int, int]] = set()
    for lr, lc in local_group:
        for tr, tc in target_group:
            top = int(tr - lr)
            left = int(tc - lc)
            shifted = {(int(r + top), int(c + left)) for r, c in local_group}
            if shifted == target:
                out.add((top, left))
    return sorted(out)


def _abs_group(group: tuple[tuple[int, int], ...], top: int, left: int) -> tuple[tuple[int, int], ...]:
    return _sorted_group([(r + top, c + left) for r, c in group])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered sprites into a relay chain that starts at the yellow anchor.

    Input structure:
    - bg = black (0). Several colored sprites are scattered without touching.
    - Exactly one anchor sprite contains a yellow (4) rectangle and one red connector pair.
    - Every non-anchor sprite has one or two red connector pairs on its boundary.

    Rule:
    - The anchor remains fixed and starts the chain.
    - The first sprite overlaps the anchor's red pair; after that, each next sprite may attach
      only to the red pair left open by the most recently attached sprite.
    - Continue relaying the open connector forward until all sprites are consumed and every red
      cell is paired exactly once.

    Key insight: The active attachment site is not "any open red in the assembly" but the unique
    current chain tail, so the state must remember which connector pair was opened last.
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

    assert len(anchor["red_groups"]) == 1, "chain variant requires exactly one anchor connector pair"
    for p in rest:
        assert len(p["red_groups"]) in (1, 2), "non-anchor pieces must have one or two connector pairs"
        for grp in p["red_groups"]:
            assert len(grp) == 2, "each connector must be a 2-cell red group"

    canvas = np.full((h, w), bg, dtype=int)
    ar0, ac0, _, _ = anchor["bbox"]
    placed = _try_place(canvas, anchor["arr"], ar0, ac0, bg)
    assert placed is not None, "anchor piece must fit on canvas at its original position"

    total_ports = int(sum(len(p["reds"]) for p in pieces))
    assert total_ports % 2 == 0, "total red connector ports must be even for complete pairing"
    target_unique_red = total_ports // 2

    red_counts: dict[tuple[int, int], int] = {}
    for rr, cc in anchor["reds"]:
        key = (int(ar0 + rr), int(ac0 + cc))
        red_counts[key] = red_counts.get(key, 0) + 1

    tail = _abs_group(anchor["red_groups"][0], ar0, ac0)
    solutions: list[np.ndarray] = []

    def dfs(
        cur: np.ndarray,
        remaining: list[dict],
        rc: dict[tuple[int, int], int],
        active_tail: tuple[tuple[int, int], ...] | None,
    ) -> None:
        if len(solutions) >= 2:
            return

        if not remaining:
            if active_tail is None and len(rc) == target_unique_red and all(v == 2 for v in rc.values()):
                solutions.append(cur)
            return

        if active_tail is None or len(rc) > target_unique_red:
            return

        moves = []
        for idx, p in enumerate(remaining):
            for red_group in p["red_groups"]:
                for top, left in _match_group_translations(red_group, active_tail):
                    ph, pw = p["shape"]
                    if top < 0 or left < 0 or top + ph > h or left + pw > w:
                        continue
                    bad = False
                    for rr, cc in p["reds"]:
                        gr, gc = top + rr, left + cc
                        if rc.get((int(gr), int(gc)), 0) >= 2:
                            bad = True
                            break
                    if bad:
                        continue
                    moves.append((idx, top, left))

        dedup = sorted(set(moves), key=lambda x: (-remaining[x[0]]["area"], x[1], x[2], x[0]))
        for idx, top, left in dedup:
            p = remaining[idx]
            next_remaining = remaining[:idx] + remaining[idx + 1 :]
            nxt = _try_place(cur, p["arr"], top, left, bg)
            if nxt is None:
                continue

            next_rc = dict(rc)
            for rr, cc in p["reds"]:
                gr, gc = int(top + rr), int(left + cc)
                next_rc[(gr, gc)] = next_rc.get((gr, gc), 0) + 1

            unpaired = _sorted_group([k for k, v in next_rc.items() if v == 1])
            if next_remaining:
                if len(unpaired) != 2:
                    continue
                next_tail = unpaired
            else:
                if unpaired:
                    continue
                next_tail = None

            dfs(nxt, next_remaining, next_rc, next_tail)
            if len(solutions) >= 2:
                return

    dfs(placed, rest, red_counts, tail)
    assert len(solutions) == 1, "chain assembly must have exactly one valid solution"
    return solutions[0]


# --------------- generator (D7 chaining variant) ---------------

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
ANCHOR_CAP = 1

# Scatter / generation
SCATTER_PADDING = 2
GEN_RETRY_LIMIT = 400

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
    arr = np.zeros(node["shape"], dtype=int)
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


def _grow_chain():
    bh, bw = random.choice(ANCHOR_BLOCK_CHOICES)
    anchor = _new_node("anchor", bh + 2, bw + 2, 4, cap=ANCHOR_CAP)
    anchor["block"] = (bh, bw)
    nodes = [anchor]
    n_pieces = random.randint(*PIECE_COUNT_RANGE)
    colors = random.sample(PIECE_COLORS, n_pieces)

    parent_i = 0
    for i in range(n_pieces):
        parent = nodes[parent_i]
        placed = False
        dims = [
            (random.randint(*PIECE_H_RANGE), random.randint(*PIECE_W_RANGE))
            for _ in range(PIECE_DIM_CANDIDATES)
        ]
        random.shuffle(dims)
        for ph, pw in dims:
            sides = [s for s in SIDES if s not in parent["sides"]]
            random.shuffle(sides)
            for side in sides:
                child = _new_node("piece", ph, pw, colors[i], cap=PIECE_CAP)
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
                parent_i = len(nodes) - 1
                placed = True
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
    nodes = _grow_chain()
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
    raise RuntimeError("failed to generate a valid chained cbebaa4b input")
