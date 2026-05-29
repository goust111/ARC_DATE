# Variant of cbebaa4b
# Dimension: D2 — 源对象的结构与属性 (src)
# Change: 散布在背景上的彩色精灵，每个精灵包含主体颜色像素和红色连接器像素，形状不规则... → 精灵带有洞（内部空缺）

# Design analysis for this D2 variant
# -----------------------------------
# spec_variant:
#   preserved_core:
#     - same black background, same fixed yellow anchor, same translation-only assembly
#     - red pixels are still the attachment markers; every red pixel must still be paired exactly once
#     - output is still the fully assembled connected structure
#   src_change:
#     - non-anchor piece-to-piece links are no longer plain boundary touches
#     - when a non-anchor piece becomes a parent in the tree, its outward connector is rendered as a slot:
#         two red pixels plus a 2-cell internal cavity directly behind them
#     - the child attached there gets a matching tab:
#         the same two red pixels plus a 2-cell protrusion on the outside of the connector
#     - after assembly, the child's tab occupies the parent's cavity cells, so input holes visibly disappear in output
#   concrete_behavior_trace:
#     - original on one edge:
#         parent red pair and child red pair overlap; bodies merely meet at the connector line
#     - variant on one edge:
#         parent has red pair + internal slit hole, child has red pair + outward tongue
#         after translation, the red pair overlaps as before, and the tongue lands inside the slit hole
#     - difference:
#         the solver cannot stop at "all reds paired"; it must also reject assemblies where any hole stays black
#   complexity_calibration:
#     - every non-anchor internal link uses exactly one of two visible states: plain (anchor links) or slot/tab (piece links)
#     - slot size is fixed to 2 cells, repeated across samples, so humans can spot the pattern from a few examples
#     - colors are still unique per non-anchor piece, avoiding accidental same-color overlap ambiguities
#   implicit_assumption_audit_main:
#     - original _try_place assumed useful overlap is only red-on-red; this still holds
#     - original DFS accepted any final board with all red counts == 2; this is no longer sufficient because holes may remain unfilled
#     - original extraction ignored internal background cells; variant needs them, so each piece now stores interior-hole coords
#     - original move ordering used only red overlap count and area; variant also benefits from preferring moves that fill known holes
#   placement_legality:
#     - generator still builds the assembled board first, then scatters pieces
#     - tabs only occupy cells that are black inside the matched slot piece, so no non-red conflicts are introduced
#     - bbox-overlap is still restricted to parent-child relations during assembly construction
#   visual_observability:
#     - in the input, some pieces show obvious 2-cell black slits inside their colored outline
#     - in the output, those slits are replaced by the neighboring piece's color
#     - anchor links remain plain, so the new behavior is localized to piece-piece interfaces and easy to compare
#   cleanliness:
#     - only one interaction layer is used: red overlap plus black-hole filling
#     - no sorting or branching depends on invisible color ids; geometry alone exposes the rule

from __future__ import annotations

import os
import sys
from collections import deque
import random

import numpy as np

sys.path.append("/home/lixinyu/snap/new/solvers")
from arc_utils import bbox_coords, components_mask, paste, random_free_location


def _interior_holes(arr: np.ndarray, bg: int) -> list[tuple[int, int]]:
    """Background cells inside the piece bbox that do not touch the bbox border."""
    h, w = arr.shape
    holes: list[tuple[int, int]] = []
    for comp in components_mask(arr == bg, connectivity=4):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            continue
        holes.extend((int(r), int(c)) for r, c in comp)
    return holes


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for idx, pts in enumerate(comps):
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        reds = np.argwhere(arr == 2)
        pieces.append(
            {
                "id": int(idx),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "reds": [(int(r), int(c)) for r, c in reds],
                "holes": _interior_holes(arr, bg),
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

    conflict = mask & (sub != bg) & ~((arr == 2) & (sub == 2))
    if np.any(conflict):
        return None

    out = canvas.copy()
    out_sub = out[top : top + ph, left : left + pw]
    out_sub[mask] = arr[mask]
    out[top : top + ph, left : left + pw] = out_sub
    return out


def _holes_filled(board: np.ndarray, placed: dict[int, tuple[dict, int, int]], bg: int) -> bool:
    for piece, top, left in placed.values():
        for rr, cc in piece["holes"]:
            if int(board[top + rr, left + cc]) == bg:
                return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered sprites around the yellow anchor, including pieces whose internal holes must be plugged by neighbors.

    Input structure:
    - bg = black (0). Several colored sprites are scattered on the grid.
    - One sprite is the fixed yellow (4) anchor block.
    - Red (2) connector pairs still mark how pieces attach, but some non-anchor sprites also contain small black cavities inside their bbox.

    Rule:
    - Keep the anchor where it is and translate the other sprites onto it one by one.
    - A move is still driven by overlapping a piece's red connector with an unpaired red connector in the growing assembly.
    - The finished assembly is valid only if every red connector is paired and every internal cavity from the input pieces is filled by another piece's colored protrusion.

    Key insight: Red pixels still tell you which translations are candidates, but the new negative-space test removes false joins — the correct assembly is the one where tabs not only meet the red markers but also occupy the slot-shaped holes.
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
    target_unique_red = total_ports // 2

    red_counts: dict[tuple[int, int], int] = {}
    for rr, cc in anchor["reds"]:
        red_counts[(ar0 + rr, ac0 + cc)] = red_counts.get((ar0 + rr, ac0 + cc), 0) + 1

    placed_map: dict[int, tuple[dict, int, int]] = {anchor["id"]: (anchor, ar0, ac0)}
    solutions: list[np.ndarray] = []

    def dfs(
        cur: np.ndarray,
        remaining: list[dict],
        rc: dict[tuple[int, int], int],
        placed_now: dict[int, tuple[dict, int, int]],
    ) -> None:
        if len(solutions) >= 2:
            return

        if not remaining:
            if len(rc) == target_unique_red and all(v == 2 for v in rc.values()) and _holes_filled(cur, placed_now, bg):
                solutions.append(cur)
            return

        if len(rc) > target_unique_red:
            return

        unpaired = [k for k, v in rc.items() if v == 1]
        if not unpaired:
            return

        rc1 = {k for k, v in rc.items() if v == 1}
        best_moves = None
        for t in unpaired:
            moves = []
            tr, tc = t
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
                    if bad:
                        continue
                    moves.append((idx, top, left, overlap))
            if not moves:
                return
            if best_moves is None or len(moves) < len(best_moves):
                best_moves = moves
                if len(best_moves) == 1:
                    break

        assert best_moves is not None

        dedup: dict[tuple[int, int, int], int] = {}
        for idx, top, left, ov in best_moves:
            key = (idx, top, left)
            dedup[key] = max(dedup.get(key, 0), ov)

        scored = []
        for (idx, top, left), ov in dedup.items():
            p = remaining[idx]
            hole_fill_gain = 0
            for _, (pp, pr0, pc0) in placed_now.items():
                for hr, hc in pp["holes"]:
                    if pr0 + hr < top or pr0 + hr >= top + p["shape"][0] or pc0 + hc < left or pc0 + hc >= left + p["shape"][1]:
                        continue
                    rr = pr0 + hr - top
                    cc = pc0 + hc - left
                    if int(p["arr"][rr, cc]) != bg:
                        hole_fill_gain += 1
            scored.append((idx, top, left, ov, hole_fill_gain))
        scored.sort(key=lambda x: (-x[4], -x[3], -remaining[x[0]]["area"], x[1], x[2], x[0]))

        for idx, top, left, _ov, _fill in scored:
            p = remaining[idx]
            next_remaining = remaining[:idx] + remaining[idx + 1 :]

            next_rc = dict(rc)
            for rr, cc in p["reds"]:
                gr, gc = top + rr, left + cc
                next_rc[(gr, gc)] = next_rc.get((gr, gc), 0) + 1

            nxt = _try_place(cur, p["arr"], top, left, bg)
            if nxt is None:
                continue

            next_placed = dict(placed_now)
            next_placed[p["id"]] = (p, top, left)
            dfs(nxt, next_remaining, next_rc, next_placed)
            if len(solutions) >= 2:
                return

    dfs(placed, rest, red_counts, placed_map)
    assert len(solutions) == 1, "jigsaw assembly must have exactly one valid solution"
    return solutions[0]


# --------------- generator (D2 slot/tab variant) ---------------

GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
MAX_ASSEMBLY_DIM = 24

PIECE_COUNT_RANGE = (4, 6)
PIECE_H_RANGE = (6, 8)
PIECE_W_RANGE = (6, 8)
PIECE_DIM_CANDIDATES = 14
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]
PIECE_CAP = 2
DECOR_SKIP_PROB = 0.35

ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]
ANCHOR_CAP_CHOICES = [1, 2]

SCATTER_PADDING = 2
GEN_RETRY_LIMIT = 400

SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}


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
        "features": {},
    }


def _port_range(shape: tuple[int, int], side: str) -> tuple[int, int]:
    h, w = shape
    span = h if side in ("left", "right") else w
    if span >= 7:
        return 2, span - 4
    return 1, span - 3


def _ensure_port(node, side):
    if side in node["ports"]:
        return
    lo, hi = _port_range(node["shape"], side)
    assert lo <= hi
    node["ports"][side] = random.randint(lo, hi)


def _port_cells(node, side):
    h, w = node["shape"]
    k = node["ports"][side]
    if node["kind"] == "anchor":
        if side == "top":
            return [(0, k), (0, k + 1)]
        if side == "bottom":
            return [(h - 1, k), (h - 1, k + 1)]
        if side == "left":
            return [(k, 0), (k + 1, 0)]
        return [(k, w - 1), (k + 1, w - 1)]

    if side == "top":
        return [(1, k), (1, k + 1)]
    if side == "bottom":
        return [(h - 2, k), (h - 2, k + 1)]
    if side == "left":
        return [(k, 1), (k + 1, 1)]
    return [(k, w - 2), (k + 1, w - 2)]


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


def _neighbors4(shape: tuple[int, int], r: int, c: int):
    h, w = shape
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _find_path(arr: np.ndarray, start: tuple[int, int], goals: set[tuple[int, int]], blocked: set[tuple[int, int]]):
    shape = arr.shape
    q = deque([start])
    prev = {start: None}
    while q:
        cur = q.popleft()
        if cur in goals:
            out = []
            while cur is not None:
                out.append(cur)
                cur = prev[cur]
            return out[::-1]
        for nxt in _neighbors4(shape, *cur):
            if nxt in prev or nxt in blocked:
                continue
            rr, cc = nxt
            if int(arr[rr, cc]) == 2 and nxt not in goals:
                continue
            prev[nxt] = cur
            q.append(nxt)
    return None


def _midpoint_endpoint(shape: tuple[int, int], side: str) -> tuple[int, int]:
    h, w = shape
    if side == "top":
        return (1, w // 2)
    if side == "bottom":
        return (h - 2, w // 2)
    if side == "left":
        return (h // 2, 1)
    return (h // 2, w - 2)


def _add_side_feature(arr: np.ndarray, side: str, k: int, mode: str, color: int):
    h, w = arr.shape
    red = []
    solids = []
    terminals = []
    holes = []

    if side == "left":
        red = [(k, 1), (k + 1, 1)]
        if mode == "plain":
            solids = [(k, 2), (k + 1, 2)]
            terminals = solids[:]
        elif mode == "tab":
            solids = [(k, 0), (k + 1, 0), (k, 2), (k + 1, 2)]
            terminals = [(k, 2), (k + 1, 2)]
        else:
            holes = [(k, 2), (k + 1, 2)]
            solids = [(k - 1, 1), (k + 2, 1), (k - 1, 2), (k + 2, 2)]
            terminals = [(k - 1, 2), (k + 2, 2)]
    elif side == "right":
        red = [(k, w - 2), (k + 1, w - 2)]
        if mode == "plain":
            solids = [(k, w - 3), (k + 1, w - 3)]
            terminals = solids[:]
        elif mode == "tab":
            solids = [(k, w - 1), (k + 1, w - 1), (k, w - 3), (k + 1, w - 3)]
            terminals = [(k, w - 3), (k + 1, w - 3)]
        else:
            holes = [(k, w - 3), (k + 1, w - 3)]
            solids = [(k - 1, w - 2), (k + 2, w - 2), (k - 1, w - 3), (k + 2, w - 3)]
            terminals = [(k - 1, w - 3), (k + 2, w - 3)]
    elif side == "top":
        red = [(1, k), (1, k + 1)]
        if mode == "plain":
            solids = [(2, k), (2, k + 1)]
            terminals = solids[:]
        elif mode == "tab":
            solids = [(0, k), (0, k + 1), (2, k), (2, k + 1)]
            terminals = [(2, k), (2, k + 1)]
        else:
            holes = [(2, k), (2, k + 1)]
            solids = [(1, k - 1), (1, k + 2), (2, k - 1), (2, k + 2)]
            terminals = [(2, k - 1), (2, k + 2)]
    else:
        red = [(h - 2, k), (h - 2, k + 1)]
        if mode == "plain":
            solids = [(h - 3, k), (h - 3, k + 1)]
            terminals = solids[:]
        elif mode == "tab":
            solids = [(h - 1, k), (h - 1, k + 1), (h - 3, k), (h - 3, k + 1)]
            terminals = [(h - 3, k), (h - 3, k + 1)]
        else:
            holes = [(h - 3, k), (h - 3, k + 1)]
            solids = [(h - 2, k - 1), (h - 2, k + 2), (h - 3, k - 1), (h - 3, k + 2)]
            terminals = [(h - 3, k - 1), (h - 3, k + 2)]

    for rr, cc in solids:
        arr[rr, cc] = color
    for rr, cc in red:
        arr[rr, cc] = 2
    for rr, cc in holes:
        arr[rr, cc] = 0
    return red, terminals, holes


def _make_piece(node):
    h, w = node["shape"]
    color = node["color"]
    used = set(node["sides"])
    arr = np.zeros((h, w), dtype=int)

    hr0 = h // 2 - 1
    hc0 = w // 2 - 1
    hub_cells = {
        (hr0, hc0),
        (hr0 + 1, hc0),
        (hr0, hc0 + 1),
        (hr0 + 1, hc0 + 1),
    }
    for rr, cc in hub_cells:
        arr[rr, cc] = color

    terminals = []
    blocked: set[tuple[int, int]] = set()
    for side in used:
        mode = node["features"][side]
        _, ts, holes = _add_side_feature(arr, side, node["ports"][side], mode, color)
        terminals.extend(ts)
        blocked.update(holes)

    decor_sides = [s for s in SIDES if s not in used]
    if len(decor_sides) > 1 and random.random() < DECOR_SKIP_PROB:
        decor_sides.remove(random.choice(decor_sides))
    for side in decor_sides:
        ep = _midpoint_endpoint((h, w), side)
        arr[ep] = color
        terminals.append(ep)

    unique_terms = []
    seen = set()
    for t in terminals:
        if t not in seen:
            unique_terms.append(t)
            seen.add(t)

    random.shuffle(unique_terms)
    for start in unique_terms:
        path = _find_path(arr, start, hub_cells, blocked)
        if path is None:
            return None
        for rr, cc in path:
            if (rr, cc) not in blocked and arr[rr, cc] != 2:
                arr[rr, cc] = color

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

                    if parent["kind"] == "anchor":
                        parent["features"][side] = "plain"
                        child["features"][incoming] = "plain"
                    else:
                        parent["features"][side] = "slot"
                        child["features"][incoming] = "tab"

                    pr, pc = _global_port(parent, side)[0]
                    cr, cc = _port_cells(child, incoming)[0]
                    child["top"] = pr - cr
                    child["left"] = pc - cc
                    child["sides"].add(incoming)

                    if not _candidate_fits(nodes, parent_i, child):
                        parent["features"].pop(side, None)
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

    piece_piece_edges = sum(1 for n in nodes[1:] if any(v == "tab" for v in n["features"].values()))
    if piece_piece_edges < 2:
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


def _holes_filled_in_nodes(out, nodes):
    for node in nodes:
        top, left = node["pos"]
        for rr, cc in _interior_holes(node["arr"], 0):
            if int(out[top + rr, left + cc]) == 0:
                return False
    return True


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
    if not _holes_filled_in_nodes(out, nodes):
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
    raise RuntimeError("failed to generate a valid cbebaa4b variant input")
