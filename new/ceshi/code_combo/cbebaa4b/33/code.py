from __future__ import annotations

# Combo variant of cbebaa4b (index 33)
# combo_spec: D2/1+D7/2+D8/0
# Name: 方向指示精灵 + 颜色编码优先级 + 链式传递

"""
Design analysis for the fused combo variant.

combo_rule:
  overview:
    - Keep the original black background, one fixed yellow anchor, and translation-only assembly.
    - Replace the old tree/radial growth with a simple relay chain.
    - Give every non-anchor piece exactly one direction-marked incoming connector side.
    - Color the connector pairs themselves with a visible priority code:
      red starts the chain, blue carries the middle relay steps, green closes the chain.
  concrete_behavior_trace:
    - Small example:
      anchor has one red connector pair.
      piece A has a cyan-backed red incoming pair and a blue outgoing pair.
      piece B has a cyan-backed blue incoming pair and a green outgoing pair.
      piece C has a cyan-backed green incoming pair and no outgoing pair.
    - Original behavior:
      any piece could try any red boundary pair against any currently open red in the growing assembly.
    - Fused behavior:
      step 1 can only use A, because the anchor exposes a red tail and A is the only piece whose
      marked incoming side is also red.
      after A attaches, the only live tail is A's blue pair, so B-like middle pieces become eligible.
      after the penultimate relay exposes green, only the green-marked terminal piece may close the chain.
    - Difference:
      direction pixels choose which side is the parent-facing side of a piece,
      connector colors choose when that side is legal,
      and chaining chooses where it is legal.
      None of the three cues is redundant with the other two.
  complexity_calibration:
    - The direction marker cannot reuse green anymore, because green now means "late-priority connector".
      A separate cyan marker keeps the two semantics visually separable.
    - The chain uses 4-6 pieces so the blue middle-priority class repeats; this prevents the color code
      from collapsing into a trivial one-off decoration.
    - Priority uses only three stable, human-visible classes (red/blue/green) instead of a hidden numeric order.
  implicit_assumption_audit:
    - Original `_try_place` allowed same-color overlap, which becomes unsafe once cyan direction markers
      repeat across many pieces. The fused solver permits overlap only for equal connector colors.
    - Original extraction only needed "all red pixels". The fused solver must decode connector side,
      connector color, and the unique cyan-backed incoming side for each non-anchor piece.
    - Original DFS assumed any open connector coordinate was a valid frontier. The fused solver instead
      tracks one active tail group plus its color.
    - Original generator could grow a tree from arbitrary parents. The fused generator must grow only
      from the newest node, otherwise blue middle pieces would not form a relay chain.
  placement_legality:
    - Output is still generated first, then scattered.
    - Each new bbox may overlap only its immediate parent bbox, and only on the matched 2-cell connector pair.
    - Ports are placed away from corners, so every side-level connector remains a clean 2-cell boundary strip.
  visual_observability:
    - Across examples, humans can see one fixed yellow anchor, cyan-marked incoming sides on all mobile pieces,
      and connector colors that progress from red near the anchor, through blue relay links, to green at closure.
    - The repeated blue middle class makes the "priority stage" idea visible instead of looking like arbitrary recoloring.
  cleanliness:
    - Cyan markers stay directly behind the incoming connector pair, so no detached clutter is added.
    - Output remains a thin chain with one live frontier at each intermediate step, avoiding dense overlaps near the anchor.
  inferability:
    - Color meaning is coupled to structure: red is always the anchor-adjacent join, blue is always a relay-stage join,
      and green is always the final closure stage.
    - Because the color cue is attached to the same 2-cell connector motif on every piece, humans can infer the mapping
      from 2-4 examples without relying on raw color ids.
  new_main_assumptions_checked_against_generator:
    - The anchor exposes exactly one red connector pair.
    - Every non-anchor piece has exactly one cyan-backed incoming connector pair and at most one extra outgoing pair.
    - Every connector pair has exactly two boundary cells of a single color from {blue, red, green}.
    - If a piece is not terminal, its unmarked outgoing pair is the unique next chain tail after placement.
"""

from pathlib import Path
import random
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(_ROOT))

from arc_utils import bbox_coords, components_mask, paste, random_free_location


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
DIR = 8

CONNECTOR_COLORS = (BLUE, RED, GREEN)
BODY_COLORS = (5, 6, 7, 9)

SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}


def _sorted_group(
    coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]
) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def _connector_mask(arr: np.ndarray) -> np.ndarray:
    return np.isin(arr, CONNECTOR_COLORS)


def _side_connectors(arr: np.ndarray) -> dict[str, dict]:
    h, w = arr.shape
    scans = {
        "top": [(0, c) for c in range(w) if int(arr[0, c]) in CONNECTOR_COLORS],
        "right": [(r, w - 1) for r in range(h) if int(arr[r, w - 1]) in CONNECTOR_COLORS],
        "bottom": [(h - 1, c) for c in range(w) if int(arr[h - 1, c]) in CONNECTOR_COLORS],
        "left": [(r, 0) for r in range(h) if int(arr[r, 0]) in CONNECTOR_COLORS],
    }
    out: dict[str, dict] = {}
    for side, cells in scans.items():
        if not cells:
            continue
        coords = _sorted_group(cells)
        assert len(coords) == 2, "each connector side must be a 2-cell boundary group"
        color = int(arr[coords[0]])
        assert all(int(arr[r, c]) == color for r, c in coords), "connector group must use one color"
        out[side] = {"cells": coords, "color": color}
    return out


def _decode_incoming_side(arr: np.ndarray, side_connectors: dict[str, dict]) -> str:
    h, w = arr.shape
    candidates: list[str] = []
    for side, info in side_connectors.items():
        dr, dc = INWARD[side]
        support = [(r + dr, c + dc) for r, c in info["cells"]]
        if all(0 <= rr < h and 0 <= cc < w and int(arr[rr, cc]) == DIR for rr, cc in support):
            candidates.append(side)
    assert len(candidates) == 1, "each non-anchor piece must have exactly one direction-marked incoming side"
    return candidates[0]


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        side_connectors = _side_connectors(arr)
        is_anchor = bool(np.any(arr == YELLOW))
        incoming_side = None if is_anchor else _decode_incoming_side(arr, side_connectors)
        outgoing_sides = [s for s in side_connectors if s != incoming_side]
        incoming_group = tuple() if incoming_side is None else side_connectors[incoming_side]["cells"]
        incoming_color = None if incoming_side is None else int(side_connectors[incoming_side]["color"])
        outgoing_side = outgoing_sides[0] if len(outgoing_sides) == 1 else None
        outgoing_group = tuple() if outgoing_side is None else side_connectors[outgoing_side]["cells"]
        outgoing_color = None if outgoing_side is None else int(side_connectors[outgoing_side]["color"])
        ports = [(int(r), int(c)) for r, c in np.argwhere(_connector_mask(arr))]
        pieces.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "shape": (int(arr.shape[0]), int(arr.shape[1])),
                "area": int(np.sum(arr != bg)),
                "is_anchor": is_anchor,
                "side_connectors": side_connectors,
                "incoming_side": incoming_side,
                "incoming_group": incoming_group,
                "incoming_set": set(incoming_group),
                "incoming_color": incoming_color,
                "outgoing_side": outgoing_side,
                "outgoing_group": outgoing_group,
                "outgoing_color": outgoing_color,
                "ports": ports,
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

    allow_overlap = _connector_mask(arr) & _connector_mask(sub) & (arr == sub)
    conflict = mask & (sub != bg) & ~allow_overlap
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


def _abs_group(
    group: tuple[tuple[int, int], ...], top: int, left: int
) -> tuple[tuple[int, int], ...]:
    return _sorted_group([(r + top, c + left) for r, c in group])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered sprites into a color-prioritized relay chain around the fixed yellow anchor.

    Input structure:
    - bg = black (0). Several disconnected sprites are scattered with gaps between them.
    - One sprite is the yellow (4) anchor and it exposes exactly one red connector pair.
    - Every non-anchor sprite has a cyan (8) direction strip behind exactly one connector pair,
      and connector pairs themselves are colored red (2), blue (1), or green (3).

    Rule:
    - Translate pieces only; do not rotate or mirror them.
    - A piece may join the assembly only through its cyan-backed incoming connector side.
    - The active chain tail determines both legality conditions: the incoming side must overlap
      the current tail coordinates exactly, and its connector color must match the current tail color.
      After attachment, the piece's other connector pair becomes the new tail; terminal pieces have no outgoing pair.

    Key insight: The task is solved by fusing three cues into one state: cyan marks which side of a piece faces its parent,
    connector color says at which stage that side becomes legal, and the chain topology says there is only one live tail to match.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = BG

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must have at least one colored piece"

    anchors = [p for p in pieces if p["is_anchor"]]
    assert len(anchors) == 1, "input must have exactly one yellow anchor piece"
    anchor = anchors[0]
    rest = [p for p in pieces if not p["is_anchor"]]

    assert len(anchor["side_connectors"]) == 1, "anchor must expose exactly one connector pair"
    anchor_side = next(iter(anchor["side_connectors"].values()))
    assert anchor_side["color"] == RED, "anchor connector color must be red"

    for p in rest:
        assert len(p["side_connectors"]) in (1, 2), "each non-anchor piece must have one or two connector pairs"
        assert len(p["incoming_group"]) == 2, "each non-anchor piece must have a marked 2-cell incoming connector"
        assert p["incoming_color"] in CONNECTOR_COLORS, "incoming connector must use a priority color"
        if p["outgoing_side"] is not None:
            assert len(p["outgoing_group"]) == 2, "outgoing connector must also be a 2-cell group"
            assert p["outgoing_color"] in CONNECTOR_COLORS, "outgoing connector must use a priority color"

    canvas = np.full((h, w), bg, dtype=int)
    ar0, ac0, _, _ = anchor["bbox"]
    placed = _try_place(canvas, anchor["arr"], ar0, ac0, bg)
    assert placed is not None, "anchor piece must fit on canvas at its original position"

    total_ports = int(sum(len(p["ports"]) for p in pieces))
    assert total_ports % 2 == 0, "total connector pixels must be even for complete pairing"
    target_unique_ports = total_ports // 2

    port_counts: dict[tuple[int, int], int] = {}
    for rr, cc in anchor_side["cells"]:
        key = (int(ar0 + rr), int(ac0 + cc))
        port_counts[key] = port_counts.get(key, 0) + 1

    active_tail = _abs_group(anchor_side["cells"], ar0, ac0)
    active_color = RED
    solutions: list[np.ndarray] = []

    def dfs(
        cur: np.ndarray,
        remaining: list[dict],
        pc: dict[tuple[int, int], int],
        tail: tuple[tuple[int, int], ...] | None,
        tail_color: int | None,
    ) -> None:
        if len(solutions) >= 2:
            return

        if not remaining:
            if tail is None and len(pc) == target_unique_ports and all(v == 2 for v in pc.values()):
                solutions.append(cur)
            return

        if tail is None or tail_color is None or len(pc) > target_unique_ports:
            return

        moves: list[tuple[int, int, int]] = []
        for idx, p in enumerate(remaining):
            if p["incoming_color"] != tail_color:
                continue
            for top, left in _match_group_translations(p["incoming_group"], tail):
                ph, pw = p["shape"]
                if top < 0 or left < 0 or top + ph > h or left + pw > w:
                    continue
                bad = False
                for rr, cc in p["ports"]:
                    gr, gc = int(top + rr), int(left + cc)
                    prev = pc.get((gr, gc), 0)
                    if (rr, cc) in p["incoming_set"]:
                        if prev != 1:
                            bad = True
                            break
                    elif prev != 0:
                        bad = True
                        break
                if not bad:
                    moves.append((idx, top, left))

        dedup = sorted(set(moves), key=lambda x: (-remaining[x[0]]["area"], x[1], x[2], x[0]))
        for idx, top, left in dedup:
            p = remaining[idx]
            next_remaining = remaining[:idx] + remaining[idx + 1 :]
            nxt = _try_place(cur, p["arr"], top, left, bg)
            if nxt is None:
                continue

            next_pc = dict(pc)
            for rr, cc in p["ports"]:
                key = (int(top + rr), int(left + cc))
                next_pc[key] = next_pc.get(key, 0) + 1

            unpaired = _sorted_group([k for k, v in next_pc.items() if v == 1])
            if p["outgoing_side"] is None:
                if unpaired:
                    continue
                next_tail = None
                next_color = None
            else:
                next_tail = _abs_group(p["outgoing_group"], top, left)
                if unpaired != next_tail:
                    continue
                next_color = p["outgoing_color"]

            dfs(nxt, next_remaining, next_pc, next_tail, next_color)
            if len(solutions) >= 2:
                return

    dfs(placed, rest, port_counts, active_tail, active_color)
    assert len(solutions) == 1, "fused chain assembly must have exactly one valid solution"
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
PIECE_DIM_CANDIDATES = 14
BAR_SKIP_PROB = 0.35

# Anchor / scatter / generation
ANCHOR_BLOCK_CHOICES = ((3, 3), (4, 5))
SCATTER_PADDING = 2
GEN_RETRY_LIMIT = 600


def _new_node(kind, h, w, body_color, top=0, left=0):
    return {
        "kind": kind,
        "shape": (int(h), int(w)),
        "body_color": int(body_color),
        "top": int(top),
        "left": int(left),
        "sides": set(),
        "ports": {},
        "side_colors": {},
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
    return random.choice([(r, c) for r in range(1, h - 1) for c in range(1, w - 1)])


def _make_piece(node):
    h, w = node["shape"]
    body = node["body_color"]
    used = set(node["sides"])
    arr = np.zeros((h, w), dtype=int)
    hub = _pick_hub(h, w)

    bar_sides = [s for s in SIDES if s not in used]
    if len(bar_sides) > 1 and random.random() < BAR_SKIP_PROB:
        bar_sides.remove(random.choice(bar_sides))
    for side in bar_sides:
        _paint_bar(arr, side, body, used)

    terminals = []
    for side in used:
        dr, dc = INWARD[side]
        conn_color = node["side_colors"][side]
        marker_side = side == node["incoming"]
        for r, c in _port_cells(node, side):
            arr[r, c] = conn_color
            rr, cc = r + dr, c + dc
            arr[rr, cc] = DIR if marker_side else body
            terminals.append((rr, cc))

    if arr[hub] == BG:
        arr[hub] = body
    order = random.randint(0, 1)
    for t in terminals:
        _draw_path(arr, t, hub, body, order)
    for side in bar_sides:
        _draw_path(arr, hub, _bar_endpoint((h, w), side, hub), body, 1 - order)

    if len(components_mask(arr != BG, connectivity=4)) != 1:
        return None
    if int(np.sum(_connector_mask(arr))) != 2 * len(used):
        return None
    if int(np.sum(arr == DIR)) != 2:
        return None
    return arr


def _make_anchor(node):
    bh, bw = node["block"]
    h, w = node["shape"]
    arr = np.zeros((h, w), dtype=int)
    arr[1 : 1 + bh, 1 : 1 + bw] = YELLOW
    for side in node["sides"]:
        conn_color = node["side_colors"][side]
        for r, c in _port_cells(node, side):
            arr[r, c] = conn_color
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


def _priority_sequence(n_pieces: int) -> list[int]:
    assert n_pieces >= 3
    return [RED] + [BLUE] * (n_pieces - 2) + [GREEN]


def _grow_chain():
    bh, bw = random.choice(ANCHOR_BLOCK_CHOICES)
    anchor = _new_node("anchor", bh + 2, bw + 2, YELLOW)
    anchor["block"] = (bh, bw)
    nodes = [anchor]

    n_pieces = random.randint(*PIECE_COUNT_RANGE)
    priorities = _priority_sequence(n_pieces)
    body_colors = [random.choice(BODY_COLORS) for _ in range(n_pieces)]

    parent_i = 0
    for i in range(n_pieces):
        parent = nodes[parent_i]
        needed_color = priorities[i]
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
                child = _new_node("piece", ph, pw, body_colors[i])
                incoming = OPPOSITE[side]
                child["incoming"] = incoming
                child["sides"].add(incoming)
                child["side_colors"][incoming] = needed_color
                _ensure_port(parent, side)
                _ensure_port(child, incoming)
                pr, pc = _global_port(parent, side)[0]
                cr, cc = _port_cells(child, incoming)[0]
                child["top"] = pr - cr
                child["left"] = pc - cc
                if not _candidate_fits(nodes, parent_i, child):
                    continue
                parent["sides"].add(side)
                parent["side_colors"][side] = needed_color
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

    mask = arr != BG
    rows, cols = np.nonzero(mask)
    g_rows, g_cols = rows + top, cols + left
    vals = arr[rows, cols]
    cur_vals = out[g_rows, g_cols]

    allow_overlap = np.isin(vals, CONNECTOR_COLORS) & np.isin(cur_vals, CONNECTOR_COLORS) & (vals == cur_vals)
    conflicts = (cur_vals != BG) & ~allow_overlap
    if np.any(conflicts):
        return None

    port_counts = {}
    port_mask = np.isin(vals, CONNECTOR_COLORS)
    for gr, gc in zip(g_rows[port_mask], g_cols[port_mask]):
        key = (int(gr), int(gc))
        port_counts[key] = port_counts.get(key, 0) + 1

    paint_mask = cur_vals == BG
    out[g_rows[paint_mask], g_cols[paint_mask]] = vals[paint_mask]
    return port_counts


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
    all_port_counts = {}
    for node in nodes:
        rc = _place_node_on_canvas(out, node, sr, sc)
        if rc is None:
            return None
        for k, v in rc.items():
            all_port_counts[k] = all_port_counts.get(k, 0) + v

    total_ports = sum(int(np.sum(_connector_mask(n["arr"]))) for n in nodes)
    if len(all_port_counts) * 2 != total_ports:
        return None
    if any(v != 2 for v in all_port_counts.values()):
        return None
    if len(components_mask(out != BG, connectivity=4)) != 1:
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
    raise RuntimeError("failed to generate a valid fused cbebaa4b input")
