# Combo variant of cbebaa4b (index 16)
# combo_spec: D1/5+D10/0
# Name: 连接器数量匹配 + 单连接器精灵

from __future__ import annotations

import random
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis for the fused variant
# a) Concrete behavior:
#    The seed puzzle used red-pixel overlap as the connector rule, and pieces could have
#    several reds. In this combo every movable piece has exactly one red pixel, so each
#    movable piece is forced to be a leaf. A small concrete example is:
#    - original: anchor top edge has two reds, a leaf piece bottom edge has two reds, and
#      the move is "translate until those two reds overlap";
#    - fused: anchor top edge has one red, a leaf piece bottom edge has one red, and the
#      move is "dock the piece above the anchor so the two single reds become adjacent".
#    This keeps the visual idea of connector docking but removes the original red-overlap
#    pairing logic. The count-matching dimension is still present, but it collapses to
#    1=1 on every used side because of the single-connector restriction.
# b) Complexity calibration:
#    If we let one-red pieces connect to each other arbitrarily, the generator would either
#    become impossible (a degree-2 internal node cannot have only one connector) or would
#    need hidden bookkeeping. To stay visible and robust, the assembled shape is a star:
#    the anchor exposes one connector on each used side and every non-anchor piece docks
#    directly to one side. We use all four sides so the rule is repeatedly demonstrated.
# c) Broken assumptions from the original main():
#    - "paired reds collapse onto the same coordinate" is no longer true;
#    - DFS over unpaired red coordinates is unnecessary because every piece has exactly one
#      legal attachment side;
#    - total red count no longer halves in the output, because the two touching red cells
#      stay as two cells instead of overlapping into one.
#    The new solver therefore switches from overlap-search to deterministic side docking.
# d) Placement legality:
#    The generator builds the output first, rejects overlaps/out-of-bounds layouts, and
#    also rejects accidental piece-to-piece touching so that every connection is clearly a
#    single red-to-red seam with the anchor.
# e) Visual observability:
#    Across examples the human-visible invariant is: each scattered piece has one red on
#    exactly one side, the anchor has one red on each used side, and the output docks each
#    piece to the side with the matching singleton count.
# f) Visual cleanliness:
#    Pieces use moderate sizes and are rejected if they touch each other in the assembled
#    output, which keeps the central yellow block readable and each seam isolated.
# g) Inferability:
#    No hidden numeric ordering or color-id semantics are used. Colors only distinguish
#    pieces; the deciding cue is the location of the lone red connector on each piece.


GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]
PIECE_COUNT = 4
PIECE_H_RANGE = (4, 7)
PIECE_W_RANGE = (4, 7)
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]
BAR_SKIP_PROB = 0.30
SCATTER_PADDING = 2
GEN_RETRY_LIMIT = 400

SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}
SIDE_ORDER = {side: i for i, side in enumerate(SIDES)}


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


def _side_from_red(shape: tuple[int, int], red: tuple[int, int]) -> str:
    h, w = shape
    r, c = red
    hits = []
    if r == 0:
        hits.append("top")
    if r == h - 1:
        hits.append("bottom")
    if c == 0:
        hits.append("left")
    if c == w - 1:
        hits.append("right")
    assert len(hits) == 1, "each connector must lie on exactly one non-corner side"
    return hits[0]


def _side_red_count(arr: np.ndarray, side: str) -> int:
    if side == "top":
        return int(np.sum(arr[0, :] == 2))
    if side == "bottom":
        return int(np.sum(arr[-1, :] == 2))
    if side == "left":
        return int(np.sum(arr[:, 0] == 2))
    return int(np.sum(arr[:, -1] == 2))


def _try_stamp(canvas: np.ndarray, arr: np.ndarray, top: int, left: int, bg: int = 0) -> np.ndarray | None:
    h, w = canvas.shape
    ph, pw = arr.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return None
    mask = arr != bg
    sub = canvas[top : top + ph, left : left + pw]
    if np.any(mask & (sub != bg)):
        return None
    out = canvas.copy()
    out[top : top + ph, left : left + pw][mask] = arr[mask]
    return out


def _adjacent_top_left(
    piece: dict, target_red: tuple[int, int], target_side: str
) -> tuple[int, int]:
    pr, pc = piece["reds"][0]
    tr, tc = target_red
    if target_side == "top":
        return tr - 1 - pr, tc - pc
    if target_side == "bottom":
        return tr + 1 - pr, tc - pc
    if target_side == "left":
        return tr - pr, tc - 1 - pc
    return tr - pr, tc + 1 - pc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered single-port sprites around the fixed yellow anchor by docking lone red connectors side-to-side.

    Input structure:
    - Background is black (0); one component contains the central yellow (4) block and stays fixed.
    - Every non-anchor sprite is a single 4-connected colored piece with exactly one red (2) pixel on one side.
    - The anchor exposes one red connector on each used side.

    Rule:
    - Keep the anchor in its original position.
    - For each scattered piece, read which side its single red lies on; it must dock to the opposite side of the anchor.
    - The matching condition is side connector count: piece side count 1 must match anchor side count 1.
    - Dock by translation only, with the two red pixels becoming orthogonally adjacent rather than overlapping.

    Key insight: Because every movable piece has exactly one connector, it can only be a leaf; the task is no longer red-overlap pairing but deterministic side docking based on the lone connector's side and the 1=1 side-count match.
    """
    g = np.asarray(grid, dtype=int)
    bg = 0
    h, w = g.shape

    pieces = _extract_pieces(g, bg)
    assert len(pieces) >= 2, "input must contain the anchor and at least one movable piece"

    anchors = [p for p in pieces if np.any(p["arr"] == 4)]
    assert len(anchors) == 1, "input must have exactly one yellow anchor component"
    anchor = anchors[0]
    rest = [p for p in pieces if p is not anchor]

    anchor_side_to_red: dict[str, tuple[int, int]] = {}
    for red in anchor["reds"]:
        side = _side_from_red(anchor["shape"], red)
        assert side not in anchor_side_to_red, "generator guarantees one anchor connector per used side"
        anchor_side_to_red[side] = red
    assert len(anchor_side_to_red) == len(rest), "single-port leaves require one anchor port per piece"
    for side in anchor_side_to_red:
        assert _side_red_count(anchor["arr"], side) == 1, "count match uses singleton anchor sides"

    canvas = np.zeros((h, w), dtype=int)
    ar0, ac0, _, _ = anchor["bbox"]
    cur = _try_stamp(canvas, anchor["arr"], ar0, ac0, bg)
    assert cur is not None, "anchor must fit in place"

    ordered = []
    for p in rest:
        assert len(p["reds"]) == 1, "each movable piece must have exactly one connector"
        piece_side = _side_from_red(p["shape"], p["reds"][0])
        target_side = OPPOSITE[piece_side]
        assert target_side in anchor_side_to_red, "each piece must have a corresponding anchor side"
        assert _side_red_count(p["arr"], piece_side) == 1, "piece-side count must be 1"
        ordered.append((SIDE_ORDER[target_side], p, target_side))
    ordered.sort(key=lambda x: x[0])

    used_sides = set()
    for _, piece, target_side in ordered:
        assert target_side not in used_sides, "each anchor side accepts exactly one leaf piece"
        local_red = anchor_side_to_red[target_side]
        target_red = (ar0 + local_red[0], ac0 + local_red[1])
        top, left = _adjacent_top_left(piece, target_red, target_side)
        nxt = _try_stamp(cur, piece["arr"], top, left, bg)
        assert nxt is not None, "piece must dock without overlap"
        cur = nxt
        used_sides.add(target_side)

    assert len(components_mask(cur != bg, connectivity=4)) == 1, "assembled output must be connected"
    return cur


def _new_anchor() -> dict:
    bh, bw = random.choice(ANCHOR_BLOCK_CHOICES)
    h, w = bh + 2, bw + 2
    ports = {}
    for side in SIDES:
        span = h if side in ("left", "right") else w
        ports[side] = random.randint(1, span - 2)
    return {
        "kind": "anchor",
        "shape": (h, w),
        "block": (bh, bw),
        "ports": ports,
        "top": 0,
        "left": 0,
    }


def _new_piece(attach_side: str, color: int) -> dict:
    h = random.randint(*PIECE_H_RANGE)
    w = random.randint(*PIECE_W_RANGE)
    piece_side = OPPOSITE[attach_side]
    span = h if piece_side in ("left", "right") else w
    return {
        "kind": "piece",
        "shape": (h, w),
        "color": color,
        "side": piece_side,
        "attach_side": attach_side,
        "port": random.randint(1, span - 2),
    }


def _port_cell(shape: tuple[int, int], side: str, k: int) -> tuple[int, int]:
    h, w = shape
    if side == "top":
        return 0, k
    if side == "bottom":
        return h - 1, k
    if side == "left":
        return k, 0
    return k, w - 1


def _draw_path(arr: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: int, order: int) -> None:
    r, c = start
    er, ec = end
    axes = [(r, er, c, True), (c, ec, er, False)] if order == 0 else [(c, ec, r, False), (r, er, ec, True)]
    for a0, a1, fixed, is_row in axes:
        step = 1 if a1 >= a0 else -1
        for v in range(a0, a1 + step, step):
            rr, cc = (v, fixed) if is_row else (fixed, v)
            if arr[rr, cc] != 2:
                arr[rr, cc] = color


def _paint_bar(arr: np.ndarray, side: str, color: int, blocked: set[str]) -> None:
    h, w = arr.shape
    if side in ("top", "bottom"):
        row = 0 if side == "top" else h - 1
        for c in range(w):
            if c == 0 and "left" in blocked:
                continue
            if c == w - 1 and "right" in blocked:
                continue
            arr[row, c] = color
        return
    col = 0 if side == "left" else w - 1
    for r in range(h):
        if r == 0 and "top" in blocked:
            continue
        if r == h - 1 and "bottom" in blocked:
            continue
        arr[r, col] = color


def _bar_endpoint(shape: tuple[int, int], side: str, hub: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    hr, hc = hub
    if side == "top":
        return 0, min(max(hc, 1), w - 2)
    if side == "bottom":
        return h - 1, min(max(hc, 1), w - 2)
    if side == "left":
        return min(max(hr, 1), h - 2), 0
    return min(max(hr, 1), h - 2), w - 1


def _make_anchor(node: dict) -> np.ndarray:
    bh, bw = node["block"]
    h, w = node["shape"]
    arr = np.zeros((h, w), dtype=int)
    arr[1 : 1 + bh, 1 : 1 + bw] = 4
    for side in SIDES:
        r, c = _port_cell(node["shape"], side, node["ports"][side])
        arr[r, c] = 2
    return arr


def _make_piece(node: dict) -> np.ndarray | None:
    h, w = node["shape"]
    color = node["color"]
    side = node["side"]
    arr = np.zeros((h, w), dtype=int)
    hub = (random.randint(1, h - 2), random.randint(1, w - 2))
    order = random.randint(0, 1)

    bar_sides = [s for s in SIDES if s != side]
    if len(bar_sides) > 2 and random.random() < BAR_SKIP_PROB:
        bar_sides.remove(random.choice(bar_sides))
    for bar_side in bar_sides:
        _paint_bar(arr, bar_side, color, {side})

    rr, cc = _port_cell(node["shape"], side, node["port"])
    dr, dc = INWARD[side]
    arr[rr, cc] = 2
    inner = (rr + dr, cc + dc)
    arr[inner] = color
    arr[hub] = color
    _draw_path(arr, inner, hub, color, order)
    for bar_side in bar_sides:
        _draw_path(arr, hub, _bar_endpoint(node["shape"], bar_side, hub), color, 1 - order)

    if len(components_mask(arr != 0, connectivity=4)) != 1:
        return None
    if int(np.sum(arr == 2)) != 1:
        return None
    return arr


def _nonzero_coords(arr: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    rows, cols = np.nonzero(arr != 0)
    return {(int(top + r), int(left + c)) for r, c in zip(rows, cols)}


def _red_coords(arr: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    rows, cols = np.nonzero(arr == 2)
    return {(int(top + r), int(left + c)) for r, c in zip(rows, cols)}


def _touches(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        if (r - 1, c) in b or (r + 1, c) in b or (r, c - 1) in b or (r, c + 1) in b:
            return True
    return False


def _assemble_layout(h: int, w: int) -> tuple[np.ndarray, list[dict]] | None:
    anchor = _new_anchor()
    anchor["arr"] = _make_anchor(anchor)
    pieces = []
    colors = random.sample(PIECE_COLORS, PIECE_COUNT)

    for side, color in zip(SIDES, colors):
        node = _new_piece(side, color)
        node["arr"] = _make_piece(node)
        if node["arr"] is None:
            return None
        ar, ac = _port_cell(anchor["shape"], side, anchor["ports"][side])
        pr, pc = _port_cell(node["shape"], node["side"], node["port"])
        node["top"], node["left"] = _adjacent_top_left({"reds": [(pr, pc)]}, (ar, ac), side)
        pieces.append(node)

    nodes = [anchor] + pieces
    coords_by_node = [_nonzero_coords(node["arr"], node["top"], node["left"]) for node in nodes]
    reds_by_node = [_red_coords(node["arr"], node["top"], node["left"]) for node in nodes]

    for i, j in combinations(range(len(nodes)), 2):
        if coords_by_node[i] & coords_by_node[j]:
            return None
        if i > 0 and j > 0 and _touches(coords_by_node[i], coords_by_node[j]):
            return None

    for idx, piece in enumerate(pieces, start=1):
        if not _touches(coords_by_node[0], coords_by_node[idx]):
            return None
        seam = set()
        for r, c in reds_by_node[idx]:
            for nbr in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nbr in reds_by_node[0]:
                    seam.add((r, c))
        if len(seam) != 1:
            return None

    all_coords = set().union(*coords_by_node)
    rows = [r for r, _ in all_coords]
    cols = [c for _, c in all_coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if (r1 - r0 + 1) > h or (c1 - c0 + 1) > w:
        return None

    sr = random.randint(-r0, h - 1 - r1)
    sc = random.randint(-c0, w - 1 - c1)
    out = np.zeros((h, w), dtype=int)
    for node in nodes:
        node["pos"] = (node["top"] + sr, node["left"] + sc)
        stamped = _try_stamp(out, node["arr"], node["pos"][0], node["pos"][1])
        if stamped is None:
            return None
        out = stamped

    if len(components_mask(out != 0, connectivity=4)) != 1:
        return None
    if int(np.sum(out == 2)) != PIECE_COUNT * 2:
        return None
    return out, nodes


def _scatter(nodes: list[dict], h: int, w: int) -> np.ndarray | None:
    inp = np.zeros((h, w), dtype=int)
    anchor = nodes[0]
    paste(inp, anchor["arr"], *anchor["pos"], transparent=0)
    rest = nodes[1:].copy()
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


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    assembled = _assemble_layout(h, w)
    if assembled is None:
        return None
    out, nodes = assembled
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
