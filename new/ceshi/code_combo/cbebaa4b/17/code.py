# Combo variant of cbebaa4b (index 17)
# combo_spec: D1/0+D2/3
# Name: 形状轮廓互补 + 带洞精灵
#
# Design analysis for the fused variant
# -------------------------------------
# spec_variant:
#   preserved_core:
#     - input and output stay on the same black canvas
#     - one yellow anchor piece stays fixed; the other colored sprites are only translated
#     - the output is the fully assembled connected structure
#   fusion_rule:
#     - red connectors are removed completely
#     - attachment is now geometric: each parent exposes a side cavity/notch and the child has a matching tab
#     - non-leaf colored pieces also expose their own outward cavity, so assembly alternates between:
#         1) filling the previous piece's hole
#         2) creating the next hole further along the chain
#   concrete_behavior_trace:
#     - original task:
#         child translation was proposed by overlapping red pixels, and validity was mostly about red counts
#     - fused task:
#         anchor has visible notches on two opposite sides
#         each root piece has a matching tab that fits one notch and extends outward
#         if that piece is not a leaf, its far edge contains another visible cavity
#         the next piece must plug that cavity while extending farther along the same strip
#     - difference:
#         the search is no longer "which red aligns with which red?"
#         it becomes "which piece exactly covers this open notch while respecting the strip and leaving the right next cavity?"
#   complexity_calibration:
#     - only two chains are used, each of length 2 or 3; total pieces remain 4-6
#     - strip widths are limited to 4 or 5 and cavity depths to 1 or 2, so the geometry repeats across samples
#     - tabs are narrower than the strip, which makes concave/convex shoulders visible without exploding the case count
#     - each non-anchor piece has a unique color, preventing color-swap ambiguity while keeping the rule geometric
#   implicit_assumption_audit_main:
#     - the original solver assumed the only meaningful overlap cue was red-on-red; this is gone
#     - the original solver also assumed every internal black cell should eventually disappear; that breaks here
#       because each piece keeps a parent-facing concavity around its incoming tab
#     - therefore main() now tracks an outward side for each chain head and only treats the cavity on that side
#       as the fill target; the parent-facing concavity is ignored as a permanent silhouette feature
#     - the original "pair all connectors" terminal condition is replaced by:
#         * all anchor notches are filled
#         * every placed non-anchor piece has no remaining cavity on its outward side
#         * all pieces are used and the result is a single connected component
#   placement_legality:
#     - generator is still output-first: build the assembled structure, then scatter the colored pieces
#     - pieces never overlap on non-background cells in the assembled board; children only occupy the parent's hole cells
#     - opposite-side chains keep the layout readable and avoid chain-chain collisions
#   visual_observability:
#     - in the input, the yellow anchor shows open notches and many colored pieces show a visible side hole
#     - in the output, those holes disappear one step at a time because neighboring tabs occupy them
#     - the absence of red pixels is itself a visible signal that matching is now by silhouette, not color markers
#   cleanliness:
#     - the same geometric language is reused everywhere: notch/tab matching plus outward cavity filling
#     - no ordering or decisions depend on hidden metadata such as color ids; the solver reads only visible geometry

from __future__ import annotations

import random
import sys

import numpy as np

sys.path.append("/home/lixinyu/snap/new/solvers")
from arc_utils import bbox_coords, components_mask, paste, random_free_location


GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
SCATTER_PADDING = 2
GEN_RETRY_LIMIT = 500

ANCHOR_CORE_CHOICES = [(3, 3), (4, 5)]
CHAIN_SIDE_PAIRS = [("top", "bottom"), ("left", "right")]
CHAIN_LENGTH_CHOICES = [(2, 2), (2, 3), (3, 3)]
STRIP_W_CHOICES = [4, 5]
PIECE_LEN_RANGE = (5, 7)
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]

SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}


def _shape_from_arr(arr: np.ndarray) -> dict:
    coords = np.argwhere(arr != 0)
    rel = [(int(r), int(c)) for r, c in coords]
    return {
        "rel": rel,
        "rel_set": set(rel),
        "h": int(arr.shape[0]),
        "w": int(arr.shape[1]),
        "area": int(coords.shape[0]),
    }


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for idx, pts in enumerate(comps):
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        pieces.append(
            {
                "id": int(idx),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "shape": _shape_from_arr(arr),
                "area": int(np.sum(arr != bg)),
            }
        )
    return pieces


def _pixels(shape: dict, top: int, left: int) -> set[tuple[int, int]]:
    return {(top + r, left + c) for r, c in shape["rel"]}


def _touches(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        if (r - 1, c) in b or (r + 1, c) in b or (r, c - 1) in b or (r, c + 1) in b:
            return True
    return False


def _axis_from_side(side: str) -> str:
    return "cols" if side in ("top", "bottom") else "rows"


def _side_outside_bbox(pts: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> bool:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return any(r < r0 for r, _ in pts)
    if side == "bottom":
        return any(r > r1 for r, _ in pts)
    if side == "left":
        return any(c < c0 for _, c in pts)
    return any(c > c1 for _, c in pts)


def _placement_pts(
    shape: dict,
    top: int,
    left: int,
    occ: set[tuple[int, int]],
    board_shape: tuple[int, int],
    *,
    must_cover: set[tuple[int, int]] | None = None,
    strip: tuple[str, int, int] | None = None,
) -> set[tuple[int, int]] | None:
    h, w = int(shape["h"]), int(shape["w"])
    H, W = map(int, board_shape)
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return None
    if strip is not None:
        axis, A, B = strip
        if axis == "cols":
            if left != A or w != B - A + 1:
                return None
        else:
            if top != A or h != B - A + 1:
                return None
    pts = _pixels(shape, top, left)
    if pts & occ:
        return None
    if must_cover is not None and not must_cover.issubset(pts):
        return None
    return pts


def _aligned_positions(axis: str, A: int, cover: set[tuple[int, int]], shape: dict):
    if axis == "cols":
        left = A
        tops = set()
        for r_need, c_need in cover:
            for r_rel, c_rel in shape["rel"]:
                if c_need == left + c_rel:
                    tops.add(r_need - r_rel)
        for top in sorted(tops):
            yield int(top), int(left)
    else:
        top = A
        lefts = set()
        for r_need, c_need in cover:
            for r_rel, c_rel in shape["rel"]:
                if r_need == top + r_rel:
                    lefts.add(c_need - c_rel)
        for left in sorted(lefts):
            yield int(top), int(left)


def _border_bg_components(arr: np.ndarray, bg: int) -> list[dict]:
    h, w = map(int, arr.shape)
    out = []
    for comp in components_mask(arr == bg, connectivity=4):
        cells = {(int(r), int(c)) for r, c in comp}
        sides = []
        if any(r == 0 for r, _ in cells):
            sides.append("top")
        if any(r == h - 1 for r, _ in cells):
            sides.append("bottom")
        if any(c == 0 for _, c in cells):
            sides.append("left")
        if any(c == w - 1 for _, c in cells):
            sides.append("right")
        if sides:
            out.append({"cells": cells, "sides": tuple(sides), "size": len(cells)})
    return out


def _anchor_notches(anchor_arr: np.ndarray, bg: int) -> list[dict]:
    notches = []
    for info in _border_bg_components(anchor_arr, bg):
        if len(info["sides"]) != 1:
            continue
        notches.append(info)
    notches.sort(key=lambda d: (SIDES.index(d["sides"][0]), min(d["cells"])))
    return notches


def _outward_cavity(board: np.ndarray, placed_piece: dict, outward_side: str, bg: int) -> set[tuple[int, int]] | None:
    piece = placed_piece["piece"]
    top, left = int(placed_piece["top"]), int(placed_piece["left"])
    h, w = int(piece["shape"]["h"]), int(piece["shape"]["w"])
    sub = board[top : top + h, left : left + w]
    matches = []
    for info in _border_bg_components(sub, bg):
        if outward_side in info["sides"]:
            matches.append(info)
    if not matches:
        return None
    matches.sort(key=lambda d: (-d["size"], len(d["sides"])))
    chosen = matches[0]
    return {(top + r, left + c) for r, c in chosen["cells"]}


def _anchor_notches_filled(board: np.ndarray, anchor: dict, bg: int) -> bool:
    r0, c0, _, _ = anchor["bbox"]
    h, w = anchor["arr"].shape
    sub = board[r0 : r0 + h, c0 : c0 + w]
    return not _anchor_notches(sub, bg)


def _terminal_ok(board: np.ndarray, anchor: dict, placed_pieces: list[dict], bg: int) -> bool:
    if not _anchor_notches_filled(board, anchor, bg):
        return False
    for placed in placed_pieces:
        if _outward_cavity(board, placed, placed["outward_side"], bg) is not None:
            return False
    return len(components_mask(board != bg, connectivity=4)) == 1


def _paint_piece(board: np.ndarray, piece: dict, top: int, left: int) -> np.ndarray:
    out = board.copy()
    paste(out, piece["arr"], top, left, transparent=0)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered sprites around the fixed yellow anchor by matching side notches and filling outward cavities.

    Input structure:
    - bg = black (0). One yellow anchor sprite stays in its final position while 4-6 colored sprites are scattered elsewhere.
    - The anchor has visible side notches instead of red connectors.
    - Colored sprites have tab-like incoming ends, and many also contain a second visible hole on their outward side.

    Rule:
    - Keep the yellow anchor where it is and translate the other sprites only.
    - A root sprite is valid when its convex tab exactly occupies one anchor notch and the rest of the sprite extends outward from that side.
    - Each non-leaf sprite may expose one more outward cavity; the next sprite in that chain must fill that cavity while staying on the same strip.

    Key insight: The puzzle is solved by geometry, not color markers: the correct move is the translation where a piece exactly covers the currently open notch/hole and continues the chain outward, causing the visible cavities to disappear in the final assembly.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must have at least one piece"

    anchors = [p for p in pieces if np.any(p["arr"] == 4)]
    assert len(anchors) == 1, "input must contain exactly one yellow anchor"
    anchor = anchors[0]
    other_pieces = [p for p in pieces if p is not anchor]

    board = np.zeros((H, W), dtype=int)
    ar0, ac0, ar1, ac1 = anchor["bbox"]
    paste(board, anchor["arr"], ar0, ac0, transparent=0)
    anchor_occ = _pixels(anchor["shape"], ar0, ac0)
    anchor_bbox = (int(ar0), int(ac0), int(ar1), int(ac1))

    notch_infos = _anchor_notches(anchor["arr"], bg)
    assert len(notch_infos) >= 1

    root_cands_by_notch: list[list[dict]] = []
    for notch in notch_infos:
        side = notch["sides"][0]
        cover = {(ar0 + r, ac0 + c) for r, c in notch["cells"]}
        axis = _axis_from_side(side)
        cands = []
        for piece in other_pieces:
            seen = set()
            for r_need, c_need in cover:
                for r_rel, c_rel in piece["shape"]["rel"]:
                    top = r_need - r_rel
                    left = c_need - c_rel
                    key = (piece["id"], int(top), int(left))
                    if key in seen:
                        continue
                    seen.add(key)
                    pts = _placement_pts(
                        piece["shape"],
                        int(top),
                        int(left),
                        anchor_occ,
                        (H, W),
                        must_cover=cover,
                    )
                    if pts is None:
                        continue
                    if not _side_outside_bbox(pts, anchor_bbox, side):
                        continue
                    cands.append(
                        {
                            "piece": piece,
                            "top": int(top),
                            "left": int(left),
                            "side": side,
                            "axis": axis,
                            "A": int(left if axis == "cols" else top),
                            "B": int(left + piece["shape"]["w"] - 1 if axis == "cols" else top + piece["shape"]["h"] - 1),
                            "pts": pts,
                        }
                    )
        dedup = {}
        for cand in cands:
            key = (cand["piece"]["id"], cand["top"], cand["left"])
            dedup[key] = cand
        cands2 = list(dedup.values())
        cands2.sort(key=lambda d: (-d["piece"]["area"], d["top"], d["left"], d["piece"]["id"]))
        assert cands2, "every anchor notch must have at least one candidate root"
        root_cands_by_notch.append(cands2)

    notch_order = sorted(range(len(root_cands_by_notch)), key=lambda i: len(root_cands_by_notch[i]))
    solutions: list[np.ndarray] = []

    def extend_search(
        cur_board: np.ndarray,
        occ: set[tuple[int, int]],
        chains: list[dict],
        remaining_ids: set[int],
        placed_now: list[dict],
    ) -> None:
        if len(solutions) >= 2:
            return
        if not remaining_ids:
            if _terminal_ok(cur_board, anchor, placed_now, bg):
                solutions.append(cur_board.copy())
            return

        chain_moves = []
        for chain_idx, chain in enumerate(chains):
            cavity = _outward_cavity(cur_board, chain["head"], chain["side"], bg)
            if cavity is None:
                continue
            head = chain["head"]
            head_pts = _pixels(head["piece"]["shape"], head["top"], head["left"])
            parent_bbox = (
                int(head["top"]),
                int(head["left"]),
                int(head["top"] + head["piece"]["shape"]["h"] - 1),
                int(head["left"] + head["piece"]["shape"]["w"] - 1),
            )
            moves = []
            for piece in other_pieces:
                if piece["id"] not in remaining_ids:
                    continue
                for top, left in _aligned_positions(chain["axis"], chain["A"], cavity, piece["shape"]):
                    pts = _placement_pts(
                        piece["shape"],
                        top,
                        left,
                        occ,
                        (H, W),
                        must_cover=cavity,
                        strip=(chain["axis"], chain["A"], chain["B"]),
                    )
                    if pts is None:
                        continue
                    if not _touches(pts, head_pts):
                        continue
                    if not _side_outside_bbox(pts, parent_bbox, chain["side"]):
                        continue
                    moves.append({"piece": piece, "top": top, "left": left, "pts": pts})
            if moves:
                moves.sort(key=lambda d: (-d["piece"]["area"], d["top"], d["left"], d["piece"]["id"]))
                chain_moves.append((len(moves), len(cavity), chain_idx, moves))

        if not chain_moves:
            return

        chain_moves.sort()
        _, _, chain_idx, moves = chain_moves[0]
        for mv in moves:
            piece = mv["piece"]
            next_board = _paint_piece(cur_board, piece, mv["top"], mv["left"])
            next_occ = occ | mv["pts"]
            next_chains = list(chains)
            next_placed_piece = {
                "piece": piece,
                "top": int(mv["top"]),
                "left": int(mv["left"]),
                "outward_side": next_chains[chain_idx]["side"],
            }
            next_chains[chain_idx] = dict(next_chains[chain_idx])
            next_chains[chain_idx]["head"] = next_placed_piece
            next_remaining = set(remaining_ids)
            next_remaining.remove(piece["id"])
            extend_search(
                next_board,
                next_occ,
                next_chains,
                next_remaining,
                placed_now + [next_placed_piece],
            )
            if len(solutions) >= 2:
                return

    def assign_roots(
        idx: int,
        cur_board: np.ndarray,
        occ: set[tuple[int, int]],
        used_ids: set[int],
        chains: list[dict],
        placed_now: list[dict],
    ) -> None:
        if len(solutions) >= 2:
            return
        if idx == len(notch_order):
            remaining_ids = {p["id"] for p in other_pieces} - used_ids
            extend_search(cur_board, occ, chains, remaining_ids, placed_now)
            return

        notch_i = notch_order[idx]
        for cand in root_cands_by_notch[notch_i]:
            piece = cand["piece"]
            if piece["id"] in used_ids:
                continue
            if cand["pts"] & occ:
                continue
            next_board = _paint_piece(cur_board, piece, cand["top"], cand["left"])
            placed_piece = {
                "piece": piece,
                "top": int(cand["top"]),
                "left": int(cand["left"]),
                "outward_side": cand["side"],
            }
            assign_roots(
                idx + 1,
                next_board,
                occ | cand["pts"],
                used_ids | {piece["id"]},
                chains
                + [
                    {
                        "side": cand["side"],
                        "axis": cand["axis"],
                        "A": cand["A"],
                        "B": cand["B"],
                        "head": placed_piece,
                    }
                ],
                placed_now + [placed_piece],
            )
            if len(solutions) >= 2:
                return

    assign_roots(0, board, anchor_occ, set(), [], [])
    assert len(solutions) == 1, "assembly must have exactly one valid solution"
    return solutions[0]


def _incoming_side(side: str) -> str:
    return OPPOSITE[side]


def _make_notch_spec(strip_w: int) -> dict:
    width = 2 if strip_w == 4 else random.choice([2, 3])
    depth = random.choice([1, 2])
    offset = random.randint(1, strip_w - width - 1)
    return {"width": int(width), "depth": int(depth), "offset": int(offset)}


def _choose_piece_length(incoming: dict, outgoing: dict | None) -> int:
    min_len = incoming["depth"] + 2
    if outgoing is not None:
        min_len += outgoing["depth"] + 1
    lo = max(PIECE_LEN_RANGE[0], min_len)
    hi = max(lo, PIECE_LEN_RANGE[1])
    return random.randint(lo, hi)


def _side_slices(side: str, depth: int, offset: int, width: int, h: int, w: int):
    if side == "top":
        return slice(0, depth), slice(offset, offset + width)
    if side == "bottom":
        return slice(h - depth, h), slice(offset, offset + width)
    if side == "left":
        return slice(offset, offset + width), slice(0, depth)
    return slice(offset, offset + width), slice(w - depth, w)


def _apply_incoming_tab(arr: np.ndarray, side: str, spec: dict, color: int) -> None:
    h, w = arr.shape
    rows, cols = _side_slices(side, spec["depth"], spec["offset"], spec["width"], h, w)
    if side in ("top", "bottom"):
        arr[rows, :] = 0
        arr[rows, cols] = color
    else:
        arr[:, cols] = 0
        arr[rows, cols] = color


def _apply_outgoing_cavity(arr: np.ndarray, side: str, spec: dict) -> None:
    h, w = arr.shape
    rows, cols = _side_slices(side, spec["depth"], spec["offset"], spec["width"], h, w)
    arr[rows, cols] = 0


def _make_piece_arr(side: str, strip_w: int, length: int, color: int, incoming: dict, outgoing: dict | None) -> np.ndarray:
    if side in ("top", "bottom"):
        h, w = int(length), int(strip_w)
    else:
        h, w = int(strip_w), int(length)
    arr = np.full((h, w), int(color), dtype=int)
    _apply_incoming_tab(arr, _incoming_side(side), incoming, color)
    if outgoing is not None:
        _apply_outgoing_cavity(arr, side, outgoing)
    assert len(components_mask(arr != 0, connectivity=4)) == 1
    return arr


def _make_anchor_arr(outer_h: int, outer_w: int, core_h: int, core_w: int, chains: list[dict]) -> np.ndarray:
    arr = np.full((outer_h, outer_w), 4, dtype=int)
    for ch in chains:
        side = ch["side"]
        spec = ch["incoming"]
        start = ch["strip_start"]
        if side == "top":
            arr[0 : spec["depth"], start + spec["offset"] : start + spec["offset"] + spec["width"]] = 0
        elif side == "bottom":
            arr[outer_h - spec["depth"] : outer_h, start + spec["offset"] : start + spec["offset"] + spec["width"]] = 0
        elif side == "left":
            arr[start + spec["offset"] : start + spec["offset"] + spec["width"], 0 : spec["depth"]] = 0
        else:
            arr[start + spec["offset"] : start + spec["offset"] + spec["width"], outer_w - spec["depth"] : outer_w] = 0

    # Keep the original task's central yellow block visually explicit.
    r0 = (outer_h - core_h) // 2
    c0 = (outer_w - core_w) // 2
    arr[r0 : r0 + core_h, c0 : c0 + core_w] = 4
    assert len(components_mask(arr != 0, connectivity=4)) == 1
    return arr


def _bbox_of_node(node: dict) -> tuple[int, int, int, int]:
    h, w = node["arr"].shape
    return (node["top"], node["left"], node["top"] + h - 1, node["left"] + w - 1)


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_piece_along_chain(
    prev_node: dict,
    side: str,
    arr: np.ndarray,
    incoming: dict,
    *,
    strip_start: int | None = None,
) -> tuple[int, int]:
    ph, pw = prev_node["arr"].shape
    h, w = arr.shape
    base_top = int(prev_node["top"])
    base_left = int(prev_node["left"])
    if prev_node.get("kind") == "anchor":
        assert strip_start is not None
        if side in ("top", "bottom"):
            base_left += int(strip_start)
        else:
            base_top += int(strip_start)
    if side == "top":
        return int(base_top - (h - incoming["depth"])), int(base_left)
    if side == "bottom":
        return int(base_top + ph - incoming["depth"]), int(base_left)
    if side == "left":
        return int(base_top), int(base_left - (w - incoming["depth"]))
    return int(base_top), int(base_left + pw - incoming["depth"])


def _render_nodes(nodes: list[dict], H: int, W: int) -> np.ndarray | None:
    out = np.zeros((H, W), dtype=int)
    for i, node in enumerate(nodes):
        top, left = int(node["top"]), int(node["left"])
        arr = node["arr"]
        h, w = arr.shape
        if top < 0 or left < 0 or top + h > H or left + w > W:
            return None
        sub = out[top : top + h, left : left + w]
        mask = arr != 0
        if np.any(mask & (sub != 0)):
            return None
        paste(out, arr, top, left, transparent=0)
        for j in range(i):
            if not _rects_overlap(_bbox_of_node(node), _bbox_of_node(nodes[j])):
                continue
            # Overlap of bboxes is allowed only through background cavity cells.
            aj = nodes[j]["arr"]
            tj, lj = int(nodes[j]["top"]), int(nodes[j]["left"])
            rr0 = max(top, tj)
            cc0 = max(left, lj)
            rr1 = min(top + h - 1, tj + aj.shape[0] - 1)
            cc1 = min(left + w - 1, lj + aj.shape[1] - 1)
            for rr in range(rr0, rr1 + 1):
                for cc in range(cc0, cc1 + 1):
                    v1 = int(arr[rr - top, cc - left])
                    v2 = int(aj[rr - tj, cc - lj])
                    if v1 != 0 and v2 != 0:
                        return None
    if len(components_mask(out != 0, connectivity=4)) != 1:
        return None
    return out


def _scatter(nodes: list[dict], H: int, W: int) -> np.ndarray | None:
    inp = np.zeros((H, W), dtype=int)
    anchor = nodes[0]
    paste(inp, anchor["arr"], anchor["top"], anchor["left"], transparent=0)
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


def _shift_nodes(nodes: list[dict], dr: int, dc: int) -> list[dict]:
    out = []
    for node in nodes:
        copied = dict(node)
        copied["top"] = int(node["top"] + dr)
        copied["left"] = int(node["left"] + dc)
        out.append(copied)
    return out


def _build_nodes() -> list[dict] | None:
    core_h, core_w = random.choice(ANCHOR_CORE_CHOICES)
    outer_h, outer_w = core_h + 4, core_w + 4
    side_a, side_b = random.choice(CHAIN_SIDE_PAIRS)
    len_a, len_b = random.choice(CHAIN_LENGTH_CHOICES)
    chain_lengths = {side_a: len_a, side_b: len_b}
    colors = random.sample(PIECE_COLORS, len_a + len_b)
    color_iter = iter(colors)

    chain_specs = []
    if side_a in ("top", "bottom"):
        strip_limit = outer_w
    else:
        strip_limit = outer_h

    for side in (side_a, side_b):
        strip_w = random.choice(STRIP_W_CHOICES)
        if side in ("top", "bottom"):
            start = random.randint(0, outer_w - strip_w)
        else:
            start = random.randint(0, outer_h - strip_w)
        incoming = _make_notch_spec(strip_w)
        chain_specs.append(
            {
                "side": side,
                "strip_w": int(strip_w),
                "strip_start": int(start),
                "incoming": incoming,
                "length": int(chain_lengths[side]),
            }
        )

    # Keep opposite-side strips readable by discouraging the exact same band.
    if side_a in ("top", "bottom"):
        if abs(chain_specs[0]["strip_start"] - chain_specs[1]["strip_start"]) <= 1:
            return None
    else:
        if abs(chain_specs[0]["strip_start"] - chain_specs[1]["strip_start"]) <= 1:
            return None

    anchor_arr = _make_anchor_arr(outer_h, outer_w, core_h, core_w, chain_specs)
    nodes = [{"kind": "anchor", "arr": anchor_arr, "top": 0, "left": 0}]

    for ch in chain_specs:
        prev_node = nodes[0]
        incoming = ch["incoming"]
        for idx in range(ch["length"]):
            outgoing = _make_notch_spec(ch["strip_w"]) if idx + 1 < ch["length"] else None
            length = _choose_piece_length(incoming, outgoing)
            color = next(color_iter)
            arr = _make_piece_arr(ch["side"], ch["strip_w"], length, color, incoming, outgoing)
            top, left = _place_piece_along_chain(
                prev_node,
                ch["side"],
                arr,
                incoming,
                strip_start=ch["strip_start"],
            )
            piece_node = {
                "kind": "piece",
                "arr": arr,
                "top": int(top),
                "left": int(left),
                "side": ch["side"],
                "incoming": dict(incoming),
                "outgoing": None if outgoing is None else dict(outgoing),
            }
            nodes.append(piece_node)
            prev_node = piece_node
            if outgoing is not None:
                incoming = outgoing

    # Quick local overlap rejection before global shifting.
    for i in range(len(nodes)):
        for j in range(i):
            bi = _bbox_of_node(nodes[i])
            bj = _bbox_of_node(nodes[j])
            if not _rects_overlap(bi, bj):
                continue
            ai = nodes[i]["arr"]
            aj = nodes[j]["arr"]
            ti, li = nodes[i]["top"], nodes[i]["left"]
            tj, lj = nodes[j]["top"], nodes[j]["left"]
            rr0 = max(ti, tj)
            cc0 = max(li, lj)
            rr1 = min(ti + ai.shape[0] - 1, tj + aj.shape[0] - 1)
            cc1 = min(li + ai.shape[1] - 1, lj + aj.shape[1] - 1)
            for rr in range(rr0, rr1 + 1):
                for cc in range(cc0, cc1 + 1):
                    if int(ai[rr - ti, cc - li]) != 0 and int(aj[rr - tj, cc - lj]) != 0:
                        return None
    return nodes


def _fusion_metrics(inp: np.ndarray, out: np.ndarray) -> dict:
    pieces = _extract_pieces(inp, 0)
    anchor = next(p for p in pieces if np.any(p["arr"] == 4))
    non_anchor = [p for p in pieces if p is not anchor]
    open_cavity_cells = 0
    pieces_with_outward_hole = 0
    for p in non_anchor:
        comps = _border_bg_components(p["arr"], 0)
        comps = [c for c in comps if len(c["sides"]) == 1]
        if comps:
            pieces_with_outward_hole += 1
            open_cavity_cells += max(c["size"] for c in comps)
    anchor_notch_cells = sum(c["size"] for c in _anchor_notches(anchor["arr"], 0))
    out_pieces = _extract_pieces(out, 0)
    out_anchor = next(p for p in out_pieces if np.any(p["arr"] == 4))
    remaining_anchor_notches = sum(c["size"] for c in _anchor_notches(out_anchor["arr"], 0))
    return {
        "pieces_with_outward_hole": int(pieces_with_outward_hole),
        "open_cavity_cells": int(open_cavity_cells),
        "anchor_notch_cells": int(anchor_notch_cells),
        "remaining_anchor_notches": int(remaining_anchor_notches),
    }


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    nodes = _build_nodes()
    if nodes is None:
        return None

    bboxes = [_bbox_of_node(n) for n in nodes]
    r0 = min(b[0] for b in bboxes)
    c0 = min(b[1] for b in bboxes)
    r1 = max(b[2] for b in bboxes)
    c1 = max(b[3] for b in bboxes)
    ah = r1 - r0 + 1
    aw = c1 - c0 + 1
    if ah > H or aw > W:
        return None

    dr = random.randint(-r0, H - 1 - r1)
    dc = random.randint(-c0, W - 1 - c1)
    shifted = _shift_nodes(nodes, dr, dc)
    out = _render_nodes(shifted, H, W)
    if out is None:
        return None
    inp = _scatter(shifted, H, W)
    if inp is None:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    metrics = _fusion_metrics(inp, out)
    if metrics["anchor_notch_cells"] <= 0 or metrics["open_cavity_cells"] <= 0:
        return None
    if metrics["pieces_with_outward_hole"] < 2:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cbebaa4b combo variant input")
