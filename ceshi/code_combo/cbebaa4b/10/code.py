# Combo variant of cbebaa4b (index 10)
# combo_spec: D4/1+D14/2
# Name: 擦除锚点 + 封闭环形结构
#
# Design analysis
# ---------------
# a) Concrete behavior and trace:
#    - Original cbebaa4b reconstructs one connected red-paired assembly and keeps
#      the yellow anchor in the output.
#    - This combo keeps the same translation-only red-overlap assembly process,
#      but changes the accepted end state in two coupled ways:
#      1. the colored sprites must close around the anchor as a loop rather than
#         ending as an open tree;
#      2. once the unique solved placement is found, the yellow anchor is erased.
#    - Small mental example: if four arc-like sprites sit around a 4x5 yellow
#      block, the original task would return the ring plus the yellow center.
#      Here the same connector search still reconstructs that arrangement first,
#      but the returned image repaints the yellow center to black, so the visible
#      result is a hollow ring. A candidate that pairs all reds yet becomes two
#      pieces after anchor removal is rejected.
# b) Complexity calibration:
#    - A free-form original generator would often produce trees. Erasing the
#      anchor from those would create disconnected spokes or ambiguous near-rings.
#    - To keep the fusion legible from 2-4 ARC examples, the generator fixes a
#      compact four-piece ring template around a larger 4x5 anchor. This sharply
#      limits ambiguity while keeping enough color variation to show that the rule
#      depends on geometry, not on color ids.
# c) Implicit assumption review:
#    - `_extract_pieces` assumes scattered sprites are separate 4-connected
#      components; generator keeps Chebyshev gaps >= 2.
#    - Anchor detection assumes exactly one component contains yellow(4); the
#      generator builds exactly one such piece.
#    - `_try_place` assumes overlaps are only legal for identical colors or
#      red-on-red; solved layouts are constructed using red connector overlaps
#      only, so this remains valid.
#    - The original DFS assumption "all red coordinates end with count 2" still
#      holds, but is no longer sufficient by itself. The combo adds a terminal
#      topology check after yellow erasure: the remaining nonzero mask must be one
#      component with exactly one bounded hole.
#    - New solver assumption: a valid puzzle has exactly one placement whose
#      post-erase silhouette is a hollow ring. The generator verifies this by
#      calling `main()` on every accepted sample.
# d) Placement legality:
#    - The generator is output-first. It builds the solved anchor+ring layout in
#      local coordinates, shifts it into the canvas, checks bounds and red pairing,
#      then scatters only the colored pieces while keeping the anchor fixed.
#    - No non-red overlap is allowed; `_try_place` enforces that during assembly
#      and generator validation.
# e) Visual observability:
#    - Humans can infer the rule because the yellow anchor is visibly central in
#      the input, the colored pieces visibly form a closed contour in the output,
#      and the anchor disappears. The decisive signal is the shape around the
#      anchor, not any hidden ordering or color ranking.
# f) Visual cleanliness:
#    - The ring uses four corner-like pieces with sparse red contact runs. This
#      avoids dense multi-layer overlap and leaves one clean rectangular cavity.
# g) Change inferability:
#    - Colors only label pieces; they are randomly permuted and never transformed.
#      The learnable rule is spatial: assemble by red connectors, close the loop,
#      then delete the yellow helper. Repeated examples support the same geometry.

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


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

    conflict = mask & (sub != bg) & ~(((arr == 2) & (sub == 2)) | (arr == sub))
    if np.any(conflict):
        return None

    out = canvas.copy()
    out_sub = out[top : top + ph, left : left + pw]
    out_sub[mask] = arr[mask]
    out[top : top + ph, left : left + pw] = out_sub
    return out


def _bounded_hole_count(mask: np.ndarray) -> int:
    if not np.any(mask):
        return 0
    comps = components_mask(mask, connectivity=4)
    if len(comps) != 1:
        return -1
    r0, c0, r1, c1 = bbox_coords(np.argwhere(mask))
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    holes = 0
    for pts in components_mask(~sub, connectivity=4):
        rs = pts[:, 0]
        cs = pts[:, 1]
        if np.any(rs == 0) or np.any(rs == sub.shape[0] - 1):
            continue
        if np.any(cs == 0) or np.any(cs == sub.shape[1] - 1):
            continue
        holes += 1
    return int(holes)


def _finalize_combo(cur: np.ndarray) -> np.ndarray | None:
    out = cur.copy()
    out[out == 4] = 0
    mask = out != 0
    if len(components_mask(mask, connectivity=4)) != 1:
        return None
    if _bounded_hole_count(mask) != 1:
        return None
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered connector sprites into a closed loop around the yellow anchor, then delete the anchor.

    Input structure:
    - bg = black (0). One fixed anchor sprite contains yellow (4) cells plus red (2) connector runs.
    - Several other colored sprites are scattered elsewhere, each as its own 4-connected component.
    - Red connector pixels mark the only places where different sprites may overlap in the solved layout.

    Rule:
    - Keep the yellow anchor at its input position and translate the remaining sprites so red connectors pair up exactly.
    - Among all red-paired assemblies, accept only the one whose colored structure closes into a loop around the anchor.
    - After the full assembly is determined, erase the yellow anchor cells back to background so the loop becomes hollow.

    Key insight:
    - The yellow block is a temporary jig, not part of the final answer. Red overlaps reconstruct the full placement, and the correct placement is the one that still forms a single one-hole ring after the yellow center is removed.
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

    solutions: list[np.ndarray] = []

    def _push_solution(arr: np.ndarray) -> None:
        for prev in solutions:
            if np.array_equal(prev, arr):
                return
        solutions.append(arr)

    def dfs(cur: np.ndarray, remaining: list[dict], rc: dict[tuple[int, int], int]) -> None:
        if len(solutions) >= 2:
            return

        if not remaining:
            if len(rc) == target_unique_red and all(v == 2 for v in rc.values()):
                final = _finalize_combo(cur)
                if final is not None:
                    _push_solution(final)
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
        for idx, top, left, ov in best_moves:
            key = (idx, top, left)
            dedup[key] = max(dedup.get(key, 0), ov)
        moves2 = [(idx, top, left, ov) for (idx, top, left), ov in dedup.items()]
        moves2.sort(key=lambda x: (-x[3], -remaining[x[0]]["area"], x[1], x[2], x[0]))

        for idx, top, left, _ov in moves2:
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
    assert len(solutions) == 1, "combo ring assembly must have exactly one valid solution"
    return solutions[0]


# --------------- generator (o5x) ---------------

GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]
SCATTER_PADDING = 2
GEN_RETRY_LIMIT = 300
ANCHOR_BLOCK = (4, 5)


def _line_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    step = 1 if r1 >= r0 else -1
    return [(r, c0) for r in range(r0, r1 + step, step)]


def _cells_to_piece(
    cells: list[tuple[int, int]], reds: set[tuple[int, int]], color: int
) -> dict:
    uniq = sorted(set(cells))
    coords = np.array(uniq, dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    arr = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in uniq:
        arr[r - r0, c - c0] = 2 if (r, c) in reds else color
    return {
        "arr": arr,
        "local_bbox": (int(r0), int(c0), int(r1), int(c1)),
        "reds": [(int(r - r0), int(c - c0)) for r, c in uniq if (r, c) in reds],
    }


def _build_local_entities() -> tuple[np.ndarray, list[dict]]:
    bh, bw = ANCHOR_BLOCK
    oh, ow = bh + 2, bw + 2

    top_join = {(0, 3), (0, 4)}
    right_join = {(3, ow - 1), (4, ow - 1)}
    bottom_join = {(oh - 1, 2), (oh - 1, 3)}
    left_join = {(2, 0), (3, 0)}

    top_anchor = {(0, 1), (0, 2)}
    right_anchor = {(1, ow - 1), (2, ow - 1)}
    bottom_anchor = {(oh - 1, 4), (oh - 1, 5)}
    left_anchor = {(4, 0), (5, 0)}

    pieces = []

    tl_cells = _line_cells(0, 0, 0, 4) + _line_cells(0, 0, 3, 0)
    tl_reds = top_join | left_join | top_anchor
    pieces.append(_cells_to_piece(tl_cells, tl_reds, -1))

    tr_cells = _line_cells(0, 3, 0, ow - 1) + _line_cells(0, ow - 1, 4, ow - 1)
    tr_reds = top_join | right_join | right_anchor
    pieces.append(_cells_to_piece(tr_cells, tr_reds, -1))

    br_cells = _line_cells(3, ow - 1, oh - 1, ow - 1) + _line_cells(oh - 1, 2, oh - 1, ow - 1)
    br_reds = right_join | bottom_join | bottom_anchor
    pieces.append(_cells_to_piece(br_cells, br_reds, -1))

    bl_cells = _line_cells(2, 0, oh - 1, 0) + _line_cells(oh - 1, 0, oh - 1, 3)
    bl_reds = left_join | bottom_join | left_anchor
    pieces.append(_cells_to_piece(bl_cells, bl_reds, -1))

    for piece, color in zip(pieces, random.sample(PIECE_COLORS, len(pieces))):
        arr = piece["arr"].copy()
        arr[arr == -1] = color
        piece["arr"] = arr

    anchor = np.zeros((oh, ow), dtype=int)
    anchor[1 : 1 + bh, 1 : 1 + bw] = 4
    for r, c in top_anchor | right_anchor | bottom_anchor | left_anchor:
        anchor[r, c] = 2

    return anchor, pieces


def _place_piece(out: np.ndarray, arr: np.ndarray, top: int, left: int) -> None:
    mask = arr != 0
    rows, cols = np.nonzero(mask)
    out[rows + top, cols + left] = arr[rows, cols]


def _assembled_output(
    anchor: np.ndarray, pieces: list[dict], h: int, w: int
) -> tuple[np.ndarray, np.ndarray, list[dict], tuple[int, int]] | None:
    ah, aw = anchor.shape
    top = random.randint(0, h - ah)
    left = random.randint(0, w - aw)

    with_anchor = np.zeros((h, w), dtype=int)
    _place_piece(with_anchor, anchor, top, left)

    placed_pieces = []
    red_counts: dict[tuple[int, int], int] = {}
    for rr, cc in np.argwhere(anchor == 2):
        key = (int(top + rr), int(left + cc))
        red_counts[key] = red_counts.get(key, 0) + 1

    for piece in pieces:
        r0, c0, _r1, _c1 = piece["local_bbox"]
        pr = top + r0
        pc = left + c0
        placed = _try_place(with_anchor, piece["arr"], pr, pc, 0)
        if placed is None:
            return None
        with_anchor = placed

        for rr, cc in piece["reds"]:
            key = (int(pr + rr), int(pc + cc))
            red_counts[key] = red_counts.get(key, 0) + 1

        placed_pieces.append({"arr": piece["arr"], "pos": (pr, pc)})

    total_reds = int(np.sum(anchor == 2)) + sum(int(np.sum(p["arr"] == 2)) for p in placed_pieces)
    if len(red_counts) * 2 != total_reds:
        return None
    if any(v != 2 for v in red_counts.values()):
        return None

    out = _finalize_combo(with_anchor)
    if out is None:
        return None

    return with_anchor, out, placed_pieces, (top, left)


def _scatter(
    anchor: np.ndarray,
    anchor_pos: tuple[int, int],
    pieces: list[dict],
    h: int,
    w: int,
) -> np.ndarray | None:
    inp = np.zeros((h, w), dtype=int)
    paste(inp, anchor, *anchor_pos, transparent=0)

    order = pieces[:]
    random.shuffle(order)
    for piece in order:
        try:
            r, c = random_free_location(
                inp,
                piece["arr"],
                transparent=0,
                padding=SCATTER_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inp, piece["arr"], r, c, transparent=0)
    return inp


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    anchor, pieces = _build_local_entities()
    assembled = _assembled_output(anchor, pieces, h, w)
    if assembled is None:
        return None
    _with_anchor, expected, placed_pieces, anchor_pos = assembled

    inp = _scatter(anchor, anchor_pos, placed_pieces, h, w)
    if inp is None:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(solved, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cbebaa4b combo input")
