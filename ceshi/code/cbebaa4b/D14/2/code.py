# Variant of cbebaa4b
# Dimension: D14 — 组装后结构的目标形状 (meta)
# Change: 组装后形成不规则的连通结构，无特定目标形状约束... → 组装后形成封闭环形（中心有空洞）
#
# Design analysis
# - Concrete shape:
#   The assembled state is no longer an arbitrary tree-like jigsaw. It is a fixed
#   hollow rectangle: four colored corner-arc sprites form a closed loop around a
#   central yellow anchor. Each arc overlaps neighbors on red 2-pixel connector
#   runs, and each arc also overlaps one inward red connector run from the anchor.
#   After the full assembly is found, the yellow anchor cells are deleted, leaving
#   a single 4-connected ring with exactly one enclosed hole.
# - Behavior trace:
#   Original: the DFS only needed "all red counts become 2" and returned the whole
#   assembly including yellow. A tree-like solution was valid.
#   Variant: the same red-pairing search can still place pieces, but the terminal
#   state must additionally survive anchor removal. Example: a candidate assembly
#   may pair every red correctly yet still fail if deleting yellow disconnects the
#   shape or if no bounded hole remains. The solver therefore keeps the pairing
#   search but changes the terminal acceptance check from "paired structure" to
#   "paired structure whose post-removal silhouette is a 1-hole ring".
# - Complexity calibration:
#   A 3x3 anchor with four quarter-ring pieces is too cramped: one piece would be
#   forced to reuse the same cells for both a neighbor join and an anchor join,
#   making the puzzle visually muddy and increasing ambiguity. To keep the D14
#   change visible and the connector layout readable, the generator uses the larger
#   original anchor size (4x5) and fixes the ring to four quarter pieces.
# - Implicit assumptions audited in main():
#   `_extract_pieces` still assumes one 4-connected scattered component per sprite;
#   the generator preserves that.
#   `_try_place` still assumes only identical colors or red-red overlaps may stack;
#   the generator constructs overlaps only on red runs.
#   The original `target_unique_red = total_ports // 2` remains valid because every
#   red cell is still paired exactly once, even though the connection graph now has
#   a cycle instead of a tree.
#   The old terminal condition "all reds paired => done" is no longer sufficient,
#   so `_finalize_ring` now checks the post-removal topology.
# - Placement legality:
#   The generator works output-first. It first creates the solved ring+anchor in a
#   local 6x7 template, shifts that template into the canvas, records each piece's
#   solved bbox, then scatters only the four colored sprites while leaving the
#   anchor fixed. Every overlap in the solved template is a deliberate red-red
#   overlap, and the scattered input is placed with padding so components remain
#   separate and solvable.
# - Visual observability:
#   Across samples, the distinctive event is obvious: colored parts close into a
#   loop and the yellow center disappears, producing a hollow frame. No sorting or
#   transformations depend on color ids; colors merely label pieces.
# - Visual cleanliness:
#   The solved output has one loop and one empty cavity. There is no dense internal
#   overlap besides the red connector cells, so the ring silhouette remains easy to
#   read from a few examples.

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


def _finalize_ring(cur: np.ndarray) -> np.ndarray | None:
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
    Reassemble scattered ring fragments around a yellow anchor, then erase the anchor to reveal a hollow loop.

    Input structure:
    - bg = black (0). Several colored sprites are scattered apart from one central yellow anchor.
    - The anchor is the unique sprite containing yellow (4) cells plus red (2) docking pixels.
    - Each colored sprite carries red connector runs that can overlap matching runs in the final assembly.

    Rule:
    - Keep the anchor fixed and translate the remaining sprites until their red connector pixels pair up.
    - The colored sprites must close into a loop that surrounds the anchor rather than ending as an open tree.
    - After the unique full assembly is found, remove the yellow anchor cells from the result, leaving the enclosed hole behind.

    Key insight:
    - Red pixels still determine how pieces can mate, but correct pairing is only half of the answer: the valid end state is the one whose silhouette remains a single one-hole ring after the yellow anchor is deleted.
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
                final = _finalize_ring(cur)
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

        assert best_moves is not None, "there must be at least one feasible connector move"

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
            for rr, cc in p["reds"]:
                gr, gc = top + rr, left + cc
                next_rc[(gr, gc)] = next_rc.get((gr, gc), 0) + 1

            nxt = _try_place(cur, p["arr"], top, left, bg)
            if nxt is None:
                continue
            dfs(nxt, next_remaining, next_rc)
            if len(solutions) >= 2:
                return

    dfs(placed, rest, red_counts)
    assert len(solutions) == 1, "ring assembly must have exactly one valid solution"
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


def _cells_to_piece(cells: list[tuple[int, int]], reds: set[tuple[int, int]], color: int) -> dict:
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


def _assembled_output(anchor: np.ndarray, pieces: list[dict], h: int, w: int) -> tuple[np.ndarray, list[dict]] | None:
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
        r0, c0, r1, c1 = piece["local_bbox"]
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

    out = with_anchor.copy()
    out[out == 4] = 0
    if len(components_mask(out != 0, connectivity=4)) != 1:
        return None
    if _bounded_hole_count(out != 0) != 1:
        return None

    return with_anchor, placed_pieces


def _scatter(anchor: np.ndarray, anchor_pos: tuple[int, int], pieces: list[dict], h: int, w: int) -> np.ndarray | None:
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
    with_anchor, placed_pieces = assembled

    anchor_global = bbox_coords(np.argwhere(with_anchor == 4))
    ar0, ac0, _, _ = anchor_global
    ah, aw = anchor.shape
    # The anchor bbox in the assembled canvas matches the sprite bbox because all
    # anchor reds lie on its border.
    anchor_top = ar0 - 1
    anchor_left = ac0 - 1
    if anchor_top < 0 or anchor_left < 0 or anchor_top + ah > h or anchor_left + aw > w:
        return None

    inp = _scatter(anchor, (anchor_top, anchor_left), placed_pieces, h, w)
    if inp is None:
        return None

    expected = with_anchor.copy()
    expected[expected == 4] = 0

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
    raise RuntimeError("failed to generate a valid cbebaa4b ring input")
