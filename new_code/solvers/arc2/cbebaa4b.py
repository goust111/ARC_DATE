import numpy as np

from arc_utils import bbox_pts as _bbox_pts, cc4 as _cc4


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = _cc4(g != bg)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = _bbox_pts(pts)
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


def _candidate_translations(
    unpaired_reds: list[tuple[int, int]], piece: dict, canvas: np.ndarray
) -> list[tuple[int, int, int]]:
    """
    Generate feasible (top,left,overlap_red_count) placements for piece by aligning one of its red
    pixels to an *unpaired* red pixel already in the assembly.
    """
    if not unpaired_reds or not piece["reds"]:
        return []
    ph, pw = piece["shape"]
    arr = piece["arr"]
    h, w = canvas.shape

    cand: dict[tuple[int, int], int] = {}
    for rr, cc in unpaired_reds:
        for pr, pc in piece["reds"]:
            top = rr - pr
            left = cc - pc
            if top < 0 or left < 0 or top + ph > h or left + pw > w:
                continue
            sub = canvas[top : top + ph, left : left + pw]
            overlap = int(np.sum((arr == 2) & (sub == 2)))
            if overlap <= 0:
                continue
            cand[(top, left)] = max(cand.get((top, left), 0), overlap)

    out = [(t, l, ov) for (t, l), ov in cand.items()]
    out.sort(key=lambda x: (-x[2], x[0], x[1]))
    return out


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered jigsaw sprites around a central yellow anchor by pairing red connector pixels.

    Input structure:
    - bg = black (0). Several colored sprites scattered on the grid.
    - One anchor sprite contains a 3×3 yellow (4) block.
    - Each sprite has red (2) connector pixels along its edges.

    Rule:
    - The anchor stays in place; remaining sprites attach one at a time.
    - Each sprite is translated so that one of its red pixels overlaps an unpaired red pixel on the growing assembly.
    - Every red connector must pair with exactly one other, forming a complete joined structure.

    Key insight: Red pixels act as jigsaw pegs — each must match exactly one counterpart, and the assembly order is determined by which unpaired connectors remain.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    if not pieces:
        return g.copy()

    # Anchor is the unique piece containing the 3x3 yellow (color 4) block.
    anchors = [p for p in pieces if np.any(p["arr"] == 4)]
    if len(anchors) != 1:
        return g.copy()
    anchor = anchors[0]
    rest = [p for p in pieces if p is not anchor]

    canvas = np.full((h, w), bg, dtype=int)
    ar0, ac0, _, _ = anchor["bbox"]
    placed = _try_place(canvas, anchor["arr"], ar0, ac0, bg)
    if placed is None:
        return g.copy()

    total_ports = int(sum(len(p["reds"]) for p in pieces))
    if total_ports % 2 != 0:
        return g.copy()
    target_unique_red = total_ports // 2

    # Track how many original red pixels map to each assembled red coordinate.
    red_counts: dict[tuple[int, int], int] = {}
    for rr, cc in anchor["reds"]:
        red_counts[(ar0 + rr, ac0 + cc)] = red_counts.get((ar0 + rr, ac0 + cc), 0) + 1

    # Backtracking assembly by pairing up red connector pixels.
    best: np.ndarray | None = None

    def dfs(cur: np.ndarray, remaining: list[dict], rc: dict[tuple[int, int], int]) -> None:
        nonlocal best
        if best is not None:
            return

        if not remaining:
            if len(rc) == target_unique_red and all(v == 2 for v in rc.values()):
                best = cur
            return

        if len(rc) > target_unique_red:
            return

        # Unpaired reds are exactly those with count==1; always pair one of them next.
        unpaired = [k for k, v in rc.items() if v == 1]
        if not unpaired:
            return

        rc1 = {k for k, v in rc.items() if v == 1}

        # Pick the most constrained unpaired red coordinate: fewest ways to cover it.
        best_target = None
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
                    # quick red-count feasibility
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
                    # must overlap at least the target (guaranteed), but keep for scoring
                    moves.append((idx, top, left, overlap))
            if not moves:
                continue
            if best_moves is None or len(moves) < len(best_moves):
                best_target = t
                best_moves = moves
                if len(best_moves) == 1:
                    break

        if best_moves is None:
            return

        # Deduplicate moves by (piece, top, left) keeping max overlap score.
        dedup: dict[tuple[int, int, int], int] = {}
        for idx, top, left, ov in best_moves:
            k = (idx, top, left)
            dedup[k] = max(dedup.get(k, 0), ov)
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
            if best is not None:
                return

    dfs(placed, rest, red_counts)
    return g.copy() if best is None else best
