from __future__ import annotations
from collections import Counter, defaultdict
import numpy as np
from arc_utils import get_bg, label_mask
import random
from arc_utils import paste, random_free_location, find_components

def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered fragments into one image by matching overlapping regions.

    Input structure:
    - bg = background. Several disconnected foreground fragments (connected components of non-bg).
    - Each fragment includes bg pixels inside its bounding box.

    Rule:
    - Pairs of fragments share identical rectangular overlap zones (at least 3 pixels wide or tall).
    - Attach fragments by aligning at the maximum-area overlap.
    - Output is the tight crop of the assembled composite.

    Key insight: bg pixels inside each fragment's bbox are part of the pattern and must match exactly in overlap zones.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    lab, n = label_mask(g != int(bg), connectivity=4)
    assert not (int(n) == 0)

    # Extract bbox patches (include bg-colored pixels inside bbox).
    pieces = []
    for k in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(k))
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        arr = g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()
        nz = [(int(r), int(c), int(arr[int(r), int(c)])) for r, c in np.argwhere(arr != int(bg))]
        bycol: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for r, c, v in nz:
            bycol[int(v)].append((int(r), int(c)))
        pieces.append({"arr": arr, "h": int(arr.shape[0]), "w": int(arr.shape[1]), "nz": nz, "bycol": bycol})

    def overlap_rect(a: np.ndarray, b: np.ndarray, dr: int, dc: int) -> tuple[int, int, int, int]:
        r0 = max(0, int(dr))
        c0 = max(0, int(dc))
        r1 = min(int(a.shape[0]), int(dr) + int(b.shape[0]))
        c1 = min(int(a.shape[1]), int(dc) + int(b.shape[1]))
        return int(r0), int(c0), int(r1), int(c1)

    def overlap_all_bg(a: np.ndarray, b: np.ndarray, dr: int, dc: int) -> bool:
        r0, c0, r1, c1 = overlap_rect(a, b, dr, dc)
        if r0 >= r1 or c0 >= c1:
            return True
        sub_a = a[int(r0) : int(r1), int(c0) : int(c1)]
        sub_b = b[int(r0 - dr) : int(r1 - dr), int(c0 - dc) : int(c1 - dc)]
        return bool(np.all(sub_a == int(bg)) and np.all(sub_b == int(bg)))

    def valid_overlap_area(i: int, j: int, dr: int, dc: int) -> int | None:
        """Return overlap area if it's an exact-match overlap with max(h,w)>=3, else None."""
        a = pieces[int(i)]["arr"]
        b = pieces[int(j)]["arr"]
        r0, c0, r1, c1 = overlap_rect(a, b, dr, dc)
        if r0 >= r1 or c0 >= c1:
            return None
        oh = int(r1 - r0)
        ow = int(c1 - c0)
        if max(int(oh), int(ow)) < 3:
            return None
        sub_a = a[int(r0) : int(r1), int(c0) : int(c1)]
        sub_b = b[int(r0 - dr) : int(r1 - dr), int(c0 - dc) : int(c1 - dc)]
        if not np.array_equal(sub_a, sub_b):
            return None
        return int(oh * ow)

    # Precompute best (max-area) overlap shifts between ordered pairs.
    n_p = int(len(pieces))
    best_area = [[0] * n_p for _ in range(n_p)]
    best_shifts: list[list[list[tuple[int, int]]]] = [[[] for _ in range(n_p)] for __ in range(n_p)]
    for i in range(n_p):
        for j in range(n_p):
            if int(i) == int(j):
                continue
            cand: set[tuple[int, int]] = set()
            for col, as_ in pieces[int(i)]["bycol"].items():
                bs_ = pieces[int(j)]["bycol"].get(int(col))
                if not bs_:
                    continue
                for ra, ca in as_:
                    for rb, cb in bs_:
                        cand.add((int(ra - rb), int(ca - cb)))

            best = 0
            goods: list[tuple[int, int]] = []
            for dr, dc in cand:
                area = valid_overlap_area(int(i), int(j), int(dr), int(dc))
                if area is None:
                    continue
                if int(area) > int(best):
                    best = int(area)
                    goods = [(int(dr), int(dc))]
                elif int(area) == int(best):
                    goods.append((int(dr), int(dc)))

            best_area[int(i)][int(j)] = int(best)
            best_shifts[int(i)][int(j)] = sorted(set(goods))

    # Root = largest bbox area (robust).
    root = int(max(range(n_p), key=lambda idx: int(pieces[int(idx)]["arr"].size)))

    def add_piece(canvas: dict[tuple[int, int], int], idx: int, r0: int, c0: int):
        # Only store non-bg pixels in canvas; bg will be filled at the end.
        for r, c, v in pieces[int(idx)]["nz"]:
            key = (int(r0 + r), int(c0 + c))
            if key in canvas and int(canvas[key]) != int(v):
                return None
            canvas[key] = int(v)
        return canvas

    def bbox_of(canvas: dict[tuple[int, int], int]) -> tuple[int, int, int, int]:
        rs = [r for r, _c in canvas]
        cs = [c for _r, c in canvas]
        return int(min(rs)), int(max(rs)), int(min(cs)), int(max(cs))

    def crop_canvas(canvas: dict[tuple[int, int], int]) -> np.ndarray:
        r0, r1, c0, c1 = bbox_of(canvas)
        out = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
        for (r, c), v in canvas.items():
            out[int(r - r0), int(c - c0)] = int(v)
        return out

    def pair_ok(i: int, j: int, pi: tuple[int, int], pj: tuple[int, int]) -> bool:
        """If pieces overlap meaningfully, the relative shift must be a best shift."""
        dr = int(pj[0] - pi[0])
        dc = int(pj[1] - pi[1])
        a = pieces[int(i)]["arr"]
        b = pieces[int(j)]["arr"]
        r0, c0, r1, c1 = overlap_rect(a, b, dr, dc)
        if r0 >= r1 or c0 >= c1:
            return True
        if overlap_all_bg(a, b, dr, dc):
            return True
        return (int(dr), int(dc)) in set(best_shifts[int(i)][int(j)])

    placed: dict[int, tuple[int, int]] = {int(root): (0, 0)}
    canvas: dict[tuple[int, int], int] = {}
    add_piece(canvas, int(root), 0, 0)

    def dfs(placed: dict[int, tuple[int, int]], canvas: dict[tuple[int, int], int]) -> np.ndarray | None:
        if int(len(placed)) == int(n_p):
            # Strict checks on every tree-edge pair (pieces whose bboxes overlap in the
            # final assembly). Both conditions must hold for the task to be well-formed;
            # the solver raises so that any caller (e.g. a generator's oracle check)
            # sees the problem loudly instead of a silently-picked alternative assembly.
            #   (a) the max-area shift between the pair must be UNIQUE (no ties);
            #   (b) the overlap zone must contain at least 2 non-bg pixels (otherwise
            #       the pattern is too weak to uniquely identify placement).
            for i in range(n_p):
                ri, ci = placed[int(i)]
                hi, wi = int(pieces[int(i)]["h"]), int(pieces[int(i)]["w"])
                for j in range(int(i) + 1, n_p):
                    rj, cj = placed[int(j)]
                    hj, wj = int(pieces[int(j)]["h"]), int(pieces[int(j)]["w"])
                    orr, orc = max(ri, rj), max(ci, cj)
                    err, erc = min(ri + hi, rj + hj), min(ci + wi, cj + wj)
                    if err <= orr or erc <= orc:
                        continue  # not a tree edge
                    if len(best_shifts[int(i)][int(j)]) > 1:
                        raise AssertionError(
                            f"ambiguous tree-edge overlap: pair ({i},{j}) has "
                            f"{len(best_shifts[int(i)][int(j)])} tied max-area shifts"
                        )
                    sub = pieces[int(i)]["arr"][orr - ri : err - ri, orc - ci : erc - ci]
                    nonbg = int((sub != int(bg)).sum())
                    if nonbg < 2:
                        raise AssertionError(
                            f"tree-edge overlap has only {nonbg} non-bg pixel(s) "
                            f"(need >=2): pair ({i},{j})"
                        )
            return crop_canvas(canvas)

        moves = []
        for i, (ri, ci) in placed.items():
            for j in range(n_p):
                if int(j) in placed:
                    continue
                for dr, dc in best_shifts[int(i)][int(j)]:
                    moves.append((int(best_area[int(i)][int(j)]), int(i), int(j), int(ri + dr), int(ci + dc)))

        if not moves:
            return None

        moves.sort(reverse=True)
        seen = set()

        for _area, _i, j, rj, cj in moves:
            if (int(j), int(rj), int(cj)) in seen:
                continue
            seen.add((int(j), int(rj), int(cj)))

            new_canvas = dict(canvas)
            if add_piece(new_canvas, int(j), int(rj), int(cj)) is None:
                continue

            ok = True
            for k, (rk, ck) in placed.items():
                if not pair_ok(int(k), int(j), (int(rk), int(ck)), (int(rj), int(cj))):
                    ok = False
                    break
                if not pair_ok(int(j), int(k), (int(rj), int(cj)), (int(rk), int(ck))):
                    ok = False
                    break
            if not ok:
                continue

            new_placed = dict(placed)
            new_placed[int(j)] = (int(rj), int(cj))
            res = dfs(new_placed, new_canvas)
            if res is not None:
                return res

        return None

    res = dfs(placed, canvas)
    assert not (res is None)
    return res

