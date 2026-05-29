# Combo variant of 446ef5d2 (index 22)
# combo_spec: D1/2+D3/1+D8/2
# Name: 旋转匹配拼正方形配合十字标记

from __future__ import annotations

from collections import Counter
from math import isqrt
import random

import numpy as np

from arc_utils import paste, random_free_location


# Design analysis
# ===============
# This fused variant combines three changes that interact rather than stack independently:
#
# a) Concreteization and behavior trace
# - Original task:
#   A 3-cell L marker points to one rectangle corner. The solver infers one anchored corner,
#   enumerates rectangle factor pairs from total piece area, and translates unrotated pieces.
# - Fused task:
#   The marker is now a 5-cell plus whose center marks the center of the destination, not a
#   corner. The destination must be a square, so total scattered piece area determines one
#   side length exactly: if area is 81, the destination must be 9x9, centered on the plus.
#   Pieces may also need quarter-turn rotation before edge colors match.
# - Small trace:
#   Suppose the plus center is at (10, 12) and the scattered pieces have total area 81.
#   Old logic: search for an L-shaped color with count 3, derive a corner, then test many
#   rectangle factor pairs such as 3x27, 9x9, 1x81.
#   New logic: detect a unique plus with count 5, compute side = sqrt(81) = 9, place the
#   target square at top-left (10-4, 12-4), then backtrack over piece placements and 0/90/180/270
#   rotations. A 2x4 piece scattered as 4x2 may only fit after rotating back by 90 degrees.
#
# b) Complexity calibration
# - The square constraint removes width/height ambiguity, which is important because the center
#   marker otherwise leaves no corner anchor. That reduction pays for the added rotation search.
# - The generator forces at least one visibly quarter-turned non-square piece, so the rotation
#   rule is observable in samples instead of being an optional edge case.
# - We keep only 1-3 interior colors and 4-12 pieces, preserving the original task's visual scale.
#
# c) Old assumptions that break
# - Marker detection via "a color appearing exactly 3 times in an L" is invalid; we now need
#   "a color appearing exactly 5 times in a plus".
# - Corner-derived top-left inference and anchor-piece preplacement are invalid because the marker
#   no longer identifies a visible corner cell that belongs to a scattered piece.
# - Fixed piece orientation is invalid because placement legality now depends on the chosen rotation.
# - Rectangle factor-pair enumeration is overkill and misleading now; the fused rule assumes the
#   total area is an odd perfect square, so one side length is implied directly by area.
#
# d) Placement legality
# - The generator still uses output-first construction, partitions the final square into rectangles,
#   rotates some scattered pieces, then places them with >=1 background cell padding.
# - The plus marker is placed first and pieces are scattered around it with padding, so the symbol
#   remains legible and never overlaps a piece in the input.
#
# e) Visual observability
# - Across examples, a human sees a plus marker near the middle of the eventual square, not near
#   a corner, and also sees at least one rectangular piece whose orientation changes between input
#   and output. The destination is visibly square in all outputs.
# - Because area -> side length is exact for squares, the center marker and piece area cooperate in
#   a way that is visually recoverable from a few examples.
#
# f) Visual cleanliness
# - The plus marker has a dedicated color and a sparse 3x3 footprint.
# - Pieces remain plain axis-aligned rectangles in the input, separated by background. No overlays
#   or extra annotations share the same cells.
#
# g) Color inferability
# - No arbitrary color remapping is introduced. Colors still mean: background, one marker color,
#   one dominant rect_col wildcard/border color, and 1-3 interior colors.
# - The generator rejects cases where any non-marker piece color appears exactly 5 times, avoiding
#   accidental confusion with the plus-marker detection rule.
#
# h) Degeneracy guard
# - The generator rejects samples that the same assembly logic can solve with rotations disabled.
#   This prevents silent fallback to the easier center-marked square-without-rotation task.
#
# i) New solver assumptions and generator guarantees
# - Assumption: the target area is an odd perfect square.
#   Guarantee: the generator samples only odd square side lengths.
# - Assumption: the plus marker is unique among all colors.
#   Guarantee: marker color appears exactly 5 times; no other non-marker color is allowed total count 5.
# - Assumption: every scattered object is still a full rectangle after rotation.
#   Guarantee: the generator partitions into rectangles, rotates whole rectangles only, then scatters.
# - Assumption: at least one rotation matters.
#   Guarantee: the generator forces at least one non-square piece to use a 90/270 degree rotation and
#   validates that a no-rotation solver no longer recovers the expected output.


def _find_plus_marker(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int]]:
    cnt = Counter(map(int, g.flatten()))
    for col in sorted(c for c, n in cnt.items() if int(c) != int(bg) and int(n) == 5):
        pts = {(int(r), int(c)) for r, c in np.argwhere(g == int(col))}
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if (r1 - r0) != 2 or (c1 - c0) != 2:
            continue
        center = (r0 + 1, c0 + 1)
        want = {
            center,
            (center[0] - 1, center[1]),
            (center[0] + 1, center[1]),
            (center[0], center[1] - 1),
            (center[0], center[1] + 1),
        }
        if pts == want:
            return int(col), center
    raise AssertionError("plus marker not found")


def _extract_rectangles(g: np.ndarray, occ: np.ndarray) -> list[tuple[int, int, int, int, np.ndarray]]:
    GH, GW = g.shape
    used = np.zeros_like(occ, dtype=bool)
    rects: list[tuple[int, int, int, int, np.ndarray]] = []
    while True:
        idx = np.argwhere(occ & (~used))
        if idx.size == 0:
            break
        rr, cc = map(int, idx[0])
        w = 0
        while cc + w < GW and bool(occ[rr, cc + w]) and not bool(used[rr, cc + w]):
            w += 1
        h = 0
        while rr + h < GH and bool((occ[rr + h, cc : cc + w] & (~used[rr + h, cc : cc + w])).all()):
            h += 1
        used[rr : rr + h, cc : cc + w] = True
        rects.append((rr, cc, h, w, g[rr : rr + h, cc : cc + w].copy()))
    return rects


def _unique_rotations(piece: np.ndarray, allow_rotation: bool) -> list[tuple[int, np.ndarray]]:
    ks = [0, 1, 2, 3] if allow_rotation else [0]
    seen: set[tuple[tuple[int, int], bytes]] = set()
    out: list[tuple[int, np.ndarray]] = []
    for k in ks:
        arr = np.rot90(piece, int(k))
        key = (tuple(map(int, arr.shape)), arr.tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append((int(k), arr))
    return out


def _top_left_from_center(center: tuple[int, int], side: int) -> tuple[int, int]:
    cr, cc = map(int, center)
    off = int(side) // 2
    return cr - off, cc - off


def _compat(v1: np.ndarray, v2: np.ndarray, rect_col: int) -> bool:
    a = np.array(v1, dtype=int)
    b = np.array(v2, dtype=int)
    m = (a != int(rect_col)) & (b != int(rect_col))
    return bool((a[m] == b[m]).all())


def _solve(grid: np.ndarray, allow_rotation: bool, return_meta: bool = False):
    g = np.array(grid, dtype=int)
    GH, GW = g.shape
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])

    marker_col, center = _find_plus_marker(g, bg)
    occ = (g != int(bg)) & (g != int(marker_col))
    rects = _extract_rectangles(g, occ)
    assert rects

    total_area = int(sum(int(h) * int(w) for _r, _c, h, w, _a in rects))
    side = int(isqrt(total_area))
    assert side * side == total_area
    assert side >= 5 and side % 2 == 1
    tr, tc = _top_left_from_center(center, side)
    assert 0 <= tr and 0 <= tc and tr + side <= GH and tc + side <= GW

    allpix: list[int] = []
    for _r, _c, _h, _w, a in rects:
        allpix.extend(map(int, a.flatten()))
    rect_col = int(Counter(allpix).most_common(1)[0][0])

    pieces = [a for _r, _c, _h, _w, a in rects]
    variants = [_unique_rotations(p, allow_rotation) for p in pieces]
    piece_sizes = [(int(p.shape[0]), int(p.shape[1])) for p in pieces]

    board = np.full((side, side), -1, dtype=int)
    used = [False] * len(pieces)
    chosen_k: list[int | None] = [None] * len(pieces)

    def boundary_ok(a: np.ndarray, r: int, c: int) -> bool:
        h, w = map(int, a.shape)
        if r == 0 and not np.all(a[0, :] == int(rect_col)):
            return False
        if r + h == side and not np.all(a[-1, :] == int(rect_col)):
            return False
        if c == 0 and not np.all(a[:, 0] == int(rect_col)):
            return False
        if c + w == side and not np.all(a[:, -1] == int(rect_col)):
            return False
        return True

    def neighbor_ok(a: np.ndarray, r: int, c: int) -> bool:
        h, w = map(int, a.shape)
        if c > 0:
            left = board[r : r + h, c - 1]
            m = left != -1
            if m.any() and not _compat(left[m], a[:, 0][m], rect_col):
                return False
        if c + w < side:
            right = board[r : r + h, c + w]
            m = right != -1
            if m.any() and not _compat(right[m], a[:, -1][m], rect_col):
                return False
        if r > 0:
            up = board[r - 1, c : c + w]
            m = up != -1
            if m.any() and not _compat(up[m], a[0, :][m], rect_col):
                return False
        if r + h < side:
            down = board[r + h, c : c + w]
            m = down != -1
            if m.any() and not _compat(down[m], a[-1, :][m], rect_col):
                return False
        return True

    def can_place(a: np.ndarray, r: int, c: int) -> bool:
        h, w = map(int, a.shape)
        if r + h > side or c + w > side:
            return False
        if (board[r : r + h, c : c + w] != -1).any():
            return False
        return boundary_ok(a, r, c) and neighbor_ok(a, r, c)

    def place(i: int, k: int, a: np.ndarray, r: int, c: int) -> None:
        h, w = map(int, a.shape)
        board[r : r + h, c : c + w] = a
        used[i] = True
        chosen_k[i] = int(k)

    def unplace(i: int, a: np.ndarray, r: int, c: int) -> None:
        h, w = map(int, a.shape)
        board[r : r + h, c : c + w] = -1
        used[i] = False
        chosen_k[i] = None

    order = sorted(
        range(len(pieces)),
        key=lambda i: (
            -piece_sizes[i][0] * piece_sizes[i][1],
            len(variants[i]),
            -max(piece_sizes[i]),
            -min(piece_sizes[i]),
            i,
        ),
    )

    def next_empty() -> tuple[int, int] | None:
        idx = np.argwhere(board == -1)
        if idx.size == 0:
            return None
        return tuple(map(int, idx[0]))

    solved = False

    def bt() -> None:
        nonlocal solved
        nxt = next_empty()
        if nxt is None:
            solved = True
            return
        r, c = nxt
        for i in order:
            if used[i]:
                continue
            for k, a in variants[i]:
                if can_place(a, r, c):
                    place(i, k, a, r, c)
                    bt()
                    if solved:
                        return
                    unplace(i, a, r, c)

    bt()
    assert solved

    out = np.full((GH, GW), int(bg), dtype=int)
    out[tr : tr + side, tc : tc + side] = board
    if return_meta:
        rots = [0 if k is None else int(k) for k in chosen_k]
        quarter_turn_non_square = sum(
            1
            for i, k in enumerate(rots)
            if k in (1, 3) and piece_sizes[i][0] != piece_sizes[i][1]
        )
        return out, {
            "side": int(side),
            "piece_rotations": rots,
            "rotated_piece_count": sum(1 for k in rots if k != 0),
            "quarter_turn_non_square_count": int(quarter_turn_non_square),
            "piece_sizes": piece_sizes,
        }
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered rectangular pieces into an odd square centered on a 5-cell plus marker.

    Rule:
    - The plus center gives the square center.
    - The total scattered-piece area is the square area, so the side length is sqrt(area).
    - Piece borders touching the square exterior must be rect_col.
    - Touching piece edges match on every non-rect_col pixel; rect_col is a wildcard.
    - Pieces may need rotation by 90/180/270 degrees before placement.
    """
    return _solve(grid, allow_rotation=True)


def _center_marker_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    cr, cc = map(int, center)
    return [(cr, cc), (cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)]


def _build_square(side: int, rect_col: int, interior_cols: list[int]) -> np.ndarray | None:
    rect = np.full((side, side), int(rect_col), dtype=int)
    iH, iW = side - 2, side - 2
    if iH < 1 or iW < 1:
        return None

    interior = np.full((iH, iW), int(rect_col), dtype=int)
    target_fill = random.uniform(0.38, 0.82)
    target_pixels = max(1, int(target_fill * iH * iW))
    blob_occupied = np.zeros((iH, iW), dtype=bool)

    for ci, color in enumerate(interior_cols):
        if ci == len(interior_cols) - 1:
            budget = target_pixels - int(blob_occupied.sum())
        else:
            budget = max(1, target_pixels // len(interior_cols))
        budget = max(1, int(budget))

        seed = None
        for _ in range(200):
            sr, sc = random.randint(0, iH - 1), random.randint(0, iW - 1)
            if blob_occupied[sr, sc]:
                continue
            ok = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = sr + dr, sc + dc
                    if 0 <= nr < iH and 0 <= nc < iW and blob_occupied[nr, nc]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                seed = (sr, sc)
                break
        if seed is None:
            continue

        blob = {seed}
        frontier = [seed]
        while len(blob) < budget and frontier:
            random.shuffle(frontier)
            new_frontier = []
            for fr, fc in frontier:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = fr + dr, fc + dc
                    if (nr, nc) in blob:
                        continue
                    if nr < 0 or nr >= iH or nc < 0 or nc >= iW:
                        continue
                    if blob_occupied[nr, nc]:
                        continue
                    conflict = False
                    for dr2 in (-1, 0, 1):
                        for dc2 in (-1, 0, 1):
                            if dr2 == 0 and dc2 == 0:
                                continue
                            ar, ac = nr + dr2, nc + dc2
                            if 0 <= ar < iH and 0 <= ac < iW and blob_occupied[ar, ac] and (ar, ac) not in blob:
                                conflict = True
                                break
                        if conflict:
                            break
                    if conflict:
                        continue
                    blob.add((nr, nc))
                    new_frontier.append((nr, nc))
                    if len(blob) >= budget:
                        break
                if len(blob) >= budget:
                    break
            frontier = new_frontier

        for br, bc in blob:
            interior[br, bc] = int(color)
            blob_occupied[br, bc] = True

    rect[1:-1, 1:-1] = interior

    rc = side // 2
    for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
        rect[rc + dr, rc + dc] = int(rect_col)

    fill_ratio = float(np.mean(rect[1:-1, 1:-1] != int(rect_col)))
    if fill_ratio < 0.28 or fill_ratio > 0.88:
        return None

    counts = Counter(map(int, rect.flatten()))
    for col, n in counts.items():
        if int(col) != int(rect_col) and int(n) == 5:
            return None

    return rect


def _rand_splits(total: int, min_s: int = 2, max_s: int = 7) -> list[int] | None:
    eff_max = min(max_s, max(min_s, random.randint(min_s, max_s)))
    segs: list[int] = []
    rem = int(total)
    while rem > 0:
        if rem <= eff_max:
            if rem < min_s:
                return None
            segs.append(int(rem))
            rem = 0
        else:
            hi = min(eff_max, rem - min_s)
            if min_s > hi:
                return None
            s = random.randint(min_s, hi)
            segs.append(int(s))
            rem -= s
    return segs


def _partition_square(side: int) -> list[tuple[int, int, int, int]] | None:
    row_segs = _rand_splits(side)
    col_segs = _rand_splits(side)
    if row_segs is None or col_segs is None:
        return None

    row_starts: list[int] = []
    s = 0
    for rs in row_segs:
        row_starts.append(int(s))
        s += rs
    col_starts: list[int] = []
    s = 0
    for cs in col_segs:
        col_starts.append(int(s))
        s += cs

    nr, nc = len(row_segs), len(col_segs)
    cell_id = [[i * nc + j for j in range(nc)] for i in range(nr)]
    n_cells = nr * nc
    if n_cells < 4:
        return None
    target_pieces = random.randint(4, min(12, n_cells))

    def get_bbox(pid: int) -> tuple[int, int, int, int] | None:
        cells = [(i, j) for i in range(nr) for j in range(nc) if cell_id[i][j] == pid]
        if not cells:
            return None
        ri = [r for r, _ in cells]
        ci = [c for _, c in cells]
        return min(ri), min(ci), max(ri), max(ci)

    def pixel_size(r0i: int, c0i: int, r1i: int, c1i: int) -> tuple[int, int]:
        h = sum(row_segs[i] for i in range(r0i, r1i + 1))
        w = sum(col_segs[j] for j in range(c0i, c1i + 1))
        return int(h), int(w)

    merges = []
    for i in range(nr):
        for j in range(nc):
            if j + 1 < nc:
                merges.append(("h", i, j))
            if i + 1 < nr:
                merges.append(("v", i, j))
    random.shuffle(merges)

    def count_pieces() -> int:
        return len({cell_id[i][j] for i in range(nr) for j in range(nc)})

    for direction, mi, mj in merges:
        if count_pieces() <= target_pieces:
            break
        pid1 = cell_id[mi][mj]
        pid2 = cell_id[mi][mj + 1] if direction == "h" else cell_id[mi + 1][mj]
        if pid1 == pid2:
            continue
        bb1 = get_bbox(pid1)
        bb2 = get_bbox(pid2)
        if bb1 is None or bb2 is None:
            continue
        mr0 = min(bb1[0], bb2[0])
        mc0 = min(bb1[1], bb2[1])
        mr1 = max(bb1[2], bb2[2])
        mc1 = max(bb1[3], bb2[3])
        ph, pw = pixel_size(mr0, mc0, mr1, mc1)
        if ph > 7 or pw > 7:
            continue
        ok = True
        for ii in range(mr0, mr1 + 1):
            for jj in range(mc0, mc1 + 1):
                if cell_id[ii][jj] not in (pid1, pid2):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        for ii in range(nr):
            for jj in range(nc):
                if cell_id[ii][jj] == pid2:
                    cell_id[ii][jj] = pid1

    pieces: list[tuple[int, int, int, int]] = []
    seen = set()
    for i in range(nr):
        for j in range(nc):
            pid = cell_id[i][j]
            if pid in seen:
                continue
            seen.add(pid)
            bb = get_bbox(pid)
            assert bb is not None
            r0i, c0i, r1i, c1i = bb
            pr = row_starts[r0i]
            pc = col_starts[c0i]
            ph = sum(row_segs[ii] for ii in range(r0i, r1i + 1))
            pw = sum(col_segs[jj] for jj in range(c0i, c1i + 1))
            pieces.append((int(pr), int(pc), int(ph), int(pw)))

    if not (4 <= len(pieces) <= 12):
        return None
    return pieces


def _piece_rotation_options(piece: np.ndarray) -> list[int]:
    out = []
    for k in (1, 2, 3):
        if not np.array_equal(np.rot90(piece, int(k)), piece):
            out.append(int(k))
    return out


def _choose_rotations(piece_arrays: list[np.ndarray]) -> list[int] | None:
    n = len(piece_arrays)
    ks = [0] * n
    quarter_turn_candidates = [
        i
        for i, p in enumerate(piece_arrays)
        if p.shape[0] != p.shape[1] and any(k in (1, 3) for k in _piece_rotation_options(p))
    ]
    if not quarter_turn_candidates:
        return None

    forced = random.choice(quarter_turn_candidates)
    forced_opts = [k for k in (1, 3) if k in _piece_rotation_options(piece_arrays[forced])]
    if not forced_opts:
        return None
    ks[forced] = random.choice(forced_opts)

    others = [i for i in range(n) if i != forced and _piece_rotation_options(piece_arrays[i])]
    random.shuffle(others)
    for i in others:
        opts = _piece_rotation_options(piece_arrays[i])
        if not opts:
            continue
        if random.random() < 0.55:
            ks[i] = random.choice(opts)

    if sum(1 for k in ks if k != 0) == 0:
        return None
    if not any(piece_arrays[i].shape[0] != piece_arrays[i].shape[1] and ks[i] in (1, 3) for i in range(n)):
        return None
    return ks


def _gen_attempt() -> np.ndarray | None:
    GH = random.randint(10, 26)
    GW = random.randint(10, 26)

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    rect_col = palette[1]
    marker_col = palette[2]
    n_interior = random.randint(1, 3)
    interior_cols = palette[3 : 3 + n_interior]

    max_side = min(21, GH, GW)
    odd_sides = [s for s in range(7, max_side + 1, 2)]
    if not odd_sides:
        return None
    side = random.choice(odd_sides)
    if random.random() < 0.6:
        larger = [s for s in odd_sides if s >= 9]
        if larger:
            side = random.choice(larger)

    centers = [
        (cr, cc)
        for cr in range(side // 2, GH - side // 2)
        for cc in range(side // 2, GW - side // 2)
    ]
    if not centers:
        return None
    center = random.choice(centers)

    rect = _build_square(side, rect_col, interior_cols)
    if rect is None:
        return None

    pieces = _partition_square(side)
    if pieces is None:
        return None

    piece_arrays = [rect[pr : pr + ph, pc : pc + pw].copy() for pr, pc, ph, pw in pieces]
    if not any(p.shape[0] != p.shape[1] for p in piece_arrays):
        return None

    for pr, pc, ph, pw in pieces:
        piece = rect[pr : pr + ph, pc : pc + pw]
        if ph < 2 or pw < 2 or ph > 7 or pw > 7:
            return None
        fill = float(np.mean(piece != int(rect_col)))
        if fill < 0.10 or fill > 0.60:
            return None
        if not any(np.any(edge != int(rect_col)) for edge in [piece[0, :], piece[-1, :], piece[:, 0], piece[:, -1]]):
            return None
        if pr == 0 and not np.all(piece[0, :] == int(rect_col)):
            return None
        if pr + ph == side and not np.all(piece[-1, :] == int(rect_col)):
            return None
        if pc == 0 and not np.all(piece[:, 0] == int(rect_col)):
            return None
        if pc + pw == side and not np.all(piece[:, -1] == int(rect_col)):
            return None

    rotations = _choose_rotations(piece_arrays)
    if rotations is None:
        return None

    canvas = np.full((GH, GW), int(bg), dtype=int)
    for mr, mc in _center_marker_cells(center):
        if mr < 0 or mr >= GH or mc < 0 or mc >= GW:
            return None
        canvas[mr, mc] = int(marker_col)

    order = list(range(len(piece_arrays)))
    random.shuffle(order)
    for idx in order:
        pa = np.rot90(piece_arrays[idx], int(rotations[idx]))
        try:
            r0, c0 = random_free_location(canvas, pa, transparent=bg, padding=1)
        except ValueError:
            return None
        paste(canvas, pa, r0, c0)

    tr, tc = _top_left_from_center(center, side)
    expected = np.full((GH, GW), int(bg), dtype=int)
    expected[tr : tr + side, tc : tc + side] = rect

    try:
        actual, meta = _solve(canvas, allow_rotation=True, return_meta=True)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if int(meta["quarter_turn_non_square_count"]) < 1:
        return None

    try:
        no_rot = _solve(canvas, allow_rotation=False)
    except Exception:
        no_rot = None
    if no_rot is not None and np.array_equal(no_rot, expected):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(500):
        canvas = _gen_attempt()
        if canvas is not None:
            return canvas
    raise RuntimeError("generate_input: failed after 500 attempts")
