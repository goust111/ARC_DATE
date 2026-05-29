# Combo variant of 446ef5d2 (index 11)
# combo_spec: D3/3+D10/2
# Name: 无标记角配合部分外边约束

from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
# ---------------------------------------------------------------------------
#
# combo_intent:
#   - D3/3 removes the original 3-cell L marker entirely.
#   - D10/2 weakens the border rule: only one adjacent pair of outer sides is forced
#     to be pure rect_col; the other two outer sides must visibly expose non-rect_col.
#   - The two changes must cooperate, not merely coexist:
#     once the marker disappears, the identity of the "special corner" can only be
#     recovered from which adjacent pair of assembled outer sides stays pure.
#
# concrete_pixel_behavior:
#   input:
#     - Background is the most common grid color.
#     - Several non-overlapping rectangular fragments are scattered on the background.
#     - There is no separate marker object, and we additionally reject accidental
#       3-cell L-shaped color triplets so the removed marker does not visually reappear.
#     - rect_col is still the most common color across all fragment pixels.
#   output:
#     - The fragments are translated (never rotated/reflected) into one rectangle.
#     - Exactly two adjacent outer sides of that rectangle are fully rect_col.
#     - The other two outer sides each contain some non-rect_col pixels.
#     - Because removing the marker makes absolute placement ambiguous, we adopt the
#       same minimal anchoring used in the standalone D3 variant:
#       the solved rectangle's top-left is aligned to the top-left of the input pieces'
#       occupied bbox. This anchor resolves only translation, not orientation.
#
# behavior_trace:
#   small_example:
#     - Original task:
#       a marker gap might explicitly tell us "the rectangle's TL corner is at (5,7)".
#       The solver then only needs to tile pieces while enforcing all four sides pure.
#     - D10-only task:
#       the marker still says which corner matters, but only the two sides touching
#       that corner must be pure; the opposite sides may show colored leaks.
#     - This combo:
#       there is no marker at all. Suppose the input pieces' bbox starts at (3,4).
#       The final rectangle must also start at (3,4), but the solver does not know
#       whether the pure pair is {top,left}, {top,right}, {bottom,left}, or
#       {bottom,right}. It must try candidate rectangle dimensions and candidate pure
#       side pairs, assemble the pieces, and keep only completions where:
#         1. internal seams match under rect_col wildcard compatibility;
#         2. exactly one adjacent side pair is fully rect_col;
#         3. both remaining sides visibly contain non-rect_col pixels.
#     - Difference from the original:
#       the original pipeline was "detect marker -> know corner/orientation -> pack".
#       The combo pipeline is "anchor by bbox -> infer corner/orientation from edge
#       purity asymmetry -> pack".
#
# complexity_calibration:
#   - Removing the anchor marker and weakening border constraints both increase search.
#     To keep the task solvable, the generator narrows the rectangle size and piece
#     count relative to the original: typically 4-9 pieces and rectangle sides <= 16.
#   - To make the D10 change observable instead of occasionally accidental, the
#     generator forces each free outer side to contain a colored band; the combo never
#     quietly degenerates back to the original "all four sides pure" family.
#   - To make the D3 change observable, the generator rejects any accidental marker-like
#     3-pixel L color pattern in the scattered input.
#   - To reduce ambiguity, the generator checks the piece multiset with the same solver
#     search used by main() and rejects samples unless the assembled rectangle is the
#     only distinct completion across all candidate dimensions and all four pure-corner
#     hypotheses.
#
# implicit_assumption_review:
#   original_assumptions_broken:
#     - The original `cnt == 3` marker detection and missing-corner mapping are invalid.
#     - The original "anchor piece contains the marker corner" pre-placement branch is
#       invalid because there is no absolute corner cell in the input anymore.
#     - The original `fits(H,W)` logic tied candidate dimensions to the marker gap;
#       now dimensions are only bounded by the fixed bbox anchor.
#   new_solver_assumptions:
#     - Every non-bg component is a filled axis-aligned rectangle. The generator keeps
#       one-cell background gaps between scattered pieces so rectangle extraction is safe.
#     - rect_col remains the mode of all piece pixels. The generator preserves this by
#       keeping much of the rectangle interior and both pure sides in rect_col.
#     - The solved rectangle is written with top-left equal to the input occupied bbox
#       top-left. The generator guarantees this by pinning one piece at that location and
#       placing all other pieces inside the sub-canvas to its lower-right.
#     - Each free outer side contains at least one non-rect_col pixel. main() depends on
#       this as part of candidate validation; the generator explicitly paints such bands.
#     - The intended assembly is globally unique up to identical-piece permutations that
#       yield the same final board. The generator enforces uniqueness at board level.
#
# placement_legality:
#   - The assembled rectangle is created first, then partitioned into rectangular pieces.
#   - Each piece stays within 2..7 cells per side, matching the original family.
#   - When scattering, one piece is pinned at the target anchor `(tr, tc)` and the rest
#     are placed via `random_free_location(..., padding=1)` in the sub-canvas
#     `canvas[tr:, tc:]`, ensuring no overlaps, no touching, no cells above/left of the
#     anchor, and therefore a stable occupied bbox.
#   - Colored free-edge bands are painted so they never overwrite cells on pure sides,
#     including the corners shared with a pure side.
#
# visual_observability_and_cleanliness:
#   - Humans can see that no marker exists, yet the solved outputs repeatedly show only
#     one adjacent pair of clean rect_col borders; this repeated asymmetry is the clue.
#   - The grid remains tidy: background + rectangular fragments in input, one rectangle
#     in output, no layered annotations, no hidden use of color ids or sorting tricks.
#   - The number of colors stays modest (bg + rect_col + 1..3 interior colors), and the
#     free-edge leaks are contiguous bands rather than isolated noise, so the difference
#     from the original task is visually legible across a few examples.
#
# post_change_main_audit:
#   - `boundary_ok` now checks only the hypothesized pure sides; this assumes the free
#     sides are allowed to be mixed, which the generator guarantees.
#   - `final_edges_ok` requires both free sides to be non-pure, preventing accidental
#     fallback to the original four-pure-side behavior.
#   - The global solver enumerates all four adjacent-side hypotheses because no marker
#     chooses one in advance; the generator verifies exactly one distinct final board
#     survives this search.
#

CORNER_ORDER = ("TL", "TR", "BL", "BR")
ALL_SIDES = ("top", "bottom", "left", "right")


def _extract_rectangles(g: np.ndarray, bg: int) -> list[tuple[int, int, int, int, np.ndarray]]:
    """Extract scattered rectangular pieces from non-background cells."""
    occ = g != int(bg)
    used = np.zeros_like(occ, dtype=bool)
    rects: list[tuple[int, int, int, int, np.ndarray]] = []
    GH, GW = g.shape
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


def _candidate_dims(total_area: int, max_h: int, max_w: int) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    for hh in range(1, int(total_area**0.5) + 1):
        if total_area % hh != 0:
            continue
        ww = total_area // hh
        for H, W in ((hh, ww), (ww, hh)):
            if 5 <= H <= max_h and 5 <= W <= max_w:
                dims.append((int(H), int(W)))
    return sorted(set(dims), key=lambda t: (-min(t), -max(t), -t[0], -t[1]))


def _corner_side_sets(corner: str) -> tuple[set[str], set[str]]:
    if corner == "TL":
        return {"top", "left"}, {"bottom", "right"}
    if corner == "TR":
        return {"top", "right"}, {"bottom", "left"}
    if corner == "BL":
        return {"bottom", "left"}, {"top", "right"}
    return {"bottom", "right"}, {"top", "left"}


def _edge_values(arr: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return arr[0, :]
    if side == "bottom":
        return arr[-1, :]
    if side == "left":
        return arr[:, 0]
    return arr[:, -1]


def _compat(v1: np.ndarray, v2: np.ndarray, rect_col: int) -> bool:
    mask = (v1 != int(rect_col)) & (v2 != int(rect_col))
    return bool((v1[mask] == v2[mask]).all())


def _assemble_rectangle(
    pieces: list[np.ndarray],
    rect_col: int,
    H: int,
    W: int,
    pure_sides: set[str],
    *,
    solution_limit: int = 2,
) -> tuple[np.ndarray | None, int, int]:
    """
    Backtracking packer for one rectangle size and one pure-side hypothesis.

    Returns:
      (first_solution_board_or_None, number_of_solutions_up_to_limit, free_edge_score)
    """
    if not pieces:
        return None, 0, -1

    shapes = [(int(p.shape[0]), int(p.shape[1])) for p in pieces]
    if any(h > H or w > W for h, w in shapes):
        return None, 0, -1

    side_ok = {
        "top": [bool(np.all(p[0, :] == int(rect_col))) for p in pieces],
        "bottom": [bool(np.all(p[-1, :] == int(rect_col))) for p in pieces],
        "left": [bool(np.all(p[:, 0] == int(rect_col))) for p in pieces],
        "right": [bool(np.all(p[:, -1] == int(rect_col))) for p in pieces],
    }
    free_sides = [side for side in ALL_SIDES if side not in pure_sides]

    order = sorted(
        range(len(pieces)),
        key=lambda i: (
            -shapes[i][0] * shapes[i][1],
            -sum(int(np.sum(_edge_values(pieces[i], side) != int(rect_col))) for side in free_sides),
            -shapes[i][0],
            -shapes[i][1],
            i,
        ),
    )

    board = np.full((H, W), -1, dtype=int)
    used = [False] * len(pieces)
    first_solution: np.ndarray | None = None
    solution_count = 0
    first_score = -1

    def boundary_ok(i: int, r: int, c: int) -> bool:
        h, w = shapes[i]
        if "top" in pure_sides and r == 0 and not side_ok["top"][i]:
            return False
        if "bottom" in pure_sides and r + h == H and not side_ok["bottom"][i]:
            return False
        if "left" in pure_sides and c == 0 and not side_ok["left"][i]:
            return False
        if "right" in pure_sides and c + w == W and not side_ok["right"][i]:
            return False
        return True

    def neighbor_ok(i: int, r: int, c: int) -> bool:
        a = pieces[i]
        h, w = shapes[i]
        if c > 0:
            left = board[r : r + h, c - 1]
            mask = left != -1
            if mask.any() and not _compat(left[mask], a[:, 0][mask], rect_col):
                return False
        if c + w < W:
            right = board[r : r + h, c + w]
            mask = right != -1
            if mask.any() and not _compat(right[mask], a[:, -1][mask], rect_col):
                return False
        if r > 0:
            up = board[r - 1, c : c + w]
            mask = up != -1
            if mask.any() and not _compat(up[mask], a[0, :][mask], rect_col):
                return False
        if r + h < H:
            down = board[r + h, c : c + w]
            mask = down != -1
            if mask.any() and not _compat(down[mask], a[-1, :][mask], rect_col):
                return False
        return True

    def can_place(i: int, r: int, c: int) -> bool:
        h, w = shapes[i]
        if r + h > H or c + w > W:
            return False
        if (board[r : r + h, c : c + w] != -1).any():
            return False
        return boundary_ok(i, r, c) and neighbor_ok(i, r, c)

    def place(i: int, r: int, c: int) -> None:
        a = pieces[i]
        h, w = shapes[i]
        board[r : r + h, c : c + w] = a
        used[i] = True

    def unplace(i: int, r: int, c: int) -> None:
        h, w = shapes[i]
        board[r : r + h, c : c + w] = -1
        used[i] = False

    def next_empty() -> tuple[int, int] | None:
        idx = np.argwhere(board == -1)
        if idx.size == 0:
            return None
        return int(idx[0, 0]), int(idx[0, 1])

    def final_edges_ok() -> tuple[bool, int]:
        if np.any(board == -1):
            return False, -1
        for side in pure_sides:
            if not np.all(_edge_values(board, side) == int(rect_col)):
                return False, -1
        score = 0
        for side in free_sides:
            edge = _edge_values(board, side)
            if np.all(edge == int(rect_col)):
                return False, -1
            score += int(np.sum(edge != int(rect_col)))
        return True, score

    def bt() -> bool:
        nonlocal first_solution, solution_count, first_score
        nxt = next_empty()
        if nxt is None:
            ok, score = final_edges_ok()
            if ok:
                solution_count += 1
                if first_solution is None:
                    first_solution = board.copy()
                    first_score = int(score)
                return solution_count >= solution_limit
            return False

        r, c = nxt
        for i in order:
            if used[i] or not can_place(i, r, c):
                continue
            place(i, r, c)
            if bt():
                return True
            unplace(i, r, c)
        return False

    bt()
    return first_solution, solution_count, first_score


def _candidate_key(board: np.ndarray, H: int, W: int) -> tuple[int, int, bytes]:
    return int(H), int(W), board.tobytes()


def _enumerate_solution_records(
    pieces: list[np.ndarray],
    rect_col: int,
    max_h: int,
    max_w: int,
) -> list[dict]:
    total_area = int(sum(int(p.shape[0]) * int(p.shape[1]) for p in pieces))
    dims = _candidate_dims(total_area, max_h, max_w)
    records: list[dict] = []
    for corner in CORNER_ORDER:
        pure_sides, free_sides = _corner_side_sets(corner)
        for H, W in dims:
            board, count, score = _assemble_rectangle(pieces, rect_col, H, W, pure_sides, solution_limit=2)
            if board is None or count == 0:
                continue
            records.append(
                {
                    "corner": corner,
                    "pure_sides": pure_sides,
                    "free_sides": free_sides,
                    "H": int(H),
                    "W": int(W),
                    "board": board,
                    "count": int(count),
                    "score": int(score),
                }
            )
    return records


def _select_solution(
    pieces: list[np.ndarray],
    rect_col: int,
    max_h: int,
    max_w: int,
) -> tuple[np.ndarray | None, tuple[int, int] | None, list[dict], dict]:
    """
    Select the intended completion from all dimension/corner hypotheses.

    Preference order:
      1. candidates whose own backtracking completion is unique (`count == 1`);
      2. larger free-edge exposure score (generator deliberately paints free edges);
      3. dimension ordering biased toward chunkier rectangles, matching D3.
    """
    records = _enumerate_solution_records(pieces, rect_col, max_h, max_w)
    if not records:
        return None, None, records, {}

    pool = [rec for rec in records if rec["count"] == 1]
    if not pool:
        pool = records

    dedup: dict[tuple[int, int, bytes], dict] = {}
    for rec in pool:
        key = _candidate_key(rec["board"], rec["H"], rec["W"])
        prev = dedup.get(key)
        if prev is None or (
            rec["score"],
            min(rec["H"], rec["W"]),
            max(rec["H"], rec["W"]),
            -CORNER_ORDER.index(rec["corner"]),
        ) > (
            prev["score"],
            min(prev["H"], prev["W"]),
            max(prev["H"], prev["W"]),
            -CORNER_ORDER.index(prev["corner"]),
        ):
            dedup[key] = rec

    best = sorted(
        dedup.values(),
        key=lambda rec: (
            -rec["score"],
            -min(rec["H"], rec["W"]),
            -max(rec["H"], rec["W"]),
            CORNER_ORDER.index(rec["corner"]),
        ),
    )[0]
    return best["board"], (best["H"], best["W"]), records, dedup


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered rectangular pieces into one rectangle without any explicit marker.

    Combo rule:
    - Output position is anchored to the top-left of the input pieces' occupied bbox.
    - rect_col remains the wildcard for interior seam compatibility.
    - The solver must infer which adjacent pair of outer sides is pure rect_col.
    - The other two outer sides must each expose at least one non-rect_col pixel.
    """
    g = np.array(grid, dtype=int)
    GH, GW = g.shape

    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    rects = _extract_rectangles(g, bg)
    assert rects

    pieces = [a for _r, _c, _h, _w, a in rects]
    allpix = [int(x) for a in pieces for x in a.flatten()]
    rect_col = int(Counter(allpix).most_common(1)[0][0])

    top_r = min(int(r) for r, _c, _h, _w, _a in rects)
    top_c = min(int(c) for _r, c, _h, _w, _a in rects)

    assembled, used_dim, _records, _dedup = _select_solution(pieces, rect_col, GH - top_r, GW - top_c)
    assert assembled is not None and used_dim is not None

    H, W = used_dim
    out = np.full((GH, GW), int(bg), dtype=int)
    out[top_r : top_r + H, top_c : top_c + W] = assembled
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _has_marker_like_triplet(grid: np.ndarray, bg: int) -> bool:
    """Reject accidental 3-cell L triplets so the removed marker does not visually persist."""
    cnt = Counter(map(int, grid.flatten()))
    for col, n in cnt.items():
        if int(col) == int(bg) or int(n) != 3:
            continue
        pts = [(int(r), int(c)) for r, c in np.argwhere(grid == int(col))]
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        if max(rs) - min(rs) != 1 or max(cs) - min(cs) != 1:
            continue
        all4 = {
            (min(rs), min(cs)),
            (min(rs), max(cs)),
            (max(rs), min(cs)),
            (max(rs), max(cs)),
        }
        if len(all4 - set(pts)) == 1:
            return True
    return False


def _rand_splits(total: int, rng, min_s: int = 2, max_s: int = 6) -> list[int] | None:
    """Split `total` into segments each in [min_s, max_s]."""
    eff_max = min(max_s, max(min_s, rng.randint(min_s, max_s)))
    segs: list[int] = []
    rem = int(total)
    while rem > 0:
        if rem <= eff_max:
            if rem < min_s:
                return None
            segs.append(rem)
            break
        hi = min(eff_max, rem - min_s)
        if min_s > hi:
            return None
        s = rng.randint(min_s, hi)
        segs.append(s)
        rem -= s
    return segs


def _paint_band(
    rect: np.ndarray,
    reserved: np.ndarray,
    side: str,
    color: int,
    pure_sides: set[str],
    rng,
) -> bool:
    """Paint a contiguous colored band on one free outer side, avoiding pure-side corners."""
    RH, RW = rect.shape
    if side in ("top", "bottom"):
        usable = RW - (1 if "left" in pure_sides else 0) - (1 if "right" in pure_sides else 0)
        if usable < 2 or RH < 3:
            return False
        length = rng.randint(2, min(5, usable))
        depth = rng.randint(2, min(3, RH - 1))
        c_min = 1 if "left" in pure_sides else 0
        c_max = RW - length - (1 if "right" in pure_sides else 0)
        if c_min > c_max:
            return False
        starts = list(range(c_min, c_max + 1))
        rng.shuffle(starts)
        for start in starts:
            cols = slice(start, start + length)
            rows = slice(0, depth) if side == "top" else slice(RH - depth, RH)
            if reserved[rows, cols].any():
                continue
            rect[rows, cols] = int(color)
            reserved[rows, cols] = True
            return True
        return False

    usable = RH - (1 if "top" in pure_sides else 0) - (1 if "bottom" in pure_sides else 0)
    if usable < 2 or RW < 3:
        return False
    length = rng.randint(2, min(5, usable))
    depth = rng.randint(2, min(3, RW - 1))
    r_min = 1 if "top" in pure_sides else 0
    r_max = RH - length - (1 if "bottom" in pure_sides else 0)
    if r_min > r_max:
        return False
    starts = list(range(r_min, r_max + 1))
    rng.shuffle(starts)
    for start in starts:
        rows = slice(start, start + length)
        cols = slice(0, depth) if side == "left" else slice(RW - depth, RW)
        if reserved[rows, cols].any():
            continue
        rect[rows, cols] = int(color)
        reserved[rows, cols] = True
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    import random as _r

    GH = _r.randint(12, 26)
    GW = _r.randint(12, 26)

    palette = list(range(10))
    _r.shuffle(palette)
    bg = palette[0]
    rect_col = palette[1]
    n_interior = _r.randint(1, 3)
    interior_cols = palette[2 : 2 + n_interior]

    pure_corner = _r.choice(CORNER_ORDER)
    pure_sides, free_sides = _corner_side_sets(pure_corner)

    max_side = min(16, GH - 2, GW - 2)
    if max_side < 5:
        return None
    RH = _r.randint(5, max_side)
    RW = _r.randint(5, max_side)
    if RH * RW > int(0.42 * GH * GW):
        return None

    tr = _r.randint(0, GH - RH)
    tc = _r.randint(0, GW - RW)
    if (GH - tr) * (GW - tc) < int(1.35 * RH * RW):
        return None

    rect = np.full((RH, RW), rect_col, dtype=int)
    iH, iW = RH - 2, RW - 2
    if iH < 1 or iW < 1:
        return None

    interior = np.full((iH, iW), rect_col, dtype=int)
    target_fill = _r.uniform(0.40, 0.82)
    target_pixels = max(1, int(target_fill * iH * iW))
    blob_occupied = np.zeros((iH, iW), dtype=bool)

    for ci, ic in enumerate(interior_cols):
        if ci == len(interior_cols) - 1:
            budget = target_pixels - int(blob_occupied.sum())
        else:
            budget = max(2, target_pixels // len(interior_cols))
        if budget <= 0:
            budget = 2

        attempts = 0
        seed = None
        while attempts < 200:
            sr, sc = _r.randint(0, iH - 1), _r.randint(0, iW - 1)
            if blob_occupied[sr, sc]:
                attempts += 1
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
            attempts += 1
        if seed is None:
            continue

        blob = {seed}
        frontier = [seed]
        while len(blob) < budget and frontier:
            _r.shuffle(frontier)
            nf = []
            for fr, fc in frontier:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = fr + dr, fc + dc
                    if nr < 0 or nr >= iH or nc < 0 or nc >= iW:
                        continue
                    if (nr, nc) in blob or blob_occupied[nr, nc]:
                        continue
                    conflict = False
                    for dr2 in (-1, 0, 1):
                        for dc2 in (-1, 0, 1):
                            if dr2 == 0 and dc2 == 0:
                                continue
                            ar, ac = nr + dr2, nc + dc2
                            if 0 <= ar < iH and 0 <= ac < iW:
                                if blob_occupied[ar, ac] and (ar, ac) not in blob:
                                    conflict = True
                                    break
                        if conflict:
                            break
                    if conflict:
                        continue
                    blob.add((nr, nc))
                    nf.append((nr, nc))
                    if len(blob) >= budget:
                        break
                if len(blob) >= budget:
                    break
            frontier = nf

        for br, bc in blob:
            interior[br, bc] = ic
            blob_occupied[br, bc] = True

    non_rc_count = int(np.sum(interior != rect_col))
    fill_ratio = non_rc_count / (iH * iW)
    if fill_ratio < 0.30 or fill_ratio > 0.90:
        return None
    rect[1:-1, 1:-1] = interior

    reserved = np.zeros((RH, RW), dtype=bool)
    for side in free_sides:
        color = _r.choice(interior_cols)
        if not _paint_band(rect, reserved, side, color, pure_sides, _r):
            return None

    for side in pure_sides:
        if not np.all(_edge_values(rect, side) == int(rect_col)):
            return None
    for side in free_sides:
        if np.all(_edge_values(rect, side) == int(rect_col)):
            return None

    row_segs = _rand_splits(RH, _r)
    col_segs = _rand_splits(RW, _r)
    if row_segs is None or col_segs is None:
        return None

    row_starts: list[int] = []
    s = 0
    for rs in row_segs:
        row_starts.append(s)
        s += rs
    col_starts: list[int] = []
    s = 0
    for cs in col_segs:
        col_starts.append(s)
        s += cs

    nr = len(row_segs)
    nc = len(col_segs)
    cell_id = [[i * nc + j for j in range(nc)] for i in range(nr)]
    n_cells = nr * nc
    if n_cells < 4:
        return None

    target_pieces = _r.randint(4, min(9, n_cells))

    def _get_piece_bbox(pid: int) -> tuple[int, int, int, int] | None:
        cells = [(i, j) for i in range(nr) for j in range(nc) if cell_id[i][j] == pid]
        if not cells:
            return None
        rs = [ij[0] for ij in cells]
        cs = [ij[1] for ij in cells]
        return min(rs), min(cs), max(rs), max(cs)

    def _pixel_size(r0i: int, c0i: int, r1i: int, c1i: int) -> tuple[int, int]:
        h = sum(row_segs[i] for i in range(r0i, r1i + 1))
        w = sum(col_segs[j] for j in range(c0i, c1i + 1))
        return h, w

    merges = []
    for i in range(nr):
        for j in range(nc):
            if j + 1 < nc:
                merges.append(("h", i, j))
            if i + 1 < nr:
                merges.append(("v", i, j))
    _r.shuffle(merges)

    def _count_pieces() -> int:
        return len(set(cell_id[i][j] for i in range(nr) for j in range(nc)))

    for direction, mi, mj in merges:
        if _count_pieces() <= target_pieces:
            break
        if direction == "h":
            pid1 = cell_id[mi][mj]
            pid2 = cell_id[mi][mj + 1]
        else:
            pid1 = cell_id[mi][mj]
            pid2 = cell_id[mi + 1][mj]
        if pid1 == pid2:
            continue
        bb1 = _get_piece_bbox(pid1)
        bb2 = _get_piece_bbox(pid2)
        if bb1 is None or bb2 is None:
            continue
        mr0 = min(bb1[0], bb2[0])
        mc0 = min(bb1[1], bb2[1])
        mr1 = max(bb1[2], bb2[2])
        mc1 = max(bb1[3], bb2[3])
        ph, pw = _pixel_size(mr0, mc0, mr1, mc1)
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

    pieces_meta: list[tuple[int, int, int, int]] = []
    seen = set()
    for i in range(nr):
        for j in range(nc):
            pid = cell_id[i][j]
            if pid in seen:
                continue
            seen.add(pid)
            bb = _get_piece_bbox(pid)
            assert bb is not None
            r0i, c0i, r1i, c1i = bb
            pr = row_starts[r0i]
            pc = col_starts[c0i]
            ph = sum(row_segs[ii] for ii in range(r0i, r1i + 1))
            pw = sum(col_segs[jj] for jj in range(c0i, c1i + 1))
            pieces_meta.append((pr, pc, ph, pw))

    if not (4 <= len(pieces_meta) <= 9):
        return None

    top_left_piece_idx = None
    piece_arrays: list[np.ndarray] = []
    for i, (pr, pc, ph, pw) in enumerate(pieces_meta):
        piece = rect[pr : pr + ph, pc : pc + pw].copy()
        piece_arrays.append(piece)
        if pr == 0 and pc == 0:
            top_left_piece_idx = i
        if not (2 <= ph <= 7 and 2 <= pw <= 7):
            return None
        fill = float(np.sum(piece != rect_col)) / float(ph * pw)
        if fill < 0.10 or fill > 0.65:
            return None
        has_non_rc_edge = any(np.any(_edge_values(piece, side) != int(rect_col)) for side in ALL_SIDES)
        if not has_non_rc_edge:
            return None
        if pr == 0 and "top" in pure_sides and not np.all(piece[0, :] == int(rect_col)):
            return None
        if pr + ph == RH and "bottom" in pure_sides and not np.all(piece[-1, :] == int(rect_col)):
            return None
        if pc == 0 and "left" in pure_sides and not np.all(piece[:, 0] == int(rect_col)):
            return None
        if pc + pw == RW and "right" in pure_sides and not np.all(piece[:, -1] == int(rect_col)):
            return None

    if top_left_piece_idx is None:
        return None

    all_piece_pixels = [int(x) for a in piece_arrays for x in a.flatten()]
    if int(Counter(all_piece_pixels).most_common(1)[0][0]) != int(rect_col):
        return None

    solved_rect, solved_dim, records, dedup = _select_solution(piece_arrays, rect_col, GH - tr, GW - tc)
    if solved_rect is None or solved_dim is None or len(dedup) != 1:
        return None
    only = next(iter(dedup.values()))
    if (only["H"], only["W"]) != (RH, RW):
        return None
    if not np.array_equal(only["board"], rect):
        return None
    if only["pure_sides"] != pure_sides:
        return None

    canvas = np.full((GH, GW), bg, dtype=int)
    paste(canvas, piece_arrays[top_left_piece_idx], tr, tc)

    order = [i for i in range(len(piece_arrays)) if i != top_left_piece_idx]
    _r.shuffle(order)
    for idx in order:
        pa = piece_arrays[idx]
        sub = canvas[tr:, tc:]
        try:
            rr, cc = random_free_location(sub, pa, transparent=bg, padding=1)
        except ValueError:
            return None
        paste(canvas, pa, tr + rr, tc + cc)

    if int(Counter(map(int, canvas.flatten())).most_common(1)[0][0]) != int(bg):
        return None
    if _has_marker_like_triplet(canvas, bg):
        return None

    expected_out = np.full((GH, GW), bg, dtype=int)
    expected_out[tr : tr + RH, tc : tc + RW] = rect

    try:
        actual_out = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual_out, expected_out):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(1500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 1500 attempts")
