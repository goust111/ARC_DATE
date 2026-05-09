# Combo variant of 446ef5d2 (index 23)
# combo_spec: D2/3+D13/2+D16/2
# Name: 干扰碎片配合多候选标记和多候选锚定

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import random

import numpy as np

from arc_utils import paste, random_free_location


"""
Design analysis for the fused combo
==================================

a) Concrete behavior and trace
   Original task:
   - There is one 3-cell L marker, one implied corner, one anchored piece, and
     every non-marker rectangle must be used.
   - Solver flow is therefore: read the unique marker -> infer corner -> use all
     rectangles -> tile one rectangle.

   Fused variant:
   - There are 2-3 marker candidates.
   - There are 1-2 extra distractor rectangles that use the same palette and are
     locally plausible but cannot belong to any full tiling.
   - For the true marker orientation, multiple pieces have the required rect_col
     corner signature, so anchor selection is ambiguous.
   - Correct reasoning becomes:
       1. try a marker candidate;
       2. for that marker's corner orientation, try plausible anchor pieces;
       3. use subset assembly because some rectangles are distractors;
       4. keep only the branch that completes a legal rectangle.

   Small trace:
   - If the solver picks a wrong marker, candidate dimensions or boundary checks
     fail, or no subset can tile a rectangle at that location.
   - If it picks the true marker but a wrong anchor-looking piece, the forced
     corner placement blocks a full tiling.
   - If it picks the true marker and true anchor but includes a distractor, the
     remaining subset cannot match all shared edges.
   - Only one marker+anchor+subset branch succeeds.

b) Complexity calibration
   - Marker count is kept at 2 or 3, usually 2.
   - Anchor-looking pieces are kept at 2-3, usually 2.
   - Distractors are kept at 1-2, usually 1.
   - True piece count is kept modest (4-8) to keep the search feasible while
     still making all three disambiguation layers visible.
   - Decoy markers use other corner orientations, which makes them visually
     plausible without exploding the number of successful alternative branches.

c) Implicit assumptions broken from the original solver
   - "There is exactly one marker color with count 3" is false now.
   - "Total area of all rectangles equals the target rectangle area" is false
     because distractors remain unused.
   - "A single anchor piece can be identified deterministically" is false
     because several pieces may have the needed rect_col L corner.
   - "Trying one marker is enough" is false because only one marker is valid.
   The new solver therefore:
   - enumerates all marker candidates;
   - uses subset-sum-reachable rectangle areas instead of one total area;
   - enumerates all anchor-looking pieces for each marker corner;
   - validates by full assembly.

d) Placement legality
   - All pieces remain axis-aligned rectangles and are never rotated.
   - Objects are bg-separated by at least one cell, except a marker may touch its
     associated piece exactly like the original task.
   - The true anchor piece is still pinned beside the true marker at the output
     location; other pieces and distractors are scattered.
   - Decoy markers are attached to different scattered piece corners via patches,
     so they remain visually local and non-overlapping.

e) Visual observability
   - Humans can see three visible changes across examples:
       1. extra rectangles exist, yet not all appear in the output;
       2. multiple L markers exist, but only one determines the final rectangle;
       3. several pieces share the same rect_col corner style, but only one can
          actually serve as the assembled corner piece.
   - No invisible ordering by numeric color ids is used.

f) Visual cleanliness
   - The only overlaid structure is the 3-cell L marker.
   - Candidate markers use distinct colors and stay attached to piece corners.
   - Object counts are capped and padding is enforced to avoid clutter.

g) Color and rule inferability
   - rect_col keeps the original meaning: border color and wildcard on shared
     edges.
   - Marker colors only signal "candidate marker", not any ordering rule.
   - Distractors reuse the same palette as real pieces, so the exclusion rule is
     inferred structurally from failed assembly rather than from color identity.

New solver assumptions introduced here
   - At least one marker candidate exists. The generator guarantees 2-3.
   - At least one marker has at least two anchor-looking pieces for its corner
     orientation. The generator guarantees this for the true marker.
   - There is exactly one successful output signature, and successful branches
     use exactly one marker color and one anchor piece. The generator validates
     this before accepting an instance.
"""


CORNERS = ("TL", "TR", "BL", "BR")


def _corner_from_missing(missing: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    if missing == (r0, c0):
        return "BR"
    if missing == (r0, c1):
        return "BL"
    if missing == (r1, c0):
        return "TR"
    return "TL"


def _marker_cells(mr: int, mc: int, corner: str) -> list[tuple[int, int]]:
    if corner == "TL":
        return [(mr - 1, mc - 1), (mr - 1, mc), (mr, mc - 1)]
    if corner == "TR":
        return [(mr - 1, mc), (mr - 1, mc + 1), (mr, mc + 1)]
    if corner == "BL":
        return [(mr, mc - 1), (mr + 1, mc - 1), (mr + 1, mc)]
    return [(mr, mc + 1), (mr + 1, mc), (mr + 1, mc + 1)]


def _top_left_from_corner(corner: str, mr: int, mc: int, H: int, W: int) -> tuple[int, int]:
    if corner == "TL":
        return mr, mc
    if corner == "TR":
        return mr, mc - W + 1
    if corner == "BL":
        return mr - H + 1, mc
    return mr - H + 1, mc - W + 1


def _piece_top_left_in_rect(corner: str, H: int, W: int, h: int, w: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, W - w
    if corner == "BL":
        return H - h, 0
    return H - h, W - w


def _compat(v1: np.ndarray, v2: np.ndarray, rect_col: int) -> bool:
    a = np.array(v1, dtype=int)
    b = np.array(v2, dtype=int)
    m = (a != int(rect_col)) & (b != int(rect_col))
    return bool((a[m] == b[m]).all())


def _is_corner_candidate(piece: np.ndarray, corner: str, rect_col: int) -> bool:
    if corner == "TL":
        return bool(np.all(piece[0, :] == int(rect_col)) and np.all(piece[:, 0] == int(rect_col)))
    if corner == "TR":
        return bool(np.all(piece[0, :] == int(rect_col)) and np.all(piece[:, -1] == int(rect_col)))
    if corner == "BL":
        return bool(np.all(piece[-1, :] == int(rect_col)) and np.all(piece[:, 0] == int(rect_col)))
    return bool(np.all(piece[-1, :] == int(rect_col)) and np.all(piece[:, -1] == int(rect_col)))


def _find_candidate_markers(g: np.ndarray, bg: int) -> list[dict]:
    cnt = Counter(map(int, g.flatten()))
    out: list[dict] = []
    for col in sorted(c for c, n in cnt.items() if int(c) != int(bg) and int(n) == 3):
        pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(col))]
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if (r1 - r0) != 1 or (c1 - c0) != 1:
            continue
        miss = list({(r0, c0), (r0, c1), (r1, c0), (r1, c1)} - set(pts))
        if len(miss) != 1:
            continue
        missing = tuple(map(int, miss[0]))
        out.append(
            {
                "color": int(col),
                "missing": missing,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "corner": _corner_from_missing(missing, (int(r0), int(c0), int(r1), int(c1))),
            }
        )
    return out


def _extract_rectangles(
    g: np.ndarray,
    bg: int,
    marker_colors: set[int],
) -> list[tuple[int, int, int, int, np.ndarray]]:
    GH, GW = g.shape
    occ = g != int(bg)
    for col in marker_colors:
        occ &= g != int(col)
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


def _candidate_area_counts(areas: list[int]) -> dict[int, int]:
    area_counts: dict[int, int] = {0: 1}
    for area in areas:
        nxt = dict(area_counts)
        for s, cnt_mask in area_counts.items():
            ns = s + int(area)
            nxt[ns] = nxt.get(ns, 0) | (cnt_mask << 1)
        area_counts = nxt
    return area_counts


def _dims_for_marker(
    area_counts: dict[int, int],
    corner: str,
    mr: int,
    mc: int,
    GH: int,
    GW: int,
    *,
    min_real_pieces: int = 4,
) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for area, cnt_mask in area_counts.items():
        if area <= 0 or ((cnt_mask >> min_real_pieces) == 0):
            continue
        root = int(area**0.5)
        for hh in range(1, root + 1):
            if area % hh != 0:
                continue
            for H, W in ((hh, area // hh), (area // hh, hh)):
                H = int(H)
                W = int(W)
                if (H, W) in seen:
                    continue
                tr, tc = _top_left_from_corner(corner, mr, mc, H, W)
                if 0 <= tr and 0 <= tc and (tr + H) <= GH and (tc + W) <= GW:
                    seen.add((H, W))
                    dims.append((H, W))
    dims.sort(key=lambda t: (-t[0] * t[1], -min(t[0], t[1]), -max(t[0], t[1])))
    return dims


def _attempt_marker(
    GH: int,
    GW: int,
    rects: list[tuple[int, int, int, int, np.ndarray]],
    pieces: list[np.ndarray],
    rect_col: int,
    areas: list[int],
    area_counts: dict[int, int],
    candidate: dict,
) -> list[dict]:
    mr, mc = candidate["missing"]
    corner = candidate["corner"]
    dims = _dims_for_marker(area_counts, corner, mr, mc, GH, GW)
    if not dims:
        return []

    ph = [int(p.shape[0]) for p in pieces]
    pw = [int(p.shape[1]) for p in pieces]
    n = len(pieces)
    full_mask = (1 << n) - 1
    order = sorted(range(n), key=lambda i: (-areas[i], -ph[i], -pw[i], i))

    linked_idx = None
    for i, (rr, cc, h, w, _a) in enumerate(rects):
        corners = {
            (rr, cc): "TL",
            (rr, cc + w - 1): "TR",
            (rr + h - 1, cc): "BL",
            (rr + h - 1, cc + w - 1): "BR",
        }
        if corners.get((mr, mc)) == corner:
            linked_idx = i
            break

    anchor_candidates = [i for i, piece in enumerate(pieces) if _is_corner_candidate(piece, corner, rect_col)]
    if not anchor_candidates:
        return []
    if linked_idx in anchor_candidates:
        anchor_candidates = [linked_idx] + [i for i in anchor_candidates if i != linked_idx]
    anchor_candidates = sorted(anchor_candidates, key=lambda i: (0 if i == linked_idx else 1, -areas[i], i))

    @lru_cache(maxsize=None)
    def _reachable_sums(mask: int) -> int:
        if mask == 0:
            return 1
        lb = mask & -mask
        i = lb.bit_length() - 1
        rest = mask ^ lb
        base = _reachable_sums(rest)
        return base | (base << areas[i])

    solutions: list[dict] = []

    for H, W in dims:
        for anchor_idx in anchor_candidates:
            occ2 = np.full((H, W), -1, dtype=int)

            def boundary_ok(a: np.ndarray, r: int, c: int) -> bool:
                h, w = a.shape
                if r == 0 and not np.all(a[0, :] == int(rect_col)):
                    return False
                if (r + h) == H and not np.all(a[-1, :] == int(rect_col)):
                    return False
                if c == 0 and not np.all(a[:, 0] == int(rect_col)):
                    return False
                if (c + w) == W and not np.all(a[:, -1] == int(rect_col)):
                    return False
                return True

            def neighbor_ok(a: np.ndarray, r: int, c: int) -> bool:
                h, w = a.shape
                if c > 0:
                    left = occ2[r : r + h, c - 1]
                    m = left != -1
                    if m.any() and not _compat(left[m], a[:, 0][m], rect_col):
                        return False
                if (c + w) < W:
                    right = occ2[r : r + h, c + w]
                    m = right != -1
                    if m.any() and not _compat(right[m], a[:, -1][m], rect_col):
                        return False
                if r > 0:
                    up = occ2[r - 1, c : c + w]
                    m = up != -1
                    if m.any() and not _compat(up[m], a[0, :][m], rect_col):
                        return False
                if (r + h) < H:
                    down = occ2[r + h, c : c + w]
                    m = down != -1
                    if m.any() and not _compat(down[m], a[-1, :][m], rect_col):
                        return False
                return True

            def can_place(i: int, r: int, c: int) -> bool:
                a = pieces[i]
                h, w = a.shape
                if (r + h) > H or (c + w) > W:
                    return False
                view = occ2[r : r + h, c : c + w]
                if view.shape != a.shape:
                    return False
                if (view != -1).any():
                    return False
                if not boundary_ok(a, r, c):
                    return False
                if not neighbor_ok(a, r, c):
                    return False
                return True

            def place(i: int, r: int, c: int) -> None:
                a = pieces[i]
                h, w = a.shape
                occ2[r : r + h, c : c + w] = a

            def unplace(i: int, r: int, c: int) -> None:
                a = pieces[i]
                h, w = a.shape
                occ2[r : r + h, c : c + w] = -1

            ar0, ac0 = _piece_top_left_in_rect(corner, H, W, ph[anchor_idx], pw[anchor_idx])
            if not can_place(anchor_idx, ar0, ac0):
                continue
            place(anchor_idx, ar0, ac0)

            def next_empty() -> tuple[int, int] | None:
                flat = np.flatnonzero(occ2.ravel() == -1)
                if flat.size == 0:
                    return None
                idx = int(flat[0])
                return idx // W, idx % W

            solved = False

            def bt(unused_mask: int, filled_area: int, used_count: int) -> None:
                nonlocal solved
                remain = H * W - filled_area
                if remain == 0:
                    solved = used_count >= 4
                    return
                if ((_reachable_sums(unused_mask) >> remain) & 1) == 0:
                    return
                nxt = next_empty()
                assert nxt is not None
                r, c = nxt
                for i in order:
                    if solved or ((unused_mask >> i) & 1) == 0:
                        continue
                    if can_place(i, r, c):
                        place(i, r, c)
                        bt(unused_mask ^ (1 << i), filled_area + areas[i], used_count + 1)
                        if solved:
                            return
                        unplace(i, r, c)

            bt(full_mask ^ (1 << anchor_idx), areas[anchor_idx], 1)
            if not solved:
                continue

            tr, tc = _top_left_from_corner(corner, mr, mc, H, W)
            if tr < 0 or tc < 0 or tr + H > GH or tc + W > GW:
                continue
            solutions.append(
                {
                    "marker_color": int(candidate["color"]),
                    "marker_corner": corner,
                    "missing": (int(mr), int(mc)),
                    "anchor_idx": int(anchor_idx),
                    "dims": (int(H), int(W)),
                    "top_left": (int(tr), int(tc)),
                    "assembly": occ2.copy(),
                }
            )
    return solutions


def _solve_all(grid: np.ndarray) -> list[dict]:
    g = np.array(grid, dtype=int)
    GH, GW = g.shape
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])

    markers = _find_candidate_markers(g, bg)
    assert markers
    marker_colors = {int(m["color"]) for m in markers}

    rects = _extract_rectangles(g, bg, marker_colors)
    assert rects

    pieces = [a for _r, _c, _h, _w, a in rects]
    allpix: list[int] = []
    for piece in pieces:
        allpix.extend(map(int, piece.flatten()))
    rect_col = int(Counter(allpix).most_common(1)[0][0])
    areas = [int(p.shape[0] * p.shape[1]) for p in pieces]
    area_counts = _candidate_area_counts(areas)

    sols: list[dict] = []
    for marker in markers:
        marker_solutions = _attempt_marker(GH, GW, rects, pieces, rect_col, areas, area_counts, marker)
        for sol in marker_solutions:
            out = np.full((GH, GW), int(bg), dtype=int)
            tr, tc = sol["top_left"]
            H, W = sol["dims"]
            if tr < 0 or tc < 0 or tr + H > GH or tc + W > GW:
                continue
            out[tr : tr + H, tc : tc + W] = sol["assembly"]
            sol["output"] = out
            sols.append(sol)
    return sols


def main(grid: np.ndarray) -> np.ndarray:
    sols = _solve_all(grid)
    assert sols
    sigs = {tuple(map(int, sol["output"].flatten())) for sol in sols}
    assert len(sigs) == 1
    return sols[0]["output"]


def _rand_splits(total: int, rng: random.Random, min_s: int = 2, max_s: int = 7) -> list[int] | None:
    for _ in range(60):
        segs: list[int] = []
        rem = int(total)
        while rem > 0:
            choices = [s for s in range(min_s, min(max_s, rem) + 1) if rem == s or rem - s >= min_s]
            if not choices:
                break
            s = int(rng.choice(choices))
            segs.append(s)
            rem -= s
        if rem == 0:
            return segs
    return None


def _build_rectangle(RH: int, RW: int, rect_col: int, interior_cols: list[int], rng: random.Random) -> np.ndarray | None:
    rect = np.full((RH, RW), rect_col, dtype=int)
    iH, iW = RH - 2, RW - 2
    if iH < 1 or iW < 1:
        return None

    interior = np.full((iH, iW), rect_col, dtype=int)
    target_fill = rng.uniform(0.45, 0.82)
    target_pixels = max(1, int(target_fill * iH * iW))
    blob_occupied = np.zeros((iH, iW), dtype=bool)

    for ci, ic in enumerate(interior_cols):
        if ci == len(interior_cols) - 1:
            budget = target_pixels - int(blob_occupied.sum())
        else:
            budget = max(1, target_pixels // len(interior_cols))
        if budget <= 0:
            budget = 1

        seed = None
        for _ in range(200):
            sr, sc = rng.randint(0, iH - 1), rng.randint(0, iW - 1)
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
            rng.shuffle(frontier)
            nf = []
            for fr, fc in frontier:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = fr + dr, fc + dc
                    if (nr, nc) in blob or nr < 0 or nr >= iH or nc < 0 or nc >= iW:
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
                    nf.append((nr, nc))
                    if len(blob) >= budget:
                        break
                if len(blob) >= budget:
                    break
            frontier = nf

        for br, bc in blob:
            interior[br, bc] = ic
            blob_occupied[br, bc] = True

    fill_ratio = float(np.mean(interior != rect_col))
    if fill_ratio < 0.30 or fill_ratio > 0.90:
        return None
    rect[1:-1, 1:-1] = interior
    return rect


def _partition_rectangle(
    rect: np.ndarray,
    rng: random.Random,
    *,
    min_pieces: int = 5,
    max_pieces: int = 9,
) -> list[tuple[int, int, int, int]] | None:
    RH, RW = rect.shape
    row_segs = _rand_splits(RH, rng)
    col_segs = _rand_splits(RW, rng)
    if row_segs is None or col_segs is None:
        return None

    row_starts: list[int] = []
    col_starts: list[int] = []
    s = 0
    for rs in row_segs:
        row_starts.append(s)
        s += rs
    s = 0
    for cs in col_segs:
        col_starts.append(s)
        s += cs

    nr, nc_grid = len(row_segs), len(col_segs)
    n_cells = nr * nc_grid
    if n_cells < min_pieces:
        return None
    cell_id = [[i * nc_grid + j for j in range(nc_grid)] for i in range(nr)]

    def piece_bbox(pid: int) -> tuple[int, int, int, int] | None:
        cells = [(i, j) for i in range(nr) for j in range(nc_grid) if cell_id[i][j] == pid]
        if not cells:
            return None
        ris = [r for r, _ in cells]
        cis = [c for _, c in cells]
        return min(ris), min(cis), max(ris), max(cis)

    def pixel_size(r0i: int, c0i: int, r1i: int, c1i: int) -> tuple[int, int]:
        h = sum(row_segs[i] for i in range(r0i, r1i + 1))
        w = sum(col_segs[j] for j in range(c0i, c1i + 1))
        return h, w

    target_pieces = rng.randint(min_pieces, min(max_pieces, n_cells))
    merges: list[tuple[str, int, int]] = []
    for i in range(nr):
        for j in range(nc_grid):
            if j + 1 < nc_grid:
                merges.append(("h", i, j))
            if i + 1 < nr:
                merges.append(("v", i, j))
    rng.shuffle(merges)

    def count_pieces() -> int:
        return len(set(cell_id[i][j] for i in range(nr) for j in range(nc_grid)))

    for direction, mi, mj in merges:
        if count_pieces() <= target_pieces:
            break
        if direction == "h":
            pid1 = cell_id[mi][mj]
            pid2 = cell_id[mi][mj + 1]
        else:
            pid1 = cell_id[mi][mj]
            pid2 = cell_id[mi + 1][mj]
        if pid1 == pid2:
            continue
        bb1 = piece_bbox(pid1)
        bb2 = piece_bbox(pid2)
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
                if cell_id[ii][jj] != pid1 and cell_id[ii][jj] != pid2:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        for ii in range(nr):
            for jj in range(nc_grid):
                if cell_id[ii][jj] == pid2:
                    cell_id[ii][jj] = pid1

    pieces: list[tuple[int, int, int, int]] = []
    seen = set()
    for i in range(nr):
        for j in range(nc_grid):
            pid = cell_id[i][j]
            if pid in seen:
                continue
            seen.add(pid)
            bb = piece_bbox(pid)
            if bb is None:
                return None
            r0i, c0i, r1i, c1i = bb
            pr = row_starts[r0i]
            pc = col_starts[c0i]
            ph = sum(row_segs[ii] for ii in range(r0i, r1i + 1))
            pw = sum(col_segs[jj] for jj in range(c0i, c1i + 1))
            pieces.append((pr, pc, ph, pw))
    if len(pieces) < min_pieces or len(pieces) > max_pieces:
        return None
    return pieces


def _piece_supports_corner(piece_info: tuple[int, int, int, int], corner: str, RH: int, RW: int) -> bool:
    pr, pc, ph, pw = piece_info
    if corner == "TL":
        return pr == 0 and pc == 0
    if corner == "TR":
        return pr == 0 and (pc + pw) == RW
    if corner == "BL":
        return (pr + ph) == RH and pc == 0
    return (pr + ph) == RH and (pc + pw) == RW


def _piece_stats(piece: np.ndarray, rect_col: int) -> tuple[float, bool]:
    ph, pw = piece.shape
    fill = float(np.sum(piece != rect_col)) / float(ph * pw)
    has_non_rc_edge = any(np.any(edge != rect_col) for edge in (piece[0, :], piece[-1, :], piece[:, 0], piece[:, -1]))
    return fill, bool(has_non_rc_edge)


def _force_candidate_corner(rect: np.ndarray, piece_box: tuple[int, int, int, int], corner: str, rect_col: int) -> None:
    pr, pc, ph, pw = piece_box
    if corner == "TL":
        rect[pr, pc : pc + pw] = rect_col
        rect[pr : pr + ph, pc] = rect_col
    elif corner == "TR":
        rect[pr, pc : pc + pw] = rect_col
        rect[pr : pr + ph, pc + pw - 1] = rect_col
    elif corner == "BL":
        rect[pr + ph - 1, pc : pc + pw] = rect_col
        rect[pr : pr + ph, pc] = rect_col
    else:
        rect[pr + ph - 1, pc : pc + pw] = rect_col
        rect[pr : pr + ph, pc + pw - 1] = rect_col


def _repair_piece_signals(
    rect: np.ndarray,
    pieces: list[tuple[int, int, int, int]],
    rect_col: int,
    interior_cols: list[int],
    rng: random.Random,
) -> None:
    RH, RW = rect.shape
    for pr, pc, ph, pw in pieces:
        piece = rect[pr : pr + ph, pc : pc + pw]
        fill, has_non_rc_edge = _piece_stats(piece, rect_col)

        if not has_non_rc_edge:
            edge_cells: list[tuple[int, int]] = []
            if pr > 0:
                edge_cells.extend((pr, pc + j) for j in range(pw))
            if pr + ph < RH:
                edge_cells.extend((pr + ph - 1, pc + j) for j in range(pw))
            if pc > 0:
                edge_cells.extend((pr + i, pc) for i in range(ph))
            if pc + pw < RW:
                edge_cells.extend((pr + i, pc + pw - 1) for i in range(ph))
            edge_cells = [(rr, cc) for rr, cc in edge_cells if 0 < rr < RH - 1 and 0 < cc < RW - 1]
            if edge_cells:
                rng.shuffle(edge_cells)
                col = int(rng.choice(interior_cols))
                for rr, cc in edge_cells[: max(1, min(3, len(edge_cells) // 2 or 1))]:
                    rect[rr, cc] = col
                piece = rect[pr : pr + ph, pc : pc + pw]
                fill, has_non_rc_edge = _piece_stats(piece, rect_col)

        min_non_rc = max(1, int(np.ceil(0.03 * ph * pw)))
        cur_non_rc = int(np.sum(piece != rect_col))
        if cur_non_rc >= min_non_rc:
            continue

        candidates: list[tuple[int, int]] = []
        for rr in range(pr, pr + ph):
            for cc in range(pc, pc + pw):
                if rr == 0 or rr == RH - 1 or cc == 0 or cc == RW - 1:
                    continue
                candidates.append((rr, cc))
        if not candidates:
            continue
        rng.shuffle(candidates)
        need = min_non_rc - cur_non_rc
        for rr, cc in candidates[:need]:
            rect[rr, cc] = int(rng.choice(interior_cols))


def _slot_side_checks(
    piece: np.ndarray,
    slot: tuple[int, int, int, int],
    rect: np.ndarray,
    rect_col: int,
) -> list[bool]:
    RH, RW = rect.shape
    pr, pc, ph, pw = slot
    if piece.shape != (ph, pw):
        return [False, False, False, False]
    checks: list[bool] = []
    if pr == 0:
        checks.append(bool(np.all(piece[0, :] == rect_col)))
    else:
        checks.append(_compat(piece[0, :], rect[pr - 1, pc : pc + pw], rect_col))
    if pr + ph == RH:
        checks.append(bool(np.all(piece[-1, :] == rect_col)))
    else:
        checks.append(_compat(piece[-1, :], rect[pr + ph, pc : pc + pw], rect_col))
    if pc == 0:
        checks.append(bool(np.all(piece[:, 0] == rect_col)))
    else:
        checks.append(_compat(piece[:, 0], rect[pr : pr + ph, pc - 1], rect_col))
    if pc + pw == RW:
        checks.append(bool(np.all(piece[:, -1] == rect_col)))
    else:
        checks.append(_compat(piece[:, -1], rect[pr : pr + ph, pc + pw], rect_col))
    return checks


def _make_distractor(
    true_piece: np.ndarray,
    slot: tuple[int, int, int, int],
    rect: np.ndarray,
    rect_col: int,
    interior_cols: list[int],
    slots: list[tuple[int, int, int, int]],
    existing: list[np.ndarray],
    rng: random.Random,
) -> tuple[np.ndarray, dict[str, int]] | None:
    pr, pc, ph, pw = slot
    RH, RW = rect.shape
    if len(interior_cols) < 2:
        return None

    side_specs: list[tuple[str, np.ndarray, int]] = []
    if pr > 0:
        side_specs.append(("top", rect[pr - 1, pc : pc + pw], 0))
    if pr + ph < RH:
        side_specs.append(("bottom", rect[pr + ph, pc : pc + pw], ph - 1))
    if pc > 0:
        side_specs.append(("left", rect[pr : pr + ph, pc - 1], 0))
    if pc + pw < RW:
        side_specs.append(("right", rect[pr : pr + ph, pc + pw], pw - 1))
    rng.shuffle(side_specs)

    for side, neigh, fixed in side_specs:
        indices = [i for i, val in enumerate(neigh) if int(val) != int(rect_col)]
        rng.shuffle(indices)
        for pos in indices:
            neigh_val = int(neigh[pos])
            options = [c for c in interior_cols if int(c) != neigh_val]
            if not options:
                continue
            cand = true_piece.copy()
            new_col = int(rng.choice(options))
            if side in ("top", "bottom"):
                cand[fixed, pos] = new_col
            else:
                cand[pos, fixed] = new_col

            fill = float(np.mean(cand != rect_col))
            if fill < 0.03 or fill > 0.75:
                continue
            if not any(np.any(edge != rect_col) for edge in (cand[0, :], cand[-1, :], cand[:, 0], cand[:, -1])):
                continue
            if any(np.array_equal(cand, other) for other in existing):
                continue

            side_checks = _slot_side_checks(cand, slot, rect, rect_col)
            if sum(side_checks) < 2 or all(side_checks):
                continue
            if any(all(_slot_side_checks(cand, s, rect, rect_col)) for s in slots):
                continue

            return cand, {"partial_sides": int(sum(side_checks))}
    return None


def _make_piece_marker_patch(piece: np.ndarray, corner: str, marker_col: int, bg: int) -> np.ndarray:
    ph, pw = piece.shape
    marker_coords = {
        "TL": [(-1, -1), (-1, 0), (0, -1)],
        "TR": [(-1, pw - 1), (-1, pw), (0, pw)],
        "BL": [(ph - 1, -1), (ph, -1), (ph, 0)],
        "BR": [(ph - 1, pw), (ph, pw - 1), (ph, pw)],
    }[corner]
    coords = [(0, 0), (ph - 1, pw - 1)] + marker_coords
    min_r = min(r for r, _ in coords)
    max_r = max(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    max_c = max(c for _, c in coords)
    patch = np.full((max_r - min_r + 1, max_c - min_c + 1), int(bg), dtype=int)
    pr0 = -min_r
    pc0 = -min_c
    patch[pr0 : pr0 + ph, pc0 : pc0 + pw] = piece
    for rr, cc in marker_coords:
        patch[rr - min_r, cc - min_c] = int(marker_col)
    return patch


def _gen_attempt(return_meta: bool = False, debug: bool = False):
    rng = random.Random()

    def bail(reason: str):
        if debug:
            return ("FAIL", reason)
        return None

    GH = rng.randint(15, 26)
    GW = rng.randint(15, 26)

    n_markers = 2
    n_distractors = 1

    palette = list(range(10))
    rng.shuffle(palette)
    bg = palette[0]
    rect_col = palette[1]
    marker_cols = palette[2 : 2 + n_markers]
    n_interior = rng.randint(2, min(3, 10 - (2 + n_markers)))
    interior_cols = palette[2 + n_markers : 2 + n_markers + n_interior]

    corner = rng.choice(list(CORNERS))

    min_side = 6
    max_side = min(14, min(GH, GW) - 4)
    if max_side < min_side:
        return bail("grid_too_small")
    RH = rng.randint(min_side, max_side)
    RW = rng.randint(min_side, max_side)

    if corner == "TL":
        mr_min, mr_max = 2, GH - RH
        mc_min, mc_max = 2, GW - RW
    elif corner == "TR":
        mr_min, mr_max = 2, GH - RH
        mc_min, mc_max = RW - 1, GW - 3
    elif corner == "BL":
        mr_min, mr_max = RH - 1, GH - 3
        mc_min, mc_max = 2, GW - RW
    else:
        mr_min, mr_max = RH - 1, GH - 3
        mc_min, mc_max = RW - 1, GW - 3
    if mr_min > mr_max or mc_min > mc_max:
        return bail("marker_range")
    mr = rng.randint(mr_min, mr_max)
    mc = rng.randint(mc_min, mc_max)

    rect = _build_rectangle(RH, RW, rect_col, interior_cols, rng)
    if rect is None:
        return bail("build_rectangle")

    pieces = _partition_rectangle(rect, rng, min_pieces=5, max_pieces=9)
    if pieces is None:
        return bail("partition")

    _repair_piece_signals(rect, pieces, rect_col, interior_cols, rng)

    if len(pieces) < max(5, n_markers + 2):
        return bail("piece_count_small")

    for pr, pc, ph, pw in pieces:
        piece = rect[pr : pr + ph, pc : pc + pw]
        fill, has_non_rc_edge = _piece_stats(piece, rect_col)
        if ph < 2 or pw < 2 or ph > 7 or pw > 7:
            return bail("piece_dim")
        if fill < 0.03:
            return bail("piece_fill_low")
        if fill > 0.75:
            return bail("piece_fill_high")
        if not has_non_rc_edge:
            return bail("piece_no_signal_edge")
        if pr == 0 and not np.all(piece[0, :] == rect_col):
            return bail("top_border")
        if pr + ph == RH and not np.all(piece[-1, :] == rect_col):
            return bail("bottom_border")
        if pc == 0 and not np.all(piece[:, 0] == rect_col):
            return bail("left_border")
        if pc + pw == RW and not np.all(piece[:, -1] == rect_col):
            return bail("right_border")

    anchor_cell = {
        "TL": (0, 0),
        "TR": (0, RW - 1),
        "BL": (RH - 1, 0),
        "BR": (RH - 1, RW - 1),
    }[corner]
    true_anchor_idx = None
    for i, (pr, pc, ph, pw) in enumerate(pieces):
        if pr <= anchor_cell[0] < pr + ph and pc <= anchor_cell[1] < pc + pw:
            true_anchor_idx = i
            break
    if true_anchor_idx is None or not _piece_supports_corner(pieces[true_anchor_idx], corner, RH, RW):
        return bail("true_anchor_missing")

    preferred_decoys: list[tuple[int, int, int]] = []
    fallback_decoys: list[tuple[int, int, int]] = []
    for i, box in enumerate(pieces):
        if i == true_anchor_idx:
            continue
        pr, pc, ph, pw = box
        piece = rect[box[0] : box[0] + box[2], box[1] : box[1] + box[3]]
        before_fill, _ = _piece_stats(piece, rect_col)
        if before_fill < 0.08:
            continue
        shadow = piece.copy()
        if corner == "TL":
            shadow[0, :] = rect_col
            shadow[:, 0] = rect_col
        elif corner == "TR":
            shadow[0, :] = rect_col
            shadow[:, -1] = rect_col
        elif corner == "BL":
            shadow[-1, :] = rect_col
            shadow[:, 0] = rect_col
        else:
            shadow[-1, :] = rect_col
            shadow[:, -1] = rect_col
        after_fill, has_non_rc_edge = _piece_stats(shadow, rect_col)
        if after_fill < 0.03 or not has_non_rc_edge:
            continue
        changed = int(np.sum(piece != shadow))
        target = preferred_decoys if (pr > 0 and pc > 0 and pr + ph < RH and pc + pw < RW) else fallback_decoys
        target.append((ph * pw, changed, i))
    decoy_pool = preferred_decoys or fallback_decoys
    if not decoy_pool:
        return bail("no_anchor_decoys")

    decoy_pool.sort()
    candidate_indices = {true_anchor_idx}
    target_candidates = 2
    for _area, _changed, idx in decoy_pool[: max(1, target_candidates - 1)]:
        candidate_indices.add(idx)
    if len(candidate_indices) < 2:
        return bail("anchor_candidate_count")

    rect2 = rect.copy()
    for idx in candidate_indices:
        _force_candidate_corner(rect2, pieces[idx], corner, rect_col)

    piece_arrays = [rect2[pr : pr + ph, pc : pc + pw].copy() for pr, pc, ph, pw in pieces]
    actual_anchor_candidates = [i for i, piece in enumerate(piece_arrays) if _is_corner_candidate(piece, corner, rect_col)]
    if true_anchor_idx not in actual_anchor_candidates or len(actual_anchor_candidates) < 2:
        return bail("actual_anchor_candidates")

    for i, (pr, pc, ph, pw) in enumerate(pieces):
        piece = piece_arrays[i]
        fill, has_non_rc_edge = _piece_stats(piece, rect_col)
        if fill < 0.03:
            return bail("piece_fill_low_after_corner")
        if fill > 0.75:
            return bail("piece_fill_high_after_corner")
        if not has_non_rc_edge:
            return bail("piece_no_signal_edge_after_corner")
        if pr == 0 and not np.all(piece[0, :] == rect_col):
            return bail("top_border_after_corner")
        if pr + ph == RH and not np.all(piece[-1, :] == rect_col):
            return bail("bottom_border_after_corner")
        if pc == 0 and not np.all(piece[:, 0] == rect_col):
            return bail("left_border_after_corner")
        if pc + pw == RW and not np.all(piece[:, -1] == rect_col):
            return bail("right_border_after_corner")

    distractors: list[np.ndarray] = []
    distractor_meta: list[dict[str, int]] = []
    candidate_order = list(range(len(piece_arrays)))
    rng.shuffle(candidate_order)
    for idx in candidate_order:
        made = _make_distractor(
            piece_arrays[idx],
            pieces[idx],
            rect2,
            rect_col,
            interior_cols,
            pieces,
            piece_arrays + distractors,
            rng,
        )
        if made is None:
            continue
        arr, meta = made
        distractors.append(arr)
        distractor_meta.append(meta)
        if len(distractors) >= n_distractors:
            break
    if len(distractors) != n_distractors:
        return bail("distractor_count")

    marker_cells = _marker_cells(mr, mc, corner)
    for cr, cc in marker_cells:
        if cr < 0 or cr >= GH or cc < 0 or cc >= GW:
            return bail("marker_oob")

    tr, tc = _top_left_from_corner(corner, mr, mc, RH, RW)
    expected_out = np.full((GH, GW), bg, dtype=int)
    expected_out[tr : tr + RH, tc : tc + RW] = rect2

    canvas = np.full((GH, GW), bg, dtype=int)
    for cr, cc in marker_cells:
        canvas[cr, cc] = marker_cols[0]

    apr, apc, aph, apw = pieces[true_anchor_idx]
    anchor_abs_r = tr + apr
    anchor_abs_c = tc + apc
    if anchor_abs_r < 0 or anchor_abs_c < 0 or anchor_abs_r + aph > GH or anchor_abs_c + apw > GW:
        return bail("anchor_abs_oob")
    paste(canvas, piece_arrays[true_anchor_idx], anchor_abs_r, anchor_abs_c)
    placed_piece_indices = {true_anchor_idx}

    for decoy_col in marker_cols[1:]:
        candidate_pairs: list[tuple[int, str]] = []
        for idx, piece_info in enumerate(pieces):
            if idx in placed_piece_indices:
                continue
            for decoy_corner in CORNERS:
                if decoy_corner == corner:
                    continue
                if not _piece_supports_corner(piece_info, decoy_corner, RH, RW):
                    candidate_pairs.append((idx, decoy_corner))
        rng.shuffle(candidate_pairs)
        placed = False
        for idx, decoy_corner in candidate_pairs:
            patch = _make_piece_marker_patch(piece_arrays[idx], decoy_corner, decoy_col, bg)
            try:
                r0, c0 = random_free_location(canvas, patch, transparent=bg, padding=1, border=1)
            except ValueError:
                continue
            paste(canvas, patch, r0, c0, transparent=bg)
            placed_piece_indices.add(idx)
            placed = True
            break
        if not placed:
            return bail("place_decoy_marker")

    remaining_piece_indices = [i for i in range(len(piece_arrays)) if i not in placed_piece_indices]
    rng.shuffle(remaining_piece_indices)
    for idx in remaining_piece_indices:
        try:
            r0, c0 = random_free_location(canvas, piece_arrays[idx], transparent=bg, padding=1, border=1)
        except ValueError:
            return bail("place_piece")
        paste(canvas, piece_arrays[idx], r0, c0)

    all_arrays = list(distractors)
    rng.shuffle(all_arrays)
    for arr in all_arrays:
        try:
            r0, c0 = random_free_location(canvas, arr, transparent=bg, padding=1, border=1)
        except ValueError:
            return bail("place_distractor")
        paste(canvas, arr, r0, c0)

    counts = Counter(map(int, canvas.flatten()))
    non_marker_colors = set(int(c) for c in interior_cols + [rect_col])
    if any(counts[col] == 3 for col in non_marker_colors):
        return bail("spurious_count3")

    try:
        sols = _solve_all(canvas)
    except Exception as e:
        return bail(f"solve_exception:{type(e).__name__}:{e}")
    if not sols:
        return bail("no_solution")

    sigs = {tuple(map(int, sol["output"].flatten())) for sol in sols}
    valid_marker_colors = {int(sol["marker_color"]) for sol in sols}
    valid_anchor_indices = {int(sol["anchor_idx"]) for sol in sols}
    if sigs != {tuple(map(int, expected_out.flatten()))}:
        return bail("wrong_output")
    if valid_marker_colors != {int(marker_cols[0])}:
        return bail("multi_valid_markers")
    if valid_anchor_indices != {int(true_anchor_idx)}:
        return bail("multi_valid_anchors")
    if len(_find_candidate_markers(canvas, bg)) != n_markers:
        return bail("marker_detection_count")

    meta = {
        "marker_count": int(n_markers),
        "distractor_count": int(len(distractors)),
        "anchor_candidate_count": int(len(actual_anchor_candidates)),
        "valid_marker_count": int(len(valid_marker_colors)),
        "valid_anchor_count": int(len(valid_anchor_indices)),
        "solution_count": int(len(sols)),
        "true_piece_count": int(len(piece_arrays)),
        "partial_sides_min": int(min(m["partial_sides"] for m in distractor_meta)),
        "partial_sides_max": int(max(m["partial_sides"] for m in distractor_meta)),
    }
    if return_meta:
        return canvas, meta
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
