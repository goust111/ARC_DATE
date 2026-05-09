# Combo variant of 446ef5d2 (index 12)
# combo_spec: D2/3+D11/0
# Name: 干扰碎片配合极少碎片

from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np

from arc_utils import paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    Original task: every scattered rectangle is genuine, so `main()` can sum all
#    piece areas, enumerate factor pairs of that single area, and tile with all
#    pieces. Here that fails in two places at once:
#    - D2/3 adds 1-2 distractor rectangles, so "use every rectangle" is wrong.
#    - D11/0 shrinks the true set to only 2-3 large rectangles, so the useful
#      search object is no longer "all pieces", but "which 2-3 pieces form the
#      anchored rectangle".
#    Example trace:
#    - Input has 4 rectangles total: 2 large true halves plus 2 mutated decoys.
#    - Old logic would sum area of all 4 and search for a 4-piece tiling.
#    - New logic instead enumerates only 2-piece and 3-piece subsets, scores
#      larger / larger-piece subsets first, and assembles the first subset that
#      tiles a rectangle anchored at the marker. The decoys stay unused.
# b) Complexity calibration:
#    True piece count is restricted to 2-3 to keep assembly itself simple, but
#    the large pieces have richer internal patterns and every internal seam is
#    required to have multiple non-rect-col matching pixels. This keeps the
#    "which pieces belong" step visible instead of collapsing into trivial plain
#    halves. Distractors are limited to 1-2 pieces so the grid stays readable.
# c) Broken implicit assumptions from the original solver:
#    - `total_area = sum(all piece areas)` is false once distractors exist.
#    - "all rectangles must be used" is false.
#    - Searching over arbitrary piece counts is unnecessary noise now that the
#      generator guarantees exactly 2-3 genuine pieces.
#    The new solver therefore only tests 2-piece and 3-piece subsets.
# d) Placement legality:
#    Generator remains output-first, then scatter. Every rectangle is axis-aligned,
#    non-overlapping, and placed with padding >= 1. Marker rules are unchanged.
#    Larger pieces mean scatter can fail more often, so rectangle sizes are kept
#    moderate relative to the canvas and the generator validates by re-solving.
# e) Visual observability:
#    Across examples, humans can see there are only a few large genuine chunks
#    plus extra rectangles. The output always uses only 2-3 large pieces. The
#    generator also enforces at least two informative matched pixels on each true
#    internal seam so the join is visibly encoded in the artwork, not hidden in
#    wildcard-only rect_col borders.
# f) Visual cleanliness:
#    No extra symbols are introduced beyond the original L marker. Because the
#    true pieces are few and large, and distractors are only 1-2, the grid does
#    not become cluttered even though the piece-selection step is now essential.
# g) Color inference:
#    No new color mapping exists. Distractors reuse the same rect_col and the
#    same 2-3 interior colors; only structure decides whether a rectangle belongs.
# h) Quantitative anti-degeneracy check:
#    The key combo metric is:
#    - `true_piece_count` must be 2 or 3;
#    - `distractor_count` must be >= 1;
#    - `same_as_original_rate` should be 0 on sampled inputs, because the
#      original solver wrongly assumes all pieces are genuine.
#    We also track `adj_signal_min`, the minimum number of visibly matching
#    non-rect-col pixels along any true internal seam. Keeping it >= 2 ensures
#    the join evidence is human-visible rather than hidden behind wildcard edges.
# 9) New solver assumptions and generator guarantees:
#    - Assumption: the intended solution uses exactly 2 or 3 rectangles.
#      Guarantee: `_partition_large()` only creates 2 or 3 true pieces.
#    - Assumption: the best successful subset is the intended one.
#      Guarantee: generator validates `main(input) == expected_output`.
#    - Assumption: every true seam has visible non-rect matching evidence.
#      Guarantee: `_adjacency_signal_counts()` rejects weak seam layouts.
#    - Assumption: distractors are plausible but invalid.
#      Guarantee: distractors are mutated true pieces that keep shape/palette
#      while failing at least one required side and fully matching no true slot.
# 10) Code-path assumption review:
#    - `_candidate_dims()` assumes area comes from a 2-3 piece subset, not from
#      all rectangles. `_gen_attempt()` guarantees that by construction.
#    - `_assemble_subset()` assumes every true piece keeps its orientation and
#      tiles an axis-aligned rectangle without rotation. The generator scatters
#      by translation only, so this remains valid.
#    - `main()` ranks subsets by larger total area / larger minimum piece area.
#      This assumes distractors stay smaller or no better than the intended
#      subset. Generator-side validation rejects any case where that ranking
#      could return a different assembled rectangle.


def _find_marker(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int], tuple[int, int, int, int]]:
    cnt = Counter(map(int, g.flatten()))
    for col in sorted(c for c, n in cnt.items() if int(c) != int(bg) and int(n) == 3):
        pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(col))]
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if (r1 - r0) != 1 or (c1 - c0) != 1:
            continue
        all4 = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        miss = list(all4 - set(pts))
        if len(miss) == 1:
            return int(col), miss[0], (r0, c0, r1, c1)
    raise AssertionError("marker not found")


def _marker_corner(missing: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    if missing == (r0, c0):
        return "BR"
    if missing == (r0, c1):
        return "BL"
    if missing == (r1, c0):
        return "TR"
    return "TL"


def _fit_top_left(corner: str, mr: int, mc: int, H: int, W: int) -> tuple[int, int]:
    if corner == "TL":
        return mr, mc
    if corner == "TR":
        return mr, mc - W + 1
    if corner == "BL":
        return mr - H + 1, mc
    return mr - H + 1, mc - W + 1


def _extract_rectangles(g: np.ndarray, bg: int, mark: int) -> list[tuple[int, int, int, int, np.ndarray]]:
    GH, GW = g.shape
    occ = (g != int(bg)) & (g != int(mark))
    used = np.zeros_like(occ, dtype=bool)
    rects: list[tuple[int, int, int, int, np.ndarray]] = []
    while True:
        idx = np.argwhere(occ & (~used))
        if idx.size == 0:
            break
        rr, cc = map(int, idx[0])
        w = 0
        while cc + w < GW and bool(occ[rr, cc + w]) and (not bool(used[rr, cc + w])):
            w += 1
        h = 0
        while rr + h < GH and bool((occ[rr + h, cc : cc + w] & (~used[rr + h, cc : cc + w])).all()):
            h += 1
        used[rr : rr + h, cc : cc + w] = True
        rects.append((rr, cc, h, w, g[rr : rr + h, cc : cc + w].copy()))
    return rects


def _compat(v1: np.ndarray, v2: np.ndarray, rect_col: int) -> bool:
    a = np.array(v1, dtype=int)
    b = np.array(v2, dtype=int)
    m = (a != int(rect_col)) & (b != int(rect_col))
    return bool((a[m] == b[m]).all())


def _piece_has_non_rc_edge(piece: np.ndarray, rect_col: int) -> bool:
    return any(np.any(edge != int(rect_col)) for edge in (piece[0, :], piece[-1, :], piece[:, 0], piece[:, -1]))


def _candidate_dims(corner: str, mr: int, mc: int, GH: int, GW: int, area: int) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for hh in range(1, int(area**0.5) + 1):
        if area % hh != 0:
            continue
        for H, W in ((hh, area // hh), (area // hh, hh)):
            H = int(H)
            W = int(W)
            if (H, W) in seen:
                continue
            tr, tc = _fit_top_left(corner, mr, mc, H, W)
            if 0 <= tr and 0 <= tc and (tr + H) <= GH and (tc + W) <= GW:
                seen.add((H, W))
                dims.append((H, W))
    dims.sort(key=lambda t: (-t[0] * t[1], -min(t[0], t[1]), -max(t[0], t[1])))
    return dims


def _assemble_subset(piece_arrays: list[np.ndarray], rect_col: int, H: int, W: int) -> np.ndarray | None:
    occ = np.full((H, W), -1, dtype=int)
    areas = [int(a.shape[0] * a.shape[1]) for a in piece_arrays]
    order = sorted(range(len(piece_arrays)), key=lambda i: (-areas[i], -piece_arrays[i].shape[0], -piece_arrays[i].shape[1], i))
    used = [False] * len(piece_arrays)

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
            left = occ[r : r + h, c - 1]
            m = left != -1
            if m.any() and (not _compat(left[m], a[:, 0][m], rect_col)):
                return False
        if (c + w) < W:
            right = occ[r : r + h, c + w]
            m = right != -1
            if m.any() and (not _compat(right[m], a[:, -1][m], rect_col)):
                return False
        if r > 0:
            up = occ[r - 1, c : c + w]
            m = up != -1
            if m.any() and (not _compat(up[m], a[0, :][m], rect_col)):
                return False
        if (r + h) < H:
            down = occ[r + h, c : c + w]
            m = down != -1
            if m.any() and (not _compat(down[m], a[-1, :][m], rect_col)):
                return False
        return True

    def can_place(i: int, r: int, c: int) -> bool:
        a = piece_arrays[i]
        h, w = a.shape
        if (r + h) > H or (c + w) > W:
            return False
        if (occ[r : r + h, c : c + w] != -1).any():
            return False
        if not boundary_ok(a, r, c):
            return False
        if not neighbor_ok(a, r, c):
            return False
        return True

    def place(i: int, r: int, c: int) -> None:
        a = piece_arrays[i]
        h, w = a.shape
        occ[r : r + h, c : c + w] = a
        used[i] = True

    def unplace(i: int, r: int, c: int) -> None:
        a = piece_arrays[i]
        h, w = a.shape
        occ[r : r + h, c : c + w] = -1
        used[i] = False

    def next_empty() -> tuple[int, int] | None:
        flat = np.flatnonzero(occ.ravel() == -1)
        if flat.size == 0:
            return None
        idx = int(flat[0])
        return idx // W, idx % W

    def bt(placed_area: int) -> bool:
        if placed_area == H * W:
            return True
        nxt = next_empty()
        if nxt is None:
            return False
        r, c = nxt
        for i in order:
            if used[i]:
                continue
            if can_place(i, r, c):
                place(i, r, c)
                if bt(placed_area + areas[i]):
                    return True
                unplace(i, r, c)
        return False

    if bt(0):
        return occ
    return None


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    GH, GW = g.shape
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    mark, missing, mb = _find_marker(g, bg)
    corner = _marker_corner(missing, mb)
    rects = _extract_rectangles(g, bg, mark)
    assert rects

    allpix: list[int] = []
    for _r, _c, _h, _w, a in rects:
        allpix.extend(map(int, a.flatten()))
    rect_col = int(Counter(allpix).most_common(1)[0][0])

    pieces = [a for _r, _c, _h, _w, a in rects]
    areas = [int(p.shape[0] * p.shape[1]) for p in pieces]
    mr, mc = map(int, missing)

    candidates: list[tuple[tuple[int, int, int, int, int, int], tuple[int, ...], int, int]] = []
    for k in (2, 3):
        for idxs in combinations(range(len(pieces)), k):
            subset_areas = [areas[i] for i in idxs]
            area = int(sum(subset_areas))
            dims = _candidate_dims(corner, mr, mc, GH, GW, area)
            if not dims:
                continue
            score = (
                int(area),
                int(min(subset_areas)),
                int(max(subset_areas)),
                -int(k),
                int(max(d[0] * d[1] for d in dims)),
                -int(min(idxs)),
            )
            for H, W in dims:
                candidates.append((score, tuple(idxs), int(H), int(W)))

    candidates.sort(key=lambda item: item[0], reverse=True)

    ass = None
    used_dim = None
    for _score, idxs, H, W in candidates:
        subset = [pieces[i] for i in idxs]
        ass = _assemble_subset(subset, rect_col, H, W)
        if ass is not None:
            used_dim = (H, W)
            break

    assert ass is not None and used_dim is not None

    H, W = used_dim
    tr, tc = _fit_top_left(corner, mr, mc, H, W)
    out = np.full((GH, GW), int(bg), dtype=int)
    out[tr : tr + H, tc : tc + W] = ass
    return out


def _build_rectangle(RH: int, RW: int, rect_col: int, interior_cols: list[int], rng) -> np.ndarray | None:
    rect = np.full((RH, RW), rect_col, dtype=int)
    iH, iW = RH - 2, RW - 2
    if iH < 1 or iW < 1:
        return None

    interior = np.full((iH, iW), rect_col, dtype=int)
    target_fill = rng.uniform(0.40, 0.82)
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
            sr, sc = rng.randint(0, iH - 1), rng.randint(0, iW - 1)
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
            rng.shuffle(frontier)
            nf = []
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


def _partition_large(RH: int, RW: int, rng) -> list[tuple[int, int, int, int]] | None:
    cand2: list[list[tuple[int, int, int, int]]] = []
    cand3: list[list[tuple[int, int, int, int]]] = []

    for split in range(4, RW - 3):
        slots = [(0, 0, RH, split), (0, split, RH, RW - split)]
        if min(split, RW - split) >= 4 and min(RH * split, RH * (RW - split)) >= 18:
            cand2.append(slots)
    for split in range(4, RH - 3):
        slots = [(0, 0, split, RW), (split, 0, RH - split, RW)]
        if min(split, RH - split) >= 4 and min(split * RW, (RH - split) * RW) >= 18:
            cand2.append(slots)

    for left_w in range(4, RW - 7):
        right_w = RW - left_w
        for split_r in range(4, RH - 3):
            bottom_h = RH - split_r
            slots = [(0, 0, RH, left_w), (0, left_w, split_r, right_w), (split_r, left_w, bottom_h, right_w)]
            dims_ok = min(left_w, right_w, split_r, bottom_h) >= 4
            areas_ok = min(RH * left_w, split_r * right_w, bottom_h * right_w) >= 18
            if dims_ok and areas_ok:
                cand3.append(slots)

    for top_h in range(4, RH - 7):
        bottom_h = RH - top_h
        for split_c in range(4, RW - 3):
            right_w = RW - split_c
            slots = [(0, 0, top_h, RW), (top_h, 0, bottom_h, split_c), (top_h, split_c, bottom_h, right_w)]
            dims_ok = min(top_h, bottom_h, split_c, right_w) >= 4
            areas_ok = min(top_h * RW, bottom_h * split_c, bottom_h * right_w) >= 18
            if dims_ok and areas_ok:
                cand3.append(slots)

    if not cand2 and not cand3:
        return None

    want3 = bool(cand3) and (not cand2 or rng.random() < 0.55)
    pool = cand3 if want3 else cand2
    return list(rng.choice(pool))


def _adjacencies(slots: list[tuple[int, int, int, int]]) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    for i in range(len(slots)):
        pr, pc, ph, pw = slots[i]
        for j in range(i + 1, len(slots)):
            qr, qc, qh, qw = slots[j]
            if pc + pw == qc or qc + qw == pc:
                overlap = min(pr + ph, qr + qh) - max(pr, qr)
                if overlap > 0:
                    out.append((i, j, "v"))
            if pr + ph == qr or qr + qh == pr:
                overlap = min(pc + pw, qc + qw) - max(pc, qc)
                if overlap > 0:
                    out.append((i, j, "h"))
    return out


def _adjacency_signal_counts(rect: np.ndarray, slots: list[tuple[int, int, int, int]], rect_col: int) -> list[int]:
    counts: list[int] = []
    RH, RW = rect.shape
    for i, j, axis in _adjacencies(slots):
        pr, pc, ph, pw = slots[i]
        qr, qc, qh, qw = slots[j]
        if axis == "v":
            if pc + pw == qc:
                left_c, right_c = pc + pw - 1, qc
                r0, r1 = max(pr, qr), min(pr + ph, qr + qh)
            else:
                left_c, right_c = qc + qw - 1, pc
                r0, r1 = max(pr, qr), min(pr + ph, qr + qh)
            rows = range(max(1, r0), min(RH - 1, r1))
            count = sum(
                1
                for r in rows
                if rect[r, left_c] != int(rect_col)
                and rect[r, right_c] != int(rect_col)
                and rect[r, left_c] == rect[r, right_c]
            )
        else:
            if pr + ph == qr:
                top_r, bottom_r = pr + ph - 1, qr
                c0, c1 = max(pc, qc), min(pc + pw, qc + qw)
            else:
                top_r, bottom_r = qr + qh - 1, pr
                c0, c1 = max(pc, qc), min(pc + pw, qc + qw)
            cols = range(max(1, c0), min(RW - 1, c1))
            count = sum(
                1
                for c in cols
                if rect[top_r, c] != int(rect_col)
                and rect[bottom_r, c] != int(rect_col)
                and rect[top_r, c] == rect[bottom_r, c]
            )
        counts.append(int(count))
    return counts


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
    rng,
) -> tuple[np.ndarray, dict[str, int]] | None:
    pr, pc, ph, pw = slot
    RH, RW = rect.shape
    if len(interior_cols) < 2:
        return None

    side_specs = []
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
            if fill < 0.08 or fill > 0.75:
                continue
            if not _piece_has_non_rc_edge(cand, rect_col):
                continue
            if any(np.array_equal(cand, other) for other in existing):
                continue

            side_checks = _slot_side_checks(cand, slot, rect, rect_col)
            if sum(side_checks) < 2 or all(side_checks):
                continue
            if any(all(_slot_side_checks(cand, s, rect, rect_col)) for s in slots):
                continue

            return cand, {"partial_sides": int(sum(side_checks)), "base_area": int(ph * pw)}
    return None


def _piece_valid(
    piece: np.ndarray,
    slot: tuple[int, int, int, int],
    rect: np.ndarray,
    rect_col: int,
) -> bool:
    RH, RW = rect.shape
    pr, pc, ph, pw = slot
    if piece.shape != (ph, pw):
        return False
    if min(ph, pw) < 4:
        return False
    if ph * pw < 18:
        return False
    fill = float(np.mean(piece != rect_col))
    if fill < 0.08 or fill > 0.75:
        return False
    if not _piece_has_non_rc_edge(piece, rect_col):
        return False
    if pr == 0 and not np.all(piece[0, :] == rect_col):
        return False
    if pr + ph == RH and not np.all(piece[-1, :] == rect_col):
        return False
    if pc == 0 and not np.all(piece[:, 0] == rect_col):
        return False
    if pc + pw == RW and not np.all(piece[:, -1] == rect_col):
        return False
    return True


def _marker_cells(corner: str, mr: int, mc: int) -> list[tuple[int, int]]:
    if corner == "TL":
        return [(mr - 1, mc - 1), (mr - 1, mc), (mr, mc - 1)]
    if corner == "TR":
        return [(mr - 1, mc), (mr - 1, mc + 1), (mr, mc + 1)]
    if corner == "BL":
        return [(mr, mc - 1), (mr + 1, mc - 1), (mr + 1, mc)]
    return [(mr, mc + 1), (mr + 1, mc), (mr + 1, mc + 1)]


def _gen_attempt(return_meta: bool = False):
    import random as _r

    GH = _r.randint(16, 26)
    GW = _r.randint(16, 26)

    palette = list(range(10))
    _r.shuffle(palette)
    bg = palette[0]
    rect_col = palette[1]
    marker_col = palette[2]
    n_interior = _r.randint(2, 3)
    interior_cols = palette[3 : 3 + n_interior]

    corner = _r.choice(["TL", "TR", "BL", "BR"])

    max_side = min(16, min(GH, GW) - 4)
    if max_side < 8:
        return None
    RH = _r.randint(8, max_side)
    RW = _r.randint(8, max_side)

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
        return None
    mr = _r.randint(mr_min, mr_max)
    mc = _r.randint(mc_min, mc_max)

    rect = _build_rectangle(RH, RW, rect_col, interior_cols, _r)
    if rect is None:
        return None

    slots = _partition_large(RH, RW, _r)
    if slots is None:
        return None

    signals = _adjacency_signal_counts(rect, slots, rect_col)
    if not signals or min(signals) < 2:
        return None

    piece_arrays = [rect[pr : pr + ph, pc : pc + pw].copy() for pr, pc, ph, pw in slots]
    for slot, piece in zip(slots, piece_arrays):
        if not _piece_valid(piece, slot, rect, rect_col):
            return None

    if corner == "TL":
        anchor_r, anchor_c = 0, 0
    elif corner == "TR":
        anchor_r, anchor_c = 0, RW - 1
    elif corner == "BL":
        anchor_r, anchor_c = RH - 1, 0
    else:
        anchor_r, anchor_c = RH - 1, RW - 1

    anchor_hits = [i for i, (pr, pc, ph, pw) in enumerate(slots) if pr <= anchor_r < pr + ph and pc <= anchor_c < pc + pw]
    if len(anchor_hits) != 1:
        return None

    marker_cells = _marker_cells(corner, mr, mc)
    for cr, cc in marker_cells:
        if cr < 0 or cr >= GH or cc < 0 or cc >= GW:
            return None

    distractor_target = 1 if len(piece_arrays) == 2 else _r.randint(1, 2)
    distractors: list[np.ndarray] = []
    distractor_meta: list[dict[str, int]] = []
    candidate_order = list(range(len(piece_arrays)))
    _r.shuffle(candidate_order)
    for idx in candidate_order:
        made = _make_distractor(
            piece_arrays[idx],
            slots[idx],
            rect,
            rect_col,
            interior_cols,
            slots,
            piece_arrays + distractors,
            _r,
        )
        if made is None:
            continue
        arr, meta = made
        distractors.append(arr)
        distractor_meta.append(meta)
        if len(distractors) >= distractor_target:
            break
    if len(distractors) != distractor_target:
        return None

    canvas = np.full((GH, GW), bg, dtype=int)
    for cr, cc in marker_cells:
        canvas[cr, cc] = marker_col

    all_arrays = piece_arrays + distractors
    order = list(range(len(all_arrays)))
    _r.shuffle(order)
    for idx in order:
        pa = all_arrays[idx]
        try:
            r0, c0 = random_free_location(canvas, pa, transparent=bg, padding=1)
            paste(canvas, pa, r0, c0)
        except ValueError:
            return None

    tr, tc = _fit_top_left(corner, mr, mc, RH, RW)
    expected_out = np.full((GH, GW), bg, dtype=int)
    expected_out[tr : tr + RH, tc : tc + RW] = rect

    try:
        actual_out = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual_out, expected_out):
        return None

    if return_meta:
        return canvas, {
            "true_piece_count": int(len(piece_arrays)),
            "distractor_count": int(len(distractors)),
            "true_min_area": int(min(a.shape[0] * a.shape[1] for a in piece_arrays)),
            "true_max_area": int(max(a.shape[0] * a.shape[1] for a in piece_arrays)),
            "adj_signal_min": int(min(signals)),
            "adj_signal_max": int(max(signals)),
            "distractor_partial_min": int(min(m["partial_sides"] for m in distractor_meta)),
            "distractor_partial_max": int(max(m["partial_sides"] for m in distractor_meta)),
        }
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
