# Combo variant of 446ef5d2 (index 4)
# combo_spec: D4/0+D12/2
# Name: 重新上色配合对称内部图案

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import paste, random_free_location

"""
Design analysis for the combo variant
=====================================

combo_rule:
  - stage 1: recover the original rectangle by translating rectangular pieces so
    that shared edges agree, with `rect_col` still acting as wildcard.
  - stage 2: among valid full assemblies, prefer ones whose interior pattern is
    symmetric.
  - stage 3: after assembly, recolor every non-`rect_col` pixel to the L-marker
    color. The border stays `rect_col`.

a) concretization_and_trace:
  - baseline behavior:
      input has scattered axis-aligned rectangular pieces plus a 3-cell L marker.
      Solver finds the rectangle pinned at the marker corner and writes the
      assembled colors back unchanged.
  - combo behavior:
      the scattered pieces still tile one rectangle, but their non-border colors
      were generated from a symmetric interior motif.
      Example: suppose the true rectangle interior is vertically symmetric and
      uses colors 2 and 4 around a border color 7. The input pieces show partial
      fragments of that mirrored motif. Solver assembles candidates by edge
      matching; if several complete tilings are possible, the intended one has a
      higher symmetry score on its interior. After choosing that rectangle, every
      2/4 pixel becomes the marker color, say 9. Final output keeps only bg, 7, 9.
  - difference from original:
      symmetry is an input-side structural cue that helps verify the assembly,
      while recoloring is an output-side post-process that changes the rendered
      rectangle after the correct assembly is known.

b) complexity_calibration:
  - Recoloring alone would be too easy if the input already looked arbitrary, so
    the generator enforces 2-3 interior colors before recolor.
  - Symmetry alone would be too weak if it only produced tiny accidental matches,
    so rectangle sides are kept at least 7 and interior fill is constrained to a
    visible 38-60% band.
  - The symmetry family set is small and repeated across samples:
      vertical reflection, horizontal reflection, 180-degree point symmetry,
      horizontal translation, vertical translation.
    This gives observable regularity without exploding the hypothesis space.
  - Piece count is biased to 4-9 to keep the search tractable while preserving a
    genuine assembly task.

c) implicit_assumption_audit_of_main:
  - Marker detection still assumes exactly one non-bg color appears 3 times in a
    2x2 L shape. Generator preserves that.
  - Piece decomposition still assumes each scattered object is a solid rectangle
    of occupied cells. Generator preserves that.
  - `rect_col` inference still assumes the border/wildcard color dominates piece
    pixel counts. Generator caps interior fill so non-border colors do not exceed
    it.
  - Boundary checks still assume the assembled outer border is entirely
    `rect_col`. Generator keeps a strict 1-cell border.
  - Edge compatibility still only compares non-`rect_col` pixels across touching
    boundaries. Symmetry does not break that local rule.
  - The original solver stopped at the first valid assembly. That is the fragile
    line under D12/2, because different valid tilings can differ globally. The
    combo solver therefore scores completed assemblies by symmetry before the
    final recolor write-back.
  - The original final write-back copied the assembly as-is. That is the fragile
    line under D4/0, so the combo solver explicitly recolors all non-`rect_col`
    assembled pixels to `marker_col`.

d) placement_legality:
  - Generator is still output-first: build the full rectangle, partition it into
    legal rectangular pieces, scatter them with >= 1 background padding, then
    place the marker.
  - Marker cells are written before scattering pieces, so `random_free_location`
    prevents overlap with both marker and earlier pieces.
  - Corner-aware marker ranges guarantee the final rectangle would fit on-grid in
    the output position.
  - Every successful generated sample is validated with `main(canvas)` against
    the expected recolored output.

e) visual_observability:
  - A human can see the input fragments carry mirrored or periodic colored bands.
    That supports the “assemble then verify by symmetry” part.
  - The output then clearly collapses all interior colors to the marker color.
    Because the marker is the only 3-cell unique object, it is the obvious source
    of the recolor target.
  - No hidden use of numeric color ids is involved; only visible marker color and
    visible positional symmetry matter.

f) visual_tidiness:
  - The scene stays clean: one background, one marker, several rectangular pieces.
  - Symmetric motifs are blocky and aligned to the grid, so scattered fragments
    remain readable instead of turning into noisy speckle.
  - The output is even cleaner: one bordered rectangle with a uniform recolored
    interior.

g) change_inferability:
  - To avoid degenerating back to the original task, the generator requires:
      * at least two pre-recolor interior colors,
      * at least one detected symmetry family in the assembled interior,
      * at least one recolored pixel.
  - Therefore “assemble only” and “assemble + recolor” produce visibly different
    outputs, and the symmetric interior actually exists in the pre-recolor object.

h) new_solver_assumptions_and_generator_guarantees:
  - New solver assumption: the intended assembly should score well on the same
    symmetry detector used by the generator. Generator guarantees that by
    constructing the rectangle from one explicit symmetry family.
  - New solver assumption: recoloring every non-`rect_col` pixel is meaningful.
    Generator guarantees this by forcing 2-3 non-`rect_col` colors and nonzero
    colored area.
"""


_LAST_GENERATION_INFO: dict[str, object] = {}


def _find_marker(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int], tuple[int, int, int, int]]:
    cnt = Counter(map(int, grid.flatten()))
    for col in sorted(c for c, n in cnt.items() if int(c) != int(bg) and int(n) == 3):
        pts = [(int(r), int(c)) for r, c in np.argwhere(grid == int(col))]
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


def _corner_from_missing(missing: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    if missing == (r0, c0):
        return "BR"
    if missing == (r0, c1):
        return "BL"
    if missing == (r1, c0):
        return "TR"
    return "TL"


def _top_left_from_corner(mr: int, mc: int, H: int, W: int, corner: str) -> tuple[int, int]:
    if corner == "TL":
        return mr, mc
    if corner == "TR":
        return mr, mc - W + 1
    if corner == "BL":
        return mr - H + 1, mc
    return mr - H + 1, mc - W + 1


def _marker_cells(corner: str, mr: int, mc: int) -> list[tuple[int, int]]:
    if corner == "TL":
        return [(mr - 1, mc - 1), (mr - 1, mc), (mr, mc - 1)]
    if corner == "TR":
        return [(mr - 1, mc), (mr - 1, mc + 1), (mr, mc + 1)]
    if corner == "BL":
        return [(mr, mc - 1), (mr + 1, mc - 1), (mr + 1, mc)]
    return [(mr, mc + 1), (mr + 1, mc), (mr + 1, mc + 1)]


def _recolor_assembled(rect: np.ndarray, rect_col: int, marker_col: int) -> np.ndarray:
    out = np.array(rect, copy=True, dtype=int)
    out[out != int(rect_col)] = int(marker_col)
    return out


def _symmetry_types(arr: np.ndarray) -> list[str]:
    kinds: list[str] = []
    if arr.size == 0:
        return kinds
    if np.array_equal(arr, np.fliplr(arr)):
        kinds.append("vertical")
    if np.array_equal(arr, np.flipud(arr)):
        kinds.append("horizontal")
    if np.array_equal(arr, np.flipud(np.fliplr(arr))):
        kinds.append("point")

    H, W = arr.shape
    for p in range(2, min(4, W) + 1):
        if W % p == 0 and W // p >= 2:
            base = arr[:, :p]
            if all(np.array_equal(arr[:, c : c + p], base) for c in range(0, W, p)):
                kinds.append(f"translation_x:{p}")
                break
    for p in range(2, min(4, H) + 1):
        if H % p == 0 and H // p >= 2:
            base = arr[:p, :]
            if all(np.array_equal(arr[r : r + p, :], base) for r in range(0, H, p)):
                kinds.append(f"translation_y:{p}")
                break
    return kinds


def _assembly_score(rect: np.ndarray, rect_col: int) -> tuple[int, int, int]:
    if rect.shape[0] <= 2 or rect.shape[1] <= 2:
        return (0, 0, 0)
    interior = rect[1:-1, 1:-1]
    kinds = _symmetry_types(interior)
    non_rect = int(np.sum(interior != int(rect_col)))
    n_colors = len(set(map(int, np.unique(interior[interior != int(rect_col)]))))
    return (len(kinds), non_rect, n_colors)


def _decompose_rectangles(grid: np.ndarray, bg: int, mark: int) -> list[tuple[int, int, int, int, np.ndarray]]:
    GH, GW = grid.shape
    occ = (grid != int(bg)) & (grid != int(mark))
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
        rects.append((rr, cc, h, w, grid[rr : rr + h, cc : cc + w].copy()))
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered rectangular pieces into a rectangle at an L-marker corner.

    Combo-specific behavior:
    1. solve the original tiling problem using `rect_col` as wildcard,
    2. prefer assemblies whose interior pattern is symmetric,
    3. recolor all non-`rect_col` assembled pixels to the marker color.
    """
    g = np.array(grid, dtype=int)
    GH, GW = g.shape
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])

    mark, missing, mb = _find_marker(g, bg)
    corner = _corner_from_missing(missing, mb)
    mr, mc = map(int, missing)

    rects = _decompose_rectangles(g, bg, mark)
    total_area = int(sum(int(h) * int(w) for _r, _c, h, w, _a in rects))

    allpix: list[int] = []
    for _r, _c, _h, _w, arr in rects:
        allpix.extend(map(int, arr.flatten()))
    rect_col = int(Counter(allpix).most_common(1)[0][0])

    anchor_idx = None
    for i, (rr, cc, h, w, _arr) in enumerate(rects):
        if rr <= mr < rr + h and cc <= mc < cc + w:
            anchor_idx = i
            break

    pieces = [arr for _r, _c, _h, _w, arr in rects]
    ph = [int(p.shape[0]) for p in pieces]
    pw = [int(p.shape[1]) for p in pieces]

    dims: list[tuple[int, int]] = []
    for hh in range(1, int(total_area**0.5) + 1):
        if total_area % hh != 0:
            continue
        for H, W in ((hh, total_area // hh), (total_area // hh, hh)):
            tr, tc = _top_left_from_corner(mr, mc, int(H), int(W), corner)
            if 0 <= tr and 0 <= tc and (tr + int(H)) <= GH and (tc + int(W)) <= GW:
                dims.append((int(H), int(W)))
    dims = sorted(set(dims), key=lambda t: (-t[0] * t[1], -t[0], -t[1]))

    def compat(v1: np.ndarray, v2: np.ndarray) -> bool:
        mask = (v1 != int(rect_col)) & (v2 != int(rect_col))
        return bool((v1[mask] == v2[mask]).all())

    def assemble_best(H: int, W: int) -> np.ndarray | None:
        occ2 = np.full((H, W), -1, dtype=int)
        usedp = [False] * len(pieces)
        best_ass: np.ndarray | None = None
        best_score = (-1, -1, -1)
        solutions_seen = 0
        solution_cap = 12

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
                if m.any() and (not compat(left[m], a[:, 0][m])):
                    return False
            if (c + w) < W:
                right = occ2[r : r + h, c + w]
                m = right != -1
                if m.any() and (not compat(right[m], a[:, -1][m])):
                    return False
            if r > 0:
                up = occ2[r - 1, c : c + w]
                m = up != -1
                if m.any() and (not compat(up[m], a[0, :][m])):
                    return False
            if (r + h) < H:
                down = occ2[r + h, c : c + w]
                m = down != -1
                if m.any() and (not compat(down[m], a[-1, :][m])):
                    return False
            return True

        def can_place(i: int, r: int, c: int) -> bool:
            a = pieces[i]
            h, w = a.shape
            if (r + h) > H or (c + w) > W:
                return False
            if (occ2[r : r + h, c : c + w] != -1).any():
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
            usedp[i] = True

        def unplace(i: int, r: int, c: int) -> None:
            a = pieces[i]
            h, w = a.shape
            occ2[r : r + h, c : c + w] = -1
            usedp[i] = False

        if anchor_idx is not None:
            rr, cc, h0, w0, _arr = rects[anchor_idx]
            corners = {
                (rr, cc): "TL",
                (rr, cc + w0 - 1): "TR",
                (rr + h0 - 1, cc): "BL",
                (rr + h0 - 1, cc + w0 - 1): "BR",
            }
            if (mr, mc) in corners and corners[(mr, mc)] == corner:
                ah, aw = ph[anchor_idx], pw[anchor_idx]
                ar0, ac0 = _top_left_from_corner(mr, mc, ah, aw, corner)
                tr, tc = _top_left_from_corner(mr, mc, H, W, corner)
                ar0 -= tr
                ac0 -= tc
                if can_place(anchor_idx, ar0, ac0):
                    place(anchor_idx, ar0, ac0)

        order = sorted(range(len(pieces)), key=lambda i: (-ph[i] * pw[i], -ph[i], -pw[i], i))

        def next_empty() -> tuple[int, int] | None:
            idx = np.argwhere(occ2 == -1)
            if idx.size == 0:
                return None
            return tuple(map(int, idx[0]))

        def bt() -> bool:
            nonlocal best_ass, best_score, solutions_seen
            nxt = next_empty()
            if nxt is None:
                solutions_seen += 1
                score = _assembly_score(occ2, rect_col)
                if score > best_score:
                    best_score = score
                    best_ass = occ2.copy()
                return solutions_seen < solution_cap

            r, c = nxt
            for i in order:
                if usedp[i] or not can_place(i, r, c):
                    continue
                place(i, r, c)
                keep_going = bt()
                unplace(i, r, c)
                if not keep_going:
                    return False
            return True

        bt()
        return best_ass

    best_ass: np.ndarray | None = None
    best_dim: tuple[int, int] | None = None
    best_score = (-1, -1, -1)
    for H, W in dims:
        ass = assemble_best(H, W)
        if ass is None:
            continue
        score = _assembly_score(ass, rect_col)
        if score > best_score:
            best_score = score
            best_ass = ass
            best_dim = (H, W)

    assert not (best_ass is None or best_dim is None)

    H, W = best_dim
    tr, tc = _top_left_from_corner(mr, mc, H, W, corner)
    out = np.full((GH, GW), int(bg), dtype=int)
    out[tr : tr + H, tc : tc + W] = _recolor_assembled(best_ass, rect_col, mark)
    return out


def _band_mask(length: int, rng: random.Random, frac_lo: float, frac_hi: float) -> np.ndarray:
    target_lo = max(1, int(np.ceil(length * frac_lo)))
    target_hi = max(target_lo, int(np.floor(length * frac_hi)))
    target = rng.randint(target_lo, min(length, target_hi))
    mask = np.zeros(length, dtype=bool)
    attempts = 0
    while int(mask.sum()) < target and attempts < 100:
        seg = rng.randint(1, max(1, min(length, 1 + length // 3)))
        start = rng.randint(0, length - seg)
        mask[start : start + seg] = True
        attempts += 1
    return mask


def _band_assign(length: int, n_colors: int, rng: random.Random, frac_lo: float, frac_hi: float) -> np.ndarray:
    target_lo = max(1, int(np.ceil(length * frac_lo)))
    target_hi = max(target_lo, int(np.floor(length * frac_hi)))
    target = rng.randint(target_lo, min(length, target_hi))
    arr = np.full(length, -1, dtype=int)
    filled = 0
    attempts = 0
    color_cycle = list(range(n_colors))
    rng.shuffle(color_cycle)
    cycle_i = 0
    while filled < target and attempts < 120:
        seg = rng.randint(1, max(1, min(length, 1 + length // 3)))
        start = rng.randint(0, length - seg)
        if np.any(arr[start : start + seg] != -1):
            attempts += 1
            continue
        arr[start : start + seg] = color_cycle[cycle_i % n_colors]
        filled += seg
        cycle_i += 1
        attempts += 1
    if np.all(arr == -1):
        arr[rng.randrange(length)] = 0
    for ci in range(n_colors):
        if ci not in arr:
            arr[rng.randrange(length)] = ci
    return arr


def _paint_vertical(out: np.ndarray, row_mask: np.ndarray, col_assign: np.ndarray, colors: list[int]) -> None:
    H, W = out.shape
    for r in range(H):
        if not bool(row_mask[r]):
            continue
        for c_left, idx in enumerate(col_assign):
            if int(idx) == -1:
                continue
            color = int(colors[int(idx)])
            c_right = W - 1 - c_left
            out[r, c_left] = color
            out[r, c_right] = color


def _paint_point_row(out: np.ndarray, r: int, col_assign: np.ndarray, colors: list[int]) -> None:
    H, W = out.shape
    for c, idx in enumerate(col_assign):
        if int(idx) == -1:
            continue
        color = int(colors[int(idx)])
        out[r, c] = color
        out[H - 1 - r, W - 1 - c] = color


def _build_vertical_pattern(H: int, W: int, rect_col: int, colors: list[int], rng: random.Random) -> np.ndarray:
    out = np.full((H, W), int(rect_col), dtype=int)
    half_w = (W + 1) // 2
    row_mask1 = _band_mask(H, rng, 0.55, 0.78)
    row_mask2 = _band_mask(H, rng, 0.18, 0.32) & (~row_mask1)
    col_assign1 = _band_assign(half_w, len(colors), rng, 0.45, 0.68)
    col_assign2 = np.roll(_band_assign(half_w, len(colors), rng, 0.25, 0.45), 1 if half_w > 1 else 0)
    _paint_vertical(out, row_mask1, col_assign1, colors)
    _paint_vertical(out, row_mask2, col_assign2, colors)
    return out


def _build_horizontal_pattern(H: int, W: int, rect_col: int, colors: list[int], rng: random.Random) -> np.ndarray:
    return _build_vertical_pattern(W, H, rect_col, colors, rng).T


def _build_point_pattern(H: int, W: int, rect_col: int, colors: list[int], rng: random.Random) -> np.ndarray:
    out = np.full((H, W), int(rect_col), dtype=int)
    top_h = (H + 1) // 2
    row_mask1 = _band_mask(top_h, rng, 0.55, 0.78)
    row_mask2 = _band_mask(top_h, rng, 0.18, 0.32) & (~row_mask1)
    base1 = _band_assign(W, len(colors), rng, 0.42, 0.64)
    base2 = np.roll(_band_assign(W, len(colors), rng, 0.24, 0.42), rng.randrange(max(1, W)))
    for r in range(top_h):
        if bool(row_mask1[r]):
            _paint_point_row(out, r, np.roll(base1, r % max(1, W)), colors)
        if bool(row_mask2[r]):
            _paint_point_row(out, r, np.roll(base2, (2 * r + 1) % max(1, W)), colors)
    return out


def _choose_period(n: int, rng: random.Random) -> int | None:
    opts = [p for p in range(2, min(4, n) + 1) if n % p == 0 and n // p >= 2]
    return None if not opts else rng.choice(opts)


def _build_translation_x_pattern(H: int, W: int, rect_col: int, colors: list[int], rng: random.Random) -> np.ndarray | None:
    period = _choose_period(W, rng)
    if period is None:
        return None
    out = np.full((H, W), int(rect_col), dtype=int)
    row_mask1 = _band_mask(H, rng, 0.55, 0.78)
    row_mask2 = _band_mask(H, rng, 0.18, 0.32) & (~row_mask1)
    base1 = _band_assign(period, len(colors), rng, 0.45, 0.80)
    base2 = np.roll(_band_assign(period, len(colors), rng, 0.25, 0.55), 1 if period > 1 else 0)
    for r in range(H):
        assign = None
        if bool(row_mask1[r]):
            assign = np.roll(base1, r % period)
        elif bool(row_mask2[r]):
            assign = np.roll(base2, (2 * r + 1) % period)
        if assign is None:
            continue
        for c in range(W):
            idx = int(assign[c % period])
            if idx != -1:
                out[r, c] = int(colors[idx])
    return out


def _build_translation_y_pattern(H: int, W: int, rect_col: int, colors: list[int], rng: random.Random) -> np.ndarray | None:
    arr = _build_translation_x_pattern(W, H, rect_col, colors, rng)
    return None if arr is None else arr.T


def _build_symmetric_interior(
    iH: int,
    iW: int,
    rect_col: int,
    colors: list[int],
    rng: random.Random,
) -> tuple[np.ndarray, str, list[str]] | None:
    builders = [
        ("vertical", _build_vertical_pattern),
        ("horizontal", _build_horizontal_pattern),
        ("point", _build_point_pattern),
        ("translation_x", _build_translation_x_pattern),
        ("translation_y", _build_translation_y_pattern),
    ]
    rng.shuffle(builders)
    for want, builder in builders:
        for _ in range(10):
            interior = builder(iH, iW, rect_col, colors, rng)
            if interior is None:
                continue
            fill = float(np.mean(interior != int(rect_col)))
            used = set(map(int, np.unique(interior[interior != int(rect_col)])))
            if used != set(map(int, colors)):
                continue
            if fill < 0.38 or fill > 0.60:
                continue
            kinds = _symmetry_types(interior)
            if want not in kinds and not any(k.startswith(want) for k in kinds):
                continue
            counts = Counter(map(int, interior[interior != int(rect_col)].flatten()))
            if min(counts.values()) < 3:
                continue
            return interior, want, kinds
    return None


def _rand_splits(total: int, rng: random.Random, min_s: int = 2, max_s: int = 7) -> list[int] | None:
    eff_max = min(max_s, max(min_s, rng.randint(min_s, max_s)))
    segs: list[int] = []
    rem = total
    while rem > 0:
        if rem <= eff_max:
            if rem < min_s:
                return None
            segs.append(rem)
            rem = 0
        else:
            hi = min(eff_max, rem - min_s)
            if min_s > hi:
                return None
            s = rng.randint(min_s, hi)
            segs.append(s)
            rem -= s
    return segs


def _partition_rect(rect: np.ndarray, rect_col: int, rng: random.Random) -> list[tuple[int, int, int, int]] | None:
    RH, RW = rect.shape
    for _ in range(50):
        row_segs = _rand_splits(RH, rng)
        col_segs = _rand_splits(RW, rng)
        if row_segs is None or col_segs is None:
            continue

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

        nr, nc = len(row_segs), len(col_segs)
        n_cells = nr * nc
        if n_cells < 4:
            continue
        cell_id = [[i * nc + j for j in range(nc)] for i in range(nr)]
        target_pieces = rng.randint(4, min(9, n_cells))

        def get_piece_bbox(pid: int) -> tuple[int, int, int, int] | None:
            cells = [(i, j) for i in range(nr) for j in range(nc) if cell_id[i][j] == pid]
            if not cells:
                return None
            ris = [c[0] for c in cells]
            cis = [c[1] for c in cells]
            return min(ris), min(cis), max(ris), max(cis)

        def pixel_size(r0i: int, c0i: int, r1i: int, c1i: int) -> tuple[int, int]:
            h = sum(row_segs[i] for i in range(r0i, r1i + 1))
            w = sum(col_segs[j] for j in range(c0i, c1i + 1))
            return h, w

        merges: list[tuple[str, int, int]] = []
        for i in range(nr):
            for j in range(nc):
                if j + 1 < nc:
                    merges.append(("h", i, j))
                if i + 1 < nr:
                    merges.append(("v", i, j))
        rng.shuffle(merges)

        def count_pieces() -> int:
            return len(set(cell_id[i][j] for i in range(nr) for j in range(nc)))

        for direction, mi, mj in merges:
            if count_pieces() <= target_pieces:
                break
            pid1 = cell_id[mi][mj]
            pid2 = cell_id[mi][mj + 1] if direction == "h" else cell_id[mi + 1][mj]
            if pid1 == pid2:
                continue
            bb1 = get_piece_bbox(pid1)
            bb2 = get_piece_bbox(pid2)
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
                for jj in range(nc):
                    if cell_id[ii][jj] == pid2:
                        cell_id[ii][jj] = pid1

        pieces: list[tuple[int, int, int, int]] = []
        seen: set[int] = set()
        for i in range(nr):
            for j in range(nc):
                pid = cell_id[i][j]
                if pid in seen:
                    continue
                seen.add(pid)
                bb = get_piece_bbox(pid)
                if bb is None:
                    continue
                r0i, c0i, r1i, c1i = bb
                pr = row_starts[r0i]
                pc = col_starts[c0i]
                ph = sum(row_segs[ii] for ii in range(r0i, r1i + 1))
                pw = sum(col_segs[jj] for jj in range(c0i, c1i + 1))
                pieces.append((pr, pc, ph, pw))

        if len(pieces) < 4 or len(pieces) > 9:
            continue

        valid = True
        for pr, pc, ph, pw in pieces:
            piece = rect[pr : pr + ph, pc : pc + pw]
            fill = float(np.mean(piece != int(rect_col)))
            if ph < 2 or pw < 2 or ph > 7 or pw > 7:
                valid = False
                break
            if fill < 0.10 or fill > 0.60:
                valid = False
                break
            if not any(np.any(edge != int(rect_col)) for edge in (piece[0, :], piece[-1, :], piece[:, 0], piece[:, -1])):
                valid = False
                break
            if pr == 0 and not np.all(piece[0, :] == int(rect_col)):
                valid = False
                break
            if pr + ph == RH and not np.all(piece[-1, :] == int(rect_col)):
                valid = False
                break
            if pc == 0 and not np.all(piece[:, 0] == int(rect_col)):
                valid = False
                break
            if pc + pw == RW and not np.all(piece[:, -1] == int(rect_col)):
                valid = False
                break
        if valid:
            return pieces
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    GH = rng.randint(10, 26)
    GW = rng.randint(10, 26)

    palette = list(range(10))
    rng.shuffle(palette)
    bg = palette[0]
    rect_col = palette[1]
    marker_col = palette[2]
    n_interior = rng.randint(2, 3)
    interior_cols = palette[3 : 3 + n_interior]

    corner = rng.choice(["TL", "TR", "BL", "BR"])

    min_side, max_side = 7, min(20, min(GH, GW) - 3)
    if max_side < min_side:
        return None
    RH = rng.randint(min_side, max_side)
    RW = rng.randint(min_side, max_side)
    if rng.random() < 0.5 and max_side >= 10:
        RH = rng.randint(max(8, min_side), max_side)
        RW = rng.randint(max(8, min_side), max_side)

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
    mr = rng.randint(mr_min, mr_max)
    mc = rng.randint(mc_min, mc_max)

    rect = np.full((RH, RW), int(rect_col), dtype=int)
    interior_info = _build_symmetric_interior(RH - 2, RW - 2, rect_col, interior_cols, rng)
    if interior_info is None:
        return None
    interior, sym_kind, sym_kinds = interior_info
    rect[1:-1, 1:-1] = interior

    pieces = _partition_rect(rect, rect_col, rng)
    if pieces is None:
        return None

    if corner == "TL":
        anchor_r, anchor_c = 0, 0
    elif corner == "TR":
        anchor_r, anchor_c = 0, RW - 1
    elif corner == "BL":
        anchor_r, anchor_c = RH - 1, 0
    else:
        anchor_r, anchor_c = RH - 1, RW - 1

    anchor_idx = None
    for i, (pr, pc, ph, pw) in enumerate(pieces):
        if pr <= anchor_r < pr + ph and pc <= anchor_c < pc + pw:
            anchor_idx = i
            break
    if anchor_idx is None:
        return None
    apr, apc, aph, apw = pieces[anchor_idx]
    if corner == "TL" and not (apr == 0 and apc == 0):
        return None
    if corner == "TR" and not (apr == 0 and apc + apw == RW):
        return None
    if corner == "BL" and not (apr + aph == RH and apc == 0):
        return None
    if corner == "BR" and not (apr + aph == RH and apc + apw == RW):
        return None

    marker_cells = _marker_cells(corner, mr, mc)
    for cr, cc in marker_cells:
        if cr < 0 or cr >= GH or cc < 0 or cc >= GW:
            return None

    piece_arrays = [rect[pr : pr + ph, pc : pc + pw].copy() for pr, pc, ph, pw in pieces]

    canvas = None
    for _ in range(24):
        trial = np.full((GH, GW), int(bg), dtype=int)
        for cr, cc in marker_cells:
            trial[cr, cc] = int(marker_col)
        order = list(range(len(piece_arrays)))
        rng.shuffle(order)
        ok = True
        for idx in order:
            pa = piece_arrays[idx]
            try:
                r0, c0 = random_free_location(trial, pa, transparent=bg, padding=1)
                paste(trial, pa, r0, c0)
            except ValueError:
                ok = False
                break
        if ok:
            canvas = trial
            break
    if canvas is None:
        return None

    tr, tc = _top_left_from_corner(mr, mc, RH, RW, corner)
    recolored = _recolor_assembled(rect, rect_col, marker_col)
    expected_out = np.full((GH, GW), int(bg), dtype=int)
    expected_out[tr : tr + RH, tc : tc + RW] = recolored

    try:
        actual_out = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual_out, expected_out):
        return None

    changed_pixels = int(np.sum(rect != recolored))
    if changed_pixels <= 0:
        return None

    global _LAST_GENERATION_INFO
    _LAST_GENERATION_INFO = {
        "sym_kind": sym_kind,
        "symmetry_types": tuple(sym_kinds),
        "pre_recolor_colors": len(interior_cols),
        "changed_pixels": changed_pixels,
        "rect_shape": (RH, RW),
        "piece_count": len(pieces),
        "fill_ratio": float(np.mean(interior != int(rect_col))),
    }
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
