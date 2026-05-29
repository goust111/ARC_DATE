# Combo variant of 446ef5d2 (index 19)
# combo_spec: D12/2+D17/2
# Name: 对称内部图案配合加强颜色分离

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
# ---------------------------------------------------------------------------
#
# source_task:
#   - Base task 446ef5d2: assemble scattered rectangular pieces into one
#     rectangle anchored at the missing cell of a 3-cell L marker.
#   - D12/2 contributes "interior is globally symmetric".
#   - D17/2 contributes "different interior colors are separated by a 2-cell
#     `rect_col` moat", making the symmetric motif sparser and cleaner.
#
# a) concretization_and_trace:
#   - Original interior:
#       pieces come from one rectangle whose inside contains irregular blobs;
#       local solving is only about edge compatibility with `rect_col` as a
#       wildcard. Example internal boundary:
#         left edge  [rect, 3, rect, 7]
#         right edge [rect, 3, rect, 7]
#       This is enough to place two pieces together.
#   - Combo interior:
#       the full rectangle still uses the same tiling rule, but the inside is a
#       symmetric arrangement of sparse 2x2 colored modules. Different colors
#       never get closer than Chebyshev distance 3, which corresponds to at
#       least a 2-cell-wide `rect_col` buffer.
#   - Concrete small trace:
#       imagine a 9x9 assembled rectangle with a 7x7 interior. The generator
#       places 2x2 red modules at macro-cells (top-left and top-right) and 2x2
#       yellow modules at the mirrored lower pair. The 2 empty columns/rows
#       between macro-cells make the color groups visibly separated. When pieces
#       are cut and scattered, local edge matching still reconstructs the same
#       rectangle, but among multiple tilings the intended one is the one whose
#       whole interior is both symmetric and strongly color-separated.
#   - Difference:
#       symmetry is no longer a mild stylistic bias. Because the colors are
#       sparse and buffered, the symmetry becomes the dominant global signal.
#
# b) complexity_calibration:
#   - To avoid an over-simple "four isolated pixels" pattern, each motif unit is
#     a 2x2 block rather than a single pixel.
#   - To keep the 2-cell moat physically realizable while still leaving enough
#     colored content for piece matching, the generator works on a macro-grid of
#     2x2 blocks spaced every 4 cells.
#   - The symmetry family set is kept to three human-legible options:
#       vertical reflection, horizontal reflection, and 180-degree point
#       symmetry. This is enough variety without making the rule impossible to
#       infer from 2-4 ARC examples.
#   - Interior color count is restricted to 2-3, not 1. This is deliberate:
#     D17/2 should be visibly active in generated examples instead of becoming a
#     vacuous constraint.
#
# c) implicit_assumptions_review_in_main:
#   - Marker detection assumes one non-bg color appears exactly 3 times in a 2x2
#     L shape. Combo does not change that.
#   - Rectangle extraction assumes each scattered object is a solid rectangle of
#     occupied cells. Generator still scatters full rectangles only.
#   - `rect_col` inference assumes that color remains globally most common across
#     all piece pixels. The sparse 2x2-module generator keeps interior fill well
#     below the border mass, preserving this.
#   - Boundary checks assume the full outer border is entirely `rect_col`. The
#     generator keeps the 1-cell border untouched.
#   - Neighbor matching assumes only shared-edge non-`rect_col` pixels matter.
#     That still holds. The new signal is global, so solver changes belong in
#     candidate ranking rather than local compatibility.
#   - The original solver effectively accepted the first full tiling. That is
#     the line most tied to the old irregular interiors. In the combo version we
#     enumerate several complete tilings per dimension and rank them by a joint
#     score: zero separation violations first, then symmetry strength.
#
# d) placement_legality:
#   - Marker placement still guarantees the final rectangle fits in-bounds.
#   - Pieces are scattered with >=1 background cell padding and no overlap.
#   - The macro-grid motif builder paints only within the inner rectangle, and
#     each 2x2 block is placed using precomputed valid positions, so no module
#     crosses the border or another module.
#
# e) visual_observability:
#   - The combo is visible directly in pixels:
#       * mirrored or point-symmetric 2x2 color modules
#       * at least two rows/cols of `rect_col` between different colors
#   - No hidden ordering by numeric color ids is used.
#   - Because 2-3 colors recur in multiple symmetric locations, a human can
#     notice both "same pattern repeats symmetrically" and "different colors are
#     kept far apart" from a few examples.
#
# f) visual_tidiness:
#   - There are still only three semantic layers: background, marker, and
#     scattered rectangular pieces.
#   - The interior is cleaner than the base task because the colored content is
#     blocky, sparse, and aligned to a macro-grid; the 2-cell moats prevent
#     clutter from adjacent colors.
#
# g) color_inference:
#   - Colors do not undergo remapping. They participate only by repeated
#     placement in symmetry orbits.
#   - Restricting to 2-3 interior colors keeps the color logic inferable.
#
# h) new_solver_assumptions_and_generator_guarantees:
#   - Solver assumes the intended rectangle scores well on both properties:
#     detectable symmetry and no different-color pair closer than distance 3.
#   - Generator guarantees exactly that by constructing the interior from
#     symmetric macro-cell orbits on a 4-step lattice and validating with the
#     solver before returning the input.
#
# i) why this does not collapse back to the base task:
#   - The base generator grows arbitrary blobs.
#   - This generator never uses blob growth. It builds the interior only from
#     symmetric 2x2 modules with enforced 2-cell moats.
#   - Therefore both control variables in the combo are active and measurable.


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
    return kinds


def _separation_stats(arr: np.ndarray, rect_col: int) -> tuple[int, int, int]:
    coords = np.argwhere(arr != int(rect_col))
    if coords.size == 0:
        return 99, 0, 0
    vals = [int(arr[r, c]) for r, c in coords]
    used_colors = len(set(vals))
    if used_colors <= 1:
        return 99, 0, used_colors

    min_sep = 99
    violations = 0
    n = len(coords)
    for i in range(n):
        r1, c1 = map(int, coords[i])
        v1 = int(vals[i])
        for j in range(i + 1, n):
            if v1 == int(vals[j]):
                continue
            r2, c2 = map(int, coords[j])
            dist = max(abs(r1 - r2), abs(c1 - c2))
            if dist < min_sep:
                min_sep = dist
            if dist < 3:
                violations += 1
    if min_sep == 99:
        min_sep = 99
    return int(min_sep), int(violations), int(used_colors)


def _combo_score(rect: np.ndarray, rect_col: int) -> tuple[int, int, int, int, int]:
    if rect.shape[0] <= 2 or rect.shape[1] <= 2:
        return (0, 0, 0, 0, 0)
    interior = rect[1:-1, 1:-1]
    kinds = _symmetry_types(interior)
    min_sep, violations, used_colors = _separation_stats(interior, rect_col)
    non_rect = int(np.sum(interior != int(rect_col)))
    return (int(violations == 0), len(kinds), min(min_sep, 99), used_colors, non_rect)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the scattered rectangle pieces and prefer candidates whose interior
    is both symmetric and strongly color-separated.
    """
    g = np.array(grid, dtype=int)
    GH, GW = g.shape
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])

    mark, missing, mb = _find_marker(g, bg)
    r0, c0, r1, c1 = mb
    if missing == (r0, c0):
        corner = "BR"
    elif missing == (r0, c1):
        corner = "BL"
    elif missing == (r1, c0):
        corner = "TR"
    else:
        corner = "TL"

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

    total_area = int(sum(int(h) * int(w) for _r, _c, h, w, _a in rects))
    allpix: list[int] = []
    for _r, _c, _h, _w, a in rects:
        allpix.extend(map(int, a.flatten()))
    rect_col = int(Counter(allpix).most_common(1)[0][0])

    mr, mc = map(int, missing)
    anchor_idx = None
    for i, (rr, cc, h, w, _a) in enumerate(rects):
        if rr <= mr < rr + h and cc <= mc < cc + w:
            anchor_idx = i
            break

    pieces = [a for _r, _c, _h, _w, a in rects]
    ph = [int(p.shape[0]) for p in pieces]
    pw = [int(p.shape[1]) for p in pieces]
    pnon = [int(np.sum(p != int(rect_col))) for p in pieces]

    dims: list[tuple[int, int]] = []
    for hh in range(1, int(total_area**0.5) + 1):
        if total_area % hh == 0:
            dims.append((hh, total_area // hh))
            if hh * hh != total_area:
                dims.append((total_area // hh, hh))

    def fits(H: int, W: int) -> bool:
        if corner == "TL":
            tr, tc = mr, mc
        elif corner == "TR":
            tr, tc = mr, mc - W + 1
        elif corner == "BL":
            tr, tc = mr - H + 1, mc
        else:
            tr, tc = mr - H + 1, mc - W + 1
        return 0 <= tr and 0 <= tc and (tr + H) <= GH and (tc + W) <= GW

    dims = [(int(H), int(W)) for H, W in dims if fits(int(H), int(W))]
    dims = sorted(dims, key=lambda t: (-t[0] * t[1], -t[0], -t[1]))

    def compat(v1: np.ndarray, v2: np.ndarray) -> bool:
        m = (v1 != int(rect_col)) & (v2 != int(rect_col))
        return bool((v1[m] == v2[m]).all())

    def assemble(H: int, W: int) -> tuple[np.ndarray | None, tuple[int, int, int, int, int]]:
        occ2 = np.full((H, W), -1, dtype=int)
        usedp = [False] * len(pieces)
        best: np.ndarray | None = None
        best_score = (-1, -1, -1, -1, -1)
        solutions_seen = 0
        max_solutions = 16

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
            rr, cc, h0, w0, _ = rects[anchor_idx]
            corners = {
                (rr, cc): "TL",
                (rr, cc + w0 - 1): "TR",
                (rr + h0 - 1, cc): "BL",
                (rr + h0 - 1, cc + w0 - 1): "BR",
            }
            if (mr, mc) in corners and corners[(mr, mc)] == corner:
                ah, aw = ph[anchor_idx], pw[anchor_idx]
                if corner == "TL":
                    ar0, ac0 = 0, 0
                elif corner == "TR":
                    ar0, ac0 = 0, W - aw
                elif corner == "BL":
                    ar0, ac0 = H - ah, 0
                else:
                    ar0, ac0 = H - ah, W - aw
                if can_place(anchor_idx, ar0, ac0):
                    place(anchor_idx, ar0, ac0)

        order = sorted(range(len(pieces)), key=lambda i: (-pnon[i], -(ph[i] * pw[i]), -ph[i], -pw[i], i))

        def next_empty() -> tuple[int, int] | None:
            idx = np.argwhere(occ2 == -1)
            if idx.size == 0:
                return None
            return tuple(map(int, idx[0]))

        def good_enough(score: tuple[int, int, int, int, int]) -> bool:
            return score[0] == 1 and score[1] >= 1 and score[2] >= 3 and score[3] >= 2

        def bt() -> bool:
            nonlocal best, best_score, solutions_seen
            nxt = next_empty()
            if nxt is None:
                solutions_seen += 1
                score = _combo_score(occ2, rect_col)
                if score > best_score:
                    best_score = score
                    best = occ2.copy()
                return solutions_seen >= max_solutions or good_enough(best_score)

            r, c = nxt
            for i in order:
                if usedp[i]:
                    continue
                if can_place(i, r, c):
                    place(i, r, c)
                    should_stop = bt()
                    unplace(i, r, c)
                    if should_stop:
                        return True
            return False

        bt()
        return best, best_score

    best_ass: np.ndarray | None = None
    best_dim: tuple[int, int] | None = None
    best_score = (-1, -1, -1, -1, -1)
    for H, W in dims:
        ass, score = assemble(H, W)
        if ass is None:
            continue
        if score > best_score:
            best_score = score
            best_ass = ass
            best_dim = (H, W)

    assert not (best_ass is None or best_dim is None)

    H, W = best_dim
    if corner == "TL":
        tr, tc = mr, mc
    elif corner == "TR":
        tr, tc = mr, mc - W + 1
    elif corner == "BL":
        tr, tc = mr - H + 1, mc
    else:
        tr, tc = mr - H + 1, mc - W + 1

    out = np.full((GH, GW), int(bg), dtype=int)
    out[tr : tr + H, tc : tc + W] = best_ass
    return out


# ---------------------------------------------------------------------------
# Generator helpers
# ---------------------------------------------------------------------------


def _macro_positions(length: int, rng: random.Random) -> list[int]:
    count = max(1, (length + 2) // 4)
    used = 4 * count - 2
    slack = max(0, length - used)
    offset = rng.randint(0, slack)
    return [offset + 4 * k for k in range(count)]


def _orbit_map(rows: int, cols: int, sym_kind: str) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    orbits: list[list[tuple[int, int]]] = []
    for r in range(rows):
        for c in range(cols):
            if (r, c) in seen:
                continue
            if sym_kind == "vertical":
                orbit = {(r, c), (r, cols - 1 - c)}
            elif sym_kind == "horizontal":
                orbit = {(r, c), (rows - 1 - r, c)}
            else:
                orbit = {(r, c), (rows - 1 - r, cols - 1 - c)}
            cell_list = sorted((int(rr), int(cc)) for rr, cc in orbit)
            seen.update(cell_list)
            orbits.append(cell_list)
    return orbits


def _build_symmetric_sparse_interior(
    iH: int, iW: int, rect_col: int, colors: list[int], rng: random.Random
) -> tuple[np.ndarray, str] | None:
    macro_rows = _macro_positions(iH, rng)
    macro_cols = _macro_positions(iW, rng)
    MR, MC = len(macro_rows), len(macro_cols)

    choices: list[str] = []
    if MC >= 2:
        choices.append("vertical")
    if MR >= 2:
        choices.append("horizontal")
    if MR >= 2 and MC >= 2:
        choices.append("point")
    if not choices:
        return None

    area = iH * iW
    block_lo = max(len(colors) + 1, int(np.ceil(area * 0.18 / 4.0)))
    block_hi = max(block_lo, int(np.floor(area * 0.34 / 4.0)))
    block_hi = min(block_hi, MR * MC)

    for sym_kind in rng.sample(choices, len(choices)):
        orbits = _orbit_map(MR, MC, sym_kind)
        for _ in range(30):
            rng.shuffle(orbits)
            target = rng.randint(block_lo, block_hi)
            active: list[list[tuple[int, int]]] = []
            filled = 0
            for orbit in orbits:
                if filled + len(orbit) > target:
                    continue
                active.append(orbit)
                filled += len(orbit)
                if filled >= target:
                    break
            if filled < max(len(colors), block_lo):
                continue
            if len(active) < len(colors):
                continue

            grid = np.full((MR, MC), -1, dtype=int)
            color_cycle = list(range(len(colors)))
            rng.shuffle(color_cycle)
            for orbit_idx, orbit in enumerate(active):
                idx = color_cycle[orbit_idx % len(color_cycle)]
                for rr, cc in orbit:
                    grid[rr, cc] = idx

            used_idxs = set(map(int, grid[grid != -1].flatten()))
            if used_idxs != set(range(len(colors))):
                continue

            interior = np.full((iH, iW), int(rect_col), dtype=int)
            for rr in range(MR):
                for cc in range(MC):
                    idx = int(grid[rr, cc])
                    if idx == -1:
                        continue
                    r0 = macro_rows[rr]
                    c0 = macro_cols[cc]
                    interior[r0 : r0 + 2, c0 : c0 + 2] = int(colors[idx])

            fill = float(np.mean(interior != int(rect_col)))
            kinds = _symmetry_types(interior)
            min_sep, violations, used_colors = _separation_stats(interior, rect_col)
            if sym_kind not in kinds:
                continue
            if violations != 0 or min_sep < 3:
                continue
            if used_colors != len(colors):
                continue
            if fill < 0.18 or fill > 0.34:
                continue
            return interior, sym_kind
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


def _rand_k_splits(total: int, k: int, rng: random.Random, min_s: int = 2, max_s: int = 7) -> list[int] | None:
    if k * min_s > total or k * max_s < total:
        return None
    segs: list[int] = []
    rem = total
    for i in range(k):
        left = k - i - 1
        lo = max(min_s, rem - left * max_s)
        hi = min(max_s, rem - left * min_s)
        if lo > hi:
            return None
        s = rng.randint(lo, hi)
        segs.append(s)
        rem -= s
    return segs if rem == 0 else None


def _partition_rect(rect: np.ndarray, rect_col: int, rng: random.Random) -> list[tuple[int, int, int, int]] | None:
    RH, RW = rect.shape
    for _ in range(80):
        candidates = [(nr, nc) for nr in range(2, 5) for nc in range(2, 5) if 4 <= nr * nc <= 12]
        nr, nc = rng.choice(candidates)
        row_segs = _rand_k_splits(RH, nr, rng)
        col_segs = _rand_k_splits(RW, nc, rng)
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

        pieces: list[tuple[int, int, int, int]] = []
        valid = True
        edge_signal_pieces = 0
        for i, pr in enumerate(row_starts):
            for j, pc in enumerate(col_starts):
                ph = row_segs[i]
                pw = col_segs[j]
                piece = rect[pr : pr + ph, pc : pc + pw]
                fill = float(np.mean(piece != int(rect_col)))
                if fill < 0.06 or fill > 0.50:
                    valid = False
                    break
                if not np.any(piece != int(rect_col)):
                    valid = False
                    break
                has_non_rect_edge = any(
                    np.any(edge != int(rect_col)) for edge in (piece[0, :], piece[-1, :], piece[:, 0], piece[:, -1])
                )
                if has_non_rect_edge:
                    edge_signal_pieces += 1
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
                pieces.append((pr, pc, ph, pw))
            if not valid:
                break
        if valid and edge_signal_pieces >= max(2, len(pieces) // 2) and 4 <= len(pieces) <= 12:
            return pieces
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    GH = rng.randint(12, 26)
    GW = rng.randint(12, 26)

    palette = list(range(10))
    rng.shuffle(palette)
    bg = palette[0]
    rect_col = palette[1]
    marker_col = palette[2]
    n_interior = rng.randint(2, 3)
    interior_cols = palette[3 : 3 + n_interior]

    corner = rng.choice(["TL", "TR", "BL", "BR"])

    min_side, max_side = 9, min(22, min(GH, GW) - 3)
    if max_side < min_side:
        return None
    RH = rng.randint(min_side, max_side)
    RW = rng.randint(min_side, max_side)
    if rng.random() < 0.5 and max_side >= 11:
        RH = rng.randint(max(10, min_side), max_side)
        RW = rng.randint(max(10, min_side), max_side)

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
    interior_info = _build_symmetric_sparse_interior(RH - 2, RW - 2, rect_col, interior_cols, rng)
    if interior_info is None:
        return None
    interior, _sym_kind = interior_info
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

    if corner == "TL":
        marker_cells = [(mr - 1, mc - 1), (mr - 1, mc), (mr, mc - 1)]
    elif corner == "TR":
        marker_cells = [(mr - 1, mc), (mr - 1, mc + 1), (mr, mc + 1)]
    elif corner == "BL":
        marker_cells = [(mr, mc - 1), (mr + 1, mc - 1), (mr + 1, mc)]
    else:
        marker_cells = [(mr, mc + 1), (mr + 1, mc), (mr + 1, mc + 1)]
    for cr, cc in marker_cells:
        if cr < 0 or cr >= GH or cc < 0 or cc >= GW:
            return None

    piece_arrays = [rect[pr : pr + ph, pc : pc + pw].copy() for pr, pc, ph, pw in pieces]

    canvas = None
    for _ in range(30):
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

    if corner == "TL":
        tr, tc = mr, mc
    elif corner == "TR":
        tr, tc = mr, mc - RW + 1
    elif corner == "BL":
        tr, tc = mr - RH + 1, mc
    else:
        tr, tc = mr - RH + 1, mc - RW + 1

    expected_out = np.full((GH, GW), int(bg), dtype=int)
    expected_out[tr : tr + RH, tc : tc + RW] = rect

    try:
        actual_out = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual_out, expected_out):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
