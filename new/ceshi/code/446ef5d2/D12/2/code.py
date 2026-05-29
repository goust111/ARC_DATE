# Variant of 446ef5d2
# Dimension: D12 — interior_cols 的数量与分布 (meta)
# Change: 1-3 种 interior_cols，在矩形内部形成不规则连通色块，不同颜色之间 8 连通分离... → interior_cols 形成对称图案

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the D12 variant
# ---------------------------------------------------------------------------
#
# source_task:
#   None. No external reference solver needs to be consulted.
#
# a) concretization_and_trace:
#   - Original task: the rectangle interior is produced by 1-3 irregular blobs, so
#     solver success depends only on rectangular tiling plus edge-color matching.
#   - Variant: the rectangle interior still uses 1-3 colors and still sits inside a
#     1-cell `rect_col` border, but the colored content is now globally symmetric.
#     The symmetry family is chosen per instance:
#       * vertical reflection
#       * horizontal reflection
#       * 180-degree point symmetry
#       * horizontal-periodic or vertical-periodic translation symmetry
#   - Concrete pixel behavior:
#       * input still shows scattered rectangular pieces and one 3-cell L marker.
#       * output is still one assembled rectangle anchored at the marker corner.
#       * the visual difference is inside the rectangle: instead of amorphous blobs,
#         rows/columns now repeat in mirrored or periodic bands.
#   - Local trace on one edge-matching step:
#       * original logic: piece A right edge `[rect, 2, rect]` can match piece B left
#         edge `[rect, 2, rect]`, and solver stops at the first full tiling.
#       * variant logic: that same edge match remains valid, but among complete tilings
#         the intended one also yields a symmetric whole interior. Solver therefore keeps
#         the old compatibility rule and adds a symmetry score to rank successful tilings.
#       * difference: symmetry does not replace edge matching; it acts as a global
#         plausibility signal aligned with the new D12 behavior.
#
# b) complexity_calibration:
#   - Purely random symmetric single pixels would be too weak visually and could leave
#     some pieces nearly empty. I therefore generate banded / tiled motifs with 38-58%
#     interior fill, so the symmetry is obvious and most pieces inherit non-rect_col
#     structure.
#   - Rectangle side minimum is raised from 5 to 7. This is a targeted adaptation:
#     small 3x3 interiors make symmetry visually trivial or accidental.
#   - Symmetry family count is kept small (4 effective families) so 2-4 ARC examples
#     can support the rule without exploding the hypothesis space.
#
# c) implicit_assumptions_review_in_main:
#   - Marker detection depends only on "a non-bg color appearing exactly 3 times in a
#     2x2 L". D12 does not change this; safe.
#   - Rectangle decomposition assumes every scattered object is a solid rectangle of
#     occupied pixels. Generator still scatters full rectangles; safe.
#   - `rect_col` inference assumes that color is globally most common across all piece
#     pixels. The new generator therefore caps interior fill so one interior color never
#     dominates the border/wildcard color.
#   - Boundary checks assume the assembled rectangle border is entirely `rect_col`.
#     Generator preserves the 1-cell border exactly; safe.
#   - Neighbor compatibility uses `rect_col` as wildcard and does not inspect interior
#     connectivity. Safe: symmetry changes only the global layout, not local edge rules.
#   - Original solver returned the first valid assembly. In the variant this is the only
#     line that is "quietly tied" to the old blob distribution: multiple edge-compatible
#     tilings could exist, but only the intended one should look symmetric. This is the
#     relevant solver change, so candidate assemblies are now ranked by symmetry score.
#
# d) placement_legality:
#   - Marker placement still guarantees the final rectangle fits entirely in-grid.
#   - Pieces are still scattered with at least 1 background cell padding and no overlap.
#   - Partitioning is retried until all pieces satisfy the original rectangular-piece
#     constraints and the new symmetric rectangle still solves back to the expected output.
#
# e) visual_observability:
#   - The new rule is visible from raw pixels: mirrored columns, mirrored rows, rotated
#     180-degree row patterns, or repeated periodic columns/rows.
#   - No hidden ordering by numeric color ids is used. Colors are assigned structurally
#     by band placement, not by magnitude.
#
# f) visual_tidiness:
#   - Only three visual layers remain: background, marker, scattered rectangles.
#   - Symmetric motifs are blocky and aligned to the grid, which is much cleaner than
#     allowing many tiny disconnected random specks.
#
# g) color_inference:
#   - Color count remains 1-3 interior colors.
#   - Colors do not undergo arbitrary remapping. They simply participate in mirrored or
#     periodic placement, which is observable from position and repetition.
#
# h) new_solver_assumptions_and_generator_guarantees:
#   - Solver now prefers assemblies with detectable symmetry in the interior.
#   - Generator guarantees each sample has at least one detected symmetry type and uses
#     the same exact assembled rectangle when validating `main(input) == expected_output`.
#
# i) why the variant does not collapse back to the original task:
#   - The original generator made irregular connected blobs by random growth.
#   - This generator never uses blob growth. It constructs interiors from mirrored or
#     periodic band assignments and explicitly measures the resulting symmetry family.
#   - Therefore the key control variable for D12 is active and testable.


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
    types: list[str] = []
    if arr.size == 0:
        return types
    if np.array_equal(arr, np.fliplr(arr)):
        types.append("vertical")
    if np.array_equal(arr, np.flipud(arr)):
        types.append("horizontal")
    if np.array_equal(arr, np.flipud(np.fliplr(arr))):
        types.append("point")

    H, W = arr.shape
    for p in range(2, min(4, W) + 1):
        if W % p == 0 and W // p >= 2:
            base = arr[:, :p]
            if all(np.array_equal(arr[:, k : k + p], base) for k in range(0, W, p)):
                types.append(f"translation_x:{p}")
                break
    for p in range(2, min(4, H) + 1):
        if H % p == 0 and H // p >= 2:
            base = arr[:p, :]
            if all(np.array_equal(arr[k : k + p, :], base) for k in range(0, H, p)):
                types.append(f"translation_y:{p}")
                break
    return types


def _symmetry_score(rect: np.ndarray, rect_col: int) -> tuple[int, int]:
    if rect.shape[0] <= 2 or rect.shape[1] <= 2:
        return (0, 0)
    interior = rect[1:-1, 1:-1]
    kinds = _symmetry_types(interior)
    non_rect = int(np.sum(interior != int(rect_col)))
    return (len(kinds), non_rect)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered rectangular pieces into a rectangle at an L-marker corner.

    Variant-specific adjustment:
    - Edge compatibility and the `rect_col` wildcard remain unchanged.
    - If multiple full assemblies are possible, prefer the one whose interior exhibits
      clear symmetry, matching the D12 generator.
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

    def assemble(H: int, W: int) -> np.ndarray | None:
        occ2 = np.full((H, W), -1, dtype=int)
        usedp = [False] * len(pieces)

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

        order = sorted(range(len(pieces)), key=lambda i: (-ph[i] * pw[i], -ph[i], -pw[i], i))
        solved = False

        def next_empty() -> tuple[int, int] | None:
            idx = np.argwhere(occ2 == -1)
            if idx.size == 0:
                return None
            return tuple(map(int, idx[0]))

        def bt() -> None:
            nonlocal solved
            nxt = next_empty()
            if nxt is None:
                solved = True
                return
            r, c = nxt
            for i in order:
                if usedp[i]:
                    continue
                if can_place(i, r, c):
                    place(i, r, c)
                    bt()
                    if solved:
                        return
                    unplace(i, r, c)

        bt()
        return occ2 if solved else None

    best_ass: np.ndarray | None = None
    best_dim: tuple[int, int] | None = None
    best_score = (-1, -1)
    for H, W in dims:
        ass = assemble(H, W)
        if ass is None:
            continue
        score = _symmetry_score(ass, rect_col)
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


def _band_mask(length: int, rng: random.Random, frac_lo: float, frac_hi: float) -> np.ndarray:
    target_lo = max(1, int(np.ceil(length * frac_lo)))
    target_hi = max(target_lo, int(np.floor(length * frac_hi)))
    target = rng.randint(target_lo, min(length, target_hi))
    mask = np.zeros(length, dtype=bool)
    attempts = 0
    while int(mask.sum()) < target and attempts < 100:
        seg = rng.randint(1, max(1, min(length, 1 + length // 3)))
        seg = min(seg, length)
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
        seg = min(seg, length)
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
            col = int(colors[int(idx)])
            c_right = W - 1 - c_left
            out[r, c_left] = col
            out[r, c_right] = col


def _paint_point_row(out: np.ndarray, r: int, col_assign: np.ndarray, colors: list[int]) -> None:
    H, W = out.shape
    for c, idx in enumerate(col_assign):
        if int(idx) == -1:
            continue
        col = int(colors[int(idx)])
        out[r, c] = col
        out[H - 1 - r, W - 1 - c] = col


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
    base2 = _band_assign(W, len(colors), rng, 0.24, 0.42)
    shift2 = rng.randrange(W) if W > 0 else 0
    base2 = np.roll(base2, shift2)
    for r in range(top_h):
        if bool(row_mask1[r]):
            _paint_point_row(out, r, np.roll(base1, r % max(1, W)), colors)
        if bool(row_mask2[r]):
            _paint_point_row(out, r, np.roll(base2, (2 * r + 1) % max(1, W)), colors)
    return out


def _choose_period(n: int, rng: random.Random) -> int | None:
    opts = [p for p in range(2, min(4, n) + 1) if n % p == 0 and n // p >= 2]
    if not opts:
        return None
    return rng.choice(opts)


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


def _build_symmetric_interior(iH: int, iW: int, rect_col: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, str] | None:
    builders = [
        ("vertical", _build_vertical_pattern),
        ("horizontal", _build_horizontal_pattern),
        ("point", _build_point_pattern),
        ("translation_x", _build_translation_x_pattern),
        ("translation_y", _build_translation_y_pattern),
    ]
    rng.shuffle(builders)
    for want, builder in builders:
        for _ in range(8):
            interior = builder(iH, iW, rect_col, colors, rng)
            if interior is None:
                continue
            fill = float(np.mean(interior != int(rect_col)))
            used = set(map(int, np.unique(interior[interior != int(rect_col)])))
            if used != set(map(int, colors)):
                continue
            if fill < 0.38 or fill > 0.58:
                continue
            kinds = _symmetry_types(interior)
            if want not in kinds and not any(k.startswith(want) for k in kinds):
                continue
            return interior, want
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
    for _ in range(40):
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
        cell_id = [[i * nc + j for j in range(nc)] for i in range(nr)]
        n_cells = nr * nc
        if n_cells < 4:
            continue
        target_pieces = rng.randint(4, min(12, n_cells))

        def get_piece_bbox(pid: int) -> tuple[int, int, int, int] | None:
            cells = [(i, j) for i in range(nr) for j in range(nc) if cell_id[i][j] == pid]
            if not cells:
                return None
            ri = [c[0] for c in cells]
            ci = [c[1] for c in cells]
            return min(ri), min(ci), max(ri), max(ci)

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

        if len(pieces) < 4 or len(pieces) > 12:
            continue

        valid = True
        for pr, pc, ph, pw in pieces:
            if ph < 2 or pw < 2 or ph > 7 or pw > 7:
                valid = False
                break
            piece = rect[pr : pr + ph, pc : pc + pw]
            fill = float(np.mean(piece != int(rect_col)))
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
    n_interior = rng.randint(1, 3)
    interior_cols = palette[3 : 3 + n_interior]

    corner = rng.choice(["TL", "TR", "BL", "BR"])

    min_side, max_side = 7, min(22, min(GH, GW) - 3)
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
    for _ in range(20):
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
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
