# Variant of 4c7dc4dd
# Dimension: D10 — 框架尺寸一致性 (meta)
# Change: 所有4个框架使用相同的外部尺寸 S（6-8），内部尺寸 n = S-2（4-6）... → 混合框架尺寸
#
# Design analysis
# YAML-like notes used as the implementation contract for this variant:
# - concretization:
#   - Keep the original "4 framed sprites on a noisy 30x30 canvas" presentation.
#   - The two training frames use one square size `S_train`; the test frame and the blank frame
#     use another size `S_test`, with `S_train != S_test`.
#   - The rule is still learned only from the training pair's interiors and then applied to the
#     test interior, but now the learned rule must transfer from an `(S_train-2)x(S_train-2)`
#     sprite to an `(S_test-2)x(S_test-2)` sprite.
#   - Example trace: original `fill` logic compared raw training/test non-zero counts because
#     all interiors had the same area. In this variant a 4x4 training sprite and a 6x6 test
#     sprite may both encode the same "fill the gaps" behavior, so count-based tie-breaking must
#     be replaced by size-normalized reasoning.
# - complexity_calibration:
#   - The rule family stays the same to isolate D10 as the main change.
#   - Only size-agnostic rule parameters are generated. In particular, `sym_fill` may either keep
#     no forced-zero cells or keep exactly the reflected corner orbit; arbitrary boolean masks are
#     avoided because they do not transfer cleanly across sizes.
#   - The blank frame shares the test size, making the mixed-size structure visually obvious as a
#     2+2 size split rather than a single odd-sized outlier.
# - implicit_assumptions_audit:
#   - `find_square_frames()` already scans all square sizes, so it does not assume a shared size.
#   - The original `sym_fill` application used `keep0.shape == y.shape`; this silently dropped the
#     learned reserved-zero pattern when test size changed. Replaced with a size-agnostic mode.
#   - The original direction score used `abs(src_n - test_n)`, which assumes equal-area interiors.
#     Replaced with variable-pixel density comparison.
# - placement_legality:
#   - Frames still occupy four quadrants with size-aware anchor ranges, so no overlap or overflow
#     occurs even when the sizes differ.
#   - Validation explicitly checks: exactly 4 detected frames, exactly 2 frame sizes with counts
#     {2, 2}, one blank, one color-matched training pair, and a test size different from pair size.
# - visual_observability:
#   - Humans can see the D10 change directly because the training pair and the test/blank pair have
#     visibly different frame sizes in every generated sample.
#   - The transformation itself remains shape-based; no invisible ordering on color ids is used.
# - cleanliness_and_color_reasoning:
#   - Sprite palettes remain limited to 1-2 colors and the test palette stays disjoint from the
#     training palette, preserving the original "match by palette, generalize by structure" cue.

import numpy as np
import random
from arc_utils import colors_present, draw_line, paste


# ------- CFG: constants governing sampling distribution -------
# Canvas
H = W = 30

# Palette pools
SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
ALL_COLORS = list(range(10))
BG_PERIOD_CHOICES = (4, 5)
BG_PERTURB_FRACTION = (0.30, 0.50)
BG_EXTRA_NOISE = (0,)

# Frame layout
FRAME_SIZE_CHOICES = (6, 7, 8)
QUAD_LOW_MIN, QUAD_LOW_MAX = 1, 8
QUAD_BOTTOM_MARGIN = 8

# Rule set
RULES = ["endpoints", "fill", "sym_fill", "comp_checker", "invert0", "sym"]

# Sprite pixel-count bounds
SPRITE_NZ_MIN, SPRITE_NZ_MAX = 3, 32

# sym_fill
SYMFILL_KEEP0_PROB = 0.4

# Rule-specific build bounds
FILL_RECT_AREA_MIN, FILL_RECT_AREA_MAX = 5, 25
ENDPOINTS_LINE_LEN_MIN = 3
COMP_CHECKER_M_MAX = 10
INVERT0_STABLE_MAX = 3
INVERT0_VAR_MAX = 12
SYM_BUILD_RETRIES = 30
SPARSE_COUNT_MAX = 10

# Outer retry loop
GEN_MAX_ATTEMPTS = 400


def find_square_frames(grid: np.ndarray) -> list[tuple[int, int, int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    cands = []
    for size in range(4, min(h, w) + 1):
        for r0 in range(0, h - size + 1):
            r1 = r0 + size - 1
            for c0 in range(0, w - size + 1):
                c1 = c0 + size - 1
                col = int(g[r0, c0])
                if int(g[r0, c1]) != col or int(g[r1, c0]) != col or int(g[r1, c1]) != col:
                    continue
                if not (
                    np.all(g[r0, c0 : c1 + 1] == col)
                    and np.all(g[r1, c0 : c1 + 1] == col)
                    and np.all(g[r0 : r1 + 1, c0] == col)
                    and np.all(g[r0 : r1 + 1, c1] == col)
                ):
                    continue
                cands.append((size, col, r0, c0, r1, c1))
    cands.sort(key=lambda t: (-t[0], t[2], t[3]))
    picked = []
    for cand in cands:
        _, _, r0, c0, r1, c1 = cand
        overlaps = False
        for _, _, pr0, pc0, pr1, pc1 in picked:
            if not (r1 < pr0 or pr1 < r0 or c1 < pc0 or pc1 < c0):
                overlaps = True
                break
        if not overlaps:
            picked.append(cand)
    return picked


def stable_colors(src: np.ndarray, dst: np.ndarray) -> set[int]:
    st = set()
    for c in set(int(x) for x in np.unique(src)):
        if c == 0:
            continue
        if np.array_equal(src == c, dst == c):
            st.add(c)
    return st


def pick_var_color(x: np.ndarray, st: set[int]) -> int | None:
    vals = [int(v) for v in np.unique(x) if int(v) != 0 and int(v) not in st]
    if not vals:
        return None
    best = None
    for v in vals:
        cnt = int(np.sum(x == v))
        if best is None or cnt > best[0]:
            best = (cnt, v)
    return best[1]


def symmetries(arr: np.ndarray) -> list[np.ndarray]:
    outs = []
    for k in range(4):
        r = np.rot90(arr, k)
        outs.append(r)
        outs.append(np.fliplr(r))
    return outs


def _bincount_colors(values: np.ndarray) -> dict[int, int]:
    out: dict[int, int] = {}
    for v in values.tolist():
        if v == 0:
            continue
        out[v] = out.get(v, 0) + 1
    return out


def orbit_cells(n: int, r: int, c: int) -> list[tuple[int, int]]:
    cells = {
        (r, c),
        (r, n - 1 - c),
        (n - 1 - r, c),
        (n - 1 - r, n - 1 - c),
    }
    return sorted(cells)


def corner_orbit_mask(n: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    for r, c in orbit_cells(n, 0, 0):
        mask[r, c] = True
    return mask


def sym_fill_base(arr: np.ndarray) -> np.ndarray | None:
    y = np.zeros_like(arr)
    for t in (lambda z: z, np.fliplr, np.flipud, lambda z: np.flipud(np.fliplr(z))):
        zz = t(arr)
        rr, cc = np.where(zz != 0)
        for r, c in zip(rr.tolist(), cc.tolist()):
            val = int(zz[r, c])
            if y[r, c] != 0 and int(y[r, c]) != val:
                return None
            y[r, c] = val
    return y


def infer_template(src: np.ndarray, dst: np.ndarray):
    st = stable_colors(src, dst)
    v = pick_var_color(dst, st) or pick_var_color(src, st)

    if np.array_equal(dst == 0, src != 0):
        nz = np.argwhere(dst != 0)
        if nz.size > 0:
            even_cols = set()
            odd_cols = set()
            for r, c in nz:
                col = int(dst[int(r), int(c)])
                if ((int(r) + int(c)) & 1) == 0:
                    even_cols.add(col)
                else:
                    odd_cols.add(col)
            if len(even_cols) == 1 and len(odd_cols) == 1:
                return ("comp_checker", {"stable": st})

    sym_src = sym_fill_base(src)
    if sym_src is not None and np.all(dst[sym_src != 0] == sym_src[sym_src != 0]):
        fill_cols = sorted(set(int(x) for x in dst[sym_src == 0].ravel() if int(x) != 0))
        if len(fill_cols) <= 1:
            fill_col = fill_cols[0] if fill_cols else None
            if fill_col is None or not np.any((sym_src == 0) & (dst != 0) & (dst != fill_col)):
                keep0 = (sym_src == 0) & (dst == 0)
                if not np.any(keep0):
                    return ("sym_fill", {"stable": st, "keep0_mode": "none"})
                if np.array_equal(keep0, corner_orbit_mask(src.shape[0])):
                    return ("sym_fill", {"stable": st, "keep0_mode": "corner"})

    if v is not None:
        pred = src.copy()
        m0 = src == 0
        mv = src == v
        pred[m0] = v
        pred[mv] = 0
        for c in st:
            pred[dst == c] = c
        if np.array_equal(pred, dst):
            return ("invert0", {"stable": st})

    for idx, var in enumerate(symmetries(src)):
        if np.array_equal(var, dst):
            return ("sym", {"idx": idx})

    if v is not None:
        coords = np.argwhere(src == v)
        s = set((int(r), int(c)) for r, c in coords)
        keep = np.zeros_like(src, dtype=bool)
        for r, c in s:
            deg = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    if (r + dr, c + dc) in s:
                        deg += 1
            if deg <= 1:
                keep[r, c] = True
        pred = src.copy()
        pred[src == v] = 0
        pred[keep] = v
        if np.array_equal(pred, dst):
            return ("endpoints", {"stable": st})

        pred = src.copy()
        h, w = pred.shape
        stable_mask = np.zeros_like(pred, dtype=bool)
        for c in st:
            stable_mask |= pred == c
        for r in range(h):
            nz = np.where(pred[r, :] != 0)[0]
            if nz.size >= 2:
                cmin = int(nz.min())
                cmax = int(nz.max())
                for cc in range(cmin, cmax + 1):
                    if pred[r, cc] == 0 and not stable_mask[r, cc]:
                        pred[r, cc] = v
        for c in range(w):
            nz = np.where(pred[:, c] != 0)[0]
            if nz.size >= 2:
                rmin = int(nz.min())
                rmax = int(nz.max())
                for rr in range(rmin, rmax + 1):
                    if pred[rr, c] == 0 and not stable_mask[rr, c]:
                        pred[rr, c] = v
        if np.array_equal(pred, dst):
            return ("fill", {"stable": st})

    return None


def apply_template(kind: str, params: dict, x: np.ndarray) -> np.ndarray:
    if kind == "sym":
        return symmetries(x)[params["idx"]].copy()

    st = set(params.get("stable", set()))
    v = pick_var_color(x, st)

    if kind == "comp_checker":
        h, w = x.shape
        rr, cc = np.indices((h, w))
        parity = (rr + cc) & 1
        even_mask = (x != 0) & (parity == 0)
        odd_mask = (x != 0) & (parity == 1)
        even_counts = _bincount_colors(x[even_mask])
        odd_counts = _bincount_colors(x[odd_mask])
        if even_counts:
            c_even = max(even_counts.items(), key=lambda t: t[1])[0]
        elif odd_counts:
            c_even = max(odd_counts.items(), key=lambda t: t[1])[0]
        else:
            return x.copy()
        c_odd = max(odd_counts.items(), key=lambda t: t[1])[0] if odd_counts else c_even
        y = np.where(parity == 0, c_even, c_odd)
        y[x != 0] = 0
        return y.astype(x.dtype)

    if kind == "sym_fill":
        nz = np.argwhere(x != 0)
        if nz.size == 0:
            return x.copy()
        best = None
        counts = {}
        h, w = x.shape
        for r, c in nz:
            col = int(x[int(r), int(c)])
            d = min(int(r), int(c), h - 1 - int(r), w - 1 - int(c))
            counts[col] = counts.get(col, 0) + 1
            if best is None or d > best[0] or (d == best[0] and counts[col] > best[1]):
                best = (d, counts[col], col)
        fill_col = best[2]
        y = sym_fill_base(x)
        if y is None:
            return x.copy()
        keep0_mode = params.get("keep0_mode", "none")
        keep0 = np.zeros_like(y, dtype=bool)
        if keep0_mode == "corner":
            keep0 = corner_orbit_mask(y.shape[0])
        y[(y == 0) & ~keep0] = fill_col
        return y

    if v is None:
        return x.copy()

    if kind == "invert0":
        y = x.copy()
        m0 = x == 0
        mv = x == v
        y[m0] = v
        y[mv] = 0
        return y

    if kind == "endpoints":
        coords = np.argwhere(x == v)
        s = set((int(r), int(c)) for r, c in coords)
        keep = np.zeros_like(x, dtype=bool)
        for r, c in s:
            deg = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    if (r + dr, c + dc) in s:
                        deg += 1
            if deg <= 1:
                keep[r, c] = True
        y = x.copy()
        y[x == v] = 0
        y[keep] = v
        return y

    if kind == "fill":
        y = x.copy()
        h, w = y.shape
        stable_mask = np.zeros_like(y, dtype=bool)
        for c in st:
            stable_mask |= y == c
        for r in range(h):
            nz = np.where(y[r, :] != 0)[0]
            if nz.size >= 2:
                cmin = int(nz.min())
                cmax = int(nz.max())
                for cc in range(cmin, cmax + 1):
                    if y[r, cc] == 0 and not stable_mask[r, cc]:
                        y[r, cc] = v
        for c in range(w):
            nz = np.where(y[:, c] != 0)[0]
            if nz.size >= 2:
                rmin = int(nz.min())
                rmax = int(nz.max())
                for rr in range(rmin, rmax + 1):
                    if y[rr, c] == 0 and not stable_mask[rr, c]:
                        y[rr, c] = v
        return y

    return x.copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a sprite transformation from a framed training pair, then transfer it to a differently sized test frame.

    Input structure:
    - The canvas contains four hollow square frames over a noisy background, with two frame sizes rather than one.
    - Two non-blank frames of one size form the training pair, while the test frame and the blank frame use the other size.
    - Interior palettes still signal the analogy: the training pair shares a non-zero color set, and the test sprite uses a different color set.

    Rule:
    - Find the training pair by matching interior color sets instead of by frame size.
    - Learn which structural rule maps the training source interior to the training destination interior.
    - Apply that same size-agnostic rule to the test interior, producing an output whose side length matches the test frame's inner size.

    Key insight:
    - The important invariant is the transformation on the sprite pattern, not the absolute frame dimensions, so the learned operation must generalize from one square size to another.
    """
    g = np.asarray(grid, dtype=int)
    frames = find_square_frames(g)
    if len(frames) < 3:
        return g.copy()

    inners = []
    for size, col, r0, c0, r1, c1 in frames:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        inners.append(((size, r0, c0, r1, c1), inner))

    blank_idx = None
    for i, (_, inner) in enumerate(inners):
        if inner.size > 0 and np.all(inner == 0):
            blank_idx = i
            break
    if blank_idx is None:
        blank_idx = int(np.argmin([np.count_nonzero(inner) for _, inner in inners]))

    nonblank = [i for i in range(len(inners)) if i != blank_idx]
    sigs = []
    for i in nonblank:
        inner = inners[i][1]
        nz = sorted(set(int(x) for x in inner.ravel() if int(x) != 0))
        sigs.append((i, tuple(nz)))

    pair = None
    for a in range(len(sigs)):
        for b in range(a + 1, len(sigs)):
            if sigs[a][1] == sigs[b][1]:
                pair = (sigs[a][0], sigs[b][0])
                break
        if pair is not None:
            break
    if pair is None:
        counts = [(i, int(np.count_nonzero(inners[i][1]))) for i in nonblank]
        counts.sort(key=lambda t: t[1])
        pair = (counts[0][0], counts[1][0])

    remaining = [i for i in nonblank if i not in pair]
    if not remaining:
        return inners[blank_idx][1].copy()
    test_idx = remaining[0]

    src_a = inners[pair[0]][1]
    src_b = inners[pair[1]][1]
    test_inner = inners[test_idx][1]

    best = None
    for src, dst in ((src_a, src_b), (src_b, src_a)):
        tpl = infer_template(src, dst)
        if tpl is None:
            continue
        kind, params = tpl
        st = set(params.get("stable", set()))
        v_src = pick_var_color(src, st)
        v_dst = pick_var_color(dst, st)
        score = 1
        if kind == "fill" and v_src is not None and v_dst is not None:
            if int(np.sum(dst == v_dst)) >= int(np.sum(src == v_src)):
                score += 2
        if kind == "endpoints" and v_src is not None and v_dst is not None:
            if int(np.sum(dst == v_dst)) <= int(np.sum(src == v_src)):
                score += 2
        src_n = int(np.sum((src != 0) & ~np.isin(src, list(st))))
        test_n = int(np.sum((test_inner != 0) & ~np.isin(test_inner, list(st))))
        src_density = src_n / max(1, src.size)
        test_density = test_n / max(1, test_inner.size)
        score2 = score * 1000 - abs(src_density - test_density)
        if best is None or score2 > best[0]:
            best = (score2, kind, params)

    if best is None:
        return test_inner.copy()

    _, kind, params = best
    return apply_template(kind, params, test_inner)


def choose_palette(rng: random.Random, count: int, forbidden: set[int]) -> list[int]:
    pool = [c for c in SPRITE_COLORS if c not in forbidden]
    if len(pool) < count:
        raise ValueError("not enough sprite colors")
    return rng.sample(pool, count)


def build_fill_pair(n: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    v = colors[0]
    dst = np.zeros((n, n), dtype=int)
    while True:
        h = rng.randint(2, n)
        w = rng.randint(2, n)
        if FILL_RECT_AREA_MIN <= h * w <= FILL_RECT_AREA_MAX:
            break
    r0 = rng.randint(0, n - h)
    c0 = rng.randint(0, n - w)
    dst[r0 : r0 + h, c0 : c0 + w] = v
    src = np.zeros_like(dst)
    src[r0, c0] = v
    src[r0, c0 + w - 1] = v
    src[r0 + h - 1, c0] = v
    src[r0 + h - 1, c0 + w - 1] = v
    return src, dst


def build_endpoints_pair(n: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    v = colors[0]
    src = np.zeros((n, n), dtype=int)
    dst = np.zeros_like(src)
    row_pairs = [(0, n - 1)]
    if n >= 5:
        row_pairs.extend([(0, n - 2), (1, n - 1)])
    r1, r2 = rng.choice(row_pairs)
    len1 = rng.randint(ENDPOINTS_LINE_LEN_MIN, n)
    len2 = rng.randint(ENDPOINTS_LINE_LEN_MIN, n)
    c1 = rng.randint(0, n - len1)
    c2 = rng.randint(0, n - len2)
    draw_line(src, r1, c1, end_r=r1, end_c=c1 + len1 - 1, color=v)
    draw_line(src, r2, c2, end_r=r2, end_c=c2 + len2 - 1, color=v)
    dst[r1, c1] = v
    dst[r1, c1 + len1 - 1] = v
    dst[r2, c2] = v
    dst[r2, c2 + len2 - 1] = v
    return src, dst


def build_comp_checker_pair(n: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    if len(colors) == 1:
        even_color = odd_color = colors[0]
    else:
        even_color, odd_color = colors
    min_src = 4 if n == 6 else 3
    m = rng.randint(min_src, min(COMP_CHECKER_M_MAX, n * n - 3))
    all_cells = [(r, c) for r in range(n) for c in range(n)]
    while True:
        chosen = rng.sample(all_cells, m)
        parities = {(r + c) & 1 for r, c in chosen}
        if len(colors) == 1 or len(parities) == 2:
            break
    rr, cc = np.indices((n, n))
    full = np.where(((rr + cc) & 1) == 0, even_color, odd_color).astype(int)
    hole_mask = np.zeros((n, n), dtype=bool)
    for r, c in chosen:
        hole_mask[r, c] = True
    src = np.where(hole_mask, full, 0)
    dst = np.where(hole_mask, 0, full)
    return src, dst


def build_invert0_pair(n: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    v, s = colors
    all_cells = [(r, c) for r in range(n) for c in range(n)]
    stable_n = rng.randint(1, min(INVERT0_STABLE_MAX, n))
    stable_cells = set(rng.sample(all_cells, stable_n))
    remaining = [cell for cell in all_cells if cell not in stable_cells]
    var_hi = max(2, min(len(remaining) - 2, INVERT0_VAR_MAX))
    var_n = rng.randint(2, var_hi)
    var_cells = set(rng.sample(remaining, var_n))
    dst = np.zeros((n, n), dtype=int)
    for r, c in stable_cells:
        dst[r, c] = s
    for r, c in var_cells:
        dst[r, c] = v
    src = dst.copy()
    src[dst == 0] = v
    src[dst == v] = 0
    return src, dst


def inverse_symmetry(arr: np.ndarray, idx: int) -> np.ndarray:
    if idx == 0:
        return arr.copy()
    if idx == 1:
        return np.fliplr(arr)
    if idx == 2:
        return np.rot90(arr, -1)
    if idx == 3:
        return np.rot90(np.fliplr(arr), -1)
    if idx == 4:
        return np.rot90(arr, 2)
    if idx == 5:
        return np.flipud(arr)
    if idx == 6:
        return np.rot90(arr, 1)
    if idx == 7:
        return np.rot90(np.fliplr(arr), 1)
    raise ValueError(idx)


def random_sparse_pattern(n: int, colors: list[int], rng: random.Random) -> np.ndarray:
    arr = np.zeros((n, n), dtype=int)
    count = rng.randint(3, min(SPARSE_COUNT_MAX, n * n - 1))
    cells = rng.sample([(r, c) for r in range(n) for c in range(n)], count)
    used = [colors[0]] * count
    if len(colors) == 2:
        for i in range(count):
            used[i] = colors[rng.randint(0, 1)]
        used[0] = colors[0]
        used[1] = colors[1]
    for (r, c), col in zip(cells, used):
        arr[r, c] = col
    return arr


def build_sym_pair(n: int, colors: list[int], rng: random.Random, idx: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(SYM_BUILD_RETRIES):
        dst = random_sparse_pattern(n, colors, rng)
        src = inverse_symmetry(dst, idx)
        if not np.array_equal(src, dst):
            return src, dst
    raise ValueError("failed to build sym pair")


def build_sym_fill_pair(
    n: int,
    colors: list[int],
    rng: random.Random,
    keep0_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    fill_col, accent_col = colors
    keep0 = np.zeros((n, n), dtype=bool)
    if keep0_mode == "corner":
        keep0 = corner_orbit_mask(n)
    src = np.zeros((n, n), dtype=int)
    orbits = []
    seen = set()
    for r in range(n):
        for c in range(n):
            orb = tuple(orbit_cells(n, r, c))
            if orb not in seen:
                seen.add(orb)
                orbits.append(list(orb))
    orbits.sort(
        key=lambda orb: (
            -max(min(r, c, n - 1 - r, n - 1 - c) for r, c in orb),
            -len(orb),
            orb[0],
        )
    )
    fill_orbit = next(orb for orb in orbits if not any(keep0[r, c] for r, c in orb))
    src[fill_orbit[0]] = fill_col
    accent_orbits = [orb for orb in orbits if orb != fill_orbit and not any(keep0[r, c] for r, c in orb)]
    picks = []
    for orb in accent_orbits:
        if len(orb) >= 2:
            picks.append(orb)
        if len(picks) >= 2:
            break
    if len(picks) < 2:
        raise ValueError("not enough accent orbits")
    for orb in picks:
        src[orb[0]] = accent_col
    dst = apply_template("sym_fill", {"keep0_mode": keep0_mode}, src)
    return src, dst


def choose_rule_setup(n_train: int, n_test: int, rng: random.Random) -> tuple[str, dict, int]:
    kind = rng.choice(RULES)
    params: dict = {}
    if kind == "invert0":
        color_count = 2
    elif kind == "sym_fill":
        color_count = 2
        keep0_mode = "none"
        if min(n_train, n_test) >= 5 and rng.random() < SYMFILL_KEEP0_PROB:
            keep0_mode = "corner"
        params["keep0_mode"] = keep0_mode
    elif kind == "sym":
        color_count = rng.choice([1, 2])
        params["idx"] = rng.randint(1, 7)
    elif kind == "comp_checker":
        color_count = rng.choice([1, 2])
    else:
        color_count = 1
    return kind, params, color_count


def build_training_pair(kind: str, params: dict, n: int, colors: list[int], rng: random.Random):
    if kind == "fill":
        return build_fill_pair(n, colors, rng)
    if kind == "endpoints":
        return build_endpoints_pair(n, colors, rng)
    if kind == "comp_checker":
        return build_comp_checker_pair(n, colors, rng)
    if kind == "invert0":
        return build_invert0_pair(n, colors, rng)
    if kind == "sym":
        return build_sym_pair(n, colors, rng, params["idx"])
    if kind == "sym_fill":
        return build_sym_fill_pair(n, colors, rng, params["keep0_mode"])
    raise ValueError(kind)


def frame_anchor(slot: int, s: int, rng: random.Random) -> tuple[int, int]:
    top_low, top_high = QUAD_LOW_MIN, QUAD_LOW_MAX
    bottom_low = H - s - QUAD_BOTTOM_MARGIN
    bottom_high = H - s - 1
    if slot == 0:
        return rng.randint(top_low, top_high), rng.randint(top_low, top_high)
    if slot == 1:
        return rng.randint(top_low, top_high), rng.randint(bottom_low, bottom_high)
    if slot == 2:
        return rng.randint(bottom_low, bottom_high), rng.randint(top_low, top_high)
    return rng.randint(bottom_low, bottom_high), rng.randint(bottom_low, bottom_high)


def render_background(
    frame_boxes: list[tuple[int, int, int, int]],
    bg_palette: list[int],
    rng: random.Random,
) -> np.ndarray:
    period = len(bg_palette)
    direction = rng.choice([1, -1])
    rr, cc = np.indices((H, W))
    idx = (rr + direction * cc) % period
    grid = np.asarray(bg_palette, dtype=int)[idx]

    reserved = np.zeros((H, W), dtype=bool)
    for r0, c0, r1, c1 in frame_boxes:
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True
    bg_cells = [(r, c) for r, c in zip(*np.where(~reserved))]
    lo, hi = BG_PERTURB_FRACTION
    perturb_n = int(round(len(bg_cells) * rng.uniform(lo, hi)))
    noise_choices = list(bg_palette) + [c for c in BG_EXTRA_NOISE if c not in bg_palette]
    for r, c in rng.sample(bg_cells, perturb_n):
        grid[r, c] = rng.choice(noise_choices)
    return grid


def draw_frame(grid: np.ndarray, r0: int, c0: int, s: int, frame_color: int, sprite: np.ndarray) -> None:
    r1, c1 = r0 + s - 1, c0 + s - 1
    grid[r0 : r1 + 1, c0 : c1 + 1] = 0
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color
    paste(grid, sprite, r0 + 1, c0 + 1, transparent=0)


def validate_scene(grid: np.ndarray, expected_output: np.ndarray) -> bool:
    frames = find_square_frames(grid)
    if len(frames) != 4:
        return False
    sizes = [size for size, _, *_ in frames]
    size_set = set(sizes)
    if len(size_set) != 2:
        return False
    if sorted(sizes.count(s) for s in size_set) != [2, 2]:
        return False
    inners = [grid[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1 in frames]
    blank_ids = [i for i, inner in enumerate(inners) if np.all(inner == 0)]
    if len(blank_ids) != 1:
        return False
    blank_idx = blank_ids[0]

    nonblank = [i for i in range(4) if i != blank_idx]
    sigs = [(i, tuple(colors_present(inners[i], exclude={0}))) for i in nonblank]
    pair = None
    for a in range(len(sigs)):
        for b in range(a + 1, len(sigs)):
            if sigs[a][1] == sigs[b][1]:
                pair = (sigs[a][0], sigs[b][0])
                break
        if pair is not None:
            break
    if pair is None:
        return False
    remaining = [i for i in nonblank if i not in pair]
    if len(remaining) != 1:
        return False
    test_idx = remaining[0]
    if sizes[pair[0]] != sizes[pair[1]]:
        return False
    if sizes[test_idx] != sizes[blank_idx]:
        return False
    if sizes[test_idx] == sizes[pair[0]]:
        return False
    if np.array_equal(inners[test_idx], expected_output):
        return False
    return np.array_equal(main(grid), expected_output)


def _valid_nz(arr: np.ndarray) -> bool:
    return SPRITE_NZ_MIN <= int(np.count_nonzero(arr)) <= SPRITE_NZ_MAX


def _build_sprites(
    kind: str,
    setup_params: dict,
    n_train: int,
    n_test: int,
    p1: list[int],
    p2: list[int],
    rng: random.Random,
):
    train_src, train_dst = build_training_pair(kind, setup_params, n_train, p1, rng)
    inferred = infer_template(train_src, train_dst)
    if inferred is None or inferred[0] != kind:
        return None
    inferred_kind, inferred_params = inferred

    test_src, _ = build_training_pair(kind, setup_params, n_test, p2, rng)
    test_dst = apply_template(inferred_kind, inferred_params, test_src)

    if np.array_equal(test_dst, test_src):
        return None
    for arr in (train_src, train_dst, test_src, test_dst):
        if not _valid_nz(arr):
            return None
    train_sig = tuple(colors_present(train_src, exclude={0}))
    if train_sig != tuple(colors_present(train_dst, exclude={0})):
        return None
    if tuple(colors_present(test_src, exclude={0})) == train_sig:
        return None
    return train_src, train_dst, test_src, test_dst


def _choose_frame_and_bg(used: set[int], rng: random.Random) -> tuple[int, list[int]]:
    frame_color = rng.choice([c for c in ALL_COLORS if c not in used])
    bg_pool = [c for c in ALL_COLORS if c not in used and c != frame_color]
    bg_palette = rng.sample(bg_pool, rng.choice(BG_PERIOD_CHOICES))
    return frame_color, bg_palette


def _assign_slots(rng: random.Random) -> tuple[list[int], int, int]:
    order = [0, 1, 2, 3]
    train_slots = sorted(rng.sample(order, 2))
    remaining = [i for i in order if i not in train_slots]
    test_slot = rng.choice(remaining)
    blank_slot = [i for i in remaining if i != test_slot][0]
    return train_slots, test_slot, blank_slot


def _render_scene(
    sizes: list[int],
    positions: list[tuple[int, int]],
    sprites: list[np.ndarray],
    frame_color: int,
    bg_palette: list[int],
    rng: random.Random,
) -> np.ndarray:
    frame_boxes = [(r0, c0, r0 + s - 1, c0 + s - 1) for (r0, c0), s in zip(positions, sizes)]
    grid = render_background(frame_boxes, bg_palette, rng)
    for (r0, c0), s, sprite in zip(positions, sizes, sprites):
        draw_frame(grid, r0, c0, s, frame_color, sprite)
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    s_train, s_test = rng.sample(FRAME_SIZE_CHOICES, 2)
    n_train = s_train - 2
    n_test = s_test - 2

    kind, setup_params, color_count = choose_rule_setup(n_train, n_test, rng)
    p1 = choose_palette(rng, color_count, set())
    p2 = choose_palette(rng, color_count, set(p1))
    built = _build_sprites(kind, setup_params, n_train, n_test, p1, p2, rng)
    if built is None:
        return None
    train_src, train_dst, test_src, test_dst = built

    frame_color, bg_palette = _choose_frame_and_bg(set(p1) | set(p2), rng)
    train_slots, test_slot, blank_slot = _assign_slots(rng)

    sizes = [s_test] * 4
    for idx in train_slots:
        sizes[idx] = s_train
    sizes[test_slot] = s_test
    sizes[blank_slot] = s_test
    positions = [frame_anchor(i, sizes[i], rng) for i in range(4)]

    sprites = [np.zeros((sizes[i] - 2, sizes[i] - 2), dtype=int) for i in range(4)]
    sprites[train_slots[0]] = train_src
    sprites[train_slots[1]] = train_dst
    sprites[test_slot] = test_src

    grid = _render_scene(sizes, positions, sprites, frame_color, bg_palette, rng)
    if not validate_scene(grid, test_dst):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
