# Variant of 4c7dc4dd
# Dimension: D13 — 精灵颜色数量约束 (meta)
# Change: 每个精灵使用 1-2 种非背景色，训练对和测试精灵的颜色集不相交... → 训练对与测试共享部分颜色

import random

import numpy as np

from arc_utils import colors_present, draw_line, paste


# ---------------------------------------------------------------------------
# Design analysis for the D13 variant
#
# concrete_behavior:
#   - Each non-blank sprite now contains exactly two visible non-background
#     colors: one stable anchor color and one variable rule color.
#   - The training pair shares both colors, but the stable color is the
#     important one: it occupies the exact same cells in source and destination.
#   - The test sprite reuses that same stable color id, but its stable anchor
#     sits in a different corner motif. Its variable color is different from the
#     training pair's variable color.
#   - The rule acts only on the variable layer. The stable anchor is copied
#     through unchanged.
#
# behavior_trace:
#   - Original pairing logic: "pick the two non-blank interiors whose non-zero
#     color sets are equal".
#   - Variant pairing logic: evaluate all candidate pairs, detect whether a
#     color is unchanged in the same cells across the pair, strip that stable
#     layer from rule inference, and prefer the pair whose stable color is also
#     reused by the remaining test sprite.
#   - Small example:
#       train src colors = {yellow stable, red variable}
#       train dst colors = {yellow stable, red variable}
#       test src colors  = {yellow stable, blue variable}
#     Original code relied on test being color-disjoint from training. Here all
#     non-blank sprites overlap in yellow, so solver logic must inspect the
#     unchanged yellow mask instead of only looking at palette separation.
#
# complexity_calibration:
#   - The stable layer is restricted to one small 3-cell corner motif chosen
#     from four corner options. This is enough to be visually obvious without
#     flooding the sprite.
#   - The variable layer still uses one of the original rule families
#     {endpoints, fill, sym_fill, invert0, sym}. "comp_checker" is not sampled
#     by the generator because a persistent stable anchor plus the 2-color cap
#     makes it visually collapse into an invert0-like complement rule.
#   - Stable motifs repeat across source/destination within the training pair,
#     but test uses a different motif. That keeps the stable color observable
#     while preventing the input from looking like three identical color labels.
#
# implicit_assumption_audit:
#   - The original `main()` assumed the training pair could be found by exact
#     equality of interior color sets. That assumption is removed.
#   - The original rule inference sometimes used stable colors as blockers
#     (e.g. fill / invert0). In this variant the solver explicitly strips the
#     stable layer, but it still carries a blocked mask so stable cells are not
#     overwritten when the variable-layer rule is applied.
#   - Symmetry rules originally transformed the whole sprite. Here they apply to
#     the variable layer only; the stable anchor remains fixed. Generator-side
#     builders therefore avoid blocked cells and solver-side symmetry checks run
#     on the stripped variable layer.
#
# placement_legality:
#   - The generator still places four same-size non-overlapping square frames in
#     quadrant-like positions on a 30x30 canvas.
#   - Stable anchors are part of the interior sprites, so no extra objects are
#     introduced outside the original frame structure.
#   - Builders receive the stable mask as a blocked mask to prevent overlap
#     between anchor cells and variable rule cells.
#
# visual_observability:
#   - Humans can see one color acting as a fixed anchor inside the training
#     example while the other color changes by a familiar geometric rule.
#   - The test sprite clearly shares the anchor color but not the variable
#     color, which makes the D13 change visible in 2-3 examples without any
#     reliance on hidden color-id ordering.
# ---------------------------------------------------------------------------


# ------- CFG: constants governing sampling distribution -------
H = W = 30

SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
ALL_COLORS = list(range(10))
BG_PERIOD_CHOICES = (4, 5)
BG_PERTURB_FRACTION = (0.30, 0.50)
BG_EXTRA_NOISE = (0,)

FRAME_SIZE_CHOICES = (6, 7, 8)
QUAD_LOW_MIN, QUAD_LOW_MAX = 1, 8
QUAD_BOTTOM_MARGIN = 8

RULES = ["endpoints", "fill", "sym_fill", "invert0", "sym"]

SPRITE_NZ_MIN, SPRITE_NZ_MAX = 3, 32

FILL_RECT_AREA_MIN, FILL_RECT_AREA_MAX = 5, 25
ENDPOINTS_LINE_LEN_MIN = 3
INVERT0_VAR_MAX = 29
SYM_BUILD_RETRIES = 40
SPARSE_COUNT_MAX = 10
GEN_MAX_ATTEMPTS = 400


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer an analogy rule from framed sprites that share a stable anchor color, then solve the test frame.

    Input structure:
    - The grid contains four same-size hollow square frames on top of a noisy background.
    - Three frame interiors are non-blank. Every non-blank sprite uses a stable anchor color plus one variable rule color.
    - The training pair keeps the stable color in exactly the same cells, while the test sprite reuses that stable color id in a different anchor shape.

    Rule:
    - Find the training pair by looking for two non-blank sprites that preserve at least one non-zero color mask exactly.
    - Treat the preserved color as a stable anchor, infer the geometric rule on the remaining variable layer only, and keep the anchor blocked during the transform.
    - Apply that learned variable-layer rule to the remaining test sprite and restore its own stable anchor cells.

    Key insight:
    - Shared colors alone do not identify the pair anymore; the unchanged placement of the shared anchor color shows which two frames form the example and which color should be ignored while learning the real transform.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    frames = find_square_frames(g)
    if len(frames) < 3:
        return g.copy()

    inners = [g[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1 in frames]

    blank_idx = None
    for i, inner in enumerate(inners):
        if inner.size > 0 and np.all(inner == 0):
            blank_idx = i
            break
    if blank_idx is None:
        blank_idx = int(np.argmin([np.count_nonzero(inner) for inner in inners]))

    nonblank = [i for i in range(len(inners)) if i != blank_idx]
    if len(nonblank) < 2:
        return inners[blank_idx].copy()

    best = None
    for a in range(len(nonblank)):
        for b in range(a + 1, len(nonblank)):
            i = nonblank[a]
            j = nonblank[b]
            remaining = [k for k in nonblank if k not in (i, j)]
            if not remaining:
                continue
            test_idx = remaining[0]
            test_inner = inners[test_idx]
            for src_idx, dst_idx in ((i, j), (j, i)):
                tpl = infer_template(inners[src_idx], inners[dst_idx])
                if tpl is None:
                    continue
                kind, params = tpl
                pred = apply_template(kind, params, test_inner)
                score = candidate_score(inners[src_idx], inners[dst_idx], test_inner, pred, kind, params)
                if best is None or score > best[0]:
                    best = (score, pred)

    if best is None:
        return inners[nonblank[0]].copy()
    return best[1]


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
        if len(picked) >= 6:
            break
    return picked


def nonzero_colors(arr: np.ndarray, exclude: set[int] | None = None) -> tuple[int, ...]:
    ex = {0}
    if exclude is not None:
        ex |= set(exclude)
    return tuple(sorted(int(v) for v in np.unique(arr) if int(v) not in ex))


def stable_colors(src: np.ndarray, dst: np.ndarray) -> set[int]:
    st = set()
    for c in set(int(x) for x in np.unique(src)):
        if c == 0:
            continue
        if np.array_equal(src == c, dst == c):
            st.add(c)
    return st


def stable_mask(arr: np.ndarray, st: set[int]) -> np.ndarray:
    if not st:
        return np.zeros_like(arr, dtype=bool)
    return np.isin(arr, list(st))


def pick_var_color(x: np.ndarray, blocked: np.ndarray | None = None) -> int | None:
    vals = [int(v) for v in np.unique(x) if int(v) != 0]
    if not vals:
        return None
    best = None
    for v in vals:
        cnt = int(np.sum((x == v) & (~blocked if blocked is not None else True)))
        if best is None or cnt > best[0]:
            best = (cnt, v)
    return best[1]


def symmetries(arr: np.ndarray) -> list[np.ndarray]:
    outs = []
    for k in range(4):
        rot = np.rot90(arr, k)
        outs.append(rot)
        outs.append(np.fliplr(rot))
    return outs


def _bincount_colors(values: np.ndarray) -> dict[int, int]:
    out: dict[int, int] = {}
    for v in values.tolist():
        if v == 0:
            continue
        out[v] = out.get(v, 0) + 1
    return out


def sym_fill_base(arr: np.ndarray) -> np.ndarray | None:
    y = np.zeros_like(arr)
    for transform in (lambda z: z, np.fliplr, np.flipud, lambda z: np.flipud(np.fliplr(z))):
        zz = transform(arr)
        rr, cc = np.where(zz != 0)
        for r, c in zip(rr.tolist(), cc.tolist()):
            val = int(zz[r, c])
            if y[r, c] != 0 and int(y[r, c]) != val:
                return None
            y[r, c] = val
    return y


def infer_template(src: np.ndarray, dst: np.ndarray):
    st = stable_colors(src, dst)
    if not st:
        return None
    blocked = stable_mask(src, st)
    src_var = src.copy()
    dst_var = dst.copy()
    src_var[blocked] = 0
    dst_var[blocked] = 0
    core = infer_template_core(src_var, dst_var, blocked)
    if core is None:
        return None
    kind, params = core
    full_params = dict(params)
    full_params["stable"] = set(st)
    return kind, full_params


def infer_template_core(src: np.ndarray, dst: np.ndarray, blocked: np.ndarray):
    free = ~blocked
    v = pick_var_color(dst, blocked) or pick_var_color(src, blocked)

    if np.array_equal((dst == 0) & free, (src != 0) & free):
        nz = np.argwhere((dst != 0) & free)
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
                return ("comp_checker", {})

    sym_src = sym_fill_base(src)
    if sym_src is not None and np.all(sym_src[blocked] == 0):
        if np.all(dst[sym_src != 0] == sym_src[sym_src != 0]):
            fill_cols = sorted(set(int(x) for x in dst[(sym_src == 0) & free].ravel() if int(x) != 0))
            if len(fill_cols) <= 1:
                fill_col = fill_cols[0] if fill_cols else None
                ok = True
                if fill_col is not None:
                    bad = (sym_src == 0) & free & (dst != 0) & (dst != fill_col)
                    if np.any(bad):
                        ok = False
                if ok:
                    keep0 = (sym_src == 0) & free & (dst == 0)
                    return ("sym_fill", {"keep0": keep0})

    if v is not None:
        pred = src.copy()
        pred[(src == 0) & free] = v
        pred[src == v] = 0
        if np.array_equal(pred, dst):
            return ("invert0", {})

    for idx, var in enumerate(symmetries(src)):
        if np.array_equal(var, dst):
            return ("sym", {"idx": idx})

    if v is not None:
        coords = np.argwhere(src == v)
        cells = set((int(r), int(c)) for r, c in coords)
        keep = np.zeros_like(src, dtype=bool)
        for r, c in cells:
            deg = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    if (r + dr, c + dc) in cells:
                        deg += 1
            if deg <= 1:
                keep[r, c] = True
        pred = src.copy()
        pred[src == v] = 0
        pred[keep] = v
        if np.array_equal(pred, dst):
            return ("endpoints", {})

        pred = src.copy()
        h, w = pred.shape
        for r in range(h):
            nz = np.where(pred[r, :] != 0)[0]
            if nz.size >= 2:
                cmin = int(nz.min())
                cmax = int(nz.max())
                for cc in range(cmin, cmax + 1):
                    if free[r, cc] and pred[r, cc] == 0:
                        pred[r, cc] = v
        for c in range(w):
            nz = np.where(pred[:, c] != 0)[0]
            if nz.size >= 2:
                rmin = int(nz.min())
                rmax = int(nz.max())
                for rr in range(rmin, rmax + 1):
                    if free[rr, c] and pred[rr, c] == 0:
                        pred[rr, c] = v
        if np.array_equal(pred, dst):
            return ("fill", {})

    return None


def apply_template(kind: str, params: dict, x: np.ndarray) -> np.ndarray:
    st = set(params.get("stable", set()))
    blocked = stable_mask(x, st)
    stable_layer = np.where(blocked, x, 0)
    var_layer = x.copy()
    var_layer[blocked] = 0
    out_var = apply_template_core(kind, params, var_layer, blocked)
    out = out_var.copy()
    out[blocked] = stable_layer[blocked]
    return out


def apply_template_core(kind: str, params: dict, x: np.ndarray, blocked: np.ndarray) -> np.ndarray:
    free = ~blocked
    if kind == "sym":
        y = symmetries(x)[params["idx"]].copy()
        y[blocked] = 0
        return y

    v = pick_var_color(x, blocked)

    if kind == "comp_checker":
        h, w = x.shape
        rr, cc = np.indices((h, w))
        parity = (rr + cc) & 1
        even_mask = (x != 0) & free & (parity == 0)
        odd_mask = (x != 0) & free & (parity == 1)
        even_counts = _bincount_colors(x[even_mask])
        odd_counts = _bincount_colors(x[odd_mask])
        if even_counts:
            c_even = max(even_counts.items(), key=lambda t: t[1])[0]
        elif odd_counts:
            c_even = max(odd_counts.items(), key=lambda t: t[1])[0]
        else:
            return x.copy()
        c_odd = max(odd_counts.items(), key=lambda t: t[1])[0] if odd_counts else c_even
        y = np.zeros_like(x)
        y[free & (parity == 0)] = c_even
        y[free & (parity == 1)] = c_odd
        y[x != 0] = 0
        y[blocked] = 0
        return y.astype(x.dtype)

    if kind == "sym_fill":
        nz = np.argwhere((x != 0) & free)
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
        keep0 = params.get("keep0", None)
        if keep0 is None or keep0.shape != y.shape:
            keep0 = np.zeros_like(y, dtype=bool)
        y[(y == 0) & free & ~keep0] = fill_col
        y[blocked] = 0
        return y

    if v is None:
        return x.copy()

    if kind == "invert0":
        y = x.copy()
        y[(x == 0) & free] = v
        y[x == v] = 0
        return y

    if kind == "endpoints":
        coords = np.argwhere(x == v)
        cells = set((int(r), int(c)) for r, c in coords)
        keep = np.zeros_like(x, dtype=bool)
        for r, c in cells:
            deg = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    if (r + dr, c + dc) in cells:
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
        for r in range(h):
            nz = np.where(y[r, :] != 0)[0]
            if nz.size >= 2:
                cmin = int(nz.min())
                cmax = int(nz.max())
                for cc in range(cmin, cmax + 1):
                    if free[r, cc] and y[r, cc] == 0:
                        y[r, cc] = v
        for c in range(w):
            nz = np.where(y[:, c] != 0)[0]
            if nz.size >= 2:
                rmin = int(nz.min())
                rmax = int(nz.max())
                for rr in range(rmin, rmax + 1):
                    if free[rr, c] and y[rr, c] == 0:
                        y[rr, c] = v
        return y

    return x.copy()


def candidate_score(
    src: np.ndarray,
    dst: np.ndarray,
    test: np.ndarray,
    pred: np.ndarray,
    kind: str,
    params: dict,
) -> int:
    st = set(params.get("stable", set()))
    if not st:
        return -10_000

    train_var = nonzero_colors(src, st)
    test_var = nonzero_colors(test, st)
    train_stable_cells = int(np.sum(stable_mask(src, st)))
    shared_with_test = len(st & set(nonzero_colors(test)))
    stable_mask_diff = sum(1 for c in st if not np.array_equal(src == c, test == c))
    src_var_n = int(np.sum((src != 0) & ~stable_mask(src, st)))
    test_var_n = int(np.sum((test != 0) & ~stable_mask(test, st)))

    score = 1000
    score += 200 * shared_with_test
    score += 30 * stable_mask_diff
    score += 10 * train_stable_cells
    if train_var and test_var and train_var != test_var:
        score += 80
    if not np.array_equal(pred, test):
        score += 20
    if kind == "fill":
        score += 10
    if kind == "endpoints":
        score += 10
    score -= abs(src_var_n - test_var_n)
    return score


def orbit_cells(n: int, r: int, c: int) -> list[tuple[int, int]]:
    cells = {
        (r, c),
        (r, n - 1 - c),
        (n - 1 - r, c),
        (n - 1 - r, n - 1 - c),
    }
    return sorted(cells)


def choose_palette(rng: random.Random) -> tuple[int, int, int]:
    stable, train_var, test_var = rng.sample(SPRITE_COLORS, 3)
    return stable, train_var, test_var


def stable_motif_masks(n: int) -> list[np.ndarray]:
    motifs = [
        [(0, 0), (0, 1), (1, 0)],
        [(0, n - 1), (0, n - 2), (1, n - 1)],
        [(n - 1, 0), (n - 2, 0), (n - 1, 1)],
        [(n - 1, n - 1), (n - 2, n - 1), (n - 1, n - 2)],
    ]
    out = []
    for cells in motifs:
        mask = np.zeros((n, n), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        out.append(mask)
    return out


def choose_stable_masks(n: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    motifs = stable_motif_masks(n)
    train_idx, test_idx = rng.sample(range(len(motifs)), 2)
    return motifs[train_idx], motifs[test_idx]


def overlay_stable(variable: np.ndarray, mask: np.ndarray, stable_color: int) -> np.ndarray:
    out = variable.copy()
    out[mask] = stable_color
    return out


def free_cells(blocked: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in zip(*np.where(~blocked))]


def build_fill_pair(n: int, color: int, blocked: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(120):
        while True:
            h = rng.randint(2, n)
            w = rng.randint(2, n)
            if FILL_RECT_AREA_MIN <= h * w <= FILL_RECT_AREA_MAX:
                break
        r0 = rng.randint(0, n - h)
        c0 = rng.randint(0, n - w)
        if np.any(blocked[r0 : r0 + h, c0 : c0 + w]):
            continue
        dst = np.zeros((n, n), dtype=int)
        dst[r0 : r0 + h, c0 : c0 + w] = color
        src = np.zeros_like(dst)
        src[r0, c0] = color
        src[r0, c0 + w - 1] = color
        src[r0 + h - 1, c0] = color
        src[r0 + h - 1, c0 + w - 1] = color
        return src, dst
    raise ValueError("fill builder failed")


def row_segments(blocked: np.ndarray, min_len: int) -> list[tuple[int, int, int]]:
    n = blocked.shape[1]
    out = []
    for r in range(blocked.shape[0]):
        start = None
        for c in range(n + 1):
            free = c < n and not bool(blocked[r, c])
            if free and start is None:
                start = c
            if (not free or c == n) and start is not None:
                end = c - 1
                if end - start + 1 >= min_len:
                    out.append((r, start, end))
                start = None
    return out


def build_endpoints_pair(n: int, color: int, blocked: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    segs = row_segments(blocked, ENDPOINTS_LINE_LEN_MIN)
    by_row: dict[int, list[tuple[int, int, int]]] = {}
    for item in segs:
        by_row.setdefault(item[0], []).append(item)
    rows = list(by_row)
    if len(rows) < 2:
        raise ValueError("not enough free rows for endpoints")
    for _ in range(80):
        r1, r2 = rng.sample(rows, 2)
        if abs(r1 - r2) <= 1:
            continue
        s1 = rng.choice(by_row[r1])
        s2 = rng.choice(by_row[r2])
        src = np.zeros((n, n), dtype=int)
        dst = np.zeros_like(src)
        len1 = rng.randint(ENDPOINTS_LINE_LEN_MIN, s1[2] - s1[1] + 1)
        len2 = rng.randint(ENDPOINTS_LINE_LEN_MIN, s2[2] - s2[1] + 1)
        c1 = rng.randint(s1[1], s1[2] - len1 + 1)
        c2 = rng.randint(s2[1], s2[2] - len2 + 1)
        draw_line(src, r1, c1, end_r=r1, end_c=c1 + len1 - 1, color=color)
        draw_line(src, r2, c2, end_r=r2, end_c=c2 + len2 - 1, color=color)
        dst[r1, c1] = color
        dst[r1, c1 + len1 - 1] = color
        dst[r2, c2] = color
        dst[r2, c2 + len2 - 1] = color
        return src, dst
    raise ValueError("endpoints builder failed")


def build_invert0_pair(n: int, color: int, blocked: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    free = free_cells(blocked)
    free_n = len(free)
    lo = max(1, free_n - INVERT0_VAR_MAX)
    hi = min(INVERT0_VAR_MAX, free_n - 1)
    if lo > hi:
        raise ValueError("invert0 free-range failed")
    var_n = rng.randint(lo, hi)
    chosen = set(rng.sample(free, var_n))
    src = np.zeros((n, n), dtype=int)
    dst = np.zeros_like(src)
    for cell in free:
        if cell in chosen:
            dst[cell] = color
        else:
            src[cell] = color
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


def random_sparse_pattern(n: int, color: int, blocked: np.ndarray, rng: random.Random) -> np.ndarray:
    arr = np.zeros((n, n), dtype=int)
    cells = free_cells(blocked)
    max_count = min(SPARSE_COUNT_MAX, len(cells) - 1, SPRITE_NZ_MAX - 3)
    count = rng.randint(3, max_count)
    for r, c in rng.sample(cells, count):
        arr[r, c] = color
    return arr


def build_sym_pair(n: int, color: int, blocked: np.ndarray, rng: random.Random, idx: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(SYM_BUILD_RETRIES):
        dst = random_sparse_pattern(n, color, blocked, rng)
        src = inverse_symmetry(dst, idx)
        if np.any(src[blocked] != 0) or np.any(dst[blocked] != 0):
            continue
        if np.array_equal(src, dst):
            continue
        return src, dst
    raise ValueError("sym builder failed")


def sym_fill_keep0(n: int) -> np.ndarray:
    keep0 = np.zeros((n, n), dtype=bool)
    if n == 6:
        for r, c in orbit_cells(n, 2, 2):
            keep0[r, c] = True
    return keep0


def build_sym_fill_pair(
    n: int,
    color: int,
    blocked: np.ndarray,
    rng: random.Random,
    keep0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    src = np.zeros((n, n), dtype=int)
    seen = set()
    orbits = []
    for r in range(n):
        for c in range(n):
            orb = tuple(orbit_cells(n, r, c))
            if orb not in seen:
                seen.add(orb)
                orbits.append(list(orb))
    candidates = [
        orb
        for orb in orbits
        if all(not blocked[r, c] and not keep0[r, c] for r, c in orb)
    ]
    if not candidates:
        raise ValueError("sym_fill no candidate orbit")
    rng.shuffle(candidates)
    picks = candidates[: max(1, min(2, len(candidates)))]
    for orb in picks:
        src[orb[0]] = color
    dst = apply_template_core("sym_fill", {"keep0": keep0}, src, blocked)
    if np.array_equal(src, dst):
        raise ValueError("sym_fill degenerate")
    return src, dst


def choose_rule_setup(n: int, rng: random.Random) -> tuple[str, dict]:
    kind = rng.choice(RULES)
    params: dict = {}
    if kind == "sym":
        params["idx"] = rng.randint(1, 7)
    elif kind == "sym_fill":
        params["keep0"] = sym_fill_keep0(n)
    return kind, params


def build_training_pair(kind: str, params: dict, n: int, color: int, blocked: np.ndarray, rng: random.Random):
    if kind == "fill":
        return build_fill_pair(n, color, blocked, rng)
    if kind == "endpoints":
        return build_endpoints_pair(n, color, blocked, rng)
    if kind == "invert0":
        return build_invert0_pair(n, color, blocked, rng)
    if kind == "sym":
        return build_sym_pair(n, color, blocked, rng, params["idx"])
    if kind == "sym_fill":
        return build_sym_fill_pair(n, color, blocked, rng, params["keep0"])
    raise ValueError(kind)


def _valid_nz(arr: np.ndarray) -> bool:
    return SPRITE_NZ_MIN <= int(np.count_nonzero(arr)) <= SPRITE_NZ_MAX


def _build_sprites(
    kind: str,
    setup_params: dict,
    n: int,
    stable_color: int,
    train_var: int,
    test_var: int,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    rng: random.Random,
):
    train_src_var, train_dst_var = build_training_pair(kind, setup_params, n, train_var, train_mask, rng)
    train_src = overlay_stable(train_src_var, train_mask, stable_color)
    train_dst = overlay_stable(train_dst_var, train_mask, stable_color)

    inferred = infer_template(train_src, train_dst)
    if inferred is None or inferred[0] != kind:
        return None
    inferred_kind, inferred_params = inferred
    if stable_color not in inferred_params.get("stable", set()):
        return None

    test_src_var, _ = build_training_pair(kind, setup_params, n, test_var, test_mask, rng)
    test_src = overlay_stable(test_src_var, test_mask, stable_color)
    test_dst = apply_template(inferred_kind, inferred_params, test_src)

    if np.array_equal(test_dst, test_src):
        return None
    if not np.array_equal(test_dst == stable_color, test_mask):
        return None

    for arr in (train_src, train_dst, test_src, test_dst):
        if not _valid_nz(arr):
            return None
        cols = nonzero_colors(arr)
        if len(cols) != 2:
            return None

    train_sig = tuple(colors_present(train_src, exclude={0}))
    test_sig = tuple(colors_present(test_src, exclude={0}))
    if train_sig != tuple(colors_present(train_dst, exclude={0})):
        return None
    if stable_color not in test_sig or train_var in test_sig or test_var in train_sig:
        return None
    if np.array_equal(train_mask, test_mask):
        return None
    if int(np.sum(test_src == test_var)) == 0 or int(np.sum(train_src == train_var)) == 0:
        return None
    return train_src, train_dst, test_src, test_dst


def _choose_frame_and_bg(used: set[int], rng: random.Random) -> tuple[int, list[int]]:
    frame_color = rng.choice([c for c in ALL_COLORS if c not in used])
    bg_pool = [c for c in ALL_COLORS if c not in used and c != frame_color]
    bg_palette = rng.sample(bg_pool, rng.choice(BG_PERIOD_CHOICES))
    return frame_color, bg_palette


def quadrant_positions(s: int, rng: random.Random) -> list[tuple[int, int]]:
    low, high = QUAD_LOW_MIN, QUAD_LOW_MAX
    bottom_low = H - s - QUAD_BOTTOM_MARGIN
    bottom_high = H - s - 1
    return [
        (rng.randint(low, high), rng.randint(low, high)),
        (rng.randint(low, high), rng.randint(bottom_low, bottom_high)),
        (rng.randint(bottom_low, bottom_high), rng.randint(low, high)),
        (rng.randint(bottom_low, bottom_high), rng.randint(bottom_low, bottom_high)),
    ]


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
    bg_cells = [(int(r), int(c)) for r, c in zip(*np.where(~reserved))]
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


def _assign_slots(positions: list[tuple[int, int]], rng: random.Random) -> tuple[list[int], int]:
    order = sorted(range(4), key=lambda i: positions[i])
    train_slots = sorted(rng.sample(order, 2), key=order.index)
    remaining = [i for i in order if i not in train_slots]
    test_slot = rng.choice(remaining)
    return train_slots, test_slot


def _render_scene(
    s: int,
    positions: list[tuple[int, int]],
    sprites: list[np.ndarray],
    frame_color: int,
    bg_palette: list[int],
    rng: random.Random,
) -> np.ndarray:
    frame_boxes = [(r0, c0, r0 + s - 1, c0 + s - 1) for r0, c0 in positions]
    grid = render_background(frame_boxes, bg_palette, rng)
    for (r0, c0), sprite in zip(positions, sprites):
        draw_frame(grid, r0, c0, s, frame_color, sprite)
    return grid


def validate_scene(grid: np.ndarray, expected_output: np.ndarray) -> bool:
    frames = find_square_frames(grid)
    if len(frames) != 4:
        return False
    sizes = {size for size, _, *_ in frames}
    if len(sizes) != 1 or next(iter(sizes)) not in set(FRAME_SIZE_CHOICES):
        return False
    inners = [grid[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1 in frames]
    if sum(int(np.all(inner == 0)) for inner in inners) != 1:
        return False
    nonblank = [inner for inner in inners if not np.all(inner == 0)]
    if len(nonblank) != 3:
        return False
    overlap_ok = False
    for i in range(3):
        for j in range(i + 1, 3):
            if set(nonzero_colors(nonblank[i])) & set(nonzero_colors(nonblank[j])):
                overlap_ok = True
    if not overlap_ok:
        return False
    return np.array_equal(main(grid), expected_output)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    s = rng.choice(FRAME_SIZE_CHOICES)
    n = s - 2

    stable_color, train_var, test_var = choose_palette(rng)
    train_mask, test_mask = choose_stable_masks(n, rng)
    kind, setup_params = choose_rule_setup(n, rng)

    built = _build_sprites(
        kind,
        setup_params,
        n,
        stable_color,
        train_var,
        test_var,
        train_mask,
        test_mask,
        rng,
    )
    if built is None:
        return None
    train_src, train_dst, test_src, test_dst = built

    frame_color, bg_palette = _choose_frame_and_bg({stable_color, train_var, test_var}, rng)
    positions = quadrant_positions(s, rng)
    train_slots, test_slot = _assign_slots(positions, rng)

    sprites = [np.zeros((n, n), dtype=int) for _ in range(4)]
    sprites[train_slots[0]] = train_src
    sprites[train_slots[1]] = train_dst
    sprites[test_slot] = test_src

    grid = _render_scene(s, positions, sprites, frame_color, bg_palette, rng)
    if not validate_scene(grid, test_dst):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
