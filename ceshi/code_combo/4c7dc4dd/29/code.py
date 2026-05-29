# Combo variant of 4c7dc4dd (index 29)
# combo_spec: D2/1+D9/0
# Name: 线性邻近配对
#
# Design analysis for the fusion:
# a) Concretization:
#    Original 4c7dc4dd let the solver find the training pair by matching interior palettes, with
#    four frames scattered across quadrants. Here the four frames sit on one exact row or one
#    exact column. The three nonblank interiors deliberately share the same palette, so palette
#    matching becomes ambiguous. The training pair is instead the uniquely closest adjacent
#    nonblank pair on that line, and the line order fixes src -> dst -> test.
# b) Complexity calibration:
#    The new cue is strong but not trivial: the line is obvious to the eye, the closest gap is
#    unique, and there is still exactly one hidden analogy rule from the original rule bank.
#    We keep only one blank frame and one shared palette so the scene stays readable.
# c) Broken old assumptions:
#    Any code that assumes "same palette identifies the pair" is invalid now. Any code that treats
#    the pair as directionless is also weaker, because the line order supplies source/destination
#    roles. The solver below therefore first infers the linear axis, then uses geometric adjacency.
# d) Placement legality:
#    Four same-size frames must fit in 30 cells along one axis. Because 4*8 > 30, size 8 from the
#    base task is no longer legal for a strict 1-D layout, so the generator restricts to sizes 6-7.
#    Gap templates are chosen so frames never overlap or leave the canvas.
# e) Visual observability:
#    Humans can see three facts immediately: all meaningful frames are on one line, the tightest
#    pair is adjacent, and the third nonblank frame lies later in the same sequence. That makes the
#    fusion cue visible without reading color ids or hidden metadata.
# f) Visual cleanliness:
#    The generator keeps exact row/column alignment, moderate background noise, and a larger gap
#    away from the blank frame whenever space allows. This avoids the quadrant clutter of the base
#    task and makes the nearest-pair relation easy to spot.
# g) Color inference:
#    All three nonblank interiors use the same non-zero color set on purpose. This prevents color
#    signatures from carrying the pairing information, while still leaving enough structure for the
#    original sprite-transform family to be learned from shape changes.

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, draw_line, paste


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
        if c != 0 and np.array_equal(src == c, dst == c):
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
                return ("sym_fill", {"stable": st, "keep0": keep0})

    if v is not None:
        pred = src.copy()
        pred[src == 0] = v
        pred[src == v] = 0
        for c in st:
            pred[dst == c] = c
        if np.array_equal(pred, dst):
            return ("invert0", {"stable": st})

    for idx, var in enumerate(symmetries(src)):
        if np.array_equal(var, dst):
            return ("sym", {"idx": idx})

    if v is not None:
        coords = np.argwhere(src == v)
        pts = set((int(r), int(c)) for r, c in coords)
        keep = np.zeros_like(src, dtype=bool)
        for r, c in pts:
            deg = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    if (r + dr, c + dc) in pts:
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
        h, w = x.shape
        best = None
        counts = {}
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
        y[(y == 0) & ~keep0] = fill_col
        return y

    if v is None:
        return x.copy()

    if kind == "invert0":
        y = x.copy()
        y[x == 0] = v
        y[x == v] = 0
        return y

    if kind == "endpoints":
        coords = np.argwhere(x == v)
        pts = set((int(r), int(c)) for r, c in coords)
        keep = np.zeros_like(x, dtype=bool)
        for r, c in pts:
            deg = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    if (r + dr, c + dc) in pts:
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


def _bincount_colors(values: np.ndarray) -> dict[int, int]:
    out: dict[int, int] = {}
    for v in values.tolist():
        if v != 0:
            out[v] = out.get(v, 0) + 1
    return out


def detect_blank_index(inners: list[np.ndarray]) -> int:
    for i, inner in enumerate(inners):
        if inner.size > 0 and np.all(inner == 0):
            return i
    return int(np.argmin([np.count_nonzero(inner) for inner in inners]))


def frame_centers(boxes: list[tuple[int, int, int, int]]) -> list[tuple[float, float]]:
    return [((r0 + r1) / 2.0, (c0 + c1) / 2.0) for r0, c0, r1, c1 in boxes]


def detect_linear_axis(boxes: list[tuple[int, int, int, int]]) -> int:
    centers = frame_centers(boxes)
    row_spread = max(r for r, _ in centers) - min(r for r, _ in centers)
    col_spread = max(c for _, c in centers) - min(c for _, c in centers)
    return 1 if col_spread >= row_spread else 0


def linear_order(indices: list[int], boxes: list[tuple[int, int, int, int]], axis: int) -> list[int]:
    centers = frame_centers(boxes)
    return sorted(indices, key=lambda i: (centers[i][axis], centers[i][1 - axis], i))


def adjacent_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], axis: int) -> int:
    if axis == 1:
        return int(b[1] - a[3] - 1)
    return int(b[0] - a[2] - 1)


def choose_roles(boxes: list[tuple[int, int, int, int]], inners: list[np.ndarray]):
    blank_idx = detect_blank_index(inners)
    nonblank = [i for i in range(len(inners)) if i != blank_idx]
    if len(nonblank) < 3:
        return None
    axis = detect_linear_axis(boxes)
    ordered = linear_order(nonblank, boxes, axis)
    gaps = [adjacent_gap(boxes[ordered[i]], boxes[ordered[i + 1]], axis) for i in range(len(ordered) - 1)]
    pair_pos = int(np.argmin(gaps))
    src_idx = ordered[pair_pos]
    dst_idx = ordered[pair_pos + 1]
    remain = [i for i in ordered if i not in (src_idx, dst_idx)]
    if not remain:
        return None
    test_idx = remain[0]
    return src_idx, dst_idx, test_idx, blank_idx, axis, ordered, gaps


def fallback_roles(inners: list[np.ndarray]):
    blank_idx = detect_blank_index(inners)
    nonblank = [i for i in range(len(inners)) if i != blank_idx]
    sigs = []
    for i in nonblank:
        inner = inners[i]
        nz = tuple(sorted(set(int(x) for x in inner.ravel() if int(x) != 0)))
        sigs.append((i, nz))
    pair = None
    for a in range(len(sigs)):
        for b in range(a + 1, len(sigs)):
            if sigs[a][1] == sigs[b][1]:
                pair = (sigs[a][0], sigs[b][0])
                break
        if pair is not None:
            break
    if pair is None:
        counts = sorted(((i, int(np.count_nonzero(inners[i]))) for i in nonblank), key=lambda t: t[1])
        pair = (counts[0][0], counts[1][0])
    remain = [i for i in nonblank if i not in pair]
    if not remain:
        return None
    return pair[0], pair[1], remain[0], blank_idx


def solve_from_pair(src_inner: np.ndarray, dst_inner: np.ndarray, test_inner: np.ndarray) -> np.ndarray:
    tpl = infer_template(src_inner, dst_inner)
    if tpl is not None:
        kind, params = tpl
        return apply_template(kind, params, test_inner)

    best = None
    for src, dst in ((src_inner, dst_inner), (dst_inner, src_inner)):
        tpl = infer_template(src, dst)
        if tpl is None:
            continue
        kind, params = tpl
        st = set(params.get("stable", set()))
        v_src = pick_var_color(src, st)
        v_dst = pick_var_color(dst, st)
        score = 1
        if kind == "fill" and v_src is not None and v_dst is not None and int(np.sum(dst == v_dst)) >= int(np.sum(src == v_src)):
            score += 2
        if kind == "endpoints" and v_src is not None and v_dst is not None and int(np.sum(dst == v_dst)) <= int(np.sum(src == v_src)):
            score += 2
        src_n = int(np.sum((src != 0) & ~np.isin(src, list(st))))
        test_n = int(np.sum((test_inner != 0) & ~np.isin(test_inner, list(st))))
        score2 = score * 100 - abs(src_n - test_n)
        if best is None or score2 > best[0]:
            best = (score2, kind, params)
    if best is None:
        return test_inner.copy()
    return apply_template(best[1], best[2], test_inner)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a sprite transform from a linearly ordered framed analogy and apply it to the third sprite.

    Input structure:
    - The grid contains four equal square frames arranged on one exact row or one exact column.
    - One frame is blank inside; the other three contain sprites and intentionally share the same
      non-zero interior color set.
    - Along the line, one adjacent nonblank pair is visibly closer than the other spacing.

    Rule:
    - Use the 1-D layout, not palette equality, to identify roles.
    - The uniquely closest adjacent nonblank pair is the training example; its earlier frame on the
      line is the source and its later frame is the destination.
    - The remaining nonblank frame is the test input. Learn the src->dst transform from the close
      pair, then apply that transform to the test sprite.

    Key insight:
    - The palette cue from the original task is now deliberately ambiguous. The reliable signal is
      geometric: line up the frames, find the tight pair, and read the analogy in line order.
    """
    g = np.asarray(grid, dtype=int)
    frames = find_square_frames(g)
    if len(frames) < 3:
        return g.copy()

    boxes = [(r0, c0, r1, c1) for _, _, r0, c0, r1, c1 in frames]
    inners = [g[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1 in frames]

    roles = choose_roles(boxes, inners)
    if roles is None:
        fb = fallback_roles(inners)
        if fb is None:
            return g.copy()
        src_idx, dst_idx, test_idx, blank_idx = fb
    else:
        src_idx, dst_idx, test_idx, blank_idx, _, _, _ = roles

    if test_idx >= len(inners):
        return inners[blank_idx].copy()
    return solve_from_pair(inners[src_idx], inners[dst_idx], inners[test_idx])


# --------------- generator ---------------

H = W = 30
SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
ALL_COLORS = list(range(10))
BG_PERIOD_CHOICES = (4, 5)
BG_PERTURB_FRACTION = (0.20, 0.35)
FRAME_SIZE_CHOICES = (6, 7)
RULES = ["endpoints", "fill", "sym_fill", "comp_checker", "invert0", "sym"]
SPRITE_NZ_MIN, SPRITE_NZ_MAX = 3, 32
SYMFILL_KEEP0_PROB = 0.4
FILL_RECT_AREA_MIN, FILL_RECT_AREA_MAX = 5, 25
ENDPOINTS_LINE_LEN_MIN = 3
COMP_CHECKER_M_MAX = 10
INVERT0_STABLE_MAX = 3
INVERT0_VAR_MAX = 12
SYM_BUILD_RETRIES = 30
SPARSE_COUNT_MAX = 10
GEN_MAX_ATTEMPTS = 400


def orbit_cells(n: int, r: int, c: int) -> list[tuple[int, int]]:
    cells = {
        (r, c),
        (r, n - 1 - c),
        (n - 1 - r, c),
        (n - 1 - r, n - 1 - c),
    }
    return sorted(cells)


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
    var_n = rng.randint(2, max(2, min(len(remaining) - 2, INVERT0_VAR_MAX)))
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


def build_sym_fill_pair(n: int, colors: list[int], rng: random.Random, keep0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fill_col, accent_col = colors
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
    dst = apply_template("sym_fill", {"keep0": keep0}, src)
    return src, dst


def choose_rule_setup(n: int, rng: random.Random) -> tuple[str, dict, int]:
    kind = rng.choice(RULES)
    params: dict = {}
    if kind == "invert0":
        color_count = 2
    elif kind == "sym_fill":
        color_count = 2
        keep0 = np.zeros((n, n), dtype=bool)
        if n == 6 or rng.random() < SYMFILL_KEEP0_PROB:
            for r, c in orbit_cells(n, 0, 0):
                keep0[r, c] = True
        params["keep0"] = keep0
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
        return build_sym_fill_pair(n, colors, rng, params["keep0"])
    raise ValueError(kind)


def _valid_nz(arr: np.ndarray) -> bool:
    return SPRITE_NZ_MIN <= int(np.count_nonzero(arr)) <= SPRITE_NZ_MAX


def _build_sprites(kind: str, setup_params: dict, n: int, palette: list[int], rng: random.Random):
    train_src, train_dst = build_training_pair(kind, setup_params, n, palette, rng)
    inferred = infer_template(train_src, train_dst)
    if inferred is None or inferred[0] != kind:
        return None
    inferred_kind, inferred_params = inferred

    train_sig = tuple(colors_present(train_src, exclude={0}))
    if train_sig != tuple(colors_present(train_dst, exclude={0})):
        return None

    for _ in range(20):
        test_src, _ = build_training_pair(kind, setup_params, n, palette, rng)
        test_dst = apply_template(inferred_kind, inferred_params, test_src)
        if np.array_equal(test_src, test_dst):
            continue
        if tuple(colors_present(test_src, exclude={0})) != train_sig:
            continue
        if not all(_valid_nz(arr) for arr in (train_src, train_dst, test_src, test_dst)):
            continue
        return train_src, train_dst, test_src, test_dst
    return None


def choose_frame_and_bg(used: set[int], rng: random.Random) -> tuple[int, list[int]]:
    frame_color = rng.choice([c for c in ALL_COLORS if c not in used])
    bg_pool = [c for c in ALL_COLORS if c not in used and c != frame_color]
    bg_palette = rng.sample(bg_pool, rng.choice(BG_PERIOD_CHOICES))
    return frame_color, bg_palette


def gap_patterns_for_size(s: int) -> list[tuple[int, int, int]]:
    if s == 7:
        return [(1, 0, 1), (1, 0, 1)]
    return [(2, 0, 2), (2, 0, 3), (3, 0, 2), (2, 1, 2)]


def linear_positions(s: int, rng: random.Random) -> tuple[list[tuple[int, int]], int, str]:
    axis = rng.choice([0, 1])
    blank_side = rng.choice(["start", "end"])
    far_gap, small_gap, mid_gap = rng.choice(gap_patterns_for_size(s))
    gaps = [far_gap, small_gap, mid_gap] if blank_side == "start" else [small_gap, mid_gap, far_gap]
    total_len = 4 * s + sum(gaps)
    slack = H - total_len
    start_margin = rng.randint(0, max(0, slack))
    orth = rng.randint(0, H - s)

    anchors = []
    cur = start_margin
    for i in range(4):
        anchors.append(cur)
        if i < 3:
            cur += s + gaps[i]

    if axis == 1:
        positions = [(orth, c) for c in anchors]
    else:
        positions = [(r, orth) for r in anchors]
    return positions, axis, blank_side


def render_background(frame_boxes: list[tuple[int, int, int, int]], bg_palette: list[int], rng: random.Random) -> np.ndarray:
    period = len(bg_palette)
    direction = rng.choice([1, -1])
    rr, cc = np.indices((H, W))
    idx = (rr + direction * cc) % period
    grid = np.asarray(bg_palette, dtype=int)[idx]

    reserved = np.zeros((H, W), dtype=bool)
    for r0, c0, r1, c1 in frame_boxes:
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True
    bg_cells = [(r, c) for r, c in zip(*np.where(~reserved))]
    perturb_n = int(round(len(bg_cells) * rng.uniform(*BG_PERTURB_FRACTION)))
    noise_choices = list(bg_palette)
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


def render_scene(s: int, positions: list[tuple[int, int]], sprites: list[np.ndarray], frame_color: int, bg_palette: list[int], rng: random.Random) -> np.ndarray:
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
    boxes = [(r0, c0, r1, c1) for _, _, r0, c0, r1, c1 in frames]
    inners = [grid[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1 in frames]
    if sum(int(np.all(inner == 0)) for inner in inners) != 1:
        return False

    axis = detect_linear_axis(boxes)
    centers = frame_centers(boxes)
    off_axis = [centers[i][1 - axis] for i in range(4)]
    if max(off_axis) - min(off_axis) > 0.01:
        return False

    blank_idx = detect_blank_index(inners)
    nonblank = [i for i in range(4) if i != blank_idx]
    sigs = [tuple(colors_present(inners[i], exclude={0})) for i in nonblank]
    dup_count = sum(1 for i in range(3) for j in range(i + 1, 3) if sigs[i] == sigs[j])
    if dup_count != 3:
        return False

    ordered = linear_order(nonblank, boxes, axis)
    gaps = [adjacent_gap(boxes[ordered[i]], boxes[ordered[i + 1]], axis) for i in range(2)]
    if not (gaps[0] < gaps[1]):
        return False

    return np.array_equal(main(grid), expected_output)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    s = rng.choice(FRAME_SIZE_CHOICES)
    n = s - 2
    kind, setup_params, color_count = choose_rule_setup(n, rng)
    palette = choose_palette(rng, color_count, set())

    built = _build_sprites(kind, setup_params, n, palette, rng)
    if built is None:
        return None
    train_src, train_dst, test_src, test_dst = built

    frame_color, bg_palette = choose_frame_and_bg(set(palette), rng)
    positions, _, blank_side = linear_positions(s, rng)

    blank = np.zeros((n, n), dtype=int)
    if blank_side == "start":
        sprites = [blank, train_src, train_dst, test_src]
    else:
        sprites = [train_src, train_dst, test_src, blank]

    grid = render_scene(s, positions, sprites, frame_color, bg_palette, rng)
    if not validate_scene(grid, test_dst):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
