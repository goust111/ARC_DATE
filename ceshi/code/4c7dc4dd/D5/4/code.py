# Variant of 4c7dc4dd
# Dimension: D5 — 精灵变换操作集 (transform)
# Change: 六种规则之一: endpoints（取端点）、fill（填充间隙）、sym_fill（对称填充）、comp_checker（棋盘互补）、invert0（零与变量色互换）、sym（旋转/翻转） → 连接线绘制

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, draw_line, paste


# Design analysis for the D5 connector_draw variant
# --------------------------------------------------
# a) Concrete behavior and trace
#    The original D5 "fill" case scanned each row/column and filled every zero
#    between the outermost non-zero cells. The variant instead joins only paired
#    variable-color pixels that already lie on the same row or column, and only
#    when the path between them contains no foreign color. Small example:
#
#        src row: [v, 0, v, 0, s]
#        fill   : [v, v, v, v, s]
#        new dst: [v, v, v, 0, s]
#
#    The connector only bridges the matching v endpoints; it does not keep
#    extending just because another non-zero color appears farther along the row.
#
# b) Complexity calibration
#    To make the new rule visible and avoid collapsing back into fill, generator
#    samples connector sprites with exactly two interior colors:
#    - one variable color v that grows into line segments
#    - one stable accent color s that stays fixed
#    At least one row/column is constructed so that old fill would over-extend or
#    would react to an obstacle differently, guaranteeing a visible distinction.
#
# c) Hidden-assumption audit for main()
#    - stable_colors() still works: the accent color occupies identical cells in
#      src and dst, so it is a valid stable color.
#    - pick_var_color() assumes a unique transforming color; generator enforces
#      only one non-stable color inside the training pair.
#    - connector inference assumes rows/cols can be interpreted as independent
#      pairs. Generator keeps variable endpoints sparse enough that the inferred
#      pairing is deterministic, and infer_template() is verified on each sample.
#    - Training and test palettes remain disjoint as in the source task, so the
#      connector application step selects the active test color structurally
#      (which color can actually form bridges) instead of relying on literal
#      training color ids.
#    - The old fill code was harmlessly monotone because it only existed as one
#      candidate rule. In this variant, keeping that behavior would create false
#      positives, so generator explicitly rejects scenes where connector_draw and
#      old fill produce the same test output.
#
# d) Placement legality
#    Frames stay non-overlapping and quadrant-separated exactly as in the source
#    task. Connector endpoints and accents are sampled only inside the n x n
#    interiors, and the output is still the transformed test interior.
#
# e) Visual observability
#    Human-visible evidence comes from repeated "same-color dots get bridged"
#    behavior plus nearby accent cells that do not trigger bridging. No sorting
#    or color remapping depends on color ids.
#
# f) Visual cleanliness
#    Each connector sprite uses only 1-2 actual segments and 1-2 accent cells.
#    This keeps the framed interiors legible even on the smallest n = 4 case.
#
# g) Color-change inferability
#    There is no arbitrary color mapping: the same variable color expands into
#    the bridge, while the accent color remains unchanged. Two colors are enough
#    to show the rule without introducing palette ambiguity.


def find_square_frames(grid: np.ndarray, limit: int | None = None) -> list[tuple[int, int, int, int, int, int]]:
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
        if limit is not None and len(picked) >= limit:
            break
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
        if int(v) == 0:
            continue
        out[int(v)] = out.get(int(v), 0) + 1
    return out


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


def span_fill_transform(x: np.ndarray, st: set[int], v: int | None) -> np.ndarray:
    y = x.copy()
    if v is None:
        return y
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


def connector_draw_transform(x: np.ndarray, st: set[int], v: int | None) -> np.ndarray:
    y = x.copy()
    if v is None:
        return y
    base = np.asarray(x, dtype=int)
    h, w = base.shape
    for r in range(h):
        cols = np.where(base[r, :] == v)[0].tolist()
        for i in range(0, len(cols) - 1, 2):
            c0 = int(cols[i])
            c1 = int(cols[i + 1])
            segment = base[r, c0 : c1 + 1]
            if np.any((segment != 0) & (segment != v)):
                continue
            for cc in range(c0 + 1, c1):
                if base[r, cc] == 0:
                    y[r, cc] = v
    for c in range(w):
        rows = np.where(base[:, c] == v)[0].tolist()
        for i in range(0, len(rows) - 1, 2):
            r0 = int(rows[i])
            r1 = int(rows[i + 1])
            segment = base[r0 : r1 + 1, c]
            if np.any((segment != 0) & (segment != v)):
                continue
            for rr in range(r0 + 1, r1):
                if base[rr, c] == 0:
                    y[rr, c] = v
    return y


def pick_connector_color(x: np.ndarray, st: set[int]) -> int | None:
    candidates = [int(v) for v in np.unique(x) if int(v) != 0 and int(v) not in st]
    if not candidates:
        candidates = [int(v) for v in np.unique(x) if int(v) != 0]
    best = None
    for color in candidates:
        pred = connector_draw_transform(x, st, color)
        added = int(np.sum(pred == color) - np.sum(x == color))
        cnt = int(np.sum(x == color))
        cand = (added, cnt, color)
        if best is None or cand > best:
            best = cand
    if best is not None and best[0] > 0:
        return best[2]
    return pick_var_color(x, st)


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
    if sym_src is not None:
        if np.all(dst[sym_src != 0] == sym_src[sym_src != 0]):
            fill_cols = sorted(set(int(x) for x in dst[sym_src == 0].ravel() if int(x) != 0))
            if len(fill_cols) <= 1:
                fill_col = fill_cols[0] if fill_cols else None
                ok = True
                if fill_col is not None:
                    bad = (sym_src == 0) & (dst != 0) & (dst != fill_col)
                    if np.any(bad):
                        ok = False
                if ok:
                    keep0 = (sym_src == 0) & (dst == 0)
                    return ("sym_fill", {"stable": st, "keep0": keep0})

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

        pred = connector_draw_transform(src, st, v)
        if np.array_equal(pred, dst):
            return ("connector_draw", {"stable": st})

    return None


def apply_template(kind: str, params: dict, x: np.ndarray) -> np.ndarray:
    if kind == "sym":
        return symmetries(x)[params["idx"]].copy()

    st = set(params.get("stable", set()))
    if kind == "connector_draw":
        return connector_draw_transform(x, st, pick_connector_color(x, st))

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
        h, w = x.shape
        nz = np.argwhere(x != 0)
        if nz.size == 0:
            return x.copy()
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

    return x.copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn a same-color connector rule from one framed sprite pair, then run it on a test sprite.

    Input structure:
    - The board contains several same-style hollow square frames on a noisy background.
    - Inside each frame is a small sprite; one frame is blank, two frames share a palette and act as the training pair.
    - The remaining non-blank frame is the test sprite whose transformed interior must be produced.

    Rule:
    - Match the training pair by their shared non-zero color set and treat the sparser member as the source pattern.
    - The active color draws straight horizontal or vertical bridges only between aligned pixels of that same color, and only through empty cells; unchanged accent colors do not participate.
    - Apply that connector rule to the test interior and return only the transformed inner square.

    Key insight:
    - This is an analogy puzzle where one color acts like endpoints and another may act like passive clutter; the visible cue is that matching-colored dots become connected without flooding every span that merely contains some non-zero cell.
    """
    g = np.asarray(grid, dtype=int)
    frames = find_square_frames(g, limit=6)
    if len(frames) < 3:
        return g.copy()

    inners = []
    for _, _, r0, c0, r1, c1 in frames:
        inners.append(((r0, c0, r1, c1), g[r0 + 1 : r1, c0 + 1 : c1]))

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
        if kind == "connector_draw" and v_src is not None and v_dst is not None:
            if int(np.sum(dst == v_dst)) >= int(np.sum(src == v_src)):
                score += 2
        if kind == "endpoints" and v_src is not None and v_dst is not None:
            if int(np.sum(dst == v_dst)) <= int(np.sum(src == v_src)):
                score += 2
        src_n = int(np.sum((src != 0) & ~np.isin(src, list(st))))
        test_n = int(np.sum((test_inner != 0) & ~np.isin(test_inner, list(st))))
        score2 = score * 100 - abs(src_n - test_n)
        if best is None or score2 > best[0]:
            best = (score2, kind, params)

    if best is None:
        return test_inner.copy()

    _, kind, params = best
    return apply_template(kind, params, test_inner)


# --------------- generator (connector_draw variant) ---------------

H = W = 30

SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
ALL_COLORS = list(range(10))
BG_PERIOD_CHOICES = (4, 5)
BG_PERTURB_FRACTION = (0.30, 0.50)
BG_EXTRA_NOISE = (0,)

FRAME_SIZE_CHOICES = (6, 7, 8)
QUAD_LOW_MIN, QUAD_LOW_MAX = 1, 8
QUAD_BOTTOM_MARGIN = 8

RULES = ["connector_draw"]

SPRITE_NZ_MIN, SPRITE_NZ_MAX = 3, 32

SYMFILL_KEEP0_PROB = 0.4
ENDPOINTS_LINE_LEN_MIN = 3
COMP_CHECKER_M_MAX = 10
INVERT0_STABLE_MAX = 3
INVERT0_VAR_MAX = 12
SYM_BUILD_RETRIES = 30
SPARSE_COUNT_MAX = 10
CONNECTOR_BUILD_RETRIES = 100

GEN_MAX_ATTEMPTS = 300


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


def build_sym_fill_pair(
    n: int,
    colors: list[int],
    rng: random.Random,
    keep0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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


def _line_cells(orient: str, fixed: int, coords: list[int]) -> list[tuple[int, int]]:
    if orient == "h":
        return [(fixed, c) for c in coords]
    return [(r, fixed) for r in coords]


def _counts_ok(var_cells: set[tuple[int, int]]) -> bool:
    row_counts: dict[int, int] = {}
    col_counts: dict[int, int] = {}
    for r, c in var_cells:
        row_counts[r] = row_counts.get(r, 0) + 1
        col_counts[c] = col_counts.get(c, 0) + 1
        if row_counts[r] > 2 or col_counts[c] > 2:
            return False
    return True


def _clear_candidates(
    n: int,
    occupied: set[tuple[int, int]],
    var_cells: set[tuple[int, int]],
) -> list[tuple[str, int, int, int, list[tuple[int, int]]]]:
    cands = []
    for orient in ("h", "v"):
        for fixed in range(n):
            for a in range(n):
                for b in range(a + 2, n):
                    cells = _line_cells(orient, fixed, [a, b])
                    if any(cell in occupied for cell in cells):
                        continue
                    if not _counts_ok(var_cells | set(cells)):
                        continue
                    cands.append((orient, fixed, a, b, cells))
    return cands


def _tail_witness_candidates(
    n: int,
    occupied: set[tuple[int, int]],
    var_cells: set[tuple[int, int]],
) -> list[tuple[str, int, int, int, int, list[tuple[int, int]]]]:
    cands = []
    for orient in ("h", "v"):
        for fixed in range(n):
            for a in range(n):
                for b in range(a + 2, n):
                    for p in range(n):
                        if p in range(a, b + 1):
                            continue
                        if p < a and a - p < 2:
                            continue
                        if p > b and p - b < 2:
                            continue
                        cells = _line_cells(orient, fixed, [a, b, p])
                        if any(cell in occupied for cell in cells):
                            continue
                        if not _counts_ok(var_cells | set(cells[:2])):
                            continue
                        cands.append((orient, fixed, a, b, p, cells))
    return cands


def _blocked_witness_candidates(
    n: int,
    occupied: set[tuple[int, int]],
    var_cells: set[tuple[int, int]],
) -> list[tuple[str, int, int, int, int, list[tuple[int, int]]]]:
    cands = []
    for orient in ("h", "v"):
        for fixed in range(n):
            for a in range(n):
                for b in range(a + 2, n):
                    for p in range(a + 1, b):
                        cells = _line_cells(orient, fixed, [a, b, p])
                        if any(cell in occupied for cell in cells):
                            continue
                        if not _counts_ok(var_cells | set(cells[:2])):
                            continue
                        cands.append((orient, fixed, a, b, p, cells))
    return cands


def build_connector_draw_pair(n: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    v, s = colors
    for _ in range(CONNECTOR_BUILD_RETRIES):
        src = np.zeros((n, n), dtype=int)
        occupied: set[tuple[int, int]] = set()
        var_cells: set[tuple[int, int]] = set()

        clear = _clear_candidates(n, occupied, var_cells)
        if not clear:
            continue
        orient, fixed, a, b, cells = rng.choice(clear)
        for r, c in cells:
            src[r, c] = v
        occupied |= set(cells)
        var_cells |= set(cells)

        witness_kind = "tail" if (n >= 5 and rng.random() < 0.65) else "blocked"
        if witness_kind == "tail":
            witness = _tail_witness_candidates(n, occupied, var_cells)
            if not witness:
                witness_kind = "blocked"
        if witness_kind == "blocked":
            witness = _blocked_witness_candidates(n, occupied, var_cells)
        if not witness:
            continue
        orient, fixed, a, b, p, cells = rng.choice(witness)
        for r, c in cells[:2]:
            src[r, c] = v
        sr, sc = cells[2]
        src[sr, sc] = s
        occupied |= set(cells)
        var_cells |= set(cells[:2])

        if rng.random() < 0.35:
            extra = _clear_candidates(n, occupied, var_cells)
            if extra:
                orient, fixed, a, b, cells = rng.choice(extra)
                for r, c in cells:
                    src[r, c] = v
                occupied |= set(cells)
                var_cells |= set(cells)

        dst = apply_template("connector_draw", {"stable": {s}}, src)
        fill_dst = span_fill_transform(src, {s}, v)
        added = int(np.sum(dst == v) - np.sum(src == v))
        if added < 2:
            continue
        if np.array_equal(dst, src):
            continue
        if np.array_equal(fill_dst, dst):
            continue
        inferred = infer_template(src, dst)
        if inferred is None or inferred[0] != "connector_draw":
            continue
        return src, dst

    raise ValueError("failed to build connector_draw pair")


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
    elif kind == "connector_draw":
        color_count = 2
    else:
        color_count = 1
    return kind, params, color_count


def build_training_pair(kind: str, params: dict, n: int, colors: list[int], rng: random.Random):
    if kind == "endpoints":
        return build_endpoints_pair(n, colors, rng)
    if kind == "connector_draw":
        return build_connector_draw_pair(n, colors, rng)
    if kind == "comp_checker":
        return build_comp_checker_pair(n, colors, rng)
    if kind == "invert0":
        return build_invert0_pair(n, colors, rng)
    if kind == "sym":
        return build_sym_pair(n, colors, rng, params["idx"])
    if kind == "sym_fill":
        return build_sym_fill_pair(n, colors, rng, params["keep0"])
    raise ValueError(kind)


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
    sizes = {size for size, _, *_ in frames}
    if len(sizes) != 1 or next(iter(sizes)) not in set(FRAME_SIZE_CHOICES):
        return False
    inners = [grid[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1 in frames]
    if sum(int(np.all(inner == 0)) for inner in inners) != 1:
        return False
    nonblank_sigs = [tuple(colors_present(inner, exclude={0})) for inner in inners]
    nonblank_sigs = [s for s in nonblank_sigs if s]
    dup_count = sum(
        1 for i in range(len(nonblank_sigs)) for j in range(i + 1, len(nonblank_sigs))
        if nonblank_sigs[i] == nonblank_sigs[j]
    )
    if dup_count != 1:
        return False
    return np.array_equal(main(grid), expected_output)


def _valid_nz(arr: np.ndarray) -> bool:
    return SPRITE_NZ_MIN <= int(np.count_nonzero(arr)) <= SPRITE_NZ_MAX


def _build_sprites(
    kind: str,
    setup_params: dict,
    n: int,
    p1: list[int],
    p2: list[int],
    rng: random.Random,
):
    train_src, train_dst = build_training_pair(kind, setup_params, n, p1, rng)
    inferred = infer_template(train_src, train_dst)
    if inferred is None or inferred[0] != kind:
        return None
    inferred_kind, inferred_params = inferred

    test_src, _ = build_training_pair(kind, setup_params, n, p2, rng)
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
    if kind == "connector_draw":
        st_train = stable_colors(train_src, train_dst)
        v_train = pick_var_color(train_src, st_train)
        st_test = stable_colors(test_src, test_dst)
        v_test = pick_var_color(test_src, st_test)
        if np.array_equal(span_fill_transform(train_src, st_train, v_train), train_dst):
            return None
        if np.array_equal(span_fill_transform(test_src, st_test, v_test), test_dst):
            return None
    return train_src, train_dst, test_src, test_dst


def _choose_frame_and_bg(used: set[int], rng: random.Random) -> tuple[int, list[int]]:
    frame_color = rng.choice([c for c in ALL_COLORS if c not in used])
    bg_pool = [c for c in ALL_COLORS if c not in used and c != frame_color]
    bg_palette = rng.sample(bg_pool, rng.choice(BG_PERIOD_CHOICES))
    return frame_color, bg_palette


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


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    s = rng.choice(FRAME_SIZE_CHOICES)
    n = s - 2

    kind, setup_params, color_count = choose_rule_setup(n, rng)
    p1 = choose_palette(rng, color_count, set())
    p2 = choose_palette(rng, color_count, set(p1))

    built = _build_sprites(kind, setup_params, n, p1, p2, rng)
    if built is None:
        return None
    train_src, train_dst, test_src, test_dst = built

    frame_color, bg_palette = _choose_frame_and_bg(set(p1) | set(p2), rng)
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
