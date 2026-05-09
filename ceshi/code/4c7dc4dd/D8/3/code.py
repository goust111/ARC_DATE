# Variant of 4c7dc4dd
# Dimension: D8 — 框架数量与角色分配 (meta)
# Change: 恰好4个框架：2个训练对 + 1个测试精灵 + 1个空白... → 带干扰框架

# Design analysis for the D8 variant
# a) Concrete behavior and trace:
#    - The scene now has five same-size hollow square frames.
#    - Four frames still form the real analogy: one per quadrant, containing
#      train source, train target, test source, and one blank interior.
#    - A fifth frame is placed near the canvas center and contains an unrelated
#      random sprite. It is visually inside the frame family but outside the
#      2x2 analogy layout.
#    - Old flow: detect frames -> find blank -> find the matching training pair
#      -> the only remaining non-blank frame is the test.
#    - New flow: detect frames -> keep the four corner-most frames (one per
#      quadrant) -> ignore the central frame -> run the original blank/pair/test
#      logic on the retained quartet. The behavioral difference is exactly at
#      role assignment: "remaining non-blank" is no longer unique until the
#      distractor is filtered out.
# b) Complexity calibration:
#    - The distractor is not encoded via color ids or hidden metadata. It is
#      visible because the real puzzle still occupies the four outer quadrants,
#      while the extra frame sits centrally and does not fit the analogy grid.
#    - This keeps the change obvious to humans without multiplying the visual
#      rule space; the sprite transformation itself stays in the original rule set.
# c) Implicit assumptions from the old solver that must change:
#    - "All detected frames belong to the puzzle" no longer holds.
#    - "After removing the blank and the matched pair, exactly one non-blank
#      frame remains" no longer holds on the full five-frame set.
#    - The fix is to select the four corner frames first. The old inference and
#      application logic can then remain unchanged.
# d) Placement legality:
#    - The generator reserves four jittered corner anchors plus one fixed center
#      anchor. With frame sizes 6-8 on a 30x30 canvas, these five boxes are kept
#      non-overlapping and in-bounds by construction.
#    - The training/test/blank roles are assigned only among the four corner slots.
# e) Visual observability:
#    - A human sees the familiar 2x2 analogy plus one extra frame near the middle.
#      The intended action is to ignore the intruder and solve the outer quartet.
#    - No color ordering or numeric color semantics are used for role selection.
# f) Visual cleanliness:
#    - Only one extra layer is added: a central frame, separated from the four
#      corner frames. Nothing overlaps, and the distractor is structurally clear.
# g) Color inferability:
#    - Color-based rule inference is unchanged from the base task. The distractor
#      uses a palette disjoint from both train and test palettes so it cannot
#      masquerade as a training-pair candidate.
# h) Variant-specific metric:
#    - A valid generated instance must contain exactly five frames and the ignored
#      frame must be the unique one closest to the grid center; this confirms the
#      distractor branch is active rather than degenerating back to the 4-frame task.
# i) New solver assumptions audited:
#    - The retained four frames must occupy four distinct quadrants relative to the
#      grid center; the generator guarantees this.
#    - The center distractor must be nearer to the grid center than any corner
#      frame; the generator guarantees this by construction.
#    - Among the retained four frames, there is still exactly one blank interior
#      and exactly one matching non-zero color-set pair; the generator verifies both.

from itertools import combinations
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
        if len(picked) >= 8:
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
        if v == 0:
            continue
        out[v] = out.get(v, 0) + 1
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


def frame_center(frame: tuple[int, int, int, int, int, int]) -> tuple[float, float]:
    _, _, r0, c0, r1, c1 = frame
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def frame_center_distance(frame: tuple[int, int, int, int, int, int], h: int, w: int) -> float:
    cr, cc = frame_center(frame)
    mr = (h - 1) / 2.0
    mc = (w - 1) / 2.0
    return abs(cr - mr) + abs(cc - mc)


def select_core_frame_indices(
    frames: list[tuple[int, int, int, int, int, int]],
    h: int,
    w: int,
) -> list[int]:
    if len(frames) <= 4:
        return sorted(range(len(frames)), key=lambda i: (frames[i][2], frames[i][3]))

    mr = (h - 1) / 2.0
    mc = (w - 1) / 2.0
    best = None
    for subset in combinations(range(len(frames)), 4):
        quads = set()
        dist_sum = 0.0
        min_dist = None
        ok = True
        for idx in subset:
            cr, cc = frame_center(frames[idx])
            quad = (0 if cr < mr else 1, 0 if cc < mc else 1)
            if quad in quads:
                ok = False
                break
            quads.add(quad)
            dist = abs(cr - mr) + abs(cc - mc)
            dist_sum += dist
            min_dist = dist if min_dist is None else min(min_dist, dist)
        if not ok or len(quads) != 4:
            continue
        score = (dist_sum, min_dist if min_dist is not None else -1.0)
        if best is None or score > best[0]:
            best = (score, subset)

    if best is not None:
        subset = best[1]
    else:
        ranked = sorted(
            range(len(frames)),
            key=lambda i: frame_center_distance(frames[i], h, w),
            reverse=True,
        )
        subset = ranked[:4]
    return sorted(subset, key=lambda i: (frames[i][2], frames[i][3]))


def extract_scene_frames(
    grid: np.ndarray,
) -> tuple[list[tuple[int, int, int, int, int, int]], list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    frames = find_square_frames(g)
    if len(frames) < 4:
        return frames, [], []
    size_counts: dict[int, int] = {}
    for size, _, _, _, _, _ in frames:
        size_counts[size] = size_counts.get(size, 0) + 1
    best_size = max(size_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
    same_size = [frame for frame in frames if frame[0] == best_size]
    core = select_core_frame_indices(same_size, g.shape[0], g.shape[1])
    ignored = [i for i in range(len(same_size)) if i not in core]
    return same_size, core, ignored


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the sprite rule from the real analogy frames while ignoring one extra decoy frame.

    Input structure:
    - Grid contains five hollow square frames of the same size.
    - Four outer frames, one in each quadrant, are the true puzzle: a training source, a training target, a test sprite, and one blank interior.
    - A fifth frame sits near the middle and contains an unrelated distractor sprite.

    Rule:
    - First separate the four corner frames from the central distractor.
    - Among those four corner frames, find the training pair by matching non-zero interior color sets; the remaining non-blank corner frame is the test input.
    - Infer the same visual transformation as in the base task and apply it to the test sprite.

    Key insight:
    - The extra frame is not another analogy role. The real puzzle is still the outer 2x2 "A:B :: C:?" arrangement, so layout filtering comes before rule induction.
    """
    g = np.asarray(grid, dtype=int)
    frames, core_idxs, _ignored = extract_scene_frames(g)
    if len(core_idxs) < 4:
        return g.copy()

    core_frames = [frames[i] for i in core_idxs]
    inners = []
    for _, _, r0, c0, r1, c1 in core_frames:
        inners.append(g[r0 + 1 : r1, c0 + 1 : c1])

    blank_idx = None
    for i, inner in enumerate(inners):
        if inner.size > 0 and np.all(inner == 0):
            blank_idx = i
            break
    if blank_idx is None:
        blank_idx = int(np.argmin([np.count_nonzero(inner) for inner in inners]))

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
        counts = [(i, int(np.count_nonzero(inners[i]))) for i in nonblank]
        counts.sort(key=lambda t: t[1])
        pair = (counts[0][0], counts[1][0])

    remaining = [i for i in nonblank if i not in pair]
    if not remaining:
        return inners[blank_idx].copy()
    test_idx = remaining[0]

    src_a = inners[pair[0]]
    src_b = inners[pair[1]]
    test_inner = inners[test_idx]

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
        score2 = score * 100 - abs(src_n - test_n)
        if best is None or score2 > best[0]:
            best = (score2, kind, params)

    if best is None:
        return test_inner.copy()

    _, kind, params = best
    return apply_template(kind, params, test_inner)


# --------------- generator ---------------

H = W = 30

SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
ALL_COLORS = list(range(10))
BG_PERIOD_CHOICES = (4, 5)
BG_PERTURB_FRACTION = (0.30, 0.50)
BG_EXTRA_NOISE = (0,)

FRAME_SIZE_CHOICES = (6, 7, 8)
CORNER_LOW_MIN, CORNER_LOW_MAX = 1, 4
CORNER_MARGIN = 1

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


def corner_positions(s: int, rng: random.Random) -> list[tuple[int, int]]:
    top_low = CORNER_LOW_MIN
    top_high = CORNER_LOW_MAX
    bottom_low = H - s - CORNER_LOW_MAX - CORNER_MARGIN
    bottom_high = H - s - CORNER_LOW_MIN - CORNER_MARGIN
    left_low = CORNER_LOW_MIN
    left_high = CORNER_LOW_MAX
    right_low = W - s - CORNER_LOW_MAX - CORNER_MARGIN
    right_high = W - s - CORNER_LOW_MIN - CORNER_MARGIN
    return [
        (rng.randint(top_low, top_high), rng.randint(left_low, left_high)),
        (rng.randint(top_low, top_high), rng.randint(right_low, right_high)),
        (rng.randint(bottom_low, bottom_high), rng.randint(left_low, left_high)),
        (rng.randint(bottom_low, bottom_high), rng.randint(right_low, right_high)),
    ]


def center_position(s: int) -> tuple[int, int]:
    return ((H - s) // 2, (W - s) // 2)


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


def _valid_nz(arr: np.ndarray) -> bool:
    return SPRITE_NZ_MIN <= int(np.count_nonzero(arr)) <= SPRITE_NZ_MAX


def _build_sprites(
    kind: str,
    setup_params: dict,
    n: int,
    p1: list[int],
    p2: list[int],
    distractor_palette: list[int],
    rng: random.Random,
):
    train_src, train_dst = build_training_pair(kind, setup_params, n, p1, rng)
    inferred = infer_template(train_src, train_dst)
    if inferred is None or inferred[0] != kind:
        return None
    inferred_kind, inferred_params = inferred

    test_src, _test_pair_dst = build_training_pair(kind, setup_params, n, p2, rng)
    test_dst = apply_template(inferred_kind, inferred_params, test_src)
    distractor = random_sparse_pattern(n, distractor_palette, rng)

    if np.array_equal(test_dst, test_src):
        return None
    for arr in (train_src, train_dst, test_src, test_dst, distractor):
        if not _valid_nz(arr):
            return None
    train_sig = tuple(colors_present(train_src, exclude={0}))
    if train_sig != tuple(colors_present(train_dst, exclude={0})):
        return None
    if tuple(colors_present(test_src, exclude={0})) == train_sig:
        return None
    distractor_sig = tuple(colors_present(distractor, exclude={0}))
    if distractor_sig in {
        train_sig,
        tuple(colors_present(test_src, exclude={0})),
        tuple(colors_present(test_dst, exclude={0})),
    }:
        return None
    return train_src, train_dst, test_src, test_dst, distractor


def _choose_frame_and_bg(used: set[int], rng: random.Random) -> tuple[int, list[int]]:
    frame_color = rng.choice([c for c in ALL_COLORS if c not in used])
    bg_pool = [c for c in ALL_COLORS if c not in used and c != frame_color]
    bg_size_choices = [k for k in BG_PERIOD_CHOICES if k <= len(bg_pool)]
    if not bg_size_choices:
        raise ValueError("not enough colors left for background palette")
    bg_palette = rng.sample(bg_pool, rng.choice(bg_size_choices))
    return frame_color, bg_palette


def _assign_corner_slots(rng: random.Random) -> tuple[list[int], int]:
    corner_slots = [0, 1, 2, 3]
    train_slots = sorted(rng.sample(corner_slots, 2))
    remaining = [i for i in corner_slots if i not in train_slots]
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
    frames, core, ignored = extract_scene_frames(grid)
    if len(frames) != 5 or len(core) != 4 or len(ignored) != 1:
        return False
    sizes = {size for size, _, _, _, _, _ in frames}
    if len(sizes) != 1 or next(iter(sizes)) not in set(FRAME_SIZE_CHOICES):
        return False

    core_frames = [frames[i] for i in core]
    ignored_frame = frames[ignored[0]]
    ignored_dist = frame_center_distance(ignored_frame, H, W)
    core_dists = [frame_center_distance(frame, H, W) for frame in core_frames]
    if not all(ignored_dist < dist for dist in core_dists):
        return False

    inners = [grid[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1 in core_frames]
    if sum(int(np.all(inner == 0)) for inner in inners) != 1:
        return False
    nonblank_sigs = [tuple(colors_present(inner, exclude={0})) for inner in inners]
    nonblank_sigs = [s for s in nonblank_sigs if s]
    dup_count = sum(
        1
        for i in range(len(nonblank_sigs))
        for j in range(i + 1, len(nonblank_sigs))
        if nonblank_sigs[i] == nonblank_sigs[j]
    )
    if dup_count != 1:
        return False
    return np.array_equal(main(grid), expected_output)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    s = rng.choice(FRAME_SIZE_CHOICES)
    n = s - 2

    kind, setup_params, color_count = choose_rule_setup(n, rng)
    p1 = choose_palette(rng, color_count, set())
    p2 = choose_palette(rng, color_count, set(p1))
    used_sprite_colors = set(p1) | set(p2)
    max_distractor_colors = min(2, max(1, 5 - len(used_sprite_colors)))
    distractor_palette = choose_palette(rng, rng.randint(1, max_distractor_colors), used_sprite_colors)

    built = _build_sprites(kind, setup_params, n, p1, p2, distractor_palette, rng)
    if built is None:
        return None
    train_src, train_dst, test_src, test_dst, distractor = built

    frame_color, bg_palette = _choose_frame_and_bg(set(p1) | set(p2) | set(distractor_palette), rng)

    positions = corner_positions(s, rng) + [center_position(s)]
    train_slots, test_slot = _assign_corner_slots(rng)

    sprites = [np.zeros((n, n), dtype=int) for _ in range(5)]
    sprites[train_slots[0]] = train_src
    sprites[train_slots[1]] = train_dst
    sprites[test_slot] = test_src
    sprites[4] = distractor

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
