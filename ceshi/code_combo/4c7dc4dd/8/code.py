# Combo variant of 4c7dc4dd (index 8)
# combo_spec: D12/2+D13/1
# Name: 共享色下的角色反转
#
# Design analysis
# ---------------
# concrete_behavior:
#   - Keep the original 30x30 "four framed sprites" scene.
#   - Every non-blank sprite uses exactly two interior colors.
#   - One color is shared by all three non-blank sprites. In the original D13-style
#     reading this is the cross-example carrier/stable color, because it is the only
#     color reused by train and test.
#   - The second color is local: the training pair shares one frozen color, while the
#     test sprite uses a different frozen color. This local color keeps the same mask
#     within each src->dst example.
#   - D12/2 swaps the old roles: the shared carrier color is now the one that
#     transforms, and the local color is the one that stays fixed.
#
# behavior_trace:
#   - Small fill example:
#       train src: shared yellow at four rectangle corners, local red anchor in a corner
#       train dst: yellow fills the rectangle, red anchor unchanged
#       test src:  shared yellow at four other rectangle corners, local blue anchor elsewhere
#       output:    yellow fills its rectangle, blue anchor unchanged
#   - Original 4c7dc4dd logic could identify the pair by equal palettes and then act on
#     the non-stable color. In this combo, the visually repeated train/test color is the
#     moving one, so the solver must separate "shared across examples" from "unchanged
#     within a transform".
#
# complexity_calibration:
#   - The generator only samples {fill, endpoints, invert0}. These three families make
#     the role swap visible without introducing ambiguous whole-sprite symmetries.
#   - The frozen layer is a tiny 3-cell corner motif. That keeps the scene readable and
#     leaves most of the sprite free for the active rule shape.
#   - Train and test use different frozen colors and different frozen motifs, so the
#     repeated shared color is the only reliable cross-example cue.
#
# implicit_assumption_audit:
#   - Original assumption "training pair = exact equal non-zero color set" is no longer
#     used in `main()`. It may still happen accidentally, but solver selection does not
#     depend on it.
#   - Original assumption "shared/reused color is also the unchanged color" is false.
#     Candidate scoring instead prefers pairs where the changed color is the one reused
#     by the test sprite.
#   - Rule application still needs a blocked mask so the frozen local color is preserved.
#     Builders therefore keep frozen cells disjoint from the active rule geometry.
#
# placement_legality:
#   - Frames stay same-size, non-overlapping, and quadrant-distributed on a 30x30 canvas.
#   - Frozen corner motifs are placed strictly inside the sprite interior and are passed
#     as blocked masks into the active-layer builders, so no overlap or out-of-bounds
#     writes occur.
#
# visual_observability_and_cleanliness:
#   - Humans can see one color repeated across all three non-blank sprites, but the
#     training example shows that this repeated color is the one that changes.
#   - The local anchor color is sparse, fixed, and visually separate from the active
#     geometry, so the two roles can be distinguished from only 2-3 examples.
#   - Generator rejection explicitly removes samples where applying the old unswapped
#     interpretation would produce the same test output.

import random

import numpy as np

from arc_utils import colors_present, draw_line, paste


H = W = 30

SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
ALL_COLORS = list(range(10))
BG_PERIOD_CHOICES = (4, 5)
BG_PERTURB_FRACTION = (0.30, 0.50)
BG_EXTRA_NOISE = (0,)

FRAME_SIZE_CHOICES = (6, 7, 8)
QUAD_LOW_MIN, QUAD_LOW_MAX = 1, 8
QUAD_BOTTOM_MARGIN = 8

RULES = ["endpoints", "fill", "invert0"]
RULE_SELECTION_WEIGHTS = {
    "endpoints": 2,
    "fill": 4,
    "invert0": 2,
}

SPRITE_NZ_MIN, SPRITE_NZ_MAX = 3, 32

FILL_RECT_AREA_MIN, FILL_RECT_AREA_MAX = 6, 24
ENDPOINTS_LINE_LEN_MIN = 3
INVERT0_VAR_MAX = 29
GEN_MAX_ATTEMPTS = 500


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


def parse_scene(grid: np.ndarray):
    frames = find_square_frames(grid)
    if len(frames) < 3:
        return None

    inners = []
    for size, col, r0, c0, r1, c1 in frames:
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        inners.append(((r0, c0, r1, c1), inner))

    blank_idx = None
    for i, (_, inner) in enumerate(inners):
        if inner.size > 0 and np.all(inner == 0):
            blank_idx = i
            break
    if blank_idx is None:
        blank_idx = int(np.argmin([np.count_nonzero(inner) for _, inner in inners]))

    nonblank = [i for i in range(len(inners)) if i != blank_idx]
    if len(nonblank) < 2:
        return None

    return {
        "frames": frames,
        "inners": inners,
        "blank_idx": blank_idx,
        "nonblank": nonblank,
    }


def nonzero_colors(arr: np.ndarray, exclude: set[int] | None = None) -> tuple[int, ...]:
    ex = {0}
    if exclude is not None:
        ex |= set(exclude)
    return tuple(sorted(int(v) for v in np.unique(arr) if int(v) not in ex))


def unchanged_colors(src: np.ndarray, dst: np.ndarray) -> set[int]:
    out = set()
    for c in set(int(x) for x in np.unique(src)):
        if c == 0:
            continue
        if np.array_equal(src == c, dst == c):
            out.add(c)
    return out


def pick_active_color(x: np.ndarray, frozen: set[int]) -> int | None:
    vals = [int(v) for v in np.unique(x) if int(v) != 0 and int(v) not in frozen]
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
    frozen = unchanged_colors(src, dst)
    active = pick_active_color(dst, frozen) or pick_active_color(src, frozen)

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
                return ("comp_checker", {"unchanged": frozen})

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
                    return ("sym_fill", {"unchanged": frozen, "keep0": keep0})

    if active is not None:
        pred = src.copy()
        pred[src == 0] = active
        pred[src == active] = 0
        if np.array_equal(pred, dst):
            return ("invert0", {"unchanged": frozen})

    for idx, var in enumerate(symmetries(src)):
        if np.array_equal(var, dst):
            return ("sym", {"idx": idx})

    if active is not None:
        coords = np.argwhere(src == active)
        active_set = set((int(r), int(c)) for r, c in coords)
        keep = np.zeros_like(src, dtype=bool)
        for r, c in active_set:
            deg = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    if (r + dr, c + dc) in active_set:
                        deg += 1
            if deg <= 1:
                keep[r, c] = True
        pred = src.copy()
        pred[src == active] = 0
        pred[keep] = active
        if np.array_equal(pred, dst):
            return ("endpoints", {"unchanged": frozen})

        pred = src.copy()
        h, w = pred.shape
        frozen_mask = np.zeros_like(pred, dtype=bool)
        for c in frozen:
            frozen_mask |= pred == c
        for r in range(h):
            nz = np.where(pred[r, :] != 0)[0]
            if nz.size >= 2:
                cmin = int(nz.min())
                cmax = int(nz.max())
                for cc in range(cmin, cmax + 1):
                    if pred[r, cc] == 0 and not frozen_mask[r, cc]:
                        pred[r, cc] = active
        for c in range(w):
            nz = np.where(pred[:, c] != 0)[0]
            if nz.size >= 2:
                rmin = int(nz.min())
                rmax = int(nz.max())
                for rr in range(rmin, rmax + 1):
                    if pred[rr, c] == 0 and not frozen_mask[rr, c]:
                        pred[rr, c] = active
        if np.array_equal(pred, dst):
            return ("fill", {"unchanged": frozen})

    return None


def apply_kind_to_color(
    kind: str,
    params: dict,
    x: np.ndarray,
    active_color: int | None,
    frozen_colors: set[int] | None = None,
) -> np.ndarray:
    if frozen_colors is None:
        frozen_colors = set()

    if kind == "sym":
        return symmetries(x)[params["idx"]].copy()

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

    if active_color is None:
        return x.copy()

    if kind == "invert0":
        y = x.copy()
        y[x == 0] = active_color
        y[x == active_color] = 0
        return y

    if kind == "endpoints":
        coords = np.argwhere(x == active_color)
        active_set = set((int(r), int(c)) for r, c in coords)
        keep = np.zeros_like(x, dtype=bool)
        for r, c in active_set:
            deg = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    if (r + dr, c + dc) in active_set:
                        deg += 1
            if deg <= 1:
                keep[r, c] = True
        y = x.copy()
        y[x == active_color] = 0
        y[keep] = active_color
        return y

    if kind == "fill":
        y = x.copy()
        h, w = y.shape
        frozen_mask = np.zeros_like(y, dtype=bool)
        for c in frozen_colors:
            frozen_mask |= y == c
        for r in range(h):
            nz = np.where(y[r, :] != 0)[0]
            if nz.size >= 2:
                cmin = int(nz.min())
                cmax = int(nz.max())
                for cc in range(cmin, cmax + 1):
                    if y[r, cc] == 0 and not frozen_mask[r, cc]:
                        y[r, cc] = active_color
        for c in range(w):
            nz = np.where(y[:, c] != 0)[0]
            if nz.size >= 2:
                rmin = int(nz.min())
                rmax = int(nz.max())
                for rr in range(rmin, rmax + 1):
                    if y[rr, c] == 0 and not frozen_mask[rr, c]:
                        y[rr, c] = active_color
        return y

    return x.copy()


def apply_template(kind: str, params: dict, x: np.ndarray) -> np.ndarray:
    frozen = set(params.get("unchanged", set()))
    active = pick_active_color(x, frozen)
    return apply_kind_to_color(kind, params, x, active, frozen)


def candidate_score(
    src: np.ndarray,
    dst: np.ndarray,
    test: np.ndarray,
    pred: np.ndarray,
    kind: str,
    params: dict,
) -> int:
    frozen = set(params.get("unchanged", set()))
    if len(frozen) != 1:
        return -100_000

    active = pick_active_color(src, frozen) or pick_active_color(dst, frozen)
    if active is None:
        return -100_000

    src_cols = set(nonzero_colors(src))
    dst_cols = set(nonzero_colors(dst))
    test_cols = set(nonzero_colors(test))
    pred_cols = set(nonzero_colors(pred))
    frozen_color = next(iter(frozen))

    if active not in src_cols or active not in dst_cols:
        return -100_000

    score = 1000
    if len(src_cols) == 2 and len(dst_cols) == 2 and len(test_cols) == 2:
        score += 150
    if active in test_cols:
        score += 280
    else:
        score -= 600
    if frozen_color not in test_cols:
        score += 220
    else:
        score -= 600
    if active in pred_cols and frozen_color not in pred_cols:
        score += 120
    if not np.array_equal(pred, test):
        score += 80

    src_active_n = int(np.sum(src == active))
    dst_active_n = int(np.sum(dst == active))
    test_active_n = int(np.sum(test == active))
    pred_active_n = int(np.sum(pred == active))
    score -= abs(src_active_n - test_active_n)

    if kind == "fill" and dst_active_n > src_active_n and pred_active_n >= test_active_n:
        score += 40
    if kind == "endpoints" and dst_active_n < src_active_n and pred_active_n <= test_active_n:
        score += 40
    if kind == "invert0" and dst_active_n != src_active_n:
        score += 20

    return score


def main(grid: np.ndarray) -> np.ndarray:
    """
    Solve a framed-sprite analogy where the color shared across train and test is the one that moves.

    Input structure:
    - The 30x30 grid contains four same-size hollow square frames on a noisy background, with one blank interior.
    - Every non-blank interior uses exactly two colors: one color is reused across all three non-blank sprites, and each example also has its own sparse companion color.
    - In the training pair, the reused color changes shape from source to destination while the companion color keeps the same cells.

    Rule:
    - Enumerate candidate training pairs among the non-blank frames instead of assuming palette equality is enough.
    - For each directed candidate, identify the unchanged local color, infer the geometric rule acting on the other color, and score candidates by preferring the interpretation where the changing color is also the color reused by the test sprite.
    - Apply that same rule to the reused color in the test sprite, leaving the test's own companion color untouched.

    Key insight:
    - Cross-sprite sharing and within-pair stability no longer point to the same color: the repeated color across frames is the active one, so the solver must first separate color roles before applying the geometry rule.
    """
    g = np.asarray(grid, dtype=int)
    parsed = parse_scene(g)
    if parsed is None:
        return g.copy()

    inners = [inner for _, inner in parsed["inners"]]
    nonblank = parsed["nonblank"]

    if len(nonblank) < 2:
        return inners[parsed["blank_idx"]].copy()

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


def choose_palette(rng: random.Random) -> tuple[int, int, int]:
    shared, frozen_train, frozen_test = rng.sample(SPRITE_COLORS, 3)
    return shared, frozen_train, frozen_test


def frozen_motif_masks(n: int) -> list[np.ndarray]:
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


def choose_frozen_masks(n: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    motifs = frozen_motif_masks(n)
    train_idx, test_idx = rng.sample(range(len(motifs)), 2)
    return motifs[train_idx], motifs[test_idx]


def overlay_frozen(active_layer: np.ndarray, mask: np.ndarray, color: int) -> np.ndarray:
    out = active_layer.copy()
    out[mask] = color
    return out


def free_cells(blocked: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in zip(*np.where(~blocked))]


def build_fill_pair(n: int, color: int, blocked: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(120):
        while True:
            h = rng.randint(2, n)
            w = rng.randint(2, n)
            area = h * w
            if FILL_RECT_AREA_MIN <= area <= FILL_RECT_AREA_MAX:
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


def choose_rule_setup(rng: random.Random) -> str:
    weights = [RULE_SELECTION_WEIGHTS[k] for k in RULES]
    return rng.choices(RULES, weights=weights, k=1)[0]


def build_training_pair(kind: str, n: int, color: int, blocked: np.ndarray, rng: random.Random):
    if kind == "fill":
        return build_fill_pair(n, color, blocked, rng)
    if kind == "endpoints":
        return build_endpoints_pair(n, color, blocked, rng)
    if kind == "invert0":
        return build_invert0_pair(n, color, blocked, rng)
    raise ValueError(kind)


def _valid_nz(arr: np.ndarray) -> bool:
    return SPRITE_NZ_MIN <= int(np.count_nonzero(arr)) <= SPRITE_NZ_MAX


def _counterfactual_old_role(kind: str, params: dict, x: np.ndarray, shared_color: int, frozen_color: int) -> np.ndarray:
    return apply_kind_to_color(kind, params, x, frozen_color, {shared_color})


def _build_sprites(
    kind: str,
    n: int,
    shared_color: int,
    frozen_train_color: int,
    frozen_test_color: int,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    rng: random.Random,
):
    train_src_var, train_dst_var = build_training_pair(kind, n, shared_color, train_mask, rng)
    train_src = overlay_frozen(train_src_var, train_mask, frozen_train_color)
    train_dst = overlay_frozen(train_dst_var, train_mask, frozen_train_color)

    inferred = infer_template(train_src, train_dst)
    if inferred is None or inferred[0] != kind:
        return None
    inferred_kind, inferred_params = inferred
    frozen = set(inferred_params.get("unchanged", set()))
    if frozen != {frozen_train_color}:
        return None
    if pick_active_color(train_src, frozen) != shared_color:
        return None

    test_src_var, _ = build_training_pair(kind, n, shared_color, test_mask, rng)
    test_src = overlay_frozen(test_src_var, test_mask, frozen_test_color)
    test_dst = apply_template(inferred_kind, inferred_params, test_src)

    if np.array_equal(test_dst, test_src):
        return None
    if not np.array_equal(test_src == frozen_test_color, test_dst == frozen_test_color):
        return None
    if int(np.sum(test_src == shared_color)) == int(np.sum(test_dst == shared_color)):
        return None
    if np.array_equal(_counterfactual_old_role(inferred_kind, inferred_params, train_src, shared_color, frozen_train_color), train_dst):
        return None
    if np.array_equal(_counterfactual_old_role(inferred_kind, inferred_params, test_src, shared_color, frozen_test_color), test_dst):
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
    if shared_color not in train_sig or shared_color not in test_sig:
        return None
    if frozen_train_color not in train_sig or frozen_test_color not in test_sig:
        return None
    if frozen_train_color in test_sig or frozen_test_color in train_sig:
        return None
    if np.array_equal(train_mask, test_mask):
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
    parsed = parse_scene(grid)
    if parsed is None:
        return False
    frames = parsed["frames"]
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

    common = set(nonzero_colors(nonblank[0]))
    for inner in nonblank[1:]:
        common &= set(nonzero_colors(inner))
    if len(common) != 1:
        return False

    if not np.array_equal(main(grid), expected_output):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    s = rng.choice(FRAME_SIZE_CHOICES)
    n = s - 2

    shared_color, frozen_train_color, frozen_test_color = choose_palette(rng)
    train_mask, test_mask = choose_frozen_masks(n, rng)
    kind = choose_rule_setup(rng)

    built = _build_sprites(
        kind,
        n,
        shared_color,
        frozen_train_color,
        frozen_test_color,
        train_mask,
        test_mask,
        rng,
    )
    if built is None:
        return None
    train_src, train_dst, test_src, test_dst = built

    frame_color, bg_palette = _choose_frame_and_bg({shared_color, frozen_train_color, frozen_test_color}, rng)
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
