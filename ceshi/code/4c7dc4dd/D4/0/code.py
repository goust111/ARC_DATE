from __future__ import annotations

# Variant of 4c7dc4dd
# Dimension: D4 — 目标输出形式 (dst)
# Change: 输出为测试框架内部 (S-2)x(S-2) 的紧凑裁剪区域... → 填充空白框架
#
# Design analysis:
#   variant:
#     core_change:
#       old_behavior: solver returned only the transformed test interior as a compact n x n crop.
#       new_behavior: solver must return the full 30x30 scene and write the transformed test interior
#         into the unique blank frame's interior.
#     concrete_pixel_form:
#       input still has 4 same-size hollow square frames on a noisy background.
#       2 non-blank frames are the training analogy pair, 1 non-blank frame is the test source,
#       and 1 frame is an empty destination container with an all-zero interior.
#       output keeps the entire scene unchanged except that the blank frame interior is replaced by
#       the inferred result for the test sprite.
#     behavior_trace:
#       original_flow:
#         - locate blank frame only to exclude it from training/test role selection
#         - infer the source->destination rule from the matching-color training pair
#         - return apply_template(rule, test_inner)
#       variant_flow:
#         - locate the same blank frame, but retain its coordinates as the destination carrier
#         - infer the rule in the same way
#         - compute transformed_test = apply_template(rule, test_inner)
#         - paste transformed_test into the blank frame interior and return the whole grid
#       visible_difference:
#         - original outputs are 4x4..6x6 crops
#         - variant outputs are always 30x30 and force the solver to identify where the answer belongs
#     complexity_calibration:
#       the rule family stays unchanged so the puzzle remains an analogy task rather than becoming
#       a new object-recognition task. The extra difficulty comes only from using the blank frame
#       as a destination placeholder, similar in spirit to hole_fill, but without introducing
#       additional matching dimensions.
#     implicit_assumptions_review:
#       - the original `return test_inner.copy()` fallback assumed the output carrier was just the
#         transformed sprite. In the variant, every exit path must return a full-grid object.
#       - the original final return dropped all frame locations after inference. In the variant,
#         main must keep the blank frame bbox so the result can be written back at the correct spot.
#       - pasting only non-zero cells would be harmless in the old task. Here we clear the blank
#         interior before pasting so zero cells in the predicted sprite remain visibly blank.
#       - shape compatibility between test and blank interiors is now required for safe pasting;
#         the generator guarantees same-size frames.
#     placement_legality:
#       generator still uses 4 non-overlapping same-size frames, one per quadrant, so the blank
#       container is always easy to localize and never overlaps the training/test frames.
#     observability:
#       the blank frame is a strong visual cue. Across examples, the answer appears inside the
#       previously empty frame, making the D4 change obvious even though the internal transform
#       rule may vary by sample.
#     visual_cleanliness:
#       only one region changes between input and output: the blank frame interior. Background,
#       frame borders, and the three informative sprites stay fixed, keeping the scene readable.
#     color_inference:
#       no new color mapping is introduced. The blank frame simply hosts the transformed test sprite,
#       so all color reasoning still comes from the original training analogy.
#     generator_guarantees_for_new_main:
#       - exactly one frame interior is all zero in the input
#       - all 4 frames share the same size, so transformed_test fits the blank frame interior
#       - expected output differs from input only inside that blank interior

import numpy as np
import random
from arc_utils import colors_present, draw_line, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a sprite transformation from a framed training pair, then place the answer in the empty frame.

    Input structure:
    - The grid contains several non-overlapping hollow square frames of equal size.
    - Two non-blank frame interiors form a training pair because they share the same non-zero color set.
    - A third non-blank frame is the test sprite, and one frame is an all-zero blank destination box.

    Rule:
    - Detect the training pair by matching interior palettes, leaving the other non-blank frame as the test input.
    - Infer the interior transformation demonstrated by that pair (reflection, fill, endpoint extraction, inversion, checker complement, or symmetry fill).
    - Apply the same transformation to the test sprite and write the result into the blank frame's interior, returning the full grid.

    Key insight: This is still an analogy puzzle, but the empty frame is now the answer canvas. Solving requires both discovering the rule and identifying where the completed result belongs.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape

    def find_square_frames():
        cands = []
        for size in range(4, min(H, W) + 1):
            for r0 in range(0, H - size + 1):
                r1 = r0 + size - 1
                for c0 in range(0, W - size + 1):
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
        for size, col, r0, c0, r1, c1 in cands:
            overlaps = False
            for _, _, pr0, pc0, pr1, pc1 in picked:
                if not (r1 < pr0 or pr1 < r0 or c1 < pc0 or pc1 < c0):
                    overlaps = True
                    break
            if not overlaps:
                picked.append((size, col, r0, c0, r1, c1))
            if len(picked) >= 6:
                break
        return picked

    frames = find_square_frames()
    if len(frames) < 3:
        return g.copy()

    inners = []
    for size, col, r0, c0, r1, c1 in frames:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        inners.append(((r0, c0, r1, c1), inner))

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
        return g.copy()
    test_idx = remaining[0]

    blank_box, blank_inner = inners[blank_idx]
    src_a = inners[pair[0]][1]
    src_b = inners[pair[1]][1]
    test_inner = inners[test_idx][1]
    if blank_inner.shape != test_inner.shape:
        return g.copy()

    def stable_colors(src, dst):
        st = set()
        for c in set(int(x) for x in np.unique(src)):
            if c == 0:
                continue
            if np.array_equal(src == c, dst == c):
                st.add(c)
        return st

    def pick_var_color(x, st):
        vals = [int(v) for v in np.unique(x) if int(v) != 0 and int(v) not in st]
        if not vals:
            return None
        best = None
        for v in vals:
            cnt = int(np.sum(x == v))
            if best is None or cnt > best[0]:
                best = (cnt, v)
        return best[1]

    def symmetries(arr):
        outs = []
        for k in range(4):
            r = np.rot90(arr, k)
            outs.append(r)
            outs.append(np.fliplr(r))
        return outs

    def infer_template(src, dst):
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

        def sym_fill_base(a):
            y = np.zeros_like(a)
            for t in (lambda z: z, np.fliplr, np.flipud, lambda z: np.flipud(np.fliplr(z))):
                zz = t(a)
                rr, cc = np.where(zz != 0)
                for r, c in zip(rr.tolist(), cc.tolist()):
                    val = int(zz[r, c])
                    if y[r, c] != 0 and int(y[r, c]) != val:
                        return None
                    y[r, c] = val
            return y

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
            S = set((int(r), int(c)) for r, c in coords)
            keep = np.zeros_like(src, dtype=bool)
            for r, c in S:
                deg = 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        if (r + dr, c + dc) in S:
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

    def apply_template(kind, params, x):
        if kind == "sym":
            return symmetries(x)[params["idx"]].copy()

        st = set(params.get("stable", set()))
        v = pick_var_color(x, st)

        if kind == "comp_checker":
            nz = np.argwhere(x != 0)
            even_counts = {}
            odd_counts = {}
            for r, c in nz:
                col = int(x[int(r), int(c)])
                if ((int(r) + int(c)) & 1) == 0:
                    even_counts[col] = even_counts.get(col, 0) + 1
                else:
                    odd_counts[col] = odd_counts.get(col, 0) + 1
            if even_counts:
                c_even = max(even_counts.items(), key=lambda t: t[1])[0]
            elif odd_counts:
                c_even = max(odd_counts.items(), key=lambda t: t[1])[0]
            else:
                return x.copy()
            if odd_counts:
                c_odd = max(odd_counts.items(), key=lambda t: t[1])[0]
            else:
                c_odd = c_even
            h, w = x.shape
            y = np.zeros_like(x)
            for r in range(h):
                for c in range(w):
                    if x[r, c] != 0:
                        y[r, c] = 0
                    else:
                        y[r, c] = c_even if ((r + c) & 1) == 0 else c_odd
            return y

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

            def sym_fill_base(a):
                y = np.zeros_like(a)
                for t in (lambda z: z, np.fliplr, np.flipud, lambda z: np.flipud(np.fliplr(z))):
                    zz = t(a)
                    rr, cc = np.where(zz != 0)
                    for r, c in zip(rr.tolist(), cc.tolist()):
                        val = int(zz[r, c])
                        if y[r, c] != 0 and int(y[r, c]) != val:
                            return a.copy()
                        y[r, c] = val
                return y

            y = sym_fill_base(x)
            keep0 = params.get("keep0", None)
            if keep0 is None or keep0.shape != y.shape:
                keep0 = np.zeros_like(y, dtype=bool)
            for r in range(h):
                for c in range(w):
                    if y[r, c] == 0 and not bool(keep0[r, c]):
                        y[r, c] = fill_col
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
            S = set((int(r), int(c)) for r, c in coords)
            keep = np.zeros_like(x, dtype=bool)
            for r, c in S:
                deg = 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        if (r + dr, c + dc) in S:
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
        return g.copy()

    _, kind, params = best
    transformed = apply_template(kind, params, test_inner)
    return write_into_blank_frame(g, blank_box, transformed)

# --------------- generator (attempt 2 (Codex)) ---------------

# ------- CFG: constants governing sampling distribution -------
# Canvas
H = W = 30

# Palette pools
SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]       # colors usable inside sprite interiors
ALL_COLORS = list(range(10))
BG_PERIOD_CHOICES = (4, 5)                  # diagonal-stripe period for background
BG_PERTURB_FRACTION = (0.30, 0.50)          # min/max fraction of bg cells perturbed to noise
BG_EXTRA_NOISE = (0,)                       # extra noise digits beyond bg palette

# Frame layout
FRAME_SIZE_CHOICES = (6, 7, 8)              # outer side S (inner = S-2 -> 4/5/6)
QUAD_LOW_MIN, QUAD_LOW_MAX = 1, 8            # top/left anchor range inside a quadrant
QUAD_BOTTOM_MARGIN = 8                       # bottom/right anchor range is [H-s-margin, H-s-1]

# Rule set
RULES = ["endpoints", "fill", "sym_fill", "comp_checker", "invert0", "sym"]

# Sprite pixel-count bounds (non-zero cells in every sprite interior)
SPRITE_NZ_MIN, SPRITE_NZ_MAX = 3, 32

# sym_fill: probability to reserve the (0,0) orbit as forced-zero
SYMFILL_KEEP0_PROB = 0.4                     # ignored when n==6 (always True then)

# Rule-specific build bounds
FILL_RECT_AREA_MIN, FILL_RECT_AREA_MAX = 5, 25   # inclusive area bounds (h*w)
ENDPOINTS_LINE_LEN_MIN = 3                   # each row line's min length
COMP_CHECKER_M_MAX = 10                      # max holes carved from the checker
INVERT0_STABLE_MAX = 3                       # upper bound on stable cells in invert0 src
INVERT0_VAR_MAX = 12                         # upper bound on variable cells
SYM_BUILD_RETRIES = 30                       # attempts to find a non-identity sym pair
SPARSE_COUNT_MAX = 10                        # max cells for random_sparse_pattern

# Outer retry loop
GEN_MAX_ATTEMPTS = 300


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
    """All 8 dihedral symmetries of arr (idx 0 = identity, 1 = fliplr, ...)."""
    outs = []
    for k in range(4):
        r = np.rot90(arr, k)
        outs.append(r)
        outs.append(np.fliplr(r))
    return outs


def _bincount_colors(values: np.ndarray) -> dict[int, int]:
    """Return {color: count} for a 1-D int array, skipping 0."""
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
        # Pick fill color: the non-zero color appearing closest to the border,
        # breaking ties by frequency (matches original tie-break semantics).
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
    # Build the full checker, then punch `holes` into src (and clear in dst).
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
        # Ensure both colors appear, then shuffle the rest uniformly.
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
    """Pick a rule and any rule-specific parameters; also return the palette size needed."""
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
    """Dispatch to the per-rule pair builder; returns (src, dst)."""
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


def quadrant_positions(s: int, rng: random.Random) -> list[tuple[int, int]]:
    """Four frame anchors, one per quadrant, with small jitter."""
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
    """Paint a noisy diagonal-stripe background, reserving frame regions as-is."""
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
    """Draw a hollow square frame at (r0,c0) of side s, clear the interior to 0,
    then stamp sprite inside (sprite 0s remain as 0 background)."""
    r1, c1 = r0 + s - 1, c0 + s - 1
    grid[r0:r1 + 1, c0:c1 + 1] = 0           # clear full square to 0 (overwrites bg)
    grid[r0, c0:c1 + 1] = frame_color
    grid[r1, c0:c1 + 1] = frame_color
    grid[r0:r1 + 1, c0] = frame_color
    grid[r0:r1 + 1, c1] = frame_color
    paste(grid, sprite, r0 + 1, c0 + 1, transparent=0)


def write_into_blank_frame(
    grid: np.ndarray,
    frame_box: tuple[int, int, int, int],
    sprite: np.ndarray,
) -> np.ndarray:
    """Return a full-grid copy with `sprite` written into the given frame interior."""
    r0, c0, r1, c1 = frame_box
    out = np.array(grid, copy=True)
    out[r0 + 1 : r1, c0 + 1 : c1] = 0
    paste(out, sprite, r0 + 1, c0 + 1)
    return out


def validate_scene(grid: np.ndarray, expected_output: np.ndarray) -> bool:
    """Oracle check for the D4 variant: solver must fill only the blank frame and return the full grid."""
    frames = find_square_frames(grid)
    if len(frames) != 4:
        return False
    sizes = {size for size, _, *_ in frames}
    if len(sizes) != 1 or next(iter(sizes)) not in set(FRAME_SIZE_CHOICES):
        return False
    if expected_output.shape != grid.shape:
        return False
    inners = [grid[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1 in frames]
    blank_ids = [i for i, inner in enumerate(inners) if np.all(inner == 0)]
    if len(blank_ids) != 1:
        return False
    blank_idx = blank_ids[0]
    nonblank_sigs = [tuple(colors_present(inner, exclude={0})) for inner in inners]
    nonblank_sigs = [s for s in nonblank_sigs if s]
    dup_count = sum(
        1 for i in range(len(nonblank_sigs)) for j in range(i + 1, len(nonblank_sigs))
        if nonblank_sigs[i] == nonblank_sigs[j]
    )
    if dup_count != 1:
        return False
    _, _, r0, c0, r1, c1 = frames[blank_idx]
    allowed = np.zeros_like(grid, dtype=bool)
    allowed[r0 + 1 : r1, c0 + 1 : c1] = True
    diff = expected_output != grid
    if not np.any(diff & allowed):
        return False
    if np.any(diff & ~allowed):
        return False
    return np.array_equal(main(grid), expected_output)


def _valid_nz(arr: np.ndarray) -> bool:
    return SPRITE_NZ_MIN <= int(np.count_nonzero(arr)) <= SPRITE_NZ_MAX


def _build_sprites(kind: str, setup_params: dict, n: int, p1: list[int], p2: list[int],
                   rng: random.Random):
    """Build (train_src, train_dst, test_src, test_dst) or return None on failure.

    Verifies: rule is self-consistent (inferable from training pair), test output is
    non-degenerate, pixel counts in bounds, and color signatures match the spec.
    """
    train_src, train_dst = build_training_pair(kind, setup_params, n, p1, rng)
    inferred = infer_template(train_src, train_dst)
    if inferred is None or inferred[0] != kind:
        return None
    inferred_kind, inferred_params = inferred

    test_src, _test_pair_dst = build_training_pair(kind, setup_params, n, p2, rng)
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
    """Pick a single frame color and a bg palette disjoint from sprite + frame colors."""
    frame_color = rng.choice([c for c in ALL_COLORS if c not in used])
    bg_pool = [c for c in ALL_COLORS if c not in used and c != frame_color]
    bg_palette = rng.sample(bg_pool, rng.choice(BG_PERIOD_CHOICES))
    return frame_color, bg_palette


def _assign_slots(positions: list[tuple[int, int]], rng: random.Random) -> tuple[list[int], int, int]:
    """Pick (train_slots [2], test_slot, blank_slot) out of the 4 quadrant positions.

    The remaining slot is implicitly blank (its sprite stays all-zero)."""
    order = sorted(range(4), key=lambda i: positions[i])
    train_slots = sorted(rng.sample(order, 2), key=order.index)
    remaining = [i for i in order if i not in train_slots]
    test_slot = rng.choice(remaining)
    blank_slot = next(i for i in remaining if i != test_slot)
    return train_slots, test_slot, blank_slot


def _render_scene(s: int, positions: list[tuple[int, int]], sprites: list[np.ndarray],
                  frame_color: int, bg_palette: list[int], rng: random.Random) -> np.ndarray:
    """Render bg, then stamp 4 frames with their interiors."""
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
    train_slots, test_slot, blank_slot = _assign_slots(positions, rng)

    sprites = [np.zeros((n, n), dtype=int) for _ in range(4)]
    sprites[train_slots[0]] = train_src
    sprites[train_slots[1]] = train_dst
    sprites[test_slot] = test_src
    # The fourth slot keeps its all-zero sprite (blank interior).

    grid = _render_scene(s, positions, sprites, frame_color, bg_palette, rng)
    blank_r0, blank_c0 = positions[blank_slot]
    expected_output = write_into_blank_frame(
        grid,
        (blank_r0, blank_c0, blank_r0 + s - 1, blank_c0 + s - 1),
        test_dst,
    )
    if not validate_scene(grid, expected_output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
