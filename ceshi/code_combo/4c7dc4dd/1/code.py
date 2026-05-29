# Combo variant of 4c7dc4dd (index 1)
# combo_spec: D2/0+D6/0
# Name: 彩框配对矩形容器
#
# Design analysis:
# a) Concretization and behavior trace:
#    - Fused samples contain exactly four same-size hollow rectangles with height != width.
#      Each rectangle encloses a rectangular sprite area; one interior is blank, three are
#      non-blank. The two training rectangles share a border color, while the test rectangle
#      uses a different border color and the blank rectangle uses a third one.
#    - Small trace: suppose the four outer boxes are 6x8, so the sprite areas are 4x6.
#      Reading-order border colors on the three non-blank frames might be [2, 7, 2], and all
#      three non-blank interiors might use the same visible palette (3, 8). Original logic
#      would fail twice: square search would miss the containers, and even if rectangles were
#      detected, matching by interior palette would be ambiguous because all three signatures
#      are identical. The fused solver must first detect rectangles, then select the unique
#      repeated border color as the training pair, then learn the interior rule from that pair.
# b) Complexity calibration:
#    - Rectangle sizes come from a short repeated menu of obvious non-square dimensions so the
#      D6 change is visible but not chaotic.
#    - Border-color matching is made necessary by forcing all three non-blank sprites to share
#      one visible color signature; otherwise D2 would collapse back to the original cue.
#    - Only one repeated non-blank border color is allowed, so the training pair is unique.
# c) Implicit assumptions audit:
#    - `find_square_frames()` and square slicing are invalid once containers become rectangles;
#      replace them with `find_rect_frames()` and rectangular inner extraction.
#    - 90/270-degree rotations no longer preserve shape, so the symmetry rule must use only
#      left-right flip, up-down flip, and 180-degree rotation.
#    - The old `pair` logic assumed exactly one duplicated interior color-set among non-blank
#      frames. Under this combo, generator deliberately makes all three signatures identical,
#      so solver must key off border colors first and only keep old heuristics as a fallback.
#    - The blank finder still assumes one all-zero interior; generator preserves that by
#      clearing every frame interior to 0 before pasting the sprite.
# d) Placement legality:
#    - All four boxes share one sampled outer rectangle size and are placed one per quadrant.
#      Their bounding boxes are checked for overlap before rendering.
#    - The interior builders all operate on the derived rectangular inner size `(h-2, w-2)`,
#      so sprites never exceed the frame interior.
# e) Visual observability:
#    - A human can see both changes directly: the containers are plainly rectangular, and the
#      repeated border color is the only reliable cue for which two examples belong together.
#    - Because the three non-blank interiors share the same palette, the border cue is not a
#      cosmetic detail; it is required to identify the training pair.
# f) Visual cleanliness:
#    - The scene still uses only four large containers, sparse interior sprites, and a noisy
#      but low-frequency background. Border colors are kept distinct from sprite and background
#      colors so the containers remain legible.
# g) Change inferability:
#    - No hidden ordering by numeric color id is introduced. Border-color repetition is a
#      visible grouping signal, while interior rule inference still comes from shape structure.
# h) Fusion metric used after implementation:
#    - Measure two quantities on generated samples: rectangle aspect ratios must all be
#      non-square, and the three non-blank interior color signatures must collapse to exactly
#      one shared signature while the non-blank border colors form counts [2,1]. If either
#      condition fails, one side of the combo has degraded.
# i) New solver assumptions checked against the generator:
#    - Solver assumes a unique repeated border color among non-blank rectangles; generator
#      assigns exactly one repeated training color plus distinct test/blank colors.
#    - Solver assumes all candidate training/test interiors have the same rectangular shape;
#      generator uses one global frame size per scene.
#    - Solver scores both training directions after the pair is identified; generator still
#      builds the pair from one of the original six rule families so one direction is valid.

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, draw_line, paste


# ------- CFG: constants governing sampling distribution -------
# Canvas
H = W = 30

# Palette pools
SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
FRAME_COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
ALL_COLORS = list(range(10))
BG_PERIOD_CHOICES = (4, 5)
BG_PERTURB_FRACTION = (0.30, 0.50)
BG_EXTRA_NOISE = (0,)

# Frame layout: all outer sizes are rectangles with aspect ratio != 1
FRAME_RECT_CHOICES = ((6, 8), (8, 6), (6, 9), (9, 6), (7, 8), (8, 7))
FRAME_DIM_SET = set(FRAME_RECT_CHOICES)
QUAD_LOW_MIN, QUAD_LOW_MAX = 1, 5
QUAD_EDGE_MARGIN = 5

# Rule set
RULES = ["endpoints", "fill", "sym_fill", "comp_checker", "invert0", "sym"]
_RULE_BAG: list[str] = []

# Sprite pixel-count bounds
SPRITE_NZ_MIN, SPRITE_NZ_MAX = 3, 32

# Rule-specific build bounds
SYMFILL_KEEP0_PROB = 0.4
FILL_RECT_AREA_MIN, FILL_RECT_AREA_MAX = 5, 25
ENDPOINTS_LINE_LEN_MIN = 3
COMP_CHECKER_M_MAX = 10
INVERT0_STABLE_MAX = 3
INVERT0_VAR_MAX = 12
SYM_BUILD_RETRIES = 30
SPARSE_COUNT_MAX = 10
TEST_BUILD_RETRIES = 30

# Outer retry loop
GEN_MAX_ATTEMPTS = 300


def find_rect_frames(grid: np.ndarray) -> list[tuple[int, int, int, int, int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    cands = []
    for fh, fw in FRAME_RECT_CHOICES:
        if fh > h or fw > w:
            continue
        for r0 in range(0, h - fh + 1):
            r1 = r0 + fh - 1
            for c0 in range(0, w - fw + 1):
                c1 = c0 + fw - 1
                col = int(g[r0, c0])
                if (
                    int(g[r0, c1]) != col
                    or int(g[r1, c0]) != col
                    or int(g[r1, c1]) != col
                ):
                    continue
                if not (
                    np.all(g[r0, c0 : c1 + 1] == col)
                    and np.all(g[r1, c0 : c1 + 1] == col)
                    and np.all(g[r0 : r1 + 1, c0] == col)
                    and np.all(g[r0 : r1 + 1, c1] == col)
                ):
                    continue
                area = fh * fw
                cands.append((area, fh, fw, col, r0, c0, r1, c1))
    cands.sort(key=lambda t: (-t[0], -max(t[1], t[2]), t[4], t[5]))
    picked = []
    for cand in cands:
        _, _, _, _, r0, c0, r1, c1 = cand
        overlaps = False
        for _, _, _, _, pr0, pc0, pr1, pc1 in picked:
            if not (r1 < pr0 or pr1 < r0 or c1 < pc0 or pc1 < c0):
                overlaps = True
                break
        if not overlaps:
            picked.append(cand)
        if len(picked) >= 6:
            break
    return picked


def find_square_frames_legacy(grid: np.ndarray) -> list[tuple[int, int, int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    cands = []
    for size in range(4, min(h, w) + 1):
        for r0 in range(0, h - size + 1):
            r1 = r0 + size - 1
            for c0 in range(0, w - size + 1):
                c1 = c0 + size - 1
                col = int(g[r0, c0])
                if (
                    int(g[r0, c1]) != col
                    or int(g[r1, c0]) != col
                    or int(g[r1, c1]) != col
                ):
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


def shape_preserving_symmetries(arr: np.ndarray) -> list[np.ndarray]:
    return [
        arr.copy(),
        np.fliplr(arr),
        np.flipud(arr),
        np.flipud(np.fliplr(arr)),
    ]


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

    for idx, var in enumerate(shape_preserving_symmetries(src)):
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
        return shape_preserving_symmetries(x)[params["idx"]].copy()

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a rectangular sprite rule from border-marked examples, then apply it to the held-out sprite.

    Input structure:
    - The grid contains several same-size hollow rectangular frames, and each frame has a non-square aspect ratio.
    - Each frame encloses a small rectangular sprite; exactly one frame interior is entirely blank.
    - Among the non-blank frames, the two frames sharing a border color form the training pair; the remaining non-blank frame is the test source.

    Rule:
    - Detect the non-square monochrome rectangle frames and slice out their rectangular interiors.
    - Use the repeated border color to locate the source/destination example pair, then infer one interior transformation rule.
    - Apply the same rule to the test sprite and return only the transformed interior.

    Key insight:
    - This is still an analogy puzzle, but two cues moved at once:
    - the containers must be recognized as rectangles rather than squares, and the training pair is identified by matching frame colors rather than matching sprite palettes.
    """
    g = np.asarray(grid, dtype=int)
    frames = find_rect_frames(g)
    if len(frames) < 3:
        return g.copy()

    inners = []
    frame_colors = []
    for _, fh, fw, col, r0, c0, r1, c1 in frames:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.shape != (fh - 2, fw - 2):
            continue
        inners.append(((r0, c0, r1, c1), inner))
        frame_colors.append(int(col))
    if len(inners) < 3:
        return g.copy()

    blank_idx = None
    for i, (_, inner) in enumerate(inners):
        if inner.size > 0 and np.all(inner == 0):
            blank_idx = i
            break
    if blank_idx is None:
        blank_idx = int(np.argmin([np.count_nonzero(inner) for _, inner in inners]))

    nonblank = [i for i in range(len(inners)) if i != blank_idx]
    pair = None
    by_border: dict[int, list[int]] = {}
    for i in nonblank:
        by_border.setdefault(frame_colors[i], []).append(i)
    border_pairs = [tuple(idxs) for idxs in by_border.values() if len(idxs) == 2]
    if len(border_pairs) == 1:
        pair = border_pairs[0]
    if pair is None:
        sigs = []
        for i in nonblank:
            inner = inners[i][1]
            nz = sorted(set(int(x) for x in inner.ravel() if int(x) != 0))
            sigs.append((i, tuple(nz)))
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
        score2 = score * 100 - abs(src_n - test_n)
        if best is None or score2 > best[0]:
            best = (score2, kind, params)

    if best is None:
        return test_inner.copy()

    _, kind, params = best
    return apply_template(kind, params, test_inner)


def orbit_cells(h: int, w: int, r: int, c: int) -> list[tuple[int, int]]:
    cells = {
        (r, c),
        (r, w - 1 - c),
        (h - 1 - r, c),
        (h - 1 - r, w - 1 - c),
    }
    return sorted(cells)


def choose_palette(rng: random.Random, count: int, forbidden: set[int]) -> list[int]:
    pool = [c for c in SPRITE_COLORS if c not in forbidden]
    if len(pool) < count:
        raise ValueError("not enough sprite colors")
    return rng.sample(pool, count)


def build_fill_pair(h: int, w: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    v = colors[0]
    dst = np.zeros((h, w), dtype=int)
    while True:
        rh = rng.randint(2, h)
        rw = rng.randint(2, w)
        if FILL_RECT_AREA_MIN <= rh * rw <= FILL_RECT_AREA_MAX:
            break
    r0 = rng.randint(0, h - rh)
    c0 = rng.randint(0, w - rw)
    dst[r0 : r0 + rh, c0 : c0 + rw] = v
    src = np.zeros_like(dst)
    src[r0, c0] = v
    src[r0, c0 + rw - 1] = v
    src[r0 + rh - 1, c0] = v
    src[r0 + rh - 1, c0 + rw - 1] = v
    return src, dst


def build_endpoints_pair(h: int, w: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    v = colors[0]
    src = np.zeros((h, w), dtype=int)
    dst = np.zeros_like(src)
    row_pairs = [(0, h - 1)]
    if h >= 5:
        row_pairs.extend([(0, h - 2), (1, h - 1)])
    r1, r2 = rng.choice(row_pairs)
    len1 = rng.randint(ENDPOINTS_LINE_LEN_MIN, w)
    len2 = rng.randint(ENDPOINTS_LINE_LEN_MIN, w)
    c1 = rng.randint(0, w - len1)
    c2 = rng.randint(0, w - len2)
    draw_line(src, r1, c1, end_r=r1, end_c=c1 + len1 - 1, color=v)
    draw_line(src, r2, c2, end_r=r2, end_c=c2 + len2 - 1, color=v)
    dst[r1, c1] = v
    dst[r1, c1 + len1 - 1] = v
    dst[r2, c2] = v
    dst[r2, c2 + len2 - 1] = v
    return src, dst


def build_comp_checker_pair(h: int, w: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    if len(colors) == 1:
        even_color = odd_color = colors[0]
    else:
        even_color, odd_color = colors
    min_src = 4 if max(h, w) >= 6 else 3
    m = rng.randint(min_src, min(COMP_CHECKER_M_MAX, h * w - 3))
    all_cells = [(r, c) for r in range(h) for c in range(w)]
    while True:
        chosen = rng.sample(all_cells, m)
        parities = {(r + c) & 1 for r, c in chosen}
        if len(colors) == 1 or len(parities) == 2:
            break
    rr, cc = np.indices((h, w))
    full = np.where(((rr + cc) & 1) == 0, even_color, odd_color).astype(int)
    hole_mask = np.zeros((h, w), dtype=bool)
    for r, c in chosen:
        hole_mask[r, c] = True
    src = np.where(hole_mask, full, 0)
    dst = np.where(hole_mask, 0, full)
    return src, dst


def build_invert0_pair(h: int, w: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    v, s = colors
    all_cells = [(r, c) for r in range(h) for c in range(w)]
    stable_n = rng.randint(1, min(INVERT0_STABLE_MAX, h))
    stable_cells = set(rng.sample(all_cells, stable_n))
    remaining = [cell for cell in all_cells if cell not in stable_cells]
    var_cap = min(len(remaining) - 2, INVERT0_VAR_MAX)
    if var_cap < 2:
        raise ValueError("not enough remaining cells for invert0")
    var_n = rng.randint(2, var_cap)
    var_cells = set(rng.sample(remaining, var_n))
    dst = np.zeros((h, w), dtype=int)
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
        return np.flipud(arr)
    if idx == 3:
        return np.flipud(np.fliplr(arr))
    raise ValueError(idx)


def random_sparse_pattern(h: int, w: int, colors: list[int], rng: random.Random) -> np.ndarray:
    arr = np.zeros((h, w), dtype=int)
    count = rng.randint(3, min(SPARSE_COUNT_MAX, h * w - 1))
    cells = rng.sample([(r, c) for r in range(h) for c in range(w)], count)
    used = [colors[0]] * count
    if len(colors) == 2:
        for i in range(count):
            used[i] = colors[rng.randint(0, 1)]
        used[0] = colors[0]
        used[1] = colors[1]
    for (r, c), col in zip(cells, used):
        arr[r, c] = col
    return arr


def build_sym_pair(h: int, w: int, colors: list[int], rng: random.Random, idx: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(SYM_BUILD_RETRIES):
        dst = random_sparse_pattern(h, w, colors, rng)
        src = inverse_symmetry(dst, idx)
        if not np.array_equal(src, dst):
            return src, dst
    raise ValueError("failed to build sym pair")


def build_sym_fill_pair(
    h: int,
    w: int,
    colors: list[int],
    rng: random.Random,
    keep0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fill_col, accent_col = colors
    src = np.zeros((h, w), dtype=int)
    orbits = []
    seen = set()
    for r in range(h):
        for c in range(w):
            orb = tuple(orbit_cells(h, w, r, c))
            if orb not in seen:
                seen.add(orb)
                orbits.append(list(orb))
    orbits.sort(
        key=lambda orb: (
            -max(min(r, c, h - 1 - r, w - 1 - c) for r, c in orb),
            -len(orb),
            orb[0],
        )
    )
    fill_candidates = [orb for orb in orbits if not any(keep0[r, c] for r, c in orb)]
    if not fill_candidates:
        raise ValueError("no fill orbit available")
    # Randomize orbit choices so train/test examples differ, while still preferring
    # centrally placed orbits that make the symmetry-completion rule easy to see.
    fill_orbit = rng.choice(fill_candidates[: min(3, len(fill_candidates))])
    src[rng.choice(fill_orbit)] = fill_col
    accent_orbits = [
        orb for orb in fill_candidates if orb != fill_orbit and len(orb) >= 2
    ]
    if len(accent_orbits) < 2:
        raise ValueError("not enough accent orbits")
    pick_n = min(len(accent_orbits), rng.randint(2, min(3, len(accent_orbits))))
    for orb in rng.sample(accent_orbits, pick_n):
        src[rng.choice(orb)] = accent_col
    dst = apply_template("sym_fill", {"keep0": keep0}, src)
    return src, dst


def next_rule_kind() -> str:
    global _RULE_BAG
    if not _RULE_BAG:
        _RULE_BAG = RULES.copy()
        random.shuffle(_RULE_BAG)
    return _RULE_BAG.pop()


def choose_rule_setup(
    h: int, w: int, rng: random.Random, forced_kind: str | None = None
) -> tuple[str, dict, int]:
    kind = forced_kind or next_rule_kind()
    params: dict = {}
    if kind == "invert0":
        color_count = 2
    elif kind == "sym_fill":
        color_count = 2
        keep0 = np.zeros((h, w), dtype=bool)
        if min(h, w) == 4 or rng.random() < SYMFILL_KEEP0_PROB:
            for r, c in orbit_cells(h, w, 0, 0):
                keep0[r, c] = True
        params["keep0"] = keep0
    elif kind == "sym":
        color_count = rng.choice([1, 2])
        params["idx"] = rng.randint(1, 3)
    elif kind == "comp_checker":
        color_count = rng.choice([1, 2])
    else:
        color_count = 1
    return kind, params, color_count


def build_training_pair(kind: str, params: dict, h: int, w: int, colors: list[int], rng: random.Random):
    if kind == "fill":
        return build_fill_pair(h, w, colors, rng)
    if kind == "endpoints":
        return build_endpoints_pair(h, w, colors, rng)
    if kind == "comp_checker":
        return build_comp_checker_pair(h, w, colors, rng)
    if kind == "invert0":
        return build_invert0_pair(h, w, colors, rng)
    if kind == "sym":
        return build_sym_pair(h, w, colors, rng, params["idx"])
    if kind == "sym_fill":
        return build_sym_fill_pair(h, w, colors, rng, params["keep0"])
    raise ValueError(kind)


def quadrant_positions(frame_h: int, frame_w: int, rng: random.Random) -> list[tuple[int, int]]:
    bottom_low = H - frame_h - QUAD_EDGE_MARGIN
    bottom_high = H - frame_h - 1
    right_low = W - frame_w - QUAD_EDGE_MARGIN
    right_high = W - frame_w - 1
    return [
        (rng.randint(QUAD_LOW_MIN, QUAD_LOW_MAX), rng.randint(QUAD_LOW_MIN, QUAD_LOW_MAX)),
        (rng.randint(QUAD_LOW_MIN, QUAD_LOW_MAX), rng.randint(right_low, right_high)),
        (rng.randint(bottom_low, bottom_high), rng.randint(QUAD_LOW_MIN, QUAD_LOW_MAX)),
        (rng.randint(bottom_low, bottom_high), rng.randint(right_low, right_high)),
    ]


def _boxes_non_overlapping(boxes: list[tuple[int, int, int, int]]) -> bool:
    for i in range(len(boxes)):
        r0, c0, r1, c1 = boxes[i]
        for j in range(i + 1, len(boxes)):
            rr0, cc0, rr1, cc1 = boxes[j]
            if not (r1 < rr0 or rr1 < r0 or c1 < cc0 or cc1 < c0):
                return False
    return True


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


def draw_frame(
    grid: np.ndarray,
    r0: int,
    c0: int,
    frame_h: int,
    frame_w: int,
    frame_color: int,
    sprite: np.ndarray,
) -> None:
    r1, c1 = r0 + frame_h - 1, c0 + frame_w - 1
    grid[r0 : r1 + 1, c0 : c1 + 1] = 0
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color
    paste(grid, sprite, r0 + 1, c0 + 1, transparent=0)


def validate_scene(grid: np.ndarray, expected_output: np.ndarray) -> bool:
    frames = find_rect_frames(grid)
    if len(frames) != 4:
        return False
    if len(find_square_frames_legacy(grid)) >= 4:
        return False
    dims = {(fh, fw) for _, fh, fw, _, *_ in frames}
    if len(dims) != 1 or next(iter(dims)) not in FRAME_DIM_SET:
        return False
    if any(fh == fw for fh, fw in dims):
        return False
    inners = [grid[r0 + 1 : r1, c0 + 1 : c1] for _, _, _, _, r0, c0, r1, c1 in frames]
    blank_idxs = [i for i, inner in enumerate(inners) if np.all(inner == 0)]
    if len(blank_idxs) != 1:
        return False
    blank_idx = blank_idxs[0]
    nonblank = [i for i in range(4) if i != blank_idx]
    by_border: dict[int, list[int]] = {}
    for i in nonblank:
        by_border.setdefault(int(frames[i][3]), []).append(i)
    if sorted(len(v) for v in by_border.values()) != [1, 2]:
        return False
    nonblank_sigs = [tuple(colors_present(inners[i], exclude={0})) for i in nonblank]
    if len(set(nonblank_sigs)) != 1:
        return False
    return np.array_equal(main(grid), expected_output)


def _valid_nz(arr: np.ndarray) -> bool:
    return SPRITE_NZ_MIN <= int(np.count_nonzero(arr)) <= SPRITE_NZ_MAX


def _color_sig(arr: np.ndarray) -> tuple[int, ...]:
    return tuple(colors_present(arr, exclude={0}))


def _build_sprites(
    kind: str,
    setup_params: dict,
    inner_h: int,
    inner_w: int,
    palette: list[int],
    rng: random.Random,
):
    train_src, train_dst = build_training_pair(kind, setup_params, inner_h, inner_w, palette, rng)
    inferred = infer_template(train_src, train_dst)
    if inferred is None or inferred[0] != kind:
        return None
    inferred_kind, inferred_params = inferred

    for arr in (train_src, train_dst):
        if not _valid_nz(arr):
            return None
    train_sig = _color_sig(train_src)
    if train_sig != _color_sig(train_dst):
        return None
    for _ in range(TEST_BUILD_RETRIES):
        test_src, _ = build_training_pair(kind, setup_params, inner_h, inner_w, palette, rng)
        test_dst = apply_template(inferred_kind, inferred_params, test_src)
        if np.array_equal(test_dst, test_src):
            continue
        if any(not _valid_nz(arr) for arr in (test_src, test_dst)):
            continue
        if _color_sig(test_src) != train_sig or _color_sig(test_dst) != train_sig:
            continue
        if np.array_equal(test_src, train_src) or np.array_equal(test_src, train_dst):
            continue
        if np.array_equal(test_dst, train_src) or np.array_equal(test_dst, train_dst):
            continue
        return train_src, train_dst, test_src, test_dst
    return None


def _choose_frame_colors_and_bg(used: set[int], rng: random.Random) -> tuple[int, int, int, list[int]]:
    pool = [c for c in FRAME_COLORS if c not in used]
    train_frame_color, test_frame_color, blank_frame_color = rng.sample(pool, 3)
    bg_pool = [
        c for c in ALL_COLORS
        if c not in used and c not in {train_frame_color, test_frame_color, blank_frame_color}
    ]
    bg_palette = rng.sample(bg_pool, rng.choice(BG_PERIOD_CHOICES))
    return train_frame_color, test_frame_color, blank_frame_color, bg_palette


def _assign_slots(positions: list[tuple[int, int]], rng: random.Random) -> tuple[list[int], int, int]:
    order = sorted(range(4), key=lambda i: positions[i])
    patterns = [
        ("blank", "test", "train_src", "train_dst"),
        ("test", "blank", "train_src", "train_dst"),
        ("train_src", "test", "blank", "train_dst"),
        ("train_src", "blank", "test", "train_dst"),
        ("test", "train_src", "blank", "train_dst"),
        ("blank", "train_src", "test", "train_dst"),
    ]
    labels = rng.choice(patterns)
    slots = {}
    for idx, label in zip(order, labels):
        slots[label] = idx
    train_slots = [slots["train_src"], slots["train_dst"]]
    return train_slots, slots["test"], slots["blank"]


def _render_scene(
    frame_h: int,
    frame_w: int,
    positions: list[tuple[int, int]],
    sprites: list[np.ndarray],
    frame_colors: list[int],
    bg_palette: list[int],
    rng: random.Random,
) -> np.ndarray:
    frame_boxes = [(r0, c0, r0 + frame_h - 1, c0 + frame_w - 1) for r0, c0 in positions]
    if not _boxes_non_overlapping(frame_boxes):
        raise ValueError("overlapping frame boxes")
    grid = render_background(frame_boxes, bg_palette, rng)
    for (r0, c0), sprite, frame_color in zip(positions, sprites, frame_colors):
        draw_frame(grid, r0, c0, frame_h, frame_w, frame_color, sprite)
    return grid


def _gen_attempt(forced_kind: str | None = None) -> np.ndarray | None:
    rng = random.Random()
    frame_h, frame_w = rng.choice(FRAME_RECT_CHOICES)
    inner_h, inner_w = frame_h - 2, frame_w - 2

    kind, setup_params, color_count = choose_rule_setup(inner_h, inner_w, rng, forced_kind)
    palette = choose_palette(rng, color_count, set())

    built = _build_sprites(kind, setup_params, inner_h, inner_w, palette, rng)
    if built is None:
        return None
    train_src, train_dst, test_src, test_dst = built

    train_frame_color, test_frame_color, blank_frame_color, bg_palette = _choose_frame_colors_and_bg(
        set(palette), rng
    )
    positions = quadrant_positions(frame_h, frame_w, rng)
    train_slots, test_slot, blank_slot = _assign_slots(positions, rng)

    sprites = [np.zeros((inner_h, inner_w), dtype=int) for _ in range(4)]
    sprites[train_slots[0]] = train_src
    sprites[train_slots[1]] = train_dst
    sprites[test_slot] = test_src
    frame_colors = [blank_frame_color] * 4
    frame_colors[train_slots[0]] = train_frame_color
    frame_colors[train_slots[1]] = train_frame_color
    frame_colors[test_slot] = test_frame_color
    frame_colors[blank_slot] = blank_frame_color

    try:
        grid = _render_scene(frame_h, frame_w, positions, sprites, frame_colors, bg_palette, rng)
    except ValueError:
        return None
    if not validate_scene(grid, test_dst):
        return None
    return grid


def generate_input() -> np.ndarray:
    forced_kind = next_rule_kind()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(forced_kind)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
