# Variant of 4c7dc4dd
# Dimension: D6 — 框架检测方式 (symbol_instructions)
# Change: 通过搜索单色空心正方形来检测框架，优先选择更大的框架，框架不重叠... → 非正方形矩形框架

# Design analysis for this variant
# a) Concretization and behavior trace:
#    - The visible change is that all four containers are now same-size hollow rectangles
#      with height != width, not squares. Their interiors are therefore h x w rectangles.
#    - Original D6 path: frame finder enumerated side length S, sliced square interiors, and
#      the "sym" rule could use any dihedral symmetry because n x n stayed square after rot90.
#    - Variant D6 path: frame finder must enumerate rectangle height/width pairs, pick the
#      largest non-overlapping monochrome hollow rectangles, and slice rectangular interiors.
#      Example: an outer 6x8 frame now yields a 4x6 sprite area; the solver must detect
#      (r0,c0,r1,c1) from border rows/cols, not from a single side length.
#    - This changes the solver exactly at frame detection + interior extraction, and also at
#      the symmetry rule because 90/270 degree rotations would swap h and w and no longer fit
#      the rectangular test interior. The variant therefore keeps only shape-preserving
#      symmetries: horizontal flip, vertical flip, and 180-degree rotation.
# b) Complexity calibration:
#    - To keep the rectangles visually obvious, the generator samples from a small set of
#      repeated non-square outer sizes: 6x8, 8x6, 6x9, 9x6, 7x8, 8x7.
#    - Using a small repeated size set prevents ambiguous accidental rectangles while still
#      making the new D6 behavior clear across samples.
#    - Inner sizes remain within 4x6 to 7x4 / 5x6 / 6x5 so the original six transformation
#      templates still produce readable sprites with 3-32 non-zero cells.
# c) Implicit assumptions audit:
#    - Old square finder assumed one scalar `size`; that fails once height != width.
#    - Old `symmetries()` assumed every rot90 output is comparable to the destination; that is
#      false for rectangles because the shape changes. The new `shape_preserving_symmetries()`
#      only returns same-shape transforms.
#    - Old orbit construction for `sym_fill` assumed `(n,n)` and corner orbit `(0,0)` under
#      square reflections. The new `orbit_cells(h,w,...)` computes horizontal/vertical/both
#      reflection orbits on rectangles.
#    - Old placement ranges were tuned for squares and could approach overlap. The new layout
#      uses tighter quadrant anchor ranges that guarantee separation for the chosen rectangle
#      sizes.
# d) Placement legality:
#    - All four frames share the same outer (h,w), are placed one per quadrant, and boxes are
#      validated as non-overlapping before rendering.
#    - Background is rendered first, then each frame clears its full rectangle to 0, draws a
#      single-color border, and pastes the inner sprite into the rectangular interior.
# e) Visual observability:
#    - The aspect ratio change is visible immediately because every container is clearly wider
#      than tall or taller than wide.
#    - The transformation rule inside the frame still comes from the training pair, but the
#      frame geometry itself now forces the solver to reason about rectangles rather than
#      defaulting to square assumptions.
# f) Visual cleanliness:
#    - Only four large frames are present, one blank, three with sparse structured sprites.
#      Frames never overlap and backgrounds are rejected unless exactly the intended four
#      rectangles are detected.
# g) Color inference:
#    - No invisible ordering by color id is introduced. Training-pair detection still uses
#      visible shared non-zero palettes, and rule application depends on spatial structure.
# h) Variant metric used after implementation:
#    - For generated samples we measure detected frame aspect ratios and compare with the old
#      square-frame detector. The variant is valid only if the new generator consistently
#      produces four non-square frames and the old square detector does not recover the same
#      four-frame structure.
# i) New solver assumptions checked against the generator:
#    - `find_rect_frames()` assumes the true frames use one of the sampled non-square sizes and
#      are hollow monochrome rectangles. `generate_input()` enforces exactly that.
#    - `shape_preserving_symmetries()` assumes source and destination interiors have identical
#      rectangular shape. The generator keeps all four frames the same size, so both training
#      sprites and the test sprite share one interior shape.

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
    Infer a sprite-to-sprite transformation from framed examples, then apply it to the held-out sprite.

    Input structure:
    - The grid contains several same-size hollow rectangular frames whose width and height differ.
    - Each frame encloses a small rectangular sprite; exactly one frame interior is completely blank.
    - Two non-blank frames form the training pair because their interior color sets match; the remaining non-blank frame is the test source.

    Rule:
    - Detect the non-square monochrome rectangle frames, preferring larger non-overlapping frames.
    - Use the matched training pair to infer one interior rule (flip/180-rotation, inversion, fill, endpoint extraction, symmetric fill, or checkerboard complement).
    - Apply that inferred rule to the test sprite and return only the transformed rectangular interior.

    Key insight:
    - The analogy is still "A transforms to B, so C transforms to ?",
    - but the containers are no longer square, so frame detection and any symmetry reasoning must respect rectangular geometry.
    """
    g = np.asarray(grid, dtype=int)
    frames = find_rect_frames(g)
    if len(frames) < 3:
        return g.copy()

    inners = []
    for _, fh, fw, _, r0, c0, r1, c1 in frames:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.shape != (fh - 2, fw - 2):
            continue
        inners.append(((r0, c0, r1, c1), inner))
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


def choose_rule_setup(h: int, w: int, rng: random.Random) -> tuple[str, dict, int]:
    kind = rng.choice(RULES)
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


def _valid_nz(arr: np.ndarray) -> bool:
    return SPRITE_NZ_MIN <= int(np.count_nonzero(arr)) <= SPRITE_NZ_MAX


def _build_sprites(
    kind: str,
    setup_params: dict,
    inner_h: int,
    inner_w: int,
    p1: list[int],
    p2: list[int],
    rng: random.Random,
):
    train_src, train_dst = build_training_pair(kind, setup_params, inner_h, inner_w, p1, rng)
    inferred = infer_template(train_src, train_dst)
    if inferred is None or inferred[0] != kind:
        return None
    inferred_kind, inferred_params = inferred

    test_src, _ = build_training_pair(kind, setup_params, inner_h, inner_w, p2, rng)
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
    frame_color = rng.choice([c for c in FRAME_COLORS if c not in used])
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
    frame_h: int,
    frame_w: int,
    positions: list[tuple[int, int]],
    sprites: list[np.ndarray],
    frame_color: int,
    bg_palette: list[int],
    rng: random.Random,
) -> np.ndarray:
    frame_boxes = [(r0, c0, r0 + frame_h - 1, c0 + frame_w - 1) for r0, c0 in positions]
    if not _boxes_non_overlapping(frame_boxes):
        raise ValueError("overlapping frame boxes")
    grid = render_background(frame_boxes, bg_palette, rng)
    for (r0, c0), sprite in zip(positions, sprites):
        draw_frame(grid, r0, c0, frame_h, frame_w, frame_color, sprite)
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    frame_h, frame_w = rng.choice(FRAME_RECT_CHOICES)
    inner_h, inner_w = frame_h - 2, frame_w - 2

    kind, setup_params, color_count = choose_rule_setup(inner_h, inner_w, rng)
    p1 = choose_palette(rng, color_count, set())
    p2 = choose_palette(rng, color_count, set(p1))

    built = _build_sprites(kind, setup_params, inner_h, inner_w, p1, p2, rng)
    if built is None:
        return None
    train_src, train_dst, test_src, test_dst = built

    frame_color, bg_palette = _choose_frame_and_bg(set(p1) | set(p2), rng)
    positions = quadrant_positions(frame_h, frame_w, rng)
    train_slots, test_slot = _assign_slots(positions, rng)

    sprites = [np.zeros((inner_h, inner_w), dtype=int) for _ in range(4)]
    sprites[train_slots[0]] = train_src
    sprites[train_slots[1]] = train_dst
    sprites[test_slot] = test_src

    try:
        grid = _render_scene(frame_h, frame_w, positions, sprites, frame_color, bg_palette, rng)
    except ValueError:
        return None
    if not validate_scene(grid, test_dst):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
