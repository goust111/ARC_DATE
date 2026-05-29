# Combo variant of 4c7dc4dd (index 12)
# combo_spec: D6/0+D9/0+D14/1
# Name: 长方形网格中的线性矩形框架

# Design analysis for this combo variant
# a) Concretization and behavior trace:
#    - The original task placed four same-size square frames in the four quadrants of a 30x30
#      grid, then identified the training pair by matching interior color sets.
#    - This combo changes three linked stages at once:
#      1. the whole canvas is rectangular (20x30 or 30x20);
#      2. every frame is a non-square hollow rectangle;
#      3. the four frames are laid out in one visible line along the long axis, so the order
#         itself means training source -> training destination -> test source -> blank target.
#    - Example trace on a 20x30 grid: four 7x5 frames appear left-to-right in one row. The
#      first interior is the source sprite, the second is its transformed version, the third is
#      the test sprite, and the last is blank. Solver logic therefore sorts the detected frames
#      by the long-axis coordinate and reads the first three roles directly instead of hunting
#      for a palette-matched pair.
# b) Complexity calibration:
#    - The generator uses only two rectangular canvas shapes and a small repeated set of frame
#      sizes that clearly fit a 4-frame line. This keeps the visual change obvious without
#      introducing too many geometry cases.
#    - Rectangular interiors remain large enough for the original rule family
#      {endpoints, fill, sym_fill, comp_checker, invert0, sym} to stay legible.
#    - All three non-blank sprites reuse the same palette, which removes the original
#      "palette tells you which two frames are the training pair" shortcut and makes the
#      left-to-right / top-to-bottom order the dominant cue.
# c) Implicit assumptions audit:
#    - Old `find_square_frames()` assumed one scalar side length. That fails here; we now
#      enumerate allowed rectangle height/width pairs.
#    - Old symmetry inference allowed 90/270 degree rotations because square interiors kept the
#      same shape. Rectangles break that assumption, so only same-shape symmetries
#      {identity, fliplr, flipud, rot180} are allowed.
#    - Old role assignment relied on "exactly one duplicated non-zero color-set pair". This
#      combo intentionally breaks that cue by reusing the same palette in all three non-blank
#      frames, so `main()` must trust spatial order instead.
#    - The long canvas axis now matters: sorting frames by columns on wide grids and rows on
#      tall grids is a new solver assumption. The generator guarantees that alignment.
# d) Placement legality:
#    - Frames are placed by partitioning the long axis into five positive gaps
#      (margin, gap, gap, gap, margin), so the four frame boxes are guaranteed non-overlapping
#      and in-bounds.
#    - The short-axis coordinate is shared by all four frames, yielding a clean single row or
#      single column. Background is drawn first, then each frame clears its rectangle to 0,
#      draws the border, and stamps the sprite.
# e) Visual observability:
#    - The non-square grid is visible immediately because either width > height or height >
#      width, unlike the old 30x30 board.
#    - The frames are obviously rectangular and collinear, so a human can read the analogy in
#      order just by scanning the line.
#    - Because the test sprite uses the same palette as the training pair, semantic order is a
#      stronger clue than color-set matching.
# f) Visual cleanliness:
#    - Only four large frames appear, on a single axis, with one blank slot at the end. There is
#      no quadrant clutter and no overlapping annotations in the same region.
# g) Change inferability:
#    - No hidden color-id ordering is used. The only new semantic signal is visible geometry:
#      rectangular frame shape, rectangular canvas shape, and serial placement order.
# h) Fusion metric used after implementation:
#    - We measure four properties on generated samples: canvas aspect ratio != 1, frame aspect
#      ratio != 1, all frame anchors share one short-axis coordinate, and the first three
#      ordered frames all share the same non-zero palette. This checks that D6, D9, and D14 are
#      all active simultaneously.
# i) New solver assumptions checked against the generator:
#    - `main()` assumes the intended frames use one of the allowed rectangle sizes, exactly four
#      such frames are present, and their semantic order is the spatial order along the long
#      axis. `validate_scene()` enforces these properties before accepting a sample.

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, draw_line, paste


# ------- CFG: constants governing sampling distribution -------
GRID_SHAPES = ((20, 30), (30, 20))
ROW_FRAME_CHOICES = ((6, 5), (7, 5), (8, 6))
COL_FRAME_CHOICES = ((5, 6), (5, 7), (6, 8))
FRAME_RECT_CHOICES = tuple(sorted(set(ROW_FRAME_CHOICES + COL_FRAME_CHOICES)))
FRAME_DIM_SET = set(FRAME_RECT_CHOICES)

SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
FRAME_COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
ALL_COLORS = list(range(10))
BG_PERIOD_CHOICES = (4, 5)
BG_PERTURB_FRACTION = (0.30, 0.50)
BG_EXTRA_NOISE = (0,)

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

GEN_MAX_ATTEMPTS = 300


def find_rect_frames(grid: np.ndarray) -> list[tuple[int, int, int, int, int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    cands = []
    for fh, fw in FRAME_RECT_CHOICES:
        if fh > h or fw > w or fh == fw:
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


def long_axis_kind(shape: tuple[int, int]) -> str:
    h, w = shape
    return "row" if w > h else "col"


def sort_frames_linearly(
    frames: list[tuple[int, int, int, int, int, int, int, int]],
    shape: tuple[int, int],
) -> list[tuple[int, int, int, int, int, int, int, int]]:
    axis = long_axis_kind(shape)
    if axis == "row":
        return sorted(frames, key=lambda t: (t[5], t[4]))
    return sorted(frames, key=lambda t: (t[4], t[5]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read an ordered analogy from a line of framed sprites and apply the learned transformation to the test sprite.

    Input structure:
    - The board is rectangular rather than square, and it contains exactly four same-size hollow rectangular frames.
    - Those frames are arranged in one row on a wide board or one column on a tall board.
    - In order along that line, the interiors mean training source, training destination, test source, and a blank placeholder.

    Rule:
    - Detect the non-square rectangular frames and sort them along the grid's long axis.
    - Infer one sprite transformation from the first interior to the second using the original rule family
    - then apply that same transformation to the third interior.

    Key insight:
    - The puzzle is still an analogy, but the role labels are now encoded by spatial order instead of by quadrant placement
    - or by matching a unique pair of color signatures.
    """
    g = np.asarray(grid, dtype=int)
    frames = find_rect_frames(g)
    if len(frames) < 4:
        return g.copy()

    ordered = sort_frames_linearly(frames, g.shape)[:4]
    inners = []
    for _, fh, fw, _, r0, c0, r1, c1 in ordered:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.shape != (fh - 2, fw - 2):
            return g.copy()
        inners.append(inner)

    src = inners[0]
    dst = inners[1]
    test_inner = inners[2]

    best = None
    for cand_src, cand_dst in ((src, dst), (dst, src)):
        tpl = infer_template(cand_src, cand_dst)
        if tpl is None:
            continue
        kind, params = tpl
        st = set(params.get("stable", set()))
        v_src = pick_var_color(cand_src, st)
        v_dst = pick_var_color(cand_dst, st)
        score = 1
        if kind == "fill" and v_src is not None and v_dst is not None:
            if int(np.sum(cand_dst == v_dst)) >= int(np.sum(cand_src == v_src)):
                score += 2
        if kind == "endpoints" and v_src is not None and v_dst is not None:
            if int(np.sum(cand_dst == v_dst)) <= int(np.sum(cand_src == v_src)):
                score += 2
        src_n = int(np.sum((cand_src != 0) & ~np.isin(cand_src, list(st))))
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


def choose_palette(rng: random.Random, count: int) -> list[int]:
    if len(SPRITE_COLORS) < count:
        raise ValueError("not enough sprite colors")
    return rng.sample(SPRITE_COLORS, count)


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
    stable_n = rng.randint(1, min(INVERT0_STABLE_MAX, h * w - 2))
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
    usable_orbits = [orb for orb in orbits if not any(keep0[r, c] for r, c in orb)]
    accent_orbits = [orb for orb in usable_orbits if len(orb) >= 2]
    if len(usable_orbits) < 3 or len(accent_orbits) < 2:
        raise ValueError("not enough accent orbits")
    fill_orbit = rng.choice(usable_orbits)
    src[rng.choice(fill_orbit)] = fill_col

    accent_pool = [orb for orb in accent_orbits if orb != fill_orbit]
    if len(accent_pool) < 2:
        raise ValueError("not enough accent orbits")
    for orb in rng.sample(accent_pool, 2):
        src[rng.choice(orb)] = accent_col
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
        if min(h, w) == 3 or rng.random() < SYMFILL_KEEP0_PROB:
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


def random_same_palette_sprite(kind: str, params: dict, h: int, w: int, colors: list[int], rng: random.Random):
    return build_training_pair(kind, params, h, w, colors, rng)[0]


def render_background(
    shape: tuple[int, int],
    frame_boxes: list[tuple[int, int, int, int]],
    bg_palette: list[int],
    rng: random.Random,
) -> np.ndarray:
    h, w = shape
    period = len(bg_palette)
    direction = rng.choice([1, -1])
    rr, cc = np.indices((h, w))
    idx = (rr + direction * cc) % period
    grid = np.asarray(bg_palette, dtype=int)[idx]

    reserved = np.zeros((h, w), dtype=bool)
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


def _boxes_non_overlapping(boxes: list[tuple[int, int, int, int]]) -> bool:
    for i in range(len(boxes)):
        r0, c0, r1, c1 = boxes[i]
        for j in range(i + 1, len(boxes)):
            rr0, cc0, rr1, cc1 = boxes[j]
            if not (r1 < rr0 or rr1 < r0 or c1 < cc0 or cc1 < c0):
                return False
    return True


def _positive_partition(total: int, parts: int, rng: random.Random) -> list[int]:
    if total < parts:
        raise ValueError("cannot build positive partition")
    remaining = total - parts
    cuts = sorted(rng.sample(range(remaining + parts - 1), parts - 1))
    vals = []
    prev = -1
    for cut in cuts + [remaining + parts - 1]:
        vals.append(cut - prev)
        prev = cut
    return vals


def line_positions(shape: tuple[int, int], frame_h: int, frame_w: int, rng: random.Random) -> list[tuple[int, int]]:
    h, w = shape
    axis = long_axis_kind(shape)
    positions = []
    if axis == "row":
        slack = w - 4 * frame_w
        gaps = _positive_partition(slack, 5, rng)
        r0 = rng.randint(1, h - frame_h - 1)
        c = gaps[0]
        for i in range(4):
            positions.append((r0, c))
            c += frame_w
            if i < 3:
                c += gaps[i + 1]
    else:
        slack = h - 4 * frame_h
        gaps = _positive_partition(slack, 5, rng)
        c0 = rng.randint(1, w - frame_w - 1)
        r = gaps[0]
        for i in range(4):
            positions.append((r, c0))
            r += frame_h
            if i < 3:
                r += gaps[i + 1]
    return positions


def validate_scene(grid: np.ndarray, expected_output: np.ndarray) -> bool:
    frames = find_rect_frames(grid)
    if len(frames) != 4:
        return False
    if len(find_square_frames_legacy(grid)) >= 4:
        return False
    if tuple(grid.shape) not in GRID_SHAPES:
        return False
    dims = {(fh, fw) for _, fh, fw, _, *_ in frames}
    if len(dims) != 1 or next(iter(dims)) not in FRAME_DIM_SET:
        return False
    if any(fh == fw for fh, fw in dims):
        return False

    ordered = sort_frames_linearly(frames, grid.shape)
    inners = [grid[r0 + 1 : r1, c0 + 1 : c1] for _, _, _, _, r0, c0, r1, c1 in ordered]
    if not np.all(inners[3] == 0):
        return False
    if any(np.all(inner == 0) for inner in inners[:3]):
        return False

    axis = long_axis_kind(grid.shape)
    if axis == "row":
        rows = {r0 for _, _, _, _, r0, _, _, _ in ordered}
        if len(rows) != 1:
            return False
    else:
        cols = {c0 for _, _, _, _, _, c0, _, _ in ordered}
        if len(cols) != 1:
            return False

    sigs = [tuple(colors_present(inner, exclude={0})) for inner in inners[:3]]
    if len(set(sigs)) != 1:
        return False
    return np.array_equal(main(grid), expected_output)


def _valid_nz(arr: np.ndarray) -> bool:
    return SPRITE_NZ_MIN <= int(np.count_nonzero(arr)) <= SPRITE_NZ_MAX


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

    for _ in range(40):
        test_src = random_same_palette_sprite(kind, setup_params, inner_h, inner_w, palette, rng)
        test_dst = apply_template(inferred_kind, inferred_params, test_src)
        if np.array_equal(test_src, train_src) or np.array_equal(test_src, train_dst):
            continue
        if np.array_equal(test_dst, test_src):
            continue
        if np.array_equal(test_dst, train_dst):
            continue
        arrays = (train_src, train_dst, test_src, test_dst)
        if not all(_valid_nz(arr) for arr in arrays):
            continue
        sigs = [tuple(colors_present(arr, exclude={0})) for arr in (train_src, train_dst, test_src)]
        if len(set(sigs)) != 1:
            continue
        return train_src, train_dst, test_src, test_dst
    return None


def _choose_frame_and_bg(used: set[int], rng: random.Random) -> tuple[int, list[int]]:
    frame_color = rng.choice([c for c in FRAME_COLORS if c not in used])
    bg_pool = [c for c in ALL_COLORS if c not in used and c != frame_color]
    bg_palette = rng.sample(bg_pool, rng.choice(BG_PERIOD_CHOICES))
    return frame_color, bg_palette


def _render_scene(
    shape: tuple[int, int],
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
    grid = render_background(shape, frame_boxes, bg_palette, rng)
    for (r0, c0), sprite in zip(positions, sprites):
        draw_frame(grid, r0, c0, frame_h, frame_w, frame_color, sprite)
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    shape = rng.choice(GRID_SHAPES)
    axis = long_axis_kind(shape)
    frame_h, frame_w = rng.choice(ROW_FRAME_CHOICES if axis == "row" else COL_FRAME_CHOICES)
    inner_h, inner_w = frame_h - 2, frame_w - 2

    kind, setup_params, color_count = choose_rule_setup(inner_h, inner_w, rng)
    palette = choose_palette(rng, color_count)

    built = _build_sprites(kind, setup_params, inner_h, inner_w, palette, rng)
    if built is None:
        return None
    train_src, train_dst, test_src, test_dst = built

    frame_color, bg_palette = _choose_frame_and_bg(set(palette), rng)
    positions = line_positions(shape, frame_h, frame_w, rng)

    sprites = [
        train_src,
        train_dst,
        test_src,
        np.zeros((inner_h, inner_w), dtype=int),
    ]

    try:
        grid = _render_scene(shape, frame_h, frame_w, positions, sprites, frame_color, bg_palette, rng)
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
