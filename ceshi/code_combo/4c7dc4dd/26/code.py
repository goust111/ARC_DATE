# Combo variant of 4c7dc4dd (index 26)
# combo_spec: D6/0+D9/2+D10/0
# Name: 嵌套异形异尺寸框架
#
# Design analysis for the fused variant
# a) Concretization and behavior trace:
#    - Visible scene: there are 2 large hollow outer rectangles. Each outer rectangle contains
#      exactly 2 smaller hollow inner rectangles. The top/bottom order is not semantic; the
#      containment relation is.
#    - One outer frame encloses the training pair: two non-blank inner rectangles of the same
#      rectangular size and the same non-zero palette. The other outer frame encloses the test
#      case: one non-blank inner rectangle plus one blank inner rectangle, using a different
#      rectangular size.
#    - Example trace: original code scanned 4 peer square frames in separate quadrants, matched
#      the training pair globally by palette, then applied the learned rule to the remaining
#      non-blank frame of the same size. Here the solver must first detect 6 non-square
#      rectangles, use nesting to split them into two semantic groups, learn the rule from the
#      doubly non-blank group, and transfer it to the non-blank member of the other group even
#      though that target rectangle has a different size.
# b) Complexity calibration:
#    - Inner frames use a small repeated set of non-square sizes so the D6 change is visually
#      obvious without creating too many shape cases.
#    - Outer frames are derived deterministically from the contained inner-frame size, so the
#      nesting cue is strong and consistent across samples.
#    - The rule family stays the same as the original task, but only size-agnostic parameters are
#      allowed for `sym_fill`, because arbitrary learned masks would not transfer across sizes.
# c) Implicit assumptions audit:
#    - Original `find_square_frames()` assumed square peer containers and non-overlap. That fails
#      here because the relevant frames are rectangles and some frames are nested. Replaced with a
#      rectangle finder over a constrained dimension set and a later containment-based parse.
#    - Original `symmetries()` allowed 90/270 rotations. That breaks on rectangular interiors
#      because shape changes. Replaced with shape-preserving symmetries only: identity, horizontal
#      flip, vertical flip, 180-degree rotation.
#    - Original training/test disambiguation used all frames as one flat pool. That misses the D9
#      semantic cue. Replaced with grouped reasoning: one outer group must contain two non-blank
#      members, the other must contain exactly one blank member.
#    - Original direction scoring compared raw non-zero counts. That assumes equal inner area and
#      breaks under D10 mixed sizes. Replaced with variable-pixel density comparison.
# d) Placement legality:
#    - Every outer frame is positioned fully inside the 30x30 canvas and the 2 outer frames never
#      overlap.
#    - Inner frames are placed at fixed left/right slots inside their outer frame with one-cell
#      padding and one-cell separation, so they never overlap each other or the outer border.
# e) Visual observability:
#    - The non-square aspect ratio is immediately visible on both outer and inner frames.
#    - The nesting cue is visually explicit: training examples live together inside one container,
#      while the test sprite and blank sprite live together inside the other.
#    - The size split is also visible because the training inner frames and test/blank inner
#      frames use different rectangular dimensions in every valid sample.
# f) Visual cleanliness:
#    - Only 6 frames are present and they are organized into 2 clear containers rather than a
#      cluttered free-form layout.
#    - Outer and inner borders use distinct colors, reducing accidental merged-border illusions.
# g) Change inferability:
#    - Color ids never encode hidden ordering. The only palette cue is the visibly shared
#      non-zero color set inside the training pair, while the group structure is conveyed by
#      geometric containment that a human can inspect directly.
# h) Variant metric used after implementation:
#    - Samples are considered non-degenerate only if detection yields exactly 2 outer + 4 inner
#      rectangles, each outer contains exactly 2 inners, and the training inner size differs from
#      the test/blank inner size.
# i) New solver assumptions checked against the generator:
#    - `parse_scene()` assumes outer dimensions come from the derived outer-size family and inner
#      dimensions come from the sampled inner-size family. `generate_input()` enforces that.
#    - `parse_scene()` assumes one group has two non-blank matching-palette inners and the other
#      has one blank and one non-blank inner. The generator constructs exactly that semantic split.

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

# Inner frames: non-square rectangles only
INNER_FRAME_CHOICES = ((6, 8), (8, 6), (6, 9), (9, 6), (7, 8), (8, 7))
INNER_FRAME_DIM_SET = set(INNER_FRAME_CHOICES)

# Outer frames are derived from the inner-frame size they contain:
# outer_h = inner_h + 4, outer_w = 2*inner_w + 5
OUTER_FRAME_CHOICES = tuple(sorted(set((fh + 4, 2 * fw + 5) for fh, fw in INNER_FRAME_CHOICES)))
OUTER_FRAME_DIM_SET = set(OUTER_FRAME_CHOICES)
FRAME_DIM_SET = INNER_FRAME_DIM_SET | OUTER_FRAME_DIM_SET

# Outer placement
OUTER_TOP_BAND = (1, 3)
OUTER_BOTTOM_MARGIN = (1, 3)
OUTER_GAP_MIN = 2

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
GEN_MAX_ATTEMPTS = 500


def find_rect_frames(grid: np.ndarray) -> list[tuple[int, int, int, int, int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    cands = []
    for fh, fw in FRAME_DIM_SET:
        if fh > h or fw > w:
            continue
        for r0 in range(0, h - fh + 1):
            r1 = r0 + fh - 1
            for c0 in range(0, w - fw + 1):
                c1 = c0 + fw - 1
                col = int(g[r0, c0])
                if col == 0:
                    continue
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
    return cands


def contains_frame(outer: dict, inner: dict) -> bool:
    return (
        outer["r0"] < inner["r0"]
        and outer["c0"] < inner["c0"]
        and inner["r1"] < outer["r1"]
        and inner["c1"] < outer["c1"]
    )


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


def orbit_cells(h: int, w: int, r: int, c: int) -> list[tuple[int, int]]:
    cells = {
        (r, c),
        (r, w - 1 - c),
        (h - 1 - r, c),
        (h - 1 - r, w - 1 - c),
    }
    return sorted(cells)


def corner_orbit_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in orbit_cells(h, w, 0, 0):
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


def classify_keep0_mode(keep0: np.ndarray) -> str | None:
    if not np.any(keep0):
        return "none"
    if np.array_equal(keep0, corner_orbit_mask(*keep0.shape)):
        return "corner"
    return None


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
            bad = (sym_src == 0) & (dst != 0)
            if fill_col is None:
                bad &= False
            else:
                bad &= dst != fill_col
            if not np.any(bad):
                keep0_mode = classify_keep0_mode((sym_src == 0) & (dst == 0))
                if keep0_mode is not None:
                    return ("sym_fill", {"stable": st, "keep0_mode": keep0_mode})

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
        keep0 = np.zeros_like(y, dtype=bool)
        if params.get("keep0_mode") == "corner":
            keep0 = corner_orbit_mask(*y.shape)
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


def parse_scene(grid: np.ndarray) -> dict | None:
    g = np.asarray(grid, dtype=int)
    raw = find_rect_frames(g)
    frames = []
    seen = set()
    for area, fh, fw, col, r0, c0, r1, c1 in raw:
        key = (fh, fw, col, r0, c0, r1, c1)
        if key in seen:
            continue
        seen.add(key)
        frames.append(
            {
                "area": area,
                "fh": fh,
                "fw": fw,
                "color": col,
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "sprite": g[r0 + 1 : r1, c0 + 1 : c1],
            }
        )

    outers = [f for f in frames if (f["fh"], f["fw"]) in OUTER_FRAME_DIM_SET]
    inners = [f for f in frames if (f["fh"], f["fw"]) in INNER_FRAME_DIM_SET]
    if len(outers) != 2 or len(inners) != 4:
        return None

    groups = []
    for outer in sorted(outers, key=lambda f: (f["r0"], f["c0"])):
        contained = [inner for inner in inners if contains_frame(outer, inner)]
        contained.sort(key=lambda f: (f["r0"], f["c0"]))
        if len(contained) != 2:
            return None
        groups.append((outer, contained))

    for inner in inners:
        owner_count = sum(int(contains_frame(outer, inner)) for outer in outers)
        if owner_count != 1:
            return None

    group_infos = []
    for outer, contained in groups:
        blank = [f for f in contained if np.all(f["sprite"] == 0)]
        nonblank = [f for f in contained if not np.all(f["sprite"] == 0)]
        sigs = [tuple(colors_present(f["sprite"], exclude={0})) for f in nonblank]
        group_infos.append(
            {
                "outer": outer,
                "inners": contained,
                "blank": blank,
                "nonblank": nonblank,
                "sigs": sigs,
            }
        )

    train_group = None
    test_group = None
    for info in group_infos:
        if len(info["blank"]) == 0 and len(info["nonblank"]) == 2:
            sigs = info["sigs"]
            if len(sigs) == 2 and sigs[0] and sigs[0] == sigs[1]:
                train_group = info
        elif len(info["blank"]) == 1 and len(info["nonblank"]) == 1:
            test_group = info
    if train_group is None or test_group is None:
        return None

    train_dims = {(f["fh"], f["fw"]) for f in train_group["inners"]}
    test_dims = {(f["fh"], f["fw"]) for f in test_group["inners"]}
    if len(train_dims) != 1 or len(test_dims) != 1 or train_dims == test_dims:
        return None

    return {
        "outers": [g["outer"] for g in group_infos],
        "inners": inners,
        "train_group": train_group,
        "test_group": test_group,
        "train_dim": next(iter(train_dims)),
        "test_dim": next(iter(test_dims)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a sprite transformation from one nested training container and transfer it to a differently sized nested test container.

    Input structure:
    - The grid contains 2 large hollow rectangular outer frames, and each outer frame encloses exactly 2 smaller hollow rectangular inner frames.
    - One outer frame holds the training pair: 2 non-blank inner sprites with the same non-zero palette and the same inner-frame size.
    - The other outer frame holds the test case: 1 non-blank inner sprite and 1 blank inner sprite, using a different inner-frame size from the training pair.

    Rule:
    - Detect the non-square inner and outer rectangles, then use containment to separate the 6 detected frames into a training container and a test container.
    - Infer the transformation relating the 2 training sprites inside the doubly non-blank container.
    - Apply that same size-agnostic rule to the non-blank sprite in the other container and return only the transformed test interior.

    Key insight:
    - The analogy is no longer encoded by four peer frames in a flat layout; instead, spatial nesting marks semantic roles, and the learned sprite rule must generalize across rectangular inner frames of different sizes.
    """
    g = np.asarray(grid, dtype=int)
    scene = parse_scene(g)
    if scene is None:
        return g.copy()

    train_a, train_b = scene["train_group"]["nonblank"]
    test_inner = scene["test_group"]["nonblank"][0]["sprite"]
    src_a = train_a["sprite"]
    src_b = train_b["sprite"]

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
    keep0_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    fill_col, accent_col = colors
    keep0 = np.zeros((h, w), dtype=bool)
    if keep0_mode == "corner":
        keep0 = corner_orbit_mask(h, w)
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
    fill_orbit = next((orb for orb in orbits if not any(keep0[r, c] for r, c in orb)), None)
    if fill_orbit is None:
        raise ValueError("no fill orbit")
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


def choose_rule_setup(
    train_shape: tuple[int, int],
    test_shape: tuple[int, int],
    rng: random.Random,
) -> tuple[str, dict, int]:
    kind = rng.choice(RULES)
    params: dict = {}
    if kind == "invert0":
        color_count = 2
    elif kind == "sym_fill":
        color_count = 2
        keep0_mode = "none"
        if min(*train_shape, *test_shape) >= 4 and rng.random() < SYMFILL_KEEP0_PROB:
            keep0_mode = "corner"
        params["keep0_mode"] = keep0_mode
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
        return build_sym_fill_pair(h, w, colors, rng, params["keep0_mode"])
    raise ValueError(kind)


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
    fh: int,
    fw: int,
    frame_color: int,
    sprite: np.ndarray | None,
) -> None:
    r1, c1 = r0 + fh - 1, c0 + fw - 1
    grid[r0 : r1 + 1, c0 : c1 + 1] = 0
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color
    if sprite is not None:
        paste(grid, sprite, r0 + 1, c0 + 1, transparent=0)


def _valid_nz(arr: np.ndarray) -> bool:
    return SPRITE_NZ_MIN <= int(np.count_nonzero(arr)) <= SPRITE_NZ_MAX


def _build_sprites(
    kind: str,
    setup_params: dict,
    train_shape: tuple[int, int],
    test_shape: tuple[int, int],
    p1: list[int],
    p2: list[int],
    rng: random.Random,
):
    train_src, train_dst = build_training_pair(kind, setup_params, train_shape[0], train_shape[1], p1, rng)
    inferred = infer_template(train_src, train_dst)
    if inferred is None or inferred[0] != kind:
        return None
    inferred_kind, inferred_params = inferred

    test_src, _ = build_training_pair(kind, setup_params, test_shape[0], test_shape[1], p2, rng)
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


def _choose_frame_and_bg(used: set[int], rng: random.Random) -> tuple[int, int, list[int]]:
    colors = [c for c in FRAME_COLORS if c not in used]
    inner_frame_color, outer_frame_color = rng.sample(colors, 2)
    bg_pool = [c for c in ALL_COLORS if c not in used and c not in {inner_frame_color, outer_frame_color}]
    periods = [p for p in BG_PERIOD_CHOICES if p <= len(bg_pool)]
    if not periods:
        raise ValueError("not enough colors for background palette")
    bg_palette = rng.sample(bg_pool, rng.choice(periods))
    return inner_frame_color, outer_frame_color, bg_palette


def make_outer_dims(inner_frame_dims: tuple[int, int]) -> tuple[int, int]:
    fh, fw = inner_frame_dims
    return fh + 4, 2 * fw + 5


def choose_outer_positions(
    top_dims: tuple[int, int],
    bottom_dims: tuple[int, int],
    rng: random.Random,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    top_h, top_w = top_dims
    bottom_h, bottom_w = bottom_dims
    top_r = rng.randint(*OUTER_TOP_BAND)
    bottom_r_low = H - bottom_h - OUTER_BOTTOM_MARGIN[1]
    bottom_r_high = H - bottom_h - OUTER_BOTTOM_MARGIN[0]
    if bottom_r_low > bottom_r_high:
        return None
    bottom_r = rng.randint(bottom_r_low, bottom_r_high)
    if bottom_r < top_r + top_h + OUTER_GAP_MIN:
        return None
    top_c = rng.randint(1, W - top_w - 1)
    bottom_c = rng.randint(1, W - bottom_w - 1)
    return (top_r, top_c), (bottom_r, bottom_c)


def inner_slots(outer_pos: tuple[int, int], inner_dims: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = outer_pos
    fh, fw = inner_dims
    left = (r0 + 2, c0 + 2)
    right = (r0 + 2, c0 + fw + 3)
    return [left, right]


def validate_scene(grid: np.ndarray, expected_output: np.ndarray) -> bool:
    scene = parse_scene(grid)
    if scene is None:
        return False
    train_group = scene["train_group"]
    test_group = scene["test_group"]
    if len(train_group["nonblank"]) != 2 or len(test_group["blank"]) != 1 or len(test_group["nonblank"]) != 1:
        return False
    if np.array_equal(test_group["nonblank"][0]["sprite"], expected_output):
        return False
    return np.array_equal(main(grid), expected_output)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    train_frame_dims, test_frame_dims = rng.sample(INNER_FRAME_CHOICES, 2)
    train_shape = (train_frame_dims[0] - 2, train_frame_dims[1] - 2)
    test_shape = (test_frame_dims[0] - 2, test_frame_dims[1] - 2)

    kind, setup_params, color_count = choose_rule_setup(train_shape, test_shape, rng)
    p1 = choose_palette(rng, color_count, set())
    p2 = choose_palette(rng, color_count, set(p1))
    built = _build_sprites(kind, setup_params, train_shape, test_shape, p1, p2, rng)
    if built is None:
        return None
    train_src, train_dst, test_src, test_dst = built

    inner_frame_color, outer_frame_color, bg_palette = _choose_frame_and_bg(set(p1) | set(p2), rng)

    train_outer_dims = make_outer_dims(train_frame_dims)
    test_outer_dims = make_outer_dims(test_frame_dims)
    top_is_train = rng.choice([True, False])
    top_dims = train_outer_dims if top_is_train else test_outer_dims
    bottom_dims = test_outer_dims if top_is_train else train_outer_dims
    outer_positions = choose_outer_positions(top_dims, bottom_dims, rng)
    if outer_positions is None:
        return None
    top_outer_pos, bottom_outer_pos = outer_positions

    train_outer_pos = top_outer_pos if top_is_train else bottom_outer_pos
    test_outer_pos = bottom_outer_pos if top_is_train else top_outer_pos

    train_slots = inner_slots(train_outer_pos, train_frame_dims)
    test_slots = inner_slots(test_outer_pos, test_frame_dims)
    rng.shuffle(train_slots)
    rng.shuffle(test_slots)

    frame_boxes = []
    outer_boxes = [
        (train_outer_pos[0], train_outer_pos[1], train_outer_pos[0] + train_outer_dims[0] - 1, train_outer_pos[1] + train_outer_dims[1] - 1),
        (test_outer_pos[0], test_outer_pos[1], test_outer_pos[0] + test_outer_dims[0] - 1, test_outer_pos[1] + test_outer_dims[1] - 1),
    ]
    frame_boxes.extend(outer_boxes)
    for (r0, c0), dims in [(train_slots[0], train_frame_dims), (train_slots[1], train_frame_dims), (test_slots[0], test_frame_dims), (test_slots[1], test_frame_dims)]:
        frame_boxes.append((r0, c0, r0 + dims[0] - 1, c0 + dims[1] - 1))

    grid = render_background(frame_boxes, bg_palette, rng)

    draw_frame(grid, train_outer_pos[0], train_outer_pos[1], train_outer_dims[0], train_outer_dims[1], outer_frame_color, None)
    draw_frame(grid, test_outer_pos[0], test_outer_pos[1], test_outer_dims[0], test_outer_dims[1], outer_frame_color, None)

    draw_frame(grid, train_slots[0][0], train_slots[0][1], train_frame_dims[0], train_frame_dims[1], inner_frame_color, train_src)
    draw_frame(grid, train_slots[1][0], train_slots[1][1], train_frame_dims[0], train_frame_dims[1], inner_frame_color, train_dst)

    blank_left = rng.choice([True, False])
    if blank_left:
        blank_slot, test_slot = test_slots[0], test_slots[1]
    else:
        test_slot, blank_slot = test_slots[0], test_slots[1]
    draw_frame(grid, test_slot[0], test_slot[1], test_frame_dims[0], test_frame_dims[1], inner_frame_color, test_src)
    draw_frame(
        grid,
        blank_slot[0],
        blank_slot[1],
        test_frame_dims[0],
        test_frame_dims[1],
        inner_frame_color,
        np.zeros(test_shape, dtype=int),
    )

    if not validate_scene(grid, test_dst):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
