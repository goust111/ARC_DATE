# Variant of 4c7dc4dd
# Dimension: D12 — 稳定色与变量色的区分 (transform)
# Change: 精灵内部有稳定色（变换前后位置不变）和变量色（参与变换的主色），通过比较训练对自动识别... → 颜色角色交换
#
# Design analysis
# ---------------
# variant:
#   core_change:
#     - Keep the framed-analogy scene structure from 4c7dc4dd.
#     - Make every generated train/test sprite use exactly two interior colors:
#       one unchanged carrier color and one transformed color.
#     - Relative to the original generator family, the transformed color is now the
#       color that used to play the "stable accent" role, while the old main color
#       stays fixed. The solver should therefore reason from observed behavior
#       ("which mask actually changes?"), not from a hard-coded color role.
#   concrete_behavior:
#     - fill:
#       src shows four corner seeds of color A inside a rectangle while color B
#       sits unchanged inside that rectangle; dst fills the rectangle with A but
#       preserves the B cells.
#     - endpoints:
#       src shows long A segments plus a few fixed B cells; dst keeps only the A
#       endpoints while the B cells remain untouched.
#     - invert0:
#       src swaps A with black everywhere except for fixed B cells; dst restores
#       the sparse A pattern while B is unchanged.
#   flow_trace:
#     - Original fill-family logic: dense main color moves, sparse accent stays.
#     - Swapped D12 fill here: sparse corner/fill color moves, carrier accent stays.
#     - In code this means the solver must apply the template to "the color whose
#       mask differs between train src/dst", even if that color is not the dominant
#       visual motif.
#   complexity_calibration:
#     - comp_checker has no meaningful unchanged non-zero color, and plain sym often
#       makes color-role swap visually ambiguous. Those branches are left supported
#       in the solver but are not sampled by the generator.
#     - Generation uses {fill, endpoints, invert0}; each rule repeats the same
#       two-color role pattern so the D12 change is visible in only 2-3 examples.
#   assumption_review:
#     - Frame finding, blank-frame detection, and pair matching by color set still
#       hold unchanged.
#     - The original stable-color detector already compares masks between train
#       src/dst; that remains valid after the role swap because the observed
#       unchanged color is still the one to freeze during application.
#     - fill scans row/column spans over all non-zero cells, so unchanged cells for
#       fill must be placed inside the intended filled rectangle; otherwise they
#       would alter the inferred span and create a different rule.
#     - endpoints computes degree only on the transformed color, so unchanged cells
#       must not overlap that color's pixels.
#   placement_legality:
#     - Keep the original 30x30 canvas, four non-overlapping same-size frames, one
#       frame per quadrant, and one blank interior.
#     - Training pair shares a two-color palette; test uses a disjoint two-color
#       palette with the same swapped-role structure.
#   observability_and_cleanliness:
#     - Every accepted sample must have exactly one unchanged non-zero color and one
#       changed non-zero color in the training pair.
#     - A counterfactual "old-role" application is computed on the test sprite by
#       forcing the rule onto the unchanged color; if that yields the same output,
#       the sample is rejected as visually degenerate.

import numpy as np
import random
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
    sigs = []
    for i in nonblank:
        inner = inners[i][1]
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
        counts = [(i, int(np.count_nonzero(inners[i][1]))) for i in nonblank]
        counts.sort(key=lambda t: t[1])
        pair = (counts[0][0], counts[1][0])

    remaining = [i for i in nonblank if i not in pair]
    if not remaining:
        return None
    test_idx = remaining[0]
    return {
        "frames": frames,
        "inners": inners,
        "blank_idx": blank_idx,
        "pair": pair,
        "test_idx": test_idx,
    }


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
        m0 = src == 0
        ma = src == active
        pred[m0] = active
        pred[ma] = 0
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
        m0 = x == 0
        ma = x == active_color
        y[m0] = active_color
        y[ma] = 0
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a rule from one framed sprite pair, then apply it to a third framed sprite.

    Input structure:
    - The 30x30 grid contains four non-overlapping hollow square frames of one shared border color.
    - One frame is blank inside, two frames form the training pair, and the remaining non-blank frame is the test input.
    - In the D12 variant, each non-blank sprite uses two interior colors: one color keeps the same footprint across the training pair while the other color is the one that actually changes.

    Rule:
    - Find the training pair by matching interior color sets, and treat the last non-blank frame as the test sprite.
    - From the training pair, infer both the transform type (fill, endpoints, invert-0, symmetry family, etc.) and which color is the changing color.
    - Apply that same transform to the corresponding changing color in the test sprite, while preserving the test sprite's unchanged companion color.

    Key insight:
    - This is still an analogy puzzle, but color role matters: the visually dominant color is not guaranteed to be the one that transforms, so the solver must compare train source and destination masks before applying the rule.
    """
    g = np.asarray(grid, dtype=int)
    parsed = parse_scene(g)
    if parsed is None:
        return g.copy()

    src_a = parsed["inners"][parsed["pair"][0]][1]
    src_b = parsed["inners"][parsed["pair"][1]][1]
    test_inner = parsed["inners"][parsed["test_idx"]][1]

    best = None
    for src, dst in ((src_a, src_b), (src_b, src_a)):
        tpl = infer_template(src, dst)
        if tpl is None:
            continue
        kind, params = tpl
        frozen = set(params.get("unchanged", set()))
        a_src = pick_active_color(src, frozen)
        a_dst = pick_active_color(dst, frozen)
        score = 1
        if kind == "fill" and a_src is not None and a_dst is not None:
            if int(np.sum(dst == a_dst)) >= int(np.sum(src == a_src)):
                score += 2
        if kind == "endpoints" and a_src is not None and a_dst is not None:
            if int(np.sum(dst == a_dst)) <= int(np.sum(src == a_src)):
                score += 2
        src_n = int(np.sum((src != 0) & ~np.isin(src, list(frozen))))
        test_n = int(np.sum((test_inner != 0) & ~np.isin(test_inner, list(frozen))))
        score2 = score * 100 - abs(src_n - test_n)
        if best is None or score2 > best[0]:
            best = (score2, kind, params)

    if best is None:
        return test_inner.copy()

    _, kind, params = best
    return apply_template(kind, params, test_inner)


# --------------- generator (D12 role-swap variant) ---------------

# Canvas
H = W = 30

# Palette pools
SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
ALL_COLORS = list(range(10))
BG_PERIOD_CHOICES = (4, 5)
BG_PERTURB_FRACTION = (0.30, 0.50)
BG_EXTRA_NOISE = (0,)

# Frame layout
FRAME_SIZE_CHOICES = (6, 7, 8)
QUAD_LOW_MIN, QUAD_LOW_MAX = 1, 8
QUAD_BOTTOM_MARGIN = 8

# Only keep rules where the swapped unchanged/active color roles are visible.
RULES = ["endpoints", "fill", "invert0"]

# Sprite pixel-count bounds
SPRITE_NZ_MIN, SPRITE_NZ_MAX = 3, 32

# Rule-specific build bounds
FILL_RECT_AREA_MIN, FILL_RECT_AREA_MAX = 6, 24
ENDPOINTS_LINE_LEN_MIN = 3
INVERT0_FIXED_MAX = 5
INVERT0_ACTIVE_MAX = 12

# Misc sampling
FIXED_CELLS_MIN, FIXED_CELLS_MAX = 1, 3

# Outer retry loop
GEN_MAX_ATTEMPTS = 400


def choose_palette(rng: random.Random, count: int, forbidden: set[int]) -> list[int]:
    pool = [c for c in SPRITE_COLORS if c not in forbidden]
    if len(pool) < count:
        raise ValueError("not enough sprite colors")
    return rng.sample(pool, count)


def _sample_cells(rng: random.Random, cells: list[tuple[int, int]], lo: int, hi: int) -> list[tuple[int, int]]:
    if len(cells) < lo:
        raise ValueError("not enough cells")
    k = rng.randint(lo, min(hi, len(cells)))
    return rng.sample(cells, k)


def _valid_role_swap_pair(src: np.ndarray, dst: np.ndarray) -> bool:
    sig_src = tuple(colors_present(src, exclude={0}))
    sig_dst = tuple(colors_present(dst, exclude={0}))
    if len(sig_src) != 2 or sig_src != sig_dst:
        return False
    frozen = unchanged_colors(src, dst)
    if len(frozen) != 1:
        return False
    active = pick_active_color(src, frozen)
    if active is None:
        return False
    return not np.array_equal(src == active, dst == active)


def build_fill_pair(n: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    active, fixed = colors
    dst = np.zeros((n, n), dtype=int)
    while True:
        h = rng.randint(2, n)
        w = rng.randint(2, n)
        area = h * w
        if FILL_RECT_AREA_MIN <= area <= FILL_RECT_AREA_MAX and area >= 5:
            break
    r0 = rng.randint(0, n - h)
    c0 = rng.randint(0, n - w)

    src = np.zeros_like(dst)
    corners = [(r0, c0), (r0, c0 + w - 1), (r0 + h - 1, c0), (r0 + h - 1, c0 + w - 1)]
    for r, c in corners:
        src[r, c] = active
        dst[r, c] = active
    for r in range(r0, r0 + h):
        for c in range(c0, c0 + w):
            dst[r, c] = active

    fixed_candidates = [(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w) if (r, c) not in corners]
    fixed_cells = _sample_cells(rng, fixed_candidates, FIXED_CELLS_MIN, FIXED_CELLS_MAX)
    for r, c in fixed_cells:
        src[r, c] = fixed
        dst[r, c] = fixed
    return src, dst


def build_endpoints_pair(n: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    active, fixed = colors
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
    draw_line(src, r1, c1, end_r=r1, end_c=c1 + len1 - 1, color=active)
    draw_line(src, r2, c2, end_r=r2, end_c=c2 + len2 - 1, color=active)
    dst[r1, c1] = active
    dst[r1, c1 + len1 - 1] = active
    dst[r2, c2] = active
    dst[r2, c2 + len2 - 1] = active

    occupied = {(r1, c) for c in range(c1, c1 + len1)} | {(r2, c) for c in range(c2, c2 + len2)}
    fixed_candidates = [(r, c) for r in range(n) for c in range(n) if (r, c) not in occupied]
    fixed_cells = _sample_cells(rng, fixed_candidates, FIXED_CELLS_MIN, FIXED_CELLS_MAX)
    for r, c in fixed_cells:
        src[r, c] = fixed
        dst[r, c] = fixed
    return src, dst


def build_invert0_pair(n: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    active, fixed = colors
    all_cells = [(r, c) for r in range(n) for c in range(n)]
    fixed_n = rng.randint(2, min(INVERT0_FIXED_MAX, max(2, n)))
    fixed_cells = set(rng.sample(all_cells, fixed_n))
    remaining = [cell for cell in all_cells if cell not in fixed_cells]
    active_hi = max(2, min(len(remaining) - 2, INVERT0_ACTIVE_MAX))
    if active_hi < 2:
        raise ValueError("not enough active cells")
    active_n = rng.randint(2, active_hi)
    active_cells = set(rng.sample(remaining, active_n))
    dst = np.zeros((n, n), dtype=int)
    for r, c in fixed_cells:
        dst[r, c] = fixed
    for r, c in active_cells:
        dst[r, c] = active
    src = dst.copy()
    src[dst == 0] = active
    src[dst == active] = 0
    return src, dst


def choose_rule_setup(_n: int, rng: random.Random) -> tuple[str, dict, int]:
    kind = rng.choice(RULES)
    return kind, {}, 2


def build_training_pair(kind: str, params: dict, n: int, colors: list[int], rng: random.Random):
    if kind == "fill":
        return build_fill_pair(n, colors, rng)
    if kind == "endpoints":
        return build_endpoints_pair(n, colors, rng)
    if kind == "invert0":
        return build_invert0_pair(n, colors, rng)
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
    nonblank_sigs = [tuple(colors_present(inner, exclude={0})) for inner in inners]
    nonblank_sigs = [s for s in nonblank_sigs if s]
    dup_count = sum(
        1 for i in range(len(nonblank_sigs)) for j in range(i + 1, len(nonblank_sigs))
        if nonblank_sigs[i] == nonblank_sigs[j]
    )
    if dup_count != 1:
        return False

    train_src = parsed["inners"][parsed["pair"][0]][1]
    train_dst = parsed["inners"][parsed["pair"][1]][1]
    if not (_valid_role_swap_pair(train_src, train_dst) or _valid_role_swap_pair(train_dst, train_src)):
        return False
    return np.array_equal(main(grid), expected_output)


def _valid_nz(arr: np.ndarray) -> bool:
    return SPRITE_NZ_MIN <= int(np.count_nonzero(arr)) <= SPRITE_NZ_MAX


def _counterfactual_old_role(kind: str, params: dict, x: np.ndarray, colors: list[int]) -> np.ndarray:
    active, fixed = colors
    return apply_kind_to_color(kind, params, x, fixed, {active})


def _build_sprites(kind: str, setup_params: dict, n: int, p1: list[int], p2: list[int], rng: random.Random):
    train_src, train_dst = build_training_pair(kind, setup_params, n, p1, rng)
    if not _valid_role_swap_pair(train_src, train_dst):
        return None

    inferred = infer_template(train_src, train_dst)
    if inferred is None or inferred[0] != kind:
        return None
    inferred_kind, inferred_params = inferred

    test_src, _unused = build_training_pair(kind, setup_params, n, p2, rng)
    test_dst = apply_template(inferred_kind, inferred_params, test_src)

    if np.array_equal(test_dst, test_src):
        return None
    if not _valid_role_swap_pair(test_src, test_dst):
        return None
    if np.array_equal(_counterfactual_old_role(inferred_kind, inferred_params, test_src, p2), test_dst):
        return None
    if np.array_equal(_counterfactual_old_role(inferred_kind, inferred_params, train_src, p1), train_dst):
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
