from __future__ import annotations

# Combo variant of 4c7dc4dd (index 7)
# combo_spec: D4/0+D18/2
# Name: 颜色标记目标框架填充
#
# Design analysis
# ===============
# a) Concretization and behavior trace
# - The original task shows 4 equal hollow square frames on a 30x30 noisy
#   background. Two non-blank interiors form a training pair, one non-blank
#   interior is the test sprite, and one interior is blank.
# - In this combo, the blank frame becomes the output carrier (D4), and frame
#   border colors now group roles (D18):
#   * the two training frames share one border color;
#   * the test frame and the blank frame share another border color.
# - Concrete trace:
#   * Original behavior: find the blank only to exclude it, identify the
#     training pair by interior palette match, infer the sprite rule, and return
#     only the transformed test crop.
#   * Combo behavior: find the blank and keep its box, use border-color grouping
#     to identify the training pair and the test/blank pair, infer the same
#     sprite rule, apply it to the test sprite, and write the result into the
#     blank frame while returning the whole 30x30 scene.
# - The visible difference is therefore twofold: border colors explicitly mark
#   roles, and outputs modify only the previously empty frame instead of
#   returning a tight crop.
#
# b) Complexity calibration
# - Border grouping uses exactly 2 colors in a 2+2 split. This is strong enough
#   to be visible without creating many categories.
# - To make the grouping cue necessary rather than decorative, the generator
#   samples two palette modes for the test sprite:
#   * `same`: test sprite uses the same non-zero palette as the training pair;
#   * `disjoint`: test sprite uses a disjoint palette.
# - `same` mode is favored for most rules so many samples are ambiguous under
#   the original palette-matching role logic. `sym_fill` stays in `disjoint`
#   mode because otherwise accepted samples collapse too often to the legacy
#   role guesser. The combo still keeps the original transform family
#   unchanged, so difficulty rises only from role decoding and output
#   placement.
#
# c) Hidden-assumption review of the original main()
# - Original assumption: training roles are recoverable from matching interior
#   color sets. This breaks in `same` mode, so border-color grouping must be the
#   primary role-identification path.
# - Original assumption: failure paths may return a compact interior. Under D4,
#   every path must return a full-grid object.
# - Original assumption: frame coordinates can be discarded after inference.
#   Under D4 they must be preserved so the transformed test sprite can be pasted
#   into the blank frame.
# - Original assumption: transparent pasting is enough. Under D4 we must clear
#   the blank interior first so zero cells in the predicted sprite remain black.
# - New combo interaction: the test frame and the blank frame share a border
#   color, so the same cue identifies both the input side of the analogy and the
#   destination carrier. The solver therefore needs a consistent blank-first role
#   pass, and the generator must guarantee exactly one blank interior and a 2+2
#   frame-color split.
#
# d) Placement legality
# - The generator keeps the original quadrant-based placement, equal frame size,
#   and non-overlap constraints.
# - All 4 frames use the same outer size S in {6,7,8}, so the transformed test
#   interior always fits the blank frame interior exactly.
# - The blank frame is chosen after slot assignment and its expected output is
#   constructed by writing only inside that frame.
#
# e) Visual observability
# - Across examples, humans can see two repeated border colors. One color always
#   groups the two example frames; the other always groups the test frame with
#   the empty destination frame.
# - In `same` mode, three non-blank interiors may share the same visible palette,
#   so the border cue becomes causally necessary and easy to notice.
# - No invisible ordering by numeric color id is used; only visible equality of
#   border color matters.
#
# f) Visual cleanliness
# - The combo adds no extra markers beyond border-color grouping.
# - Only the blank interior changes between input and output; background, border
#   outlines, and the three informative interiors remain fixed.
# - Because frames remain separated by quadrant jitter, the grouped colors are
#   readable without overlap or crowding.
#
# g) Change inferability
# - No new sprite-color remapping is introduced; sprite colors still participate
#   only through the original analogy rule.
# - The only new color semantics is binary equality of frame borders, repeated
#   twice per scene, which gives enough support for quick inference from a few
#   examples.
#
# h) Generator anti-degeneration policy
# - In `same` mode, the generator places the test frame earlier than the
#   training pair in frame-order, making the legacy palette-based role finder
#   likely to pick the wrong pair.
# - Such scenes are rejected if a legacy full-grid solver that ignores border
#   colors still produces the same output. `sym_fill` is forced into
#   `disjoint` mode because its accepted `same`-mode samples are too sparse,
#   which would otherwise starve that rule branch. This keeps D18 functionally
#   active while preserving balanced generator coverage.
#
# i) New solver assumptions audit
# - `_roles_from_border_grouping()` assumes exactly two frame colors and a 2+2
#   split. `generate_input()` enforces that.
# - `write_into_blank_frame()` assumes the blank and test interiors have equal
#   shape. Equal frame sizes guarantee this.
# - `main()` assumes there is exactly one blank frame. The generator constructs
#   one all-zero interior and validation rejects anything else.

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
        if len(picked) >= 6:
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


def _find_blank_index(inners: list[np.ndarray]) -> int | None:
    blank_idx = None
    for i, inner in enumerate(inners):
        if inner.size > 0 and np.all(inner == 0):
            blank_idx = i
            break
    if blank_idx is None and inners:
        blank_idx = int(np.argmin([np.count_nonzero(inner) for inner in inners]))
    return blank_idx


def _roles_from_palette_match(inners: list[np.ndarray], blank_idx: int | None):
    if blank_idx is None:
        return None, None
    nonblank = [i for i in range(len(inners)) if i != blank_idx]
    sigs = []
    for i in nonblank:
        inner = inners[i]
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
        counts = [(i, int(np.count_nonzero(inners[i]))) for i in nonblank]
        counts.sort(key=lambda t: t[1])
        if len(counts) >= 2:
            pair = (counts[0][0], counts[1][0])
    if pair is None:
        return None, None

    remaining = [i for i in nonblank if i not in pair]
    if not remaining:
        return pair, None
    return pair, remaining[0]


def _roles_from_border_grouping(
    frames: list[tuple[int, int, int, int, int, int]],
    blank_idx: int | None,
):
    if blank_idx is None:
        return None, None

    groups: dict[int, list[int]] = {}
    for i, frame in enumerate(frames):
        groups.setdefault(int(frame[1]), []).append(i)
    if len(groups) != 2:
        return None, None

    blank_color = int(frames[blank_idx][1])
    test_group = groups.get(blank_color, [])
    test_candidates = [i for i in test_group if i != blank_idx]
    other_colors = [c for c in groups if c != blank_color]
    if len(other_colors) != 1:
        return None, None
    pair = tuple(groups[other_colors[0]])

    if len(test_group) != 2 or len(test_candidates) != 1 or len(pair) != 2:
        return None, None
    if blank_idx in pair:
        return None, None
    return pair, test_candidates[0]


def _choose_best_rule(
    src_a: np.ndarray,
    src_b: np.ndarray,
    test_inner: np.ndarray,
):
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
    return best


def write_into_blank_frame(
    grid: np.ndarray,
    frame_box: tuple[int, int, int, int],
    sprite: np.ndarray,
) -> np.ndarray:
    r0, c0, r1, c1 = frame_box
    out = np.array(grid, copy=True)
    out[r0 + 1 : r1, c0 + 1 : c1] = 0
    paste(out, sprite, r0 + 1, c0 + 1)
    return out


def solve_with_role_strategy(
    grid: np.ndarray,
    *,
    prefer_border_grouping: bool,
    fill_blank_frame: bool,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    frames = find_square_frames(g)
    if len(frames) < 3:
        return g.copy()

    boxes = [(r0, c0, r1, c1) for _, _, r0, c0, r1, c1 in frames]
    inners = [g[r0 + 1 : r1, c0 + 1 : c1] for r0, c0, r1, c1 in boxes]
    blank_idx = _find_blank_index(inners)

    pair = None
    test_idx = None
    if prefer_border_grouping:
        pair, test_idx = _roles_from_border_grouping(frames, blank_idx)
    if pair is None or test_idx is None:
        pair, test_idx = _roles_from_palette_match(inners, blank_idx)

    if blank_idx is None or pair is None or test_idx is None:
        return g.copy() if fill_blank_frame else np.zeros((1, 1), dtype=int)

    blank_box = boxes[blank_idx]
    blank_inner = inners[blank_idx]
    src_a = inners[pair[0]]
    src_b = inners[pair[1]]
    test_inner = inners[test_idx]
    if blank_inner.shape != test_inner.shape:
        return g.copy() if fill_blank_frame else test_inner.copy()

    best = _choose_best_rule(src_a, src_b, test_inner)
    if best is None:
        return g.copy() if fill_blank_frame else test_inner.copy()

    _, kind, params = best
    transformed = apply_template(kind, params, test_inner)
    if not fill_blank_frame:
        return transformed
    return write_into_blank_frame(g, blank_box, transformed)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a framed sprite analogy, then write the answer into the empty same-colored target frame.

    Input structure:
    - The grid contains four non-overlapping hollow square frames of equal size on a noisy background.
    - Two non-blank frames share one border color and form the training source/destination example pair.
    - A third non-blank frame and the one blank frame share a different border color; the non-blank one is the test sprite and the blank one is the destination canvas.

    Rule:
    - Use border-color grouping to separate the training pair from the test/blank pair, with the blank interior marking the output location.
    - Infer the interior transformation shown by the training pair (reflection, fill, endpoint extraction, inversion, checker complement, or symmetry fill).
    - Apply the same transformation to the test sprite and place the result inside the blank frame, returning the full grid.

    Key insight:
    - The border colors now carry role information, and the empty frame is no longer just something to ignore. The solver must combine both cues: one color says which two frames teach the rule, while the other color links the test sprite to the frame where the completed answer belongs.
    """
    return solve_with_role_strategy(grid, prefer_border_grouping=True, fill_blank_frame=True)


def legacy_fill_main(grid: np.ndarray) -> np.ndarray:
    return solve_with_role_strategy(grid, prefer_border_grouping=False, fill_blank_frame=True)


def legacy_compact_main(grid: np.ndarray) -> np.ndarray:
    return solve_with_role_strategy(grid, prefer_border_grouping=False, fill_blank_frame=False)


# --------------- generator ---------------

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

# Rule set
RULES = ["endpoints", "fill", "sym_fill", "comp_checker", "invert0", "sym"]

# Sprite pixel-count bounds
SPRITE_NZ_MIN, SPRITE_NZ_MAX = 3, 32

# Variant-specific palette relation
SAME_TEST_PALETTE_PROB = 0.7

# sym_fill
SYMFILL_KEEP0_PROB = 0.4

# Rule-specific build bounds
FILL_RECT_AREA_MIN, FILL_RECT_AREA_MAX = 5, 25
ENDPOINTS_LINE_LEN_MIN = 3
COMP_CHECKER_M_MAX = 10
INVERT0_STABLE_MAX = 3
INVERT0_VAR_MAX = 12
SYM_BUILD_RETRIES = 30
SPARSE_COUNT_MAX = 10

# Outer retry loop
GEN_MAX_ATTEMPTS = 500


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
    if expected_output.shape != grid.shape:
        return False

    frame_groups: dict[int, list[int]] = {}
    for i, frame in enumerate(frames):
        frame_groups.setdefault(int(frame[1]), []).append(i)
    if sorted(len(v) for v in frame_groups.values()) != [2, 2]:
        return False

    inners = [grid[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1 in frames]
    blank_indices = [i for i, inner in enumerate(inners) if np.all(inner == 0)]
    if len(blank_indices) != 1:
        return False
    blank_idx = blank_indices[0]
    blank_color = int(frames[blank_idx][1])
    same_group = frame_groups[blank_color]
    if len(same_group) != 2:
        return False
    test_candidates = [i for i in same_group if i != blank_idx]
    if len(test_candidates) != 1:
        return False

    train_group = [idxs for color, idxs in frame_groups.items() if color != blank_color]
    if len(train_group) != 1 or len(train_group[0]) != 2:
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


def _build_sprites(
    kind: str,
    setup_params: dict,
    n: int,
    p1: list[int],
    p2: list[int],
    palette_mode: str,
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
    test_sig = tuple(colors_present(test_src, exclude={0}))
    if palette_mode == "same":
        if test_sig != train_sig:
            return None
    else:
        if test_sig == train_sig:
            return None

    return train_src, train_dst, test_src, test_dst


def _choose_frame_and_bg(used: set[int], rng: random.Random) -> tuple[int, int, list[int]]:
    frame_pool = [c for c in ALL_COLORS if c not in used]
    train_frame_color, test_frame_color = rng.sample(frame_pool, 2)
    bg_pool = [c for c in ALL_COLORS if c not in used and c not in {train_frame_color, test_frame_color}]
    bg_sizes = [k for k in BG_PERIOD_CHOICES if k <= len(bg_pool)]
    if not bg_sizes:
        raise ValueError("not enough colors left for background palette")
    bg_palette = rng.sample(bg_pool, rng.choice(bg_sizes))
    return train_frame_color, test_frame_color, bg_palette


def _assign_slots(
    positions: list[tuple[int, int]],
    palette_mode: str,
    rng: random.Random,
) -> tuple[list[int], int, int]:
    order = sorted(range(4), key=lambda i: positions[i])
    if palette_mode == "same":
        blank_slot = rng.choice(order[1:])
        nonblank = [i for i in order if i != blank_slot]
        test_slot = nonblank[0]
        train_slots = nonblank[1:]
        return train_slots, test_slot, blank_slot

    train_slots = sorted(rng.sample(order, 2), key=order.index)
    remaining = [i for i in order if i not in train_slots]
    test_slot = rng.choice(remaining)
    blank_slot = [i for i in remaining if i != test_slot][0]
    return train_slots, test_slot, blank_slot


def _render_scene(
    s: int,
    positions: list[tuple[int, int]],
    sprites: list[np.ndarray],
    frame_colors: list[int],
    bg_palette: list[int],
    rng: random.Random,
) -> np.ndarray:
    frame_boxes = [(r0, c0, r0 + s - 1, c0 + s - 1) for r0, c0 in positions]
    grid = render_background(frame_boxes, bg_palette, rng)
    for (r0, c0), sprite, frame_color in zip(positions, sprites, frame_colors):
        draw_frame(grid, r0, c0, s, frame_color, sprite)
    return grid


def _choose_test_palette_mode(kind: str, rng: random.Random) -> str:
    # `sym_fill` almost always collapses back to the legacy role logic in
    # `same` mode, so keep it in the disjoint branch to preserve rule coverage.
    if kind == "sym_fill":
        return "disjoint"
    return "same" if rng.random() < SAME_TEST_PALETTE_PROB else "disjoint"


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    s = rng.choice(FRAME_SIZE_CHOICES)
    n = s - 2

    kind, setup_params, color_count = choose_rule_setup(n, rng)
    palette_mode = _choose_test_palette_mode(kind, rng)
    p1 = choose_palette(rng, color_count, set())
    p2 = list(p1) if palette_mode == "same" else choose_palette(rng, color_count, set(p1))

    built = _build_sprites(kind, setup_params, n, p1, p2, palette_mode, rng)
    if built is None:
        return None
    train_src, train_dst, test_src, test_dst = built

    train_frame_color, test_frame_color, bg_palette = _choose_frame_and_bg(set(p1) | set(p2), rng)

    positions = quadrant_positions(s, rng)
    train_slots, test_slot, blank_slot = _assign_slots(positions, palette_mode, rng)

    sprites = [np.zeros((n, n), dtype=int) for _ in range(4)]
    sprites[train_slots[0]] = train_src
    sprites[train_slots[1]] = train_dst
    sprites[test_slot] = test_src
    sprites[blank_slot] = np.zeros((n, n), dtype=int)

    frame_colors = [test_frame_color] * 4
    frame_colors[train_slots[0]] = train_frame_color
    frame_colors[train_slots[1]] = train_frame_color

    grid = _render_scene(s, positions, sprites, frame_colors, bg_palette, rng)
    blank_r0, blank_c0 = positions[blank_slot]
    expected_output = write_into_blank_frame(
        grid,
        (blank_r0, blank_c0, blank_r0 + s - 1, blank_c0 + s - 1),
        test_dst,
    )
    if not validate_scene(grid, expected_output):
        return None
    if palette_mode == "same" and np.array_equal(legacy_fill_main(grid), expected_output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
