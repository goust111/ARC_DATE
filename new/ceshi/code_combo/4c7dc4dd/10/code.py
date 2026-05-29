# Combo variant of 4c7dc4dd (index 10)
# combo_spec: D2/0+D8/3+D18/2
# Name: 多色框架中的配对与干扰排除

from itertools import combinations
import random

import numpy as np

from arc_utils import colors_present, draw_line, paste


# Design analysis for the combo variant
# =====================================
# a) Concretization and behavior trace
# - The base task had 4 same-size hollow frames: one blank, one test, and one
#   training pair found by matching interior palettes.
# - This combo changes two different points in the reasoning chain:
#   1) frame selection: there are now 5 frames, but only the four outer frames
#      belong to the analogy; the centered frame is a distractor.
#   2) role identification inside the real quartet: the two training frames
#      share one border color, while the test frame and the blank frame share a
#      second border color.
# - Concrete trace:
#   * Old logic on a 5-frame scene would inspect all detected frames together.
#     That is already unsafe because the distractor creates an extra non-blank
#     candidate.
#   * Even if an ablated solver first removed the center frame, old palette
#     matching would still be ambiguous because this generator makes all three
#     real non-blank corner frames reuse the same interior palette.
#   * New logic must therefore run as:
#       detect frames -> keep the four corner-most frames -> find the blank in
#       that quartet -> use border-color grouping there to split
#       {train_src, train_dst} from {test, blank} -> infer the sprite rule ->
#       apply it to the test sprite.
# - The distractor border may deliberately reuse one of the real border colors,
#   so "group all frames by border color" is also unsafe until the center frame
#   has been removed.
#
# b) Complexity calibration
# - Border grouping uses exactly two colors on the real quartet, each repeated
#   twice. This is visually salient and easy to count.
# - The distractor has only one extra frame and sits near the center, so the
#   structural change is clear without exploding the search space.
# - The interior transformation family is unchanged from the base task, keeping
#   the conceptual load focused on role discovery rather than inventing a new
#   sprite rule.
# - To make the border cue necessary instead of cosmetic, the generator forces
#   the real train/test sprites to share the same visible color signature.
#
# c) Hidden-assumption review
# - Original assumption: every detected frame belongs to the analogy. Broken by
#   D8/3; fixed by selecting the four corner frames first.
# - Original assumption: among non-blank frames, exactly one palette-matched
#   pair identifies training. Broken by D2/0 plus D18/2 because the test sprite
#   intentionally shares that palette too.
# - Original assumption: border colors are irrelevant decoration. Broken here:
#   inside the retained quartet, border equality is the only reliable cue for
#   training-vs-test grouping.
# - New combined assumption: the centered frame is the unique non-core frame and
#   the retained quartet occupies four distinct quadrants. The generator
#   enforces this geometrically.
#
# d) Placement legality
# - The four real frames are placed at separated corner anchors; the distractor
#   uses a central anchor. With frame sizes 6-8 on a 30x30 canvas these boxes
#   are non-overlapping and in-bounds by construction.
# - All frames share the same size so the visual family stays coherent, and the
#   solver can still use the base square-frame detector.
#
# e) Visual observability
# - A human can see the familiar outer 2x2 analogy plus one center intruder.
# - Inside the outer quartet, two border colors form a visible 2+2 split; one
#   color labels the training pair, the other labels the test+blank pair.
# - No invisible ordering by numeric color id is used. Only visible equality of
#   border colors and visible centrality of the distractor matter.
#
# f) Visual cleanliness
# - There is only one added layer of complexity: a single central distractor.
# - The distractor sprite uses a palette disjoint from the real quartet so it
#   does not visually fuse with the analogy sprites.
# - Frames never overlap, and interiors remain black-backed sprites inside clean
#   hollow squares.
#
# g) Change inferability
# - There is no free-form color remapping to infer. The human only needs to
#   notice repeated border-color equality and ignore the centered outlier frame.
# - The transformation itself is still learned structurally from the training
#   pair, exactly as in the base task.
#
# h) Anti-degeneration policy
# - Generation rejects any sample where a D8-only ablation
#   (remove center distractor but still pair by interior palette) succeeds.
# - Generation also rejects any sample where a D2+D18-only ablation
#   (use border grouping on all frames without first filtering the center
#   distractor) succeeds.
# - This makes both halves of the fusion causally necessary on the same input.
#
# i) New solver assumptions audit
# - `select_core_frame_indices()` assumes the true quartet occupies four
#   different quadrants and is farther from the grid center than the distractor.
#   The generator guarantees this.
# - `_roles_from_border_grouping()` assumes the retained quartet has exactly two
#   border colors in a 2+2 split, and that the blank shares its border color
#   with the test frame. The generator guarantees this.
# - The inference code still assumes one of the six base sprite rules explains
#   the training pair; `_build_core_sprites()` validates that before rendering.


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


def _find_blank_index(inners: list[np.ndarray]) -> int | None:
    for i, inner in enumerate(inners):
        if inner.size > 0 and np.all(inner == 0):
            return i
    if inners:
        return int(np.argmin([np.count_nonzero(inner) for inner in inners]))
    return None


def _roles_from_palette_match(inners: list[np.ndarray], blank_idx: int | None):
    if blank_idx is None:
        return None, None
    nonblank = [i for i in range(len(inners)) if i != blank_idx]
    sigs = []
    for i in nonblank:
        nz = tuple(sorted(set(int(x) for x in inners[i].ravel() if int(x) != 0)))
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
    if sorted(len(v) for v in groups.values()) != [2, 2]:
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


def _infer_best_template(src_a: np.ndarray, src_b: np.ndarray, test_inner: np.ndarray):
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


def solve_frames(
    grid: np.ndarray,
    frames: list[tuple[int, int, int, int, int, int]],
    *,
    prefer_border_grouping: bool,
    allow_palette_fallback: bool,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    if len(frames) < 3:
        return g.copy()

    inners = [g[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1 in frames]
    blank_idx = _find_blank_index(inners)
    if blank_idx is None:
        return g.copy()

    pair = None
    test_idx = None
    if prefer_border_grouping:
        pair, test_idx = _roles_from_border_grouping(frames, blank_idx)
    if (pair is None or test_idx is None) and allow_palette_fallback:
        pair, test_idx = _roles_from_palette_match(inners, blank_idx)
    if pair is None or test_idx is None:
        return inners[blank_idx].copy()

    src_a = inners[pair[0]]
    src_b = inners[pair[1]]
    test_inner = inners[test_idx]
    best = _infer_best_template(src_a, src_b, test_inner)
    if best is None:
        return test_inner.copy()
    _, kind, params = best
    return apply_template(kind, params, test_inner)


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


def solve_core_scene(grid: np.ndarray, *, prefer_border_grouping: bool, allow_palette_fallback: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    frames, core_idxs, _ignored = extract_scene_frames(g)
    if len(core_idxs) < 4:
        return g.copy()
    core_frames = [frames[i] for i in core_idxs]
    return solve_frames(
        g,
        core_frames,
        prefer_border_grouping=prefer_border_grouping,
        allow_palette_fallback=allow_palette_fallback,
    )


def all_frames_border_only_main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    return solve_frames(
        g,
        find_square_frames(g),
        prefer_border_grouping=True,
        allow_palette_fallback=False,
    )


def core_palette_only_main(grid: np.ndarray) -> np.ndarray:
    return solve_core_scene(grid, prefer_border_grouping=False, allow_palette_fallback=True)


def legacy_main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    return solve_frames(
        g,
        find_square_frames(g),
        prefer_border_grouping=False,
        allow_palette_fallback=True,
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the sprite analogy from the outer quartet, ignoring a central decoy and using border-color pairing.

    Input structure:
    - The grid contains five same-size hollow square frames on a noisy background.
    - Four corner frames are the real puzzle: one blank interior, one test sprite, and a training source/target pair.
    - One extra frame sits nearer the center and is a distractor; it may even reuse one of the real border colors.

    Rule:
    - First isolate the four corner frames and discard the centered distractor.
    - Within that retained quartet, the two frames sharing one border color form the training pair, while the blank frame shares the other border color with the test frame.
    - Infer the same sprite transformation as in the base task and apply it to the test sprite, returning only the transformed interior.

    Key insight:
    - The puzzle now has a two-stage visual cue: layout tells you which frame to ignore, and border-color equality inside the remaining quartet tells you which two sprites define the example transformation.
    """
    return solve_core_scene(grid, prefer_border_grouping=True, allow_palette_fallback=False)


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

CORE_TEST_BUILD_RETRIES = 40
DISTRACTOR_BORDER_MODES = ("train", "test", "unique")
DISTRACTOR_BORDER_WEIGHTS = (0.40, 0.40, 0.20)

GEN_MAX_ATTEMPTS = 600


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
                depth = max(min(rr, cc, n - 1 - rr, n - 1 - cc) for rr, cc in orb)
                orbits.append((depth, list(orb)))

    usable_orbits = [(depth, orb) for depth, orb in orbits if not any(keep0[r, c] for r, c in orb)]
    if not usable_orbits:
        raise ValueError("not enough usable orbits")

    max_depth = max(depth for depth, _ in usable_orbits)
    fill_choices = [orb for depth, orb in usable_orbits if depth == max_depth]
    fill_orbit = rng.choice(fill_choices)
    src[rng.choice(fill_orbit)] = fill_col

    accent_orbits = [
        orb
        for depth, orb in usable_orbits
        if orb != fill_orbit and len(orb) >= 2 and depth < max_depth
    ]
    if len(accent_orbits) < 2:
        accent_orbits = [orb for _depth, orb in usable_orbits if orb != fill_orbit and len(orb) >= 2]
    if len(accent_orbits) < 2:
        raise ValueError("not enough accent orbits")
    for orb in rng.sample(accent_orbits, 2):
        src[rng.choice(orb)] = accent_col
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


def _color_sig(arr: np.ndarray) -> tuple[int, ...]:
    return tuple(colors_present(arr, exclude={0}))


def _build_core_sprites(kind: str, setup_params: dict, n: int, palette: list[int], rng: random.Random):
    train_src, train_dst = build_training_pair(kind, setup_params, n, palette, rng)
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

    for _ in range(CORE_TEST_BUILD_RETRIES):
        test_src, _ = build_training_pair(kind, setup_params, n, palette, rng)
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


def _choose_distractor(kind: str, n: int, palette: list[int], rng: random.Random) -> np.ndarray:
    # Keep the distractor visually simple and unrelated to the real transform.
    for _ in range(20):
        if kind == "sym":
            arr = random_sparse_pattern(n, palette, rng)
        else:
            arr = random_sparse_pattern(n, palette, rng)
        if _valid_nz(arr):
            return arr
    raise ValueError("failed to build distractor sprite")


def _choose_frame_colors(
    used: set[int],
    rng: random.Random,
) -> tuple[int, int, int, str, list[int]]:
    frame_pool = [c for c in ALL_COLORS if c not in used]
    train_frame_color, test_frame_color, unique_distractor_color = rng.sample(frame_pool, 3)
    distractor_mode = rng.choices(DISTRACTOR_BORDER_MODES, weights=DISTRACTOR_BORDER_WEIGHTS, k=1)[0]
    if distractor_mode == "train":
        distractor_frame_color = train_frame_color
    elif distractor_mode == "test":
        distractor_frame_color = test_frame_color
    else:
        distractor_frame_color = unique_distractor_color
    bg_pool = [
        c
        for c in ALL_COLORS
        if c not in used and c not in {train_frame_color, test_frame_color, distractor_frame_color}
    ]
    bg_sizes = [k for k in BG_PERIOD_CHOICES if k <= len(bg_pool)]
    if not bg_sizes:
        raise ValueError("not enough colors left for background palette")
    bg_palette = rng.sample(bg_pool, rng.choice(bg_sizes))
    return train_frame_color, test_frame_color, distractor_frame_color, distractor_mode, bg_palette


def _assign_corner_slots(positions: list[tuple[int, int]], rng: random.Random) -> tuple[list[int], int, int]:
    # Force the first two non-blank frames in reading order to not both be
    # training, so palette-only matching on the core quartet fails.
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

    core_inners = [grid[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1 in core_frames]
    blank_indices = [i for i, inner in enumerate(core_inners) if np.all(inner == 0)]
    if len(blank_indices) != 1:
        return False
    blank_idx = blank_indices[0]

    groups: dict[int, list[int]] = {}
    for i, frame in enumerate(core_frames):
        groups.setdefault(int(frame[1]), []).append(i)
    if sorted(len(v) for v in groups.values()) != [2, 2]:
        return False
    blank_color = int(core_frames[blank_idx][1])
    if len(groups.get(blank_color, [])) != 2:
        return False

    nonblank_sigs = [_color_sig(inner) for inner in core_inners if np.count_nonzero(inner) > 0]
    if len(nonblank_sigs) != 3 or len(set(nonblank_sigs)) != 1:
        return False

    if not np.array_equal(main(grid), expected_output):
        return False
    if np.array_equal(core_palette_only_main(grid), expected_output):
        return False
    if np.array_equal(all_frames_border_only_main(grid), expected_output):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    s = rng.choice(FRAME_SIZE_CHOICES)
    n = s - 2

    kind, setup_params, color_count = choose_rule_setup(n, rng)
    core_palette = choose_palette(rng, color_count, set())
    distractor_color_count = rng.choice([1, 2])
    distractor_palette = choose_palette(rng, distractor_color_count, set(core_palette))

    built = _build_core_sprites(kind, setup_params, n, core_palette, rng)
    if built is None:
        return None
    train_src, train_dst, test_src, test_dst = built

    distractor = _choose_distractor(kind, n, distractor_palette, rng)
    if _color_sig(distractor) == _color_sig(train_src):
        return None

    try:
        train_frame_color, test_frame_color, distractor_frame_color, _mode, bg_palette = _choose_frame_colors(
            set(core_palette) | set(distractor_palette),
            rng,
        )
    except ValueError:
        return None

    positions = corner_positions(s, rng) + [center_position(s)]
    train_slots, test_slot, blank_slot = _assign_corner_slots(positions[:4], rng)

    sprites = [np.zeros((n, n), dtype=int) for _ in range(5)]
    sprites[train_slots[0]] = train_src
    sprites[train_slots[1]] = train_dst
    sprites[test_slot] = test_src
    sprites[blank_slot] = np.zeros((n, n), dtype=int)
    sprites[4] = distractor

    frame_colors = [test_frame_color] * 5
    frame_colors[train_slots[0]] = train_frame_color
    frame_colors[train_slots[1]] = train_frame_color
    frame_colors[blank_slot] = test_frame_color
    frame_colors[4] = distractor_frame_color

    grid = _render_scene(s, positions, sprites, frame_colors, bg_palette, rng)
    if not validate_scene(grid, test_dst):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
