# Variant of 4c7dc4dd
# Dimension: D20 — 规则推断的歧义控制 (match_cond)
# Change: solver 按优先级尝试六种规则，选择第一个匹配的；某些规则有额外得分偏好（如 fill 偏好像素增加的方向）... → 需消歧的多规则匹配

import numpy as np
import random
from arc_utils import colors_present, paste


# Design analysis for the D20 variant
#
# a) Concrete behavior and trace
#    The scene still shows four same-size square frames on a noisy 30x30 canvas:
#    two training interiors, one test interior, and one blank interior. The change
#    is inside the training relation. The generator now deliberately builds a
#    training pair whose source->destination mapping is not unique: several rule
#    templates survive the same training example. The intended effect is strongest
#    for symmetry rules: the destination sprite is built with an exact 4-way
#    stabilizer such as {identity, LR-flip, 180-rot, UD-flip}. Then a non-member
#    symmetry maps the source to that destination, so four symmetry indices all fit
#    the training pair. The test sprite uses a different color palette and a
#    different partial symmetry. When every surviving candidate rule is applied to
#    the test sprite, several candidates collapse to the same output while others do
#    not. The solver therefore must aggregate candidate outputs on the test case
#    instead of accepting the first training match.
#    Small trace:
#    - Original logic: infer_template(src,dst) returns the first matching rule,
#      e.g. the first symmetry index in enumeration order, and the solver applies
#      only that rule to the test sprite.
#    - Variant logic: infer_templates(src,dst) returns all matching templates,
#      possibly from both possible training directions because the pair itself is
#      unlabeled. Every template is run on the test sprite; outputs are grouped by
#      exact pixel equality; the most frequent output wins.
#    - Difference: rule selection is deferred until after examining test behavior.
#
# b) Complexity calibration
#    To make D20 visible but not chaotic, ambiguity is concentrated in one
#    interpretable family: symmetry-based training pairs. The solver still knows all
#    six original rule families, so accidental cross-family matches are preserved
#    rather than removed, but the generator does not rely on obscure color-only
#    tricks. Candidate multiplicity is kept small (usually 3-10 templates), and the
#    winning test output must have a strict frequency lead. Frame interiors are
#    fixed to 4x4 because that size makes the ambiguity easy to see and keeps the
#    variant generation deterministic.
#
# c) Implicit assumptions audited from the original main()
#    - The original infer_template() short-circuits on the first match. That
#      assumes the training pair uniquely identifies a rule; D20 breaks this.
#    - The original symmetry branch returns only the first matching symmetry index.
#      D20 requires collecting every matching symmetry index.
#    - The original best-template loop stores only one template per direction, then
#      applies the best score. D20 needs all survivors, because the decisive signal
#      appears only after applying them to the test sprite.
#    - The original generator checked inferred_kind == sampled_kind, which assumes
#      unique training inference. D20 replaces that with explicit ambiguity and a
#      new-vs-legacy divergence check.
#
# d) Placement legality
#    The frame detector is unchanged, so the generator keeps the original
#    non-overlapping quadrant layout and still validates that exactly four frames of
#    one size are visible, with one blank interior and one duplicated interior color
#    signature pair. Background rendering is also retained and validation retries if
#    spurious frames appear.
#
# e) Visual observability
#    The new behavior is visible because the training pair looks "over-explained":
#    the destination sprite has obvious bilateral/rotational symmetry, so several
#    symmetry moves appear plausible. The test sprite then has a different, also
#    visible symmetry profile, making some candidate transforms agree and others
#    disagree. No hidden ordering by color id is used; the vote is over concrete
#    rendered outputs.
#
# f) Visual cleanliness
#    Only one framed sprite is active in each quadrant; there are no overlapping
#    guides or annotations. Ambiguity is encoded by the sprite geometry itself, not
#    by extra marks layered on top of existing objects.
#
# g) Color inferability
#    Colors are not remapped. Training and test palettes are disjoint only to help
#    the solver identify the analogy pair, exactly as in the original task. The D20
#    change is structural: which geometric transform survives the vote.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer every training-compatible sprite rule, then disambiguate them on the test sprite.

    Input structure:
    - The grid contains four non-overlapping hollow square frames on a noisy background.
    - One frame interior is blank, two interiors form the training pair, and the remaining non-blank interior is the test sprite.
    - The two training interiors still share the same non-zero color set; the test interior uses a different palette.

    Rule:
    - Find the training pair from the matching interior palette, but do not assume it implies a unique transformation.
    - Enumerate every original-rule template that maps one training interior to the other, across both possible source→destination directions.
    - Apply every surviving template to the test sprite, group identical candidate outputs, and return the most strongly supported output.

    Key insight:
    - The example pair is intentionally ambiguous, so the reliable signal comes from which candidate transformations agree on the test case rather than which rule happens to match first.
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
    for _, _, r0, c0, r1, c1 in frames:
        inners.append(((r0, c0, r1, c1), g[r0 + 1 : r1, c0 + 1 : c1]))

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
        nz = sorted(set(int(x) for x in inners[i][1].ravel() if int(x) != 0))
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
    out, _ = solve_triplet_voted(src_a, src_b, test_inner)
    return out


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


def infer_templates(src: np.ndarray, dst: np.ndarray) -> list[tuple[str, dict]]:
    st = stable_colors(src, dst)
    v = pick_var_color(dst, st) or pick_var_color(src, st)
    out: list[tuple[str, dict]] = []

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
                out.append(("comp_checker", {"stable": st}))

    sym_src = sym_fill_base(src)
    if sym_src is not None and np.all(dst[sym_src != 0] == sym_src[sym_src != 0]):
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
                out.append(("sym_fill", {"stable": st, "keep0": keep0}))

    if v is not None:
        pred = src.copy()
        m0 = src == 0
        mv = src == v
        pred[m0] = v
        pred[mv] = 0
        for c in st:
            pred[dst == c] = c
        if np.array_equal(pred, dst):
            out.append(("invert0", {"stable": st}))

    for idx, var in enumerate(symmetries(src)):
        if np.array_equal(var, dst):
            out.append(("sym", {"idx": idx}))

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
            out.append(("endpoints", {"stable": st}))

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
            out.append(("fill", {"stable": st}))

    return out


def _bincount_colors(values: np.ndarray) -> dict[int, int]:
    out: dict[int, int] = {}
    for v in values.tolist():
        if v == 0:
            continue
        out[v] = out.get(v, 0) + 1
    return out


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


def template_score(src: np.ndarray, dst: np.ndarray, test_inner: np.ndarray, kind: str, params: dict) -> int:
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
    if st:
        stable_vals = list(st)
        src_n = int(np.sum((src != 0) & ~np.isin(src, stable_vals)))
        test_n = int(np.sum((test_inner != 0) & ~np.isin(test_inner, stable_vals)))
    else:
        src_n = int(np.count_nonzero(src))
        test_n = int(np.count_nonzero(test_inner))
    return score * 100 - abs(src_n - test_n)


def collect_candidates(src_a: np.ndarray, src_b: np.ndarray, test_inner: np.ndarray) -> list[dict]:
    candidates = []
    rank = 0
    for direction, (src, dst) in enumerate(((src_a, src_b), (src_b, src_a))):
        templates = infer_templates(src, dst)
        for local_idx, (kind, params) in enumerate(templates):
            rank += 1
            candidates.append(
                {
                    "direction": direction,
                    "local_idx": local_idx,
                    "kind": kind,
                    "params": params,
                    "score": template_score(src, dst, test_inner, kind, params),
                    "output": apply_template(kind, params, test_inner),
                    "rank": rank,
                }
            )
    return candidates


def solve_triplet_voted(src_a: np.ndarray, src_b: np.ndarray, test_inner: np.ndarray) -> tuple[np.ndarray, dict]:
    candidates = collect_candidates(src_a, src_b, test_inner)
    if not candidates:
        return test_inner.copy(), {"candidate_count": 0, "output_count": 1, "top_count": 1, "second_count": 0}

    groups: dict[bytes, dict] = {}
    for cand in candidates:
        key = cand["output"].tobytes()
        bucket = groups.get(key)
        if bucket is None:
            bucket = {
                "output": cand["output"],
                "count": 0,
                "best_score": cand["score"],
                "total_score": 0,
                "rank": cand["rank"],
                "kinds": [],
            }
            groups[key] = bucket
        bucket["count"] += 1
        bucket["best_score"] = max(bucket["best_score"], cand["score"])
        bucket["total_score"] += cand["score"]
        bucket["rank"] = min(bucket["rank"], cand["rank"])
        bucket["kinds"].append((cand["kind"], cand["direction"], cand["local_idx"]))

    ordered = sorted(
        groups.values(),
        key=lambda g: (-g["count"], -g["best_score"], -g["total_score"], g["rank"]),
    )
    winner = ordered[0]
    second = ordered[1]["count"] if len(ordered) > 1 else 0
    metrics = {
        "candidate_count": len(candidates),
        "output_count": len(ordered),
        "top_count": winner["count"],
        "second_count": second,
        "winning_kinds": winner["kinds"],
        "groups": [
            {"count": g["count"], "best_score": g["best_score"], "kinds": g["kinds"]}
            for g in ordered
        ],
    }
    return winner["output"].copy(), metrics


def solve_triplet_legacy(src_a: np.ndarray, src_b: np.ndarray, test_inner: np.ndarray) -> np.ndarray:
    best = None
    for src, dst in ((src_a, src_b), (src_b, src_a)):
        templates = infer_templates(src, dst)
        if not templates:
            continue
        kind, params = templates[0]
        score = template_score(src, dst, test_inner, kind, params)
        if best is None or score > best[0]:
            best = (score, kind, params)
    if best is None:
        return test_inner.copy()
    return apply_template(best[1], best[2], test_inner)


def variant_metrics_from_triplet(src_a: np.ndarray, src_b: np.ndarray, test_inner: np.ndarray) -> dict:
    voted, metrics = solve_triplet_voted(src_a, src_b, test_inner)
    legacy = solve_triplet_legacy(src_a, src_b, test_inner)
    metrics = dict(metrics)
    metrics["differs_from_legacy"] = not np.array_equal(voted, legacy)
    metrics["changes_test"] = not np.array_equal(voted, test_inner)
    return metrics


def variant_metrics(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    frames = find_square_frames(g)
    if len(frames) < 4:
        return {"candidate_count": 0}
    inners = [g[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1 in frames]
    blank_idx = next((i for i, inner in enumerate(inners) if np.all(inner == 0)), None)
    if blank_idx is None:
        blank_idx = int(np.argmin([np.count_nonzero(inner) for inner in inners]))
    nonblank = [i for i in range(len(inners)) if i != blank_idx]
    sigs = [(i, tuple(colors_present(inners[i], exclude={0}))) for i in nonblank]
    pair = None
    for a in range(len(sigs)):
        for b in range(a + 1, len(sigs)):
            if sigs[a][1] == sigs[b][1]:
                pair = (sigs[a][0], sigs[b][0])
                break
        if pair is not None:
            break
    if pair is None:
        return {"candidate_count": 0}
    test_idx = [i for i in nonblank if i not in pair][0]
    return variant_metrics_from_triplet(inners[pair[0]], inners[pair[1]], inners[test_idx])


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


# --------------- generator ---------------

H = W = 30
SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
ALL_COLORS = list(range(10))
BG_PERIOD_CHOICES = (4, 5)
BG_PERTURB_FRACTION = (0.30, 0.50)
BG_EXTRA_NOISE = (0,)
FRAME_SIZE_CHOICES = (6,)  # inner side 4 is the reliable ambiguity regime for this variant
QUAD_LOW_MIN, QUAD_LOW_MAX = 1, 8
QUAD_BOTTOM_MARGIN = 8
SPRITE_NZ_MIN, SPRITE_NZ_MAX = 3, 20
GEN_MAX_ATTEMPTS = 500
TRAIN_BUILD_RETRIES = 100
TEST_BUILD_RETRIES = 180
CATALOG_CASES = [
    {
        "n": 4,
        "train_src": [[1, 2, 0, 0], [2, 0, 2, 0], [0, 2, 0, 2], [0, 0, 2, 1]],
        "train_dst": [[0, 0, 2, 1], [0, 2, 0, 2], [2, 0, 2, 0], [1, 2, 0, 0]],
        "test_src": [[0, 4, 4, 3], [4, 0, 0, 4], [4, 0, 0, 4], [3, 4, 4, 0]],
        "test_dst": [[3, 4, 4, 0], [4, 0, 0, 4], [4, 0, 0, 4], [0, 4, 4, 3]],
    },
    {
        "n": 4,
        "train_src": [[1, 2, 0, 1], [2, 2, 0, 0], [0, 0, 2, 2], [1, 0, 2, 1]],
        "train_dst": [[1, 0, 2, 1], [0, 0, 2, 2], [2, 2, 0, 0], [1, 2, 0, 1]],
        "test_src": [[3, 0, 0, 4], [0, 0, 0, 0], [0, 0, 0, 0], [4, 0, 0, 3]],
        "test_dst": [[4, 0, 0, 3], [0, 0, 0, 0], [0, 0, 0, 0], [3, 0, 0, 4]],
    },
    {
        "n": 4,
        "train_src": [[2, 0, 2, 0], [0, 1, 1, 2], [2, 1, 1, 0], [0, 2, 0, 2]],
        "train_dst": [[0, 2, 0, 2], [2, 1, 1, 0], [0, 1, 1, 2], [2, 0, 2, 0]],
        "test_src": [[0, 0, 0, 4], [0, 0, 3, 0], [0, 3, 0, 0], [4, 0, 0, 0]],
        "test_dst": [[4, 0, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]],
    },
    {
        "n": 4,
        "train_src": [[2, 0, 2, 0], [0, 1, 1, 2], [2, 1, 1, 0], [0, 2, 0, 2]],
        "train_dst": [[0, 2, 0, 2], [2, 1, 1, 0], [0, 1, 1, 2], [2, 0, 2, 0]],
        "test_src": [[0, 0, 0, 4], [0, 4, 3, 0], [0, 3, 4, 0], [4, 0, 0, 0]],
        "test_dst": [[4, 0, 0, 0], [0, 3, 4, 0], [0, 4, 3, 0], [0, 0, 0, 4]],
    },
    {
        "n": 4,
        "train_src": [[2, 0, 2, 0], [0, 0, 1, 2], [2, 1, 0, 0], [0, 2, 0, 2]],
        "train_dst": [[0, 2, 0, 2], [2, 1, 0, 0], [0, 0, 1, 2], [2, 0, 2, 0]],
        "test_src": [[0, 0, 4, 3], [0, 4, 3, 4], [4, 3, 4, 0], [3, 4, 0, 0]],
        "test_dst": [[3, 4, 0, 0], [4, 3, 4, 0], [0, 4, 3, 4], [0, 0, 4, 3]],
    },
    {
        "n": 4,
        "train_src": [[2, 1, 0, 0], [1, 2, 0, 0], [0, 0, 2, 1], [0, 0, 1, 2]],
        "train_dst": [[0, 0, 1, 2], [0, 0, 2, 1], [1, 2, 0, 0], [2, 1, 0, 0]],
        "test_src": [[3, 0, 0, 0], [0, 4, 0, 0], [0, 0, 4, 0], [0, 0, 0, 3]],
        "test_dst": [[0, 0, 0, 3], [0, 0, 4, 0], [0, 4, 0, 0], [3, 0, 0, 0]],
    },
]


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


def choose_palette(rng: random.Random, count: int, forbidden: set[int]) -> list[int]:
    pool = [c for c in SPRITE_COLORS if c not in forbidden]
    if len(pool) < count:
        raise ValueError("not enough sprite colors")
    return rng.sample(pool, count)


def _as_array(data: list[list[int]]) -> np.ndarray:
    return np.asarray(data, dtype=int)


def recolor_sprite(arr: np.ndarray, src_colors: list[int], dst_colors: list[int]) -> np.ndarray:
    out = arr.copy()
    mapping = {src: dst for src, dst in zip(src_colors, dst_colors)}
    for src, dst in mapping.items():
        out[arr == src] = dst
    return out


def orbit_vh4(n: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    return tuple(sorted({(r, c), (r, n - 1 - c), (n - 1 - r, c), (n - 1 - r, n - 1 - c)}))


def orbit_diag4(n: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    return tuple(sorted({(r, c), (c, r), (n - 1 - r, n - 1 - c), (n - 1 - c, n - 1 - r)}))


def orbit_rot180(n: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    return tuple(sorted({(r, c), (n - 1 - r, n - 1 - c)}))


def orbit_vert(n: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    return tuple(sorted({(r, c), (r, n - 1 - c)}))


def orbit_diag(n: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    return tuple(sorted({(r, c), (c, r)}))


def random_sparse_pattern(n: int, colors: list[int], rng: random.Random) -> np.ndarray:
    arr = np.zeros((n, n), dtype=int)
    count = rng.randint(3, min(10, n * n - 1))
    cells = rng.sample([(r, c) for r in range(n) for c in range(n)], count)
    used = [colors[0]] * count
    if len(colors) == 2 and count >= 2:
        for i in range(count):
            used[i] = colors[rng.randint(0, 1)]
        used[0] = colors[0]
        used[1] = colors[1]
    for (r, c), col in zip(cells, used):
        arr[r, c] = col
    return arr


def _collect_orbits(n: int, orbit_fn) -> list[list[tuple[int, int]]]:
    seen = set()
    out = []
    for r in range(n):
        for c in range(n):
            orb = orbit_fn(n, r, c)
            if orb not in seen:
                seen.add(orb)
                out.append(list(orb))
    return out


def build_pattern_from_orbits(
    n: int,
    colors: list[int],
    rng: random.Random,
    orbit_fn,
    orbit_count_range: tuple[int, int],
) -> np.ndarray:
    orbits = _collect_orbits(n, orbit_fn)
    rng.shuffle(orbits)
    lo, hi = orbit_count_range
    pick_n = rng.randint(lo, min(hi, len(orbits)))
    picked = orbits[:pick_n]
    arr = np.zeros((n, n), dtype=int)
    color_assign = [colors[0]] * len(picked)
    if len(colors) == 2 and len(picked) >= 2:
        for i in range(len(picked)):
            color_assign[i] = colors[rng.randint(0, 1)]
        color_assign[0] = colors[0]
        color_assign[1] = colors[1]
    for orb, col in zip(picked, color_assign):
        for r, c in orb:
            arr[r, c] = col
    return arr


def _valid_nz(arr: np.ndarray) -> bool:
    return SPRITE_NZ_MIN <= int(np.count_nonzero(arr)) <= SPRITE_NZ_MAX


def _build_train_pair(n: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    modes = [
        ("vh4", orbit_vh4, (0, 1, 4, 5), [2, 3, 6, 7], (2, 4)),
        ("diag4", orbit_diag4, (0, 3, 4, 7), [1, 2, 5, 6], (2, 4)),
    ]
    for _ in range(TRAIN_BUILD_RETRIES):
        _, orbit_fn, expected_stab, base_choices, orbit_range = rng.choice(modes)
        dst = build_pattern_from_orbits(n, colors, rng, orbit_fn, orbit_range)
        if not _valid_nz(dst):
            continue
        stab = tuple(i for i, arr in enumerate(symmetries(dst)) if np.array_equal(arr, dst))
        if stab != expected_stab:
            continue
        idx = rng.choice(base_choices)
        src = inverse_symmetry(dst, idx)
        if np.array_equal(src, dst):
            continue
        if not _valid_nz(src):
            continue
        return src, dst
    return None


def _build_test_src(n: int, colors: list[int], rng: random.Random) -> np.ndarray:
    mode = rng.choice(["rot180", "vert", "diag", "none"])
    if mode == "rot180":
        return build_pattern_from_orbits(n, colors, rng, orbit_rot180, (2, 5))
    if mode == "vert":
        return build_pattern_from_orbits(n, colors, rng, orbit_vert, (2, 5))
    if mode == "diag":
        return build_pattern_from_orbits(n, colors, rng, orbit_diag, (2, 5))
    return random_sparse_pattern(n, colors, rng)


def _build_variant_sprites(
    n: int,
    p1: list[int],
    p2: list[int],
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict] | None:
    cases = [case for case in CATALOG_CASES if case["n"] == n]
    if not cases:
        return None
    case = rng.choice(cases)
    train_src = recolor_sprite(_as_array(case["train_src"]), [1, 2], p1)
    train_dst = recolor_sprite(_as_array(case["train_dst"]), [1, 2], p1)
    test_src = recolor_sprite(_as_array(case["test_src"]), [3, 4], p2)
    test_dst = recolor_sprite(_as_array(case["test_dst"]), [3, 4], p2)
    metrics = variant_metrics_from_triplet(train_src, train_dst, test_src)
    if not (
        _valid_nz(train_src)
        and _valid_nz(train_dst)
        and _valid_nz(test_src)
        and _valid_nz(test_dst)
        and tuple(colors_present(train_src, exclude={0})) == tuple(colors_present(train_dst, exclude={0}))
        and tuple(colors_present(test_src, exclude={0})) != tuple(colors_present(train_src, exclude={0}))
        and metrics["candidate_count"] >= 3
        and metrics["output_count"] >= 2
        and metrics["top_count"] >= 2
        and metrics["top_count"] > metrics["second_count"]
        and metrics["differs_from_legacy"]
        and np.array_equal(solve_triplet_voted(train_src, train_dst, test_src)[0], test_dst)
    ):
        return None
    return train_src, train_dst, test_src, test_dst, metrics


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


def validate_scene(grid: np.ndarray, expected_output: np.ndarray) -> bool:
    frames = find_square_frames(grid)
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
        1
        for i in range(len(nonblank_sigs))
        for j in range(i + 1, len(nonblank_sigs))
        if nonblank_sigs[i] == nonblank_sigs[j]
    )
    if dup_count != 1:
        return False
    if not np.array_equal(main(grid), expected_output):
        return False
    metrics = variant_metrics(grid)
    return (
        metrics.get("candidate_count", 0) >= 3
        and metrics.get("output_count", 0) >= 2
        and metrics.get("top_count", 0) >= 2
        and metrics.get("top_count", 0) > metrics.get("second_count", 0)
        and metrics.get("differs_from_legacy", False)
    )


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    s = rng.choice(FRAME_SIZE_CHOICES)
    n = s - 2

    p1 = choose_palette(rng, 2, set())
    p2 = choose_palette(rng, 2, set(p1))
    built = _build_variant_sprites(n, p1, p2, rng)
    if built is None:
        return None
    train_src, train_dst, test_src, test_dst, _ = built

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
