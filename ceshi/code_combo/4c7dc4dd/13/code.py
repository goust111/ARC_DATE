from __future__ import annotations

# Combo variant of 4c7dc4dd (index 13)
# combo_spec: D4/0+D16/0+D20/0
# Name: 双训练验证消歧后原位填充

import random
from collections import Counter

import numpy as np

from arc_utils import colors_present, paste


# Design analysis for the combo variant
#
# a) Concrete behavior and trace
#    The original task had 4 frames: one training pair, one test sprite, and one
#    blank frame was only an excluded distractor. The solver inferred one rule
#    from one unlabeled pair and returned a cropped transformed sprite.
#    This combo turns the scene into a 2x3 layout of 6 equal frames:
#    - top row:    training source 1, training destination 1, test source
#    - bottom row: training source 2, training destination 2, blank output box
#    Each training row uses its own color, but both rows demonstrate the same
#    geometric transformation. Each row is intentionally ambiguous on its own:
#    it supports two symmetry candidates rather than one. The correct candidate
#    is the only one that appears in both rows, so both training rows must be
#    consulted. The solver then applies every surviving row-compatible candidate
#    to the test sprite, groups identical outputs, and writes the winning output
#    into the blank frame, returning the full 30x30 grid.
#    Small trace:
#    - original: infer one rule from one pair -> apply to test -> return n x n crop
#    - combo: sort 6 frames into 2 rows x 3 cols -> collect candidates from both
#      left->right training rows -> vote by test outputs -> paste winner into the
#      blank frame -> return full scene
#    - visible difference: there are now two example rows and the answer appears
#      inside the formerly empty bottom-right frame.
#
# b) Complexity calibration
#    The rule family is narrowed to symmetry transforms because that is the most
#    controllable way to create human-visible ambiguity without hidden metadata.
#    Every sample still requires analogy reasoning, but the ambiguity is kept to
#    exactly two candidates per training row. Two different ambiguity patterns are
#    used (for example vertical-symmetric and horizontal-symmetric destinations),
#    so one symmetry candidate is repeated across both rows while the wrong one is
#    row-specific. This makes the D16 change materially useful instead of cosmetic.
#    Frame size is fixed to 6 (inner size 4): that is the most reliable regime for
#    building exact-two-candidate symmetric sprites while still fitting 6 frames
#    cleanly on a 30x30 board.
#
# c) Implicit assumptions reviewed from the original main()
#    - The original solver assumed one training pair was enough; this combo breaks
#      that and requires aggregating evidence from two separate rows.
#    - The original solver treated pair direction as unlabeled. Here the layout
#      makes direction explicit: within each training row, left is source and
#      middle is destination. The generator guarantees this.
#    - The original solver could discard frame coordinates after extracting the
#      test sprite. Here it must preserve the blank frame box because the output is
#      the full grid with an in-place fill.
#    - The original solver only needed 4 frames and could identify the pair by a
#      single duplicated palette. Here it assumes a 2x3 frame layout, exactly one
#      blank interior, and two duplicated training palettes corresponding to the
#      two left->middle rows. The generator validates all of these.
#
# d) Placement legality
#    The generator uses six non-overlapping frame anchors arranged in a jittered
#    2x3 lattice. Because frame size is fixed to 6, all boxes stay in bounds with
#    clear horizontal and vertical gaps. Validation rejects any sample where the
#    frame detector sees extra accidental frames.
#
# e) Visual observability
#    A human can read the puzzle as two horizontal examples stacked over each
#    other, with the test on the top-right and the blank answer box on the
#    bottom-right. Each training destination is visibly symmetric enough that a
#    second symmetry move also looks plausible, but the second training row uses a
#    different symmetry profile, so only one transform is supported twice.
#    Nothing relies on color ids or invisible ordering; the repeated support is
#    encoded by visible geometry.
#
# f) Visual cleanliness
#    No guides or markers overlap sprites. Each frame contains one sparse interior,
#    the background stays outside frame boxes, and only the blank frame changes
#    between input and output. The two training rows share structure but use
#    different colors, which helps the analogy read cleanly.
#
# g) Change inferability
#    Colors are not remapped. Each row keeps one sprite color from source to
#    destination, and the test uses a third color. The entire inference burden is
#    geometric: find the repeated symmetry transform and place its result into the
#    blank box.
#
# Main-side assumptions that the generator now guarantees
#    - exactly 6 detected frames of the same size
#    - sorted row-major, they form a stable 2x3 layout
#    - top-left->top-middle and bottom-left->bottom-middle are the training rows
#    - top-right is non-blank test input, bottom-right is blank destination
#    - each training row yields exactly two symmetry candidates
#    - exactly one symmetry candidate key appears in both rows
#    - that shared candidate produces the unique highest-vote test output


# ------- CFG -------
H = W = 30
FRAME_SIZE = 6
INNER_SIZE = FRAME_SIZE - 2
SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
ALL_COLORS = list(range(10))
BG_PERIOD_CHOICES = (4, 5)
BG_PERTURB_FRACTION = (0.30, 0.50)
BG_EXTRA_NOISE = (0,)
GEN_MAX_ATTEMPTS = 600
PATTERN_BUILD_RETRIES = 120
TEST_BUILD_RETRIES = 200
SPRITE_NZ_MIN, SPRITE_NZ_MAX = 3, 20
# Only keep intended symmetries that admit a unique cross-row disambiguation under
# this generator's exact-two-candidate construction.
VIABLE_INTENDED_SPECS = {
    2: [("diag", "antidiag")],
    3: [("vertical", "horizontal")],
    5: [("vertical", "diag")],
    6: [("horizontal", "diag")],
    7: [("vertical", "horizontal"), ("horizontal", "antidiag")],
}


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
        pts = set((int(r), int(c)) for r, c in coords)
        keep = np.zeros_like(src, dtype=bool)
        for r, c in pts:
            deg = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    if (r + dr, c + dc) in pts:
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
        pts = set((int(r), int(c)) for r, c in coords)
        keep = np.zeros_like(x, dtype=bool)
        for r, c in pts:
            deg = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    if (r + dr, c + dc) in pts:
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


def canonical_key(kind: str, params: dict) -> tuple:
    if kind == "sym":
        return ("sym", int(params["idx"]))
    if kind == "sym_fill":
        keep0 = params.get("keep0")
        if isinstance(keep0, np.ndarray):
            return ("sym_fill", keep0.shape, keep0.tobytes())
    return (kind,)


def dedup_templates(templates: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    out = []
    seen = set()
    for kind, params in templates:
        key = canonical_key(kind, params)
        if key in seen:
            continue
        seen.add(key)
        out.append((kind, params))
    return out


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


def layout_frames(frames: list[tuple[int, int, int, int, int, int]]):
    if len(frames) < 6:
        return None
    frames = sorted(frames[:6], key=lambda t: (t[2], t[3]))
    top = sorted(frames[:3], key=lambda t: t[3])
    bottom = sorted(frames[3:6], key=lambda t: t[3])
    if top[-1][2] >= bottom[0][2]:
        return None
    return top + bottom


def solve_triplets_to_blank(
    train_rows: list[tuple[np.ndarray, np.ndarray]],
    test_inner: np.ndarray,
) -> tuple[np.ndarray, dict]:
    candidates = []
    for pair_id, (src, dst) in enumerate(train_rows):
        templates = dedup_templates(infer_templates(src, dst))
        for local_idx, (kind, params) in enumerate(templates):
            candidates.append(
                {
                    "pair_id": pair_id,
                    "local_idx": local_idx,
                    "kind": kind,
                    "params": params,
                    "key": canonical_key(kind, params),
                    "score": template_score(src, dst, test_inner, kind, params),
                    "output": apply_template(kind, params, test_inner),
                    "rank": len(candidates),
                }
            )

    if not candidates:
        return test_inner.copy(), {
            "candidate_count": 0,
            "output_count": 1,
            "top_count": 1,
            "second_count": 0,
            "key_support_max": 0,
            "shared_key_count": 0,
        }

    key_support = Counter(cand["key"] for cand in candidates)
    groups: dict[bytes, dict] = {}
    for cand in candidates:
        okey = cand["output"].tobytes()
        bucket = groups.get(okey)
        if bucket is None:
            bucket = {
                "output": cand["output"],
                "count": 0,
                "best_score": cand["score"],
                "total_score": 0,
                "rank": cand["rank"],
                "pair_ids": set(),
                "keys": set(),
                "kinds": [],
            }
            groups[okey] = bucket
        bucket["count"] += 1
        bucket["best_score"] = max(bucket["best_score"], cand["score"])
        bucket["total_score"] += cand["score"]
        bucket["rank"] = min(bucket["rank"], cand["rank"])
        bucket["pair_ids"].add(cand["pair_id"])
        bucket["keys"].add(cand["key"])
        bucket["kinds"].append((cand["kind"], cand["pair_id"], cand["local_idx"]))

    ordered = sorted(
        groups.values(),
        key=lambda g: (-g["count"], -len(g["pair_ids"]), -g["best_score"], -g["total_score"], g["rank"]),
    )
    winner = ordered[0]
    second = ordered[1]["count"] if len(ordered) > 1 else 0
    metrics = {
        "candidate_count": len(candidates),
        "output_count": len(ordered),
        "top_count": winner["count"],
        "second_count": second,
        "key_support_max": max(key_support.values()),
        "shared_key_count": sum(1 for v in key_support.values() if v >= 2),
        "winning_kinds": winner["kinds"],
        "row_candidate_counts": [sum(1 for cand in candidates if cand["pair_id"] == i) for i in range(len(train_rows))],
    }
    return winner["output"].copy(), metrics


def solve_single_row_legacy(src: np.ndarray, dst: np.ndarray, test_inner: np.ndarray) -> np.ndarray:
    templates = dedup_templates(infer_templates(src, dst))
    if not templates:
        return test_inner.copy()
    kind, params = templates[0]
    return apply_template(kind, params, test_inner)


def combo_metrics_from_parts(
    train1_src: np.ndarray,
    train1_dst: np.ndarray,
    train2_src: np.ndarray,
    train2_dst: np.ndarray,
    test_inner: np.ndarray,
) -> dict:
    row1 = dedup_templates(infer_templates(train1_src, train1_dst))
    row2 = dedup_templates(infer_templates(train2_src, train2_dst))
    voted, metrics = solve_triplets_to_blank([(train1_src, train1_dst), (train2_src, train2_dst)], test_inner)
    key1 = {canonical_key(kind, params) for kind, params in row1}
    key2 = {canonical_key(kind, params) for kind, params in row2}
    shared = key1 & key2
    all_sym = all(kind == "sym" for kind, _ in row1 + row2)
    out = dict(metrics)
    out["row1_candidate_count"] = len(row1)
    out["row2_candidate_count"] = len(row2)
    out["row1_keys"] = sorted(key1)
    out["row2_keys"] = sorted(key2)
    out["shared_keys"] = sorted(shared)
    out["all_sym"] = all_sym
    out["changes_test"] = not np.array_equal(voted, test_inner)
    out["differs_from_single_row"] = not np.array_equal(voted, solve_single_row_legacy(train1_src, train1_dst, test_inner))
    return out


def combo_metrics(grid: np.ndarray) -> dict:
    frames = layout_frames(find_square_frames(grid))
    if frames is None or len(frames) != 6:
        return {"candidate_count": 0}
    inners = [grid[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1 in frames]
    return combo_metrics_from_parts(inners[0], inners[1], inners[3], inners[4], inners[2])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use two framed training analogies to resolve an ambiguous symmetry rule, then fill the empty frame.

    Input structure:
    - The grid contains six same-size hollow square frames arranged as two visual rows of three.
    - In each row, the left and middle frames form a source→destination training example with the same interior palette.
    - The top-right frame is the test sprite, and the bottom-right frame is an all-zero blank destination box.

    Rule:
    - Read both training rows left-to-right, enumerate every original-rule template that fits each row, and keep the full multi-row candidate pool instead of trusting one row alone.
    - Apply every surviving candidate to the test sprite, then group identical predicted outputs so candidates supported by both rows reinforce the same result.
    - Write the winning transformed test sprite into the blank frame and return the whole grid.

    Key insight:
    - Each training row is deliberately underdetermined by itself; the correct transform is the one whose test-time output is jointly supported across the two rows, and that answer belongs inside the empty frame rather than as a cropped patch.
    """
    g = np.asarray(grid, dtype=int)
    frames = layout_frames(find_square_frames(g))
    if frames is None or len(frames) != 6:
        return g.copy()

    boxes = [(r0, c0, r1, c1) for _, _, r0, c0, r1, c1 in frames]
    inners = [g[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1 in frames]

    train1_src, train1_dst, test_inner = inners[0], inners[1], inners[2]
    train2_src, train2_dst, blank_inner = inners[3], inners[4], inners[5]
    blank_box = boxes[5]

    if blank_inner.shape != test_inner.shape or not np.all(blank_inner == 0):
        return g.copy()

    predicted, _ = solve_triplets_to_blank(
        [(train1_src, train1_dst), (train2_src, train2_dst)],
        test_inner,
    )
    return write_into_blank_frame(g, blank_box, predicted)


def choose_palette(rng: random.Random, count: int, forbidden: set[int]) -> list[int]:
    pool = [c for c in SPRITE_COLORS if c not in forbidden]
    if len(pool) < count:
        raise ValueError("not enough sprite colors")
    return rng.sample(pool, count)


def _valid_nz(arr: np.ndarray) -> bool:
    return SPRITE_NZ_MIN <= int(np.count_nonzero(arr)) <= SPRITE_NZ_MAX


def stabilizer_indices(arr: np.ndarray) -> tuple[int, ...]:
    return tuple(i for i, var in enumerate(symmetries(arr)) if np.array_equal(var, arr))


def random_sparse_pattern(n: int, color: int, rng: random.Random) -> np.ndarray:
    arr = np.zeros((n, n), dtype=int)
    count = rng.randint(3, min(8, n * n - 1))
    for r, c in rng.sample([(r, c) for r in range(n) for c in range(n)], count):
        arr[r, c] = color
    return arr


def build_vertical_symmetric(n: int, color: int, rng: random.Random) -> np.ndarray:
    arr = np.zeros((n, n), dtype=int)
    domain = [(r, c) for r in range(n) for c in range((n + 1) // 2)]
    pick_n = rng.randint(2, min(5, len(domain)))
    for r, c in rng.sample(domain, pick_n):
        arr[r, c] = color
        arr[r, n - 1 - c] = color
    return arr


def build_horizontal_symmetric(n: int, color: int, rng: random.Random) -> np.ndarray:
    arr = np.zeros((n, n), dtype=int)
    domain = [(r, c) for r in range((n + 1) // 2) for c in range(n)]
    pick_n = rng.randint(2, min(5, len(domain)))
    for r, c in rng.sample(domain, pick_n):
        arr[r, c] = color
        arr[n - 1 - r, c] = color
    return arr


def build_diag_symmetric(n: int, color: int, rng: random.Random) -> np.ndarray:
    arr = np.zeros((n, n), dtype=int)
    domain = [(r, c) for r in range(n) for c in range(r, n)]
    pick_n = rng.randint(2, min(5, len(domain)))
    for r, c in rng.sample(domain, pick_n):
        arr[r, c] = color
        arr[c, r] = color
    return arr


def build_antidiag_symmetric(n: int, color: int, rng: random.Random) -> np.ndarray:
    arr = np.zeros((n, n), dtype=int)
    domain = [(r, c) for r in range(n) for c in range(n) if r + c <= n - 1]
    pick_n = rng.randint(2, min(5, len(domain)))
    for r, c in rng.sample(domain, pick_n):
        arr[r, c] = color
        arr[n - 1 - c, n - 1 - r] = color
    return arr


SYMMETRY_BUILDERS = {
    "vertical": build_vertical_symmetric,
    "horizontal": build_horizontal_symmetric,
    "diag": build_diag_symmetric,
    "antidiag": build_antidiag_symmetric,
}


def build_ambiguous_pair(
    n: int,
    color: int,
    intended_idx: int,
    mode: str,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, set[tuple]] | None:
    builder = SYMMETRY_BUILDERS[mode]
    for _ in range(PATTERN_BUILD_RETRIES):
        dst = builder(n, color, rng)
        if not _valid_nz(dst):
            continue
        stab = stabilizer_indices(dst)
        if len(stab) != 2 or 0 not in stab:
            continue
        src = inverse_symmetry(dst, intended_idx)
        if np.array_equal(src, dst) or not _valid_nz(src):
            continue
        templates = dedup_templates(infer_templates(src, dst))
        keys = {canonical_key(kind, params) for kind, params in templates}
        if len(keys) != 2:
            continue
        if any(kind != "sym" for kind, _ in templates):
            continue
        if ("sym", intended_idx) not in keys:
            continue
        return src, dst, keys
    return None


def build_test_sprite(
    n: int,
    color: int,
    rng: random.Random,
    expected_count: int,
) -> np.ndarray | None:
    for _ in range(TEST_BUILD_RETRIES):
        test = random_sparse_pattern(n, color, rng)
        if int(np.count_nonzero(test)) != expected_count:
            continue
        if len(stabilizer_indices(test)) != 1:
            continue
        return test
    return None


def frame_positions(s: int, rng: random.Random) -> list[tuple[int, int]]:
    x_ranges = [(1, 3), (12, 14), (21, 23)]
    y_ranges = [(2, 5), (18, 21)]
    return [
        (rng.randint(y_ranges[0][0], y_ranges[0][1]), rng.randint(x_ranges[0][0], x_ranges[0][1])),
        (rng.randint(y_ranges[0][0], y_ranges[0][1]), rng.randint(x_ranges[1][0], x_ranges[1][1])),
        (rng.randint(y_ranges[0][0], y_ranges[0][1]), rng.randint(x_ranges[2][0], x_ranges[2][1])),
        (rng.randint(y_ranges[1][0], y_ranges[1][1]), rng.randint(x_ranges[0][0], x_ranges[0][1])),
        (rng.randint(y_ranges[1][0], y_ranges[1][1]), rng.randint(x_ranges[1][0], x_ranges[1][1])),
        (rng.randint(y_ranges[1][0], y_ranges[1][1]), rng.randint(x_ranges[2][0], x_ranges[2][1])),
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


def choose_frame_and_bg(used: set[int], rng: random.Random) -> tuple[int, list[int]]:
    frame_color = rng.choice([c for c in ALL_COLORS if c not in used])
    bg_pool = [c for c in ALL_COLORS if c not in used and c != frame_color]
    bg_palette = rng.sample(bg_pool, rng.choice(BG_PERIOD_CHOICES))
    return frame_color, bg_palette


def validate_scene(grid: np.ndarray, expected_output: np.ndarray) -> bool:
    frames = layout_frames(find_square_frames(grid))
    if frames is None or len(frames) != 6:
        return False
    sizes = {size for size, _, *_ in frames}
    if sizes != {FRAME_SIZE}:
        return False
    if expected_output.shape != grid.shape:
        return False
    inners = [grid[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1 in frames]
    if not np.all(inners[5] == 0):
        return False
    if tuple(colors_present(inners[0], exclude={0})) != tuple(colors_present(inners[1], exclude={0})):
        return False
    if tuple(colors_present(inners[3], exclude={0})) != tuple(colors_present(inners[4], exclude={0})):
        return False
    if tuple(colors_present(inners[0], exclude={0})) == tuple(colors_present(inners[3], exclude={0})):
        return False
    metrics = combo_metrics_from_parts(inners[0], inners[1], inners[3], inners[4], inners[2])
    diff = expected_output != grid
    blank_mask = np.zeros_like(grid, dtype=bool)
    _, _, r0, c0, r1, c1 = frames[5]
    blank_mask[r0 + 1 : r1, c0 + 1 : c1] = True
    if not np.any(diff & blank_mask):
        return False
    if np.any(diff & ~blank_mask):
        return False
    return (
        np.array_equal(main(grid), expected_output)
        and metrics.get("row1_candidate_count", 0) == 2
        and metrics.get("row2_candidate_count", 0) == 2
        and metrics.get("all_sym", False)
        and metrics.get("shared_key_count", 0) == 1
        and metrics.get("key_support_max", 0) == 2
        and metrics.get("top_count", 0) == 2
        and metrics.get("second_count", 0) == 1
        and metrics.get("output_count", 0) >= 3
        and metrics.get("changes_test", False)
        and metrics.get("differs_from_single_row", False)
    )


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    s = FRAME_SIZE
    n = INNER_SIZE
    intended_idx = rng.choice(sorted(VIABLE_INTENDED_SPECS))

    train1_color = choose_palette(rng, 1, set())[0]
    train2_color = choose_palette(rng, 1, {train1_color})[0]
    test_color = choose_palette(rng, 1, {train1_color, train2_color})[0]

    mode_pairs = list(VIABLE_INTENDED_SPECS[intended_idx])
    rng.shuffle(mode_pairs)

    built = None
    for mode1, mode2 in mode_pairs:
        pair1 = build_ambiguous_pair(n, train1_color, intended_idx, mode1, rng)
        if pair1 is None:
            continue
        pair2 = build_ambiguous_pair(n, train2_color, intended_idx, mode2, rng)
        if pair2 is None:
            continue
        key1 = pair1[2]
        key2 = pair2[2]
        shared = key1 & key2
        if shared != {("sym", intended_idx)}:
            continue
        built = (pair1, pair2)
        break
    if built is None:
        return None

    (train1_src, train1_dst, _), (train2_src, train2_dst, _) = built
    expected_count = int(np.count_nonzero(train1_src))
    test_src = build_test_sprite(n, test_color, rng, expected_count)
    if test_src is None:
        return None

    metrics = combo_metrics_from_parts(train1_src, train1_dst, train2_src, train2_dst, test_src)
    if not (
        metrics["row1_candidate_count"] == 2
        and metrics["row2_candidate_count"] == 2
        and metrics["all_sym"]
        and metrics["shared_key_count"] == 1
        and metrics["key_support_max"] == 2
        and metrics["top_count"] == 2
        and metrics["second_count"] == 1
        and metrics["output_count"] >= 3
        and metrics["changes_test"]
        and metrics["differs_from_single_row"]
    ):
        return None

    voted_test, _ = solve_triplets_to_blank([(train1_src, train1_dst), (train2_src, train2_dst)], test_src)
    if not _valid_nz(voted_test):
        return None

    frame_color, bg_palette = choose_frame_and_bg({train1_color, train2_color, test_color}, rng)
    positions = frame_positions(s, rng)
    frame_boxes = [(r0, c0, r0 + s - 1, c0 + s - 1) for r0, c0 in positions]
    grid = render_background(frame_boxes, bg_palette, rng)

    sprites = [
        train1_src,
        train1_dst,
        test_src,
        train2_src,
        train2_dst,
        np.zeros((n, n), dtype=int),
    ]
    for (r0, c0), sprite in zip(positions, sprites):
        draw_frame(grid, r0, c0, s, frame_color, sprite)

    expected_output = write_into_blank_frame(grid, frame_boxes[5], voted_test)
    if not validate_scene(grid, expected_output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
