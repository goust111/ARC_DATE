# Combo variant of 5dbc8537 (index 18)
# combo_spec: D1/3+D13/2
# Name: 间距编码顺序+面积排序匹配

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, crop_bbox, find_components, get_bg, paste


# Design analysis for the fused variant
# a) Concrete behavior:
#    Original 5dbc8537 lets the solver fit each outside rectangle into any inside hole
#    whose bbox matches exactly. Here the inside corridor is still decomposable into a
#    left-to-right or top-to-bottom sequence of rectangles, but the intended mapping is:
#    order holes by their position along the strip; order sprites by area ascending; if
#    multiple sprites have the same area, break that tie by the visible gap immediately
#    before the sprite in the outer lane. Example: hole areas [4, 6, 6, 9] and outside
#    sprites shuffled as [6b, 4, 9, 6a] with lead-gaps [2, -, 1, 1] become [4, 6a, 6b, 9].
# b) Complexity calibration:
#    If every area were unique, the spacing signal would be inert. The generator therefore
#    forces at least one equal-area group of size 2 or 3 and assigns distinct lead-gaps
#    1/2(/3) inside that group. If every hole had arbitrary geometry, the rule would be
#    visually noisy, so holes remain rectangles embedded in the original corridor strip.
# c) Broken assumptions from the original solver:
#    - "candidate hole = any rectangle with matching bbox" is no longer sufficient.
#    - DFS exact cover is unnecessary because the strip generator makes the hole sequence
#      recoverable directly from per-row/per-column intervals.
#    - duplicate bbox sizes are now allowed and intentionally ambiguous unless spacing is used.
# d) Placement legality:
#    Template holes are placed as consecutive strip segments with overlapping cross-axis
#    intervals, so they stay in-bounds and visually coherent. Outer sprites are placed on a
#    single lane in the free band outside the template with explicit integer gaps >= 1.
# e) Visual observability:
#    A human sees two stable cues across examples: hole order is spatial along the strip,
#    and equal-area outside sprites are disambiguated by visibly different blank gaps.
#    No invisible feature such as color id magnitude is used for ranking.
# f) Visual cleanliness:
#    The outer sprites live in one uncluttered band; the template remains a single strip.
#    There are no overlaid markers or mixed annotations competing in the same region.
# g) New solver assumptions that the generator must guarantee:
#    - Each strip row/column has at most one contiguous hole interval.
#    - Outside sprites form a one-dimensional lane aligned to the strip's long axis.
#    - The first lane sprite is a singleton-area case so the "no previous gap" sentinel
#      never needs to decide an equal-area tie.


GRID_MIN, GRID_MAX = 15, 25
THICK_MIN, THICK_MAX = 7, 10
SPRITE_COUNT_MIN, SPRITE_COUNT_MAX = 6, 9
MAX_LONG_LEN = 4
HOLE_FRAC_LO, HOLE_FRAC_HI = 0.35, 0.52
MAX_RETRIES = 500


def _pick_base_color(g: np.ndarray, bg: int) -> int:
    base_candidates: list[tuple[int, int]] = []
    for col in colors_present(g, exclude={bg}):
        max_area = 0
        for coords in components_mask(g == int(col), connectivity=8):
            max_area = max(int(max_area), int(coords.shape[0]))
        base_candidates.append((int(max_area), int(col)))
    assert base_candidates
    base_candidates.sort(reverse=True)
    if len(base_candidates) > 1:
        assert base_candidates[0][0] > base_candidates[1][0]
    return int(base_candidates[0][1])


def _hole_interval(line: np.ndarray, base_col: int) -> tuple[int, int] | None:
    cols = np.flatnonzero(line != int(base_col))
    if cols.size == 0:
        return None
    c0 = int(cols[0])
    c1 = int(cols[-1])
    assert c1 - c0 + 1 == int(cols.size)
    return c0, c1


def _extract_ordered_holes(dst: np.ndarray, base_col: int) -> tuple[bool, list[dict[str, int]]]:
    dh, dw = map(int, dst.shape)
    vertical = dh > dw
    holes: list[dict[str, int]] = []

    if vertical:
        tokens = [_hole_interval(dst[r, :], base_col) for r in range(dh)]
    else:
        tokens = [_hole_interval(dst[:, c], base_col) for c in range(dw)]

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok is None:
            i += 1
            continue
        j = i
        while j + 1 < len(tokens) and tokens[j + 1] == tok:
            j += 1
        a0, a1 = tok
        if vertical:
            holes.append({"r": int(i), "c": int(a0), "h": int(j - i + 1), "w": int(a1 - a0 + 1)})
        else:
            holes.append({"r": int(a0), "c": int(i), "h": int(a1 - a0 + 1), "w": int(j - i + 1)})
        i = j + 1

    assert holes
    hole_area = sum(int(h["h"]) * int(h["w"]) for h in holes)
    assert hole_area == int((dst != int(base_col)).sum())
    return vertical, holes


def _extract_sprites(g: np.ndarray, bg: int, dst_bbox: tuple[int, int, int, int], vertical: bool) -> list[dict[str, object]]:
    r0, c0, r1, c1 = dst_bbox
    masked = g.copy()
    masked[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)
    comps = find_components(masked, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites: list[dict[str, object]] = []
    for comp in comps:
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        ph, pw = map(int, patch.shape)
        assert bool((patch != int(bg)).all())
        sprites.append(
            {
                "r0": int(pr0),
                "c0": int(pc0),
                "r1": int(pr1),
                "c1": int(pc1),
                "h": int(ph),
                "w": int(pw),
                "area": int(ph * pw),
                "patch": patch,
            }
        )

    assert sprites
    lane = sorted(sprites, key=lambda s: (int(s["r0"]), int(s["c0"])) if vertical else (int(s["c0"]), int(s["r0"])))
    big_gap = 10**6
    for idx, sp in enumerate(lane):
        sp["lane_index"] = int(idx)
        if idx == 0:
            sp["lead_gap"] = int(big_gap)
            continue
        prev = lane[idx - 1]
        if vertical:
            gap = int(sp["r0"]) - int(prev["r1"]) - 1
        else:
            gap = int(sp["c0"]) - int(prev["c1"]) - 1
        assert gap >= 1
        sp["lead_gap"] = int(gap)
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a strip template by reading hole order from the strip and sprite order from area-plus-spacing cues.

    Input structure:
    - The background is the most frequent color. One large non-background color forms a full-height or full-width strip.
    - Inside that strip, the missing cells appear as a sequence of rectangular slices along the strip's long axis.
    - Outside the strip, the source sprites are scattered on a single lane in the remaining side band, with visible blank gaps between consecutive sprites.

    Rule:
    - Recover the strip bbox from the largest non-background component and decompose its hole corridor into rectangles in spatial order.
    - Sort outside sprites by rectangle area from small to large.
    - If several sprites have the same area, break that tie by the gap immediately before the sprite on the outer lane: smaller gap means earlier.
    - Paste the resulting sprite order into the ordered strip holes and return the tight strip crop.

    Key insight:
    The task no longer hinges on direct bbox-to-bbox matching. The strip already tells you where the destinations are, while the outside lane tells you how to rank equal-area candidates via spacing.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    base_col = _pick_base_color(g, bg)

    base_coords = np.argwhere(g == int(base_col))
    assert not (base_coords.size == 0)
    dst_bbox = tuple(map(int, bbox_coords(base_coords)))
    dst = crop_bbox(g, dst_bbox)
    vertical, holes = _extract_ordered_holes(dst, base_col)

    sprites = _extract_sprites(g, bg, dst_bbox, vertical)
    # Matching is driven by the fused cues only:
    # 1) global area order, 2) within equal-area groups, smaller lead-gap first.
    sprites_sorted = sorted(sprites, key=lambda s: (int(s["area"]), int(s["lead_gap"]), int(s["lane_index"])))

    assert len(holes) == len(sprites_sorted)
    out = np.full(dst.shape, int(base_col), dtype=int)
    for hole, sp in zip(holes, sprites_sorted):
        assert int(hole["h"]) * int(hole["w"]) == int(sp["area"])
        # Geometry compatibility is guaranteed by the generator; it is not the ranking cue.
        assert (int(hole["h"]), int(hole["w"])) == (int(sp["h"]), int(sp["w"]))
        paste(out, np.array(sp["patch"], dtype=int), int(hole["r"]), int(hole["c"]), overwrite=True)
    return out


def _choose_group_sizes(total: int) -> list[int]:
    for _ in range(200):
        groups = random.randint(4, min(6, total))
        sizes = [1] * groups
        rem = total - groups
        while rem > 0:
            choices = [i for i, s in enumerate(sizes) if s < 3]
            if not choices:
                break
            sizes[random.choice(choices)] += 1
            rem -= 1
        if rem == 0 and any(s > 1 for s in sizes) and any(s == 1 for s in sizes):
            random.shuffle(sizes)
            return sizes
    raise RuntimeError("failed to choose area-group sizes")


def _factor_pairs(max_w: int) -> dict[int, list[tuple[int, int]]]:
    out: dict[int, list[tuple[int, int]]] = {}
    for h in range(1, MAX_LONG_LEN + 1):
        for w in range(2, max_w + 1):
            out.setdefault(int(h * w), []).append((int(h), int(w)))
    return out


def _build_sequence_spec(thick: int, total: int) -> dict[str, object] | None:
    max_w = int(thick - 2)
    if max_w < 3:
        return None

    pair_map = _factor_pairs(max_w)
    area_pool = sorted(a for a, pairs in pair_map.items() if pairs)
    if len(area_pool) < 6:
        return None

    for _ in range(400):
        group_sizes = _choose_group_sizes(total)
        if len(group_sizes) > len(area_pool):
            continue

        group_areas = sorted(random.sample(area_pool, len(group_sizes)))
        groups: list[list[dict[str, int]]] = []
        specs: list[dict[str, int]] = []
        idx = 0
        for gid, (size, area) in enumerate(zip(group_sizes, group_areas)):
            pairs = pair_map[int(area)]
            weighted = [pair for pair in pairs for _ in range(pair[1])]
            h, w = random.choice(weighted)
            members: list[dict[str, int]] = []
            for local_rank in range(size):
                rec = {
                    "idx": int(idx),
                    "group": int(gid),
                    "rank": int(local_rank + 1),
                    "area": int(area),
                    "long_len": int(h),
                    "cross_len": int(w),
                }
                members.append(rec)
                specs.append(rec)
                idx += 1
            groups.append(members)

        singleton_groups = [g for g in groups if len(g) == 1]
        if not singleton_groups:
            continue

        first_group = random.choice(singleton_groups)
        other_groups = [g for g in groups if g is not first_group]
        random.shuffle(other_groups)
        lane_order: list[int] = [int(first_group[0]["idx"])]
        for g in other_groups:
            members = [int(m["idx"]) for m in g]
            if len(members) > 1:
                perm = members[:]
                while perm == members:
                    random.shuffle(perm)
                lane_order.extend(perm)
            else:
                lane_order.extend(members)

        if lane_order == list(range(total)):
            continue

        area_by_idx = {int(rec["idx"]): int(rec["area"]) for rec in specs}
        lane_areas = [int(area_by_idx[idx]) for idx in lane_order]
        if all(lane_areas[i] <= lane_areas[i + 1] for i in range(len(lane_areas) - 1)):
            continue

        lead_gap_by_idx = {int(rec["idx"]): int(rec["rank"]) for rec in specs}
        hole_len = sum(int(rec["long_len"]) for rec in specs)
        gap_sum = sum(int(lead_gap_by_idx[i]) for i in lane_order[1:])
        lane_span = int(hole_len + gap_sum)
        hole_area = sum(int(rec["area"]) for rec in specs)

        long_lo = max(
            GRID_MIN,
            lane_span,
            math.ceil(hole_area / (HOLE_FRAC_HI * thick)),
        )
        long_hi = min(
            GRID_MAX,
            math.floor(hole_area / (HOLE_FRAC_LO * thick)),
        )
        if long_lo > long_hi:
            continue

        long_dim = random.randint(int(long_lo), int(long_hi))
        return {
            "specs": specs,
            "lane_order": lane_order,
            "lead_gap_by_idx": lead_gap_by_idx,
            "long_dim": int(long_dim),
            "lane_span": int(lane_span),
        }
    return None


def _place_canonical_rects(specs: list[dict[str, int]], long_dim: int, thick: int) -> list[dict[str, int]] | None:
    hole_len = sum(int(rec["long_len"]) for rec in specs)
    start = random.randint(0, int(long_dim - hole_len))
    pos = int(start)
    out: list[dict[str, int]] = []
    prev_c = prev_w = None

    for rec in specs:
        w = int(rec["cross_len"])
        hi = int(thick - w - 1)
        if hi < 1:
            return None
        if prev_c is None:
            c = random.randint(1, hi)
        else:
            lo = max(1, int(prev_c - w + 1))
            hi2 = min(hi, int(prev_c + prev_w - 1))
            if lo > hi2:
                return None
            c = random.randint(lo, hi2)
        out.append(
            {
                "idx": int(rec["idx"]),
                "area": int(rec["area"]),
                "long_pos": int(pos),
                "cross_pos": int(c),
                "long_len": int(rec["long_len"]),
                "cross_len": int(rec["cross_len"]),
            }
        )
        pos += int(rec["long_len"])
        prev_c, prev_w = int(c), int(w)
    return out


def _orient_rects(canon: list[dict[str, int]], vertical: bool) -> list[dict[str, int]]:
    rects: list[dict[str, int]] = []
    for rec in canon:
        if vertical:
            r, c = int(rec["long_pos"]), int(rec["cross_pos"])
            h, w = int(rec["long_len"]), int(rec["cross_len"])
        else:
            r, c = int(rec["cross_pos"]), int(rec["long_pos"])
            h, w = int(rec["cross_len"]), int(rec["long_len"])
        rects.append({"idx": int(rec["idx"]), "area": int(rec["area"]), "r": int(r), "c": int(c), "h": int(h), "w": int(w)})
    rects.sort(key=lambda rec: int(rec["idx"]))
    return rects


def _make_patch(h: int, w: int, cols: list[int]) -> np.ndarray:
    out = np.full((h, w), int(cols[0]), dtype=int)
    if len(cols) == 1 or h * w == 1:
        return out

    alt = int(cols[1])
    rr, cc = np.indices((h, w))
    pattern = random.choice(["checker", "dots", "split"])
    if pattern == "checker":
        out[(rr + cc + random.randint(0, 1)) % 2 == 0] = int(alt)
    elif pattern == "dots":
        step = random.choice([2, 3])
        mask = (rr * 2 + cc + random.randint(0, step - 1)) % step == 0
        if not bool(mask.any()) or bool(mask.all()):
            mask = np.zeros((h, w), dtype=bool)
            mask[random.randrange(h), random.randrange(w)] = True
        out[mask] = int(alt)
    elif h > 1 and (w == 1 or random.random() < 0.5):
        out[random.randint(1, h - 1) :, :] = int(alt)
    else:
        out[:, random.randint(1, w - 1) :] = int(alt)
    return out


def _make_pieces(rects: list[dict[str, int]], bg: int, base: int) -> dict[int, np.ndarray]:
    usable = [c for c in range(10) if c not in (bg, base)]
    random.shuffle(usable)
    pieces: dict[int, np.ndarray] = {}
    for i, rec in enumerate(rects):
        primary = usable[i % len(usable)]
        cols = [int(primary)]
        if int(rec["h"]) * int(rec["w"]) > 1 and random.random() < 0.4:
            others = [c for c in usable if c != primary]
            if others:
                cols.append(int(random.choice(others)))
        pieces[int(rec["idx"])] = _make_patch(int(rec["h"]), int(rec["w"]), cols)
    return pieces


def _render_output(dh: int, dw: int, base: int, rects: list[dict[str, int]], pieces: dict[int, np.ndarray]) -> np.ndarray:
    out = np.full((dh, dw), int(base), dtype=int)
    for rec in rects:
        paste(out, pieces[int(rec["idx"])], int(rec["r"]), int(rec["c"]))
    return out


def _render_input(
    H: int,
    W: int,
    bg: int,
    base: int,
    tr0: int,
    tc0: int,
    dh: int,
    dw: int,
    rects: list[dict[str, int]],
    pieces: dict[int, np.ndarray],
    lane_order: list[int],
    lead_gap_by_idx: dict[int, int],
    lane_span: int,
    vertical: bool,
) -> np.ndarray:
    inp = np.full((H, W), int(bg), dtype=int)
    inp[tr0 : tr0 + dh, tc0 : tc0 + dw] = int(base)
    for rec in rects:
        r, c, h, w = map(int, (rec["r"], rec["c"], rec["h"], rec["w"]))
        inp[tr0 + r : tr0 + r + h, tc0 + c : tc0 + c + w] = int(bg)

    if vertical:
        outer_c0 = 0 if tc0 > 0 else tc0 + dw
        outer_w = W - dw
        max_w = max(int(pieces[i].shape[1]) for i in lane_order)
        lane_c = outer_c0 + random.randint(0, int(outer_w - max_w))
        pos = random.randint(0, int(H - lane_span))
        for p, idx in enumerate(lane_order):
            if p > 0:
                pos += int(lead_gap_by_idx[idx])
            patch = pieces[int(idx)]
            paste(inp, patch, int(pos), int(lane_c))
            pos += int(patch.shape[0])
    else:
        outer_r0 = 0 if tr0 > 0 else tr0 + dh
        outer_h = H - dh
        max_h = max(int(pieces[i].shape[0]) for i in lane_order)
        lane_r = outer_r0 + random.randint(0, int(outer_h - max_h))
        pos = random.randint(0, int(W - lane_span))
        for p, idx in enumerate(lane_order):
            if p > 0:
                pos += int(lead_gap_by_idx[idx])
            patch = pieces[int(idx)]
            paste(inp, patch, int(lane_r), int(pos))
            pos += int(patch.shape[1])
    return inp


def _gen_attempt() -> np.ndarray | None:
    thick = random.randint(THICK_MIN, THICK_MAX)
    total = random.randint(SPRITE_COUNT_MIN, SPRITE_COUNT_MAX)
    spec = _build_sequence_spec(thick, total)
    if spec is None:
        return None

    vertical = random.random() < 0.5
    specs = list(spec["specs"])
    lane_order = list(spec["lane_order"])
    lead_gap_by_idx = dict(spec["lead_gap_by_idx"])
    long_dim = int(spec["long_dim"])
    lane_span = int(spec["lane_span"])

    canon = _place_canonical_rects(specs, long_dim, thick)
    if canon is None:
        return None
    rects = _orient_rects(canon, vertical)

    max_cross = max(int(rec["w"]) if vertical else int(rec["h"]) for rec in rects)
    if vertical:
        H = int(long_dim)
        min_W = max(GRID_MIN, thick + max_cross)
        if min_W > GRID_MAX:
            return None
        W = random.randint(int(min_W), GRID_MAX)
        dh, dw = H, int(thick)
        tr0, tc0 = 0, (0 if random.random() < 0.5 else W - dw)
    else:
        W = int(long_dim)
        min_H = max(GRID_MIN, thick + max_cross)
        if min_H > GRID_MAX:
            return None
        H = random.randint(int(min_H), GRID_MAX)
        dh, dw = int(thick), W
        tr0, tc0 = (0 if random.random() < 0.5 else H - dh), 0

    bg, base = random.sample(range(10), 2)
    pieces = _make_pieces(rects, bg, base)
    out = _render_output(dh, dw, base, rects, pieces)

    try:
        inp = _render_input(H, W, bg, base, tr0, tc0, dh, dw, rects, pieces, lane_order, lead_gap_by_idx, lane_span, vertical)
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid combo input")
