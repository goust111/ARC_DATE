# Combo variant of 5dbc8537 (index 8)
# combo_spec: D1/0+D9/0
# Name: 单色精灵边缘颜色消歧

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np

from arc_utils import (
    bbox_coords,
    colors_present,
    components_mask,
    crop_bbox,
    find_components,
    get_bg,
    paste,
    random_free_location,
)

# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior:
#    The base object is still a thick strip whose interior contains rectangular
#    bg holes. The D1 change adds a 1-cell colored stripe next to each hole, and
#    the D9 change removes all bichromatic sprite interiors: every outside sprite
#    is now a solid rectangle of one color.
#
#    Small trace:
#    - Old task: two 3x4 holes and two 3x4 sprites were distinguishable because
#      the sprite interiors could differ. Matching by size alone was enough.
#    - This combo: several 3x4 sprites can all be solid blocks. Example:
#      3x4 red, 3x4 red, 3x4 blue. The template may contain three 3x4 holes with
#      edge stripes red, red, blue. Shape-only matching is ambiguous, but the
#      intended key is now the multiset of (h, w, color), where hole-edge color
#      must equal the sprite color.
#
# b) Complexity calibration:
#    If every duplicated size group collapsed to one color, edge hints would be
#    decorative. The generator therefore enforces multiple size groups whose
#    members share size but not all color, so color is genuinely needed. At the
#    same time, at least one exact duplicate (same size and same color) is kept,
#    which makes the monochrome setting visually apparent without introducing a
#    new hidden rule.
#
# c) Implicit-assumption audit on the original main():
#    - `hole_mask = dst != base_col` no longer isolates holes, because cue stripes
#      also differ from the base color.
#    - DFS over every size-matching rectangle assumed bbox size uniquely chose a
#      destination. That is invalid once same-size monochrome sprites repeat.
#    - Initializing the output as solid base would erase the edge stripes. The
#      solver must start from the cropped template itself and only fill bg holes.
#    - The original solver allowed arbitrary sprite interiors; this combo assumes
#      each sprite component is a fully filled monochrome rectangle.
#
# d) Placement legality:
#    The hole layout remains a stacked corridor of rectangles, so every row (or
#    column, in the horizontal orientation) contains at most one bg run. That
#    makes hole recovery from stripe-adjacent bg cells stable. Each rectangle also
#    keeps one template cell on both exposed sides, so a stripe can be placed
#    without overlap or going out of bounds.
#
# e) Visual observability:
#    In a training pair, a human can see solid rectangles outside the strip and
#    equally colored side stripes next to holes inside the strip. When two holes
#    share size but their stripes differ, the intended rule "same solid color goes
#    into same-colored edge-marked hole" is legible from pixels alone.
#
# f) Visual cleanliness:
#    The cue stays outside the hole area, in the template wall, so the hole itself
#    remains a clean rectangle. Sprites are plain solid blocks, which reduces
#    clutter relative to the original mixed mono/bichromatic interiors.
#
# g) Quantified non-degeneracy target:
#    Generation accepts only layouts with:
#    - at least two duplicated size groups containing at least two colors; this
#      measures whether edge-color disambiguation is active;
#    - at least one exact duplicate (same h, w, color) group; this measures
#      whether the "all monochrome, duplicates exist" part is visible.
#
# h) New solver assumptions and generator guarantees:
#    - Every hole rectangle has exactly one full-length cue stripe.
#    - Every outside sprite is a monochrome filled rectangle.
#    - Consecutive corridor rectangles never keep the same full span, so a hole
#      recovered from one stripe-adjacent bg cell extends unambiguously.
#    - For each key (h, w, color), the number of sprites equals the number of
#      holes carrying that cue color.
# ---------------------------------------------------------------------------


def _row_bg_span(dst: np.ndarray, row: int, col: int, bg: int) -> tuple[int, int]:
    """Contiguous bg interval in `row` that contains `(row, col)`."""
    H, W = dst.shape
    assert 0 <= row < H and 0 <= col < W and int(dst[row, col]) == int(bg)
    c0 = int(col)
    c1 = int(col)
    while c0 - 1 >= 0 and int(dst[row, c0 - 1]) == int(bg):
        c0 -= 1
    while c1 + 1 < W and int(dst[row, c1 + 1]) == int(bg):
        c1 += 1
    return int(c0), int(c1)


def _col_bg_span(dst: np.ndarray, row: int, col: int, bg: int) -> tuple[int, int]:
    """Contiguous bg interval in `col` that contains `(row, col)`."""
    H, W = dst.shape
    assert 0 <= row < H and 0 <= col < W and int(dst[row, col]) == int(bg)
    r0 = int(row)
    r1 = int(row)
    while r0 - 1 >= 0 and int(dst[r0 - 1, col]) == int(bg):
        r0 -= 1
    while r1 + 1 < H and int(dst[r1 + 1, col]) == int(bg):
        r1 += 1
    return int(r0), int(r1)


def _extract_holes(dst: np.ndarray, bg: int, base: int) -> list[dict]:
    """Recover hole rectangles and their cue colors from template-side stripes."""
    H, W = dst.shape
    hints = np.argwhere((dst != int(bg)) & (dst != int(base)))
    holes_by_bbox: dict[tuple[int, int, int, int], int] = {}
    vertical = H > W

    for hr, hc in hints:
        nbrs: list[tuple[int, int]] = []
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            rr = int(hr + dr)
            cc = int(hc + dc)
            if 0 <= rr < H and 0 <= cc < W and int(dst[rr, cc]) == int(bg):
                nbrs.append((int(rr), int(cc)))
        if len(nbrs) != 1:
            continue

        rr, cc = nbrs[0]
        if vertical:
            c0, c1 = _row_bg_span(dst, int(rr), int(cc), int(bg))
            r0 = int(rr)
            while (
                r0 - 1 >= 0
                and int(dst[r0 - 1, cc]) == int(bg)
                and _row_bg_span(dst, r0 - 1, cc, int(bg)) == (c0, c1)
            ):
                r0 -= 1
            r1 = int(rr)
            while (
                r1 + 1 < H
                and int(dst[r1 + 1, cc]) == int(bg)
                and _row_bg_span(dst, r1 + 1, cc, int(bg)) == (c0, c1)
            ):
                r1 += 1
        else:
            r0, r1 = _col_bg_span(dst, int(rr), int(cc), int(bg))
            c0 = int(cc)
            while (
                c0 - 1 >= 0
                and int(dst[rr, c0 - 1]) == int(bg)
                and _col_bg_span(dst, rr, c0 - 1, int(bg)) == (r0, r1)
            ):
                c0 -= 1
            c1 = int(cc)
            while (
                c1 + 1 < W
                and int(dst[rr, c1 + 1]) == int(bg)
                and _col_bg_span(dst, rr, c1 + 1, int(bg)) == (r0, r1)
            ):
                c1 += 1

        bbox = (int(r0), int(c0), int(r1), int(c1))
        cue = int(dst[int(hr), int(hc)])
        if bbox in holes_by_bbox:
            assert holes_by_bbox[bbox] == cue
        else:
            holes_by_bbox[bbox] = cue

    holes: list[dict] = []
    covered = 0
    for bbox, cue in sorted(holes_by_bbox.items()):
        r0, c0, r1, c1 = map(int, bbox)
        patch = dst[r0 : r1 + 1, c0 : c1 + 1]
        assert bool((patch == int(bg)).all())
        holes.append(
            {
                "bbox": bbox,
                "h": int(r1 - r0 + 1),
                "w": int(c1 - c0 + 1),
                "cue": int(cue),
            }
        )
        covered += int((r1 - r0 + 1) * (c1 - c0 + 1))

    assert holes
    assert covered == int((dst == int(bg)).sum())
    return holes


def _monochrome_color(patch: np.ndarray) -> int:
    """Return the single color used by a monochrome patch."""
    vals = set(map(int, patch.flatten()))
    assert len(vals) == 1
    return int(next(iter(vals)))


def _layout_metrics(rects: list[tuple[int, int, int, int]], cues: list[int]) -> dict[str, int]:
    """Quantify whether the combo behavior is active in a sampled layout."""
    size_groups = _size_groups(rects)
    colorful_dup_groups = 0
    for key, ids in size_groups.items():
        if len(ids) >= 2 and len({int(cues[i]) for i in ids}) >= 2:
            colorful_dup_groups += 1

    exact_dup_groups = 0
    for count in Counter((int(h), int(w), int(cues[i])) for i, (_r, _c, h, w) in enumerate(rects)).values():
        if count >= 2:
            exact_dup_groups += 1

    return {
        "size_groups": int(len(size_groups)),
        "dup_size_groups": int(sum(1 for ids in size_groups.values() if len(ids) >= 2)),
        "colorful_dup_groups": int(colorful_dup_groups),
        "exact_dup_groups": int(exact_dup_groups),
    }


def _instance_metrics(grid: np.ndarray) -> dict[str, int]:
    """Recover combo metrics directly from a generated input grid."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    base_candidates: list[tuple[int, int]] = []
    for col in colors_present(g, exclude={bg}):
        max_area = 0
        for coords in components_mask(g == int(col), connectivity=8):
            max_area = max(int(max_area), int(coords.shape[0]))
        base_candidates.append((int(max_area), int(col)))
    assert base_candidates
    base_col = max(base_candidates)[1]

    base_coords = np.argwhere(g == int(base_col))
    r0, c0, r1, c1 = bbox_coords(base_coords)
    dst = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))
    holes = _extract_holes(dst, int(bg), int(base_col))

    rects = []
    cues = []
    for hole in holes:
        hr0, hc0, hr1, hc1 = map(int, hole["bbox"])
        rects.append((hr0, hc0, hr1 - hr0 + 1, hc1 - hc0 + 1))
        cues.append(int(hole["cue"]))
    return _layout_metrics(rects, cues)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill cue-marked rectangular holes in a strip template using solid-color sprites.

    Input structure:
    - The most common color is bg. A large base-colored strip forms the template, and each rectangular bg hole has a one-cell colored stripe on one exposed side.
    - Outside the strip sit separate sprites, each a fully filled monochrome rectangle.

    Rule:
    - Identify the base strip as the largest non-bg connected component and crop to its bbox.
    - Inside that crop, recover every hole rectangle and read its cue color from the adjacent stripe.
    - Group outside sprites by their rectangle size and solid color, then fill each hole with a sprite whose size matches and whose color equals the hole-edge cue.

    Key insight: Same-size rectangles are no longer resolved by geometry alone; the visible edge color on the hole must match the sprite's single fill color, and matching is by the full multiset of (height, width, color).
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # --- pick base color by largest CC size (excluding bg) ---
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
    base_col = int(base_candidates[0][1])

    # --- destination bbox = bbox of all base-color pixels ---
    base_coords = np.argwhere(g == int(base_col))
    assert not (base_coords.size == 0)
    r0, c0, r1, c1 = bbox_coords(base_coords)
    dst = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))

    # --- recover hole rectangles from bg area plus adjacent cue stripes ---
    holes = _extract_holes(dst, int(bg), int(base_col))
    holes_by_key: dict[tuple[int, int, int], list[tuple[int, int, int, int]]] = {}
    for hole in holes:
        key = (int(hole["h"]), int(hole["w"]), int(hole["cue"]))
        holes_by_key.setdefault(key, []).append(tuple(map(int, hole["bbox"])))

    # --- source sprites = connected components of non-bg pixels outside dst bbox ---
    masked = g.copy()
    masked[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)
    sprite_comps = find_components(masked, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites_by_key: dict[tuple[int, int, int], list[dict]] = {}
    for comp in sprite_comps:
        b = tuple(map(int, comp["bbox"]))
        pr0, pc0, pr1, pc1 = b
        patch = crop_bbox(g, b)
        ph, pw = map(int, patch.shape)
        m = np.zeros((ph, pw), dtype=bool)
        coords = comp["coords"]
        m[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        assert bool(m.all())
        color = _monochrome_color(patch)
        key = (int(ph), int(pw), int(color))
        sprites_by_key.setdefault(key, []).append({"mask": m, "patch": patch})

    hole_counts = {k: len(v) for k, v in holes_by_key.items()}
    sprite_counts = {k: len(v) for k, v in sprites_by_key.items()}
    assert sprite_counts == hole_counts

    # --- paste each sprite group into the matching cue-colored holes ---
    out = dst.copy()
    for key, bboxes in holes_by_key.items():
        sprites = sprites_by_key[key]
        assert len(sprites) == len(bboxes)
        for bbox, sprite in zip(sorted(bboxes), sprites):
            rr0, cc0, _rr1, _cc1 = bbox
            paste(out, sprite["patch"], int(rr0), int(cc0), mask=sprite["mask"], overwrite=True)

    return out


# --------------- generator ---------------

GRID_MIN, GRID_MAX = 15, 25
THICK_MIN, THICK_MAX = 7, 10
OUTER_GAP = 8
SPRITE_COUNT_MIN, SPRITE_COUNT_MAX = 7, 10
HOLE_FRAC_LO, HOLE_FRAC_HI = 0.37, 0.49
CORRIDOR_COVERAGE = 0.78
MAX_RECT_LEN = 8
MAX_RECT_WIDTH = 6
MIN_SHAPE_GROUPS = 3
MIN_DUP_GROUPS = 2
MIN_COLORFUL_DUP_GROUPS = 2
MIN_EXACT_DUP_GROUPS = 1
WIDTH_TOLERANCE = 2
MAX_RETRIES = 500


def _split_lengths(total: int, count: int) -> list[int] | None:
    """Partition `total` into `count` positive integers, each <= MAX_RECT_LEN."""
    if count <= 0 or count > total or count * MAX_RECT_LEN < total:
        return None
    lengths = [1] * count
    remaining = total - count
    while remaining:
        choices = [i for i, v in enumerate(lengths) if v < MAX_RECT_LEN]
        if not choices:
            return None
        i = random.choice(choices)
        add = random.randint(1, min(remaining, MAX_RECT_LEN - lengths[i]))
        lengths[i] += add
        remaining -= add
    random.shuffle(lengths)
    return lengths


def _choose_widths(lengths: list[int], thick: int, full_area: int) -> list[int] | None:
    """Pick widths so total hole area falls inside the target fraction window."""
    max_w = min(MAX_RECT_WIDTH, thick - 2)
    lo = max(0, math.ceil(HOLE_FRAC_LO * full_area))
    hi = max(lo, math.floor(HOLE_FRAC_HI * full_area))
    paths: dict[int, list[int]] = {0: []}
    for h in lengths:
        opts = list(range(1, max_w + 1))
        if h == 1 and max_w >= 2:
            opts = list(range(2, max_w + 1))
        random.shuffle(opts)
        items = list(paths.items())
        random.shuffle(items)
        nxt: dict[int, list[int]] = {}
        for cur, path in items:
            for w in opts:
                s = cur + h * w
                if s <= hi and s not in nxt:
                    nxt[s] = path + [w]
        paths = nxt
        if not paths:
            return None
    valid = [(s, p) for s, p in paths.items() if lo <= s <= hi]
    if not valid:
        return None
    target = random.randint(lo, hi)
    best = min(abs(s - target) for s, _ in valid)
    near = [p for s, p in valid if abs(s - target) <= best + WIDTH_TOLERANCE]
    return random.choice(near)


def _corridor_rects(long: int, thick: int, count: int) -> list[tuple[int, int, int, int]] | None:
    """Generate stacked rectangles with one bg run per row and repeated size groups."""
    min_len = max(count + 2, round(CORRIDOR_COVERAGE * long))
    max_len = long - 1
    full_area = long * thick
    lo = math.ceil(HOLE_FRAC_LO * full_area)
    hi = math.floor(HOLE_FRAC_HI * full_area)
    max_w = min(MAX_RECT_WIDTH, thick - 2)
    if max_w < 1:
        return None

    max_groups = min(5, count - 3)
    if max_groups < MIN_SHAPE_GROUPS:
        return None

    for _ in range(160):
        n_groups = random.randint(MIN_SHAPE_GROUPS, max_groups)
        counts = [3] + [2] * (MIN_DUP_GROUPS - 1) + [1] * (n_groups - MIN_DUP_GROUPS)
        remaining = count - sum(counts)
        if remaining < 0:
            continue
        for _ in range(remaining):
            choices = [i for i, v in enumerate(counts) if v < 7]
            if not choices:
                break
            counts[random.choice(choices)] += 1
        if sum(counts) != count:
            continue

        shapes: list[tuple[int, int]] = []
        used_shapes: set[tuple[int, int]] = set()
        ok = True
        for _ in range(n_groups):
            picked = None
            for _ in range(40):
                h = random.randint(1, min(5, MAX_RECT_LEN))
                w = random.randint(1, max_w)
                if h == 1 and w == 1:
                    continue
                shape = (int(h), int(w))
                if shape in used_shapes:
                    continue
                picked = shape
                break
            if picked is None:
                ok = False
                break
            used_shapes.add(picked)
            shapes.append(picked)
        if not ok:
            continue

        hole_len = sum(counts[i] * shapes[i][0] for i in range(n_groups))
        area = sum(counts[i] * shapes[i][0] * shapes[i][1] for i in range(n_groups))
        if not (min_len <= hole_len <= max_len and lo <= area <= hi):
            continue

        dims: list[tuple[int, int]] = []
        for cnt, shape in zip(counts, shapes):
            dims.extend([shape] * cnt)
        random.shuffle(dims)

        start = random.randint(0, long - hole_len)
        r = start
        prev_c = None
        prev_w = None
        rects: list[tuple[int, int, int, int]] = []
        for h, w in dims:
            c_hi = thick - w - 1
            if c_hi < 1:
                ok = False
                break
            if prev_c is None:
                c = random.randint(1, c_hi)
            else:
                c_lo = max(1, int(prev_c) - w + 1)
                c_hi2 = min(c_hi, int(prev_c) + int(prev_w) - 1)
                if c_lo > c_hi2:
                    ok = False
                    break
                choices = list(range(c_lo, c_hi2 + 1))
                if int(prev_w) == int(w) and int(prev_c) in choices and len(choices) > 1:
                    choices.remove(int(prev_c))
                if not choices:
                    ok = False
                    break
                c = random.choice(choices)
            rects.append((int(r), int(c), int(h), int(w)))
            r += h
            prev_c, prev_w = c, w
        if ok:
            return rects
    return None


def _size_groups(rects: list[tuple[int, int, int, int]]) -> dict[tuple[int, int], list[int]]:
    groups: dict[tuple[int, int], list[int]] = {}
    for i, (_r, _c, h, w) in enumerate(rects):
        groups.setdefault((int(h), int(w)), []).append(int(i))
    return groups


def _assign_colors(rects: list[tuple[int, int, int, int]], bg: int, base: int) -> list[int] | None:
    """Assign monochrome sprite colors while preserving both exact and color-split duplicates."""
    available = [c for c in range(1, 10) if c not in (bg, base)]
    groups = _size_groups(rects)
    dup_keys = [k for k, ids in groups.items() if len(ids) >= 2]
    triple_keys = [k for k, ids in groups.items() if len(ids) >= 3]
    if len(dup_keys) < MIN_DUP_GROUPS or not triple_keys:
        return None

    exact_key = random.choice(triple_keys)
    other_dup = [k for k in dup_keys if k != exact_key]
    if len(other_dup) < MIN_COLORFUL_DUP_GROUPS - 1:
        return None
    n_colorful = random.randint(MIN_COLORFUL_DUP_GROUPS, len(dup_keys))
    colorful_keys = {exact_key, *random.sample(other_dup, n_colorful - 1)}

    cues = [-1] * len(rects)
    group_order = list(groups)
    random.shuffle(group_order)
    for key in group_order:
        ids = groups[key][:]
        random.shuffle(ids)
        if key == exact_key:
            dup_color, alt_color = random.sample(available, 2)
            cues[ids[0]] = int(dup_color)
            cues[ids[1]] = int(dup_color)
            cues[ids[2]] = int(alt_color)
            for idx in ids[3:]:
                cues[idx] = int(random.choice([dup_color, alt_color]))
        elif key in colorful_keys:
            c0, c1 = random.sample(available, 2)
            cues[ids[0]] = int(c0)
            cues[ids[1]] = int(c1)
            for idx in ids[2:]:
                cues[idx] = int(random.choice([c0, c1]))
        else:
            fill = random.choice(available)
            for idx in ids:
                cues[idx] = int(fill)

    assert all(c >= 0 for c in cues)
    metrics = _layout_metrics(rects, cues)
    if metrics["colorful_dup_groups"] < MIN_COLORFUL_DUP_GROUPS:
        return None
    if metrics["exact_dup_groups"] < MIN_EXACT_DUP_GROUPS:
        return None
    return [int(c) for c in cues]


def _assign_sides(cues: list[int]) -> list[str] | None:
    """Choose stripe sides; avoid merging same-color adjacent stripes when possible."""
    sides = [""] * len(cues)

    def dfs(i: int) -> bool:
        if i == len(cues):
            return True
        opts = ["left", "right"]
        random.shuffle(opts)
        for side in opts:
            if i > 0 and int(cues[i - 1]) == int(cues[i]) and sides[i - 1] == side:
                continue
            sides[i] = side
            if dfs(i + 1):
                return True
        return False

    if not dfs(0):
        return None
    return sides


def _make_pieces(rects: list[tuple[int, int, int, int]], cues: list[int]) -> dict[int, np.ndarray]:
    """Build monochrome rectangle sprites."""
    pieces: dict[int, np.ndarray] = {}
    for i, (_r, _c, h, w) in enumerate(rects):
        pieces[i] = np.full((int(h), int(w)), int(cues[i]), dtype=int)
    return pieces


def _render_template(
    dh: int,
    dw: int,
    bg: int,
    base: int,
    rects: list[tuple[int, int, int, int]],
    cues: list[int],
    sides: list[str],
) -> np.ndarray:
    """Template crop with bg holes and colored side stripes."""
    tpl = np.full((dh, dw), int(base), dtype=int)
    vertical = dh > dw
    for (r, c, h, w), cue, side in zip(rects, cues, sides):
        tpl[int(r) : int(r) + int(h), int(c) : int(c) + int(w)] = int(bg)
        if vertical and side == "left":
            tpl[int(r) : int(r) + int(h), int(c) - 1] = int(cue)
        elif vertical:
            tpl[int(r) : int(r) + int(h), int(c) + int(w)] = int(cue)
        elif side == "left":
            tpl[int(r) - 1, int(c) : int(c) + int(w)] = int(cue)
        else:
            tpl[int(r) + int(h), int(c) : int(c) + int(w)] = int(cue)
    return tpl


def _render_output(
    template: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    pieces: dict[int, np.ndarray],
) -> np.ndarray:
    out = template.copy()
    for i, (r, c, _h, _w) in enumerate(rects):
        paste(out, pieces[i], int(r), int(c))
    return out


def _scatter_sprites(
    H: int,
    W: int,
    bg: int,
    base: int,
    tr0: int,
    tc0: int,
    template: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    pieces: dict[int, np.ndarray],
) -> np.ndarray:
    """Build the input grid: template on one side, matching sprites scattered outside."""
    dh, dw = map(int, template.shape)
    inp = np.full((H, W), int(bg), dtype=int)
    inp[tr0 : tr0 + dh, tc0 : tc0 + dw] = template

    placement = np.full((H, W), int(bg), dtype=int)
    placement[tr0 : tr0 + dh, tc0 : tc0 + dw] = int(base)

    order = list(range(len(rects)))
    random.shuffle(order)
    order.sort(key=lambda i: pieces[i].size, reverse=True)
    for i in order:
        patch = pieces[i]
        r, c = random_free_location(placement, patch, transparent=int(bg), padding=1, padding_connectivity=8)
        paste(inp, patch, int(r), int(c))
        paste(placement, patch, int(r), int(c))
    return inp


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: sample a layout, build monochrome sprites and stripes, oracle-verify."""
    vertical = random.random() < 0.5
    thick = random.randint(THICK_MIN, THICK_MAX)
    if vertical:
        H = random.randint(GRID_MIN, GRID_MAX)
        W = random.randint(max(GRID_MIN, thick + OUTER_GAP), GRID_MAX)
        dh, dw = H, thick
        tr0, tc0 = 0, (0 if random.random() < 0.5 else W - thick)
    else:
        H = random.randint(max(GRID_MIN, thick + OUTER_GAP), GRID_MAX)
        W = random.randint(GRID_MIN, GRID_MAX)
        dh, dw = thick, W
        tr0, tc0 = (0 if random.random() < 0.5 else H - thick), 0
    long = dh if vertical else dw

    total = random.randint(SPRITE_COUNT_MIN, SPRITE_COUNT_MAX)
    rects_canon = _corridor_rects(long, thick, total)
    if rects_canon is None:
        return None
    if len(_size_groups(rects_canon)) < MIN_SHAPE_GROUPS:
        return None

    rects = rects_canon if vertical else [(c, r, w, h) for r, c, h, w in rects_canon]

    bg, base = random.sample(range(1, 10), 2)
    cues = _assign_colors(rects, int(bg), int(base))
    if cues is None:
        return None
    sides = _assign_sides(cues)
    if sides is None:
        return None

    metrics = _layout_metrics(rects, cues)
    if metrics["colorful_dup_groups"] < MIN_COLORFUL_DUP_GROUPS:
        return None
    if metrics["exact_dup_groups"] < MIN_EXACT_DUP_GROUPS:
        return None

    pieces = _make_pieces(rects, cues)
    template = _render_template(dh, dw, int(bg), int(base), rects, cues, sides)
    out = _render_output(template, rects, pieces)
    try:
        inp = _scatter_sprites(H, W, int(bg), int(base), tr0, tc0, template, rects, pieces)
        solved = main(inp)
        recovered = _instance_metrics(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    if recovered["colorful_dup_groups"] < MIN_COLORFUL_DUP_GROUPS:
        return None
    if recovered["exact_dup_groups"] < MIN_EXACT_DUP_GROUPS:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 5dbc8537 combo input")
