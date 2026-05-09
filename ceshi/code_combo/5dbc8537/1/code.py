# Combo variant of 5dbc8537 (index 1)
# combo_spec: D1/0+D4/0
# Name: 边缘颜色引导重映射填充

from __future__ import annotations

from collections import Counter
import math
import random

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
# Design analysis for the D1/0 + D4/0 combo
#
# a) Concrete behavior and trace:
#    The original task was "match hole rectangle by bbox size, then paste the sprite
#    unchanged". This combo inserts two linked changes into that chain:
#    1. every hole has a visible colored side stripe inside the template wall;
#    2. the matching source sprite carries the same cue color on one full edge;
#    3. after the shape+cue match, the sprite's body/main color is repainted to the
#       hole cue color before pasting.
#
#    Small trace:
#    - Old: two 3x2 holes and two 3x2 sprites. Size alone decides placement, and
#      the pasted patch keeps its original colors.
#    - New: the two 3x2 holes might have blue vs red side stripes. The two 3x2
#      sprites therefore expose blue vs red edge bands, so the match key becomes
#      (h, w, cue_color). After matching, the large body color inside that sprite
#      is recolored to the cue color, leaving any small accent pixels untouched.
#
# b) Complexity calibration:
#    The generator deliberately produces repeated size groups, and within at least
#    two duplicated groups the cue colors differ. Without that, the edge cue would
#    be decorative and the combo would collapse back to the original size-only task.
#    Source body colors are drawn from a small repeated palette so recoloring is
#    visible across many pieces rather than being a one-off coincidence.
#
# c) Implicit-assumption audit on the original main():
#    - `hole_mask = dst != base_col` is no longer valid because cue stripes are
#      non-base pixels that are not holes.
#    - The original DFS over every all-hole rectangle assumed size alone identified
#      the destination. That breaks once two equal-size holes exist with different
#      edge cues.
#    - `out = full(base)` would erase the template-side cue stripes, so the combo
#      must start from the cropped template itself.
#    - Raw pasting is also invalid now because the transform stage repaints the
#      sprite body after the shape+cue match.
#
# d) Placement legality:
#    Hole rectangles are still generated as a connected corridor of stacked
#    rectangles. Each rectangle keeps one template-wall cell on both sides in the
#    cross-strip direction, which guarantees a legal left/right or top/bottom cue
#    stripe location. Source sprites are scattered with one-cell padding outside
#    the template, so connected-component extraction remains reliable.
#
# e) Visual observability:
#    The same cue color appears twice in the input: once on the hole-side stripe and
#    once on the sprite edge band. In the output, that cue color expands into the
#    sprite body. Across 2-4 examples, humans can see both "same cue -> same hole"
#    and "body color turns into cue color" without relying on hidden ids.
#
# f) Visual cleanliness:
#    Cue stripes sit in the template wall, not inside the bg cavity itself, so hole
#    geometry stays readable. On the sprite side, the cue is a single full edge band
#    rather than a noisy perimeter marker. Recoloring only the dominant body color
#    keeps small accent structure visible and avoids turning the output into a flat
#    monochrome slab everywhere.
#
# g) Change inferability:
#    The recolor rule is structurally signaled: the cue color already exists at the
#    destination boundary and on the source edge. The source body palette is small
#    and repeated, while cue colors vary across holes, so the mapping is inferable
#    as "the matched hole paints the sprite body with its own edge color" rather
#    than as an arbitrary global color substitution table.
#
# h) Quantified non-degeneracy target:
#    Every accepted sample must satisfy both:
#    - at least two duplicated size groups use multiple cue colors; and
#    - at least max(4, n_sprites // 2) sprite body cells change under the recolor.
#    These two checks separately verify that D1-style cue disambiguation and D4-style
#    recoloring are both active.
#
# i) New solver assumptions and generator guarantees:
#    - every hole rectangle is pure bg and has exactly one adjacent full cue stripe;
#    - every row (vertical strip) or column (horizontal strip) contains at most one
#      bg run, so a hole can be recovered from any cue-adjacent bg cell;
#    - source sprites are filled rectangles;
#    - in a vertical strip the source cue is the full left column; in a horizontal
#      strip it is the full top row;
#    - the body region contains a unique dominant non-cue color, which the solver
#      can safely treat as the recolor target.
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
    """Recover rectangular holes and their cue colors from adjacent template stripes."""
    H, W = dst.shape
    vertical_strip = H > W
    hints = np.argwhere((dst != int(bg)) & (dst != int(base)))
    holes_by_bbox: dict[tuple[int, int, int, int], int] = {}

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
        if vertical_strip:
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
        r0, c0, r1, c1 = bbox
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


def _sprite_cue_and_main(patch: np.ndarray, vertical_strip: bool) -> tuple[int, int]:
    """Read the source cue edge and the dominant recolor target color from a sprite."""
    h, w = map(int, patch.shape)
    if vertical_strip:
        assert w >= 2
        cue_band = patch[:, 0]
        body = patch[:, 1:]
    else:
        assert h >= 2
        cue_band = patch[0, :]
        body = patch[1:, :]

    cue_vals = set(map(int, cue_band.flatten()))
    assert len(cue_vals) == 1
    cue = int(next(iter(cue_vals)))

    body_vals = list(map(int, body.flatten()))
    assert body_vals
    counts = Counter(body_vals)
    main, _ = counts.most_common(1)[0]
    assert int(main) != int(cue)
    return int(cue), int(main)


def _recolor_main_to_cue(patch: np.ndarray, main_color: int, cue_color: int) -> np.ndarray:
    """Repaint the sprite body color to the matched hole cue color."""
    out = np.array(patch, copy=True)
    out[out == int(main_color)] = int(cue_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill cue-marked rectangular cavities by matching sprite edge colors, then repaint the sprite body from that cue.

    Input structure:
    - The most common color is background. One dominant non-background color forms a large strip template.
    - Inside that strip, rectangular bg holes are accompanied by colored side stripes on the template wall.
    - Outside the strip, rectangular source sprites are scattered separately; each sprite carries the same cue color on one full edge and a different body color inside.

    Rule:
    - Detect the template from the largest non-background connected component and crop to its bbox.
    - Recover each rectangular hole and its cue color from the adjacent stripe, then extract the outside sprites.
    - Match every sprite to the unique hole with the same rectangle size and cue color.
    - Before pasting, repaint the sprite's dominant body color to that matched cue color, while leaving any small accent pixels unchanged.

    Key insight: Solving now requires a fused chain: use shape plus edge color to decide where a piece goes, then use that same edge color again as the repaint target for the sprite body.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

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

    base_coords = np.argwhere(g == int(base_col))
    assert not (base_coords.size == 0)
    r0, c0, r1, c1 = bbox_coords(base_coords)
    dst_bbox = (int(r0), int(c0), int(r1), int(c1))
    dst = crop_bbox(g, dst_bbox)
    vertical_strip = int(dst.shape[0]) > int(dst.shape[1])

    holes = _extract_holes(dst, int(bg), int(base_col))
    hole_by_key: dict[tuple[int, int, int], tuple[int, int, int, int]] = {}
    for hole in holes:
        key = (int(hole["h"]), int(hole["w"]), int(hole["cue"]))
        assert key not in hole_by_key
        hole_by_key[key] = tuple(map(int, hole["bbox"]))

    masked = g.copy()
    masked[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)
    sprite_comps = find_components(masked, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites: list[dict] = []
    for comp in sprite_comps:
        b = tuple(map(int, comp["bbox"]))
        pr0, pc0, _pr1, _pc1 = b
        patch = crop_bbox(g, b)
        ph, pw = map(int, patch.shape)
        m = np.zeros((ph, pw), dtype=bool)
        coords = comp["coords"]
        m[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        assert bool(m.all())
        cue, main_color = _sprite_cue_and_main(patch, vertical_strip)
        sprites.append(
            {
                "h": int(ph),
                "w": int(pw),
                "cue": int(cue),
                "main": int(main_color),
                "patch": patch,
            }
        )

    assert len(sprites) == len(holes)
    sprite_keys = [(int(s["h"]), int(s["w"]), int(s["cue"])) for s in sprites]
    assert len(sprite_keys) == len(set(sprite_keys))
    assert set(sprite_keys) == set(hole_by_key.keys())

    out = np.array(dst, copy=True)
    for sp in sprites:
        rr0, cc0, _rr1, _cc1 = hole_by_key[(int(sp["h"]), int(sp["w"]), int(sp["cue"]))]
        recolored = _recolor_main_to_cue(sp["patch"], int(sp["main"]), int(sp["cue"]))
        paste(out, recolored, int(rr0), int(cc0), overwrite=True)

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
SOURCE_PALETTE_SIZE = 3
ACCENT_FRAC = 0.35
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
    """Generate stacked rectangles that yield one bg run per row/column and stay connected."""
    min_len = max(count + 2, round(CORRIDOR_COVERAGE * long))
    max_len = long - 1
    full_area = long * thick
    lo = math.ceil(HOLE_FRAC_LO * full_area)
    hi = math.floor(HOLE_FRAC_HI * full_area)
    max_w = min(MAX_RECT_WIDTH, thick - 2)
    if max_w < 2:
        return None

    max_groups = min(5, count - MIN_DUP_GROUPS + 1)
    if max_groups < MIN_SHAPE_GROUPS:
        return None

    for _ in range(120):
        n_groups = random.randint(MIN_SHAPE_GROUPS, max_groups)
        counts = [2] * MIN_DUP_GROUPS + [1] * (n_groups - MIN_DUP_GROUPS)
        remaining = count - sum(counts)
        if remaining < 0:
            return None
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
                w = random.randint(2, max_w)
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


def _assign_cues(rects: list[tuple[int, int, int, int]], bg: int, base: int) -> tuple[list[int], list[str]] | None:
    """Assign cue colors unique within each size group and visually separated across neighbors."""
    available = [c for c in range(1, 10) if c not in (bg, base)]
    groups = _size_groups(rects)
    if not groups or max(len(v) for v in groups.values()) > len(available):
        return None

    cues = [-1] * len(rects)
    sides = [""] * len(rects)
    used_in_group: dict[tuple[int, int], set[int]] = {k: set() for k in groups}

    def dfs(i: int) -> bool:
        if i == len(rects):
            return True
        h, w = rects[i][2], rects[i][3]
        key = (int(h), int(w))
        prev_cue = cues[i - 1] if i > 0 else None
        colors = [c for c in available if c not in used_in_group[key] and c != prev_cue]
        random.shuffle(colors)
        fallback = [] if prev_cue is None else [c for c in available if c not in used_in_group[key] and c == prev_cue]
        for cue in colors + fallback:
            cues[i] = int(cue)
            used_in_group[key].add(int(cue))
            side_opts = ["left", "right"]
            random.shuffle(side_opts)
            for side in side_opts:
                if i > 0 and sides[i - 1] == side and cues[i - 1] == int(cue):
                    continue
                sides[i] = side
                if dfs(i + 1):
                    return True
            sides[i] = ""
            used_in_group[key].remove(int(cue))
            cues[i] = -1
        return False

    if not dfs(0):
        return None
    return [int(x) for x in cues], sides


def _make_source_piece(
    h: int,
    w: int,
    cue: int,
    main_color: int,
    accent_color: int | None,
    vertical_strip: bool,
) -> np.ndarray:
    """Build a rectangular source sprite with one cue edge and a dominant recolorable body."""
    out = np.full((h, w), int(main_color), dtype=int)
    if vertical_strip:
        out[:, 0] = int(cue)
        body_r0, body_c0 = 0, 1
    else:
        out[0, :] = int(cue)
        body_r0, body_c0 = 1, 0

    body = out[body_r0:, body_c0:]
    body_area = int(body.size)
    if accent_color is None or body_area < 4:
        return out

    coords = [(r, c) for r in range(body.shape[0]) for c in range(body.shape[1])]
    max_accent = max(1, body_area // 3)
    accent_count = random.randint(1, max_accent)
    random.shuffle(coords)
    for r, c in coords[:accent_count]:
        body[r, c] = int(accent_color)
    return out


def _make_source_pieces(
    rects: list[tuple[int, int, int, int]],
    cues: list[int],
    bg: int,
    base: int,
    vertical_strip: bool,
) -> dict[int, np.ndarray] | None:
    """Create scattered source sprites whose body color can later be repainted from the hole cue."""
    groups = _size_groups(rects)
    if len(groups) < MIN_SHAPE_GROUPS:
        return None

    available = [c for c in range(1, 10) if c not in (bg, base)]
    if len(available) < SOURCE_PALETTE_SIZE + 1:
        return None
    source_palette = random.sample(available, SOURCE_PALETTE_SIZE)

    pieces: dict[int, np.ndarray] = {}
    for i, (_r, _c, h, w) in enumerate(rects):
        cue = int(cues[i])
        main_choices = [c for c in source_palette if c != cue]
        if not main_choices:
            return None
        main_color = int(random.choice(main_choices))
        accent_color = None
        accent_choices = [c for c in available if c not in (cue, main_color)]
        body_area = int((h * (w - 1)) if vertical_strip else ((h - 1) * w))
        if accent_choices and body_area >= 4 and random.random() < ACCENT_FRAC:
            accent_color = int(random.choice(accent_choices))
        pieces[i] = _make_source_piece(int(h), int(w), cue, main_color, accent_color, vertical_strip)
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
    vertical_strip = dh > dw
    for (r, c, h, w), cue, side in zip(rects, cues, sides):
        tpl[int(r) : int(r) + int(h), int(c) : int(c) + int(w)] = int(bg)
        if vertical_strip and side == "left":
            tpl[int(r) : int(r) + int(h), int(c) - 1] = int(cue)
        elif vertical_strip:
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
    """Expected output after shape+cue matching and body recoloring."""
    out = np.array(template, copy=True)
    vertical_strip = int(template.shape[0]) > int(template.shape[1])
    for i, (r, c, _h, _w) in enumerate(rects):
        cue, main_color = _sprite_cue_and_main(pieces[i], vertical_strip)
        recolored = _recolor_main_to_cue(pieces[i], main_color, cue)
        paste(out, recolored, int(r), int(c))
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


def _combo_metrics(rects: list[tuple[int, int, int, int]], cues: list[int], pieces: dict[int, np.ndarray], vertical_strip: bool) -> tuple[int, int]:
    """Return (disambiguated_duplicate_groups, recolored_body_cells)."""
    size_to_colors: dict[tuple[int, int], set[int]] = {}
    for (_r, _c, h, w), cue in zip(rects, cues):
        size_to_colors.setdefault((int(h), int(w)), set()).add(int(cue))
    disambiguated = sum(
        1
        for key, ids in _size_groups(rects).items()
        if len(ids) >= 2 and len(size_to_colors[key]) >= 2
    )

    changed_cells = 0
    for i in range(len(rects)):
        cue, main_color = _sprite_cue_and_main(pieces[i], vertical_strip)
        changed_cells += int((pieces[i] == int(main_color)).sum())
        assert cue == int(cues[i])
    return int(disambiguated), int(changed_cells)


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: sample layout, build template and sprites, oracle-verify."""
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

    groups = _size_groups(rects_canon)
    if len(groups) < MIN_SHAPE_GROUPS:
        return None
    dup_groups = [ids for ids in groups.values() if len(ids) >= 2]
    if len(dup_groups) < MIN_DUP_GROUPS:
        return None

    rects = rects_canon if vertical else [(c, r, w, h) for r, c, h, w in rects_canon]

    bg, base = random.sample(range(1, 10), 2)
    assigned = _assign_cues(rects, int(bg), int(base))
    if assigned is None:
        return None
    cues, sides = assigned

    pieces = _make_source_pieces(rects, cues, int(bg), int(base), vertical)
    if pieces is None:
        return None

    disambiguated, changed_cells = _combo_metrics(rects, cues, pieces, vertical)
    if disambiguated < MIN_DUP_GROUPS:
        return None
    if changed_cells < max(4, len(rects) // 2):
        return None

    template = _render_template(dh, dw, int(bg), int(base), rects, cues, sides)
    out = _render_output(template, rects, pieces)
    try:
        inp = _scatter_sprites(H, W, int(bg), int(base), tr0, tc0, template, rects, pieces)
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
    raise RuntimeError("failed to generate valid 5dbc8537 combo input")
