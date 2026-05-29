# Variant of 5dbc8537
# Dimension: D1 — 匹配条件：精灵与洞的对应方式 (match_cond)
# Change: s.bbox == d.rect — 精灵外接矩形尺寸与洞的矩形尺寸完全相等（shape_match）... → 颜色边缘匹配

from __future__ import annotations

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
# Design analysis for the D1 variant
#
# a) Concrete behavior:
#    The original task encodes every destination as a bg-colored rectangle inside a
#    base-colored strip, and the solver only needs bbox size. In this variant, each
#    hole rectangle still stays bg-colored, but one exposed side of that hole gets a
#    1-cell-thick colored stripe inside the template. The corresponding outside
#    sprite has the same cue color on its full perimeter. A same-size pair like
#    3x4-blue-edge vs 3x4-red-edge is now visually distinct even though shape alone
#    is insufficient.
#
#    Small trace:
#    - Old: two 3x4 sprites, two 3x4 holes. Solver can place either sprite into
#      either hole because it only checks h,w.
#    - New: both holes are still 3x4, but one has a blue side stripe and the other
#      a red side stripe. Sprites also expose blue/red borders. The intended match
#      key becomes (h, w, cue_color), not just (h, w).
#
# b) Complexity calibration:
#    The generator must not collapse to all-unique sizes, otherwise color cues would
#    be decorative. We therefore require multiple duplicated size groups, and within
#    each duplicated size group the cue colors are distinct.
#
# c) Implicit-assumption audit on the original main():
#    - `hole_mask = dst != base_col` used to work because only holes differed from
#      base. It is now wrong because cue stripes also differ from base.
#    - The DFS placement over every all-hole rectangle assumed size alone determined
#      the slot. That assumption is removed; we now explicitly recover each hole
#      rectangle from its adjacent cue stripe and match by (h, w, cue_color).
#    - Initializing `out` as a solid base array would erase cue stripes. The variant
#      instead starts from the cropped template itself so the edge hints remain.
#
# d) Placement legality:
#    To keep recovery reliable, the hole generator is restricted to a corridor made
#    from stacked rectangles only, without extra dot holes. This guarantees each row
#    inside the template contains at most one bg interval. Every rectangle keeps at
#    least one base cell on both left and right, so a cue stripe can be placed
#    without overlap or out-of-bounds issues.
#
# e) Visual observability:
#    The cue is a visible side stripe on the hole plus a same-colored sprite border.
#    When two holes share size, the edge stripe is the only visible distinguisher,
#    making the D1 change legible from a few examples.
#
# f) Visual cleanliness:
#    Cue stripes live in the template wall next to holes rather than inside the hole
#    area, so the rectangular cavities stay clear and the sprite fill region is not
#    cluttered by extra markers.
#
# g) Color inference:
#    The cue colors are reused across different sizes but are unique within any one
#    duplicated size group. Humans can infer "same border color -> same side-striped
#    hole" rather than memorizing arbitrary hidden ids.
#
# h) Quantified non-degeneracy target used during validation:
#    Each generated sample should contain at least one, and usually several, size
#    groups with multiplicity >= 2 and with multiple cue colors inside that size
#    group. That metric directly measures whether color disambiguation is active.
#
# i) New solver assumptions and generator guarantees:
#    - Every hole rectangle has exactly one cue color and a full-height side stripe.
#    - Every row inside the template has at most one bg run.
#    - No two consecutive rectangles share the same horizontal span, so a hole can
#      be recovered from any stripe-adjacent bg cell by extending across rows whose
#      bg span is unchanged.
#    - Every sprite is a filled rectangle whose perimeter color equals the hole cue.
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
            while r0 - 1 >= 0 and int(dst[r0 - 1, cc]) == int(bg) and _row_bg_span(dst, r0 - 1, cc, int(bg)) == (c0, c1):
                r0 -= 1
            r1 = int(rr)
            while r1 + 1 < H and int(dst[r1 + 1, cc]) == int(bg) and _row_bg_span(dst, r1 + 1, cc, int(bg)) == (c0, c1):
                r1 += 1
        else:
            r0, r1 = _col_bg_span(dst, int(rr), int(cc), int(bg))
            c0 = int(cc)
            while c0 - 1 >= 0 and int(dst[rr, c0 - 1]) == int(bg) and _col_bg_span(dst, rr, c0 - 1, int(bg)) == (r0, r1):
                c0 -= 1
            c1 = int(cc)
            while c1 + 1 < W and int(dst[rr, c1 + 1]) == int(bg) and _col_bg_span(dst, rr, c1 + 1, int(bg)) == (r0, r1):
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


def _sprite_cue_color(patch: np.ndarray) -> int:
    """Sprites are rectangular and expose a uniform cue color on their perimeter."""
    h, w = map(int, patch.shape)
    border = set(map(int, patch[0, :]))
    border.update(map(int, patch[h - 1, :]))
    border.update(map(int, patch[:, 0]))
    border.update(map(int, patch[:, w - 1]))
    assert len(border) == 1
    return int(next(iter(border)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill marked rectangular cavities in a template strip by matching sprites to edge cues.

    Input structure:
    - bg is the most common color. A large base-colored strip forms the template, and each rectangular bg hole has a colored stripe on one exposed side.
    - Filled rectangular sprites sit outside the strip, and each sprite shows a single cue color on its perimeter.

    Rule:
    - The largest non-bg connected component still identifies the base strip and its tight bbox.
    - Inside that bbox, recover each bg rectangle and read the cue color from the adjacent side stripe.
    - Match every outside sprite to the unique hole with the same rectangle size and the same cue color, then paste the sprite into that hole.

    Key insight: Bounding-box shape is no longer enough on its own — duplicated hole sizes are disambiguated by the hole-edge color that also appears on the sprite border.
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
    dst_bbox = (int(r0), int(c0), int(r1), int(c1))
    dst = crop_bbox(g, dst_bbox)

    # --- recover hole rectangles from bg area plus adjacent cue stripes ---
    holes = _extract_holes(dst, int(bg), int(base_col))
    hole_by_key: dict[tuple[int, int, int], tuple[int, int, int, int]] = {}
    for hole in holes:
        key = (int(hole["h"]), int(hole["w"]), int(hole["cue"]))
        assert key not in hole_by_key
        hole_by_key[key] = tuple(map(int, hole["bbox"]))

    # --- source sprites = connected components of non-bg pixels outside dst_bbox ---
    masked = g.copy()
    masked[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)
    sprite_comps = find_components(masked, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites: list[dict] = []
    for comp in sprite_comps:
        b = tuple(map(int, comp["bbox"]))
        pr0, pc0, pr1, pc1 = b
        patch = crop_bbox(g, b)
        ph, pw = map(int, patch.shape)
        m = np.zeros((ph, pw), dtype=bool)
        coords = comp["coords"]
        m[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        assert bool(m.all())
        sprites.append(
            {
                "h": int(ph),
                "w": int(pw),
                "cue": _sprite_cue_color(patch),
                "mask": m,
                "patch": patch,
            }
        )

    assert len(sprites) == len(holes)
    sprite_keys = [(int(s["h"]), int(s["w"]), int(s["cue"])) for s in sprites]
    assert len(sprite_keys) == len(set(sprite_keys))
    assert set(sprite_keys) == set(hole_by_key.keys())

    # --- paste each sprite into the matching cue-colored rectangle ---
    out = dst.copy()
    for sp in sprites:
        key = (int(sp["h"]), int(sp["w"]), int(sp["cue"]))
        rr0, cc0, _rr1, _cc1 = hole_by_key[key]
        paste(out, sp["patch"], int(rr0), int(cc0), mask=sp["mask"], overwrite=True)

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
BICHROMATIC_FRAC = 0.35
WIDTH_TOLERANCE = 2
MAX_RETRIES = 400


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
    """Generate stacked rectangles that yield one bg run per row and stay connected."""
    min_len = max(count + 2, round(CORRIDOR_COVERAGE * long))
    max_len = long - 1
    full_area = long * thick
    lo = math.ceil(HOLE_FRAC_LO * full_area)
    hi = math.floor(HOLE_FRAC_HI * full_area)
    max_w = min(MAX_RECT_WIDTH, thick - 2)
    if max_w < 1:
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
        for gi in range(n_groups):
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


def _assign_cues(rects: list[tuple[int, int, int, int]], bg: int, base: int) -> tuple[list[int], list[str]] | None:
    """Assign cue colors, unique inside each size group, with adjacent stripes visually separated."""
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
        if prev_cue is None:
            fallback = []
        else:
            fallback = [c for c in available if c not in used_in_group[key] and c == prev_cue]
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


def _make_piece(h: int, w: int, cue: int, accent: int | None) -> np.ndarray:
    """Build a sprite with a uniform cue-colored perimeter and optional interior accent."""
    out = np.full((h, w), int(cue), dtype=int)
    if accent is None or h < 3 or w < 3:
        return out
    inner = out[1:-1, 1:-1]
    ih, iw = inner.shape
    rr, cc = np.indices((ih, iw))
    pattern = random.choice(["checker", "dots", "split"])
    if pattern == "checker":
        inner[(rr + cc + random.randint(0, 1)) % 2 == 0] = int(accent)
    elif pattern == "dots":
        step = random.choice([2, 3])
        mask = (rr * 2 + cc + random.randint(0, step - 1)) % step == 0
        if not mask.any():
            mask[random.randrange(ih), random.randrange(iw)] = True
        inner[mask] = int(accent)
    elif ih == 1 and iw == 1:
        inner[:, :] = int(accent)
    elif ih > 1 and (iw == 1 or random.random() < 0.5):
        inner[random.randint(1, ih - 1) :, :] = int(accent)
    else:
        inner[:, random.randint(1, iw - 1) :] = int(accent)
    return out


def _make_pieces(
    rects: list[tuple[int, int, int, int]],
    cues: list[int],
    bg: int,
    base: int,
) -> dict[int, np.ndarray] | None:
    """Create the outside sprites; cue color stays on the perimeter."""
    available = [c for c in range(1, 10) if c not in (bg, base)]
    pieces: dict[int, np.ndarray] = {}
    for i, (_r, _c, h, w) in enumerate(rects):
        cue = int(cues[i])
        use_accent = h >= 3 and w >= 3 and random.random() < BICHROMATIC_FRAC
        accent = None
        if use_accent:
            accent_choices = [c for c in available if c != cue]
            if not accent_choices:
                return None
            accent = random.choice(accent_choices)
        pieces[i] = _make_piece(int(h), int(w), int(cue), accent)
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
    """Single attempt: sample a layout, build sprites and stripes, oracle-verify."""
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

    size_to_colors: dict[tuple[int, int], set[int]] = {}
    for (_r, _c, h, w), cue in zip(rects, cues):
        size_to_colors.setdefault((int(h), int(w)), set()).add(int(cue))
    disambiguated = sum(1 for k, ids in _size_groups(rects).items() if len(ids) >= 2 and len(size_to_colors[k]) >= 2)
    if disambiguated < MIN_DUP_GROUPS:
        return None

    pieces = _make_pieces(rects, cues, int(bg), int(base))
    if pieces is None:
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
    raise RuntimeError("failed to generate valid 5dbc8537 D1 input")
