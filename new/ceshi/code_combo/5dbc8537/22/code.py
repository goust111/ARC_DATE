# Combo variant of 5dbc8537 (index 22)
# combo_spec: D4/2+D9/1
# Name: 全双色精灵颜色互换填入

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

# Design analysis for the fused variant:
# a) Concrete behavior:
#    Original: a 2x4 sprite such as [[2,7,2,7],[7,2,7,2]] is pasted unchanged into the
#    matching 2x4 hole. Fused: the same sprite still matches by bbox size, but once placed
#    inside the hole its two colors are swapped, yielding [[7,2,7,2],[2,7,2,7]]. The hole
#    layout and packing logic stay unchanged; only the filled pixels differ.
# b) Complexity calibration:
#    Every sprite is forced to be bichromatic, so color swap is always visible. To avoid
#    ambiguity from repeated hole sizes, all sprites sharing the same (h,w) reuse the same
#    exact pre-swap pattern; then any size-only placement permutation still produces the
#    same swapped output.
# c) Implicit assumptions from the original solver:
#    - Largest non-bg connected component still identifies the template base color.
#    - Each source sprite remains a full rectangle, so bbox area equals hole area.
#    - Exact-cover search still relies only on rectangle dimensions; therefore the generator
#      must keep the hole region as a perfect tiling of source sprite bboxes.
#    New fused assumption: every source sprite has exactly two non-bg colors. The generator
#    guarantees this by forbidding monochrome pieces and 1x1 pieces.
# d) Placement legality:
#    Holes are built first as non-overlapping rectangles inside the strip template. Sprites
#    are then scattered only in the outer region with 8-neighbor padding, so no sprite can
#    merge with another sprite or with the template.
# e) Visual observability:
#    Because all sprites are checker / stripe / dot-style bichrome fills, each training pair
#    shows a direct local relation: same shape and same texture, but the two colors trade
#    places after insertion. No hidden ordering or color-id arithmetic is used.
# f) Visual cleanliness:
#    The template remains a single strip and sprites remain isolated rectangles, so there is
#    no overlap between marks. Restricting patterns to a small family keeps the output legible.
# g) Change inferability:
#    The key variable is local two-color exchange. Since every sprite exhibits it, samples do
#    not degenerate into the original task where some monochrome pieces would look unchanged.


def _swap_bichrome_patch(patch: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Swap the two active colors inside a sprite patch."""
    if mask is None:
        mask = np.ones_like(patch, dtype=bool)
    vals = sorted({int(x) for x in patch[mask].flatten()})
    assert len(vals) == 2
    a, b = vals
    out = patch.copy()
    out[mask & (patch == a)] = b
    out[mask & (patch == b)] = a
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a strip-template's rectangular holes using scattered sprites, then swap each sprite's two colors.

    Input structure:
    - The background is the most frequent color.
    - One dominant non-background color forms a long strip-like template containing rectangular holes.
    - Outside that strip sit separated source sprites, and every sprite is a full rectangle with exactly two colors.

    Rule:
    - Detect the template from the largest non-background connected component and crop to its bbox.
    - Match outside sprites to hole rectangles only by bounding-box size, using an exact cover of the hole area.
    - Before writing a matched sprite into the template, exchange its two colors, then return the tight template crop.

    Key insight:
    - Geometry still solves the jigsaw: each sprite fits a hole of the same rectangle size.
    - The new visual rule is independent of placement: every placed bichrome sprite keeps its pattern but reverses its two colors.
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
    dh, dw = map(int, dst.shape)
    hole_mask = dst != int(base_col)

    out = np.full((dh, dw), int(base_col), dtype=int)

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
        vals = {int(x) for x in patch[m].flatten()}
        assert len(vals) == 2
        sprites.append({"h": int(ph), "w": int(pw), "mask": m, "patch": patch})

    sprites.sort(key=lambda s: (-int(s["h"] * s["w"]), -int(s["mask"].sum())))

    candidates: list[list[tuple[int, int]]] = []
    areas = [int(s["h"] * s["w"]) for s in sprites]
    assert sum(areas) == int(hole_mask.sum())
    for sp in sprites:
        h, w = int(sp["h"]), int(sp["w"])
        pos: list[tuple[int, int]] = []
        if h <= dh and w <= dw:
            for rr0 in range(int(dh - h + 1)):
                for cc0 in range(int(dw - w + 1)):
                    if bool(hole_mask[rr0 : rr0 + h, cc0 : cc0 + w].all()):
                        pos.append((int(rr0), int(cc0)))
        candidates.append(pos)

    assert all(candidates)
    used = ~hole_mask
    placement: list[tuple[int, int] | None] = [None] * len(sprites)

    def dfs(i: int) -> bool:
        if i == len(sprites):
            return True
        h, w = int(sprites[i]["h"]), int(sprites[i]["w"])
        for rr0, cc0 in candidates[i]:
            if bool(used[rr0 : rr0 + h, cc0 : cc0 + w].any()):
                continue
            used[rr0 : rr0 + h, cc0 : cc0 + w] = True
            placement[i] = (int(rr0), int(cc0))
            if dfs(i + 1):
                return True
            placement[i] = None
            used[rr0 : rr0 + h, cc0 : cc0 + w] = False
        return False

    assert dfs(0)

    for i, pos in enumerate(placement):
        assert pos is not None
        rr0, cc0 = map(int, pos)
        sp = sprites[i]
        swapped = _swap_bichrome_patch(sp["patch"], sp["mask"])
        paste(out, swapped, rr0, cc0, mask=sp["mask"], overwrite=True)

    return out


# --------------- generator ---------------

GRID_MIN, GRID_MAX = 15, 25
THICK_MIN, THICK_MAX = 7, 10
OUTER_GAP = 9
SPRITE_COUNT_MIN, SPRITE_COUNT_MAX = 7, 13
HOLE_FRAC_LO, HOLE_FRAC_HI = 0.39, 0.50
CORRIDOR_COVERAGE = 0.78
MAX_RECT_LEN = 8
MAX_RECT_WIDTH = 6
MIN_SHAPE_GROUPS = 3
MIN_PALETTE_SIZE = 4
MAX_PALETTE_SIZE = 7
WIDTH_TOLERANCE = 2
MAX_RETRIES = 400
LAYOUT_VARIANTS = 4
_GENERATE_COUNTER = 0


def _split_lengths(total: int, count: int) -> list[int] | None:
    """Partition total corridor length into count positive lengths."""
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
    """Choose widths so hole area lands in the desired fraction of the template."""
    max_w = min(MAX_RECT_WIDTH, thick - 2)
    lo = math.ceil(HOLE_FRAC_LO * full_area)
    hi = math.floor(HOLE_FRAC_HI * full_area)
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
    """Generate connected rectangles whose union forms the hole corridor."""
    min_len = max(count + 2, round(CORRIDOR_COVERAGE * long))
    if min_len > long - 1:
        min_len = long - 1
    hole_len = random.randint(min_len, long - 1)
    lengths = _split_lengths(hole_len, count)
    if lengths is None:
        return None
    widths = _choose_widths(lengths, thick, long * thick)
    if widths is None:
        return None

    start = random.randint(0, long - hole_len)
    r = start
    prev_c = prev_w = None
    rects: list[tuple[int, int, int, int]] = []
    for h, w in zip(lengths, widths):
        c_hi = thick - w - 1
        if c_hi < 1:
            return None
        if prev_c is None:
            c = random.randint(1, c_hi)
        else:
            c_lo = max(1, prev_c - w + 1)
            c_hi = min(c_hi, prev_c + prev_w - 1)
            if c_lo > c_hi:
                return None
            c = random.randint(c_lo, c_hi)
        rects.append((r, c, h, w))
        r += h
        prev_c, prev_w = c, w
    return rects


def _make_bichrome_patch(h: int, w: int, cols: tuple[int, int]) -> np.ndarray:
    """Fill a rectangle with two colors using checker / stripes / dots."""
    a, b = map(int, cols)
    rr, cc = np.indices((h, w))
    patterns = ["checker", "stripes", "dots"]
    random.shuffle(patterns)
    for pattern in patterns:
        if pattern == "checker":
            mask = (rr + cc + random.randint(0, 1)) % 2 == 0
        elif pattern == "stripes":
            if h == 1:
                mask = (cc + random.randint(0, 1)) % 2 == 0
            elif w == 1:
                mask = (rr + random.randint(0, 1)) % 2 == 0
            elif random.random() < 0.5:
                mask = (rr + random.randint(0, 1)) % 2 == 0
            else:
                mask = (cc + random.randint(0, 1)) % 2 == 0
        else:
            step = random.choice([2, 3])
            mask = (rr * 2 + cc + random.randint(0, step - 1)) % step == 0
        if mask.any() and (~mask).any():
            out = np.full((h, w), a, dtype=int)
            out[mask] = b
            return out
    out = np.full((h, w), a, dtype=int)
    out[(rr + cc) % 2 == 0] = b
    assert len({int(x) for x in out.flatten()}) == 2
    return out


def _make_pieces(rects: list[tuple[int, int, int, int]], bg: int, base: int) -> dict[int, np.ndarray] | None:
    """Assign one shared bichrome patch to each shape group."""
    groups: dict[tuple[int, int], list[int]] = {}
    for i, (_, _, h, w) in enumerate(rects):
        groups.setdefault((h, w), []).append(i)
    if len(groups) < MIN_SHAPE_GROUPS:
        return None

    available = [c for c in range(10) if c not in (bg, base)]
    if len(available) < MIN_PALETTE_SIZE:
        return None
    palette_size = random.randint(MIN_PALETTE_SIZE, min(MAX_PALETTE_SIZE, len(available)))
    palette = random.sample(available, palette_size)

    keys = list(groups)
    random.shuffle(keys)
    group_patch: dict[tuple[int, int], np.ndarray] = {}
    for key in keys:
        a, b = random.sample(palette, 2)
        h, w = key
        patch = _make_bichrome_patch(h, w, (int(a), int(b)))
        assert len({int(x) for x in patch.flatten()}) == 2
        group_patch[key] = patch

    pieces: dict[int, np.ndarray] = {}
    for i, (_, _, h, w) in enumerate(rects):
        patch = group_patch[(h, w)].copy()
        assert len({int(x) for x in patch.flatten()}) == 2
        pieces[i] = patch
    return pieces


def _render_output(
    dh: int,
    dw: int,
    base: int,
    rects: list[tuple[int, int, int, int]],
    output_pieces: dict[int, np.ndarray],
) -> np.ndarray:
    out = np.full((dh, dw), base, dtype=int)
    for i, (r, c, _, _) in enumerate(rects):
        paste(out, output_pieces[i], r, c)
    return out


def _scatter_sprites(
    H: int,
    W: int,
    bg: int,
    base: int,
    tr0: int,
    tc0: int,
    dh: int,
    dw: int,
    rects: list[tuple[int, int, int, int]],
    input_pieces: dict[int, np.ndarray],
) -> np.ndarray:
    inp = np.full((H, W), bg, dtype=int)
    inp[tr0 : tr0 + dh, tc0 : tc0 + dw] = base
    for r, c, h, w in rects:
        inp[tr0 + r : tr0 + r + h, tc0 + c : tc0 + c + w] = bg

    placement = np.full((H, W), bg, dtype=int)
    placement[tr0 : tr0 + dh, tc0 : tc0 + dw] = base
    order = list(range(len(rects)))
    random.shuffle(order)
    order.sort(key=lambda i: input_pieces[i].size, reverse=True)
    for i in order:
        patch = input_pieces[i]
        r, c = random_free_location(placement, patch, transparent=bg, padding=1, padding_connectivity=8)
        paste(inp, patch, r, c)
        paste(placement, patch, r, c)
    return inp


def _gen_attempt(layout_variant: int) -> np.ndarray | None:
    vertical = layout_variant in (0, 1)
    place_on_low_side = layout_variant in (0, 2)
    thick = random.randint(THICK_MIN, THICK_MAX)
    if vertical:
        H = random.randint(GRID_MIN, GRID_MAX)
        W = random.randint(max(GRID_MIN, thick + OUTER_GAP), GRID_MAX)
        dh, dw = H, thick
        tr0, tc0 = 0, (0 if place_on_low_side else W - thick)
    else:
        H = random.randint(max(GRID_MIN, thick + OUTER_GAP), GRID_MAX)
        W = random.randint(GRID_MIN, GRID_MAX)
        dh, dw = thick, W
        tr0, tc0 = (0 if place_on_low_side else H - thick), 0
    long = dh if vertical else dw

    total = random.randint(SPRITE_COUNT_MIN, SPRITE_COUNT_MAX)
    rects_canon = _corridor_rects(long, thick, total)
    if rects_canon is None:
        return None
    rects = rects_canon if vertical else [(c, r, w, h) for r, c, h, w in rects_canon]

    bg, base = random.sample(range(10), 2)
    pieces_in = _make_pieces(rects, bg, base)
    if pieces_in is None:
        return None
    pieces_out = {i: _swap_bichrome_patch(patch) for i, patch in pieces_in.items()}

    out = _render_output(dh, dw, base, rects, pieces_out)
    try:
        inp = _scatter_sprites(H, W, bg, base, tr0, tc0, dh, dw, rects, pieces_in)
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GENERATE_COUNTER
    layout_variant = _GENERATE_COUNTER % LAYOUT_VARIANTS
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(layout_variant)
        if grid is not None:
            _GENERATE_COUNTER += 1
            return grid
    raise RuntimeError("failed to generate valid fused 5dbc8537 input")
