# Variant of 5dbc8537
# Dimension: D4 — 填充变换方式 (transform)
# Change: 将精灵原样粘贴到匹配的洞中（精灵的背景色像素被基底色替换），输出为紧裁后的模板外接矩形... → 颜色重映射

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
# Design analysis for the D4 recolor variant
# ---------------------------------------------------------------------------
# spec:
#   variant_core:
#     original_behavior: sprites are packed into same-sized rectangular holes and keep their colors.
#     new_behavior: after packing, each sprite is recolored from the hole-local color hints.
#   concrete_pixel_rule:
#     template_bbox: unchanged from the original task; a base-colored strip spans one full grid dimension.
#     holes: still axis-aligned bg rectangles inside that strip, tiled exactly by the outside sprites.
#     hints:
#       - every hole has one colored hint pixel on the template border, aligned with the hole midpoint
#         across the strip thickness: left border in a vertical strip, top border in a horizontal strip;
#       - the whole sprite is repainted to that hint color after being placed into the matching hole.
#     output: the cropped template is returned with the same side hints still visible, but each filled hole
#       now contains the recolored version of its matched sprite rather than the original sprite colors.
#   behavior_trace:
#     original:
#       - detect base strip, treat every non-base cell in bbox as hole, exact-cover by sprite bbox, paste patch.
#     variant:
#       - detect base strip, but only bg cells are holes because non-base hint pixels now live inside the bbox;
#       - exact-cover is still driven only by rectangle sizes;
#       - once a sprite is assigned to a hole, inspect the aligned border hint pixel and repaint every
#         non-background cell of the sprite to that color before pasting.
#     difference:
#       - the packing stage is unchanged;
#       - the transform stage is new and visible in the output;
#       - the solver can no longer initialize the destination as plain base color or it would erase the hints.
#   complexity_calibration:
#     - hints are local and repeated per hole, so the recolor rule is inferable from 2-4 examples.
#     - the target palette is small (3 colors per sample) and reused across many holes, which avoids a noisy
#       one-off mapping while still making recoloring obvious.
#     - same-shape duplicate sprites reuse one prototype patch. This is a necessary adaptation: because output
#       colors now depend on the hole, allowing different same-sized source patches would create hidden
#       assignment ambiguity under the original size-only exact-cover search.
#   implicit_assumption_audit:
#     - original `hole_mask = dst != base` is invalid now because hint pixels are non-base but not holes.
#       The variant must use `hole_mask = dst == bg`.
#     - original `out = full(base)` would delete the hint pixels. The variant must start from the cropped
#       template itself and overwrite only hole cells via sprite placement.
#     - original raw paste preserved source colors. The variant must repaint the placed sprite from a
#       visible per-hole border hint.
#     - exact-cover still assumes rectangular holes and non-overlap only; the generator preserves that.
#   placement_legality:
#     - rectangles still live strictly inside the strip with one base cell on both side walls, guaranteeing
#       valid hint locations at `c-1` and `c+w`.
#     - sprites are still scattered outside the template with one-cell padding, so component extraction works.
#     - generator oracle-checks `main(input) == expected_output` and rejects any sample where base-color
#       detection or recoloring assumptions fail.
#   visual_observability:
#     - the border hint pixels are aligned with the corresponding holes and repeated many times, so humans
#       can read them as paint chips tied to those cavities rather than unrelated decorations.
#     - source and target palettes are disjoint inside each sample, ensuring recoloring is visible instead of
#       silently matching the original colors.
#     - bichromatic source patches are generated with unequal color counts, so "main color vs accent color"
#       is visible without relying on hidden color ids or arbitrary ordering.


def _recolor_patch(patch: np.ndarray, mask: np.ndarray, bg: int, hint_color: int) -> np.ndarray:
    """Repaint every sprite pixel to the hole's hint color."""
    out = np.array(patch, copy=True)
    out[mask & (patch != int(bg))] = int(hint_color)
    return out


def _hole_hint_color(dst: np.ndarray, rr0: int, cc0: int, h: int, w: int, bg: int, base_col: int) -> int:
    """Read the border hint attached to a hole."""
    dh, dw = map(int, dst.shape)
    vertical_strip = dh > dw
    if vertical_strip:
        mid_r = int(rr0 + h // 2)
        hint = int(dst[mid_r, 0])
    else:
        mid_c = int(cc0 + w // 2)
        hint = int(dst[0, mid_c])
    assert hint not in (int(bg), int(base_col))
    return hint


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the template cavities by size, then recolor each fitted sprite from hole-side hint pixels.

    Input structure:
    - The most common color is background; one other dominant color forms the large strip template.
    - Rectangular bg holes sit inside that strip, and each hole has a colored hint pixel on the strip border
      aligned with that hole.
    - Source sprites remain scattered outside the template and still match the holes by bounding-box size.

    Rule:
    - Detect the base strip from the largest non-background connected component and crop to its bbox.
    - Treat only bg cells inside that bbox as holes; the extra non-base pixels there are side hints, not gaps.
    - Pack the outside sprites into matching hole rectangles, then repaint the fitted sprite to the color of
      that hole's aligned border hint.

    Key insight: The puzzle is still a rectangle-packing match, but the final color comes from a visible
    per-hole border paint chip rather than from the source sprite palette itself.
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
    hole_mask = dst == int(bg)

    out = np.array(dst, copy=True)

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
        hint_color = _hole_hint_color(dst, rr0, cc0, int(sp["h"]), int(sp["w"]), bg, base_col)
        recolored = _recolor_patch(sp["patch"], sp["mask"], bg, hint_color)
        paste(out, recolored, rr0, cc0, mask=sp["mask"], overwrite=True)

    return out


# --------------- generator ---------------

GRID_MIN, GRID_MAX = 15, 25
THICK_MIN, THICK_MAX = 7, 10
OUTER_GAP = 8
SPRITE_COUNT_MIN, SPRITE_COUNT_MAX = 7, 15
DOT_MIN, DOT_MAX = 1, 4
DOT_NON_DOT_BUFFER = 6
DOT_LONG_BUFFER = 4
HOLE_FRAC_LO, HOLE_FRAC_HI = 0.30, 0.42
CORRIDOR_COVERAGE = 0.78
MAX_RECT_LEN = 8
MAX_RECT_WIDTH = 6
MIN_SHAPE_GROUPS = 3
SOURCE_PALETTE_SIZE = 4
TARGET_PALETTE_SIZE = 3
BICHROMATIC_FRAC = 0.35
WIDTH_TOLERANCE = 2
MAX_RETRIES = 400


def _split_lengths(total: int, count: int) -> list[int] | None:
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


def _choose_widths(lengths: list[int], thick: int, dots: int, full_area: int) -> list[int] | None:
    max_w = min(MAX_RECT_WIDTH, thick - 2)
    lo = max(0, math.ceil(HOLE_FRAC_LO * full_area) - dots)
    hi = max(lo, math.floor(HOLE_FRAC_HI * full_area) - dots)
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


def _place_dot_markers(
    rects: list[tuple[int, int, int, int, bool]],
    occupied: np.ndarray,
    thick: int,
    dots: int,
) -> list[tuple[int, int, int, int, bool]] | None:
    for _ in range(dots):
        choices: list[tuple[int, int]] = []
        for r, c, h, w, _ in rects:
            for rr in range(r, r + h):
                for cc in (c - 1, c + w):
                    if 0 < cc < thick - 1 and not occupied[rr, cc]:
                        choices.append((rr, cc))
        if not choices:
            return None
        rr, cc = random.choice(choices)
        occupied[rr, cc] = True
        rects.append((rr, cc, 1, 1, True))
    return rects


def _corridor_rects(long: int, thick: int, count: int, dots: int) -> list[tuple[int, int, int, int, bool]] | None:
    min_len = max(count + 2, round(CORRIDOR_COVERAGE * long))
    if min_len > long - 1:
        min_len = long - 1
    hole_len = random.randint(min_len, long - 1)
    lengths = _split_lengths(hole_len, count)
    if lengths is None:
        return None
    widths = _choose_widths(lengths, thick, dots, long * thick)
    if widths is None:
        return None

    start = random.randint(0, long - hole_len)
    r = start
    prev_c = prev_w = None
    rects: list[tuple[int, int, int, int, bool]] = []
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
        rects.append((r, c, h, w, False))
        r += h
        prev_c, prev_w = c, w

    occupied = np.zeros((long, thick), dtype=bool)
    for r, c, h, w, _ in rects:
        occupied[r : r + h, c : c + w] = True

    return _place_dot_markers(rects, occupied, thick, dots)


def _make_source_patch(h: int, w: int, cols: list[int]) -> np.ndarray:
    """Create a 1-2 color source patch with a visible dominant color."""
    out = np.full((h, w), cols[0], dtype=int)
    if len(cols) == 1:
        return out

    area = int(h * w)
    assert area >= 3
    accent = int(cols[1])
    block_choices = [
        (bh, bw)
        for bh in range(1, h + 1)
        for bw in range(1, w + 1)
        if int(bh * bw) <= max(1, area // 3)
    ]
    if block_choices and random.random() < 0.55:
        bh, bw = random.choice(block_choices)
        out[h - bh :, w - bw :] = accent
        return out

    coords = [(r, c) for r in range(h) for c in range(w) if (r, c) != (0, 0)]
    max_accent = max(1, min(len(coords), max(1, area // 3)))
    accent_count = random.randint(1, max_accent)
    random.shuffle(coords)
    for r, c in coords[:accent_count]:
        out[r, c] = accent
    return out


def _make_source_pieces(
    rects: list[tuple[int, int, int, int, bool]],
    bg: int,
    base: int,
) -> tuple[dict[int, np.ndarray], list[int]] | None:
    groups: dict[tuple[int, int], list[int]] = {}
    for i, (_, _, h, w, is_dot) in enumerate(rects):
        if not is_dot:
            groups.setdefault((h, w), []).append(i)
    if len(groups) < MIN_SHAPE_GROUPS:
        return None

    nonzero = [c for c in range(1, 10) if c not in (bg, base)]
    if len(nonzero) < SOURCE_PALETTE_SIZE + TARGET_PALETTE_SIZE:
        return None
    source_palette = random.sample(nonzero, SOURCE_PALETTE_SIZE)
    target_pool = [c for c in nonzero if c not in source_palette]
    assert len(target_pool) == TARGET_PALETTE_SIZE

    keys = list(groups)
    random.shuffle(keys)
    bi_candidates = [key for key in keys if int(key[0] * key[1]) >= 3]
    bi_count = min(len(bi_candidates), max(1 if bi_candidates else 0, round(BICHROMATIC_FRAC * len(keys))))
    bi_keys = set(random.sample(bi_candidates, bi_count)) if bi_count else set()

    prototypes: dict[tuple[int, int], np.ndarray] = {}
    for key in keys:
        primary = random.choice(source_palette)
        cols = [int(primary)]
        if key in bi_keys:
            secondary_choices = [c for c in source_palette if c != primary]
            cols.append(int(random.choice(secondary_choices)))
        prototypes[key] = _make_source_patch(int(key[0]), int(key[1]), cols)

    pieces: dict[int, np.ndarray] = {}
    for i, (_, _, h, w, is_dot) in enumerate(rects):
        if is_dot:
            pieces[i] = np.array([[0]], dtype=int)
        else:
            pieces[i] = np.array(prototypes[(h, w)], copy=True)
    return pieces, target_pool


def _make_hints(
    rects: list[tuple[int, int, int, int, bool]],
    target_pool: list[int],
) -> dict[int, int]:
    hints: dict[int, int] = {}
    for i, _rect in enumerate(rects):
        hints[i] = int(random.choice(target_pool))
    return hints


def _paint_hints(
    canvas: np.ndarray,
    rects: list[tuple[int, int, int, int, bool]],
    hints: dict[int, int],
) -> None:
    vertical_strip = int(canvas.shape[0]) > int(canvas.shape[1])
    for i, (r, c, h, w, _is_dot) in enumerate(rects):
        hint = int(hints[i])
        if vertical_strip:
            mid_r = int(r + h // 2)
            canvas[mid_r, 0] = hint
        else:
            mid_c = int(c + w // 2)
            canvas[0, mid_c] = hint


def _render_output(
    dh: int,
    dw: int,
    bg: int,
    base: int,
    rects: list[tuple[int, int, int, int, bool]],
    pieces: dict[int, np.ndarray],
    hints: dict[int, int],
) -> np.ndarray:
    out = np.full((dh, dw), int(base), dtype=int)
    _paint_hints(out, rects, hints)
    for i, (r, c, _h, _w, _is_dot) in enumerate(rects):
        patch = pieces[i]
        recolored = _recolor_patch(patch, np.ones_like(patch, dtype=bool), bg, int(hints[i]))
        paste(out, recolored, r, c)
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
    rects: list[tuple[int, int, int, int, bool]],
    pieces: dict[int, np.ndarray],
    hints: dict[int, int],
) -> np.ndarray:
    inp = np.full((H, W), int(bg), dtype=int)
    inp[tr0 : tr0 + dh, tc0 : tc0 + dw] = int(base)
    template = inp[tr0 : tr0 + dh, tc0 : tc0 + dw]
    _paint_hints(template, rects, hints)
    for r, c, h, w, _ in rects:
        inp[tr0 + r : tr0 + r + h, tc0 + c : tc0 + c + w] = int(bg)

    placement = np.full((H, W), int(bg), dtype=int)
    placement[tr0 : tr0 + dh, tc0 : tc0 + dw] = inp[tr0 : tr0 + dh, tc0 : tc0 + dw]
    order = list(range(len(rects)))
    random.shuffle(order)
    order.sort(key=lambda i: pieces[i].size, reverse=True)
    for i in order:
        patch = pieces[i]
        r, c = random_free_location(placement, patch, transparent=bg, padding=1, padding_connectivity=8)
        paste(inp, patch, r, c)
        paste(placement, patch, r, c)
    return inp


def _variant_visible(
    rects: list[tuple[int, int, int, int, bool]],
    pieces: dict[int, np.ndarray],
    hints: dict[int, int],
) -> bool:
    changed = 0
    used_targets: set[int] = set()
    for i, _rect in enumerate(rects):
        patch = pieces[i]
        hint = int(hints[i])
        used_targets.add(hint)
        recolored = _recolor_patch(patch, np.ones_like(patch, dtype=bool), bg=-1, hint_color=hint)
        if not np.array_equal(recolored, patch):
            changed += 1
    return changed >= max(3, len(rects) // 3) and len(used_targets) >= 2


def _gen_attempt() -> np.ndarray | None:
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
    min_dots = max(DOT_MIN, total - (long - DOT_LONG_BUFFER))
    max_dots = min(DOT_MAX, total - DOT_NON_DOT_BUFFER)
    if min_dots > max_dots:
        return None
    dots = random.randint(min_dots, max_dots)

    canon = _corridor_rects(long, thick, total - dots, dots)
    if canon is None:
        return None
    rects = canon if vertical else [(c, r, w, h, d) for r, c, h, w, d in canon]

    bg, base = random.sample(range(1, 10), 2)
    piece_info = _make_source_pieces(rects, bg, base)
    if piece_info is None:
        return None
    pieces, target_pool = piece_info
    hints = _make_hints(rects, target_pool)
    if not _variant_visible(rects, pieces, hints):
        return None

    out = _render_output(dh, dw, bg, base, rects, pieces, hints)
    try:
        inp = _scatter_sprites(H, W, bg, base, tr0, tc0, dh, dw, rects, pieces, hints)
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
    raise RuntimeError("failed to generate valid 5dbc8537 D4 input")
