# Variant of 5dbc8537
# Dimension: D4 — 填充变换方式 (transform)
# Change: 将精灵原样粘贴到匹配的洞中（精灵的背景色像素被基底色替换），输出为紧裁后的模板外接矩形... → 镜像翻转填入

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

# Design analysis for the D4 variant
# ----------------------------------
# - Available references:
#   - The original 5dbc8537 solver and the provided spec were read in full.
#   - The prompt asks to inspect source task `b10624e5`, but inside the allowed workspace
#     there is no readable `b10624e5.py` solver copy; only dataset/json references exist.
#     The implementation below therefore keeps the reflection mechanism simple, explicit,
#     and locally verifiable rather than importing behavior from an unavailable solver.
#
# spec_variant:
#   task_type: move_to_match + placement-dependent reflection
#   template:
#     same_as_original: true
#     bbox: one full grid dimension by a 7-10 cell strip
#     hole_region: connected union of axis-aligned rectangles, tiled exactly by sprite bboxes
#   sprites:
#     location: outside the template bbox
#     matching_key: rectangle bbox size only, as in the original task
#     transform:
#       vertical_strip:
#         upper_half: horizontal mirror (left-right reflection)
#         lower_half: vertical mirror (top-bottom reflection)
#       horizontal_strip:
#         left_half: vertical mirror
#         right_half: horizontal mirror
#       branch_test: use the rectangle center relative to the strip midpoint; exact-midpoint ties
#         go to the lower/right branch
#   generator_adaptations:
#     necessity: mirror effects must be visible, so most non-dot sprites need asymmetric fills
#     visibility_guards:
#       - require at least two non-dot rectangles in each branch
#       - require several rectangles whose mirrored output differs from the raw sprite
#       - reject cases where mirrored output equals the old "paste-as-is" output
#
# behavior_trace:
#   original:
#     1. locate template strip
#     2. recover outside sprites and match them to hole rectangles by bbox
#     3. paste the sprite patch directly into the matched hole
#   variant:
#     1. same template and same bbox-matching
#     2. after a sprite is assigned to a hole, inspect that hole's position within the strip
#     3. mirror the sprite before pasting, with the axis chosen by strip orientation + half
#   example:
#     - In a vertical strip, a 3x4 sprite placed in the upper half is pasted as `np.fliplr(sprite)`.
#     - The same sprite shape placed in the lower half is pasted as `np.flipud(sprite)`.
#     - In the old task those two outputs would be identical to the input sprite; here they differ.
#
# implicit-assumption audit:
#   - Base-color detection by largest non-bg component still holds because the template frame
#     generation is unchanged.
#   - Hole matching by bbox still holds because reflection preserves rectangle size.
#   - The original final `paste(out, sp["patch"], ...)` relied on "destination transform = none".
#     That is the only solver step that becomes invalid under D4, so the main code changes only
#     there plus a small helper to choose the mirror axis from the destination slot.
#   - The DFS packer assumes the hole rectangles form an exact cover for the sprite bboxes.
#     The generator continues to guarantee this and additionally guarantees both reflection
#     branches appear and remain visually observable.
#
# legality / cleanliness:
#   - Sprites are still scattered outside the template with >=1 bg padding.
#   - The template holes remain disjoint rectangles inside a single strip, so no overlapping
#     layers are introduced.
#   - Mirror choice uses only spatially visible evidence; color ids are never used as control
#     signals for ordering or transformation.


def _slot_transform_kind(rr0: int, cc0: int, h: int, w: int, dh: int, dw: int) -> str:
    """Return 'h' for left-right mirror and 'v' for top-bottom mirror."""
    if dh > dw:  # vertical strip: split by upper/lower half
        center2 = 2 * int(rr0) + int(h) - 1
        return "h" if center2 < int(dh) - 1 else "v"

    center2 = 2 * int(cc0) + int(w) - 1
    return "v" if center2 < int(dw) - 1 else "h"


def _apply_slot_transform(patch: np.ndarray, kind: str) -> np.ndarray:
    if kind == "h":
        return np.fliplr(patch)
    if kind == "v":
        return np.flipud(patch)
    raise ValueError(f"unknown transform kind: {kind}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the template cavities with outside sprites, but mirror each sprite according to slot position.

    Input structure:
    - The most common color is the background. One large non-background strip of a single base color acts as the template.
    - Rectangular sprites lie outside that strip, while rectangular gaps of matching sizes are carved inside it.

    Rule:
    - Detect the template from the largest non-background connected component and crop to its bbox.
    - Match outside sprites to the hole rectangles by exact bbox cover, then mirror each sprite before insertion.
    - In a vertical strip, upper-half slots use a left-right mirror and lower-half slots use a top-bottom mirror; for a horizontal strip, rotate that rule with the strip.

    Key insight:
    - The placement search is unchanged from the original puzzle: only the final write step changes, because each matched slot tells you which mirror to apply.
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
        kind = _slot_transform_kind(rr0, cc0, int(sp["h"]), int(sp["w"]), dh, dw)
        patch = _apply_slot_transform(sp["patch"], kind)
        mask = _apply_slot_transform(sp["mask"].astype(int), kind).astype(bool)
        paste(out, patch, rr0, cc0, mask=mask, overwrite=True)

    return out


# --------------- generator ---------------

GRID_MIN, GRID_MAX = 15, 25
THICK_MIN, THICK_MAX = 7, 10
OUTER_GAP = 8
SPRITE_COUNT_MIN, SPRITE_COUNT_MAX = 7, 14
DOT_MIN, DOT_MAX = 1, 2
DOT_NON_DOT_BUFFER = 6
DOT_LONG_BUFFER = 4
HOLE_FRAC_LO, HOLE_FRAC_HI = 0.39, 0.50
CORRIDOR_COVERAGE = 0.78
MAX_RECT_LEN = 8
MAX_RECT_WIDTH = 6
MIN_SHAPE_GROUPS = 3
MIN_PALETTE_SIZE = 5
WIDTH_TOLERANCE = 2
MAX_RETRIES = 400
MIN_BRANCH_PIECES = 2
MIN_CHANGED_PIECES = 4
MIN_CHANGED_CELLS = 10


def _split_lengths(total: int, count: int) -> list[int] | None:
    """Partition *total* into *count* positive integers, each <= MAX_RECT_LEN."""
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
    """Pick widths so total hole area falls within the original density band."""
    max_w = min(MAX_RECT_WIDTH, thick - 2)
    lo = max(0, math.ceil(HOLE_FRAC_LO * full_area) - dots)
    hi = max(lo, math.floor(HOLE_FRAC_HI * full_area) - dots)
    paths: dict[int, list[int]] = {0: []}
    for h in lengths:
        opts = list(range(1, max_w + 1))
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
    """Attach *dots* 1x1 markers adjacent to existing rects."""
    for _ in range(dots):
        choices: list[tuple[int, int]] = []
        for r, c, h, w, _ in rects:
            for rr in range(r, r + h):
                for cc in (c - 1, c + w):
                    if 0 <= cc < thick and not occupied[rr, cc]:
                        choices.append((rr, cc))
        if not choices:
            return None
        rr, cc = random.choice(choices)
        occupied[rr, cc] = True
        rects.append((rr, cc, 1, 1, True))
    return rects


def _corridor_rects(long: int, thick: int, count: int, dots: int) -> list[tuple[int, int, int, int, bool]] | None:
    """Generate connected stacked rectangles along the long axis, plus a few dot markers."""
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


def _make_patch(h: int, w: int, cols: list[int]) -> np.ndarray:
    """Build an asymmetric 2-color rectangle so flips stay visually observable."""
    a, b = map(int, cols)
    out = np.full((h, w), a, dtype=int)

    if h == 1 and w == 1:
        out[0, 0] = a
        return out

    if h == 1:
        seg = random.randint(1, w - 1)
        if random.random() < 0.5:
            out[0, :seg] = b
        else:
            out[0, w - seg :] = b
        return out

    if w == 1:
        seg = random.randint(1, h - 1)
        if random.random() < 0.5:
            out[:seg, 0] = b
        else:
            out[h - seg :, 0] = b
        return out

    pattern = random.choice(["corner_l", "offset_split", "stair"])
    if pattern == "corner_l":
        top = random.randint(1, max(1, h - 1))
        left = random.randint(1, max(1, w - 1))
        corner = random.choice(["tl", "tr", "bl", "br"])
        if corner == "tl":
            out[:top, 0] = b
            out[0, :left] = b
        elif corner == "tr":
            out[:top, w - 1] = b
            out[0, w - left :] = b
        elif corner == "bl":
            out[h - top :, 0] = b
            out[h - 1, :left] = b
        else:
            out[h - top :, w - 1] = b
            out[h - 1, w - left :] = b
    elif pattern == "offset_split":
        if random.random() < 0.5:
            cut = random.randint(1, h - 1)
            out[cut:, :] = b
            if w > 2:
                out[:, : random.randint(1, w - 1)] = a
        else:
            cut = random.randint(1, w - 1)
            out[:, cut:] = b
            if h > 2:
                out[: random.randint(1, h - 1), :] = a
    else:
        steps = max(2, min(h, w))
        for k in range(steps):
            rr = min(h - 1, k)
            cc = min(w - 1, (2 * k) % w)
            out[rr:, cc] = b
        if np.all(out == b):
            out[0, 0] = a
    return out


def _make_pieces(
    rects: list[tuple[int, int, int, int, bool]],
    bg: int,
    base: int,
) -> dict[int, np.ndarray] | None:
    """Assign colors and build asymmetric patch arrays for each rect."""
    groups: dict[tuple[int, int], list[int]] = {}
    for i, (_, _, h, w, is_dot) in enumerate(rects):
        if not is_dot:
            groups.setdefault((h, w), []).append(i)
    if len(groups) < MIN_SHAPE_GROUPS:
        return None

    available = [c for c in range(1, 10) if c not in (bg, base)]
    if len(available) < MIN_PALETTE_SIZE:
        return None

    keys = list(groups)
    random.shuffle(keys)
    palette = random.sample(available, max(MIN_PALETTE_SIZE, min(len(available), len(keys) + 2)))
    group_cols: dict[tuple[int, int], list[int]] = {}
    for idx, key in enumerate(keys):
        a = palette[idx % len(palette)]
        b_choices = [c for c in available if c != a]
        b = random.choice(b_choices)
        group_cols[key] = [int(a), int(b)]

    pieces: dict[int, np.ndarray] = {}
    for i, (_, _, h, w, is_dot) in enumerate(rects):
        pieces[i] = np.array([[0]], dtype=int) if is_dot else _make_patch(h, w, group_cols[(h, w)])
    return pieces


def _render_output(
    dh: int,
    dw: int,
    base: int,
    rects: list[tuple[int, int, int, int, bool]],
    pieces: dict[int, np.ndarray],
    *,
    transformed: bool,
) -> np.ndarray:
    """Build either the old direct-paste output or the reflected variant output."""
    out = np.full((dh, dw), base, dtype=int)
    for i, (r, c, h, w, _) in enumerate(rects):
        patch = pieces[i]
        if transformed:
            kind = _slot_transform_kind(r, c, h, w, dh, dw)
            patch = _apply_slot_transform(patch, kind)
        paste(out, patch, r, c)
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
) -> np.ndarray:
    """Build the input grid: template frame with holes plus raw, unmirrored sprites."""
    inp = np.full((H, W), bg, dtype=int)
    inp[tr0 : tr0 + dh, tc0 : tc0 + dw] = base
    for r, c, h, w, _ in rects:
        inp[tr0 + r : tr0 + r + h, tc0 + c : tc0 + c + w] = bg

    placement = np.full((H, W), bg, dtype=int)
    placement[tr0 : tr0 + dh, tc0 : tc0 + dw] = base
    order = list(range(len(rects)))
    random.shuffle(order)
    order.sort(key=lambda i: pieces[i].size, reverse=True)
    for i in order:
        patch = pieces[i]
        r, c = random_free_location(placement, patch, transparent=bg, padding=1, padding_connectivity=8)
        paste(inp, patch, r, c)
        paste(placement, patch, r, c)
    return inp


def _branch_metrics(
    dh: int,
    dw: int,
    rects: list[tuple[int, int, int, int, bool]],
    pieces: dict[int, np.ndarray],
) -> tuple[dict[str, int], dict[str, int], int]:
    """Count branch coverage and how many pieces truly change under the mirror rule."""
    branch_counts = {"h": 0, "v": 0}
    visible_counts = {"h": 0, "v": 0}
    changed_pieces = 0
    for i, (r, c, h, w, is_dot) in enumerate(rects):
        if is_dot:
            continue
        kind = _slot_transform_kind(r, c, h, w, dh, dw)
        branch_counts[kind] += 1
        if (kind == "h" and w > 1) or (kind == "v" and h > 1):
            visible_counts[kind] += 1
        if not np.array_equal(pieces[i], _apply_slot_transform(pieces[i], kind)):
            changed_pieces += 1
    return branch_counts, visible_counts, changed_pieces


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: sample layout, build raw sprites, mirror them in output, verify with solver."""
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
    pieces = _make_pieces(rects, bg, base)
    if pieces is None:
        return None

    branch_counts, visible_counts, changed_pieces = _branch_metrics(dh, dw, rects, pieces)
    if any(branch_counts[k] < MIN_BRANCH_PIECES for k in ("h", "v")):
        return None
    if any(visible_counts[k] < MIN_BRANCH_PIECES for k in ("h", "v")):
        return None
    if changed_pieces < MIN_CHANGED_PIECES:
        return None

    out = _render_output(dh, dw, base, rects, pieces, transformed=True)
    old_out = _render_output(dh, dw, base, rects, pieces, transformed=False)
    changed_cells = int(np.sum(out != old_out))
    if changed_cells < MIN_CHANGED_CELLS:
        return None

    try:
        inp = _scatter_sprites(H, W, bg, base, tr0, tc0, dh, dw, rects, pieces)
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
