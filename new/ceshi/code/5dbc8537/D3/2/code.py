# Variant of 5dbc8537
# Dimension: D3 — 目标洞的结构与分布 (dst)
# Change: 模板框架内的矩形洞区域，所有洞形成单个4-连通的连续背景色区域，洞总面积精确等于所有精灵面积之和（完美拼贴）... → 洞内有引导标记

from __future__ import annotations

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

# -----------------------------------------------------------------------------
# Design analysis for the D3 marker-alignment variant
#
# a) Concrete behavior and flow impact
#    - The overall scene stays close to the original task: a large base-colored
#      strip acts as the destination template, while full rectangular sprites are
#      scattered outside it.
#    - The changed part is inside the strip. Each rectangular destination slot is
#      still a rectangle of non-base cells, but it is no longer pure background.
#      Instead, the slot is mostly bg with 1-2 colored guide pixels placed on a
#      corner or edge cell of that rectangle. Those guide pixels copy the target
#      sprite's own boundary colors at the same relative offsets.
#    - Small example:
#        old: a 3x4 sprite could go into any 3x4 all-bg rectangle inside the
#             connected cavity; matching relied only on the packing geometry.
#        new: two 3x4 slots may exist. One slot contains a red top-left pixel,
#             the other a yellow right-edge pixel. The solver must test
#             "same size" first, then also require that the candidate slot's
#             visible guide pixels match the sprite patch at the same offsets.
#      This changes the candidate-filtering step, not the outer template or the
#      final paste.
#
# b) Complexity calibration
#    - If the generator kept unique rectangle sizes, the guides would be merely
#      decorative because size-only placement would still solve the puzzle.
#    - To force the new mechanism to matter, the generator deliberately creates
#      repeated (h,w) slot sizes and gives same-size pieces distinct guide
#      signatures. This keeps the rule visible without introducing rotation,
#      mirroring, or hidden metadata.
#    - Guide counts are kept to 1-2 per slot so the input remains readable.
#
# c) Implicit-assumption audit against the original main()
#    - `hole_mask = dst != base_col` still works, but the meaning changes:
#      hole cells now include bg cells and colored guide cells.
#    - The old "all-hole rectangle of same bbox" search is no longer sufficient
#      because several same-sized placements can exist. A candidate must also
#      satisfy the slot's visible guide pixels.
#    - Pasting the whole sprite into the chosen slot remains safe because each
#      guide pixel already equals the sprite value at that location; there is no
#      destructive overwrite of correct information.
#
# d) Placement legality
#    - Rectangles are still non-overlapping, inside the template bbox, and their
#      union remains one 4-connected corridor-like region.
#    - Sprites are scattered outside the template with >=1 bg-cell padding, so
#      component extraction stays reliable.
#    - Every slot receives at least one guide pixel, and same-size slots are
#      assigned distinct guide signatures.
#
# e) Visual observability
#    - A human can see colored dots already sitting in the empty destination
#      rectangles and match them to the same boundary dots on outside sprites.
#    - Repeated slot sizes appear in most generated examples, so the guide rule
#      is not a rare corner case.
#
# f) Visual cleanliness
#    - Guides are only placed on boundary cells of a slot and never layered on
#      top of the base frame, so the template stays legible.
#    - There are no extra singleton distractors inside the template; every
#      colored guide pixel belongs to exactly one slot.
#
# g) Color inference
#    - No color-id ordering is used. Guide colors are literal copied pixels from
#      the target sprite, so the observed relation is structural: "this slot
#      shows a subset of the piece's own boundary colors."
#
# New solver assumptions introduced here and guaranteed by the generator:
#    1. Every real slot is a full rectangle of non-base cells.
#    2. Every real slot contains at least one non-bg guide pixel.
#    3. Among slots of the same size, guide signatures are unique.
# -----------------------------------------------------------------------------


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a template strip by matching scattered sprites to destination slots that show boundary guide pixels.

    Input structure:
    - bg = most common color. A large base-colored strip marks the destination bbox.
    - Outside that strip are full rectangular sprites; inside it, each slot is a rectangle of non-base cells that contains 1-2 colored guide pixels on its boundary.

    Rule:
    - The largest non-bg connected color is the base strip and defines the cropped output region.
    - A sprite may only occupy a same-sized slot whose visible guide pixels match the sprite at the same relative offsets.
    - Find a non-overlapping assignment of all sprites to slots, then paste the sprites into the strip and return the cropped template.

    Key insight:
    - Rectangle size still narrows the search, but repeated slot sizes are disambiguated by the guide pixels already shown inside each hole.
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
    dh, dw = map(int, dst.shape)
    hole_mask = dst != int(base_col)
    guide_mask = (dst != int(base_col)) & (dst != int(bg))

    out = np.full((dh, dw), int(base_col), dtype=int)

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
        sprites.append({"h": int(ph), "w": int(pw), "mask": m, "patch": patch})

    # Larger sprites first; tie-break with guide count in their matching windows.
    sprites.sort(key=lambda s: (-int(s["h"] * s["w"]), -int(s["mask"].sum())))

    # --- candidate placements per sprite: same-sized all-hole rectangles whose guides match the patch ---
    candidates: list[list[tuple[int, int]]] = []
    areas = [int(s["h"] * s["w"]) for s in sprites]
    assert sum(areas) == int(hole_mask.sum())
    for sp in sprites:
        h, w = int(sp["h"]), int(sp["w"])
        pos: list[tuple[int, int]] = []
        if h <= dh and w <= dw:
            for rr0 in range(int(dh - h + 1)):
                for cc0 in range(int(dw - w + 1)):
                    window_mask = hole_mask[rr0 : rr0 + h, cc0 : cc0 + w]
                    if not bool(window_mask.all()):
                        continue
                    window_guides = guide_mask[rr0 : rr0 + h, cc0 : cc0 + w]
                    if not bool(window_guides.any()):
                        continue
                    if not np.array_equal(
                        sp["patch"][window_guides],
                        dst[rr0 : rr0 + h, cc0 : cc0 + w][window_guides],
                    ):
                        continue
                    pos.append((int(rr0), int(cc0)))
        candidates.append(pos)

    assert all(candidates)
    used = ~hole_mask  # base pixels are treated as already occupied
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

    # --- paste according to the guide-constrained exact cover placement ---
    for i, pos in enumerate(placement):
        assert pos is not None
        rr0, cc0 = map(int, pos)
        sp = sprites[i]
        paste(out, sp["patch"], rr0, cc0, mask=sp["mask"], overwrite=True)

    return out


# --------------- generator ---------------

GRID_MIN, GRID_MAX = 15, 25
THICK_MIN, THICK_MAX = 7, 10
OUTER_GAP = 8
SPRITE_COUNT_MIN, SPRITE_COUNT_MAX = 7, 12
HOLE_FRAC_LO, HOLE_FRAC_HI = 0.39, 0.50
CORRIDOR_COVERAGE = 0.78
MAX_RECT_LEN = 8
MAX_RECT_WIDTH = 6
MIN_SHAPE_GROUPS = 3
MIN_DUP_GROUPS = 2
MIN_PALETTE_SIZE = 5
MAX_RETRIES = 400


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


def _shape_stats(shapes: list[tuple[int, int]]) -> tuple[int, int]:
    counts: dict[tuple[int, int], int] = {}
    for key in shapes:
        counts[key] = counts.get(key, 0) + 1
    return len(counts), sum(1 for v in counts.values() if v >= 2)


def _choose_widths(lengths: list[int], thick: int, full_area: int) -> list[int] | None:
    """Choose widths that satisfy the area range and force repeated (h,w) groups."""
    max_w = min(MAX_RECT_WIDTH, thick - 2)
    if max_w < 2:
        return None
    lo = math_ceil(HOLE_FRAC_LO * full_area)
    hi = math_floor(HOLE_FRAC_HI * full_area)
    options = list(range(2, max_w + 1))
    for _ in range(220):
        palette_size = random.randint(2, min(3, len(options)))
        palette = random.sample(options, palette_size)
        widths = [random.choice(palette) for _ in lengths]
        area = sum(int(h) * int(w) for h, w in zip(lengths, widths))
        if not (lo <= area <= hi):
            continue
        distinct, dup_groups = _shape_stats(list(zip(lengths, widths)))
        if distinct < MIN_SHAPE_GROUPS or dup_groups < MIN_DUP_GROUPS:
            continue
        return widths
    return None


def math_ceil(x: float) -> int:
    return int(np.ceil(float(x)))


def math_floor(x: float) -> int:
    return int(np.floor(float(x)))


def _corridor_rects(long: int, thick: int, count: int) -> list[tuple[int, int, int, int]] | None:
    """Generate connected stacked rectangles along the long axis."""
    min_len = max(count + 2, round(CORRIDOR_COVERAGE * long))
    if min_len > long - 1:
        min_len = long - 1
    for _ in range(120):
        hole_len = random.randint(min_len, long - 1)
        lengths = _split_lengths(hole_len, count)
        if lengths is None:
            continue
        widths = _choose_widths(lengths, thick, long * thick)
        if widths is None:
            continue

        start = random.randint(0, long - hole_len)
        r = start
        prev_c = prev_w = None
        rects: list[tuple[int, int, int, int]] = []
        ok = True
        for h, w in zip(lengths, widths):
            c_hi = thick - w - 1
            if c_hi < 1:
                ok = False
                break
            if prev_c is None:
                c = random.randint(1, c_hi)
            else:
                c_lo = max(1, prev_c - w + 1)
                c_hi = min(c_hi, prev_c + prev_w - 1)
                if c_lo > c_hi:
                    ok = False
                    break
                c = random.randint(c_lo, c_hi)
            rects.append((r, c, h, w))
            r += h
            prev_c, prev_w = c, w
        if ok:
            return rects
    return None


def _guide_positions(h: int, w: int) -> list[tuple[int, int]]:
    """Candidate guide cells: corners plus mid-edge cells when distinct."""
    pts = [
        (0, 0),
        (0, w - 1),
        (h - 1, 0),
        (h - 1, w - 1),
        (0, w // 2),
        (h - 1, w // 2),
        (h // 2, 0),
        (h // 2, w - 1),
    ]
    uniq: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for p in pts:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _make_pieces(
    rects: list[tuple[int, int, int, int]],
    bg: int,
    base: int,
) -> tuple[dict[int, np.ndarray], dict[int, tuple[tuple[int, int, int], ...]]] | None:
    """Assign simple two-color patches and unique guide signatures within each size group."""
    groups: dict[tuple[int, int], list[int]] = {}
    for i, (_r, _c, h, w) in enumerate(rects):
        groups.setdefault((h, w), []).append(i)
    distinct = len(groups)
    dup_groups = sum(1 for ids in groups.values() if len(ids) >= 2)
    if distinct < MIN_SHAPE_GROUPS or dup_groups < MIN_DUP_GROUPS:
        return None

    available = [c for c in range(1, 10) if c not in (bg, base)]
    if len(available) < MIN_PALETTE_SIZE:
        return None
    palette = random.sample(available, random.randint(MIN_PALETTE_SIZE, len(available)))

    pieces: dict[int, np.ndarray] = {}
    guides: dict[int, tuple[tuple[int, int, int], ...]] = {}
    for (h, w), ids in groups.items():
        anchors = _guide_positions(h, w)
        used_sigs: set[tuple[tuple[int, int, int], ...]] = set()
        for idx in ids:
            for _ in range(300):
                primary = random.choice(palette)
                accent_choices = [c for c in palette if c != primary]
                if not accent_choices:
                    continue
                accent = random.choice(accent_choices)
                marker_count = 2 if len(ids) > 1 else random.choice([1, 2])
                marker_count = min(marker_count, len(anchors))
                pos = tuple(sorted(random.sample(anchors, marker_count)))
                sig = tuple(sorted((int(r), int(c), int(accent)) for r, c in pos))
                if sig in used_sigs:
                    continue
                patch = np.full((h, w), int(primary), dtype=int)
                for r, c in pos:
                    patch[int(r), int(c)] = int(accent)
                pieces[idx] = patch
                guides[idx] = sig
                used_sigs.add(sig)
                break
            else:
                return None
    return pieces, guides


def _render_output(
    dh: int,
    dw: int,
    base: int,
    rects: list[tuple[int, int, int, int]],
    pieces: dict[int, np.ndarray],
) -> np.ndarray:
    out = np.full((dh, dw), base, dtype=int)
    for i, (r, c, _h, _w) in enumerate(rects):
        paste(out, pieces[i], r, c)
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
    pieces: dict[int, np.ndarray],
    guides: dict[int, tuple[tuple[int, int, int], ...]],
) -> np.ndarray:
    """Build the input grid: base strip with guide-marked slots + scattered sprites outside."""
    inp = np.full((H, W), bg, dtype=int)
    inp[tr0 : tr0 + dh, tc0 : tc0 + dw] = base

    for i, (r, c, h, w) in enumerate(rects):
        inp[tr0 + r : tr0 + r + h, tc0 + c : tc0 + c + w] = bg
        for mr, mc, color in guides[i]:
            inp[tr0 + r + mr, tc0 + c + mc] = int(color)

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


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: sample layout, build pieces, scatter sprites, oracle-verify."""
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
    canon = _corridor_rects(long, thick, total)
    if canon is None:
        return None
    rects = canon if vertical else [(c, r, w, h) for r, c, h, w in canon]

    bg, base = random.sample(range(1, 10), 2)
    made = _make_pieces(rects, bg, base)
    if made is None:
        return None
    pieces, guides = made

    out = _render_output(dh, dw, base, rects, pieces)
    try:
        inp = _scatter_sprites(H, W, bg, base, tr0, tc0, dh, dw, rects, pieces, guides)
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
    raise RuntimeError("failed to generate valid 5dbc8537 D3 input")
