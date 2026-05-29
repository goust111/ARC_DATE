# Combo variant of 5dbc8537 (index 16)
# combo_spec: D8/0+D10/0
# Name: 少量大精灵多独立洞组

from __future__ import annotations

import itertools
import random
from typing import Iterable

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
# a) Concrete behavior shift:
#    - Original: one long 4-connected hole corridor, 7-15 medium/small sprites, global size-only packing.
#    - New: 2-3 disconnected hole groups inside the same strip template, but only 3-5 large sprites.
#      A small example is a vertical strip with group A made from a 5x4 block touching a 4x3 block,
#      then two base-only rows, then group B as a single 6x5 block. A solver can no longer assume the
#      hole area is one region; it must first separate hole groups, then decide which subset of sprites
#      belongs to each group, then place within that group.
# b) Complexity calibration:
#    - The generator keeps only 3-5 sprites, but each piece is large enough that one bad assignment is
#      decisive. Group count is capped at 2-3 and per-group piece count at 1-3, which keeps the rule
#      visible and prevents an explosion of equivalent packings.
# c) Broken original assumptions:
#    - `sum(sprite areas) == single hole area` is no longer sufficient structure; the hole may have
#      several connected components.
#    - A single global DFS over all placements still works in theory, but it misses the intended
#      reasoning step. The new solver therefore decomposes the hole by 4-connectivity and searches
#      subsets per group.
#    - Dot markers and tiny fillers are removed; all sprites are full rectangles with substantially
#      larger bounding boxes.
# d) Placement legality:
#    - Every rectangle stays strictly inside the strip thickness, so the base-colored frame remains
#      connected around the holes.
#    - Groups are laid out sequentially along the strip's long axis with at least one base-only gap,
#      so they are disconnected and never overlap.
#    - Scattered source sprites are placed with `random_free_location(..., padding=1)`, keeping them
#      isolated from each other and from the template.
# e) Visual observability:
#    - A human can see 2-3 separate hole islands inside the strip and only a few large outer sprites.
#      That is a strong visible departure from the original "many fragments in one corridor" regime.
# f) Visual cleanliness:
#    - The template uses one strip, holes are grouped along one axis, and sprites are large rectangles.
#      There is no extra marker layer or color-based metadata overlaying the geometry.
# g) Color inference risk:
#    - Colors remain decorative. Placement is inferable from rectangle size and group shape, not from
#      hidden numeric color ordering. Bichromatic fills are allowed, but only as sprite texture.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct a strip template by returning large outside sprites to several disconnected hole groups.

    Input structure:
    - The most common color is background. One dominant non-background color forms a thick full-height
      or full-width strip that acts as the template frame.
    - Inside that strip, the missing cells are not one corridor anymore: they form 2-3 separate
      4-connected hole groups.
    - Outside the strip sit only a few large rectangular sprites, each fully filling its own bbox.

    Rule:
    - Detect the strip from the largest non-background connected component color and crop to its bbox.
    - Split the strip's missing cells into disconnected hole groups.
    - Partition the outside sprites across those groups, and within each group place the matching
      rectangles so that the group is tiled exactly.

    Key insight:
    - The task is no longer a single global packing problem. The crucial extra step is to recognize that
      the cavity has multiple independent islands, so sprites must be assigned to a hole group before
      local rectangle matching can finish the fill.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    base_candidates: list[tuple[int, int]] = []
    for col in colors_present(g, exclude={bg}):
        max_area = 0
        for coords in components_mask(g == int(col), connectivity=8):
            max_area = max(max_area, int(coords.shape[0]))
        base_candidates.append((max_area, int(col)))

    assert base_candidates
    base_candidates.sort(reverse=True)
    if len(base_candidates) > 1:
        assert base_candidates[0][0] > base_candidates[1][0]
    base_col = int(base_candidates[0][1])

    base_coords = np.argwhere(g == int(base_col))
    assert base_coords.size
    r0, c0, r1, c1 = bbox_coords(base_coords)
    dst = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))
    dh, dw = map(int, dst.shape)
    hole_mask = dst != int(base_col)

    hole_groups = components_mask(hole_mask, connectivity=4)
    assert hole_groups
    group_masks: list[np.ndarray] = []
    for coords in hole_groups:
        gm = np.zeros((dh, dw), dtype=bool)
        gm[coords[:, 0], coords[:, 1]] = True
        group_masks.append(gm)

    masked = g.copy()
    masked[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)
    sprite_comps = find_components(masked, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites: list[dict] = []
    for comp in sprite_comps:
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        ph, pw = map(int, patch.shape)
        mask = np.zeros((ph, pw), dtype=bool)
        coords = comp["coords"]
        mask[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        sprites.append(
            {
                "h": ph,
                "w": pw,
                "area": int(mask.sum()),
                "mask": mask,
                "patch": patch,
            }
        )

    assert sprites
    order = sorted(range(len(sprites)), key=lambda i: (-sprites[i]["area"], -sprites[i]["h"], -sprites[i]["w"]))
    sprites = [sprites[i] for i in order]
    sprite_areas = [int(sp["area"]) for sp in sprites]
    assert sum(sprite_areas) == int(hole_mask.sum())

    group_data: list[dict] = []
    for gm in group_masks:
        area = int(gm.sum())
        candidates: list[list[tuple[int, int]]] = []
        for sp in sprites:
            h, w = int(sp["h"]), int(sp["w"])
            pos: list[tuple[int, int]] = []
            if h <= dh and w <= dw:
                for rr0 in range(dh - h + 1):
                    for cc0 in range(dw - w + 1):
                        if bool(gm[rr0 : rr0 + h, cc0 : cc0 + w].all()):
                            pos.append((rr0, cc0))
            candidates.append(pos)
        feasible = sum(1 for pos in candidates if pos)
        group_data.append({"mask": gm, "area": area, "candidates": candidates, "feasible": feasible})

    group_order = sorted(
        range(len(group_data)),
        key=lambda gi: (group_data[gi]["feasible"], -group_data[gi]["area"]),
    )

    def subset_masks(indices: list[int]) -> Iterable[tuple[int, ...]]:
        n = len(indices)
        for bits in range(1, 1 << n):
            subset = [indices[j] for j in range(n) if (bits >> j) & 1]
            yield tuple(subset)

    def cover_group(gi: int, subset: tuple[int, ...]) -> dict[int, tuple[int, int]] | None:
        gm = group_data[gi]["mask"]
        used = ~gm.copy()
        placement: dict[int, tuple[int, int]] = {}
        local = sorted(subset, key=lambda si: (len(group_data[gi]["candidates"][si]), -sprites[si]["area"]))

        def dfs(idx: int) -> bool:
            if idx == len(local):
                return bool(used.all())
            si = local[idx]
            sp = sprites[si]
            h, w = int(sp["h"]), int(sp["w"])
            for rr0, cc0 in group_data[gi]["candidates"][si]:
                if bool(used[rr0 : rr0 + h, cc0 : cc0 + w].any()):
                    continue
                used[rr0 : rr0 + h, cc0 : cc0 + w] = True
                placement[si] = (rr0, cc0)
                if dfs(idx + 1):
                    return True
                used[rr0 : rr0 + h, cc0 : cc0 + w] = False
                placement.pop(si, None)
            return False

        if dfs(0):
            return placement
        return None

    def assign_groups(remaining: tuple[int, ...], order_idx: int) -> dict[int, tuple[int, int]] | None:
        if order_idx == len(group_order):
            return {} if not remaining else None

        gi = group_order[order_idx]
        target_area = int(group_data[gi]["area"])
        remaining_list = list(remaining)
        tried: set[tuple[int, ...]] = set()
        for subset in subset_masks(remaining_list):
            if subset in tried:
                continue
            tried.add(subset)
            if sum(sprite_areas[si] for si in subset) != target_area:
                continue
            if any(not group_data[gi]["candidates"][si] for si in subset):
                continue
            placement = cover_group(gi, subset)
            if placement is None:
                continue
            next_remaining = tuple(si for si in remaining if si not in subset)
            tail = assign_groups(next_remaining, order_idx + 1)
            if tail is not None:
                placement.update(tail)
                return placement
        return None

    final_placement = assign_groups(tuple(range(len(sprites))), 0)
    assert final_placement is not None

    out = np.full((dh, dw), int(base_col), dtype=int)
    for si, (rr0, cc0) in final_placement.items():
        sp = sprites[si]
        paste(out, sp["patch"], rr0, cc0, mask=sp["mask"], overwrite=True)
    return out


# --------------- generator ---------------

GRID_MIN, GRID_MAX = 22, 30
THICK_MIN, THICK_MAX = 9, 12
OUTER_GAP = 12
SPRITE_COUNT_MIN, SPRITE_COUNT_MAX = 3, 5
GROUP_COUNT_MIN, GROUP_COUNT_MAX = 2, 3
RECT_LONG_MIN, RECT_LONG_MAX = 3, 12
RECT_CROSS_MIN = 3
HOLE_FRAC_LO, HOLE_FRAC_HI = 0.34, 0.56
GROUP_GAP_MIN, GROUP_GAP_MAX = 1, 3
MIN_BIG_AREA = 12
MIN_MAX_AREA = 20
BICHROMATIC_FRAC = 0.4
MAX_RETRIES = 600


def _rand_partition(total: int, parts: int, low: int, high: int) -> list[int] | None:
    if parts <= 0:
        return None
    if total < parts * low or total > parts * high:
        return None
    vals = [low] * parts
    remaining = total - parts * low
    while remaining > 0:
        choices = [i for i, v in enumerate(vals) if v < high]
        if not choices:
            return None
        i = random.choice(choices)
        add = random.randint(1, min(remaining, high - vals[i]))
        vals[i] += add
        remaining -= add
    random.shuffle(vals)
    return vals


def _choose_group_sizes(total: int, groups: int) -> list[int] | None:
    out: list[list[int]] = []
    for vals in itertools.product((1, 2, 3), repeat=groups):
        if sum(vals) == total and max(vals) >= 2:
            out.append(list(vals))
    return random.choice(out) if out else None


def _choose_group_count(total_sprites: int) -> int:
    opts = [g for g in range(GROUP_COUNT_MIN, GROUP_COUNT_MAX + 1) if g <= total_sprites]
    assert opts
    return random.choice(opts)


def _make_patch(h: int, w: int, cols: list[int]) -> np.ndarray:
    out = np.full((h, w), cols[0], dtype=int)
    if len(cols) == 1:
        return out
    alt = cols[1]
    rr, cc = np.indices((h, w))
    mode = random.choice(["checker", "dots", "split_h", "split_w"])
    if mode == "checker":
        out[(rr + cc + random.randint(0, 1)) % 2 == 0] = alt
    elif mode == "dots":
        step = random.choice([2, 3])
        mask = (rr + 2 * cc + random.randint(0, step - 1)) % step == 0
        if mask.any() and not mask.all():
            out[mask] = alt
        else:
            out[h // 2, w // 2] = alt
    elif mode == "split_h" and h > 1:
        cut = random.randint(1, h - 1)
        out[cut:, :] = alt
    else:
        cut = random.randint(1, w - 1) if w > 1 else 0
        if w > 1:
            out[:, cut:] = alt
        else:
            out[-1:, :] = alt
    return out


def _build_group_rects(start: int, lengths: list[int], thick: int) -> list[tuple[int, int, int, int]] | None:
    rects: list[tuple[int, int, int, int]] = []
    prev_c = prev_w = None
    rr = start
    max_w = thick - 2
    if max_w < RECT_CROSS_MIN:
        return None
    for h in lengths:
        w = random.randint(RECT_CROSS_MIN, max_w)
        c_hi = thick - w - 1
        if c_hi < 1:
            return None
        if prev_c is None:
            c = random.randint(1, c_hi)
        else:
            lo = max(1, prev_c - w + 1)
            hi = min(c_hi, prev_c + prev_w - 1)
            if lo > hi:
                return None
            c = random.randint(lo, hi)
        if h * w < MIN_BIG_AREA:
            return None
        rects.append((rr, c, h, w))
        rr += h
        prev_c, prev_w = c, w
    return rects


def _build_canonical_rects(long: int, thick: int, total_sprites: int, group_count: int) -> list[tuple[int, int, int, int]] | None:
    group_sizes = _choose_group_sizes(total_sprites, group_count)
    if group_sizes is None:
        return None

    gaps = [random.randint(GROUP_GAP_MIN, GROUP_GAP_MAX) for _ in range(group_count - 1)]
    available_long = long - 2 - sum(gaps)
    if available_long < total_sprites * RECT_LONG_MIN:
        return None

    group_lengths: list[list[int]] = []
    remaining_long = available_long
    for pieces in group_sizes:
        min_need_rest = sum(group_sizes[len(group_lengths) + 1 :]) * RECT_LONG_MIN
        hi = min(RECT_LONG_MAX * pieces, remaining_long - min_need_rest)
        lo = RECT_LONG_MIN * pieces
        if lo > hi:
            return None
        span = random.randint(lo, hi)
        part = _rand_partition(span, pieces, RECT_LONG_MIN, RECT_LONG_MAX)
        if part is None:
            return None
        group_lengths.append(part)
        remaining_long -= span

    slack = long - sum(sum(part) for part in group_lengths) - sum(gaps)
    if slack < 0:
        return None
    lead = random.randint(1, max(1, slack - 1))
    tail = slack - lead
    starts: list[int] = []
    cursor = lead
    for gi, part in enumerate(group_lengths):
        starts.append(cursor)
        cursor += sum(part)
        if gi < len(gaps):
            cursor += gaps[gi]
    if tail < 1:
        return None

    rects: list[tuple[int, int, int, int]] = []
    for start, part in zip(starts, group_lengths):
        group_rects = _build_group_rects(start, part, thick)
        if group_rects is None:
            return None
        rects.extend(group_rects)

    # Keep every rectangle size unique so the filled output is inferable from geometry alone.
    dims = [(h, w) for _, _, h, w in rects]
    if len(set(dims)) != len(dims):
        return None

    areas = [h * w for _, _, h, w in rects]
    total_area = sum(areas)
    frac = total_area / float(long * thick)
    if not (HOLE_FRAC_LO <= frac <= HOLE_FRAC_HI):
        return None
    if max(areas) < MIN_MAX_AREA:
        return None
    if sum(1 for a in areas if a >= 18) < 2:
        return None
    return rects


def _make_pieces(rects: list[tuple[int, int, int, int]], bg: int, base: int) -> dict[int, np.ndarray] | None:
    available = [c for c in range(10) if c not in (bg, base)]
    if len(available) < len(rects):
        return None
    primaries = random.sample(available, len(rects))
    pieces: dict[int, np.ndarray] = {}
    for i, (_, _, h, w) in enumerate(rects):
        cols = [primaries[i]]
        if random.random() < BICHROMATIC_FRAC and len(available) >= 2:
            second_pool = [c for c in available if c != primaries[i]]
            cols.append(random.choice(second_pool))
        pieces[i] = _make_patch(h, w, cols)
    return pieces


def _render_output(dh: int, dw: int, base: int, rects: list[tuple[int, int, int, int]], pieces: dict[int, np.ndarray]) -> np.ndarray:
    out = np.full((dh, dw), base, dtype=int)
    for i, (r, c, _, _) in enumerate(rects):
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
) -> np.ndarray:
    inp = np.full((H, W), bg, dtype=int)
    inp[tr0 : tr0 + dh, tc0 : tc0 + dw] = base
    for r, c, h, w in rects:
        inp[tr0 + r : tr0 + r + h, tc0 + c : tc0 + c + w] = bg

    placement = np.full((H, W), bg, dtype=int)
    placement[tr0 : tr0 + dh, tc0 : tc0 + dw] = base
    order = list(range(len(rects)))
    random.shuffle(order)
    order.sort(key=lambda i: pieces[i].size, reverse=True)
    for i in order:
        patch = pieces[i]
        rr, cc = random_free_location(placement, patch, transparent=bg, padding=1, padding_connectivity=8)
        paste(inp, patch, rr, cc)
        paste(placement, patch, rr, cc)
    return inp


def _swap_orientation(rects: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    return [(c, r, w, h) for r, c, h, w in rects]


def _gen_attempt() -> np.ndarray | None:
    vertical = random.random() < 0.5
    thick = random.randint(THICK_MIN, THICK_MAX)
    total = random.randint(SPRITE_COUNT_MIN, SPRITE_COUNT_MAX)
    group_count = _choose_group_count(total)

    if vertical:
        H = random.randint(GRID_MIN, GRID_MAX)
        W = random.randint(max(GRID_MIN, thick + OUTER_GAP), GRID_MAX)
        dh, dw = H, thick
        tr0, tc0 = 0, 0 if random.random() < 0.5 else W - thick
        canon = _build_canonical_rects(H, thick, total, group_count)
        rects = canon
    else:
        H = random.randint(max(GRID_MIN, thick + OUTER_GAP), GRID_MAX)
        W = random.randint(GRID_MIN, GRID_MAX)
        dh, dw = thick, W
        tr0, tc0 = (0 if random.random() < 0.5 else H - thick), 0
        canon = _build_canonical_rects(W, thick, total, group_count)
        rects = None if canon is None else _swap_orientation(canon)

    if rects is None:
        return None

    bg, base = random.sample(range(10), 2)
    pieces = _make_pieces(rects, bg, base)
    if pieces is None:
        return None

    out = _render_output(dh, dw, base, rects, pieces)
    hole_mask = np.full((dh, dw), False, dtype=bool)
    for r, c, h, w in rects:
        hole_mask[r : r + h, c : c + w] = True
    groups = components_mask(hole_mask, connectivity=4)
    if not (GROUP_COUNT_MIN <= len(groups) <= GROUP_COUNT_MAX):
        return None

    base_area = dh * dw - int(hole_mask.sum())
    if base_area <= max(int(piece.size) for piece in pieces.values()):
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
    raise RuntimeError("failed to generate valid fused 5dbc8537 input")
