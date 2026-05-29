# Combo variant of 5dbc8537 (index 9)
# combo_spec: D1/3+D14/0
# Name: 面积排序分层递进填充

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path(__file__).resolve().parents[4] / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, colors_present, components_mask, crop_bbox, find_components, get_bg, paste, random_free_location


# Design analysis for the fused variant:
# - Original 5dbc8537 relied on a hidden invariant: every cavity rectangle had a unique bbox, so direct size
#   matching was sufficient. D1/3 breaks that. Here the first-layer cavities deliberately share one outer bbox,
#   so bbox equality no longer identifies which large sprite goes where.
# - Concrete miniature example: three 4x5 cavities appear top->bottom. The scattered monochrome frame sprites
#   have occupied areas 18, 17, 16 because they contain 1x2, 1x3, 2x2 windows. The solver must place the
#   larger occupied area first into the first cavity, then the next, etc. After that step the visible residual
#   holes are 1x2, 1x3, 2x2 in the same spatial order; the second layer then fills those with small bichrome
#   rectangles sorted by small area.
# - This keeps the multi-step effect visually legible: layer 1 is always monochrome and large, layer 2 is
#   always bichrome and small. Human solvers can separate the layers by appearance instead of by raw color ids.
# - Generator legality is enforced by construction: large holes are disjoint rectangles, each frame sprite fits
#   every large hole because the outer bbox is shared, each frame leaves exactly one inner rectangular window,
#   and small solid sprites match those windows. The outer scattering stage blocks full bboxes, so sprites never
#   overlap or nest inside one another.
# - New solver assumptions are explicit and generator-backed:
#   1. The template is still the largest non-bg color component.
#   2. Monochrome outside components are layer 1; bichrome components are layer 2.
#   3. Initial holes are rectangular stage-1 slots; remaining holes after stage 1 are rectangular stage-2 slots.
#   4. Stage 1 uses area-descending assignment along the strip direction; stage 2 uses area-ascending assignment
#      along the same strip direction. This asymmetry is intentional because every stage-1 frame and its stage-2
#      window are complementary inside a shared outer bbox.


def _axis_for_strip(shape: tuple[int, int]) -> int:
    dh, dw = map(int, shape)
    return 0 if dh >= dw else 1


def _slot_key(item: tuple[int, int, int, int], axis: int) -> tuple[int, int]:
    r, c, _h, _w = item
    return (int(r), int(c)) if axis == 0 else (int(c), int(r))


def _is_full_rectangle(mask: np.ndarray) -> bool:
    return bool(mask.size) and bool(mask.all())


def _sprite_info(grid: np.ndarray, comp: dict) -> dict[str, object]:
    pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
    patch = crop_bbox(grid, (pr0, pc0, pr1, pc1))
    mask = np.zeros_like(patch, dtype=bool)
    coords = comp["coords"]
    mask[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
    cols = sorted({int(v) for v in patch[mask]})
    return {
        "bbox": (pr0, pc0, pr1, pc1),
        "patch": patch,
        "mask": mask,
        "h": int(patch.shape[0]),
        "w": int(patch.shape[1]),
        "bbox_area": int(patch.shape[0] * patch.shape[1]),
        "area": int(mask.sum()),
        "colors": cols,
    }


def _rect_components(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    rects: list[tuple[int, int, int, int]] = []
    for coords in components_mask(mask, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(coords)
        piece = mask[r0 : r1 + 1, c0 : c1 + 1]
        assert _is_full_rectangle(piece)
        rects.append((int(r0), int(c0), int(r1 - r0 + 1), int(c1 - c0 + 1)))
    return rects


def _rank_large(sprite: dict[str, object]) -> tuple[int, int, int]:
    return (-int(sprite["area"]), -int(sprite["bbox_area"]), -int(sprite["h"]))


def _rank_small(sprite: dict[str, object]) -> tuple[int, int, int]:
    return (int(sprite["area"]), int(sprite["h"]), int(sprite["w"]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a strip template in two ordered passes: large monochrome frame-sprites first, then small bichrome fillers.

    Input structure:
    - The background is the most common color. One dominant base color forms a full-width or full-height strip.
    - Inside that strip, the visible holes at input time are only the coarse first-layer rectangular slots.
    - Outside the strip are two sprite families: large monochrome connected frames and small bichrome solid rectangles.

    Rule:
    - Detect the strip from the largest non-background color component and crop to its bbox.
    - Sort the initial large rectangular holes by strip direction. Sort large monochrome sprites by occupied area
      from largest to smallest and place them into those holes in that spatial order.
    - The first pass exposes a new set of smaller rectangular holes. Sort those new holes by the same strip order.
      Sort the small bichrome sprites by area from smallest to largest and place them into the new holes.

    Key insight:
    - The first-layer holes are intentionally indistinguishable by bbox, so direct shape matching is insufficient.
      The only stable cue is the layer split (monochrome vs bichrome) plus the area-order rule, and the first pass
      must happen before the second because it literally creates the second-layer holes.
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
    assert base_coords.size
    r0, c0, r1, c1 = bbox_coords(base_coords)
    dst = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))
    dh, dw = map(int, dst.shape)
    axis = _axis_for_strip((dh, dw))

    masked = g.copy()
    masked[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)
    comps = find_components(masked, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    sprites = [_sprite_info(g, comp) for comp in comps]

    large = [sp for sp in sprites if len(sp["colors"]) == 1]
    small = [sp for sp in sprites if len(sp["colors"]) >= 2]
    assert large and small

    work = dst.copy()
    stage1_slots = sorted(_rect_components(work == int(bg)), key=lambda rect: _slot_key(rect, axis))
    assert len(stage1_slots) == len(large)
    large_sorted = sorted(large, key=_rank_large)

    for rect, sp in zip(stage1_slots, large_sorted):
        rr0, cc0, h, w = rect
        assert (int(sp["h"]), int(sp["w"])) == (int(h), int(w))
        paste(work, sp["patch"], int(rr0), int(cc0), mask=sp["mask"], overwrite=True)

    stage2_slots = sorted(_rect_components(work == int(bg)), key=lambda rect: _slot_key(rect, axis))
    assert len(stage2_slots) == len(small)
    small_sorted = sorted(small, key=_rank_small)

    for rect, sp in zip(stage2_slots, small_sorted):
        rr0, cc0, h, w = rect
        assert (int(sp["h"]), int(sp["w"])) == (int(h), int(w))
        paste(work, sp["patch"], int(rr0), int(cc0), mask=sp["mask"], overwrite=True)

    assert not np.any(work == int(bg))
    return work


# --------------- generator ---------------

GRID_MIN, GRID_MAX = 17, 25
THICK_MIN, THICK_MAX = 7, 10
OUTER_GAP = 8
MAX_RETRIES = 500
BASELINE_GAP = 1
BIG_H_MIN, BIG_H_MAX = 4, 6
BIG_W_MIN, BIG_W_MAX = 4, 6
LARGE_SLOT_COUNTS = (3, 4)


def _rand_composition(total: int, parts: int) -> list[int]:
    if parts <= 0:
        return []
    if total <= 0:
        return [0] * parts
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    vals: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        vals.append(int(cut - prev - 1))
        prev = cut
    return vals


def _small_rect_choices(big_h: int, big_w: int) -> list[tuple[int, int]]:
    best_by_area: dict[int, tuple[int, int]] = {}
    for h in range(1, int(big_h - 1)):
        for w in range(1, int(big_w - 1)):
            area = int(h * w)
            if area < 2:
                continue
            cand = (int(h), int(w))
            prev = best_by_area.get(area)
            if prev is None or cand[0] + cand[1] < prev[0] + prev[1]:
                best_by_area[area] = cand
    return [best_by_area[a] for a in sorted(best_by_area)]


def _choose_layout(long_len: int, thick: int) -> list[tuple[int, int, int, int]] | None:
    feasible_by_count: dict[int, list[tuple[int, int, list[tuple[int, int]]]]] = {}
    for count in LARGE_SLOT_COUNTS:
        for big_h in range(BIG_H_MIN, BIG_H_MAX + 1):
            min_used = int(count * big_h + (count + 1) * BASELINE_GAP)
            if min_used > int(long_len):
                continue
            max_big_w = min(BIG_W_MAX, int(thick - 2))
            for big_w in range(BIG_W_MIN, max_big_w + 1):
                if int(thick - big_w) < 2:
                    continue
                choices = _small_rect_choices(big_h, big_w)
                if len(choices) >= int(count):
                    feasible_by_count.setdefault(int(count), []).append((int(big_h), int(big_w), choices))
    if not feasible_by_count:
        return None

    count = random.choice(sorted(feasible_by_count))
    big_h, big_w, _choices = random.choice(feasible_by_count[count])
    slack = int(long_len - (count * big_h + (count + 1) * BASELINE_GAP))
    gaps = _rand_composition(slack, count + 1)
    cursor = int(BASELINE_GAP + gaps[0])
    cross = random.randint(1, int(thick - big_w - 1))
    rects: list[tuple[int, int, int, int]] = []
    for i in range(count):
        rects.append((int(cursor), int(cross), int(big_h), int(big_w)))
        cursor += int(big_h + BASELINE_GAP + gaps[i + 1])
    return rects


def _make_frame_patch(big_h: int, big_w: int, color: int, inner_h: int, inner_w: int, bg: int) -> np.ndarray:
    patch = np.full((int(big_h), int(big_w)), int(color), dtype=int)
    patch[1 : 1 + int(inner_h), 1 : 1 + int(inner_w)] = int(bg)
    return patch


def _make_bichrome_patch(h: int, w: int, primary: int, accent: int) -> np.ndarray:
    patch = np.full((int(h), int(w)), int(primary), dtype=int)
    rr, cc = np.indices((int(h), int(w)))
    patch[(rr + cc) % 2 == 0] = int(accent)
    if len(set(map(int, patch.flatten()))) == 1:
        patch[0, 0] = int(accent)
        patch[-1, -1] = int(primary)
    return patch


def _order_slots(rects: list[tuple[int, int, int, int]], axis: int) -> list[tuple[int, int, int, int]]:
    return sorted(rects, key=lambda rect: _slot_key(rect, axis))


def _slot_bbox_mask(rects: list[tuple[int, int, int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c, h, w in rects:
        mask[int(r) : int(r) + int(h), int(c) : int(c) + int(w)] = True
    return mask


def _scatter_all(
    H: int,
    W: int,
    bg: int,
    base: int,
    tr0: int,
    tc0: int,
    template: np.ndarray,
    sprites: list[np.ndarray],
) -> np.ndarray:
    inp = np.full((int(H), int(W)), int(bg), dtype=int)
    paste(inp, template, int(tr0), int(tc0), overwrite=True)

    placement = np.full((int(H), int(W)), int(bg), dtype=int)
    placement[int(tr0) : int(tr0) + template.shape[0], int(tc0) : int(tc0) + template.shape[1]] = int(base)

    order = list(range(len(sprites)))
    random.shuffle(order)
    order.sort(key=lambda i: sprites[i].shape[0] * sprites[i].shape[1], reverse=True)
    for i in order:
        patch = sprites[i]
        blocker = np.full(patch.shape, int(base), dtype=int)
        rr, cc = random_free_location(placement, blocker, transparent=bg, padding=1, padding_connectivity=8)
        paste(inp, patch, int(rr), int(cc), overwrite=True)
        paste(placement, blocker, int(rr), int(cc), overwrite=True)
    return inp


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
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

    axis = _axis_for_strip((dh, dw))
    long_len = int(dh if axis == 0 else dw)
    cross_len = int(dw if axis == 0 else dh)
    rects_axis = _choose_layout(long_len, cross_len)
    if rects_axis is None:
        return None

    if axis == 0:
        slots = rects_axis
    else:
        slots = [(int(c), int(r), int(w), int(h)) for r, c, h, w in rects_axis]
    slots_sorted = _order_slots(slots, axis)
    count = len(slots_sorted)

    big_h = int(slots_sorted[0][2])
    big_w = int(slots_sorted[0][3])
    small_choices = _small_rect_choices(big_h, big_w)
    if len(small_choices) < count:
        return None
    inner_dims = random.sample(small_choices, count)
    inner_dims.sort(key=lambda hw: hw[0] * hw[1])

    bg, base = random.sample(range(10), 2)
    palette = [c for c in range(10) if c not in (bg, base)]
    if len(palette) < count + 1:
        return None
    big_colors = random.sample(palette, count)
    accent = random.choice([c for c in palette if c not in big_colors] or palette)
    small_primary = random.sample([c for c in palette if c != accent], count)

    large_patches: list[np.ndarray] = []
    small_patches: list[np.ndarray] = []
    for i, (ih, iw) in enumerate(inner_dims):
        large_patches.append(_make_frame_patch(big_h, big_w, big_colors[i], ih, iw, bg))
        small_patches.append(_make_bichrome_patch(ih, iw, small_primary[i], accent))

    large_sorted = sorted(large_patches, key=lambda p: -int(np.count_nonzero(p != bg)))
    small_sorted = sorted(small_patches, key=lambda p: int(p.shape[0] * p.shape[1]))

    template = np.full((int(dh), int(dw)), int(base), dtype=int)
    for r, c, h, w in slots:
        template[int(r) : int(r) + int(h), int(c) : int(c) + int(w)] = int(bg)

    out = template.copy()
    for (r, c, _h, _w), patch in zip(slots_sorted, large_sorted):
        paste(out, patch, int(r), int(c), transparent=bg)
    for (r, c, _h, _w), patch in zip(slots_sorted, small_sorted):
        paste(out, patch, int(r + 1), int(c + 1), overwrite=True)

    sprites = large_patches + small_patches
    try:
        inp = _scatter_all(H, W, bg, base, tr0, tc0, template, sprites)
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    # Fusion guardrails:
    # - stage 1 must be non-degenerate: at least two large sprites share the same bbox but differ in occupied area.
    # - stage 2 must truly exist: the first pass leaves smaller holes that the second layer fills.
    large_bbox_shapes = {tuple(p.shape) for p in large_patches}
    large_areas = {int(np.count_nonzero(p != bg)) for p in large_patches}
    if len(large_bbox_shapes) != 1 or len(large_areas) != len(large_patches):
        return None
    stage1_only = template.copy()
    for (r, c, _h, _w), patch in zip(slots_sorted, large_sorted):
        paste(stage1_only, patch, int(r), int(c), transparent=bg)
    if int(np.sum(stage1_only == int(bg))) <= 0:
        return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        built = _gen_attempt()
        if built is not None:
            inp, _out = built
            return inp
    raise RuntimeError("failed to generate valid fused 5dbc8537 instance")
