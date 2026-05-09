# Combo variant of 67e490f4 (index 11)
# combo_spec: D8/0+D13/1
# Name: 自由布局+固定精灵区

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import (
    bbox_coords,
    components_mask,
    crop_bbox,
    find_components,
    get_bg,
    paste,
    random_free_location,
)

# Design analysis for combo D8/0 + D13/1
# a) Concrete behavior and trace:
#    - Original 67e490f4 had two strong regularities: hole placement inside the frame was mirrored, and
#      outside sprites could be scattered anywhere around the frame.
#    - In this combo, the semantic rule stays "shape-coded outside sprites determine hole fill colors", but
#      the two spatial regularities change together:
#      1. D8/0 removes LR/UD/180 symmetry from the hole layout, so an L-hole near the upper-left no longer
#         implies partner holes elsewhere.
#      2. D13/1 constrains every sprite to one structured side band of the grid: either the left strip, the
#         right strip, the top strip, or the bottom strip.
#    - Example trace: if the frame sits in the center-right and contains a 3-cell L-hole plus a 1x2 domino
#      hole, then the input may place all clue sprites only in the left strip. The solver still ignores
#      positions semantically and matches each hole shape against every outside sprite up to D8 transforms,
#      but a human now sees a strong contrast: the frame interior looks irregular while the clues live in a
#      tidy, fixed zone.
# b) Complexity calibration:
#    - The hole family count remains 3-6 and total holes remain 6-13. This is enough repetition for humans
#      to infer shape families, while keeping the free layout visually nontrivial.
#    - Winning sprite colors are a strict plurality for each shape family, avoiding hidden dependence on the
#      solver's tie-break.
#    - The sprite zone is restricted to one side band rather than tiny corner boxes, so there is enough area
#      to host all clue sprites without forcing clutter or accidental contacts.
# c) Implicit-assumption audit of main():
#    - `find_components(... monochromatic=True)` assumes the frame is the largest monochrome non-bg component.
#      The generator keeps the frame much larger than any sprite and leaves at least one bg row/col between
#      the frame and the sprite band.
#    - `hole_mask = sub != frame_col` assumes the frame bbox contains only frame cells and holes, not clue
#      sprites. The generator keeps the whole sprite band outside the frame bbox.
#    - Hole keys use exact 4-connected masks, while sprites are indexed under all D8 transforms. Therefore
#      the generator may rotate/flip instances freely, but it must ensure every canonical hole family has at
#      least one corresponding sprite family in the fixed band.
#    - The solver does not detect the side band explicitly, so the generator must make D13/1 visible through
#      placement alone; otherwise the combo would silently collapse back to "sprites can be anywhere".
# d) Placement legality:
#    - Holes stay one cell away from the frame border, never overlap, and remain 4-disconnected.
#    - The frame is placed entirely outside the chosen sprite band with a guaranteed bg gap.
#    - Sprites are packed only inside the chosen band using `random_free_location` on a local region canvas,
#      so they remain non-overlapping, in-bounds, and separated by Chebyshev padding 1.
# e) Visual observability:
#    - The free-layout change is enforced by rejecting hole masks that are LR-, UD-, or 180-symmetric.
#    - The fixed-region change is enforced by construction: every clue sprite lies in the same side strip, so
#      2-3 training examples visibly share a "legend area" even though the frame sits elsewhere.
# f) Visual cleanliness:
#    - There are only two major visual zones: one frame zone and one sprite zone. No overlays, markers, or
#      extra distractor systems are introduced, which keeps the puzzle legible.
# g) Change inferability:
#    - Colors are inferable only through repeated shape/color evidence. The band location itself does not
#      encode color; it merely structures where clues appear, which humans can recognize from a few examples.
# h) Quantitative anti-degeneration checks used during validation:
#    - Hole symmetry metric: all generated samples must fail LR, UD, and 180-degree symmetry tests.
#    - Sprite-zone metric: all non-frame components must lie inside a single side band of thickness 9.
# i) New post-change assumptions:
#    - Because the solver still aggregates all outside components indiscriminately, the generator must avoid
#      placing any unrelated non-bg decoration in the sprite band or elsewhere outside the frame.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a perforated frame by reading shape clues from a dedicated outside sprite zone.

    Input structure:
    - The most common color is the background.
    - One large monochrome frame appears somewhere away from a structured side band, and its interior contains
      freely arranged holes that do not follow the original mirror symmetry.
    - All clue sprites lie outside the frame and are concentrated in one fixed strip of the grid rather than
      being scattered around the frame.

    Rule:
    - Identify the largest monochrome component as the frame, crop to its bounding box, and begin with a
      solid rectangle of the frame color.
    - Read each hole inside that bbox as a 4-connected shape, then collect all outside sprites and group them
      by the same shape up to rotation and reflection.
    - Fill every hole with the most frequent color among its matching sprites.

    Key insight:
    - Spatial placement of the clue sprites is now structured but still not semantic; the actual mapping comes
      entirely from shape equivalence between each hole and the sprites in the fixed clue band.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert comps
    frame_area = max(int(comp["area"]) for comp in comps)
    frame_candidates = [comp for comp in comps if int(comp["area"]) == frame_area]
    assert len(frame_candidates) == 1
    frame = frame_candidates[0]
    frame_col = int(frame["color"])
    fb = frame["bbox"]
    r0, c0, r1, c1 = map(int, fb)

    sub = crop_bbox(g, fb)
    th, tw = sub.shape

    hole_mask = sub != frame_col
    holes: list[tuple[tuple[tuple[int, int], bytes], tuple[int, int], np.ndarray]] = []
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        m = np.zeros((hh, ww), dtype=bool)
        m[rel[:, 0], rel[:, 1]] = True
        holes.append(((m.shape, m.tobytes()), (int(hr0), int(hc0)), m))

    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = bg
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites_by_key: dict[tuple[tuple[int, int], bytes], dict[int, int]] = {}

    def _all_mask_keys(mask: np.ndarray):
        m0 = mask.astype(bool)
        for flip in (False, True):
            m1 = np.flipud(m0) if flip else m0
            for k in range(4):
                m2 = np.rot90(m1, k)
                yield (m2.shape, m2.tobytes())

    for comp in spr_comps:
        coords = comp["coords"]
        sr0, sc0, sr1, sc1 = map(int, comp["bbox"])
        hh = int(sr1 - sr0 + 1)
        ww = int(sc1 - sc0 + 1)

        vals = [int(v) for v in g[coords[:, 0], coords[:, 1]]]
        scol = int(Counter(vals).most_common(1)[0][0])

        rel = coords - np.array([sr0, sc0])
        m = np.zeros((hh, ww), dtype=bool)
        m[rel[:, 0], rel[:, 1]] = True

        for key in _all_mask_keys(m):
            bucket = sprites_by_key.setdefault(key, {})
            bucket[scol] = int(bucket.get(scol, 0)) + 1

    out = np.full((th, tw), frame_col, dtype=int)
    for key, (hr0, hc0), hmask in holes:
        cmap = sprites_by_key.get(key)
        assert cmap is not None
        scol = max(cmap.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        patch = np.full(hmask.shape, scol, dtype=int)
        paste(out, patch, hr0, hc0, mask=hmask)

    return out


NUM_COLORS = 10
GRID_SIZE = 30

FRAME_H_RANGE = (7, 13)
FRAME_W_RANGE = (7, 20)

SHAPE_COUNT_RANGE = (3, 6)
MIN_SHAPES = 3
SHAPE_MAX_AREA = 8
MIN_TOTAL_HOLES = 6
MAX_TOTAL_HOLES = 13
TARGET_HOLE_FILL_RANGE = (0.16, 0.34)
FRAME_DENSITY_RANGE = (0.60, 0.86)

WINNING_SPRITE_RANGE = (3, 5)
OTHER_COLOR_MAX = 2
SPRITE_PADDING = 1
SPRITE_REGION_THICKNESS = 9
FRAME_REGION_GAP = 1

MAX_GEN_ATTEMPTS = 400


def _all_rot_flip(mask: np.ndarray):
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes()), m2


def _canonical_key(mask: np.ndarray):
    return min(key for key, _ in _all_rot_flip(mask))


def _all_orientations(mask: np.ndarray):
    seen: set[tuple[tuple[int, int], bytes]] = set()
    result: list[np.ndarray] = []
    for key, m in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            result.append(m)
    return result


def _build_shape_library():
    raw = [
        [[1]],
        [[1, 1]],
        [[1], [1]],
        [[1, 1, 1]],
        [[1, 1], [1, 1]],
        [[1, 1, 1], [1, 1, 1]],
        [[1, 0], [1, 1]],
        [[1, 0], [1, 0], [1, 1]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[1, 1, 1], [0, 1, 0], [0, 1, 0]],
        [[1, 1, 1, 1]],
        [[1, 1], [1, 0]],
        [[1, 1, 1, 1, 1]],
        [[1, 0], [1, 1], [0, 1]],
        [[1, 1, 1], [1, 0, 0]],
        [[1, 1], [0, 1], [0, 1]],
    ]
    seen: set[tuple[tuple[int, int], bytes]] = set()
    dedup: list[np.ndarray] = []
    for s in raw:
        m = np.array(s, dtype=bool)
        k = _canonical_key(m)
        if k not in seen:
            seen.add(k)
            dedup.append(m)
    return dedup


SHAPE_LIBRARY = _build_shape_library()
SPRITE_REGION_SIDES = ("left", "right", "top", "bottom")


def _symmetry_flags(mask: np.ndarray) -> tuple[bool, bool, bool]:
    return (
        bool(np.array_equal(mask, np.fliplr(mask))),
        bool(np.array_equal(mask, np.flipud(mask))),
        bool(np.array_equal(mask, np.rot90(mask, 2))),
    )


def _breaks_all_old_symmetries(mask: np.ndarray) -> bool:
    lr, ud, rot180 = _symmetry_flags(mask)
    return not lr and not ud and not rot180


def _shape_area(shape: np.ndarray) -> int:
    return int(np.sum(shape))


def _can_place(occ: np.ndarray, shape: np.ndarray, pr: int, pc: int) -> bool:
    th, tw = occ.shape
    sh, sw = shape.shape
    if pr < 1 or pc < 1 or pr + sh > th - 1 or pc + sw > tw - 1:
        return False

    cells = {(pr + dr, pc + dc) for dr, dc in np.argwhere(shape)}
    for rr, cc in cells:
        if not occ[rr, cc]:
            return False
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if 0 <= nr < th and 0 <= nc < tw and not occ[nr, nc] and (nr, nc) not in cells:
                return False
    return True


def _list_valid_placements(occ: np.ndarray, shape: np.ndarray):
    th, tw = occ.shape
    valid: list[tuple[np.ndarray, int, int]] = []
    for ori in _all_orientations(shape):
        sh, sw = ori.shape
        for pr in range(1, th - sh):
            for pc in range(1, tw - sw):
                if _can_place(occ, ori, pr, pc):
                    valid.append((ori, pr, pc))
    return valid


def _carve_hole(occ: np.ndarray, shape: np.ndarray, pr: int, pc: int) -> None:
    for dr, dc in np.argwhere(shape):
        occ[pr + dr, pc + dc] = False


def _build_hole_layout(th: int, tw: int):
    max_sh, max_sw = max(1, th - 2), max(1, tw - 2)
    available = [
        s
        for s in SHAPE_LIBRARY
        if s.shape[0] <= max_sh and s.shape[1] <= max_sw and _shape_area(s) <= SHAPE_MAX_AREA
    ]
    if len(available) < MIN_SHAPES:
        return None

    k_target = random.randint(SHAPE_COUNT_RANGE[0], min(SHAPE_COUNT_RANGE[1], len(available)))
    chosen = random.sample(available, k_target)
    chosen.sort(key=lambda s: (-_shape_area(s), -max(s.shape), -min(s.shape)))

    occ = np.ones((th, tw), dtype=bool)
    hole_groups: list[tuple[np.ndarray, list[tuple[np.ndarray, tuple[int, int]]]]] = []
    total_holes = 0
    hole_area = 0

    for shape in chosen:
        valid = _list_valid_placements(occ, shape)
        if not valid:
            return None
        ori, pr, pc = random.choice(valid)
        _carve_hole(occ, ori, pr, pc)
        hole_groups.append((shape, [(ori, (pr, pc))]))
        total_holes += 1
        hole_area += _shape_area(ori)

    bbox_area = th * tw
    low_area = max(hole_area, int(np.ceil(bbox_area * TARGET_HOLE_FILL_RANGE[0])))
    high_area = max(low_area, int(np.floor(bbox_area * TARGET_HOLE_FILL_RANGE[1])))
    target_hole_area = random.randint(low_area, high_area)

    stalled = 0
    while (hole_area < target_hole_area or total_holes < MIN_TOTAL_HOLES) and total_holes < MAX_TOTAL_HOLES:
        idx = random.randrange(len(hole_groups))
        shape, instances = hole_groups[idx]
        valid = _list_valid_placements(occ, shape)
        if not valid:
            stalled += 1
            if stalled >= 12:
                break
            continue
        ori, pr, pc = random.choice(valid)
        _carve_hole(occ, ori, pr, pc)
        instances.append((ori, (pr, pc)))
        total_holes += 1
        hole_area += _shape_area(ori)
        stalled = 0

    while total_holes < MAX_TOTAL_HOLES and hole_area < bbox_area * TARGET_HOLE_FILL_RANGE[1] and random.random() < 0.35:
        idx = random.randrange(len(hole_groups))
        shape, instances = hole_groups[idx]
        valid = _list_valid_placements(occ, shape)
        if not valid:
            break
        ori, pr, pc = random.choice(valid)
        _carve_hole(occ, ori, pr, pc)
        instances.append((ori, (pr, pc)))
        total_holes += 1
        hole_area += _shape_area(ori)

    frame_density = 1.0 - hole_area / bbox_area
    if not (FRAME_DENSITY_RANGE[0] <= frame_density <= FRAME_DENSITY_RANGE[1]):
        return None
    if total_holes < MIN_TOTAL_HOLES:
        return None

    hole_mask = ~occ
    if not _breaks_all_old_symmetries(hole_mask):
        return None

    used_rows = int(np.sum(hole_mask.any(axis=1)))
    used_cols = int(np.sum(hole_mask.any(axis=0)))
    if used_rows < 3 or used_cols < 3:
        return None

    return occ, hole_groups


def _build_output(occ: np.ndarray, hole_groups, frame_col: int, bg: int):
    th, tw = occ.shape
    avail_colors = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    if len(avail_colors) < len(hole_groups):
        return None

    shape_colors = random.sample(avail_colors, len(hole_groups))
    output = np.full((th, tw), frame_col, dtype=int)
    shape_info = []
    for (shape, instances), fill_col in zip(hole_groups, shape_colors):
        for ori, (pr, pc) in instances:
            patch = np.full(ori.shape, fill_col, dtype=int)
            paste(output, patch, pr, pc, mask=ori)
        shape_info.append((shape, fill_col, instances))
    return output, shape_info


def _sprite_region_bbox(side: str) -> tuple[int, int, int, int]:
    t = SPRITE_REGION_THICKNESS
    if side == "left":
        return 0, 0, GRID_SIZE - 1, t - 1
    if side == "right":
        return 0, GRID_SIZE - t, GRID_SIZE - 1, GRID_SIZE - 1
    if side == "top":
        return 0, 0, t - 1, GRID_SIZE - 1
    if side == "bottom":
        return GRID_SIZE - t, 0, GRID_SIZE - 1, GRID_SIZE - 1
    raise ValueError(f"unknown side: {side}")


def _frame_position_ranges(th: int, tw: int, side: str):
    t = SPRITE_REGION_THICKNESS
    gap = FRAME_REGION_GAP
    if side == "left":
        return (0, GRID_SIZE - th), (t + gap, GRID_SIZE - tw)
    if side == "right":
        return (0, GRID_SIZE - th), (0, GRID_SIZE - t - gap - tw)
    if side == "top":
        return (t + gap, GRID_SIZE - th), (0, GRID_SIZE - tw)
    if side == "bottom":
        return (0, GRID_SIZE - t - gap - th), (0, GRID_SIZE - tw)
    raise ValueError(f"unknown side: {side}")


def _place_frame(inp: np.ndarray, occ: np.ndarray, frame_col: int, bg: int, side: str):
    th, tw = occ.shape
    (r_min, r_max), (c_min, c_max) = _frame_position_ranges(th, tw, side)
    if r_max < r_min or c_max < c_min:
        return None
    fr = random.randint(r_min, r_max)
    fc = random.randint(c_min, c_max)
    paste(inp, np.where(occ, frame_col, bg).astype(int), fr, fc)
    return fr, fc, fr + th - 1, fc + tw - 1


def _place_sprite_in_region(
    inp: np.ndarray,
    placement: np.ndarray,
    region_bbox: tuple[int, int, int, int],
    ori: np.ndarray,
    color: int,
    bg: int,
) -> bool:
    patch = np.full(ori.shape, bg, dtype=int)
    patch[ori] = color
    try:
        lr, lc = random_free_location(placement, patch, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, patch, lr, lc, transparent=bg)
    rr0, cc0, _rr1, _cc1 = region_bbox
    paste(inp, patch, rr0 + lr, cc0 + lc, transparent=bg)
    return True


def _scatter_sprites_in_region(
    inp: np.ndarray,
    shape_info,
    bg: int,
    frame_col: int,
    region_bbox: tuple[int, int, int, int],
) -> bool:
    rr0, cc0, rr1, cc1 = region_bbox
    placement = np.full((rr1 - rr0 + 1, cc1 - cc0 + 1), bg, dtype=int)
    other_pool = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    ordered = sorted(shape_info, key=lambda item: -_shape_area(item[0]))

    for shape, fill_color, _instances in ordered:
        orientations = _all_orientations(shape)
        n_winning = random.randint(*WINNING_SPRITE_RANGE)
        losers = [c for c in other_pool if c != fill_color]
        n_other = random.randint(1, min(OTHER_COLOR_MAX, len(losers))) if losers else 0
        oc_list = random.sample(losers, n_other)

        for _ in range(n_winning):
            if not _place_sprite_in_region(inp, placement, region_bbox, random.choice(orientations), fill_color, bg):
                return False

        for oc in oc_list:
            n_oc = random.randint(1, max(1, n_winning - 1))
            for _ in range(n_oc):
                if not _place_sprite_in_region(inp, placement, region_bbox, random.choice(orientations), oc, bg):
                    break
    return True


def _outside_components(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]):
    bg = int(get_bg(grid))
    outside = np.array(grid, dtype=int, copy=True)
    r0, c0, r1, c1 = frame_bbox
    outside[r0 : r1 + 1, c0 : c1 + 1] = bg
    return find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")


def _components_fit_region(comps, region_bbox: tuple[int, int, int, int]) -> bool:
    rr0, cc0, rr1, cc1 = region_bbox
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 < rr0 or c0 < cc0 or r1 > rr1 or c1 > cc1:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, NUM_COLORS - 1)
    frame_col = random.choice([c for c in range(NUM_COLORS) if c != bg])
    th = random.randint(*FRAME_H_RANGE)
    tw = random.randint(*FRAME_W_RANGE)

    layout = _build_hole_layout(th, tw)
    if layout is None:
        return None
    occ, hole_groups = layout

    result = _build_output(occ, hole_groups, frame_col, bg)
    if result is None:
        return None
    output, shape_info = result

    side = random.choice(SPRITE_REGION_SIDES)
    region_bbox = _sprite_region_bbox(side)

    inp = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    frame_bbox = _place_frame(inp, occ, frame_col, bg, side)
    if frame_bbox is None:
        return None

    if not _scatter_sprites_in_region(inp, shape_info, bg, frame_col, region_bbox):
        return None

    comps = _outside_components(inp, frame_bbox)
    fits = [name for name in SPRITE_REGION_SIDES if _components_fit_region(comps, _sprite_region_bbox(name))]
    if not comps or fits != [side]:
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {MAX_GEN_ATTEMPTS} attempts")
