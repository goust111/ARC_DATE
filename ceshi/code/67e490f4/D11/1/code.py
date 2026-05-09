# Variant of 67e490f4
# Dimension: D11 — 精灵数量与洞的对应关系 (meta)
# Change: many:1 — 每种洞形状对应多个精灵（2-13个，多种颜色），通过频率消歧选择填充色... → 1:many — 一个精灵填充多个洞

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

# ---------------------------------------------------------------------------
# Design analysis for the D11 variant
#
# source_task: none
#
# a) Concretization and behavior trace
# - The scene still contains one large monochrome frame and several small outside
#   sprites on a 30x30 background.
# - The frame interior now repeats each hole shape multiple times. Outside the
#   frame, each canonical hole shape appears exactly once as a single sprite.
# - Example:
#   old behavior for one canonical shape S:
#     holes(S) = 4 copies inside frame
#     outside sprites(S) = {red, red, blue}
#     solver counted colors and filled all 4 holes with red.
#   new behavior for the same shape S:
#     holes(S) = 4 copies inside frame
#     outside sprites(S) = {blue}
#     solver simply reads blue once and broadcasts it to all 4 holes.
# - The changed step is therefore the hole-shape -> sprite-color lookup:
#   aggregation by frequency is removed; one observed sprite color is reused for
#   every matching hole instance.
#
# b) Complexity calibration
# - If a shape had only one hole, the 1:many relation would be invisible.
#   Therefore the generator forbids singleton mirror groups and requires each
#   chosen shape to occupy at least two hole instances inside the frame.
# - If there were too many outside sprites, the puzzle would visually resemble
#   the original many:1 task. Therefore we place exactly one outside sprite per
#   matched shape and no same-shape color competitors.
# - Distinct shape count stays at 3-6 so the puzzle is not trivial, while total
#   holes stay >= 8 so repetition is visible to a human in a few examples.
#
# c) Hidden-assumption audit of the original main()
# - Frame detection by largest monochrome non-bg component is still valid.
# - Hole extraction as 4-connected non-frame regions inside the frame bbox is
#   still valid.
# - The original sprite loop built key -> color -> count and assumed multiple
#   same-shape sprites might disagree in color. That assumption is no longer
#   true. Keeping "last write wins" would be unsafe if duplicates slipped in, so
#   the new solver asserts same-shape sprites do not conflict in color.
# - Filling holes one by one is still safe because same-shape holes are
#   disjoint; reusing the same color for each one cannot overwrite unrelated
#   regions.
#
# d) Placement legality
# - Holes are still carved strictly inside the frame border and remain
#   4-disconnected from one another.
# - Outside sprites are scattered with Chebyshev padding >= 1 against the frame
#   bbox and each other, so the sprite components stay separate under the
#   solver's 8-connectivity scan.
# - Each sprite is placed fully outside the frame bbox by painting onto a copy
#   of the input that already contains the frame.
#
# e) Visual observability
# - The observable rule shift is: few outside sprites, many repeated holes.
#   Humans can see that one isolated exemplar color must be copied into several
#   identical cavities.
# - Rotation/flip matching remains allowed. The outside sprite may be an
#   asymmetric orientation of the inside hole shape, so shape matching is still
#   the relevant cue rather than raw bbox size.
#
# f) Visual cleanliness
# - No extra distractor sprites are added, because the D11 change is about the
#   broadcast relation itself. This keeps the grid readable: the frame contains
#   repeated holes, and the exterior contains one clean exemplar per shape.
#
# g) Color inference
# - Colors are not transformed algorithmically and are never sorted by id for
#   semantic decisions. Each shape receives one distinct fill color directly from
#   its unique outside sprite, so the color rule is locally visible.
#
# h) Quantitative variant target
# - For every generated sample:
#   sprite_count_per_shape == 1
#   hole_count_per_shape >= 2
#   total_holes >= 8
# - These constraints make the 1:many broadcast relation structural rather than
#   an occasional accident.
#
# i) New solver assumptions and generator guarantees
# - main() assumes each canonical hole shape has a matching outside sprite.
#   _scatter_sprites places one sprite for every shape_info entry.
# - main() assumes same-shape sprites do not disagree in color.
#   _scatter_sprites emits exactly one such sprite, so the assertion is stable.
# - main() assumes the outside sprite is separable as its own connected
#   component. random_free_location(..., padding=1) guarantees that.
# ---------------------------------------------------------------------------


def _all_rot_flip(mask: np.ndarray):
    """Yield (key, oriented_mask) for all 8 rotation/flip variants of a bool mask."""
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes()), m2


def _canonical_key(mask: np.ndarray):
    """Smallest key across all rotation/flip variants."""
    return min(key for key, _ in _all_rot_flip(mask))


def _all_orientations(mask: np.ndarray):
    """Deduplicated list of all distinct rotation/flip variants."""
    seen, result = set(), []
    for key, m in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            result.append(m)
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint every repeated hole in a frame from the single matching outside sprite.

    Input structure:
    - The most common color is the background.
    - One large single-color frame contains many hole components; the same hole shape can recur several times.
    - Outside the frame, there is exactly one small sprite for each hole-shape family, possibly rotated or flipped.

    Rule:
    - Locate the largest frame and crop to its bounding box, then start with a solid frame-colored rectangle.
    - Canonicalize holes and outside sprites by shape under rotation/flip.
    - For each hole shape, copy the lone matching sprite color into every hole instance of that shape.

    Key insight: The exterior sprite is a broadcast key, not a vote: one exemplar color is shared by all same-shape holes inside the frame.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Find the template frame: largest connected component of any non-bg color (8-connected).
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

    # STEP 2) Extract hole components inside the bbox (anything not equal to frame color), 4-connected.
    hole_mask = (sub != int(frame_col))
    holes: list[tuple[tuple[tuple[int, int], bytes], tuple[int, int], np.ndarray]] = []
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        m = np.zeros((hh, ww), dtype=bool)
        m[rel[:, 0], rel[:, 1]] = True
        holes.append((_canonical_key(m), (int(hr0), int(hc0)), m))

    # STEP 3) Extract the single outside sprite for each shape.
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprite_color_by_key: dict[tuple[tuple[int, int], bytes], int] = {}
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
        key = _canonical_key(m)

        prev = sprite_color_by_key.get(key)
        assert prev is None or prev == scol
        sprite_color_by_key[key] = scol

    # STEP 4) Render output: start from solid frame color, then broadcast each shape's sprite color to all matching holes.
    out = np.full((th, tw), int(frame_col), dtype=int)
    for key, (hr0, hc0), hmask in holes:
        scol = sprite_color_by_key.get(key)
        assert scol is not None
        patch = np.full(hmask.shape, int(scol), dtype=int)
        paste(out, patch, int(hr0), int(hc0), mask=hmask)

    return out


# --------------- generator (D11 1:many) ---------------

NUM_COLORS = 10
GRID_SIZE = 30

# Frame bbox dimensions
FRAME_H_RANGE = (7, 13)
FRAME_W_RANGE = (7, 20)

# Hole layout
SHAPE_COUNT_RANGE = (3, 6)
MIN_SHAPES = 3
SHAPE_MAX_AREA = 8
FRAME_DENSITY_RANGE = (0.55, 0.90)
MIN_TOTAL_HOLES = 8
PLACEMENT_RETRIES = 80
GROUP_COUNT_EVEN_WEIGHTS = ([1, 2], [4, 1])
GROUP_COUNT_ODD_WEIGHTS = ([1, 2, 3], [3, 2, 1])

# Sprite scattering
SPRITE_PADDING = 1

# Generation retry budget
MAX_GEN_ATTEMPTS = 300


def _build_shape_library():
    raw = [
        [[1]], [[1, 1]], [[1], [1]], [[1, 1, 1]], [[1, 1], [1, 1]],
        [[1, 1, 1], [1, 1, 1]], [[1, 0], [1, 1]], [[1, 0], [1, 0], [1, 1]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]], [[1, 1, 1], [0, 1, 0], [0, 1, 0]],
        [[1, 1, 1, 1]], [[1, 1], [1, 0]], [[1, 1, 1, 1, 1]],
        [[1, 0], [1, 1], [0, 1]], [[1, 1, 1], [1, 0, 0]], [[1, 1], [0, 1], [0, 1]],
    ]
    seen, dedup = set(), []
    for s in raw:
        m = np.array(s, dtype=bool)
        k = _canonical_key(m)
        if k not in seen:
            seen.add(k)
            dedup.append(m)
    return dedup


SHAPE_LIBRARY = _build_shape_library()


def _mirror_pos(r, c, sh, sw, th, tw):
    """Return the 4-fold symmetric mirror positions (deduplicated)."""
    return list({
        (r, c),
        (r, tw - 1 - c - sw + 1),
        (th - 1 - r - sh + 1, c),
        (th - 1 - r - sh + 1, tw - 1 - c - sw + 1),
    })


def _try_place_quadrant(occ, shape, th, tw):
    """Place shape in the top-left quadrant, mirrored to all 4 quadrants."""
    sh, sw = shape.shape
    r_max, c_max = th // 2 - sh, tw // 2 - sw
    if r_max < 1 or c_max < 1:
        return None
    r, c = random.randint(1, r_max), random.randint(1, c_max)
    return _validate_positions(occ, shape, _mirror_pos(r, c, sh, sw, th, tw), th, tw)


def _try_place_hcenter(occ, shape, th, tw):
    """Place on the horizontal center band, mirrored left-right into two copies."""
    if th % 2 == 0:
        return None
    sh, sw = shape.shape
    r = th // 2 - sh // 2
    if r < 1 or r + sh > th - 1:
        return None
    c_max = tw // 2 - sw
    if c_max < 1:
        return None
    c = random.randint(1, c_max)
    c_m = tw - 1 - c - sw + 1
    if c_m == c:
        return None
    return _validate_positions(occ, shape, [(r, c), (r, c_m)], th, tw)


def _try_place_vcenter(occ, shape, th, tw):
    """Place on the vertical center band, mirrored up-down into two copies."""
    if tw % 2 == 0:
        return None
    sh, sw = shape.shape
    c = tw // 2 - sw // 2
    if c < 1 or c + sw > tw - 1:
        return None
    r_max = th // 2 - sh
    if r_max < 1:
        return None
    r = random.randint(1, r_max)
    r_m = th - 1 - r - sh + 1
    if r_m == r:
        return None
    return _validate_positions(occ, shape, [(r, c), (r_m, c)], th, tw)


def _validate_positions(occ, shape, positions, th, tw):
    """Check all positions: in bounds, no overlap, and 4-disconnected from existing holes."""
    sh, sw = shape.shape
    new_cells = set()
    for pr, pc in positions:
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    new_cells.add((pr + dr, pc + dc))

    for pr, pc in positions:
        if pr < 1 or pc < 1 or pr + sh > th - 1 or pc + sw > tw - 1:
            return None
        for dr in range(sh):
            for dc in range(sw):
                if not shape[dr, dc]:
                    continue
                rr, cc = pr + dr, pc + dc
                if not occ[rr, cc]:
                    return None
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < th and 0 <= nc < tw and not occ[nr, nc]:
                        if (nr, nc) not in new_cells:
                            return None
    return positions


def _try_place_symmetric(occ, shape, th, tw):
    """Try a random symmetric placement type that produces at least two hole copies."""
    placers = [_try_place_quadrant]
    if th % 2 == 1:
        placers.append(_try_place_hcenter)
    if tw % 2 == 1:
        placers.append(_try_place_vcenter)
    random.shuffle(placers)
    for placer in placers:
        positions = placer(occ, shape, th, tw)
        if positions is not None and len(positions) >= 2:
            return positions
    return None


def _carve_hole(occ, shape, positions):
    """Mark hole cells as unoccupied in the occupancy grid."""
    sh, sw = shape.shape
    for pr, pc in positions:
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    occ[pr + dr, pc + dc] = False


def _build_hole_layout(th, tw):
    """Place repeated holes with LR+UD symmetry. Returns (occ, hole_placements) or None."""
    max_sh, max_sw = max(1, (th - 2) // 2), max(1, (tw - 2) // 2)
    available = [
        s for s in SHAPE_LIBRARY
        if s.shape[0] <= max_sh and s.shape[1] <= max_sw and int(np.sum(s)) <= SHAPE_MAX_AREA
    ]
    if len(available) < MIN_SHAPES:
        return None

    k_target = random.randint(SHAPE_COUNT_RANGE[0], min(SHAPE_COUNT_RANGE[1], len(available)))
    random.shuffle(available)
    even_dims = (th % 2 == 0 or tw % 2 == 0)
    if even_dims:
        available.sort(key=lambda s: int(np.sum(s)))

    occ = np.ones((th, tw), dtype=bool)
    hole_placements = []  # list of (shape, placed_groups)
    group_vals, group_wts = GROUP_COUNT_EVEN_WEIGHTS if even_dims else GROUP_COUNT_ODD_WEIGHTS

    for shape in available:
        if len(hole_placements) >= k_target:
            break
        group_count = random.choices(group_vals, weights=group_wts)[0]
        placed = []
        total_instances = 0
        for _ in range(group_count):
            for _ in range(PLACEMENT_RETRIES):
                positions = _try_place_symmetric(occ, shape, th, tw)
                if positions is not None:
                    _carve_hole(occ, shape, positions)
                    placed.append(positions)
                    total_instances += len(positions)
                    break
        if total_instances >= 2:
            hole_placements.append((shape, placed))

    if len(hole_placements) < MIN_SHAPES:
        return None

    frame_density = 1.0 - int(np.sum(~occ)) / (th * tw)
    if not (FRAME_DENSITY_RANGE[0] <= frame_density <= FRAME_DENSITY_RANGE[1]):
        return None

    total_holes = sum(len(group) for _, groups in hole_placements for group in groups)
    if total_holes < MIN_TOTAL_HOLES:
        return None

    if any(sum(len(group) for group in groups) < 2 for _, groups in hole_placements):
        return None

    return occ, hole_placements


def _build_output(occ, hole_placements, frame_col, bg):
    """Render the target output canvas and assign one fill color per repeated shape."""
    th, tw = occ.shape
    avail_colors = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    if len(avail_colors) < len(hole_placements):
        return None
    shape_colors = random.sample(avail_colors, len(hole_placements))

    output = np.full((th, tw), frame_col, dtype=int)
    shape_info = []
    for (shape, groups), fill_col in zip(hole_placements, shape_colors):
        all_pos = []
        for positions in groups:
            for pr, pc in positions:
                paste(output, np.full(shape.shape, fill_col, dtype=int), pr, pc, mask=shape)
                all_pos.append((pr, pc))
        shape_info.append((shape, fill_col, all_pos))
    return output, shape_info


def _place_sprite(inp, placement, ori, color, bg):
    """Place a single sprite on inp and placement grids. Returns True on success."""
    patch = np.full(ori.shape, bg, dtype=int)
    patch[ori] = color
    try:
        r, c = random_free_location(placement, patch, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, patch, r, c, transparent=bg)
    paste(inp, patch, r, c, transparent=bg)
    return True


def _scatter_sprites(inp, shape_info, bg):
    """Place exactly one outside sprite for each hole shape."""
    placement = inp.copy()
    for shape, fill_color, _ in shape_info:
        orientations = _all_orientations(shape)
        if not _place_sprite(inp, placement, random.choice(orientations), fill_color, bg):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build repeated holes, place one sprite per shape, validate."""
    bg = random.randint(0, NUM_COLORS - 1)
    frame_col = random.choice([c for c in range(NUM_COLORS) if c != bg])
    th = random.randint(*FRAME_H_RANGE)
    tw = random.randint(*FRAME_W_RANGE)

    layout = _build_hole_layout(th, tw)
    if layout is None:
        return None
    occ, hole_placements = layout

    result = _build_output(occ, hole_placements, frame_col, bg)
    if result is None:
        return None
    output, shape_info = result

    fr = random.randint(0, GRID_SIZE - th)
    fc_pos = random.randint(0, GRID_SIZE - tw)
    inp = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    paste(inp, np.where(occ, frame_col, bg).astype(int), fr, fc_pos)

    if not _scatter_sprites(inp, shape_info, bg):
        return None

    # Oracle check plus structural D11 check.
    try:
        if not all(len(positions) >= 2 for _, _, positions in shape_info):
            return None
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
