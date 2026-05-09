# Variant of 67e490f4
# Dimension: D3 — 匹配条件：颜色消歧规则 (match_cond)
# Change: frequency_tiebreak — 当多个不同颜色的精灵匹配同一个洞形状时，选择出现频率最高的颜色填充；频率相同时... → 最少频率颜色填充（取反）
#
# Design analysis
# spec:
#   variant_core:
#     Keep the original geometry exactly: one large frame, bg-colored holes inside it,
#     and scattered outside sprites whose shapes match the holes up to rotation/flip.
#     The only semantic change is the color-disambiguation rule after shape matching:
#     choose the least frequent matching sprite color instead of the most frequent one.
#   concretization:
#     A hole shape may have outside sprites in two colors, for example:
#       red L-shape x5, green L-shape x1
#     The original task would fill the matching hole(s) with red.
#     This variant must fill the hole(s) with green.
#     The pixels do not move differently; only the counting decision changes.
#   behavior_trace:
#     example:
#       hole_shape: 2x2 L mask
#       matching_sprite_counts:
#         color_2: 4
#         color_6: 1
#       original_solver: pick color_2 because it is the plurality
#       variant_solver: pick color_6 because it is the unique minority
#       visible_effect: the output color disagrees with the visually dominant outside color
#   complexity_calibration:
#     The generator deliberately creates at least two colors per matched shape and makes
#     the minority unique in count. This prevents degeneration into the original task and
#     avoids relying on invisible tie-breaking by color id for human inference.
#   implicit_assumption_audit:
#     - Frame detection via largest monochrome non-bg component is unchanged and still valid.
#     - Hole extraction via 4-connected non-frame cells inside the frame bbox is unchanged.
#     - Sprite extraction still aggregates counts by shape key across all rotation/flip views.
#     - The old selection line assumed "higher count is better"; that assumption is no longer
#       valid and must be inverted to "lower count is better".
#     - The old generator assumed the output color should be the majority color for each shape;
#       that assumption must be replaced by a minority-color assignment.
#     - Tie-breaking by smaller color id is kept only as a defensive fallback, but generation
#       avoids equal counts so the decisive visible rule is always rarity, not color numbering.
#   placement_legality:
#     Hole placement remains symmetric, in-bounds, border-separated, and 4-disconnected.
#     Sprites remain fully outside the frame bbox and keep Chebyshev padding from each other
#     and from the frame. This preserves clean component detection for the solver.
#   visual_observability:
#     Every canonical hole shape gets a rare color and a more common rival color. The rare
#     color is the one painted into the output. Since the dominant outside color is rejected,
#     the inversion from majority-vote to minority-vote is directly visible across examples.
#   visual_cleanliness:
#     No extra marking layer is added. The scene still has just frame, holes, and sprites,
#     with sparse outside placement and distinct fill colors for different shapes.
#   color_inference:
#     The output color is always present among same-shape outside sprites, so humans can trace
#     shape correspondence first and then notice that the chosen color is the least common one.
#   new_solver_assumptions:
#     main() now assumes only that each hole shape has at least one matching outside sprite.
#     generate_input() guarantees more: every shape has exactly one unique least-frequent color,
#     at least one more common competing color, and enough spacing for stable component parsing.

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the frame interior by using outside sprites as shape-indexed color clues.

    Input structure:
    - The background is the most common color.
    - One large monochrome frame encloses several bg-colored holes of different shapes.
    - Outside that frame sit small sprites whose silhouettes match the hole shapes up to
      rotation or reflection, and the same shape may appear in multiple colors.

    Rule:
    - Crop the result to the frame bounding box and start from a solid fill of the frame color.
    - For each hole, gather every outside sprite with the same shape under rotation/flip.
    - Fill the hole with the least frequent matching sprite color; if counts tie, use the
      smaller color id as a deterministic fallback.

    Key insight:
    - Shape decides which sprites belong to a hole, but frequency decides which color wins:
      here the rare matching color is the answer, not the common one.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Detect the unique large template frame as the largest monochrome non-bg component.
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

    # STEP 2) Read each hole inside the frame bbox as a 4-connected boolean mask.
    hole_mask = sub != int(frame_col)
    holes: list[tuple[tuple[tuple[int, int], bytes], tuple[int, int], np.ndarray]] = []
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        mask = np.zeros((hh, ww), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        holes.append(((mask.shape, mask.tobytes()), (int(hr0), int(hc0)), mask))

    # STEP 3) Collect all outside sprites, aggregating color counts per shape key.
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites_by_key: dict[tuple[tuple[int, int], bytes], dict[int, int]] = {}

    def _all_mask_keys(mask: np.ndarray):
        base = mask.astype(bool)
        for flip in (False, True):
            flipped = np.flipud(base) if flip else base
            for k in range(4):
                oriented = np.rot90(flipped, k)
                yield (oriented.shape, oriented.tobytes())

    for comp in spr_comps:
        coords = comp["coords"]
        sr0, sc0, sr1, sc1 = map(int, comp["bbox"])
        hh = int(sr1 - sr0 + 1)
        ww = int(sc1 - sc0 + 1)

        vals = [int(v) for v in g[coords[:, 0], coords[:, 1]]]
        scol = int(Counter(vals).most_common(1)[0][0])

        rel = coords - np.array([sr0, sc0])
        mask = np.zeros((hh, ww), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True

        for key in _all_mask_keys(mask):
            bucket = sprites_by_key.setdefault(key, {})
            bucket[scol] = int(bucket.get(scol, 0)) + 1

    # STEP 4) Render the cropped output, choosing the least frequent color for each shape.
    out = np.full((th, tw), int(frame_col), dtype=int)
    for key, (hr0, hc0), hmask in holes:
        cmap = sprites_by_key.get(key)
        assert cmap is not None
        scol = min(cmap.items(), key=lambda kv: (kv[1], kv[0]))[0]
        patch = np.full(hmask.shape, int(scol), dtype=int)
        paste(out, patch, int(hr0), int(hc0), mask=hmask)

    return out


# --------------- generator ---------------

NUM_COLORS = 10
GRID_SIZE = 30

# Frame bbox dimensions
FRAME_H_RANGE = (7, 13)
FRAME_W_RANGE = (7, 20)

# Hole layout
SHAPE_COUNT_RANGE = (3, 6)
MIN_SHAPES = 3
SHAPE_MAX_AREA = 8
FRAME_DENSITY_RANGE = (0.55, 0.92)
MIN_TOTAL_HOLES = 5
PLACEMENT_RETRIES = 80
MULT_EVEN_WEIGHTS = ([1, 2], [4, 1])
MULT_ODD_WEIGHTS = ([1, 2, 3], [3, 2, 1])

# Sprite scattering for the minority-color variant
RARE_SPRITE_RANGE = (1, 2)
DOMINANT_ADVANTAGE_RANGE = (2, 4)
SPRITE_PADDING = 1

# Generation retry budget
MAX_GEN_ATTEMPTS = 400


def _all_rot_flip(mask: np.ndarray):
    """Yield (key, oriented_mask) for all 8 rotation/flip variants of a bool mask."""
    base = mask.astype(bool)
    for flip in (False, True):
        flipped = np.flipud(base) if flip else base
        for k in range(4):
            oriented = np.rot90(flipped, k)
            yield (oriented.shape, oriented.tobytes()), oriented


def _canonical_key(mask: np.ndarray):
    """Smallest key across all rotation/flip variants."""
    return min(key for key, _ in _all_rot_flip(mask))


def _all_orientations(mask: np.ndarray):
    """Deduplicated list of all distinct rotation/flip variants."""
    seen: set[tuple[tuple[int, int], bytes]] = set()
    result: list[np.ndarray] = []
    for key, oriented in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            result.append(oriented)
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
    for shape in raw:
        mask = np.array(shape, dtype=bool)
        key = _canonical_key(mask)
        if key not in seen:
            seen.add(key)
            dedup.append(mask)
    return dedup


SHAPE_LIBRARY = _build_shape_library()


def _mirror_pos(r, c, sh, sw, th, tw):
    """Return the 4-fold symmetric mirror positions (deduplicated)."""
    return list(
        {
            (r, c),
            (r, tw - 1 - c - sw + 1),
            (th - 1 - r - sh + 1, c),
            (th - 1 - r - sh + 1, tw - 1 - c - sw + 1),
        }
    )


def _try_place_quadrant(occ, shape, th, tw):
    """Place shape in the top-left quadrant, mirrored into all quadrants."""
    sh, sw = shape.shape
    r_max, c_max = th // 2 - sh, tw // 2 - sw
    if r_max < 1 or c_max < 1:
        return None
    r, c = random.randint(1, r_max), random.randint(1, c_max)
    return _validate_positions(occ, shape, _mirror_pos(r, c, sh, sw, th, tw), th, tw)


def _try_place_hcenter(occ, shape, th, tw):
    """Place on the horizontal center line and mirror left-right."""
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
    return _validate_positions(occ, shape, list({(r, c), (r, c_m)}), th, tw)


def _try_place_vcenter(occ, shape, th, tw):
    """Place on the vertical center line and mirror up-down."""
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
    return _validate_positions(occ, shape, list({(r, c), (r_m, c)}), th, tw)


def _try_place_center(occ, shape, th, tw):
    """Place exactly at the center. Requires odd height and odd width."""
    if th % 2 == 0 or tw % 2 == 0:
        return None
    sh, sw = shape.shape
    r, c = th // 2 - sh // 2, tw // 2 - sw // 2
    if r < 1 or c < 1 or r + sh > th - 1 or c + sw > tw - 1:
        return None
    return _validate_positions(occ, shape, [(r, c)], th, tw)


def _validate_positions(occ, shape, positions, th, tw):
    """Check all positions: in bounds, no overlap, 4-disconnected from existing holes."""
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
    """Try a random symmetry-preserving placement mode."""
    placers = [_try_place_quadrant]
    if th % 2 == 1:
        placers.append(_try_place_hcenter)
    if tw % 2 == 1:
        placers.append(_try_place_vcenter)
    if th % 2 == 1 and tw % 2 == 1:
        placers.append(_try_place_center)
    return random.choice(placers)(occ, shape, th, tw)


def _carve_hole(occ, shape, positions):
    """Mark hole cells as False in the frame occupancy grid."""
    sh, sw = shape.shape
    for pr, pc in positions:
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    occ[pr + dr, pc + dc] = False


def _build_hole_layout(th, tw):
    """Place symmetric holes. Returns (occupancy_mask, hole_placements) or None."""
    max_sh, max_sw = max(1, (th - 2) // 2), max(1, (tw - 2) // 2)
    available = [
        shape
        for shape in SHAPE_LIBRARY
        if shape.shape[0] <= max_sh and shape.shape[1] <= max_sw and int(np.sum(shape)) <= SHAPE_MAX_AREA
    ]
    if len(available) < MIN_SHAPES:
        return None

    k_target = random.randint(SHAPE_COUNT_RANGE[0], min(SHAPE_COUNT_RANGE[1], len(available)))
    random.shuffle(available)
    even_dims = th % 2 == 0 or tw % 2 == 0
    if even_dims:
        available.sort(key=lambda shape: int(np.sum(shape)))

    occ = np.ones((th, tw), dtype=bool)
    hole_placements = []

    mult_vals, mult_wts = MULT_EVEN_WEIGHTS if even_dims else MULT_ODD_WEIGHTS
    for shape in available:
        if len(hole_placements) >= k_target:
            break
        mult = random.choices(mult_vals, weights=mult_wts)[0]
        placed = []
        for _ in range(mult):
            for _ in range(PLACEMENT_RETRIES):
                positions = _try_place_symmetric(occ, shape, th, tw)
                if positions is not None:
                    _carve_hole(occ, shape, positions)
                    placed.append(positions)
                    break
        if placed:
            hole_placements.append((shape, placed))

    if len(hole_placements) < MIN_SHAPES:
        return None

    frame_density = 1.0 - int(np.sum(~occ)) / (th * tw)
    if not (FRAME_DENSITY_RANGE[0] <= frame_density <= FRAME_DENSITY_RANGE[1]):
        return None

    total_holes = sum(len(pos_group) for _, placed_groups in hole_placements for pos_group in placed_groups)
    if total_holes < MIN_TOTAL_HOLES:
        return None

    return occ, hole_placements


def _build_output(occ, hole_placements, frame_col, bg):
    """
    Render the target output and assign one rare fill color per canonical hole shape.

    Returns:
      (output, shape_info)
    where shape_info is [(shape, rare_color, dominant_color, [positions])].
    """
    th, tw = occ.shape
    avail_colors = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    if len(avail_colors) < len(hole_placements):
        return None

    rare_colors = random.sample(avail_colors, len(hole_placements))
    output = np.full((th, tw), frame_col, dtype=int)
    shape_info = []
    for (shape, placed_groups), rare_color in zip(hole_placements, rare_colors):
        dominant_choices = [c for c in avail_colors if c != rare_color]
        if not dominant_choices:
            return None
        dominant_color = random.choice(dominant_choices)
        all_pos = []
        patch = np.full(shape.shape, int(rare_color), dtype=int)
        for positions in placed_groups:
            for pr, pc in positions:
                paste(output, patch, pr, pc, mask=shape)
                all_pos.append((pr, pc))
        shape_info.append((shape, rare_color, dominant_color, all_pos))
    return output, shape_info


def _place_sprite(inp, placement, ori, color, bg):
    """Place one sprite on the input canvas and occupancy tracker."""
    patch = np.full(ori.shape, bg, dtype=int)
    patch[ori] = color
    try:
        r, c = random_free_location(placement, patch, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, patch, r, c, transparent=bg)
    paste(inp, patch, r, c, transparent=bg)
    return True


def _scatter_sprites(inp, shape_info, bg, frame_col):
    """
    Scatter outside sprites so each shape has a unique least-frequent color.

    Returns:
      stats list on success, or None on failure.
    """
    placement = inp.copy()
    stats = []

    for shape, rare_color, dominant_color, _ in shape_info:
        orientations = _all_orientations(shape)
        n_rare = random.randint(*RARE_SPRITE_RANGE)
        n_dominant = n_rare + random.randint(*DOMINANT_ADVANTAGE_RANGE)

        for _ in range(n_dominant):
            if not _place_sprite(inp, placement, random.choice(orientations), dominant_color, bg):
                return None
        for _ in range(n_rare):
            if not _place_sprite(inp, placement, random.choice(orientations), rare_color, bg):
                return None

        stats.append(
            {
                "shape_key": _canonical_key(shape),
                "rare_color": int(rare_color),
                "dominant_color": int(dominant_color),
                "rare_count": int(n_rare),
                "dominant_count": int(n_dominant),
            }
        )

    return stats


def _gen_attempt() -> np.ndarray | None:
    """Single output-first generation attempt, validated by the solver."""
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
    fc = random.randint(0, GRID_SIZE - tw)
    inp = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    paste(inp, np.where(occ, frame_col, bg).astype(int), fr, fc)

    stats = _scatter_sprites(inp, shape_info, bg, frame_col)
    if stats is None:
        return None

    # Defensive check: every shape should genuinely invert the original majority rule.
    if not all(stat["rare_color"] != stat["dominant_color"] and stat["rare_count"] < stat["dominant_count"] for stat in stats):
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
