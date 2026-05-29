# Combo variant of 67e490f4 (index 3)
# combo_spec: D4/0+D9/0
# Name: 多色精灵图案搬运

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
# Design analysis for the D4/0 + D9/0 combo
#
# a) Concrete behavior:
#    The large destination object is still a monochrome frame with bg-colored
#    holes inside it, and the outside objects are still shape clues. The combo
#    changes two linked pieces of the original rule:
#    1) source sprites are no longer single-color components; each sprite uses
#       2-3 foreground colors inside one connected silhouette,
#    2) the fill action is no longer "pick one color and flood the hole";
#       instead the matched sprite is a literal stamp and its whole colored
#       pattern is copied into the hole after orientation alignment.
#
#    Small trace:
#    - Original: three matching L sprites with colors red / red / blue make
#      every L hole become a solid red patch.
#    - Combo: three matching L sprites with full patterns A / A / B make every
#      L hole receive pattern A itself, including all internal colors.
#      Rotation or flip outside the frame only changes how the solver has to
#      align the stamp, not which pattern wins.
#
# b) Complexity calibration:
#    If the generator allowed monochrome "winning" sprites, the task would
#    collapse back toward the original color-key behavior. To keep the visual
#    change obvious without making matching chaotic, each hole shape gets:
#    - one winning multicolor pattern,
#    - 1-2 losing multicolor patterns of the same silhouette,
#    - repeated sprite copies so the winning full pattern has a strict count
#      advantage.
#    Shape count stays modest and every rule-relevant pattern repeats.
#
# c) Hidden-assumption review of the original main():
#    - Frame detection as the largest monochrome non-bg component still works.
#      Multicolor sprites split into smaller monochrome pieces under that scan,
#      while the frame remains the only huge monochrome object.
#    - Hole extraction by "sub != frame_col" still works because frame holes in
#      the input are still empty bg gaps.
#    - The original sprite summarization "take most common color per component"
#      is invalid: a multicolor sprite cannot be reduced to one scalar without
#      destroying the new rule. The solver must preserve full colored patches.
#    - The original output step built np.full(..., color), which would erase
#      the intended 2-3 color motif. The new output must paste the aligned
#      winning patch itself.
#    - Orientation ambiguity becomes more dangerous once full patterns matter.
#      The generator therefore only uses silhouettes whose 8 D4 transforms are
#      all distinct, so a sprite orientation maps to exactly one aligned patch
#      for a given hole mask.
#
# d) Placement legality:
#    Holes remain inside the frame border, pairwise 4-disconnected, and fully
#    contained in the frame bbox. Outside sprites are scattered with Chebyshev
#    padding on pure bg so multicolor components never merge with one another
#    or with the frame.
#
# e) Visual observability:
#    A human can infer the rule from a few examples because the same silhouette
#    appears several times outside the frame with one repeated internal motif,
#    and the corresponding holes in the output visibly inherit that motif. The
#    change is directly visible at pixel level; no hidden ordering on color ids
#    is needed.
#
# f) Visual cleanliness:
#    The scene still has only two layers: one frame region and separate outside
#    sprites. Inside the cropped output, holes show compact 2-3 color motifs
#    rather than a clutter of overlapping annotations.
#
# g) Change inferability / anti-degeneration:
#    The generator forces every winning pattern to use 2 or 3 colors, and the
#    copied hole contents therefore differ from any solid-fill interpretation.
#    On the same input, the original solver and this combo solver should almost
#    always disagree, which provides a measurable non-degeneration check.
#
# h) New main() assumptions and generator guarantees:
#    - Assumption: every hole shape has at least one outside sprite match.
#      Guarantee: each chosen shape scatters several winning copies.
#    - Assumption: for one hole orientation, one sprite contributes one aligned
#      patch signature.
#      Guarantee: only orientation-unique silhouettes are used.
#    - Assumption: one full pattern is the strict plurality winner per shape.
#      Guarantee: winning copy count exceeds each losing pattern count.
#    - Assumption: outside sprites are genuinely multicolor.
#      Guarantee: every scattered sprite patch uses 2 or 3 foreground colors.
# ---------------------------------------------------------------------------


def _all_rot_flip_masks(mask: np.ndarray):
    """Yield (key, oriented_mask) for all 8 dihedral transforms of a bool mask."""
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes()), m2


def _all_rot_flip_patches(patch: np.ndarray):
    """Yield deduplicated transforms of the full colored patch, including bg cells."""
    seen = set()
    p0 = np.array(patch, dtype=int)
    for flip in (False, True):
        p1 = np.flipud(p0) if flip else p0
        for k in range(4):
            p2 = np.rot90(p1, k)
            sig = (p2.shape, p2.tobytes())
            if sig in seen:
                continue
            seen.add(sig)
            yield sig, p2


def _canonical_key(mask: np.ndarray):
    return min(key for key, _ in _all_rot_flip_masks(mask))


def _canonical_patch_signature(patch: np.ndarray):
    return min(sig for sig, _ in _all_rot_flip_patches(patch))


def _orientation_count(mask: np.ndarray) -> int:
    return len({key for key, _ in _all_rot_flip_masks(mask)})


def _foreground_colors(patch: np.ndarray, bg: int) -> list[int]:
    vals = [int(v) for v in patch[patch != int(bg)].tolist()]
    return sorted(set(vals))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy multicolor sprite motifs into matching holes inside a large frame.

    Input structure:
    - bg is the most frequent color. One large monochrome frame object defines the target area.
    - Inside the frame bbox are several bg-colored holes with small silhouettes.
    - Outside the frame are separated 2-3 color sprites whose foreground masks match hole shapes up to rotation/flip.

    Rule:
    - Crop to the frame's bounding box and start from a solid rectangle of the frame color.
    - For each hole silhouette, collect all outside sprites with the same foreground mask under dihedral transforms.
    - Find the most frequent full multicolor sprite pattern for that shape, align it to the hole orientation, and paste the whole pattern into the hole.

    Key insight: The outside objects are not color votes anymore; they are reusable stamps. Shape chooses the destination hole, and repeated internal pixel layouts determine exactly which multicolor motif gets copied.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Detect the template frame as the unique largest monochrome non-bg component.
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert comps
    frame_area = max(int(comp["area"]) for comp in comps)
    frame_candidates = [comp for comp in comps if int(comp["area"]) == frame_area]
    assert len(frame_candidates) == 1
    frame = frame_candidates[0]
    frame_col = int(frame["color"])
    r0, c0, r1, c1 = map(int, frame["bbox"])

    sub = crop_bbox(g, frame["bbox"])
    th, tw = sub.shape

    # STEP 2) Extract holes inside the frame bbox as 4-connected non-frame regions.
    hole_mask = sub != int(frame_col)
    holes: list[tuple[tuple[tuple[int, int], bytes], tuple[int, int], np.ndarray]] = []
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        mask = np.zeros((hh, ww), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        holes.append((((mask.shape, mask.tobytes())), (int(hr0), int(hc0)), mask))

    # STEP 3) Extract outside sprites and count aligned full-patch patterns per shape key.
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    # key -> patch_signature -> [count, aligned_patch]
    sprites_by_key: dict[tuple[tuple[int, int], bytes], dict[tuple[tuple[int, int], bytes], list[object]]] = {}

    for comp in spr_comps:
        coords = comp["coords"]
        sr0, sc0, sr1, sc1 = map(int, comp["bbox"])
        hh = int(sr1 - sr0 + 1)
        ww = int(sc1 - sc0 + 1)
        rel = coords - np.array([sr0, sc0])

        patch = np.full((hh, ww), int(bg), dtype=int)
        patch[rel[:, 0], rel[:, 1]] = g[coords[:, 0], coords[:, 1]]

        variants_by_key: dict[tuple[tuple[int, int], bytes], dict[tuple[tuple[int, int], bytes], np.ndarray]] = {}
        for sig, oriented_patch in _all_rot_flip_patches(patch):
            fg_mask = oriented_patch != int(bg)
            key = (fg_mask.shape, fg_mask.tobytes())
            variants_by_key.setdefault(key, {})[sig] = oriented_patch

        for key, sig_map in variants_by_key.items():
            assert len(sig_map) == 1
            sig, aligned_patch = next(iter(sig_map.items()))
            bucket = sprites_by_key.setdefault(key, {})
            if sig not in bucket:
                bucket[sig] = [0, aligned_patch]
            bucket[sig][0] = int(bucket[sig][0]) + 1

    # STEP 4) Render the cropped output by stamping the plurality multicolor pattern into each hole.
    out = np.full((th, tw), int(frame_col), dtype=int)
    for key, (hr0, hc0), hmask in holes:
        patch_map = sprites_by_key.get(key)
        assert patch_map is not None
        best_count = max(int(info[0]) for info in patch_map.values())
        winners = [info[1] for info in patch_map.values() if int(info[0]) == best_count]
        assert len(winners) == 1
        winning_patch = winners[0]
        assert 2 <= len(_foreground_colors(winning_patch, bg)) <= 3
        paste(out, winning_patch, int(hr0), int(hc0), mask=hmask)

    return out


# --------------- generator (multicolor pattern-stamping combo) ---------------

NUM_COLORS = 10
GRID_SIZE = 30

# Frame bbox dimensions
FRAME_H_RANGE = (7, 13)
FRAME_W_RANGE = (7, 20)

# Hole layout
SHAPE_COUNT_RANGE = (3, 5)
MIN_SHAPES = 3
SHAPE_MAX_AREA = 8
FRAME_DENSITY_RANGE = (0.55, 0.92)
MIN_TOTAL_HOLES = 5
PLACEMENT_RETRIES = 80
MULT_EVEN_WEIGHTS = ([1, 2], [4, 1])
MULT_ODD_WEIGHTS = ([1, 2, 3], [3, 2, 1])

# Patterned sprite scattering
PATTERN_POOL_RANGE = (4, 5)
WINNING_SPRITE_RANGE = (3, 5)
LOSING_PATTERN_MAX = 2
LOSING_SPRITE_RANGE = (1, 2)
SPRITE_PADDING = 1
PATTERN_RETRIES = 100

# Generation retry budget
MAX_GEN_ATTEMPTS = 300


def _all_orientations(mask: np.ndarray):
    seen, result = set(), []
    for key, oriented in _all_rot_flip_masks(mask):
        if key in seen:
            continue
        seen.add(key)
        result.append(oriented)
    return result


def _all_patch_orientations(patch: np.ndarray):
    return [p for _sig, p in _all_rot_flip_patches(patch)]


def _build_shape_library():
    raw = [
        [[1, 0], [1, 0], [1, 1]],
        [[1, 0, 0], [1, 1, 1], [0, 1, 0]],
        [[1, 0, 0], [1, 1, 0], [1, 0, 1]],
        [[1, 0, 0], [1, 1, 1], [1, 1, 0]],
        [[1, 0, 0], [1, 1, 1], [0, 1, 1]],
        [[1, 1, 0], [1, 0, 0], [1, 1, 1]],
        [[1, 0, 1], [1, 1, 1], [0, 0, 1]],
        [[1, 0, 0], [1, 1, 0], [1, 0, 0], [1, 1, 1]],
    ]
    seen, dedup = set(), []
    for shape_spec in raw:
        shape = np.array(shape_spec, dtype=bool)
        key = _canonical_key(shape)
        if key in seen:
            continue
        seen.add(key)
        if _orientation_count(shape) == 8 and 5 <= int(np.sum(shape)) <= SHAPE_MAX_AREA:
            dedup.append(shape)
    return dedup


SHAPE_LIBRARY = _build_shape_library()


def _mirror_pos(r, c, sh, sw, th, tw):
    return list(
        {
            (r, c),
            (r, tw - 1 - c - sw + 1),
            (th - 1 - r - sh + 1, c),
            (th - 1 - r - sh + 1, tw - 1 - c - sw + 1),
        }
    )


def _try_place_quadrant(occ, shape, th, tw):
    sh, sw = shape.shape
    r_max, c_max = th // 2 - sh, tw // 2 - sw
    if r_max < 1 or c_max < 1:
        return None
    r = random.randint(1, r_max)
    c = random.randint(1, c_max)
    return _validate_positions(occ, shape, _mirror_pos(r, c, sh, sw, th, tw), th, tw)


def _try_place_hcenter(occ, shape, th, tw):
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
    if th % 2 == 0 or tw % 2 == 0:
        return None
    sh, sw = shape.shape
    r = th // 2 - sh // 2
    c = tw // 2 - sw // 2
    if r < 1 or c < 1 or r + sh > th - 1 or c + sw > tw - 1:
        return None
    return _validate_positions(occ, shape, [(r, c)], th, tw)


def _validate_positions(occ, shape, positions, th, tw):
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
                rr = pr + dr
                cc = pc + dc
                if not occ[rr, cc]:
                    return None
                for nr, nc in [(rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)]:
                    if 0 <= nr < th and 0 <= nc < tw and not occ[nr, nc] and (nr, nc) not in new_cells:
                        return None
    return positions


def _try_place_symmetric(occ, shape, th, tw):
    placers = [_try_place_quadrant]
    if th % 2 == 1:
        placers.append(_try_place_hcenter)
    if tw % 2 == 1:
        placers.append(_try_place_vcenter)
    if th % 2 == 1 and tw % 2 == 1:
        placers.append(_try_place_center)
    return random.choice(placers)(occ, shape, th, tw)


def _carve_hole(occ, shape, positions):
    sh, sw = shape.shape
    for pr, pc in positions:
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    occ[pr + dr, pc + dc] = False


def _build_hole_layout(th, tw):
    max_sh = max(1, (th - 2) // 2)
    max_sw = max(1, (tw - 2) // 2)
    available = [
        shape
        for shape in SHAPE_LIBRARY
        if shape.shape[0] <= max_sh and shape.shape[1] <= max_sw and int(np.sum(shape)) <= SHAPE_MAX_AREA
    ]
    if len(available) < MIN_SHAPES:
        return None

    k_target = random.randint(SHAPE_COUNT_RANGE[0], min(SHAPE_COUNT_RANGE[1], len(available)))
    random.shuffle(available)

    occ = np.ones((th, tw), dtype=bool)
    hole_placements = []
    even_dims = th % 2 == 0 or tw % 2 == 0
    mult_vals, mult_wts = MULT_EVEN_WEIGHTS if even_dims else MULT_ODD_WEIGHTS

    for shape in available:
        if len(hole_placements) >= k_target:
            break
        mult = random.choices(mult_vals, weights=mult_wts)[0]
        placed = []
        for _ in range(mult):
            for _ in range(PLACEMENT_RETRIES):
                positions = _try_place_symmetric(occ, shape, th, tw)
                if positions is None:
                    continue
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

    total_holes = sum(len(pos_group) for _shape, groups in hole_placements for pos_group in groups)
    if total_holes < MIN_TOTAL_HOLES:
        return None

    return occ, hole_placements


def _random_pattern(shape: np.ndarray, palette: list[int], bg: int, banned: set):
    cells = [tuple(map(int, rc)) for rc in np.argwhere(shape)]
    max_colors = min(3, len(cells), len(palette))
    if max_colors < 2:
        return None

    for _ in range(PATTERN_RETRIES):
        n_cols = random.randint(2, max_colors)
        cols = random.sample(palette, n_cols)
        dominant = int(cols[0])
        accents = [int(c) for c in cols[1:]]
        max_total_accents = min(len(cells) - 1, max(n_cols - 1, (len(cells) - 1) // 2))
        if max_total_accents < n_cols - 1:
            continue
        total_accents = random.randint(n_cols - 1, max_total_accents)

        patch = np.full(shape.shape, int(bg), dtype=int)
        for rr, cc in cells:
            patch[rr, cc] = dominant

        chosen_cells = random.sample(cells, total_accents)
        random.shuffle(chosen_cells)
        for (rr, cc), col in zip(chosen_cells[: len(accents)], accents):
            patch[rr, cc] = int(col)
        extra_cells = chosen_cells[len(accents) :]
        for rr, cc in extra_cells:
            patch[rr, cc] = int(random.choice(accents))

        fg_vals = [int(patch[rr, cc]) for rr, cc in cells]
        counts = Counter(fg_vals)
        if not (2 <= len(counts) <= 3):
            continue
        if counts[dominant] <= max(v for k, v in counts.items() if k != dominant):
            continue

        sig = _canonical_patch_signature(patch)
        if sig in banned:
            continue
        return patch, sig
    return None


def _build_output(occ, hole_placements, frame_col, bg):
    th, tw = occ.shape
    avail_colors = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    palette_size = random.randint(PATTERN_POOL_RANGE[0], min(PATTERN_POOL_RANGE[1], len(avail_colors)))
    pattern_palette = random.sample(avail_colors, palette_size)

    output = np.full((th, tw), int(frame_col), dtype=int)
    shape_info = []

    for shape, groups in hole_placements:
        built = _random_pattern(shape, pattern_palette, bg, set())
        if built is None:
            return None
        winning_patch, winning_sig = built

        banned = {winning_sig}
        losing_patches = []
        n_losing = random.randint(1, LOSING_PATTERN_MAX)
        for _ in range(n_losing):
            alt = _random_pattern(shape, pattern_palette, bg, banned)
            if alt is None:
                return None
            losing_patch, losing_sig = alt
            banned.add(losing_sig)
            losing_patches.append(losing_patch)

        all_pos = []
        for positions in groups:
            for pr, pc in positions:
                paste(output, winning_patch, pr, pc, mask=shape)
                all_pos.append((pr, pc))

        shape_info.append((shape, winning_patch, losing_patches, all_pos))

    return output, shape_info


def _place_sprite(inp, placement, patch, bg):
    try:
        r, c = random_free_location(placement, patch, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, patch, r, c, transparent=bg)
    paste(inp, patch, r, c, transparent=bg)
    return True


def _scatter_sprites(inp, shape_info, bg):
    placement = inp.copy()

    for _shape, winning_patch, losing_patches, _positions in shape_info:
        winning_orients = _all_patch_orientations(winning_patch)
        n_winning = random.randint(*WINNING_SPRITE_RANGE)
        for _ in range(n_winning):
            if not _place_sprite(inp, placement, random.choice(winning_orients), bg):
                return False

        for losing_patch in losing_patches:
            losing_orients = _all_patch_orientations(losing_patch)
            n_losing = random.randint(LOSING_SPRITE_RANGE[0], min(LOSING_SPRITE_RANGE[1], n_winning - 1))
            for _ in range(n_losing):
                if not _place_sprite(inp, placement, random.choice(losing_orients), bg):
                    break
    return True


def _gen_attempt() -> np.ndarray | None:
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

    if not _scatter_sprites(inp, shape_info, bg):
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
