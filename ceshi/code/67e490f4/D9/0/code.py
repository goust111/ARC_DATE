# Variant of 67e490f4
# Dimension: D9 — 变换：填充操作方式 (transform)
# Change: hole_fill — 每个洞的所有像素统一填充为匹配精灵的最高频颜色，生成纯色填充... → 保留精灵内部图案填充

from __future__ import annotations

import random
import numpy as np
from arc_utils import bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste, random_free_location

# ---------------------------------------------------------------------------
# Design analysis for the D9 variant
#
# a) Concrete behavior:
#    The frame-and-hole scaffold stays the same: one large monochrome frame,
#    bg-colored holes inside it, and scattered source sprites outside it.
#    The changed step is the fill action. Originally a matching sprite shape
#    only contributed one scalar: its dominant color, so every hole became a
#    flat color blob. In this variant the sprite is treated as a real stamp:
#    after rotating/flipping it to the hole's orientation, the full colored
#    pixel arrangement is copied into the hole.
#
#    Small trace:
#    - Old logic: three L-shaped sprites outside = red, red, blue ->
#      every L-shaped hole becomes solid red.
#    - New logic: three L-shaped sprites outside = pattern A, pattern A,
#      pattern B -> every L-shaped hole receives the aligned copy of pattern A.
#      If a winning sprite is seen rotated outside the frame, solver must undo
#      that orientation before stamping it into the hole.
#
# b) Complexity calibration:
#    Purely changing the solver to support multicolor sprites is not enough;
#    the generator must actually produce non-uniform patterns or the variant
#    visually collapses back to the original task. Each chosen hole shape
#    therefore gets a winning pattern with at least two colors, plus 1-2 losing
#    patterns of the same silhouette. Shape count remains small (3-5), and
#    each shape has repeated evidence via multiple sprite copies.
#
# c) Hidden-assumption review of the original main():
#    - Frame detection by largest monochrome component still holds.
#    - Hole extraction by "sub != frame_col" still holds because holes in the
#      input are still bg gaps; the new multicolor content only appears in the
#      output.
#    - The original sprite loop assumed every component could be summarized by
#      one representative color. That assumption is invalid here; the solver
#      must preserve the whole patch and count exact aligned patches instead.
#    - The original output step created np.full(hmask.shape, scol). That would
#      destroy all internal sprite structure in this variant.
#
# d) Placement legality:
#    Sprites remain fully outside the frame bbox and are scattered with at
#    least one bg cell of Chebyshev padding, so multicolor components do not
#    accidentally merge. Hole cells remain 4-disconnected from each other and
#    away from the border, so each hole bbox is well-defined and can be filled
#    independently.
#
# e) Visual observability:
#    A flat-color output would be indistinguishable from the original rule, so
#    the generator forbids monochrome winning patterns. It also prefers hole
#    shapes with unique orientation under the dihedral transforms; this removes
#    stamp-direction ambiguity and makes rotation/flip alignment meaningful.
#
# f) Visual cleanliness:
#    Nothing is overlaid on top of the frame except the frame itself and the
#    outside sprites. The output is still just the cropped frame rectangle, so
#    the only added complexity is that holes now show small internal motifs
#    rather than uniform fills.
#
# g) Color inferability:
#    The task no longer depends on hidden ordering over color ids. The
#    generator guarantees a unique plurality winner by count for every shape,
#    so the solver never needs color-based tie-breaking to decide which stamp
#    wins. Colors are drawn from a small shared palette to keep samples legible.
#
# h) Variant-specific anti-degeneration guard:
#    Winning patterns are always non-uniform. Therefore, on the same input,
#    the original "dominant-color fill" solver would generally disagree with
#    this solver; the change is not cosmetic.
#
# i) New main() assumptions and generator guarantees:
#    - Assumption: each hole shape has at least one matching outside sprite.
#      Generator: every chosen shape scatters multiple winning sprites.
#    - Assumption: for a given sprite and hole orientation, the aligned patch is
#      unique. Generator: only uses hole silhouettes whose 8 transforms are all
#      distinct, avoiding orientation ambiguity from symmetric masks.
#    - Assumption: each shape has a strict plurality winning pattern.
#      Generator: winning count is always higher than every losing pattern count.
# ---------------------------------------------------------------------------


def _all_rot_flip_masks(mask: np.ndarray):
    """Yield (key, oriented_mask) for all 8 rotation/flip variants of a bool mask."""
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes()), m2


def _all_rot_flip_patches(patch: np.ndarray):
    """Yield deduplicated full-patch transforms, including bg cells inside the bbox."""
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp hole interiors inside a large frame using shape-matched outside sprites.

    Input structure:
    - bg = most common color. One large monochrome frame object sits in the grid.
    - Inside the frame bbox: several bg-colored holes of different small shapes.
    - Outside the frame: small separated multicolor sprites whose silhouettes match hole shapes up to rotation/flip.

    Rule:
    - Crop to the frame's bounding box and start from a solid frame-colored rectangle.
    - For each hole silhouette, gather outside sprites whose foreground mask matches that hole under rotation/flip.
    - Align the most frequent full sprite pattern to the hole orientation, then paste that whole aligned pattern into the hole.

    Key insight: The outside sprites are literal stamps rather than color keys — shape tells you which hole they belong to, and the repeated internal pixel arrangement tells you exactly what to copy.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Find the template frame: the largest monochrome non-bg component.
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

    # STEP 2) Extract holes inside the frame bbox as 4-connected non-frame regions.
    hole_mask = (sub != int(frame_col))
    holes: list[tuple[tuple[tuple[int, int], bytes], tuple[int, int], np.ndarray]] = []
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        m = np.zeros((hh, ww), dtype=bool)
        m[rel[:, 0], rel[:, 1]] = True
        holes.append((((m.shape, m.tobytes())), (int(hr0), int(hc0)), m))

    # STEP 3) Extract outside sprites, preserving their full colored patches.
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
            key = ((oriented_patch != int(bg)).shape, (oriented_patch != int(bg)).tobytes())
            variants_by_key.setdefault(key, {})[sig] = oriented_patch

        for key, sig_map in variants_by_key.items():
            # Unique alignment per hole orientation is guaranteed by the generator.
            assert len(sig_map) == 1
            sig, aligned_patch = next(iter(sig_map.items()))
            bucket = sprites_by_key.setdefault(key, {})
            if sig not in bucket:
                bucket[sig] = [0, aligned_patch]
            bucket[sig][0] = int(bucket[sig][0]) + 1

    # STEP 4) Render output: solid frame first, then stamp the winning aligned pattern into each hole.
    out = np.full((th, tw), int(frame_col), dtype=int)
    for key, (hr0, hc0), hmask in holes:
        patch_map = sprites_by_key.get(key)
        assert patch_map is not None
        best_count = max(int(info[0]) for info in patch_map.values())
        winners = [info[1] for info in patch_map.values() if int(info[0]) == best_count]
        assert len(winners) == 1
        paste(out, winners[0], int(hr0), int(hc0), mask=hmask)

    return out


# --------------- generator (pattern-stamping variant) ---------------

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
PATTERN_RETRIES = 80

# Generation retry budget
MAX_GEN_ATTEMPTS = 300


def _all_orientations(mask: np.ndarray):
    seen, result = set(), []
    for key, m in _all_rot_flip_masks(mask):
        if key in seen:
            continue
        seen.add(key)
        result.append(m)
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
    for s in raw:
        m = np.array(s, dtype=bool)
        k = _canonical_key(m)
        if k in seen:
            continue
        seen.add(k)
        if _orientation_count(m) == 8 and int(np.sum(m)) <= SHAPE_MAX_AREA:
            dedup.append(m)
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
    r, c = random.randint(1, r_max), random.randint(1, c_max)
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
    r, c = th // 2 - sh // 2, tw // 2 - sw // 2
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
                rr, cc = pr + dr, pc + dc
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
    max_sh, max_sw = max(1, (th - 2) // 2), max(1, (tw - 2) // 2)
    available = [
        s
        for s in SHAPE_LIBRARY
        if s.shape[0] <= max_sh and s.shape[1] <= max_sw and int(np.sum(s)) <= SHAPE_MAX_AREA
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

    total_holes = sum(len(pos) for _, pls in hole_placements for pos in pls)
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
        patch = np.full(shape.shape, int(bg), dtype=int)
        random.shuffle(cells)

        for (rr, cc), col in zip(cells[:n_cols], cols):
            patch[rr, cc] = int(col)
        for rr, cc in cells[n_cols:]:
            patch[rr, cc] = int(random.choice(cols))

        fg_vals = [int(patch[rr, cc]) for rr, cc in cells]
        if len(set(fg_vals)) < 2:
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

    output = np.full((th, tw), frame_col, dtype=int)
    shape_info = []

    for shape, pls in hole_placements:
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
        for positions in pls:
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


def _scatter_sprites(inp, shape_info, bg, frame_col):
    placement = inp.copy()
    _ = frame_col

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

    if not _scatter_sprites(inp, shape_info, bg, frame_col):
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
