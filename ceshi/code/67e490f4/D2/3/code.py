# Variant of 67e490f4
# Dimension: D2 — 匹配条件：旋转/翻转的允许范围 (match_cond)
# Change: rotation — 匹配时允许 8 种旋转/翻转变体（0/90/180/270度旋转 x 是否垂直翻转）... → 仅允许180度旋转

# Design analysis for the D2 variant
# a) Pixel-level behavior
#    - The large outer object is still a single monochrome frame whose holes are cut out as bg-colored regions.
#    - Small colored sprites still live outside the frame and act as color keys for hole filling.
#    - The change is local but decisive: a hole now matches only sprites with the same mask or the 180-degree-rotated
#      mask. Quarter-turns and flips no longer count.
#    - Concrete trace:
#      old logic: an L-hole, a 90-degree-rotated L sprite, and a 180-degree-rotated L sprite all shared one key.
#      new logic: the hole shares a key only with the 180-degree partner, while the 90-degree version becomes a
#      different class and can receive a different color.
# b) Complexity calibration
#    - If every chosen hole shape were point-symmetric, the task would visually collapse back to "exact shape match".
#    - To prevent that, the generator always includes at least one family that splits under the new rule:
#      two hole shapes come from the same full rotation/flip family but belong to different 180-only classes.
#    - The generator also forces at least one winning sprite set to use only the 180-degree orientation when that
#      orientation is visually distinct, so the retained symmetry is observable too.
# c) Hidden assumptions in the original solver
#    - Original STEP 2 used the raw hole mask as a key because STEP 3 expanded each sprite to all 8 variants.
#      That assumption breaks here: 90-degree relatives must stop matching.
#    - Original STEP 3 assumed "canonical under full dihedral symmetry". The variant replaces that with
#      "canonical under {identity, rot180}" only.
#    - Original generator deduplicated the library under full rotation/flip symmetry. That would erase exactly the
#      distinctions this variant needs, so the new library deduplicates only under 180-degree rotation.
# d) Placement legality
#    - Holes still stay inside the frame border, remain 4-disconnected, and sprites stay fully outside the frame bbox
#      with Chebyshev padding >= 1, so component extraction remains stable.
#    - The forced contrast pair is selected before placement, but the attempt is rejected unless both shapes actually
#      get placed and at least one matching sprite is scattered for each.
# e) Visual observability
#    - Each sample contains at least one pair of shapes that would merge under the original rule but must stay separate
#      here, so a human can see that quarter-turn equivalence is gone.
#    - No color-id-based behavior is used as part of the intended concept; the generator enforces strict pluralities to
#      avoid relying on tie-breaking.
# f) Visual cleanliness
#    - The frame is still the unique largest monochrome component, holes remain separated by frame-colored cells, and
#      sprites remain isolated on background, so object boundaries stay readable.
# g) Color inferability
#    - Colors are assigned one per distinct 180-only shape class. Outside sprites repeat those colors with plurality,
#      making the mapping inferable from repeated shape-color support rather than arbitrary recoloring.
# h) New solver-side assumptions and generator guarantees
#    - `main()` assumes every hole class has at least one outside sprite in the same 180-only class.
#    - It assumes the frame is still the unique largest monochrome non-bg component.
#    - It assumes plurality colors are strict in practice, so human-solvable behavior does not depend on hidden ties.
#    - The generator guarantees all three and additionally rejects samples where the old 8-way matcher would produce
#      the same output, preventing silent regression to the source task.

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


def _iter_rot180_variants(mask: np.ndarray):
    """Yield unique identity / 180-degree variants of a bool mask."""
    m0 = mask.astype(bool)
    seen = set()
    for k in (0, 2):
        m = np.rot90(m0, k)
        key = (m.shape, m.tobytes())
        if key not in seen:
            seen.add(key)
            yield key, m


def _iter_full_variants(mask: np.ndarray):
    """Yield unique variants from 4 rotations x optional vertical flip."""
    m0 = mask.astype(bool)
    seen = set()
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            key = (m2.shape, m2.tobytes())
            if key not in seen:
                seen.add(key)
                yield key, m2


def _canonical_key_rot180(mask: np.ndarray):
    return min(key for key, _ in _iter_rot180_variants(mask))


def _canonical_key_full(mask: np.ndarray):
    return min(key for key, _ in _iter_full_variants(mask))


def _solve_with_key(grid: np.ndarray, key_fn) -> np.ndarray:
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

    hole_mask = sub != int(frame_col)
    holes: list[tuple[tuple[tuple[int, int], bytes], tuple[int, int], np.ndarray]] = []
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        m = np.zeros((hh, ww), dtype=bool)
        m[rel[:, 0], rel[:, 1]] = True
        holes.append((key_fn(m), (int(hr0), int(hc0)), m))

    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites_by_key: dict[tuple[tuple[int, int], bytes], dict[int, int]] = {}
    for comp in spr_comps:
        coords = comp["coords"]
        sr0, sc0, sr1, sc1 = map(int, comp["bbox"])
        hh = int(sr1 - sr0 + 1)
        ww = int(sc1 - sc0 + 1)

        rel = coords - np.array([sr0, sc0])
        m = np.zeros((hh, ww), dtype=bool)
        m[rel[:, 0], rel[:, 1]] = True

        vals = [int(v) for v in g[coords[:, 0], coords[:, 1]]]
        scol = int(Counter(vals).most_common(1)[0][0])

        bucket = sprites_by_key.setdefault(key_fn(m), {})
        bucket[scol] = int(bucket.get(scol, 0)) + 1

    out = np.full((th, tw), int(frame_col), dtype=int)
    for key, (hr0, hc0), hmask in holes:
        cmap = sprites_by_key.get(key)
        assert cmap is not None
        scol = max(cmap.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        patch = np.full(hmask.shape, int(scol), dtype=int)
        paste(out, patch, int(hr0), int(hc0), mask=hmask)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor holes inside a large frame using outside sprites that match only by point symmetry.

    Input structure:
    - The background is the most common color, plus one large monochrome frame component.
    - Inside that frame bbox are several hole components carved out of the frame color.
    - Outside the frame bbox are small isolated colored sprites on the background.

    Rule:
    - Crop to the frame bbox and begin from a solid block of the frame color.
    - For each hole, look for outside sprites whose mask is identical to the hole or to its 180-degree rotation.
    - Fill the hole with the most frequent color among those matching sprites.

    Key insight:
    - Shape matching is stricter than the source task: 90-degree turns and flips no longer merge classes, so
      quarter-turned relatives can drive different hole colors while 180-degree partners still count as matches.
    """
    return _solve_with_key(grid, _canonical_key_rot180)


def _legacy_fullsym_main(grid: np.ndarray) -> np.ndarray:
    """Original 8-way matching rule, kept only to reject degenerate samples during generation."""
    return _solve_with_key(grid, _canonical_key_full)


# --------------- generator ---------------

NUM_COLORS = 10
GRID_SIZE = 30

FRAME_H_RANGE = (7, 13)
FRAME_W_RANGE = (7, 20)

SHAPE_COUNT_RANGE = (3, 6)
MIN_SHAPES = 3
SHAPE_MAX_AREA = 8
FRAME_DENSITY_RANGE = (0.55, 0.92)
MIN_TOTAL_HOLES = 5
PLACEMENT_RETRIES = 80
MULT_EVEN_WEIGHTS = ([1, 2], [4, 1])
MULT_ODD_WEIGHTS = ([1, 2, 3], [3, 2, 1])

WINNING_SPRITE_RANGE = (2, 5)
OTHER_COLOR_MAX = 2
SPRITE_PADDING = 1

MAX_GEN_ATTEMPTS = 400


def _distinct_rot180_orientations(mask: np.ndarray):
    return [m.copy() for _, m in _iter_rot180_variants(mask)]


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
    entries_by_key = {}
    for s in raw:
        base = np.array(s, dtype=bool)
        family = _canonical_key_full(base)
        for _, oriented in _iter_full_variants(base):
            k180 = _canonical_key_rot180(oriented)
            if k180 in entries_by_key:
                continue
            entries_by_key[k180] = {
                "mask": oriented.copy(),
                "family": family,
                "rot180_key": k180,
                "area": int(np.sum(oriented)),
                "point_symmetric": bool(np.array_equal(oriented, np.rot90(oriented, 2))),
            }
    return list(entries_by_key.values())


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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < th and 0 <= nc < tw and not occ[nr, nc] and (nr, nc) not in new_cells:
                        return None
    return positions


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


def _select_shape_entries(th, tw):
    max_sh = max(1, (th - 2) // 2)
    max_sw = max(1, (tw - 2) // 2)
    available = [
        entry
        for entry in SHAPE_LIBRARY
        if entry["mask"].shape[0] <= max_sh
        and entry["mask"].shape[1] <= max_sw
        and entry["area"] <= SHAPE_MAX_AREA
    ]
    if len(available) < MIN_SHAPES:
        return None

    families = {}
    for entry in available:
        families.setdefault(entry["family"], []).append(entry)

    contrast_families = [
        members
        for members in families.values()
        if len(members) >= 2 and any(not entry["point_symmetric"] for entry in members)
    ]
    if not contrast_families:
        return None

    k_target = random.randint(SHAPE_COUNT_RANGE[0], min(SHAPE_COUNT_RANGE[1], len(available)))
    anchor_family = random.choice(contrast_families)
    forced_pair = random.sample(anchor_family, 2)
    forced_keys = {entry["rot180_key"] for entry in forced_pair}

    remaining = [entry for entry in available if entry["rot180_key"] not in forced_keys]
    if len(remaining) < k_target - 2:
        return None

    extras = random.sample(remaining, k_target - 2)
    chosen = forced_pair + extras
    random.shuffle(chosen)
    return chosen, forced_keys


def _build_hole_layout(th, tw):
    selected = _select_shape_entries(th, tw)
    if selected is None:
        return None
    chosen_entries, forced_keys = selected

    even_dims = th % 2 == 0 or tw % 2 == 0
    if even_dims:
        chosen_entries = sorted(chosen_entries, key=lambda entry: entry["area"])

    occ = np.ones((th, tw), dtype=bool)
    hole_placements = []
    mult_vals, mult_wts = MULT_EVEN_WEIGHTS if even_dims else MULT_ODD_WEIGHTS

    for entry in chosen_entries:
        shape = entry["mask"]
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
            hole_placements.append((entry, placed))
        elif entry["rot180_key"] in forced_keys:
            return None

    placed_keys = {entry["rot180_key"] for entry, _ in hole_placements}
    if not forced_keys.issubset(placed_keys):
        return None
    if len(hole_placements) < MIN_SHAPES:
        return None

    frame_density = 1.0 - int(np.sum(~occ)) / (th * tw)
    if not (FRAME_DENSITY_RANGE[0] <= frame_density <= FRAME_DENSITY_RANGE[1]):
        return None

    total_holes = sum(len(positions) for _, groups in hole_placements for positions in groups)
    if total_holes < MIN_TOTAL_HOLES:
        return None

    return occ, hole_placements


def _build_output(occ, hole_placements, frame_col, bg):
    th, tw = occ.shape
    avail_colors = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    if len(avail_colors) < len(hole_placements):
        return None
    shape_colors = random.sample(avail_colors, len(hole_placements))

    output = np.full((th, tw), frame_col, dtype=int)
    shape_info = []
    for (entry, groups), fill_col in zip(hole_placements, shape_colors):
        all_pos = []
        shape = entry["mask"]
        for positions in groups:
            for pr, pc in positions:
                paste(output, np.full(shape.shape, fill_col, dtype=int), pr, pc, mask=shape)
                all_pos.append((pr, pc))
        shape_info.append(
            {
                "mask": shape,
                "fill_color": int(fill_col),
                "positions": all_pos,
                "family": entry["family"],
                "rot180_key": entry["rot180_key"],
                "point_symmetric": entry["point_symmetric"],
            }
        )
    return output, shape_info


def _place_sprite(inp, placement, ori, color, bg):
    patch = np.full(ori.shape, bg, dtype=int)
    patch[ori] = color
    try:
        r, c = random_free_location(placement, patch, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, patch, r, c, transparent=bg)
    paste(inp, patch, r, c, transparent=bg)
    return True


def _scatter_sprites(inp, shape_info, bg, frame_col, forced_rot180_key):
    placement = inp.copy()
    other_pool = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    sprite_stats = []

    for info in shape_info:
        shape = info["mask"]
        fill_color = info["fill_color"]
        orientations = _distinct_rot180_orientations(shape)
        losers = [c for c in other_pool if c != fill_color]
        n_winning = random.randint(*WINNING_SPRITE_RANGE)
        n_other = random.randint(1, min(OTHER_COLOR_MAX, len(losers))) if losers else 0
        oc_list = random.sample(losers, min(n_other, len(losers)))

        force_rot180_only = info["rot180_key"] == forced_rot180_key and len(orientations) > 1
        winning_indices = []

        for _ in range(n_winning):
            if force_rot180_only:
                idx = 1
            else:
                idx = random.randrange(len(orientations))
            if not _place_sprite(inp, placement, orientations[idx], fill_color, bg):
                return False, None
            winning_indices.append(idx)

        for oc in oc_list:
            n_oc = random.randint(1, max(1, n_winning - 1))
            for _ in range(n_oc):
                idx = random.randrange(len(orientations))
                if not _place_sprite(inp, placement, orientations[idx], oc, bg):
                    break

        sprite_stats.append(
            {
                "family": info["family"],
                "rot180_key": info["rot180_key"],
                "rot180_only": force_rot180_only,
                "winning_orientations": tuple(sorted(set(winning_indices))),
            }
        )

    return True, sprite_stats


def _variant_metrics(shape_info, sprite_stats, legacy_diff):
    family_counts = Counter(info["family"] for info in shape_info)
    return {
        "legacy_diff": bool(legacy_diff),
        "contrast_families": sum(1 for count in family_counts.values() if count >= 2),
        "rot180_only_shapes": sum(1 for stat in sprite_stats if stat["rot180_only"]),
        "distinct_shape_count": len(shape_info),
    }


def _gen_attempt(return_meta: bool = False):
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

    rot180_candidates = [info for info in shape_info if len(_distinct_rot180_orientations(info["mask"])) > 1]
    if not rot180_candidates:
        return None
    forced_rot180_key = random.choice(rot180_candidates)["rot180_key"]

    fr = random.randint(0, GRID_SIZE - th)
    fc = random.randint(0, GRID_SIZE - tw)
    inp = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    paste(inp, np.where(occ, frame_col, bg).astype(int), fr, fc)

    ok, sprite_stats = _scatter_sprites(inp, shape_info, bg, frame_col, forced_rot180_key)
    if not ok:
        return None

    try:
        solved = main(inp)
        if not np.array_equal(solved, output):
            return None
        legacy_diff = not np.array_equal(_legacy_fullsym_main(inp), output)
    except Exception:
        return None

    metrics = _variant_metrics(shape_info, sprite_stats, legacy_diff)
    if not metrics["legacy_diff"]:
        return None
    if metrics["contrast_families"] < 1:
        return None
    if metrics["rot180_only_shapes"] < 1:
        return None

    if return_meta:
        return inp, metrics
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {MAX_GEN_ATTEMPTS} attempts")
