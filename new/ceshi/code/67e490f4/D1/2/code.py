# Variant of 67e490f4
# Dimension: D1 — 匹配条件：形状匹配方式 (match_cond)
# Change: shape_match — 精灵的布尔掩码与洞的布尔掩码相同（允许旋转和翻转），即 s.mask == d.mask... → 面积匹配替代形状匹配

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


# Design analysis
# ---------------
# a) Concrete behavior trace:
#    - Original task: each hole shape inside the frame is keyed by its exact boolean mask
#      (up to rotation/flip); outside sprites with the same mask vote on a fill color.
#    - Variant: the key is only the pixel count. A 3-cell L hole and a 3-cell bar hole now
#      belong to the same class, so both holes are filled by the plurality color among ALL
#      outside 3-cell sprites, regardless of shape.
#    - Small trace:
#        holes inside frame: area-3 bar, area-3 L, area-4 square
#        outside sprites: bar(blue)x4, L(red)x3, square(green)x3
#      Original: bar->blue, L->red, square->green.
#      Variant: both area-3 holes -> blue, area-4 hole -> green.
#      The changed step is the lookup key in both solver matching and generator voting.
# b) Complexity calibration:
#    - To make the change visible, generation must force at least one area class to contain
#      at least two distinct hole shapes. Otherwise area-match can collapse to the original.
#    - Distinct used areas are kept small (2-4), while total distinct shapes stay moderate
#      (3-6), so the puzzle still has repeated evidence instead of too many one-off cases.
# c) Old assumptions that break under the variant:
#    - `key = (mask.shape, mask.tobytes())` for holes/sprites is no longer valid because the
#      same area can intentionally map multiple shapes to one color.
#    - sprite votes may no longer be shape-local; the solver must aggregate all same-area
#      sprites before selecting a color.
#    - the original generator assigned one color per shape; that would erase the variant.
#      The new generator assigns one color per area and ensures a shared-area witness exists.
# d) Placement legality:
#    - Hole placement logic is unchanged: holes remain 4-disconnected, stay away from the
#      frame border, and the full frame pattern remains mirror-symmetric.
#    - Sprite placement still uses `random_free_location(..., padding=1)`, so sprites remain
#      separated from each other and from the frame bbox.
# e) Visual observability:
#    - Every generated grid contains at least one "shared area" with >=2 visibly different
#      hole shapes. Those holes receive the same output color, and their outside sprites show
#      that the shared color is determined by total count over that area class.
#    - A witness area is further biased so one shape would prefer a different color under the
#      old shape-match rule, making the variant behavior materially different.
# f) Visual cleanliness:
#    - The frame/hole layout and sparse outside scattering are preserved from the source task.
#      No extra markings or overlaps are introduced.
# g) Color inferability:
#    - Only a few area classes are active per sample, each with one strict plurality color.
#      The generator avoids ties, so humans do not need any invisible color-id tiebreak.
# h) New solver assumptions and generator guarantees:
#    - Solver assumes every hole area appears among outside sprites: generator guarantees it.
#    - Solver assumes plurality is strict enough to be readable: generator constructs a unique
#      winner per used area.
#    - Solver assumes the largest monochrome component is the frame: sprite spacing and small
#      sprite sizes preserve that property.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill holes inside a large template frame using area-matched outside sprites.

    Input structure:
    - bg = most common color. One large monochrome frame object (template).
    - Inside the frame: holes of various small shapes (non-frame-color regions).
    - Outside the frame: small monochrome sprites scattered on bg.

    Rule:
    - Crop to the frame's bounding box; fill it entirely with the frame color.
    - For each hole, count its pixels and gather all outside sprites with the same pixel area.
    - Fill the hole with the most frequent color among sprites of that area.

    Key insight: Sprites are no longer shape keys; they are area-coded votes, so different
    hole shapes with the same number of cells inherit the same winning color.
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

    hole_mask = sub != int(frame_col)
    holes: list[tuple[int, tuple[int, int], np.ndarray]] = []
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        m = np.zeros((hh, ww), dtype=bool)
        m[rel[:, 0], rel[:, 1]] = True
        holes.append((int(coords.shape[0]), (int(hr0), int(hc0)), m))

    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites_by_area: dict[int, dict[int, int]] = {}
    for comp in spr_comps:
        coords = comp["coords"]
        area = int(coords.shape[0])
        vals = [int(v) for v in g[coords[:, 0], coords[:, 1]]]
        scol = int(Counter(vals).most_common(1)[0][0])
        bucket = sprites_by_area.setdefault(area, {})
        bucket[scol] = int(bucket.get(scol, 0)) + 1

    out = np.full((th, tw), int(frame_col), dtype=int)
    for area, (hr0, hc0), hmask in holes:
        cmap = sprites_by_area.get(int(area))
        assert cmap is not None
        scol = max(cmap.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        patch = np.full(hmask.shape, int(scol), dtype=int)
        paste(out, patch, int(hr0), int(hc0), mask=hmask)

    return out


# --------------- generator ---------------

NUM_COLORS = 10
GRID_SIZE = 30

FRAME_H_RANGE = (7, 13)
FRAME_W_RANGE = (7, 20)

TOTAL_SHAPE_COUNT_RANGE = (3, 6)
AREA_GROUP_COUNT_RANGE = (2, 4)
MIN_SHAPES = 3
MIN_SHARED_AREA_SHAPES = 2
MAX_SHAPES_PER_AREA = 3
SHAPE_MAX_AREA = 8
FRAME_DENSITY_RANGE = (0.55, 0.92)
MIN_TOTAL_HOLES = 5
PLACEMENT_RETRIES = 80
MULT_EVEN_WEIGHTS = ([1, 2], [4, 1])
MULT_ODD_WEIGHTS = ([1, 2, 3], [3, 2, 1])

WINNING_SPRITE_RANGE = (2, 4)
OTHER_COLOR_MAX = 2
SPRITE_PADDING = 1

MAX_GEN_ATTEMPTS = 300


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
    seen, result = set(), []
    for key, m in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            result.append(m)
    return result


def _shape_area(mask: np.ndarray) -> int:
    return int(np.sum(mask))


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
    seen, dedup = set(), []
    for s in raw:
        m = np.array(s, dtype=bool)
        k = _canonical_key(m)
        if k not in seen:
            seen.add(k)
            dedup.append(m)
    return dedup


SHAPE_LIBRARY = _build_shape_library()


def _shapes_by_area(max_sh: int, max_sw: int) -> dict[int, list[np.ndarray]]:
    groups: dict[int, list[np.ndarray]] = {}
    for shape in SHAPE_LIBRARY:
        if shape.shape[0] > max_sh or shape.shape[1] > max_sw:
            continue
        area = _shape_area(shape)
        if area > SHAPE_MAX_AREA:
            continue
        groups.setdefault(area, []).append(shape)
    return groups


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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < th and 0 <= nc < tw and not occ[nr, nc]:
                        if (nr, nc) not in new_cells:
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


def _select_shapes_for_layout(th, tw):
    max_sh, max_sw = max(1, (th - 2) // 2), max(1, (tw - 2) // 2)
    by_area = _shapes_by_area(max_sh, max_sw)
    candidate_areas = [area for area, shapes in by_area.items() if shapes]
    shared_areas = [area for area, shapes in by_area.items() if len(shapes) >= MIN_SHARED_AREA_SHAPES]
    if len(candidate_areas) < AREA_GROUP_COUNT_RANGE[0] or not shared_areas:
        return None

    n_areas = random.randint(AREA_GROUP_COUNT_RANGE[0], min(AREA_GROUP_COUNT_RANGE[1], len(candidate_areas)))
    witness_area = random.choice(shared_areas)
    other_areas = [area for area in candidate_areas if area != witness_area]
    chosen_areas = [witness_area]
    if n_areas > 1:
        chosen_areas += random.sample(other_areas, n_areas - 1)

    counts = {area: 1 for area in chosen_areas}
    counts[witness_area] = 2
    max_total = min(TOTAL_SHAPE_COUNT_RANGE[1], sum(min(MAX_SHAPES_PER_AREA, len(by_area[a])) for a in chosen_areas))
    min_total = max(TOTAL_SHAPE_COUNT_RANGE[0], sum(counts.values()))
    if min_total > max_total:
        return None

    target_total = random.randint(min_total, max_total)
    while sum(counts.values()) < target_total:
        expandable = [
            area
            for area in chosen_areas
            if counts[area] < min(MAX_SHAPES_PER_AREA, len(by_area[area]))
        ]
        if not expandable:
            break
        counts[random.choice(expandable)] += 1

    selected = []
    for area in chosen_areas:
        pool = list(by_area[area])
        random.shuffle(pool)
        selected.extend(pool[: counts[area]])

    random.shuffle(selected)
    selected.sort(key=lambda s: (-_shape_area(s), -(s.shape[0] * s.shape[1])))
    return selected


def _build_hole_layout(th, tw):
    selected_shapes = _select_shapes_for_layout(th, tw)
    if selected_shapes is None:
        return None

    occ = np.ones((th, tw), dtype=bool)
    hole_placements = []
    mult_vals, mult_wts = MULT_EVEN_WEIGHTS if (th % 2 == 0 or tw % 2 == 0) else MULT_ODD_WEIGHTS

    for shape in selected_shapes:
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

    area_shape_counts: dict[int, int] = {}
    for shape, _ in hole_placements:
        area = _shape_area(shape)
        area_shape_counts[area] = int(area_shape_counts.get(area, 0)) + 1
    if len(area_shape_counts) < AREA_GROUP_COUNT_RANGE[0]:
        return None
    if max(area_shape_counts.values()) < MIN_SHARED_AREA_SHAPES:
        return None

    frame_density = 1.0 - int(np.sum(~occ)) / (th * tw)
    if not (FRAME_DENSITY_RANGE[0] <= frame_density <= FRAME_DENSITY_RANGE[1]):
        return None

    total_holes = sum(len(pos) for _, pls in hole_placements for pos in pls)
    if total_holes < MIN_TOTAL_HOLES:
        return None

    return occ, hole_placements


def _build_output(occ, hole_placements, frame_col, bg):
    th, tw = occ.shape
    used_areas = sorted({_shape_area(shape) for shape, _ in hole_placements})
    avail_colors = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    if len(avail_colors) < len(used_areas):
        return None
    area_colors = dict(zip(used_areas, random.sample(avail_colors, len(used_areas))))

    output = np.full((th, tw), frame_col, dtype=int)
    shape_info = []
    for shape, pls in hole_placements:
        area = _shape_area(shape)
        fill_col = int(area_colors[area])
        all_pos = []
        for positions in pls:
            for pr, pc in positions:
                paste(output, np.full(shape.shape, fill_col, dtype=int), pr, pc, mask=shape)
                all_pos.append((pr, pc))
        shape_info.append((shape, area, fill_col, all_pos))
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


def _scatter_sprites(inp, shape_info, bg, frame_col):
    placement = inp.copy()
    other_pool = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]

    area_groups: dict[int, dict[str, object]] = {}
    for shape, area, fill_color, _ in shape_info:
        bucket = area_groups.setdefault(area, {"fill_color": fill_color, "shapes": []})
        bucket["shapes"].append(shape)

    shared_areas = [area for area, info in area_groups.items() if len(info["shapes"]) >= 2]
    assert shared_areas
    witness_area = random.choice(shared_areas)

    for area, info in area_groups.items():
        fill_color = int(info["fill_color"])
        shapes = list(info["shapes"])
        orientations = {id(shape): _all_orientations(shape) for shape in shapes}
        losers = [c for c in other_pool if c != fill_color]

        winner_counts: dict[int, int] = {}
        witness_shape_idx = None
        if area == witness_area and len(shapes) >= 2 and losers:
            witness_shape_idx = random.randrange(len(shapes))
            for idx in range(len(shapes)):
                if idx == witness_shape_idx:
                    winner_counts[idx] = random.randint(1, 2)
                else:
                    winner_counts[idx] = random.randint(3, WINNING_SPRITE_RANGE[1])
        else:
            for idx in range(len(shapes)):
                winner_counts[idx] = random.randint(*WINNING_SPRITE_RANGE)

        for idx, shape in enumerate(shapes):
            oris = orientations[id(shape)]
            for _ in range(winner_counts[idx]):
                if not _place_sprite(inp, placement, random.choice(oris), fill_color, bg):
                    return False

        n_other = random.randint(1, min(OTHER_COLOR_MAX, len(losers))) if losers else 0
        oc_list = random.sample(losers, n_other)

        if area == witness_area and witness_shape_idx is not None and oc_list:
            witness_color = oc_list[0]
            witness_shape = shapes[witness_shape_idx]
            witness_oris = orientations[id(witness_shape)]
            n_witness = winner_counts[witness_shape_idx] + random.randint(1, 2)
            total_winners = sum(winner_counts.values())
            if n_witness >= total_winners:
                n_witness = total_winners - 1
            if n_witness <= winner_counts[witness_shape_idx]:
                return False
            for _ in range(n_witness):
                if not _place_sprite(inp, placement, random.choice(witness_oris), witness_color, bg):
                    return False
            oc_list = oc_list[1:]

        for oc in oc_list:
            n_oc = random.randint(1, max(1, sum(winner_counts.values()) - 1))
            for _ in range(n_oc):
                shape = random.choice(shapes)
                if not _place_sprite(inp, placement, random.choice(orientations[id(shape)]), oc, bg):
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
