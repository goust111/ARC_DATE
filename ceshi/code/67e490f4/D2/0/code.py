# Variant of 67e490f4
# Dimension: D2 — 匹配条件：旋转/翻转的允许范围 (match_cond)
# Change: rotation — 匹配时允许 8 种旋转/翻转变体（0/90/180/270度旋转 x 是否垂直翻转）... → 禁止旋转和翻转（严格形状匹配）
#
# Design analysis for this variant
# - Source-task check: this D2 variant has no separate source_task, so the baseline is the original
#   solver/generator only.
# - Concrete behavior:
#   The frame still contains background-colored holes and the outside region still contains small
#   sprites. The changed behavior is that a hole is keyed by its exact boolean bitmap, including its
#   facing. Example: an L-triomino hole shaped [[1,0],[1,1]] only matches outside sprites with that
#   exact mask. A rotated sprite [[1,1],[1,0]] used to count under the original rule, but now it is
#   a different key and must not influence this hole.
# - Flow difference on a small trace:
#   Original logic built one bucket for all 8 rotation/flip variants of a sprite, so two hole
#   orientations from the same canonical family were collapsed into one color vote. In the variant,
#   sprite votes are stored only under the sprite's native mask key. If the frame contains both an
#   upright L-hole and a rotated L-hole, they can now legitimately receive different colors.
# - Complexity calibration:
#   If the generator kept many symmetric shapes, removing rotation/flip would be visually invisible.
#   To make the D2 change observable without changing the task family, the generator now favors
#   asymmetric shape families and always includes at least one canonical family that appears in two
#   different hole orientations with different output colors. This keeps the task focused on shape
#   matching while making orientation materially relevant.
# - Assumption audit against the old main():
#   Step 1 (largest monochrome component is the frame) still holds unchanged.
#   Step 2 already extracted exact hole masks, so no change is needed there.
#   Step 3 in the old solver depended on the premise that a sprite may vote for every rotated/flipped
#   version of itself; that premise is false in this variant, so the `_all_mask_keys(...)` expansion
#   must be removed and replaced by one exact key per sprite.
#   Step 4 assumes every hole key has at least one sprite vote; the generator must now guarantee an
#   exact-orientation witness for every hole orientation, not just for its canonical family.
# - Placement legality:
#   Holes remain strictly inside the frame border, mutually 4-disconnected, and sprites are still
#   scattered outside the frame bbox with Chebyshev padding 1. Every oriented hole group receives
#   at least one exact-match sprite before any distractors are added.
# - Visual observability:
#   Each sample contains a paired canonical family with two different hole orientations. Those two
#   orientations are given different colors in the output, while the outside sprites show both
#   orientations as separate color-coded populations. This makes "direction is part of the key"
#   inferable from a few examples.
# - Cleanliness and color inference:
#   Output colors are assigned per oriented hole group from distinct visible colors. The generator
#   avoids ties in sprite-frequency voting, so the result does not depend on invisible color-id tie
#   breaking even though the solver keeps the original fallback tie rule for robustness.
# - New implicit assumptions introduced by the variant:
#   `main()` now assumes exact orientation coverage for every hole key and that different oriented
#   groups are not accidentally duplicated with conflicting target colors. `generate_input()` enforces
#   both by selecting unique exact masks per group and by scattering winning exact-match sprites for
#   each one.

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


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    m = mask.astype(bool)
    return (m.shape, m.tobytes())


def _all_rot_flip(mask: np.ndarray):
    """Yield (key, oriented_mask) for all 8 rotation/flip variants of a bool mask."""
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield _mask_key(m2), m2


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    """Canonical representative under rotation/flip, used only by the generator."""
    return min(key for key, _ in _all_rot_flip(mask))


def _all_orientations(mask: np.ndarray) -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    result: list[np.ndarray] = []
    for key, oriented in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            result.append(oriented)
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill holes inside a large template frame using strictly oriented outside sprites.

    Input structure:
    - The background is the most common color, with one large monochrome frame object.
    - Inside the frame bbox are several hole components carved out of the frame color.
    - Outside the frame bbox are small sprites on the background, usually monochrome.

    Rule:
    - Crop to the frame's bounding box and begin with a solid rectangle of the frame color.
    - For each hole, only count outside sprites whose bitmap matches the hole exactly as drawn.
    - Fill the hole with the most frequent color among those exact matches.

    Key insight:
    - Shape identity now includes direction: a rotated or flipped look-alike sprite is a different
      key and cannot color a hole unless its pixels line up exactly.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Find the template frame: largest connected monochrome non-bg component.
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

    # STEP 2) Extract hole components inside the bbox, keyed by exact mask orientation.
    hole_mask = (sub != int(frame_col))
    holes: list[tuple[tuple[tuple[int, int], bytes], tuple[int, int], np.ndarray]] = []
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        mask = np.zeros((hh, ww), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        holes.append((_mask_key(mask), (int(hr0), int(hc0)), mask))

    # STEP 3) Extract outside sprites and vote only under their native orientation.
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites_by_key: dict[tuple[tuple[int, int], bytes], dict[int, int]] = {}
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
        bucket = sprites_by_key.setdefault(_mask_key(mask), {})
        bucket[scol] = int(bucket.get(scol, 0)) + 1

    # STEP 4) Fill each hole from the exact-match color vote.
    out = np.full((th, tw), int(frame_col), dtype=int)
    for key, (hr0, hc0), hmask in holes:
        cmap = sprites_by_key.get(key)
        assert cmap is not None
        scol = max(cmap.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        paste(out, np.full(hmask.shape, int(scol), dtype=int), int(hr0), int(hc0), mask=hmask)

    return out


# --------------- generator ---------------

NUM_COLORS = 10
GRID_SIZE = 30

FRAME_H_RANGE = (7, 13)
FRAME_W_RANGE = (7, 20)

SHAPE_COUNT_RANGE = (3, 6)          # number of oriented hole groups
MIN_SHAPES = 3
SHAPE_MAX_AREA = 8
FRAME_DENSITY_RANGE = (0.55, 0.92)
MIN_TOTAL_HOLES = 5
PLACEMENT_RETRIES = 90
MULT_EVEN_WEIGHTS = ([1, 2], [4, 1])
MULT_ODD_WEIGHTS = ([1, 2, 3], [3, 2, 1])

WINNING_SPRITE_RANGE = (2, 4)
DOMINANT_BONUS_RANGE = (2, 3)       # extra exact-match sprites for one orientation in the paired family
OTHER_COLOR_MAX = 2
ALT_ORIENTATION_DECOYS = (1, 2)
SPRITE_PADDING = 1

MAX_GEN_ATTEMPTS = 400


def _build_shape_library() -> list[np.ndarray]:
    raw = [
        [[1, 0], [1, 1]],
        [[1, 0], [1, 0], [1, 1]],
        [[1, 1, 1], [1, 0, 0]],
        [[1, 1, 1], [0, 1, 0], [0, 1, 0]],
        [[1, 0], [1, 1], [0, 1]],
        [[1, 1, 0], [0, 1, 1]],
        [[1, 1, 1], [0, 0, 1]],
        [[1, 0, 0], [1, 1, 1], [0, 0, 1]],
        [[1, 1, 1], [1, 0, 1]],
        [[1, 1, 0], [1, 0, 0], [1, 1, 1]],
    ]
    seen: set[tuple[tuple[int, int], bytes]] = set()
    dedup: list[np.ndarray] = []
    for arr in raw:
        mask = np.array(arr, dtype=bool)
        key = _canonical_key(mask)
        if key not in seen:
            seen.add(key)
            if len(_all_orientations(mask)) > 1 and int(np.sum(mask)) <= SHAPE_MAX_AREA:
                dedup.append(mask)
    return dedup


SHAPE_LIBRARY = _build_shape_library()


def _mirror_pos(r: int, c: int, sh: int, sw: int, th: int, tw: int) -> list[tuple[int, int]]:
    return list(
        {
            (r, c),
            (r, tw - sw - c),
            (th - sh - r, c),
            (th - sh - r, tw - sw - c),
        }
    )


def _validate_positions(
    occ: np.ndarray,
    shape: np.ndarray,
    positions: list[tuple[int, int]],
    th: int,
    tw: int,
) -> list[tuple[int, int]] | None:
    sh, sw = shape.shape
    new_cells: set[tuple[int, int]] = set()
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


def _try_place_quadrant(occ: np.ndarray, shape: np.ndarray, th: int, tw: int):
    sh, sw = shape.shape
    r_max, c_max = th // 2 - sh, tw // 2 - sw
    if r_max < 1 or c_max < 1:
        return None
    r = random.randint(1, r_max)
    c = random.randint(1, c_max)
    return _validate_positions(occ, shape, _mirror_pos(r, c, sh, sw, th, tw), th, tw)


def _try_place_hcenter(occ: np.ndarray, shape: np.ndarray, th: int, tw: int):
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
    return _validate_positions(occ, shape, list({(r, c), (r, tw - sw - c)}), th, tw)


def _try_place_vcenter(occ: np.ndarray, shape: np.ndarray, th: int, tw: int):
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
    return _validate_positions(occ, shape, list({(r, c), (th - sh - r, c)}), th, tw)


def _try_place_center(occ: np.ndarray, shape: np.ndarray, th: int, tw: int):
    if th % 2 == 0 or tw % 2 == 0:
        return None
    sh, sw = shape.shape
    r = th // 2 - sh // 2
    c = tw // 2 - sw // 2
    if r < 1 or c < 1 or r + sh > th - 1 or c + sw > tw - 1:
        return None
    return _validate_positions(occ, shape, [(r, c)], th, tw)


def _try_place_symmetric(occ: np.ndarray, shape: np.ndarray, th: int, tw: int):
    placers = [_try_place_quadrant]
    if th % 2 == 1:
        placers.append(_try_place_hcenter)
    if tw % 2 == 1:
        placers.append(_try_place_vcenter)
    if th % 2 == 1 and tw % 2 == 1:
        placers.append(_try_place_center)
    return random.choice(placers)(occ, shape, th, tw)


def _carve_hole(occ: np.ndarray, shape: np.ndarray, positions: list[tuple[int, int]]) -> None:
    sh, sw = shape.shape
    for pr, pc in positions:
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    occ[pr + dr, pc + dc] = False


def _fitting_orientations(shape: np.ndarray, max_sh: int, max_sw: int) -> list[np.ndarray]:
    return [
        oriented
        for oriented in _all_orientations(shape)
        if oriented.shape[0] <= max_sh and oriented.shape[1] <= max_sw and int(np.sum(oriented)) <= SHAPE_MAX_AREA
    ]


def _select_oriented_groups(th: int, tw: int):
    max_sh = max(1, (th - 2) // 2)
    max_sw = max(1, (tw - 2) // 2)

    families = []
    for base in SHAPE_LIBRARY:
        orientations = _fitting_orientations(base, max_sh, max_sw)
        if orientations:
            families.append((_canonical_key(base), orientations))

    paired_candidates = [(fk, oris) for fk, oris in families if len(oris) >= 2]
    if not paired_candidates:
        return None

    min_groups = max(MIN_SHAPES, SHAPE_COUNT_RANGE[0])
    max_groups = min(SHAPE_COUNT_RANGE[1], len(families) + 1)
    if max_groups < min_groups:
        return None
    target_groups = random.randint(min_groups, max_groups)
    paired_key, paired_oris = random.choice(paired_candidates)
    paired_pick = random.sample(paired_oris, 2)
    groups = [{"family": paired_key, "shape": shape, "paired": True} for shape in paired_pick]

    remaining = [(fk, oris) for fk, oris in families if fk != paired_key]
    random.shuffle(remaining)
    extras_needed = target_groups - len(groups)
    if len(remaining) < extras_needed:
        return None
    for fk, oris in remaining[:extras_needed]:
        groups.append({"family": fk, "shape": random.choice(oris), "paired": False})

    return groups


def _build_hole_layout(th: int, tw: int):
    oriented_groups = _select_oriented_groups(th, tw)
    if oriented_groups is None or len(oriented_groups) < MIN_SHAPES:
        return None

    occ = np.ones((th, tw), dtype=bool)
    placed_groups = []
    mult_vals, mult_wts = MULT_EVEN_WEIGHTS if (th % 2 == 0 or tw % 2 == 0) else MULT_ODD_WEIGHTS

    for group in oriented_groups:
        shape = group["shape"]
        mult = random.choices(mult_vals, weights=mult_wts)[0]
        placements = []
        for _ in range(mult):
            for _ in range(PLACEMENT_RETRIES):
                positions = _try_place_symmetric(occ, shape, th, tw)
                if positions is not None:
                    _carve_hole(occ, shape, positions)
                    placements.append(positions)
                    break
        if not placements:
            return None
        placed_groups.append(
            {
                "family": group["family"],
                "shape": shape,
                "paired": group["paired"],
                "placements": placements,
            }
        )

    frame_density = 1.0 - int(np.sum(~occ)) / float(th * tw)
    if not (FRAME_DENSITY_RANGE[0] <= frame_density <= FRAME_DENSITY_RANGE[1]):
        return None

    total_holes = sum(len(pos) for group in placed_groups for pos in group["placements"])
    if total_holes < MIN_TOTAL_HOLES:
        return None

    paired_count = sum(1 for group in placed_groups if group["paired"])
    if paired_count < 2:
        return None

    return occ, placed_groups


def _build_output(occ: np.ndarray, placed_groups: list[dict], frame_col: int, bg: int):
    th, tw = occ.shape
    avail_colors = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    if len(avail_colors) < len(placed_groups):
        return None

    colors = random.sample(avail_colors, len(placed_groups))
    output = np.full((th, tw), frame_col, dtype=int)
    shape_info = []

    for group, fill_color in zip(placed_groups, colors):
        all_pos = []
        shape = group["shape"]
        for positions in group["placements"]:
            for pr, pc in positions:
                paste(output, np.full(shape.shape, fill_color, dtype=int), pr, pc, mask=shape)
                all_pos.append((pr, pc))
        shape_info.append(
            {
                "family": group["family"],
                "shape": shape,
                "fill_color": int(fill_color),
                "positions": all_pos,
                "paired": bool(group["paired"]),
            }
        )

    paired_colors = {info["fill_color"] for info in shape_info if info["paired"]}
    if len(paired_colors) < 2:
        return None

    return output, shape_info


def _place_sprite(inp: np.ndarray, placement: np.ndarray, shape: np.ndarray, color: int, bg: int) -> bool:
    patch = np.full(shape.shape, bg, dtype=int)
    patch[shape] = int(color)
    try:
        r, c = random_free_location(placement, patch, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, patch, r, c, transparent=bg)
    paste(inp, patch, r, c, transparent=bg)
    return True


def _scatter_sprites(inp: np.ndarray, shape_info: list[dict], bg: int, frame_col: int) -> bool:
    placement = inp.copy()
    other_pool = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]

    family_to_infos: dict[tuple[tuple[int, int], bytes], list[dict]] = {}
    for info in shape_info:
        family_to_infos.setdefault(info["family"], []).append(info)

    paired_family = None
    for family, infos in family_to_infos.items():
        if len(infos) >= 2:
            paired_family = family
            break
    assert paired_family is not None

    dominant_idx = random.randrange(len(family_to_infos[paired_family]))

    for family, infos in family_to_infos.items():
        family_oris = _all_orientations(infos[0]["shape"])
        used_keys = {_mask_key(info["shape"]) for info in infos}

        for idx, info in enumerate(infos):
            fill_color = int(info["fill_color"])
            base_count = random.randint(*WINNING_SPRITE_RANGE)
            if family == paired_family and idx == dominant_idx:
                base_count += random.randint(*DOMINANT_BONUS_RANGE)

            for _ in range(base_count):
                if not _place_sprite(inp, placement, info["shape"], fill_color, bg):
                    return False

            losers = [c for c in other_pool if c != fill_color]
            n_other = random.randint(1, min(OTHER_COLOR_MAX, len(losers))) if losers else 0
            for oc in random.sample(losers, n_other):
                n_oc = random.randint(1, max(1, base_count - 1))
                n_oc = min(n_oc, base_count - 1)
                for _ in range(n_oc):
                    if not _place_sprite(inp, placement, info["shape"], oc, bg):
                        break

        unused_family_oris = [ori for ori in family_oris if _mask_key(ori) not in used_keys]
        if unused_family_oris:
            decoy_color_choices = [c for c in other_pool if c not in {info["fill_color"] for info in infos}]
            if decoy_color_choices:
                decoy_color = random.choice(decoy_color_choices)
                for _ in range(random.randint(*ALT_ORIENTATION_DECOYS)):
                    if not _place_sprite(inp, placement, random.choice(unused_family_oris), decoy_color, bg):
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
    occ, placed_groups = layout

    built = _build_output(occ, placed_groups, frame_col, bg)
    if built is None:
        return None
    output, shape_info = built

    inp = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    fr = random.randint(0, GRID_SIZE - th)
    fc = random.randint(0, GRID_SIZE - tw)
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
