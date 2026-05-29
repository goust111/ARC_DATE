# Combo variant of 67e490f4 (index 14)
# combo_spec: D2/0+D12/0+D18/1
# Name: 严格方向+干扰+多联骨牌
#
# Design analysis for the fused variant
# - Concrete behavior and trace:
#   The frame still acts as the destination template, but every true hole shape is now a
#   triomino, tetromino, or pentomino. Matching is stricter than the source task: a hole
#   with exact bitmap [[1,0],[1,1]] only accepts sprites with that exact bitmap, not a
#   rotated or mirrored version. The outside region also contains 3-5 distractor sprites
#   derived from real shapes by adding or deleting exactly one cell. On a small trace,
#   two holes may belong to the same polyomino family but face differently and receive
#   different output colors; the relaxed source-task solver would merge them, while this
#   fused solver keeps them separate and ignores near-miss distractors entirely.
# - Complexity calibration:
#   The generator always includes at least one polyomino family that appears in two exact
#   orientations with different output colors, so "direction matters" is visible. It also
#   inserts several one-cell near misses, but keeps the distractor count bounded at 3-5
#   and uses only 3-5 oriented hole groups overall, so the puzzle is harder without
#   becoming visually chaotic or ambiguous.
# - Assumption audit against the old main():
#   Step 1 still assumes the unique largest monochrome non-background component is the
#   frame; generation preserves that. Step 2 already extracts exact hole masks, which is
#   now the real key instead of a canonicalized family. Step 3 in the source solver
#   expanded each sprite to all rotation/flip keys; that assumption is false here, so the
#   fused solver stores votes only under the sprite's native bitmap. Step 4 assumes every
#   queried exact hole key has at least one witness sprite; the generator enforces this
#   before any distractors are added.
# - Placement legality:
#   Holes remain inside the frame border, mutually 4-disconnected, and sprites are placed
#   outside the frame bbox with Chebyshev padding 1 via `random_free_location`. Near-miss
#   distractors are trimmed, connected polyominoes and are filtered so they do not share
#   a rotation/flip family with any real hole.
# - Visual observability:
#   A human can compare the cropped holes with the outside sprites and see two signals:
#   some true keys are the "same kind" of polyomino but with different facing, and some
#   distractors differ by only one cell. Across a few examples this makes the combined
#   rule inferable: orientation is part of the key, and approximate matches do nothing.
# - Visual cleanliness:
#   The outside field is denser than the source task but still readable because every
#   sprite is a separate monochrome component with background gaps. The frame interior is
#   unchanged except for the holes, so no region mixes multiple overlaid encodings.
# - Color inferability:
#   Output colors are assigned per exact oriented hole group, not per free polyomino
#   family. Winning sprite counts are kept strictly above any losing-color counts, so the
#   intended color is visible from repetition rather than hidden tie-breaking.
# - Behavioral verification target:
#   The generator rejects instances unless there is at least one rotation/flip family with
#   two distinct exact hole orientations and two distinct fill colors, unless all 3-5
#   non-matching sprites are edit-distance-1 near misses, and unless a relaxed
#   rotation/flip-allowed solver disagrees with the strict output.

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


MaskKey = tuple[tuple[int, int], bytes]


def _mask_key(mask: np.ndarray) -> MaskKey:
    m = mask.astype(bool)
    return (m.shape, m.tobytes())


def _all_rot_flip(mask: np.ndarray):
    """Yield all distinct rotation/flip variants, including mirrored forms."""
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield _mask_key(m2), m2


def _all_mask_keys(mask: np.ndarray):
    seen: set[MaskKey] = set()
    for key, _ in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            yield key


def _family_key(mask: np.ndarray) -> MaskKey:
    """Canonical key under rotation/flip, used only by the generator/analyzer."""
    return min(key for key, _ in _all_rot_flip(mask))


def _all_exact_orientations(mask: np.ndarray) -> list[np.ndarray]:
    seen: set[MaskKey] = set()
    result: list[np.ndarray] = []
    for key, oriented in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            result.append(oriented)
    return result


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _shape_edit_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Minimum XOR cell count between `a` and any orientation of `b`."""
    aa = a.astype(bool)
    best = None
    for bb in _all_exact_orientations(b):
        h = max(aa.shape[0], bb.shape[0])
        w = max(aa.shape[1], bb.shape[1])
        pa = np.zeros((h, w), dtype=bool)
        pb = np.zeros((h, w), dtype=bool)
        pa[: aa.shape[0], : aa.shape[1]] = aa
        pb[: bb.shape[0], : bb.shape[1]] = bb
        dist = int(np.logical_xor(pa, pb).sum())
        if best is None or dist < best:
            best = dist
    assert best is not None
    return best


def _extract_frame(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert comps
    frame_area = max(int(comp["area"]) for comp in comps)
    frame_candidates = [comp for comp in comps if int(comp["area"]) == frame_area]
    assert len(frame_candidates) == 1
    frame = frame_candidates[0]
    return int(frame["color"]), tuple(map(int, frame["bbox"]))


def _extract_holes(sub: np.ndarray, frame_col: int):
    hole_mask = sub != int(frame_col)
    holes = []
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        mask = np.zeros((hh, ww), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        holes.append((_mask_key(mask), (int(hr0), int(hc0)), mask))
    return holes


def _extract_sprite_records(g: np.ndarray, frame_bbox, bg: int):
    r0, c0, r1, c1 = map(int, frame_bbox)
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    records = []
    for comp in comps:
        coords = comp["coords"]
        sr0, sc0, sr1, sc1 = map(int, comp["bbox"])
        hh = int(sr1 - sr0 + 1)
        ww = int(sc1 - sc0 + 1)
        vals = [int(v) for v in g[coords[:, 0], coords[:, 1]]]
        color = int(Counter(vals).most_common(1)[0][0])
        rel = coords - np.array([sr0, sc0])
        mask = np.zeros((hh, ww), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        records.append({"color": color, "mask": mask})
    return records


def _solve_relaxed(grid: np.ndarray) -> np.ndarray:
    """Source-task style solver used only for behavioral verification."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frame_col, fb = _extract_frame(g, bg)
    sub = crop_bbox(g, fb)
    th, tw = sub.shape
    holes = _extract_holes(sub, frame_col)
    sprites = _extract_sprite_records(g, fb, bg)

    sprites_by_key: dict[MaskKey, dict[int, int]] = {}
    for rec in sprites:
        for key in _all_mask_keys(rec["mask"]):
            bucket = sprites_by_key.setdefault(key, {})
            color = int(rec["color"])
            bucket[color] = int(bucket.get(color, 0)) + 1

    out = np.full((th, tw), int(frame_col), dtype=int)
    for key, (hr0, hc0), hmask in holes:
        cmap = sprites_by_key.get(key)
        assert cmap is not None
        scol = max(cmap.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        paste(out, np.full(hmask.shape, int(scol), dtype=int), int(hr0), int(hc0), mask=hmask)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the cropped frame by matching holes to strictly oriented polyomino sprites.

    Input structure:
    - The background is the most common color, and one large monochrome component is the frame.
    - Inside the frame bbox are several holes carved from the frame color; every true hole is a
      triomino, tetromino, or pentomino.
    - Outside the frame bbox are many small sprites on the background, including exact witnesses,
      losing-color copies, and several one-cell near-miss distractors.

    Rule:
    - Crop to the frame bbox and start from a solid rectangle of the frame color.
    - For each hole, only count outside sprites whose exact bitmap matches the hole as drawn;
      rotated, mirrored, or one-cell-off shapes do not match.
    - Fill each hole with the most frequent color among those exact witnesses.

    Key insight:
    - The outside sprites are still shape-coded color keys, but the code now combines two strict
      filters: direction is part of the key, and almost-correct polyominoes are decoys.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frame_col, fb = _extract_frame(g, bg)
    sub = crop_bbox(g, fb)
    th, tw = sub.shape

    holes = _extract_holes(sub, frame_col)
    sprites = _extract_sprite_records(g, fb, bg)

    sprites_by_key: dict[MaskKey, dict[int, int]] = {}
    for rec in sprites:
        key = _mask_key(rec["mask"])
        bucket = sprites_by_key.setdefault(key, {})
        color = int(rec["color"])
        bucket[color] = int(bucket.get(color, 0)) + 1

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

FRAME_H_RANGE = (8, 13)
FRAME_W_RANGE = (8, 18)

SHAPE_COUNT_RANGE = (3, 5)  # number of oriented hole groups
MIN_SHAPES = 3
SHAPE_MIN_AREA = 3
SHAPE_MAX_AREA = 5
FRAME_DENSITY_RANGE = (0.58, 0.90)
MIN_TOTAL_HOLES = 6
PLACEMENT_RETRIES = 90
MULT_EVEN_WEIGHTS = ([1, 2], [4, 1])
MULT_ODD_WEIGHTS = ([1, 2, 3], [3, 2, 1])

WINNING_SPRITE_RANGE = (2, 4)
PAIRED_DOMINANT_RANGE = (5, 7)
PAIRED_MINOR_RANGE = (2, 3)
OTHER_COLOR_MAX = 2
SPRITE_PADDING = 1
DISTRACTOR_RANGE = (3, 5)
DISTRACTOR_AREA_RANGE = (2, 6)

MAX_GEN_ATTEMPTS = 700


def _build_polyomino_library() -> list[np.ndarray]:
    raw = [
        [[1, 1, 1]],  # I3
        [[1, 0], [1, 1]],  # L3
        [[1, 1, 1, 1]],  # I4
        [[1, 1], [1, 1]],  # O4
        [[1, 1, 1], [0, 1, 0]],  # T4
        [[1, 0, 0], [1, 1, 1]],  # L4
        [[1, 1, 0], [0, 1, 1]],  # S4
        [[1, 1, 1, 1, 1]],  # I5
        [[1, 0, 0, 0], [1, 1, 1, 1]],  # L5
        [[1, 1], [1, 1], [1, 0]],  # P5
        [[1, 1, 1], [0, 1, 0], [0, 1, 0]],  # T5
        [[1, 0, 1], [1, 1, 1]],  # U5
        [[1, 0, 0], [1, 0, 0], [1, 1, 1]],  # V5
        [[1, 0, 0], [1, 1, 0], [0, 1, 1]],  # W5
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],  # X5
        [[0, 1], [1, 1], [0, 1], [0, 1]],  # Y5
        [[1, 1, 0], [0, 1, 0], [0, 1, 1]],  # Z5
        [[0, 1, 1], [1, 1, 0], [0, 1, 0]],  # F5
        [[1, 1, 0, 0], [0, 1, 1, 1]],  # N5
    ]
    seen: set[MaskKey] = set()
    library: list[np.ndarray] = []
    for arr in raw:
        mask = np.array(arr, dtype=bool)
        area = int(mask.sum())
        key = _family_key(mask)
        if key in seen:
            continue
        if not (SHAPE_MIN_AREA <= area <= SHAPE_MAX_AREA):
            continue
        seen.add(key)
        library.append(mask)
    return library


POLYOMINO_LIBRARY = _build_polyomino_library()


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
    r_max = th // 2 - sh
    c_max = tw // 2 - sw
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
        for oriented in _all_exact_orientations(shape)
        if oriented.shape[0] <= max_sh and oriented.shape[1] <= max_sw and SHAPE_MIN_AREA <= int(oriented.sum()) <= SHAPE_MAX_AREA
    ]


def _select_oriented_groups(th: int, tw: int):
    max_sh = max(1, (th - 2) // 2)
    max_sw = max(1, (tw - 2) // 2)

    families = []
    for base in POLYOMINO_LIBRARY:
        orientations = _fitting_orientations(base, max_sh, max_sw)
        if orientations:
            families.append({"family": _family_key(base), "orientations": orientations})

    paired_candidates = [fam for fam in families if len(fam["orientations"]) >= 2]
    if not paired_candidates:
        return None

    min_groups = max(MIN_SHAPES, SHAPE_COUNT_RANGE[0])
    max_groups = min(SHAPE_COUNT_RANGE[1], len(families) + 1)
    if max_groups < min_groups:
        return None

    paired_family = random.choice(paired_candidates)
    paired_shapes = random.sample(paired_family["orientations"], 2)
    groups = [
        {"family": paired_family["family"], "shape": paired_shapes[0], "paired": True},
        {"family": paired_family["family"], "shape": paired_shapes[1], "paired": True},
    ]

    remaining = [fam for fam in families if fam["family"] != paired_family["family"]]
    random.shuffle(remaining)
    target_groups = random.randint(min_groups, max_groups)
    extras_needed = target_groups - len(groups)
    if len(remaining) < extras_needed:
        return None

    for fam in remaining[:extras_needed]:
        groups.append({"family": fam["family"], "shape": random.choice(fam["orientations"]), "paired": False})
    random.shuffle(groups)
    return groups


def _build_hole_layout(th: int, tw: int):
    groups = _select_oriented_groups(th, tw)
    if groups is None:
        return None

    occ = np.ones((th, tw), dtype=bool)
    placed_groups = []
    even_dims = th % 2 == 0 or tw % 2 == 0
    mult_vals, mult_wts = MULT_EVEN_WEIGHTS if even_dims else MULT_ODD_WEIGHTS

    for group in groups:
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
                "paired": bool(group["paired"]),
                "placements": placements,
            }
        )

    frame_density = 1.0 - int(np.sum(~occ)) / float(th * tw)
    if not (FRAME_DENSITY_RANGE[0] <= frame_density <= FRAME_DENSITY_RANGE[1]):
        return None

    total_holes = sum(len(positions) for group in placed_groups for positions in group["placements"])
    if total_holes < MIN_TOTAL_HOLES:
        return None

    paired_groups = [group for group in placed_groups if group["paired"]]
    if len(paired_groups) < 2:
        return None
    return occ, placed_groups


def _build_output(occ: np.ndarray, placed_groups: list[dict], frame_col: int, bg: int):
    th, tw = occ.shape
    avail_colors = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    if len(avail_colors) < len(placed_groups):
        return None

    chosen_colors = random.sample(avail_colors, len(placed_groups))
    output = np.full((th, tw), int(frame_col), dtype=int)
    shape_info = []
    for group, fill_color in zip(placed_groups, chosen_colors):
        shape = group["shape"]
        all_pos = []
        for positions in group["placements"]:
            for pr, pc in positions:
                paste(output, np.full(shape.shape, int(fill_color), dtype=int), pr, pc, mask=shape)
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

    family_to_colors: dict[MaskKey, set[int]] = {}
    family_to_keys: dict[MaskKey, set[MaskKey]] = {}
    for info in shape_info:
        family = info["family"]
        family_to_colors.setdefault(family, set()).add(int(info["fill_color"]))
        family_to_keys.setdefault(family, set()).add(_mask_key(info["shape"]))
    if not any(len(family_to_keys[f]) >= 2 and len(family_to_colors[f]) >= 2 for f in family_to_keys):
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


def _near_miss_variants(mask: np.ndarray) -> list[np.ndarray]:
    base = mask.astype(bool)
    h, w = base.shape
    out: dict[MaskKey, np.ndarray] = {}

    for rr, cc in np.argwhere(base):
        cand = base.copy()
        cand[rr, cc] = False
        if not cand.any():
            continue
        if len(components_mask(cand, connectivity=4)) != 1:
            continue
        trimmed = _trim_mask(cand)
        area = int(trimmed.sum())
        if DISTRACTOR_AREA_RANGE[0] <= area <= DISTRACTOR_AREA_RANGE[1]:
            out[_mask_key(trimmed)] = trimmed

    ext = np.zeros((h + 2, w + 2), dtype=bool)
    ext[1 : 1 + h, 1 : 1 + w] = base
    frontier = set()
    for rr, cc in np.argwhere(ext):
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if 0 <= nr < ext.shape[0] and 0 <= nc < ext.shape[1] and not ext[nr, nc]:
                frontier.add((nr, nc))
    for rr, cc in frontier:
        cand = ext.copy()
        cand[rr, cc] = True
        trimmed = _trim_mask(cand)
        area = int(trimmed.sum())
        if DISTRACTOR_AREA_RANGE[0] <= area <= DISTRACTOR_AREA_RANGE[1]:
            out[_mask_key(trimmed)] = trimmed

    out.pop(_mask_key(base), None)
    return list(out.values())


def _choose_distractor_shapes(shape_info: list[dict]) -> list[np.ndarray] | None:
    hole_exact_keys = {_mask_key(info["shape"]) for info in shape_info}
    hole_family_keys = {info["family"] for info in shape_info}

    per_base: list[list[np.ndarray]] = []
    all_unique: dict[MaskKey, np.ndarray] = {}
    used_families: dict[MaskKey, np.ndarray] = {}

    for info in shape_info:
        local = []
        local_families = set()
        for cand in _near_miss_variants(info["shape"]):
            cand_key = _mask_key(cand)
            fam_key = _family_key(cand)
            if cand_key in hole_exact_keys or fam_key in hole_family_keys or fam_key in local_families:
                continue
            local_families.add(fam_key)
            local.append(cand)
            used_families[fam_key] = cand
            all_unique[cand_key] = cand
        if local:
            random.shuffle(local)
            per_base.append(local)

    min_needed = DISTRACTOR_RANGE[0]
    if len(used_families) < min_needed:
        return None

    max_possible = min(DISTRACTOR_RANGE[1], len(used_families))
    target = random.randint(min_needed, max_possible)
    selected: list[np.ndarray] = []
    used_family_keys: set[MaskKey] = set()

    while len(selected) < target:
        progress = False
        order = list(range(len(per_base)))
        random.shuffle(order)
        for idx in order:
            while per_base[idx] and _family_key(per_base[idx][-1]) in used_family_keys:
                per_base[idx].pop()
            if not per_base[idx]:
                continue
            cand = per_base[idx].pop()
            fam_key = _family_key(cand)
            if fam_key in used_family_keys:
                continue
            selected.append(cand)
            used_family_keys.add(fam_key)
            progress = True
            if len(selected) >= target:
                break
        if not progress:
            break

    if len(selected) < target:
        leftovers = list(used_families.items())
        random.shuffle(leftovers)
        for fam_key, cand in leftovers:
            if fam_key in used_family_keys:
                continue
            selected.append(cand)
            used_family_keys.add(fam_key)
            if len(selected) >= target:
                break

    return selected if len(selected) >= min_needed else None


def _scatter_sprites(inp: np.ndarray, shape_info: list[dict], bg: int, frame_col: int) -> bool:
    placement = inp.copy()
    other_pool = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]

    family_to_infos: dict[MaskKey, list[dict]] = {}
    for info in shape_info:
        family_to_infos.setdefault(info["family"], []).append(info)

    paired_families = [family for family, infos in family_to_infos.items() if len(infos) >= 2]
    if not paired_families:
        return False
    paired_family = paired_families[0]
    dominant_idx = random.randrange(len(family_to_infos[paired_family]))

    for family, infos in family_to_infos.items():
        family_fill_colors = {int(info["fill_color"]) for info in infos}
        for idx, info in enumerate(infos):
            fill_color = int(info["fill_color"])
            if family == paired_family and idx == dominant_idx:
                n_winning = random.randint(*PAIRED_DOMINANT_RANGE)
            elif family == paired_family:
                n_winning = random.randint(*PAIRED_MINOR_RANGE)
            else:
                n_winning = random.randint(*WINNING_SPRITE_RANGE)

            for _ in range(n_winning):
                if not _place_sprite(inp, placement, info["shape"], fill_color, bg):
                    return False

            losers = [c for c in other_pool if c != fill_color and c not in family_fill_colors]
            max_other_colors = min(OTHER_COLOR_MAX, len(losers))
            n_other = random.randint(1, max_other_colors) if max_other_colors > 0 else 0
            for oc in random.sample(losers, n_other):
                limit = n_winning - 1
                if family == paired_family and idx == dominant_idx:
                    limit = n_winning - 2
                limit = max(1, limit)
                n_oc = random.randint(1, limit)
                n_oc = min(n_oc, limit)
                for _ in range(n_oc):
                    if not _place_sprite(inp, placement, info["shape"], int(oc), bg):
                        break

    distractor_shapes = _choose_distractor_shapes(shape_info)
    if distractor_shapes is None:
        return False
    distractor_colors = [info["fill_color"] for info in shape_info] + other_pool
    for shape in distractor_shapes:
        if not _place_sprite(inp, placement, shape, random.choice(distractor_colors), bg):
            return False
    return True


def _analyze_instance(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frame_col, fb = _extract_frame(g, bg)
    sub = crop_bbox(g, fb)
    holes = _extract_holes(sub, frame_col)
    strict_out = main(g)
    relaxed_out = _solve_relaxed(g)

    hole_exact_keys = set()
    hole_family_keys = set()
    hole_family_to_keys: dict[MaskKey, set[MaskKey]] = {}
    hole_family_to_colors: dict[MaskKey, set[int]] = {}
    real_shapes = []
    for key, (hr0, hc0), mask in holes:
        hole_exact_keys.add(key)
        family = _family_key(mask)
        hole_family_keys.add(family)
        hole_family_to_keys.setdefault(family, set()).add(key)
        color = int(strict_out[hr0 : hr0 + mask.shape[0], hc0 : hc0 + mask.shape[1]][mask][0])
        hole_family_to_colors.setdefault(family, set()).add(color)
        real_shapes.append(mask)

    sprites = _extract_sprite_records(g, fb, bg)
    distractor_count = 0
    near_miss_count = 0
    exact_match_sprite_count = 0

    for rec in sprites:
        key = _mask_key(rec["mask"])
        if key in hole_exact_keys:
            exact_match_sprite_count += 1
            continue
        distractor_count += 1
        dist = min(_shape_edit_distance(rec["mask"], shape) for shape in real_shapes)
        if dist == 1:
            near_miss_count += 1

    orientation_color_split_count = sum(
        1
        for family in hole_family_to_keys
        if len(hole_family_to_keys[family]) >= 2 and len(hole_family_to_colors[family]) >= 2
    )

    return {
        "distractor_count": distractor_count,
        "near_miss_count": near_miss_count,
        "exact_match_sprite_count": exact_match_sprite_count,
        "orientation_color_split_count": orientation_color_split_count,
        "relaxed_matches_strict": bool(np.array_equal(relaxed_out, strict_out)),
    }


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
        if not np.array_equal(main(inp), output):
            return None
        stats = _analyze_instance(inp)
        if not (DISTRACTOR_RANGE[0] <= stats["distractor_count"] <= DISTRACTOR_RANGE[1]):
            return None
        if stats["near_miss_count"] != stats["distractor_count"]:
            return None
        if stats["orientation_color_split_count"] < 1:
            return None
        if stats["relaxed_matches_strict"]:
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {MAX_GEN_ATTEMPTS} attempts")
