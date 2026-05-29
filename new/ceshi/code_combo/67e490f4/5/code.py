# Combo variant of 67e490f4 (index 5)
# combo_spec: D11/0+D12/0
# Name: 精确配对+大量干扰

from __future__ import annotations

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

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Original task: a large monochrome frame contains several hole shapes, and the
#      outside region contains many sprites per hole shape. Shape matching is exact up
#      to rotation/flip, but the color comes from the most frequent matching sprite.
#    - D11 changes the correspondence itself: each hole shape appears exactly once in
#      the frame and has exactly one true matching sprite outside, so the color becomes
#      a direct lookup instead of a vote.
#    - D12 changes the outside field: besides the true sprites, add 3-5 distractors
#      that differ from real shapes by exactly one cell, so they look tempting but
#      still fail exact canonical-mask matching.
#    - Small trace:
#      original: one T-hole could see red T, red T, green T outside -> fill red.
#      combo: one T-hole sees exactly one green T outside, plus several shapes such as
#      a T missing one arm cell or a T with one extra side cell -> fill green, and all
#      near-misses must be ignored.
#    - The fused difference is therefore not "fewer sprites" plus "more clutter" as
#      separate edits; the outside sprites now form a strict 1:1 dictionary whose only
#      difficulty is distinguishing the exact key from several almost-correct decoys.
#
# b) Complexity calibration
#    - Using only 1-2 holes would make the fusion visually weak, while too many holes
#      plus many distractors would make the outside field noisy. The generator therefore
#      uses 4-6 unique holes and 3-5 distractors.
#    - Hole shapes are kept small (area <= 8) but drawn from a library rich in
#      asymmetric polyominoes. This keeps exact comparison visible and usually allows at
#      least one sprite/hole orientation mismatch, so rotation/flip equivalence remains
#      observable.
#    - Distractors are edit-distance-1 near-misses derived from real shapes rather than
#      arbitrary blobs. That makes the extra clutter meaningful instead of random.
#
# c) Audit of solver assumptions affected by the fusion
#    - The original solver aggregated `key -> color -> count` and relied on plurality.
#      That assumption is false here: every true hole key must resolve to exactly one
#      outside sprite color, so the fused solver reads a direct `canonical_key -> color`
#      dictionary instead.
#    - The original solver accepted any number of exact duplicates for one shape. The
#      fused rule forbids that, so the new solver asserts unique canonical sprite keys.
#    - The original exact-match logic still matters. D12 does not relax shape matching;
#      it makes that exactness more important because many outside sprites are one-cell
#      edits of real shapes and must be ignored.
#    - Frame detection as the largest monochrome non-background component still holds as
#      long as the generator keeps the frame much larger than every sprite.
#
# d) Placement legality
#    - Every hole is placed fully inside the frame with a one-cell frame border.
#    - Holes are kept 4-disconnected so each hole remains a separate component.
#    - All outside sprites are placed with `random_free_location(..., padding=1)`,
#      which keeps them separated from the frame and from one another by background.
#    - Distractors are trimmed after one-cell edit operations and filtered so they never
#      canonicalize to a real hole key or to another already-chosen distractor key.
#
# e) Visual observability
#    - Across a few examples, a human can see that the number of true matching sprites
#      equals the number of holes, yet the outside region contains extra almost-matching
#      shapes that never affect the output.
#    - Because the output colors come directly from unique matched sprites, there is no
#      hidden frequency rule left to infer. The only nontrivial step is exact silhouette
#      comparison under rotation/flip, which is visible from the near-miss distractors.
#
# f) Visual cleanliness
#    - The frame interior contains only frame color and background holes; there are no
#      extra decorations inside the template.
#    - Outside clutter is controlled: every sprite is monochrome, isolated, and small,
#      so even with 3-5 distractors the scene stays readable.
#
# g) Change inferability
#    - Colors are not numerically transformed. Each hole simply inherits the color of
#      its unique exact-match sprite.
#    - Distractors may reuse real colors or other non-frame colors, which prevents the
#      solver from relying on color rarity. Shape exactness is the visible signal.
#
# h) Behavioral non-degeneration check
#    - `_analyze_instance` re-extracts holes and sprites from a generated input and
#      verifies three combo-defining facts: true exact matches equal hole count, there
#      are 3-5 non-matching sprites, and every distractor is edit-distance 1 from some
#      real hole shape. Samples that do not show those behaviors are rejected.
#
# i) New solver assumptions introduced by the combo and generator guarantees
#    - Each hole canonical key is unique inside the frame: guaranteed by generating each
#      selected base shape only once.
#    - Each outside canonical key is unique: guaranteed by placing exactly one true
#      sprite per hole and deduplicating distractor keys before placement.
#    - Every hole key has a matching outside sprite key: guaranteed by output-first
#      construction and oracle validation with `main(inp) == output`.


# --------------- shared shape helpers ---------------


def _all_rot_flip(mask: np.ndarray):
    """Yield `(key, oriented_mask)` for all rotation/flip variants of a bool mask."""
    base = mask.astype(bool)
    for flip in (False, True):
        flipped = np.flipud(base) if flip else base
        for k in range(4):
            oriented = np.rot90(flipped, k)
            yield (oriented.shape, oriented.tobytes()), oriented


def _canonical_key(mask: np.ndarray):
    """Canonical representative for a shape under rotation/flip."""
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


def _orientation_key(mask: np.ndarray):
    """Exact orientation key without normalization."""
    return mask.shape, mask.tobytes()


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    """Crop a boolean mask to its tight bounding box."""
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _shape_edit_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Minimum XOR cell count between `a` and any oriented version of `b`."""
    aa = a.astype(bool)
    best = None
    for bb in _all_orientations(b):
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


def _extract_frame(g: np.ndarray, bg: int):
    """Return the unique largest non-background monochrome component as the frame."""
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert comps
    frame_area = max(int(comp["area"]) for comp in comps)
    frame_candidates = [comp for comp in comps if int(comp["area"]) == frame_area]
    assert len(frame_candidates) == 1
    frame = frame_candidates[0]
    return int(frame["color"]), tuple(map(int, frame["bbox"]))


def _extract_holes(sub: np.ndarray, frame_col: int):
    """Return hole records as dicts with canonical key, exact mask, and position."""
    hole_mask = sub != int(frame_col)
    holes: list[dict[str, object]] = []
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        mask = np.zeros((hh, ww), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        holes.append(
            {
                "canonical_key": _canonical_key(mask),
                "orientation_key": _orientation_key(mask),
                "pos": (int(hr0), int(hc0)),
                "mask": mask,
            }
        )
    return holes


def _extract_sprite_records(g: np.ndarray, frame_bbox, bg: int):
    """Return outside monochrome sprites with canonicalized shape metadata."""
    r0, c0, r1, c1 = map(int, frame_bbox)
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    comps = find_components(outside, bg=bg, connectivity=8, monochromatic=True, return_format="dict")

    records: list[dict[str, object]] = []
    for comp in comps:
        coords = comp["coords"]
        sr0, sc0, sr1, sc1 = map(int, comp["bbox"])
        hh = int(sr1 - sr0 + 1)
        ww = int(sc1 - sc0 + 1)
        rel = coords - np.array([sr0, sc0])
        mask = np.zeros((hh, ww), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        records.append(
            {
                "color": int(comp["color"]),
                "mask": mask,
                "canonical_key": _canonical_key(mask),
                "orientation_key": _orientation_key(mask),
            }
        )
    return records


# --------------- solver ---------------


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the cropped frame by matching each hole to one exact outside sprite and ignoring many near-miss decoys.

    Input structure:
    - The most common color is the background.
    - One large monochrome component is the frame; inside its bounding box are several separate holes.
    - Outside the frame there is exactly one true sprite for each hole shape, plus 3-5 extra sprites whose shapes differ from real holes by one cell.

    Rule:
    - Find the frame, crop to its bounding box, and start a solid canvas of the frame color.
    - Normalize each hole and each outside sprite up to rotation/flip, then build a direct shape-to-color dictionary from the outside sprites.
    - For each hole, copy the color of the unique outside sprite with the same normalized shape; ignore all sprites whose normalized shape does not match any hole exactly.

    Key insight:
    - This fused variant removes color voting entirely: the problem is not choosing among multiple exact matches, but spotting the single perfect silhouette among several almost-correct distractors.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    frame_col, frame_bbox = _extract_frame(g, bg)
    sub = crop_bbox(g, frame_bbox)
    th, tw = sub.shape

    holes = _extract_holes(sub, frame_col)
    hole_keys = [record["canonical_key"] for record in holes]
    assert len(hole_keys) == len(set(hole_keys))

    sprite_records = _extract_sprite_records(g, frame_bbox, bg)
    sprites_by_key: dict[tuple[tuple[int, int], bytes], int] = {}
    for rec in sprite_records:
        key = rec["canonical_key"]
        assert key not in sprites_by_key
        sprites_by_key[key] = int(rec["color"])

    out = np.full((th, tw), int(frame_col), dtype=int)
    for hole in holes:
        key = hole["canonical_key"]
        scol = sprites_by_key.get(key)
        assert scol is not None
        hr0, hc0 = hole["pos"]
        mask = hole["mask"]
        patch = np.full(mask.shape, int(scol), dtype=int)
        paste(out, patch, int(hr0), int(hc0), mask=mask)
    return out


# --------------- generator ---------------

NUM_COLORS = 10
GRID_SIZE = 30

FRAME_H_RANGE = (8, 13)
FRAME_W_RANGE = (8, 18)

HOLE_COUNT_RANGE = (4, 6)
MIN_HOLES = 4
SHAPE_MAX_AREA = 8
FRAME_DENSITY_RANGE = (0.70, 0.90)
PLACEMENT_RETRIES = 160

SPRITE_PADDING = 1
DISTRACTOR_RANGE = (3, 5)
DISTRACTOR_AREA_MAX = SHAPE_MAX_AREA + 1

MAX_GEN_ATTEMPTS = 500


def _build_shape_library():
    raw = [
        [[1, 1]],
        [[1], [1]],
        [[1, 1, 1]],
        [[1, 1, 1, 1]],
        [[1, 1], [1, 1]],
        [[1, 0], [1, 1]],
        [[1, 1], [1, 0]],
        [[1, 0], [1, 0], [1, 1]],
        [[1, 1], [0, 1], [0, 1]],
        [[1, 0], [1, 1], [0, 1]],
        [[0, 1], [1, 1], [1, 0]],
        [[1, 1, 1], [1, 0, 0]],
        [[1, 1, 1], [0, 0, 1]],
        [[1, 1], [0, 1], [1, 1]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[1, 1, 1], [0, 1, 0], [0, 1, 0]],
        [[1, 1, 0], [0, 1, 1]],
        [[0, 1, 1], [1, 1, 0]],
        [[1, 1, 1], [0, 1, 1]],
    ]
    seen: set[tuple[tuple[int, int], bytes]] = set()
    library: list[np.ndarray] = []
    for spec in raw:
        mask = np.array(spec, dtype=bool)
        area = int(mask.sum())
        if area < 2 or area > SHAPE_MAX_AREA:
            continue
        key = _canonical_key(mask)
        if key in seen:
            continue
        seen.add(key)
        library.append(mask)
    return library


SHAPE_LIBRARY = _build_shape_library()


def _can_place(occ: np.ndarray, shape: np.ndarray, pr: int, pc: int) -> bool:
    """Check bounds, overlap, and 4-neighbor separation for one hole."""
    th, tw = occ.shape
    sh, sw = shape.shape
    if pr < 1 or pc < 1 or pr + sh > th - 1 or pc + sw > tw - 1:
        return False

    new_cells = set()
    for dr in range(sh):
        for dc in range(sw):
            if shape[dr, dc]:
                new_cells.add((pr + dr, pc + dc))

    for rr, cc in new_cells:
        if not occ[rr, cc]:
            return False
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if 0 <= nr < th and 0 <= nc < tw and not occ[nr, nc] and (nr, nc) not in new_cells:
                return False
    return True


def _carve_hole(occ: np.ndarray, shape: np.ndarray, pr: int, pc: int):
    """Mark one placed hole as empty in the occupancy grid."""
    sh, sw = shape.shape
    for dr in range(sh):
        for dc in range(sw):
            if shape[dr, dc]:
                occ[pr + dr, pc + dc] = False


def _near_miss_variants(mask: np.ndarray):
    """Return connected masks that differ from `mask` by exactly one cell."""
    base = mask.astype(bool)
    h, w = base.shape
    out: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}

    for rr, cc in np.argwhere(base):
        cand = base.copy()
        cand[rr, cc] = False
        if not cand.any():
            continue
        if len(components_mask(cand, connectivity=4)) != 1:
            continue
        trimmed = _trim_mask(cand)
        if int(trimmed.sum()) <= DISTRACTOR_AREA_MAX:
            out[_canonical_key(trimmed)] = trimmed

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
        if int(trimmed.sum()) <= DISTRACTOR_AREA_MAX:
            out[_canonical_key(trimmed)] = trimmed

    out.pop(_canonical_key(base), None)
    return list(out.values())


def _distractor_candidates(base_shapes: list[np.ndarray]):
    """Collect unique near-miss distractor candidates derived from real shapes."""
    real_keys = {_canonical_key(shape) for shape in base_shapes}
    per_base: list[list[np.ndarray]] = []
    unique: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}

    for shape in base_shapes:
        local: list[np.ndarray] = []
        seen_local: set[tuple[tuple[int, int], bytes]] = set()
        for cand in _near_miss_variants(shape):
            key = _canonical_key(cand)
            if key in real_keys or key in seen_local:
                continue
            seen_local.add(key)
            local.append(cand)
            unique[key] = cand
        random.shuffle(local)
        per_base.append(local)
    return unique, per_base


def _select_shapes(th: int, tw: int):
    """Choose 4-6 unique real hole shapes with enough distractor support."""
    available = [
        shape
        for shape in SHAPE_LIBRARY
        if shape.shape[0] <= th - 2 and shape.shape[1] <= tw - 2 and int(shape.sum()) <= SHAPE_MAX_AREA
    ]
    if len(available) < MIN_HOLES:
        return None

    asymmetric = [shape for shape in available if len(_all_orientations(shape)) > 1]
    if not asymmetric:
        return None

    max_k = min(HOLE_COUNT_RANGE[1], len(available), NUM_COLORS - 2)
    if max_k < MIN_HOLES:
        return None

    for _ in range(100):
        k_target = random.randint(HOLE_COUNT_RANGE[0], max_k)
        pool = available[:]
        random.shuffle(pool)

        anchor = random.choice(asymmetric)
        selected = [anchor]
        for shape in pool:
            if _canonical_key(shape) == _canonical_key(anchor):
                continue
            selected.append(shape)
            if len(selected) >= k_target:
                break

        if len(selected) < MIN_HOLES:
            continue

        random.shuffle(selected)
        selected = selected[:k_target]
        unique, _ = _distractor_candidates(selected)
        if len(unique) >= DISTRACTOR_RANGE[0]:
            selected.sort(key=lambda shape: (int(shape.sum()), shape.shape[0] * shape.shape[1]), reverse=True)
            return selected
    return None


def _build_hole_layout(th: int, tw: int):
    """Place one instance of each chosen shape. Returns `(occ, hole_records)` or None."""
    selected = _select_shapes(th, tw)
    if selected is None:
        return None

    occ = np.ones((th, tw), dtype=bool)
    hole_records: list[dict[str, object]] = []

    for base_shape in selected:
        orientations = _all_orientations(base_shape)
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            oriented = random.choice(orientations)
            sh, sw = oriented.shape
            pr = random.randint(1, th - sh - 1)
            pc = random.randint(1, tw - sw - 1)
            if not _can_place(occ, oriented, pr, pc):
                continue
            _carve_hole(occ, oriented, pr, pc)
            hole_records.append(
                {
                    "base_shape": base_shape,
                    "hole_mask": oriented,
                    "hole_pos": (int(pr), int(pc)),
                }
            )
            placed = True
            break
        if not placed:
            return None

    frame_density = 1.0 - int(np.sum(~occ)) / float(th * tw)
    if not (FRAME_DENSITY_RANGE[0] <= frame_density <= FRAME_DENSITY_RANGE[1]):
        return None
    return occ, hole_records


def _build_output(occ: np.ndarray, hole_records: list[dict[str, object]], frame_col: int, bg: int):
    """Render the output and assign one unique fill color to each real hole shape."""
    avail_colors = [color for color in range(NUM_COLORS) if color not in (bg, frame_col)]
    if len(avail_colors) < len(hole_records):
        return None

    output = np.full(occ.shape, frame_col, dtype=int)
    fill_colors = random.sample(avail_colors, len(hole_records))
    shape_info: list[dict[str, object]] = []

    for record, fill_col in zip(hole_records, fill_colors):
        hole_mask = record["hole_mask"]
        pr, pc = record["hole_pos"]
        patch = np.full(hole_mask.shape, fill_col, dtype=int)
        paste(output, patch, int(pr), int(pc), mask=hole_mask)
        shape_info.append(
            {
                "base_shape": record["base_shape"],
                "hole_mask": hole_mask,
                "hole_pos": (int(pr), int(pc)),
                "fill_color": int(fill_col),
            }
        )
    return output, shape_info


def _place_sprite(inp: np.ndarray, placement: np.ndarray, oriented: np.ndarray, color: int, bg: int) -> bool:
    """Place one monochrome sprite on the outside background."""
    patch = np.full(oriented.shape, bg, dtype=int)
    patch[oriented] = color
    try:
        r, c = random_free_location(placement, patch, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, patch, r, c, transparent=bg)
    paste(inp, patch, r, c, transparent=bg)
    return True


def _choose_distractor_shapes(shape_info: list[dict[str, object]]):
    """Pick 3-5 unique near-miss shapes distributed across real bases when possible."""
    base_shapes = [info["base_shape"] for info in shape_info]
    unique, per_base = _distractor_candidates(base_shapes)
    if len(unique) < DISTRACTOR_RANGE[0]:
        return None

    target = random.randint(DISTRACTOR_RANGE[0], min(DISTRACTOR_RANGE[1], len(unique)))
    selected: list[np.ndarray] = []
    used: set[tuple[tuple[int, int], bytes]] = set()

    while len(selected) < target:
        progress = False
        order = list(range(len(per_base)))
        random.shuffle(order)
        for idx in order:
            while per_base[idx] and _canonical_key(per_base[idx][-1]) in used:
                per_base[idx].pop()
            if not per_base[idx]:
                continue
            cand = per_base[idx].pop()
            key = _canonical_key(cand)
            if key in used:
                continue
            selected.append(cand)
            used.add(key)
            progress = True
            if len(selected) >= target:
                break
        if not progress:
            break

    if len(selected) < target:
        leftovers = list(unique.items())
        random.shuffle(leftovers)
        for key, cand in leftovers:
            if key in used:
                continue
            selected.append(cand)
            used.add(key)
            if len(selected) >= target:
                break

    return selected if len(selected) >= DISTRACTOR_RANGE[0] else None


def _scatter_sprites(inp: np.ndarray, shape_info: list[dict[str, object]], bg: int, frame_col: int):
    """Place one true sprite per hole plus 3-5 near-miss distractors."""
    placement = inp.copy()
    color_pool = [color for color in range(NUM_COLORS) if color not in (bg, frame_col)]

    mismatch_needed = True
    mismatch_count = 0

    for info in shape_info:
        base_shape = info["base_shape"]
        hole_mask = info["hole_mask"]
        fill_color = info["fill_color"]
        orientations = _all_orientations(base_shape)
        hole_key = _orientation_key(hole_mask)
        diff_orientations = [ori for ori in orientations if _orientation_key(ori) != hole_key]

        if mismatch_needed and diff_orientations:
            oriented = random.choice(diff_orientations)
            mismatch_needed = False
        else:
            oriented = random.choice(orientations)

        if not _place_sprite(inp, placement, oriented, int(fill_color), bg):
            return None
        if _orientation_key(oriented) != hole_key:
            mismatch_count += 1

    distractor_shapes = _choose_distractor_shapes(shape_info)
    if distractor_shapes is None:
        return None

    preferred_colors = [int(info["fill_color"]) for info in shape_info]
    for shape in distractor_shapes:
        distractor_color = random.choice(preferred_colors + color_pool)
        oriented = random.choice(_all_orientations(shape))
        if not _place_sprite(inp, placement, oriented, int(distractor_color), bg):
            return None

    return {
        "orientation_mismatch_count": mismatch_count,
        "distractor_count": len(distractor_shapes),
    }


def _analyze_instance(grid: np.ndarray):
    """Return stats used to verify that the combo behavior is present."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frame_col, frame_bbox = _extract_frame(g, bg)
    sub = crop_bbox(g, frame_bbox)

    holes = _extract_holes(sub, frame_col)
    hole_masks = {rec["canonical_key"]: rec["mask"] for rec in holes}
    hole_orientation_keys = {rec["canonical_key"]: rec["orientation_key"] for rec in holes}

    sprites = _extract_sprite_records(g, frame_bbox, bg)
    exact_sprite_count = 0
    distractor_count = 0
    near_miss_count = 0
    orientation_mismatch_count = 0
    exact_keys: set[tuple[tuple[int, int], bytes]] = set()
    distractor_edit_distances: list[int] = []

    for rec in sprites:
        key = rec["canonical_key"]
        if key in hole_masks:
            exact_sprite_count += 1
            exact_keys.add(key)
            if rec["orientation_key"] != hole_orientation_keys[key]:
                orientation_mismatch_count += 1
            continue
        distractor_count += 1
        dist = min(_shape_edit_distance(rec["mask"], real_mask) for real_mask in hole_masks.values())
        distractor_edit_distances.append(dist)
        if dist == 1:
            near_miss_count += 1

    return {
        "hole_count": len(holes),
        "unique_hole_key_count": len(hole_masks),
        "sprite_count": len(sprites),
        "exact_sprite_count": exact_sprite_count,
        "unique_exact_key_count": len(exact_keys),
        "distractor_count": distractor_count,
        "near_miss_count": near_miss_count,
        "orientation_mismatch_count": orientation_mismatch_count,
        "min_distractor_edit": min(distractor_edit_distances) if distractor_edit_distances else None,
        "max_distractor_edit": max(distractor_edit_distances) if distractor_edit_distances else None,
    }


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt for the exact-match-plus-near-miss combo."""
    bg = random.randint(0, NUM_COLORS - 1)
    frame_col = random.choice([color for color in range(NUM_COLORS) if color != bg])
    th = random.randint(*FRAME_H_RANGE)
    tw = random.randint(*FRAME_W_RANGE)

    layout = _build_hole_layout(th, tw)
    if layout is None:
        return None
    occ, hole_records = layout

    result = _build_output(occ, hole_records, frame_col, bg)
    if result is None:
        return None
    output, shape_info = result

    fr = random.randint(0, GRID_SIZE - th)
    fc = random.randint(0, GRID_SIZE - tw)
    inp = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    paste(inp, np.where(occ, frame_col, bg).astype(int), fr, fc)

    sprite_stats = _scatter_sprites(inp, shape_info, bg, frame_col)
    if sprite_stats is None:
        return None

    try:
        if not np.array_equal(main(inp), output):
            return None
        stats = _analyze_instance(inp)
    except Exception:
        return None

    if stats["hole_count"] < MIN_HOLES:
        return None
    if stats["hole_count"] != stats["unique_hole_key_count"]:
        return None
    if stats["exact_sprite_count"] != stats["hole_count"]:
        return None
    if stats["unique_exact_key_count"] != stats["hole_count"]:
        return None
    if not (DISTRACTOR_RANGE[0] <= stats["distractor_count"] <= DISTRACTOR_RANGE[1]):
        return None
    if stats["near_miss_count"] != stats["distractor_count"]:
        return None
    if stats["min_distractor_edit"] != 1 or stats["max_distractor_edit"] != 1:
        return None
    if stats["orientation_mismatch_count"] < 1:
        return None
    if sprite_stats["orientation_mismatch_count"] < 1:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {MAX_GEN_ATTEMPTS} attempts")
