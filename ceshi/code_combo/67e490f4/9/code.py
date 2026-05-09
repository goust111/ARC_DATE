# Combo variant of 67e490f4 (index 9)
# combo_spec: D4/2+D5/2
# Name: 框架内精灵+中空精灵

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
)


# Design analysis for combo D4/2 + D5/2
# --------------------------------------
# a) Concrete behavior trace
#    - Original task:
#      one large monochrome frame contains bg holes; small solid sprites live outside the frame;
#      hole shape and outside sprite shape are matched up to rotation/flip; the most frequent
#      sprite color per shape family fills all matching holes.
#    - Fused variant:
#      the outside field is now empty. Instead, every hole already contains one monochrome sample
#      sprite painted inside the hole. That sample is hollow: some bg cells are carved out of its
#      interior, but the sample's outer silhouette is still the same as the surrounding hole.
#    - Small trace:
#      suppose three square holes appear in the frame. Inside them we see hollow samples
#      `red ring`, `red ring`, `green ring`.
#      Original solver would have looked outside the frame and raw foreground masks would have
#      matched directly. That is now impossible because:
#        1. all evidence moved inside the holes,
#        2. the visible sample mask is not the semantic key anymore.
#      New solver behavior:
#        * recover each hollow sample's outer silhouette by filling its enclosed bg cavity,
#        * group those recovered silhouettes across the frame,
#        * use the plurality sample color of that silhouette family,
#        * flood-fill every hole of that family with that winning color.
#    - The key fused interaction is that the hole component remains recoverable as the union of
#      sample pixels plus the remaining bg cells inside the same cavity, while the sample key must
#      be derived from only the colored subset after restoring its enclosed interior.
#
# b) Complexity calibration
#    - To keep D4/2 visible, every sample sprite is forced to have at least one enclosed bg cavity;
#      otherwise the task collapses back to a plain "read the color already sitting in the hole"
#      version.
#    - To keep D5/2 visible, there are no informative components outside the frame at all.
#      All non-frame evidence is embedded inside frame holes.
#    - To keep the majority rule visible, each sample contains 3-5 shape families with repeated
#      instances, and at least one family has a strict plurality color that disagrees with at
#      least one individual hole's sample color.
#
# c) Hidden-assumption audit of the original main()
#    - Old STEP 2 assumed every non-frame cell inside the frame bbox belonged to a hole. This is
#      still valid, but the hole component is now "hole bg + sample pixels" rather than pure bg.
#      The generator guarantees one frame-colored moat between holes, so these mixed components
#      still recover the correct outer hole silhouette.
#    - Old STEP 3 assumed informative sprites live outside the frame. Broken by D5/2. The solver
#      must instead search for colored components inside the frame bbox and ignore the outside.
#    - Old STEP 3 also assumed raw sprite pixels are the matching key. Broken by D4/2. The solver
#      must fill each sample's enclosed bg cavities before computing its shape key.
#    - Old color tie-break by numeric color id is not human-visible. The generator avoids semantic
#      ties by construction, so the solver's fallback ordering is never relied upon.
#
# d) Placement legality
#    - Holes remain strictly inside the frame border, pairwise 4-disconnected, and non-overlapping.
#    - Every hole receives exactly one same-bbox sample sprite whose painted pixels stay inside the
#      hole mask; only enclosed interior cells are left as bg.
#    - Sample cavities never touch the sample bbox boundary, so `_fill_enclosed_bg()` recovers the
#      intended outer silhouette exactly.
#
# e) Visual observability
#    - A human sees no outside clues. The only colored evidence sits inside the holes themselves.
#    - The samples visibly differ from the output because input holes contain thin hollow shells,
#      while output holes become solid fills.
#    - Repeated same-shape holes with mixed sample colors make the family-wise plurality rule
#      observable across 2-3 examples.
#
# f) Visual cleanliness
#    - The grid contains only one large frame plus embedded samples. No third region of scattered
#      outside clutter remains, which keeps the scene cleaner than the source task.
#    - Hole silhouettes are chosen from thick, compact masks that can display a clear hollow
#      interior without hairline artifacts.
#
# g) Color inference
#    - Colors are copied, not numerically transformed.
#    - The winning color for each shape family is the sample color that appears most often among
#      hollow samples whose recovered outer silhouette matches that family.
#
# New solver assumptions audited against the generator
# - Every hole family has at least one embedded sample sprite, in fact one per hole instance.
# - Every sample is monochrome apart from bg cavities.
# - At least one family in every generated puzzle has a strict plurality color and at least one
#   dissenting sample color, so the task does not degenerate to "each hole keeps its own color".


def _all_rot_flip(mask: np.ndarray):
    """Yield all rotation/flip variants of a boolean mask."""
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes()), m2


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    """Canonical key under rotation/flip."""
    return min(key for key, _ in _all_rot_flip(mask))


def _all_orientations(mask: np.ndarray) -> list[np.ndarray]:
    """Deduplicated rotation/flip variants."""
    seen: set[tuple[tuple[int, int], bytes]] = set()
    out: list[np.ndarray] = []
    for key, m in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            out.append(m.copy())
    return out


def _extract_mask_from_coords(
    coords: np.ndarray, bbox: tuple[int, int, int, int]
) -> np.ndarray:
    """Return a cropped boolean mask for coords inside bbox."""
    r0, c0, r1, c1 = map(int, bbox)
    hh = int(r1 - r0 + 1)
    ww = int(c1 - c0 + 1)
    rel = coords - np.array([r0, c0])
    mask = np.zeros((hh, ww), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _fill_enclosed_bg(mask: np.ndarray) -> np.ndarray:
    """Fill bg pockets that are fully enclosed inside a sprite bbox."""
    filled = mask.astype(bool).copy()
    bg_mask = ~filled
    if not bg_mask.any():
        return filled
    hh, ww = filled.shape
    for coords in components_mask(bg_mask, connectivity=4):
        touches_edge = False
        for rr, cc in coords:
            if int(rr) == 0 or int(cc) == 0 or int(rr) == hh - 1 or int(cc) == ww - 1:
                touches_edge = True
                break
        if not touches_edge:
            filled[coords[:, 0], coords[:, 1]] = True
    return filled


def _cavity_options(mask: np.ndarray) -> list[np.ndarray]:
    """All enclosed cavity choices that keep the painted sample 8-connected."""
    mask = mask.astype(bool)
    hh, ww = mask.shape
    candidates: list[tuple[int, int]] = []
    for r in range(1, hh - 1):
        for c in range(1, ww - 1):
            if not mask[r, c]:
                continue
            if mask[r - 1, c] and mask[r + 1, c] and mask[r, c - 1] and mask[r, c + 1]:
                candidates.append((r, c))

    options: list[np.ndarray] = []
    seen: set[bytes] = set()

    def try_add(cells: list[tuple[int, int]]):
        cavity = np.zeros_like(mask, dtype=bool)
        for r, c in cells:
            cavity[r, c] = True
        painted = mask & ~cavity
        if len(components_mask(painted, connectivity=8)) != 1:
            return
        enclosed = 0
        for coords in components_mask(~painted, connectivity=4):
            touches_edge = False
            for rr, cc in coords:
                if int(rr) == 0 or int(cc) == 0 or int(rr) == hh - 1 or int(cc) == ww - 1:
                    touches_edge = True
                    break
            if not touches_edge:
                enclosed += 1
        if enclosed < 1:
            return
        key = cavity.tobytes()
        if key not in seen:
            seen.add(key)
            options.append(cavity)

    for cell in candidates:
        try_add([cell])

    cand_set = set(candidates)
    for r, c in candidates:
        for nr, nc in ((r + 1, c), (r, c + 1)):
            if (nr, nc) in cand_set:
                try_add([(r, c), (nr, nc)])

    return options


def _majority_color(color_counts: dict[int, int]) -> int:
    """Return the unique plurality color; fallback ordering is unreachable in generated data."""
    items = sorted(color_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return int(items[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill frame holes using hollow sample sprites that already sit inside those holes.

    Input structure:
    - `bg` is the most common color. One large monochrome frame occupies a rectangular bbox.
    - Inside the frame are several non-frame hole components; each hole contains one monochrome
      sample sprite of the same outer silhouette, but that sample has an enclosed bg cavity.
    - Outside the frame there are no informative sprites; the rest of the grid is just bg.

    Rule:
    - Crop to the frame bbox and start from a solid rectangle of the frame color.
    - Read hole families from their full outer silhouette, but read sample keys by first filling
      the enclosed bg cavity inside each hollow sample, then comparing those restored silhouettes
      up to rotation/flip.
    - For each hole family, choose the most frequent sample color among matching restored
      silhouettes and fill every hole of that family with that color.

    Key insight:
    - The evidence moved inside the destination holes, and the visible painted sample is not the
      true key anymore. You must recover each hollow sample's outer contour, group equal contours,
      and then propagate the plurality sample color to the full hole silhouette.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Find the frame as the unique largest monochrome non-bg component.
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert comps
    frame_area = max(int(comp["area"]) for comp in comps)
    frame_candidates = [comp for comp in comps if int(comp["area"]) == frame_area]
    assert len(frame_candidates) == 1
    frame = frame_candidates[0]
    frame_col = int(frame["color"])
    fb = frame["bbox"]
    _r0, _c0, _r1, _c1 = map(int, fb)

    sub = crop_bbox(g, fb)
    th, tw = sub.shape

    # STEP 2) Extract whole hole components as all non-frame cells inside the frame bbox.
    hole_mask = sub != int(frame_col)
    holes: list[tuple[tuple[tuple[int, int], bytes], tuple[int, int], np.ndarray]] = []
    for coords in components_mask(hole_mask, connectivity=4):
        hb = bbox_coords(coords)
        hr0, hc0, _, _ = map(int, hb)
        mask = _extract_mask_from_coords(coords, hb)
        holes.append((_canonical_key(mask), (int(hr0), int(hc0)), mask))

    # STEP 3) Extract colored sample sprites from inside the holes and recover their silhouettes.
    sample_mask = (sub != int(frame_col)) & (sub != int(bg))
    samples_by_key: dict[tuple[tuple[int, int], bytes], dict[int, int]] = {}
    for coords in components_mask(sample_mask, connectivity=8):
        bb = bbox_coords(coords)
        raw_mask = _extract_mask_from_coords(coords, bb)
        silhouette = _fill_enclosed_bg(raw_mask)
        key = _canonical_key(silhouette)
        vals = [int(v) for v in sub[coords[:, 0], coords[:, 1]]]
        scol = int(Counter(vals).most_common(1)[0][0])
        bucket = samples_by_key.setdefault(key, {})
        bucket[scol] = int(bucket.get(scol, 0)) + 1

    # STEP 4) Paint the solved crop: solid frame color first, then flood each hole family.
    out = np.full((th, tw), int(frame_col), dtype=int)
    for key, (hr0, hc0), hmask in holes:
        cmap = samples_by_key.get(key)
        assert cmap is not None
        scol = _majority_color(cmap)
        patch = np.full(hmask.shape, int(scol), dtype=int)
        paste(out, patch, int(hr0), int(hc0), mask=hmask)

    return out


# --------------- generator ---------------

NUM_COLORS = 10
GRID_SIZE = 30

FRAME_H_RANGE = (11, 17)
FRAME_W_RANGE = (11, 24)

SHAPE_COUNT_RANGE = (3, 5)
MIN_SHAPES = 3
MIN_TOTAL_HOLES = 7
FRAME_DENSITY_RANGE = (0.55, 0.88)
PLACEMENT_RETRIES = 100
MULT_EVEN_WEIGHTS = ([1, 2], [3, 2])
MULT_ODD_WEIGHTS = ([1, 2, 3], [2, 3, 1])

MAX_GEN_ATTEMPTS = 500

SUPPORTED_PARITY_BUCKETS = [
    ((1, 0), 1),  # odd x even
    ((1, 1), 1),  # odd x odd
    ((0, 1), 6),  # even x odd: upweighted because layout acceptance is much lower
]


def _build_shape_library() -> list[np.ndarray]:
    """Compact thick hole silhouettes that can host a same-footprint hollow sample."""
    raw = [
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
        [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
        [[1, 1, 1], [1, 1, 1], [1, 1, 0]],
        [[1, 1, 1], [1, 1, 1], [0, 1, 1]],
        [[1, 1, 1], [1, 1, 1], [0, 1, 0], [0, 1, 0]],
        [[1, 1, 1, 0], [1, 1, 1, 1], [0, 1, 1, 1]],
        [[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 0]],
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 0], [0, 1, 1, 0]],
    ]
    seen: set[tuple[tuple[int, int], bytes]] = set()
    out: list[np.ndarray] = []
    for rows in raw:
        mask = np.array(rows, dtype=bool)
        if len(components_mask(mask, connectivity=4)) != 1:
            continue
        if not _cavity_options(mask):
            continue
        key = _canonical_key(mask)
        if key not in seen:
            seen.add(key)
            out.append(mask)
    return out


SHAPE_LIBRARY = _build_shape_library()


def _sample_frame_dims() -> tuple[int, int]:
    """
    Sample frame dimensions from parity buckets that the symmetric layout builder can realize.

    Even x even frames are intentionally skipped: under the current mirror-placement regime they
    become a near-dead branch and hurt generator coverage without adding task signal.
    """
    heights_by_parity = {
        0: [h for h in range(FRAME_H_RANGE[0], FRAME_H_RANGE[1] + 1) if h % 2 == 0],
        1: [h for h in range(FRAME_H_RANGE[0], FRAME_H_RANGE[1] + 1) if h % 2 == 1],
    }
    widths_by_parity = {
        0: [w for w in range(FRAME_W_RANGE[0], FRAME_W_RANGE[1] + 1) if w % 2 == 0],
        1: [w for w in range(FRAME_W_RANGE[0], FRAME_W_RANGE[1] + 1) if w % 2 == 1],
    }

    valid_buckets = [
        (parity, weight)
        for parity, weight in SUPPORTED_PARITY_BUCKETS
        if heights_by_parity[parity[0]] and widths_by_parity[parity[1]]
    ]
    chosen_parity = random.choices(
        [parity for parity, _weight in valid_buckets],
        weights=[weight for _parity, weight in valid_buckets],
    )[0]
    th = random.choice(heights_by_parity[chosen_parity[0]])
    tw = random.choice(widths_by_parity[chosen_parity[1]])
    return int(th), int(tw)


def _mirror_pos(
    r: int, c: int, sh: int, sw: int, th: int, tw: int
) -> list[tuple[int, int]]:
    """Return 4-fold mirrored positions, deduplicated."""
    return list(
        {
            (r, c),
            (r, tw - 1 - c - sw + 1),
            (th - 1 - r - sh + 1, c),
            (th - 1 - r - sh + 1, tw - 1 - c - sw + 1),
        }
    )


def _validate_positions(
    occ: np.ndarray,
    shape: np.ndarray,
    positions: list[tuple[int, int]],
    th: int,
    tw: int,
) -> list[tuple[int, int]] | None:
    """Check bounds, non-overlap, and 8-disconnection from prior holes."""
    sh, sw = shape.shape
    new_cells: set[tuple[int, int]] = set()
    union_mask = np.zeros((th, tw), dtype=bool)
    for pr, pc in positions:
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    rr, cc = pr + dr, pc + dc
                    new_cells.add((rr, cc))
                    union_mask[rr, cc] = True

    # Copies produced by one mirrored placement must stay visually separate even for 8-connected
    # sample extraction, otherwise two diagonal neighbors can collapse into one sample component.
    if len(components_mask(union_mask, connectivity=8)) != len(positions):
        return None

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
                for nr, nc in (
                    (rr - 1, cc - 1),
                    (rr - 1, cc),
                    (rr - 1, cc + 1),
                    (rr, cc - 1),
                    (rr, cc + 1),
                    (rr + 1, cc - 1),
                    (rr + 1, cc),
                    (rr + 1, cc + 1),
                ):
                    if 0 <= nr < th and 0 <= nc < tw and not occ[nr, nc]:
                        if (nr, nc) not in new_cells:
                            return None
    return positions


def _try_place_quadrant(
    occ: np.ndarray, shape: np.ndarray, th: int, tw: int
) -> list[tuple[int, int]] | None:
    sh, sw = shape.shape
    r_max, c_max = th // 2 - sh, tw // 2 - sw
    if r_max < 1 or c_max < 1:
        return None
    r, c = random.randint(1, r_max), random.randint(1, c_max)
    return _validate_positions(occ, shape, _mirror_pos(r, c, sh, sw, th, tw), th, tw)


def _try_place_hcenter(
    occ: np.ndarray, shape: np.ndarray, th: int, tw: int
) -> list[tuple[int, int]] | None:
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


def _try_place_vcenter(
    occ: np.ndarray, shape: np.ndarray, th: int, tw: int
) -> list[tuple[int, int]] | None:
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


def _try_place_center(
    occ: np.ndarray, shape: np.ndarray, th: int, tw: int
) -> list[tuple[int, int]] | None:
    if th % 2 == 0 or tw % 2 == 0:
        return None
    sh, sw = shape.shape
    r, c = th // 2 - sh // 2, tw // 2 - sw // 2
    if r < 1 or c < 1 or r + sh > th - 1 or c + sw > tw - 1:
        return None
    return _validate_positions(occ, shape, [(r, c)], th, tw)


def _try_place_symmetric(
    occ: np.ndarray, shape: np.ndarray, th: int, tw: int
) -> list[tuple[int, int]] | None:
    placers = [_try_place_quadrant]
    if th % 2 == 1:
        placers.append(_try_place_hcenter)
    if tw % 2 == 1:
        placers.append(_try_place_vcenter)
    if th % 2 == 1 and tw % 2 == 1:
        placers.append(_try_place_center)
    return random.choice(placers)(occ, shape, th, tw)


def _carve_hole(occ: np.ndarray, shape: np.ndarray, positions: list[tuple[int, int]]) -> None:
    """Mark hole cells as False in the frame occupancy grid."""
    sh, sw = shape.shape
    for pr, pc in positions:
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    occ[pr + dr, pc + dc] = False


def _build_hole_layout(
    th: int, tw: int
) -> list[dict[str, object]] | None:
    """Place repeated hole families inside the frame."""
    max_sh = max(1, (th - 2) // 2)
    max_sw = max(1, (tw - 2) // 2)
    available = [s for s in SHAPE_LIBRARY if s.shape[0] <= max_sh and s.shape[1] <= max_sw]
    if len(available) < MIN_SHAPES:
        return None

    k_target = random.randint(SHAPE_COUNT_RANGE[0], min(SHAPE_COUNT_RANGE[1], len(available)))
    random.shuffle(available)
    even_dims = (th % 2 == 0 or tw % 2 == 0)
    if even_dims:
        available.sort(key=lambda s: int(np.sum(s)))

    occ = np.ones((th, tw), dtype=bool)
    families: list[dict[str, object]] = []
    mult_vals, mult_wts = MULT_EVEN_WEIGHTS if even_dims else MULT_ODD_WEIGHTS

    for base_shape in available:
        if len(families) >= k_target:
            break
        groups = random.choices(mult_vals, weights=mult_wts)[0]
        instances: list[tuple[np.ndarray, int, int]] = []
        orientations = _all_orientations(base_shape)
        for _ in range(groups):
            for _ in range(PLACEMENT_RETRIES):
                oriented = random.choice(orientations)
                positions = _try_place_symmetric(occ, oriented, th, tw)
                if positions is None:
                    continue
                _carve_hole(occ, oriented, positions)
                for pr, pc in positions:
                    instances.append((oriented.copy(), int(pr), int(pc)))
                break
        if instances:
            families.append({"shape": base_shape.copy(), "instances": instances})

    if len(families) < MIN_SHAPES:
        return None

    frame_density = 1.0 - int(np.sum(~occ)) / (th * tw)
    if not (FRAME_DENSITY_RANGE[0] <= frame_density <= FRAME_DENSITY_RANGE[1]):
        return None

    total_holes = sum(len(f["instances"]) for f in families)
    if total_holes < MIN_TOTAL_HOLES:
        return None

    if not any(len(f["instances"]) >= 3 for f in families):
        return None

    return families


def _assign_sample_colors(
    families: list[dict[str, object]], bg: int, frame_col: int
) -> list[dict[str, object]] | None:
    """Choose winner color per family and per-hole sample colors with strict plurality."""
    avail_colors = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    if len(avail_colors) < len(families):
        return None
    family_winners = random.sample(avail_colors, len(families))
    out: list[dict[str, object]] = []
    has_disagreement = False

    for family, fill_col in zip(families, family_winners):
        instances = list(family["instances"])
        n = len(instances)
        sample_colors: list[int] = []
        if n >= 3:
            n_win = random.randint(n // 2 + 1, n - 1)
        else:
            n_win = n
        sample_colors.extend([int(fill_col)] * n_win)
        rem = n - n_win
        loser_pool = [c for c in avail_colors if c != fill_col]
        if rem > 0:
            k = random.randint(1, min(2, rem, len(loser_pool)))
            chosen = random.sample(loser_pool, k)
            counts = [1] * k
            for _ in range(rem - k):
                counts[random.randrange(k)] += 1
            for color, count in zip(chosen, counts):
                sample_colors.extend([int(color)] * int(count))
            has_disagreement = True
        random.shuffle(sample_colors)
        enriched = []
        for (shape, r, c), sample_col in zip(instances, sample_colors):
            enriched.append((shape.copy(), int(r), int(c), int(sample_col)))
        out.append(
            {
                "shape": np.array(family["shape"], dtype=bool),
                "fill_color": int(fill_col),
                "instances": enriched,
            }
        )

    if not has_disagreement:
        candidates = [i for i, fam in enumerate(out) if len(fam["instances"]) >= 3]
        if not candidates:
            return None
        idx = random.choice(candidates)
        fam = out[idx]
        fill_col = int(fam["fill_color"])
        loser_pool = [c for c in avail_colors if c != fill_col]
        loser = int(random.choice(loser_pool))
        repl = list(fam["instances"])
        shape, r, c, _old = repl[0]
        repl[0] = (shape, r, c, loser)
        fam["instances"] = repl

    return out


def _render_hollow_sample(shape: np.ndarray, color: int, bg: int) -> np.ndarray | None:
    """Render a hollow monochrome sample whose restored silhouette equals `shape`."""
    options = _cavity_options(shape)
    if not options:
        return None
    cavity = random.choice(options)
    patch = np.full(shape.shape, bg, dtype=int)
    patch[shape] = int(color)
    patch[cavity] = int(bg)
    return patch


def _build_output(
    families: list[dict[str, object]], th: int, tw: int, frame_col: int
) -> np.ndarray:
    """Render the target crop with solid family-colored holes."""
    out = np.full((th, tw), int(frame_col), dtype=int)
    for fam in families:
        fill_col = int(fam["fill_color"])
        for shape, r, c, _sample_col in fam["instances"]:
            patch = np.full(shape.shape, fill_col, dtype=int)
            paste(out, patch, int(r), int(c), mask=shape)
    return out


def _build_input(
    families: list[dict[str, object]],
    th: int,
    tw: int,
    bg: int,
    frame_col: int,
) -> np.ndarray | None:
    """Render the full 30x30 input with a framed field of embedded hollow samples."""
    fr = random.randint(0, GRID_SIZE - th)
    fc = random.randint(0, GRID_SIZE - tw)

    occ = np.ones((th, tw), dtype=bool)
    for fam in families:
        for shape, r, c, _sample_col in fam["instances"]:
            for dr in range(shape.shape[0]):
                for dc in range(shape.shape[1]):
                    if shape[dr, dc]:
                        occ[int(r) + dr, int(c) + dc] = False

    inp = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    frame_patch = np.where(occ, int(frame_col), int(bg)).astype(int)
    paste(inp, frame_patch, fr, fc)

    for fam in families:
        for shape, r, c, sample_col in fam["instances"]:
            patch = _render_hollow_sample(shape, int(sample_col), int(bg))
            if patch is None:
                return None
            paste(inp, patch, fr + int(r), fc + int(c), transparent=int(bg))
    return inp


def _gen_attempt() -> np.ndarray | None:
    """Single output-first generation attempt."""
    bg = random.randint(0, NUM_COLORS - 1)
    frame_col = random.choice([c for c in range(NUM_COLORS) if c != bg])
    th, tw = _sample_frame_dims()

    families = _build_hole_layout(th, tw)
    if families is None:
        return None

    colored_families = _assign_sample_colors(families, bg, frame_col)
    if colored_families is None:
        return None

    target = _build_output(colored_families, th, tw, frame_col)
    inp = _build_input(colored_families, th, tw, bg, frame_col)
    if inp is None:
        return None

    try:
        if np.array_equal(main(inp), target):
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


def fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    """
    Quantify whether the generated sample really exhibits the fused behavior.

    Metrics used in post-generation auditing:
    - `inside_samples`: number of colored sample components inside the frame.
    - `outside_nonbg`: non-bg pixels outside the frame bbox; should be 0.
    - `hollow_samples`: number of sample components whose restored silhouette differs from raw pixels.
    - `disagreeing_holes`: holes whose embedded sample color differs from the solved fill color.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    frame = max(comps, key=lambda comp: int(comp["area"]))
    frame_col = int(frame["color"])
    r0, c0, r1, c1 = map(int, frame["bbox"])
    sub = crop_bbox(g, frame["bbox"])
    out = main(g)

    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = bg
    outside_nonbg = int(np.count_nonzero(outside != bg))

    sample_mask = (sub != frame_col) & (sub != bg)
    inside_samples = 0
    hollow_samples = 0
    disagreeing_holes = 0

    for coords in components_mask(sample_mask, connectivity=8):
        inside_samples += 1
        bb = bbox_coords(coords)
        raw_mask = _extract_mask_from_coords(coords, bb)
        restored = _fill_enclosed_bg(raw_mask)
        if not np.array_equal(raw_mask, restored):
            hollow_samples += 1

    for coords in components_mask(sub != frame_col, connectivity=4):
        bb = bbox_coords(coords)
        hr0, hc0, _, _ = map(int, bb)
        hole_mask = _extract_mask_from_coords(coords, bb)
        local = sub[hr0 : hr0 + hole_mask.shape[0], hc0 : hc0 + hole_mask.shape[1]]
        sample_vals = [int(v) for v in local[(local != frame_col) & (local != bg)]]
        assert sample_vals
        sample_color = int(Counter(sample_vals).most_common(1)[0][0])
        solved_vals = out[hr0 : hr0 + hole_mask.shape[0], hc0 : hc0 + hole_mask.shape[1]][hole_mask]
        solved_color = int(Counter(map(int, solved_vals)).most_common(1)[0][0])
        if sample_color != solved_color:
            disagreeing_holes += 1

    return {
        "inside_samples": int(inside_samples),
        "outside_nonbg": int(outside_nonbg),
        "hollow_samples": int(hollow_samples),
        "disagreeing_holes": int(disagreeing_holes),
    }
