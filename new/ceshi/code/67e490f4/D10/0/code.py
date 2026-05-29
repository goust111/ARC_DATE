# Variant of 67e490f4
# Dimension: D10 — 输出裁剪方式 (grid_shape)
# Change: tight_crop — 输出裁剪到框架的外接矩形大小（th x tw），从纯框架色开始，填充各洞... → 保留完整网格输出

# Design analysis for this D10 variant
# a) Concrete behavior and trace
#    - Input still has one large monochrome frame with background-colored holes, plus many small
#      outside sprites whose shapes match hole shapes up to rotation/flip.
#    - Original behavior: solve the frame locally, then return only the frame bbox crop.
#    - Variant behavior: solve the same hole-color assignment, but redraw the answer on the full
#      30x30 canvas. Outside sprites disappear; only the frame remains in its original position
#      with holes filled. Everything outside the frame becomes background.
#    - Small trace: if a 9x11 frame sits at rows 5..13, cols 8..18, original code returns a 9x11
#      patch. Variant code instead returns a 30x30 grid: background everywhere, that same 9x11
#      solved patch pasted back at (5, 8), and all sprite cells erased.
# b) Complexity calibration
#    - Shape matching, rotation/flip invariance, and majority-color choice stay unchanged, so the
#      puzzle is not simplified into merely cropping or recoloring.
#    - The generator still uses 3-6 distinct hole shapes and multiple sprite instances per shape,
#      ensuring repeated evidence for the mapping.
# c) Hidden-assumption review of the original main()
#    - Finding the largest monochrome non-bg component as the frame still holds.
#    - Reading holes only inside the frame bbox still holds because D10 does not alter frame
#      structure.
#    - Outside sprites are still extracted by blanking out the frame bbox; that assumption holds.
#    - The old render step assumed the correct output canvas is exactly (th, tw). That assumption
#      is false in D10, so rendering must start from the full input-sized background canvas and
#      paste the solved frame back into its original position.
# d) Placement legality
#    - The generator keeps using a frame-first layout with 4-disconnected holes.
#    - Sprites are still placed only on background cells with Chebyshev padding, so they do not
#      overlap each other or merge with the frame.
#    - A one-cell outer margin is enforced around the frame to keep visible background area where
#      sprite disappearance can be observed.
# e) Visual observability
#    - Every generated sample has a full-grid output with visible background outside the frame.
#    - Sprites are present in the input and absent in the output, so the D10 change is directly
#      visible without relying on hidden ids or bookkeeping.
# f) Visual cleanliness
#    - The output contains only two region types outside the frame: plain background and nothing
#      else. This removes clutter and makes the transform legible.
# g) Color inferability
#    - Hole-fill colors still come from repeated sprite colors keyed by shape. The generator keeps
#      a strict plurality for the winning color of each shape, so no invisible tie-breaking is
#      needed in the produced data.
# h) Variant-behavior metric used during validation
#    - Measure how many non-background cells lie outside the frame bbox in the input. In this
#      variant those cells must always vanish in the output, while the output shape stays 30x30.
# i) New solver assumptions introduced by D10
#    - The full-grid redraw assumes the frame bbox location should be preserved. The generator
#      guarantees exactly one large frame and keeps it spatially stable between input and output.

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
    Resolve shape-coded hole fills while keeping the entire canvas.

    Input structure:
    - The dominant color is the background on a 30x30 board.
    - One large monochrome component is a frame whose holes appear as non-frame cells inside its bbox.
    - Outside the frame are small separated sprites; their shapes encode which hole shape they refer to.

    Rule:
    - Locate the frame and read the hole masks inside its bounding box.
    - Match each hole shape against outside sprites under rotation and reflection, then choose the most frequent sprite color for that shape.
    - Return a full-size grid: background everywhere except the frame restored in place with each hole filled by its matched color; outside sprites vanish.

    Key insight:
    - The sprites are not moved into the output. They only act as shape-indexed color votes, and the final board is the original canvas with the solved frame left behind.
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
    hole_mask = sub != frame_col
    holes: list[tuple[tuple[tuple[int, int], bytes], tuple[int, int], np.ndarray]] = []
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        m = np.zeros((hh, ww), dtype=bool)
        m[rel[:, 0], rel[:, 1]] = True
        key = (m.shape, m.tobytes())
        holes.append((key, (int(hr0), int(hc0)), m))

    # STEP 3) Extract sprites outside the bbox.
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = bg
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    # key -> color -> count
    sprites_by_key: dict[tuple[tuple[int, int], bytes], dict[int, int]] = {}

    def _all_mask_keys(mask: np.ndarray):
        """Generate keys for all rotations and vertical-flip variants of a boolean mask."""
        m0 = mask.astype(bool)
        for flip in (False, True):
            m1 = np.flipud(m0) if flip else m0
            for k in range(4):
                m2 = np.rot90(m1, k)
                yield (m2.shape, m2.tobytes())

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

        for key in _all_mask_keys(m):
            bucket = sprites_by_key.setdefault(key, {})
            bucket[scol] = int(bucket.get(scol, 0)) + 1

    # STEP 4) Render the solved frame locally, then paste it back onto a full background canvas.
    solved_frame = np.full((th, tw), frame_col, dtype=int)
    for key, (hr0, hc0), hmask in holes:
        cmap = sprites_by_key.get(key)
        assert cmap is not None
        # The generator avoids ties; this fallback keeps behavior defined if one appears.
        scol = max(cmap.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        patch = np.full(hmask.shape, int(scol), dtype=int)
        paste(solved_frame, patch, int(hr0), int(hc0), mask=hmask)

    out = np.full(g.shape, bg, dtype=int)
    paste(out, solved_frame, r0, c0)
    return out


# --------------- generator ---------------

NUM_COLORS = 10
GRID_SIZE = 30

# Frame bbox dimensions
FRAME_H_RANGE = (7, 13)
FRAME_W_RANGE = (7, 20)
FRAME_MARGIN = 1

# Hole layout
SHAPE_COUNT_RANGE = (3, 6)
MIN_SHAPES = 3
SHAPE_MAX_AREA = 8
FRAME_DENSITY_RANGE = (0.55, 0.92)
MIN_TOTAL_HOLES = 5
PLACEMENT_RETRIES = 80
MULT_EVEN_WEIGHTS = ([1, 2], [4, 1])
MULT_ODD_WEIGHTS = ([1, 2, 3], [3, 2, 1])

# Sprite scattering
WINNING_SPRITE_RANGE = (2, 5)
OTHER_COLOR_MAX = 2
SPRITE_PADDING = 1

# Generation retry budget
MAX_GEN_ATTEMPTS = 300


def _all_rot_flip(mask: np.ndarray):
    """Yield (key, oriented_mask) for all 8 rotation/flip variants of a bool mask."""
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes()), m2


def _canonical_key(mask: np.ndarray):
    """Smallest key across all rotation/flip variants (canonical representative)."""
    return min(key for key, _ in _all_rot_flip(mask))


def _all_orientations(mask: np.ndarray):
    """Deduplicated list of all distinct rotation/flip variants."""
    seen, result = set(), []
    for key, m in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            result.append(m)
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
    return list(
        {
            (r, c),
            (r, tw - c - sw),
            (th - r - sh, c),
            (th - r - sh, tw - c - sw),
        }
    )


def _try_place_quadrant(occ, shape, th, tw):
    """Place shape in top-left quadrant, mirror to all 4 quadrants."""
    sh, sw = shape.shape
    r_max, c_max = th // 2 - sh, tw // 2 - sw
    if r_max < 1 or c_max < 1:
        return None
    r, c = random.randint(1, r_max), random.randint(1, c_max)
    return _validate_positions(occ, shape, _mirror_pos(r, c, sh, sw, th, tw), th, tw)


def _try_place_hcenter(occ, shape, th, tw):
    """Place on horizontal center line, mirror LR only (2 copies)."""
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
    c_m = tw - c - sw
    return _validate_positions(occ, shape, list({(r, c), (r, c_m)}), th, tw)


def _try_place_vcenter(occ, shape, th, tw):
    """Place on vertical center line, mirror UD only (2 copies)."""
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
    r_m = th - r - sh
    return _validate_positions(occ, shape, list({(r, c), (r_m, c)}), th, tw)


def _try_place_center(occ, shape, th, tw):
    """Place at dead center (1 copy). Requires both dims odd."""
    if th % 2 == 0 or tw % 2 == 0:
        return None
    sh, sw = shape.shape
    r, c = th // 2 - sh // 2, tw // 2 - sw // 2
    if r < 1 or c < 1 or r + sh > th - 1 or c + sw > tw - 1:
        return None
    return _validate_positions(occ, shape, [(r, c)], th, tw)


def _validate_positions(occ, shape, positions, th, tw):
    """Check all positions: in bounds, no overlap with existing holes, 4-disconnected from them."""
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


def _try_place_symmetric(occ, shape, th, tw):
    """Try a random symmetric placement type."""
    placers = [_try_place_quadrant]
    if th % 2 == 1:
        placers.append(_try_place_hcenter)
    if tw % 2 == 1:
        placers.append(_try_place_vcenter)
    if th % 2 == 1 and tw % 2 == 1:
        placers.append(_try_place_center)
    return random.choice(placers)(occ, shape, th, tw)


def _carve_hole(occ, shape, positions):
    """Mark hole cells as unoccupied in the occupancy grid."""
    sh, sw = shape.shape
    for pr, pc in positions:
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    occ[pr + dr, pc + dc] = False


def _build_hole_layout(th, tw):
    """Place holes with LR+UD+rot180 symmetry. Returns (occ, hole_placements) or None."""
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
    even_dims = th % 2 == 0 or tw % 2 == 0
    if even_dims:
        available.sort(key=lambda s: int(np.sum(s)))

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

    total_holes = sum(len(pos) for _, pls in hole_placements for pos in pls)
    if total_holes < MIN_TOTAL_HOLES:
        return None

    return occ, hole_placements


def _build_solved_frame(occ, hole_placements, frame_col, bg):
    """Render the solved frame patch and assign a unique winning color per shape."""
    th, tw = occ.shape
    avail_colors = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    if len(avail_colors) < len(hole_placements):
        return None
    shape_colors = random.sample(avail_colors, len(hole_placements))

    solved_frame = np.full((th, tw), frame_col, dtype=int)
    shape_info = []
    for (shape, pls), fill_col in zip(hole_placements, shape_colors):
        all_pos = []
        for positions in pls:
            for pr, pc in positions:
                paste(solved_frame, np.full(shape.shape, fill_col, dtype=int), pr, pc, mask=shape)
                all_pos.append((pr, pc))
        shape_info.append((shape, fill_col, all_pos))
    return solved_frame, shape_info


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


def _scatter_sprites(inp, shape_info, bg, frame_col):
    """Place winning and losing sprites for each hole shape outside the frame."""
    placement = inp.copy()
    other_pool = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]

    for shape, fill_color, _ in shape_info:
        orientations = _all_orientations(shape)
        n_winning = random.randint(*WINNING_SPRITE_RANGE)
        losers = [c for c in other_pool if c != fill_color]
        n_other = random.randint(1, min(OTHER_COLOR_MAX, len(losers))) if losers else 0
        oc_list = random.sample(losers, min(n_other, len(losers)))

        for _ in range(n_winning):
            if not _place_sprite(inp, placement, random.choice(orientations), fill_color, bg):
                return False

        for oc in oc_list:
            n_oc = random.randint(1, max(1, n_winning - 1))
            for _ in range(n_oc):
                if not _place_sprite(inp, placement, random.choice(orientations), oc, bg):
                    break
    return True


def _make_full_output(solved_frame, fr, fc, bg):
    """Paste the solved frame back onto a full-size background canvas."""
    out = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    paste(out, solved_frame, fr, fc)
    return out


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build symmetric frame, scatter sprites, validate."""
    bg = random.randint(0, NUM_COLORS - 1)
    frame_col = random.choice([c for c in range(NUM_COLORS) if c != bg])
    th = random.randint(*FRAME_H_RANGE)
    tw = random.randint(*FRAME_W_RANGE)

    layout = _build_hole_layout(th, tw)
    if layout is None:
        return None
    occ, hole_placements = layout

    result = _build_solved_frame(occ, hole_placements, frame_col, bg)
    if result is None:
        return None
    solved_frame, shape_info = result

    fr = random.randint(FRAME_MARGIN, GRID_SIZE - th - FRAME_MARGIN)
    fc = random.randint(FRAME_MARGIN, GRID_SIZE - tw - FRAME_MARGIN)

    inp = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    paste(inp, np.where(occ, frame_col, bg).astype(int), fr, fc)
    if not _scatter_sprites(inp, shape_info, bg, frame_col):
        return None

    expected = _make_full_output(solved_frame, fr, fc, bg)

    try:
        if np.array_equal(main(inp), expected):
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
