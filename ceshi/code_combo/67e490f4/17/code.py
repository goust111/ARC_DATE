# Combo variant of 67e490f4 (index 17)
# combo_spec: D3/1+D17/1
# Name: 样本块编码+唯一颜色

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste


# ---------------------------------------------------------------------------
# Design analysis for the combo variant D3/1 + D17/1
#
# variant_analysis:
#   a_concretization_and_trace:
#     scene:
#       - A large monochrome frame is still the only big monochrome object.
#       - Inside its bbox, several bg-colored holes appear in repeated shapes.
#       - On one side of the grid, there is a tidy band of separated 2-color sample
#         blocks. In each block, one shared symbol color draws a hole shape on top of
#         a tile whose background color is the answer color for that shape.
#     old_trace:
#       - In the original task, an L-shaped hole looked outside for L-shaped sprites.
#       - If those sprites appeared in red, red, and yellow, the solver counted them
#         and filled the hole red because red had the highest frequency.
#     combo_trace:
#       - Now the outside evidence is a single sample tile, for example a blue L drawn
#         on a yellow rectangle.
#       - The solver reads that block as "L-shape -> yellow" and uses the same mapping
#         for every matching hole, regardless of any frequency idea.
#     visible_difference:
#       - The outer objects are no longer answer-colored monochrome sprites.
#       - The repeated shared symbol color across all sample blocks makes the legend
#         mechanism visible, while the tile backgrounds provide the fill colors.
#
#   b_complexity_calibration:
#     - Keep 4-6 distinct canonical hole shapes and at least 5 total hole instances so
#       the task still needs several matches rather than a single lookup.
#     - Keep one sample block per shape, all aligned in a right-side band, which makes
#       the legend obvious without clutter.
#     - Keep fill colors distinct across shapes and separate from bg/frame/legend_fg,
#       so the shared symbol color is easy to spot and the mapping is unambiguous.
#
#   c_implicit_assumption_audit_for_main:
#     - Largest monochrome non-bg component is assumed to be the frame. This still
#       holds because sample blocks are deliberately multicolor.
#     - Every non-frame region inside the frame bbox is assumed to be a hole. The
#       generator still carves holes as bg and keeps the border solid frame color.
#     - Hole matching can no longer use raw oriented mask bytes, because a sample tile
#       may rotate or reflect the shape token relative to the hole. The solver must
#       canonicalize both sides under rotation/flip.
#     - Outside informative components are assumed to be exactly the sample blocks,
#       each with exactly two non-bg colors, one of which is shared globally. The
#       generator guarantees this and emits no competing outside sprite system.
#     - The original "shape -> color-count histogram" assumption is removed. The new
#       assumption is "shape -> exactly one legend payload color".
#
#   d_placement_legality:
#     - Holes stay strictly inside the frame border, are 4-disconnected, and remain
#       symmetric as in the original construction.
#     - Sample blocks are placed only in a reserved side band with bg gaps between
#       blocks and a two-column bg gap from the frame, so nothing overlaps or merges.
#     - Tile sizes are derived from shape bboxes plus a fixed margin, which keeps the
#       legend compact enough to fit within the 30x30 canvas.
#
#   e_visual_observability:
#     - Across a few examples, a human can see the same symbol color repeated in every
#       sample block while the block backgrounds vary; that makes the "dictionary"
#       reading strategy observable.
#     - Because each hole shape appears multiple times in the frame, the viewer can
#       verify that one sample entry controls several fills.
#
#   f_visual_cleanliness:
#     - The frame area contains only frame color and bg-colored holes, so the target
#       structure remains easy to parse.
#     - The legend is isolated to one side instead of mixing markers among free
#       sprites, reducing accidental visual noise.
#
#   g_change_inferability:
#     - No hidden use of numeric color ids remains. The answer color is carried by the
#       visible tile background in the sample block.
#     - Unique fill colors per shape prevent multiple legend blocks from competing for
#       the same shape, so the solver and a human both see a direct mapping.
#
#   h_variant_metric_used_after_implementation:
#     - Metric 1: every outside informative component should have exactly 2 colors.
#     - Metric 2: across all sample blocks there should be exactly 1 shared color
#       (legend_fg), and each canonical hole shape should map to exactly 1 distinct
#       payload color.
#     - Metric 3: the number of sample blocks should equal the number of distinct
#       canonical hole shapes inside the frame.
#
#   i_new_main_assumptions_checked_against_generator:
#     - main() assumes there are at least 3 sample blocks so shared-color detection by
#       intersection is meaningful; generator now enforces at least 4 distinct samples.
#     - main() assumes each sample block contributes one canonical shape key and one
#       payload color; generator builds exactly one tile per distinct hole shape.
#     - main() assumes payload colors are all different from bg, frame, and legend_fg;
#       generator samples colors from the remaining palette only.
# ---------------------------------------------------------------------------


def _all_rot_flip(mask: np.ndarray):
    """Yield (key, oriented_mask) for all 8 rotation/flip variants of a bool mask."""
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes()), m2


def _canonical_key(mask: np.ndarray):
    """Canonical representative across rotation/flip variants."""
    return min(key for key, _ in _all_rot_flip(mask))


def _all_orientations(mask: np.ndarray):
    """Deduplicated list of all distinct rotation/flip variants."""
    seen, result = set(), []
    for key, oriented in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            result.append(oriented)
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the cropped frame by learning a shape-to-color code from explicit sample blocks.

    Input structure:
    - The most common color is the background, and one large monochrome frame sits on it.
    - Inside the frame bbox, holes appear as non-frame-color regions of several small shapes.
    - Outside the frame, a side band contains separated two-color sample blocks; each block shows one
      hole shape in a shared symbol color on top of a differently colored tile.

    Rule:
    - Crop to the frame bounding box and begin with a solid canvas of the frame color.
    - Read every sample block as a mapping from its displayed shape to the tile background color,
      allowing rotation and reflection when comparing shapes.
    - Fill each frame hole with the unique color assigned to its matching sample shape.

    Key insight:
    - The outside objects are a legend, not a population to count: the repeated symbol color marks the
      shape token, and the surrounding tile color is the answer to copy into matching holes.
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

    # STEP 2) Extract holes inside the frame bbox and canonicalize their shapes.
    hole_mask = sub != int(frame_col)
    holes: list[tuple[tuple[tuple[int, int], bytes], tuple[int, int], np.ndarray]] = []
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        mask = np.zeros((hh, ww), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        holes.append((_canonical_key(mask), (int(hr0), int(hc0)), mask))

    # STEP 3) Read the outside sample blocks and recover the legend mapping.
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    legend_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert legend_comps

    color_sets = []
    for comp in legend_comps:
        coords = comp["coords"]
        colors = set(int(v) for v in g[coords[:, 0], coords[:, 1]])
        assert len(colors) == 2
        color_sets.append(colors)

    shared = set.intersection(*color_sets)
    assert len(shared) == 1
    legend_fg = next(iter(shared))

    legend_map: dict[tuple[tuple[int, int], bytes], int] = {}
    for comp in legend_comps:
        coords = comp["coords"]
        vals = g[coords[:, 0], coords[:, 1]]
        payloads = sorted({int(v) for v in vals if int(v) != legend_fg})
        assert len(payloads) == 1
        fill_color = payloads[0]

        sym_coords = coords[vals == legend_fg]
        sr0, sc0, sr1, sc1 = bbox_coords(sym_coords)
        sh = int(sr1 - sr0 + 1)
        sw = int(sc1 - sc0 + 1)
        rel = sym_coords - np.array([sr0, sc0])
        mask = np.zeros((sh, sw), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        key = _canonical_key(mask)

        prev = legend_map.get(key)
        if prev is None:
            legend_map[key] = fill_color
        else:
            assert prev == fill_color

    # STEP 4) Render the cropped output from the learned shape->color legend.
    out = np.full((th, tw), int(frame_col), dtype=int)
    for key, (hr0, hc0), hmask in holes:
        fill_color = legend_map.get(key)
        assert fill_color is not None
        patch = np.full(hmask.shape, int(fill_color), dtype=int)
        paste(out, patch, int(hr0), int(hc0), mask=hmask)

    return out


# --------------- generator ---------------

NUM_COLORS = 10
GRID_SIZE = 30

# Frame bbox dimensions
FRAME_H_RANGE = (9, 15)
FRAME_W_RANGE = (8, 13)

# Hole layout
SHAPE_COUNT_RANGE = (4, 6)
MIN_SHAPES = 4
SHAPE_MAX_AREA = 8
FRAME_DENSITY_RANGE = (0.55, 0.92)
MIN_TOTAL_HOLES = 5
PLACEMENT_RETRIES = 80
MULT_EVEN_WEIGHTS = ([1, 2], [4, 1])
MULT_ODD_WEIGHTS = ([1, 2, 3], [3, 2, 1])

# Legend construction
LEGEND_MARGIN = 1
LEGEND_GAP = 1
LEGEND_SIDE_GAP = 2
MAX_LEGEND_COLS = 2

# Generation retry budget
MAX_GEN_ATTEMPTS = 300


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
    for shape in raw:
        mask = np.array(shape, dtype=bool)
        key = _canonical_key(mask)
        if key not in seen:
            seen.add(key)
            dedup.append(mask)
    return dedup


SHAPE_LIBRARY = _build_shape_library()


def _mirror_pos(r, c, sh, sw, th, tw):
    """Return the 4-fold mirrored positions (deduplicated)."""
    return list(
        {
            (r, c),
            (r, tw - 1 - c - sw + 1),
            (th - 1 - r - sh + 1, c),
            (th - 1 - r - sh + 1, tw - 1 - c - sw + 1),
        }
    )


def _validate_positions(occ, shape, positions, th, tw):
    """Check all positions: in bounds, no overlap, and 4-disconnected from existing holes."""
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


def _build_hole_layout(th, tw):
    """Place holes inside the frame. Returns (occ, hole_placements) or None."""
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
    available.sort(key=lambda shape: (int(np.sum(shape)), shape.shape[0] * shape.shape[1], shape.shape))
    # Bias the trial order toward smaller shapes so 4-6 distinct legend entries appear
    # regularly instead of collapsing to the minimum count on tight frames.
    prefix_len = min(len(available), max(k_target + 3, MIN_SHAPES))
    priority = available[:prefix_len]
    random.shuffle(priority)
    available = priority + available[prefix_len:]

    even_dims = th % 2 == 0 or tw % 2 == 0

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

    total_holes = sum(len(positions) for _, placement_groups in hole_placements for positions in placement_groups)
    if total_holes < MIN_TOTAL_HOLES:
        return None

    return occ, hole_placements


def _assign_shape_colors(hole_placements, blocked_colors):
    """Assign one distinct fill color to each canonical hole shape."""
    blocked = set(int(c) for c in blocked_colors)
    avail_colors = [c for c in range(NUM_COLORS) if c not in blocked]
    if len(avail_colors) < len(hole_placements):
        return None
    shape_colors = random.sample(avail_colors, len(hole_placements))

    shape_info = []
    for (shape, placement_groups), fill_col in zip(hole_placements, shape_colors):
        positions = []
        for group in placement_groups:
            for pr, pc in group:
                positions.append((pr, pc))
        shape_info.append((shape, fill_col, positions))
    return shape_info


def _render_output(occ, shape_info, frame_col):
    th, tw = occ.shape
    output = np.full((th, tw), int(frame_col), dtype=int)
    for shape, fill_col, positions in shape_info:
        patch = np.full(shape.shape, int(fill_col), dtype=int)
        for pr, pc in positions:
            paste(output, patch, pr, pc, mask=shape)
    return output


def _build_legend_tile(shape, fill_color, legend_fg):
    """Create one sample block: shape in legend_fg on a tile filled with fill_color."""
    oriented = random.choice(_all_orientations(shape))
    h, w = oriented.shape
    tile = np.full((h + 2 * LEGEND_MARGIN, w + 2 * LEGEND_MARGIN), int(fill_color), dtype=int)
    patch = np.full(oriented.shape, int(legend_fg), dtype=int)
    paste(tile, patch, LEGEND_MARGIN, LEGEND_MARGIN, mask=oriented)
    return tile


def _legend_layout_dims(tiles):
    """Prefer a compact 2-column legend when it shortens the side band substantially."""
    max_h = max(int(tile.shape[0]) for tile in tiles)
    max_w = max(int(tile.shape[1]) for tile in tiles)
    n = len(tiles)
    candidates = []
    for n_cols in range(1, min(MAX_LEGEND_COLS, n) + 1):
        rows = (n + n_cols - 1) // n_cols
        total_h = rows * max_h + (rows - 1) * LEGEND_GAP
        total_w = n_cols * max_w + (n_cols - 1) * LEGEND_GAP
        if total_h > GRID_SIZE:
            continue
        candidates.append((total_w, total_h, n_cols, max_h, max_w))
    if not candidates:
        return None

    best_one = min(candidates, key=lambda cand: (cand[0], cand[1]))
    if n >= 4:
        two_col = [cand for cand in candidates if cand[2] == 2]
        if two_col:
            best_two = min(two_col, key=lambda cand: (cand[1], cand[0]))
            if best_two[0] <= best_one[0] + max_w + LEGEND_GAP and best_two[1] + max_h < best_one[1]:
                return best_two
    return best_one


def _place_legends(inp, tiles, start_col):
    dims = _legend_layout_dims(tiles)
    if dims is None:
        return False
    total_w, total_h, n_cols, max_h, max_w = dims
    band_w = GRID_SIZE - start_col
    if total_w > band_w:
        return False

    top = random.randint(0, GRID_SIZE - total_h)
    left = random.randint(0, band_w - total_w)
    order = list(range(len(tiles)))
    random.shuffle(order)

    for slot_idx, tile_idx in enumerate(order):
        row = slot_idx // n_cols
        col = slot_idx % n_cols
        slot_r = top + row * (max_h + LEGEND_GAP)
        slot_c = left + col * (max_w + LEGEND_GAP)
        tile = tiles[tile_idx]
        tr, tc = tile.shape
        r = slot_r + (max_h - tr) // 2
        c = start_col + slot_c + (max_w - tc) // 2
        paste(inp, tile, r, c)
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build frame and side legend, then validate with the solver."""
    bg = random.randint(0, NUM_COLORS - 1)
    frame_col = random.choice([c for c in range(NUM_COLORS) if c != bg])
    legend_fg = random.choice([c for c in range(NUM_COLORS) if c not in {bg, frame_col}])

    th = random.randint(*FRAME_H_RANGE)
    tw = random.randint(*FRAME_W_RANGE)
    layout = _build_hole_layout(th, tw)
    if layout is None:
        return None
    occ, hole_placements = layout

    shape_info = _assign_shape_colors(hole_placements, blocked_colors={bg, frame_col, legend_fg})
    if shape_info is None:
        return None
    output = _render_output(occ, shape_info, frame_col)

    tiles = [_build_legend_tile(shape, fill_color, legend_fg) for shape, fill_color, _ in shape_info]
    dims = _legend_layout_dims(tiles)
    if dims is None:
        return None
    legend_w = dims[0]

    max_fc = GRID_SIZE - tw - LEGEND_SIDE_GAP - legend_w
    if max_fc < 0:
        return None
    fr = random.randint(0, GRID_SIZE - th)
    fc = random.randint(0, max_fc)

    inp = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    paste(inp, np.where(occ, frame_col, bg).astype(int), fr, fc)

    legend_start = fc + tw + LEGEND_SIDE_GAP
    if not _place_legends(inp, tiles, legend_start):
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
