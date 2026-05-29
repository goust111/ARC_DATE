# Combo variant of 67e490f4 (index 16)
# combo_spec: D3/1+D6/0+D11/1
# Name: 多框架+一对多广播+唯一颜色

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
# Design analysis for the combo variant
#
# combo_analysis:
#   a_concretization_and_behavior_trace:
#     old_scene:
#       - One large frame contained many holes.
#       - Outside the frame, several same-shape sprites could appear in competing
#         colors, and the most frequent color won for that shape.
#     new_scene:
#       - The canvas now has 2-3 separate rectangular frames.
#       - Across all frames, the same canonical hole shape may recur several
#         times, including in different frames.
#       - Outside every frame bbox, each canonical shape appears exactly once as
#         one sprite, in one unique color.
#     local_trace:
#       - Suppose an L-shape hole appears twice in the left frame and once in the
#         right frame.
#       - Original logic would have searched for multiple outside L-sprites and
#         counted colors.
#       - Combo logic sees exactly one outside L-sprite, say teal. Its color is
#         broadcast to all three L-holes, even across frame boundaries.
#     visible_difference:
#       - Multiple targets must be solved in one sample.
#       - There is no same-shape color competition outside.
#       - One exemplar sprite can explain several holes in more than one frame.
#
#   b_complexity_calibration:
#     - Use 2-3 frames so the multi-target behavior is obvious but readable.
#     - Use 3-4 canonical shapes and 7-11 total holes so every sample has
#       repeated evidence, yet still fits in 30x30 with spacing.
#     - Enforce total_holes_per_shape >= 2 and at least one shape spanning
#       multiple frames; otherwise D11 or D6 would become visually weak.
#     - Keep fill colors unique across shapes to avoid hidden ambiguity.
#
#   c_implicit_assumption_audit_for_main:
#     - "Largest non-bg component is the frame" breaks under D6, so main() must
#       scan all monochrome components and filter for rectangular hole-bearing
#       frames by border test and minimum size.
#     - "Blank one bbox to collect sprites" also breaks; every detected frame bbox
#       must be removed before sprite lookup.
#     - "Hole key is orientation-specific" is no longer needed for assignment.
#       Under D11/1 one sprite should match every rotated/flipped copy, so both
#       holes and sprites are canonicalized under rotation/flip.
#     - "One shape may map to several colors and need tie-break" breaks under
#       D3/1. main() now assumes one unique color per canonical shape and asserts
#       that duplicate sprites, if any, do not disagree.
#     - "Output is one crop" breaks under D6. Solved frame crops are stitched in
#       frame reading order with a one-column bg gap.
#
#   d_placement_legality:
#     - Frame placement reserves each full bbox, not only frame-colored cells, so
#       later sprites cannot land inside holes.
#     - Holes stay strictly inside frame borders and remain 4-disconnected.
#     - Sprites use Chebyshev padding 1, keeping components separable under the
#       solver's 8-connectivity scan.
#     - Exactly one sprite is emitted per canonical shape, so the global lookup
#       cannot accidentally reintroduce plurality logic.
#
#   e_visual_observability:
#     - A human can see several cropped frames in the output and several separate
#       frames in the input.
#     - At least one shape always appears in more than one frame, so cross-frame
#       broadcasting is not a rare accident.
#     - Because each shape has one distinct outside exemplar color, the absence of
#       frequency competition is visible from just a few examples.
#
#   f_visual_cleanliness:
#     - Frame colors are distinct from bg and from sprite-fill colors.
#     - Frame count and hole count stay modest, and frame padding keeps objects
#       spatially separated.
#     - No distractor sprites are added; that would only obscure the fused rule.
#
#   g_color_inference:
#     - Color choice never depends on numeric color ids or invisible ordering.
#     - Each color assignment is supported by repeated same-shape holes, often in
#       multiple frames, so the rule is structural and human-visible.
#
#   h_post_implementation_metric:
#     - For generated samples we check:
#         1. sprite_count_per_shape == 1 for every canonical shape
#         2. total_holes_per_shape >= 2 for every canonical shape
#         3. max(frames_touched_per_shape) >= 2
#       These three numbers quantify D3/1 + D11/1 + D6/0 jointly.
#
#   i_new_main_assumptions_and_generator_guarantees:
#     - main() assumes each detected frame has at least one hole; generator only
#       places frames built from explicit hole layouts.
#     - main() assumes each canonical hole shape has exactly one outside color
#       exemplar; generator emits exactly one sprite per global shape.
#     - main() assumes non-frame sprites are smaller than frames; generator keeps
#       sprite bboxes well below the frame-size threshold used in detection.
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
    """Smallest key across all rotation/flip variants."""
    return min(key for key, _ in _all_rot_flip(mask))


def _all_orientations(mask: np.ndarray):
    """Deduplicated list of all distinct rotation/flip variants."""
    seen = set()
    result = []
    for key, oriented in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            result.append(oriented)
    return result


def _mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Convert component coords into a tight boolean mask plus its top-left anchor."""
    r0, c0, r1, c1 = bbox_coords(coords)
    rel = coords - np.array([r0, c0])
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask, (int(r0), int(c0))


def _extract_holes(sub: np.ndarray, frame_col: int):
    """Extract 4-connected holes inside one frame crop."""
    hole_mask = sub != int(frame_col)
    holes = []
    for coords in components_mask(hole_mask, connectivity=4):
        mask, (hr0, hc0) = _mask_from_coords(coords)
        holes.append((_canonical_key(mask), (int(hr0), int(hc0)), mask))
    return holes


def _collect_frames(grid: np.ndarray, bg: int):
    """Find all rectangular monochrome frame components that contain at least one hole."""
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    frames = []
    for comp in comps:
        color = int(comp["color"])
        bb = tuple(map(int, comp["bbox"]))
        sub = crop_bbox(grid, bb)
        th, tw = map(int, sub.shape)
        if min(th, tw) < DETECT_MIN_FRAME_SIDE:
            continue
        if not (
            np.all(sub[0, :] == color)
            and np.all(sub[-1, :] == color)
            and np.all(sub[:, 0] == color)
            and np.all(sub[:, -1] == color)
        ):
            continue
        holes = _extract_holes(sub, color)
        if not holes:
            continue
        frames.append({"bbox": bb, "color": color, "shape": (th, tw), "holes": holes})
    frames.sort(key=lambda item: (int(item["bbox"][0]), int(item["bbox"][1])))
    return frames


def _build_sprite_lookup(grid: np.ndarray, bg: int, frames):
    """Build a global canonical shape -> unique sprite color mapping from outside all frames."""
    outside = np.array(grid, dtype=int).copy()
    for frame in frames:
        r0, c0, r1, c1 = map(int, frame["bbox"])
        outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)

    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    sprite_color_by_key: dict[tuple[tuple[int, int], bytes], int] = {}
    for comp in spr_comps:
        coords = comp["coords"]
        vals = [int(v) for v in grid[coords[:, 0], coords[:, 1]]]
        sprite_col = int(Counter(vals).most_common(1)[0][0])
        mask, _ = _mask_from_coords(coords)
        key = _canonical_key(mask)
        prev = sprite_color_by_key.get(key)
        assert prev is None or prev == sprite_col
        sprite_color_by_key[key] = sprite_col
    return sprite_color_by_key


def _render_frame(frame: dict, sprite_color_by_key: dict[tuple[tuple[int, int], bytes], int]) -> np.ndarray:
    """Solve one frame crop using the global broadcast dictionary."""
    th, tw = map(int, frame["shape"])
    frame_col = int(frame["color"])
    out = np.full((th, tw), int(frame_col), dtype=int)
    for key, (hr0, hc0), mask in frame["holes"]:
        fill_col = sprite_color_by_key.get(key)
        assert fill_col is not None
        paste(out, np.full(mask.shape, int(fill_col), dtype=int), int(hr0), int(hc0), mask=mask)
    return out


def _stitch_frames(outputs: list[np.ndarray], bg: int, gap: int = 1) -> np.ndarray:
    """Place solved frame crops left-to-right with a one-column background separator."""
    assert outputs
    max_h = max(int(arr.shape[0]) for arr in outputs)
    total_w = sum(int(arr.shape[1]) for arr in outputs) + int(gap) * (len(outputs) - 1)
    out = np.full((max_h, total_w), int(bg), dtype=int)
    c0 = 0
    for arr in outputs:
        h, w = map(int, arr.shape)
        out[:h, c0 : c0 + w] = arr
        c0 += w + int(gap)
    return out


DETECT_MIN_FRAME_SIDE = 6
OUTPUT_GAP = 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Solve several separate frames by broadcasting one unique sprite color per shape across the whole canvas.

    Input structure:
    - The background is the most common color.
    - There are 2-3 detached rectangular monochrome frames, each containing one or more interior holes.
    - Outside all frame boxes, each canonical hole shape appears exactly once as a small sprite in its own color.

    Rule:
    - Detect every frame, crop each frame bbox, and ignore all of those bboxes while collecting outside sprites.
    - Match holes and sprites by canonical shape under rotation/reflection.
    - Use the single sprite color for a shape to fill every matching hole in every frame, then stitch the solved frame crops left-to-right in reading order with a one-column background gap.

    Key insight:
    - The outside sprites form one global broadcast dictionary: each sprite is a unique shape-colored exemplar whose color must be copied into all same-shape holes, even when those holes are split across different frames.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    frames = _collect_frames(g, bg)
    assert 2 <= len(frames) <= 3

    sprite_color_by_key = _build_sprite_lookup(g, bg, frames)
    solved = [_render_frame(frame, sprite_color_by_key) for frame in frames]
    return _stitch_frames(solved, bg, gap=OUTPUT_GAP)


# --------------- generator ---------------

NUM_COLORS = 10
GRID_SIZE = 30

FRAME_COUNT_RANGE = (2, 3)
FRAME_H_RANGE = (6, 8)
FRAME_W_RANGE = (6, 10)
FRAME_DENSITY_RANGE = (0.55, 0.88)
FRAME_PADDING = 2
FRAME_LAYOUT_RETRIES = 120

SHAPE_COUNT_RANGE = (3, 4)
MIN_HOLES_PER_FRAME = 2
MAX_HOLES_PER_FRAME = 5
TOTAL_HOLES_RANGE = (7, 11)
HOLE_PLACEMENT_RETRIES = 140

SPRITE_PADDING = 1

MAX_GEN_ATTEMPTS = 500


def _build_shape_library():
    raw = [
        [[1]],
        [[1, 1]],
        [[1], [1]],
        [[1, 1, 1]],
        [[1], [1], [1]],
        [[1, 1], [1, 1]],
        [[1, 0], [1, 1]],
        [[1, 1], [0, 1]],
        [[1, 1], [1, 0]],
        [[1, 0], [1, 0], [1, 1]],
        [[1, 1, 1], [0, 1, 0]],
        [[1, 1, 1], [1, 0, 0]],
        [[1, 1], [0, 1], [0, 1]],
    ]
    seen = set()
    out = []
    for shape in raw:
        mask = np.array(shape, dtype=bool)
        key = _canonical_key(mask)
        if key not in seen:
            seen.add(key)
            out.append(mask)
    return out


SHAPE_LIBRARY = _build_shape_library()


def _shape_area(mask: np.ndarray) -> int:
    return int(np.sum(mask))


def _shape_key_map(shapes: list[np.ndarray]):
    return {_canonical_key(shape): shape for shape in shapes}


def _can_place_hole(occ: np.ndarray, mask: np.ndarray, pr: int, pc: int) -> bool:
    """Check bounds, overlap, and 4-disconnection for one hole placement."""
    th, tw = map(int, occ.shape)
    sh, sw = map(int, mask.shape)
    cells = set()
    for dr in range(sh):
        for dc in range(sw):
            if not bool(mask[dr, dc]):
                continue
            rr, cc = int(pr + dr), int(pc + dc)
            if rr < 1 or cc < 1 or rr >= th - 1 or cc >= tw - 1:
                return False
            cells.add((rr, cc))

    for rr, cc in cells:
        if not bool(occ[rr, cc]):
            return False
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if 0 <= nr < th and 0 <= nc < tw and not bool(occ[nr, nc]) and (nr, nc) not in cells:
                return False
    return True


def _carve_hole(occ: np.ndarray, mask: np.ndarray, pr: int, pc: int) -> None:
    """Mark placed hole cells as background within the frame occupancy grid."""
    sh, sw = map(int, mask.shape)
    for dr in range(sh):
        for dc in range(sw):
            if bool(mask[dr, dc]):
                occ[int(pr + dr), int(pc + dc)] = False


def _build_frame_layout(height: int, width: int, shape_items: list[np.ndarray]):
    """Place the requested hole multiset inside one rectangular frame."""
    occ = np.ones((height, width), dtype=bool)
    placements = []
    items = sorted(shape_items, key=lambda shape: (_shape_area(shape), shape.shape[0] * shape.shape[1]), reverse=True)

    for shape in items:
        orientations = _all_orientations(shape)
        success = False
        for _ in range(HOLE_PLACEMENT_RETRIES):
            oriented = random.choice(orientations)
            sh, sw = map(int, oriented.shape)
            if sh > height - 2 or sw > width - 2:
                continue
            r_max = height - sh - 1
            c_max = width - sw - 1
            if r_max < 1 or c_max < 1:
                continue
            pr = random.randint(1, r_max)
            pc = random.randint(1, c_max)
            if _can_place_hole(occ, oriented, pr, pc):
                _carve_hole(occ, oriented, pr, pc)
                placements.append((shape, oriented, int(pr), int(pc)))
                success = True
                break
        if not success:
            return None

    density = float(np.sum(occ)) / float(height * width)
    if not (FRAME_DENSITY_RANGE[0] <= density <= FRAME_DENSITY_RANGE[1]):
        return None
    return occ, placements


def _build_frame_for_items(items: list[np.ndarray]):
    """Try several frame sizes until the requested hole multiset fits cleanly."""
    for _ in range(FRAME_LAYOUT_RETRIES):
        height = random.randint(*FRAME_H_RANGE)
        width_floor = max(FRAME_W_RANGE[0], 5 + max(0, len(items) - 2))
        if width_floor > FRAME_W_RANGE[1]:
            width_floor = FRAME_W_RANGE[1]
        width = random.randint(width_floor, FRAME_W_RANGE[1])
        built = _build_frame_layout(height, width, items)
        if built is not None:
            return built
    return None


def _assign_shapes_to_frames(n_frames: int):
    """Assign repeated canonical shapes to frames with guaranteed global broadcast structure."""
    n_shapes = random.randint(SHAPE_COUNT_RANGE[0], SHAPE_COUNT_RANGE[1])
    global_shapes = random.sample(SHAPE_LIBRARY, n_shapes)
    frames = [[] for _ in range(n_frames)]
    counts = {_canonical_key(shape): 0 for shape in global_shapes}

    shared_shape = random.choice(global_shapes)
    share_count = random.randint(2, n_frames)
    shared_frames = random.sample(range(n_frames), share_count)
    for fi in shared_frames:
        frames[fi].append(shared_shape)
        counts[_canonical_key(shared_shape)] += 1

    for shape in global_shapes:
        key = _canonical_key(shape)
        while counts[key] < 2:
            choices = [idx for idx in range(n_frames) if len(frames[idx]) < MAX_HOLES_PER_FRAME]
            if not choices:
                return None
            idx = random.choice(choices)
            frames[idx].append(shape)
            counts[key] += 1

    while any(len(items) < MIN_HOLES_PER_FRAME for items in frames):
        candidates = [idx for idx in range(n_frames) if len(frames[idx]) < MAX_HOLES_PER_FRAME]
        if not candidates:
            return None
        idx = min(candidates, key=lambda i: len(frames[i]))
        shape = random.choice(global_shapes)
        frames[idx].append(shape)
        counts[_canonical_key(shape)] += 1

    min_total = max(TOTAL_HOLES_RANGE[0], sum(len(items) for items in frames))
    max_total = min(TOTAL_HOLES_RANGE[1], n_frames * MAX_HOLES_PER_FRAME)
    if min_total > max_total:
        return None
    target_total = random.randint(min_total, max_total)
    while sum(len(items) for items in frames) < target_total:
        choices = [idx for idx in range(n_frames) if len(frames[idx]) < MAX_HOLES_PER_FRAME]
        if not choices:
            break
        idx = random.choice(choices)
        shape = random.choice(global_shapes)
        frames[idx].append(shape)
        counts[_canonical_key(shape)] += 1

    frame_sets: dict[tuple[tuple[int, int], bytes], set[int]] = {}
    for fi, items in enumerate(frames):
        for shape in items:
            frame_sets.setdefault(_canonical_key(shape), set()).add(fi)

    if min(counts.values()) < 2:
        return None
    if max(len(v) for v in frame_sets.values()) < 2:
        return None
    return global_shapes, frames


def _choose_frame_colors(bg: int, count: int):
    palette = [c for c in range(NUM_COLORS) if c != bg]
    return random.sample(palette, count)


def _render_frame_pair(occ: np.ndarray, placements, frame_col: int, bg: int, fill_by_key):
    """Render both the input-style frame patch and the solved output patch."""
    height, width = map(int, occ.shape)
    inp_patch = np.where(occ, int(frame_col), int(bg)).astype(int)
    out_patch = np.full((height, width), int(frame_col), dtype=int)
    for shape, oriented, pr, pc in placements:
        fill = int(fill_by_key[_canonical_key(shape)])
        paste(out_patch, np.full(oriented.shape, fill, dtype=int), int(pr), int(pc), mask=oriented)
    return inp_patch, out_patch


def _place_rect_object(canvas: np.ndarray, placement: np.ndarray, patch: np.ndarray, bg: int, padding: int):
    """Place a full rectangular bbox and reserve that rectangle in the placement mask."""
    block_color = (int(bg) + 1) % NUM_COLORS
    if block_color == int(bg):
        block_color = (block_color + 1) % NUM_COLORS
    block = np.full(patch.shape, int(block_color), dtype=int)
    r, c = random_free_location(placement, block, transparent=bg, padding=padding)
    paste(placement, block, r, c, transparent=bg)
    paste(canvas, patch, r, c)
    return int(r), int(c)


def _place_sprite(inp: np.ndarray, placement: np.ndarray, mask: np.ndarray, color: int, bg: int) -> bool:
    """Place one monochrome sprite onto the background without touching existing objects."""
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    try:
        r, c = random_free_location(placement, patch, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, patch, r, c, transparent=bg)
    paste(inp, patch, r, c, transparent=bg)
    return True


def _scatter_sprites(inp: np.ndarray, placement: np.ndarray, global_shapes: list[np.ndarray], fill_by_key, bg: int) -> bool:
    """Place exactly one sprite for each canonical hole shape."""
    for shape in global_shapes:
        key = _canonical_key(shape)
        fill_color = int(fill_by_key[key])
        orientations = _all_orientations(shape)
        if not _place_sprite(inp, placement, random.choice(orientations), fill_color, bg):
            return False
    return True


def _stitch_expected(frames_with_positions, bg: int) -> np.ndarray:
    ordered = sorted(frames_with_positions, key=lambda item: (int(item[0]), int(item[1])))
    outputs = [patch for _r, _c, patch in ordered]
    return _stitch_frames(outputs, bg, gap=OUTPUT_GAP)


def _gen_attempt() -> np.ndarray | None:
    """Build several frame crops plus one global sprite per shape, then validate with main()."""
    bg = random.randint(0, NUM_COLORS - 1)
    n_frames = random.randint(*FRAME_COUNT_RANGE)

    assigned = _assign_shapes_to_frames(n_frames)
    if assigned is None:
        return None
    global_shapes, frame_shape_items = assigned

    frame_layouts = []
    for items in frame_shape_items:
        built = _build_frame_for_items(items)
        if built is None:
            return None
        frame_layouts.append((built[0], built[1], items))

    frame_colors = _choose_frame_colors(bg, n_frames)
    shape_key_to_shape = _shape_key_map(global_shapes)
    sprite_palette = [c for c in range(NUM_COLORS) if c != bg and c not in frame_colors]
    if len(sprite_palette) < len(shape_key_to_shape):
        return None
    fill_colors = random.sample(sprite_palette, len(shape_key_to_shape))
    fill_by_key = {key: color for key, color in zip(shape_key_to_shape.keys(), fill_colors)}

    inp = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    placement = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)

    frames_with_positions = []
    for frame_col, (occ, placements, _items) in zip(frame_colors, frame_layouts):
        inp_patch, out_patch = _render_frame_pair(occ, placements, int(frame_col), bg, fill_by_key)
        try:
            fr, fc = _place_rect_object(inp, placement, inp_patch, bg, padding=FRAME_PADDING)
        except ValueError:
            return None
        frames_with_positions.append((fr, fc, out_patch))

    if not _scatter_sprites(inp, placement, global_shapes, fill_by_key, bg):
        return None

    expected = _stitch_expected(frames_with_positions, bg)
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate input after {MAX_GEN_ATTEMPTS} attempts")
