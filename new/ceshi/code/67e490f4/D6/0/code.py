# Variant of 67e490f4
# Dimension: D6 — 目标结构：框架的形态 (dst)
# Change: hole_fill — 单个大型单色矩形区域作为框架，bbox 为 7-13行 x 7-20列，框架色密度 0.55-0... → 多个独立框架

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

# Design analysis for the D6 variant
# a) Concrete behavior:
#    - Input no longer contains one dominant frame. Instead it contains 2-3 separated rectangular
#      monochrome frames, each with its own local hole set. Sprites stay outside every frame bbox.
#    - Shape->color matching is global across the whole canvas: one sprite family can fill same-shaped
#      holes that happen to live in different frames.
#    - Output is not one crop. Each frame is solved independently, cropped to its own bbox, then the
#      solved crops are stitched left-to-right in input reading order, separated by a single bg column.
#    - Example trace:
#      old: find unique largest frame -> clear its bbox -> build one output crop.
#      new: detect all rectangular hole-bearing frames -> clear all their bboxes -> build a single
#      global sprite dictionary -> solve every frame -> concatenate several crops.
# b) Complexity calibration:
#    - Frames are small but still large enough to hold 2-4 holes; global distinct shapes are limited
#      to 3-4 so the repeated-shape evidence appears multiple times without clutter.
#    - Generator enforces at least one canonical hole shape to appear in more than one frame; otherwise
#      the multi-frame change would be visually weak and would reduce to isolated per-frame subproblems.
# c) Original-assumption audit:
#    - "largest component is the frame" no longer holds, so main() must scan all monochrome components
#      and keep only those whose bbox border is solid and whose interior contains at least one hole.
#    - "outside = everything except one bbox" no longer holds; every frame bbox must be blanked out
#      before collecting sprites.
#    - "output shape == frame bbox" no longer holds; output assembly now depends on several solved crops.
#    - Tie-breaking by color id is avoided in the new generator by ensuring every sprite family has a
#      unique plurality color. Solver asserts that uniqueness instead of silently relying on color ids.
# d) Placement legality:
#    - Hole cells never touch frame borders and remain 4-disconnected from one another.
#    - Frames are placed using their full bbox as occupied area, so sprites cannot land inside holes.
#    - Sprites are scattered on the remaining bg with Chebyshev padding, preventing accidental merging.
# e) Visual observability:
#    - Multiple frames are always present, and at least one hole shape is reused across frames.
#    - Output concatenation preserves each frame as a distinct block with a bg separator, so the new
#      "solve many frames then assemble" behavior is visible from 2-3 examples.
# f) Visual cleanliness:
#    - Frame colors are distinct from each other and from fill colors in the same sample.
#    - Sprite counts stay modest, and no hollow sprite shapes are generated, so frame detection remains
#      visually and algorithmically unambiguous.
# g) Color inference:
#    - Colors are still determined only by repeated sprite families, not by position or id arithmetic.
#      The only color rule a human must infer is "same shape family -> most frequent sprite color".


def _all_rot_flip(mask: np.ndarray):
    """Yield (key, oriented_mask) over all rotation/flip variants of a boolean mask."""
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes()), m2


def _canonical_key(mask: np.ndarray):
    return min(key for key, _ in _all_rot_flip(mask))


def _all_orientations(mask: np.ndarray):
    seen = set()
    result = []
    for key, oriented in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            result.append(oriented)
    return result


def _mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = bbox_coords(coords)
    rel = coords - np.array([r0, c0])
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask, (int(r0), int(c0))


def _extract_holes(sub: np.ndarray, frame_col: int):
    hole_mask = (sub != int(frame_col))
    holes = []
    for coords in components_mask(hole_mask, connectivity=4):
        mask, (hr0, hc0) = _mask_from_coords(coords)
        holes.append((_canonical_key(mask), (int(hr0), int(hc0)), mask))
    return holes


def _collect_frames(grid: np.ndarray, bg: int):
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    frames = []
    for comp in comps:
        color = int(comp["color"])
        bb = tuple(map(int, comp["bbox"]))
        sub = crop_bbox(grid, bb)
        th, tw = map(int, sub.shape)
        if min(th, tw) < 5:
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
        frames.append({"bbox": bb, "color": color, "holes": holes, "shape": (th, tw)})
    frames.sort(key=lambda item: (int(item["bbox"][0]), int(item["bbox"][1])))
    return frames


def _build_sprite_lookup(grid: np.ndarray, bg: int, frames) -> dict[tuple[tuple[int, int], bytes], dict[int, int]]:
    outside = np.array(grid, dtype=int).copy()
    for frame in frames:
        r0, c0, r1, c1 = map(int, frame["bbox"])
        outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)

    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    sprites_by_key: dict[tuple[tuple[int, int], bytes], dict[int, int]] = {}
    for comp in spr_comps:
        coords = comp["coords"]
        vals = [int(v) for v in grid[coords[:, 0], coords[:, 1]]]
        scol = int(Counter(vals).most_common(1)[0][0])
        mask, _ = _mask_from_coords(coords)
        for key, _ in _all_rot_flip(mask):
            bucket = sprites_by_key.setdefault(key, {})
            bucket[scol] = int(bucket.get(scol, 0)) + 1
    return sprites_by_key


def _render_frame(frame: dict, sprites_by_key: dict[tuple[tuple[int, int], bytes], dict[int, int]]) -> np.ndarray:
    th, tw = map(int, frame["shape"])
    frame_col = int(frame["color"])
    out = np.full((th, tw), int(frame_col), dtype=int)
    for key, (hr0, hc0), mask in frame["holes"]:
        cmap = sprites_by_key.get(key)
        assert cmap is not None
        best = max(cmap.values())
        winners = [color for color, count in cmap.items() if int(count) == int(best)]
        assert len(winners) == 1
        fill_col = int(winners[0])
        paste(out, np.full(mask.shape, fill_col, dtype=int), int(hr0), int(hc0), mask=mask)
    return out


def _stitch_frames(outputs: list[np.ndarray], bg: int, gap: int = 1) -> np.ndarray:
    assert outputs
    max_h = max(int(arr.shape[0]) for arr in outputs)
    total_w = sum(int(arr.shape[1]) for arr in outputs) + int(gap) * (len(outputs) - 1)
    out = np.full((max_h, total_w), int(bg), dtype=int)
    c = 0
    for arr in outputs:
        h, w = map(int, arr.shape)
        out[:h, c : c + w] = arr
        c += w + int(gap)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill hole groups inside several small frames using one shared pool of outside sprites.

    Input structure:
    - bg is the dominant canvas color. There are 2-3 separate rectangular monochrome frames, each with
      one or more holes cut out of its interior.
    - Outside all frame boxes, small sprites are scattered on bg. Sprite families are keyed by shape and
      may correspond to holes that appear in different frames.

    Rule:
    - Detect every frame, treat each frame bbox as its own cropped target, and ignore those bboxes when
      collecting the outside sprites.
    - For each hole in each frame, find the global sprite family with the same shape up to rotation/flip.
      Fill that hole with the most frequent color among matching sprites.
    - Solve every frame independently, then stitch the solved frame crops left-to-right in input reading
      order, leaving one bg column between neighboring crops.

    Key insight: The sprites still encode a shape-to-color dictionary, but the dictionary is shared across
    multiple independent targets, so one learned shape family can fill matching holes in several frames.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    frames = _collect_frames(g, bg)
    assert 2 <= len(frames) <= 3

    sprites_by_key = _build_sprite_lookup(g, bg, frames)
    solved = [_render_frame(frame, sprites_by_key) for frame in frames]
    return _stitch_frames(solved, bg, gap=1)


# --------------- generator ---------------

NUM_COLORS = 10
GRID_SIZE = 30

FRAME_COUNT_RANGE = (2, 3)
FRAME_H_RANGE = (6, 8)
FRAME_W_RANGE = (6, 9)
FRAME_DENSITY_RANGE = (0.55, 0.88)
FRAME_PADDING = 2
FRAME_LAYOUT_RETRIES = 90

SHAPE_COUNT_RANGE = (3, 4)
MIN_HOLES_PER_FRAME = 2
MAX_HOLES_PER_FRAME = 4
TOTAL_HOLES_RANGE = (6, 10)
HOLE_PLACEMENT_RETRIES = 120

WINNING_BONUS_RANGE = (1, 3)
LOSER_COLOR_MAX = 2
SPRITE_PADDING = 1
OUTPUT_GAP = 1

MAX_GEN_ATTEMPTS = 400


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
    sh, sw = map(int, mask.shape)
    for dr in range(sh):
        for dc in range(sw):
            if bool(mask[dr, dc]):
                occ[int(pr + dr), int(pc + dc)] = False


def _build_frame_layout(height: int, width: int, shape_items: list[np.ndarray]):
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


def _assign_shapes_to_frames(n_frames: int):
    n_shapes = random.randint(SHAPE_COUNT_RANGE[0], SHAPE_COUNT_RANGE[1])
    global_shapes = random.sample(SHAPE_LIBRARY, n_shapes)
    frames = [[] for _ in range(n_frames)]

    shared_shape = random.choice(global_shapes)
    fa, fb = random.sample(range(n_frames), 2)
    frames[fa].append(shared_shape)
    frames[fb].append(shared_shape)

    for shape in global_shapes:
        if shape is shared_shape:
            continue
        frames[random.randrange(n_frames)].append(shape)

    while any(len(items) < MIN_HOLES_PER_FRAME for items in frames):
        idx = min(range(n_frames), key=lambda i: len(frames[i]))
        frames[idx].append(random.choice(global_shapes))

    min_total = max(TOTAL_HOLES_RANGE[0], sum(len(items) for items in frames))
    max_total = min(TOTAL_HOLES_RANGE[1], n_frames * MAX_HOLES_PER_FRAME)
    if min_total > max_total:
        return None
    target_total = random.randint(min_total, max_total)
    while sum(len(items) for items in frames) < target_total:
        choices = [i for i in range(n_frames) if len(frames[i]) < MAX_HOLES_PER_FRAME]
        if not choices:
            break
        idx = random.choice(choices)
        frames[idx].append(random.choice(global_shapes))

    key_counts: dict[tuple[tuple[int, int], bytes], set[int]] = {}
    for fi, items in enumerate(frames):
        for shape in items:
            key_counts.setdefault(_canonical_key(shape), set()).add(fi)
    if max(len(v) for v in key_counts.values()) < 2:
        return None
    return global_shapes, frames


def _choose_frame_colors(bg: int, count: int):
    palette = [c for c in range(NUM_COLORS) if c != bg]
    return random.sample(palette, count)


def _render_frame_pair(occ: np.ndarray, placements, frame_col: int, bg: int, fill_by_key):
    height, width = map(int, occ.shape)
    inp_patch = np.where(occ, int(frame_col), int(bg)).astype(int)
    out_patch = np.full((height, width), int(frame_col), dtype=int)
    for shape, oriented, pr, pc in placements:
        fill = int(fill_by_key[_canonical_key(shape)])
        paste(out_patch, np.full(oriented.shape, fill, dtype=int), int(pr), int(pc), mask=oriented)
    return inp_patch, out_patch


def _place_rect_object(canvas: np.ndarray, placement: np.ndarray, patch: np.ndarray, bg: int, padding: int):
    block_color = (int(bg) + 1) % NUM_COLORS
    if block_color == int(bg):
        block_color = (block_color + 1) % NUM_COLORS
    block = np.full(patch.shape, int(block_color), dtype=int)
    r, c = random_free_location(placement, block, transparent=bg, padding=padding)
    paste(placement, block, r, c, transparent=bg)
    paste(canvas, patch, r, c)
    return int(r), int(c)


def _place_sprite(inp: np.ndarray, placement: np.ndarray, mask: np.ndarray, color: int, bg: int) -> bool:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    try:
        r, c = random_free_location(placement, patch, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, patch, r, c, transparent=bg)
    paste(inp, patch, r, c, transparent=bg)
    return True


def _scatter_sprites(inp: np.ndarray, placement: np.ndarray, global_shapes: list[np.ndarray], hole_counts, fill_by_key, sprite_palette, bg: int) -> bool:
    for shape in global_shapes:
        key = _canonical_key(shape)
        fill_color = int(fill_by_key[key])
        orientations = _all_orientations(shape)
        winning = int(hole_counts[key] + random.randint(*WINNING_BONUS_RANGE))
        for _ in range(winning):
            if not _place_sprite(inp, placement, random.choice(orientations), fill_color, bg):
                return False

        loser_pool = [c for c in sprite_palette if c != fill_color]
        random.shuffle(loser_pool)
        loser_count = random.randint(0, min(LOSER_COLOR_MAX, len(loser_pool)))
        for color in loser_pool[:loser_count]:
            copies = random.randint(1, max(1, winning - 1))
            if copies >= winning:
                copies = winning - 1
            for _ in range(max(0, copies)):
                if not _place_sprite(inp, placement, random.choice(orientations), int(color), bg):
                    break
    return True


def _stitch_expected(frames_with_positions, bg: int) -> np.ndarray:
    ordered = sorted(frames_with_positions, key=lambda item: (int(item[0]), int(item[1])))
    outputs = [patch for _r, _c, patch in ordered]
    return _stitch_frames(outputs, bg, gap=OUTPUT_GAP)


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, NUM_COLORS - 1)
    n_frames = random.randint(*FRAME_COUNT_RANGE)
    assigned = _assign_shapes_to_frames(n_frames)
    if assigned is None:
        return None
    global_shapes, frame_shape_items = assigned

    common_h = random.randint(*FRAME_H_RANGE)
    frame_layouts = []
    for items in frame_shape_items:
        min_w = max(FRAME_W_RANGE[0], 5 + max(0, len(items) - 1))
        widths = list(range(min_w, FRAME_W_RANGE[1] + 1))
        random.shuffle(widths)
        built = None
        for width in widths:
            for _ in range(FRAME_LAYOUT_RETRIES):
                built = _build_frame_layout(common_h, width, items)
                if built is not None:
                    frame_layouts.append((width, built[0], built[1], items))
                    break
            if built is not None:
                break
        if built is None:
            return None

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
    hole_counts = {key: 0 for key in shape_key_to_shape}
    for frame_col, (_width, occ, placements, _items) in zip(frame_colors, frame_layouts):
        inp_patch, out_patch = _render_frame_pair(occ, placements, int(frame_col), bg, fill_by_key)
        try:
            fr, fc = _place_rect_object(inp, placement, inp_patch, bg, padding=FRAME_PADDING)
        except ValueError:
            return None
        frames_with_positions.append((fr, fc, out_patch))
        for shape, _oriented, _pr, _pc in placements:
            key = _canonical_key(shape)
            hole_counts[key] = int(hole_counts.get(key, 0)) + 1

    if max(hole_counts.values()) <= 0:
        return None

    if not _scatter_sprites(inp, placement, global_shapes, hole_counts, fill_by_key, sprite_palette, bg):
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
