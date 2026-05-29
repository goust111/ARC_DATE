# Combo variant of 67e490f4 (index 4)
# combo_spec: D6/0+D8/0
# Name: 多框架自由布局

from __future__ import annotations

from collections import Counter, defaultdict
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
# a) Concrete behavior and trace:
#    - Original task: one large symmetric frame is cropped, its holes are matched against outside sprites,
#      and the single solved crop is returned.
#    - Combo task: there are 2-3 separate small frames scattered freely on the canvas, each with its own
#      hole set. The outside sprite pool is shared globally, so the same shape family may fill holes in
#      different frames. Example trace:
#      old: identify largest frame -> blank one bbox -> solve one crop.
#      new: identify every rectangular hole-bearing frame -> blank all frame bboxes -> build one global
#      shape->color lookup -> solve every frame -> stitch solved frame crops in reading order.
#    - D8/0 removes the original mirror placement constraint inside each frame. A frame may contain an
#      L-hole near the top edge and a bar near the lower-right with no mirrored counterparts.
# b) Complexity calibration:
#    - Keep 2-3 frames and 6-10 total holes so the multi-target behavior is obvious but not crowded.
#    - Keep 3-4 global canonical shapes and force at least one shape to appear in multiple frames, so the
#      shared sprite dictionary has visible repeated evidence.
#    - Each frame carries only 2-4 holes, which keeps frame detection and shape matching readable.
# c) Implicit-assumption audit of the solver:
#    - "largest component is the frame" is invalid now; the solver must scan all monochrome components and
#      retain only rectangular border-solid components with at least one internal hole.
#    - "outside sprites live outside one bbox" is invalid; every detected frame bbox must be blanked out
#      before collecting sprites.
#    - "output shape equals one frame bbox" is invalid; output is now a left-to-right stitch of solved
#      frame crops ordered by input reading order.
#    - Hole matching still relies on D8 shape equivalence. The generator therefore guarantees that every
#      hole key exists in the sprite lookup and that winning colors are strict pluralities.
# d) Placement legality:
#    - Hole cells never touch a frame border, never overlap, and remain 4-disconnected.
#    - Frames are placed by reserving their entire bounding rectangles, so sprites cannot land inside
#      holes or touch frame borders.
#    - Sprites are scattered with Chebyshev padding, preventing accidental mergers.
# e) Visual observability:
#    - Samples always contain multiple frames, at least one cross-frame shared shape, and every frame's
#      hole mask breaks LR, UD, and 180-degree symmetry. That makes both "multi-frame" and "free layout"
#      visible from only a few examples.
# f) Visual cleanliness:
#    - Frame colors are distinct from one another and from fill colors in the same puzzle.
#    - Hollow or border-like sprite distractors are not generated, keeping frame detection unambiguous.
# g) Color inferability:
#    - Colors are still inferred only from repeated sprite families: same shape (up to rotation/flip)
#      means same candidate family, and the most frequent sprite color wins. No ordering by color id is
#      needed because the generator enforces a unique plurality for every shape.
# h) New solver assumptions and generator guarantees:
#    - _collect_frames() assumes a valid frame has a solid monochrome bbox border and at least one hole;
#      generator constructs every frame exactly that way.
#    - _render_frame() assumes each hole key maps to exactly one plurality color; generator sets winner
#      counts strictly above every loser count.
#    - _stitch_frames() assumes reading-order sorting is the intended frame order; generator derives the
#      expected output using the same ordering on placed frame top-left coordinates.


FrameKey = tuple[tuple[int, int], bytes]


def _all_rot_flip(mask: np.ndarray):
    """Yield (key, oriented_mask) over all D8 variants of a boolean mask."""
    base = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(base) if flip else base
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes()), m2


def _canonical_key(mask: np.ndarray) -> FrameKey:
    return min(key for key, _ in _all_rot_flip(mask))


def _all_orientations(mask: np.ndarray) -> list[np.ndarray]:
    seen: set[FrameKey] = set()
    out: list[np.ndarray] = []
    for key, oriented in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            out.append(oriented)
    return out


def _mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = bbox_coords(coords)
    rel = coords - np.array([r0, c0])
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask, (int(r0), int(c0))


def _extract_holes(sub: np.ndarray, frame_col: int):
    hole_mask = sub != int(frame_col)
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


def _build_sprite_lookup(
    grid: np.ndarray, bg: int, frames
) -> dict[FrameKey, dict[int, int]]:
    outside = np.array(grid, dtype=int).copy()
    for frame in frames:
        r0, c0, r1, c1 = map(int, frame["bbox"])
        outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)

    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    sprites_by_key: dict[FrameKey, dict[int, int]] = {}
    for comp in spr_comps:
        coords = comp["coords"]
        vals = [int(v) for v in grid[coords[:, 0], coords[:, 1]]]
        color = int(Counter(vals).most_common(1)[0][0])
        mask, _ = _mask_from_coords(coords)
        for key, _ in _all_rot_flip(mask):
            bucket = sprites_by_key.setdefault(key, {})
            bucket[color] = int(bucket.get(color, 0)) + 1
    return sprites_by_key


def _plurality_color(cmap: dict[int, int]) -> int:
    best = max(int(v) for v in cmap.values())
    winners = [int(color) for color, count in cmap.items() if int(count) == int(best)]
    assert len(winners) == 1
    return int(winners[0])


def _render_frame(frame: dict, sprites_by_key: dict[FrameKey, dict[int, int]]) -> np.ndarray:
    th, tw = map(int, frame["shape"])
    frame_col = int(frame["color"])
    out = np.full((th, tw), int(frame_col), dtype=int)
    for key, (hr0, hc0), mask in frame["holes"]:
        cmap = sprites_by_key.get(key)
        assert cmap is not None
        fill_col = _plurality_color(cmap)
        paste(out, np.full(mask.shape, int(fill_col), dtype=int), int(hr0), int(hc0), mask=mask)
    return out


def _stitch_frames(outputs: list[np.ndarray], bg: int, gap: int) -> np.ndarray:
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


def _symmetry_flags(mask: np.ndarray) -> tuple[bool, bool, bool]:
    return (
        bool(np.array_equal(mask, np.fliplr(mask))),
        bool(np.array_equal(mask, np.flipud(mask))),
        bool(np.array_equal(mask, np.rot90(mask, 2))),
    )


def _breaks_all_old_symmetries(mask: np.ndarray) -> bool:
    lr, ud, rot180 = _symmetry_flags(mask)
    return not lr and not ud and not rot180


def main(grid: np.ndarray) -> np.ndarray:
    """
    Solve several independent hole-frames using one shared pool of outside sprites.

    Input structure:
    - The background is the most common color.
    - There are 2-3 separate small monochrome rectangular frames, each containing one or more interior
      holes on a freely arranged, non-symmetric layout.
    - Outside every frame box, disconnected sprites sit on the background. Shape families are global and
      may correspond to holes that appear in different frames.

    Rule:
    - Detect every valid frame, crop each frame bbox, and ignore all frame bboxes when collecting the
      outside sprites.
    - For every hole in every frame, match its shape against the global sprite families up to rotation
      and reflection, then choose the unique most frequent sprite color for that shape.
    - Fill each frame independently and stitch the solved frame crops left-to-right in input reading
      order, leaving one background column between neighboring crops.

    Key insight:
    - The puzzle is still driven by shape-coded color lookup, but now the lookup table is shared across
      multiple separate targets and the hole layouts no longer follow the original symmetry shortcut.
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
FRAME_LAYOUT_RETRIES = 120

SHAPE_COUNT_RANGE = (3, 4)
MIN_HOLES_PER_FRAME = 2
MAX_HOLES_PER_FRAME = 4
TOTAL_HOLES_RANGE = (6, 10)
SHAPE_MAX_AREA = 8

WINNING_BONUS_RANGE = (1, 3)
LOSER_COLOR_MAX = 2
SPRITE_PADDING = 1
OUTPUT_GAP = 1

MAX_GEN_ATTEMPTS = 500


def _build_shape_library():
    raw = [
        [[1]],
        [[1, 1]],
        [[1], [1]],
        [[1, 1, 1]],
        [[1], [1], [1]],
        [[1, 1, 1, 1]],
        [[1, 1], [1, 1]],
        [[1, 0], [1, 1]],
        [[1, 1], [0, 1]],
        [[1, 1], [1, 0]],
        [[1, 0], [1, 0], [1, 1]],
        [[1, 1, 1], [0, 1, 0]],
        [[1, 1, 1], [1, 0, 0]],
        [[1, 0], [1, 1], [0, 1]],
    ]
    seen: set[FrameKey] = set()
    out: list[np.ndarray] = []
    for shape in raw:
        mask = np.array(shape, dtype=bool)
        if int(np.sum(mask)) > SHAPE_MAX_AREA:
            continue
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


def _list_valid_placements(occ: np.ndarray, shape: np.ndarray):
    th, tw = map(int, occ.shape)
    valid: list[tuple[np.ndarray, int, int]] = []
    for ori in _all_orientations(shape):
        sh, sw = map(int, ori.shape)
        for pr in range(1, th - sh):
            for pc in range(1, tw - sw):
                if _can_place_hole(occ, ori, pr, pc):
                    valid.append((ori, int(pr), int(pc)))
    return valid


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
        valid = _list_valid_placements(occ, shape)
        if not valid:
            return None
        ori, pr, pc = random.choice(valid)
        _carve_hole(occ, ori, pr, pc)
        placements.append((shape, ori, int(pr), int(pc)))

    density = float(np.sum(occ)) / float(height * width)
    if not (FRAME_DENSITY_RANGE[0] <= density <= FRAME_DENSITY_RANGE[1]):
        return None

    hole_mask = ~occ
    if not _breaks_all_old_symmetries(hole_mask):
        return None
    if int(np.sum(hole_mask.any(axis=1))) < 3 or int(np.sum(hole_mask.any(axis=0))) < 3:
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

    key_to_frames: dict[FrameKey, set[int]] = defaultdict(set)
    for fi, items in enumerate(frames):
        for shape in items:
            key_to_frames[_canonical_key(shape)].add(fi)
    if max(len(v) for v in key_to_frames.values()) < 2:
        return None
    return global_shapes, frames


def _render_frame_pair(occ: np.ndarray, placements, frame_col: int, bg: int, fill_by_key):
    height, width = map(int, occ.shape)
    inp_patch = np.where(occ, int(frame_col), int(bg)).astype(int)
    out_patch = np.full((height, width), int(frame_col), dtype=int)
    for shape, ori, pr, pc in placements:
        fill = int(fill_by_key[_canonical_key(shape)])
        paste(out_patch, np.full(ori.shape, int(fill), dtype=int), int(pr), int(pc), mask=ori)
    return inp_patch, out_patch


def _place_rect_object(canvas: np.ndarray, placement: np.ndarray, patch: np.ndarray, bg: int, padding: int):
    block_color = (int(bg) + 1) % NUM_COLORS
    if block_color == int(bg):
        block_color = (block_color + 1) % NUM_COLORS
    block = np.full(patch.shape, int(block_color), dtype=int)
    r, c = random_free_location(placement, block, transparent=bg, padding=padding)
    paste(placement, block, int(r), int(c), transparent=bg)
    paste(canvas, patch, int(r), int(c))
    return int(r), int(c)


def _place_sprite(inp: np.ndarray, placement: np.ndarray, mask: np.ndarray, color: int, bg: int) -> bool:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    try:
        r, c = random_free_location(placement, patch, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, patch, int(r), int(c), transparent=bg)
    paste(inp, patch, int(r), int(c), transparent=bg)
    return True


def _scatter_sprites(inp: np.ndarray, placement: np.ndarray, global_shapes, hole_counts, fill_by_key, sprite_palette, bg: int) -> bool:
    for shape in global_shapes:
        key = _canonical_key(shape)
        fill_color = int(fill_by_key[key])
        orientations = _all_orientations(shape)
        winning = int(hole_counts[key] + random.randint(*WINNING_BONUS_RANGE))
        for _ in range(winning):
            if not _place_sprite(inp, placement, random.choice(orientations), fill_color, bg):
                return False

        losers = [int(c) for c in sprite_palette if int(c) != int(fill_color)]
        random.shuffle(losers)
        n_loser_colors = random.randint(0, min(LOSER_COLOR_MAX, len(losers)))
        for color in losers[:n_loser_colors]:
            copies = random.randint(1, winning - 1)
            for _ in range(copies):
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

    frame_layouts = []
    for items in frame_shape_items:
        built = None
        for _ in range(FRAME_LAYOUT_RETRIES):
            height = random.randint(*FRAME_H_RANGE)
            min_width = max(FRAME_W_RANGE[0], 5 + max(0, len(items) - 2))
            width = random.randint(min_width, FRAME_W_RANGE[1])
            built = _build_frame_layout(height, width, items)
            if built is not None:
                frame_layouts.append((height, width, built[0], built[1], items))
                break
        if built is None:
            return None

    frame_palette = [c for c in range(NUM_COLORS) if c != bg]
    if len(frame_palette) < n_frames:
        return None
    frame_colors = random.sample(frame_palette, n_frames)

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
    for frame_col, (_height, _width, occ, placements, _items) in zip(frame_colors, frame_layouts):
        inp_patch, out_patch = _render_frame_pair(occ, placements, int(frame_col), bg, fill_by_key)
        try:
            fr, fc = _place_rect_object(inp, placement, inp_patch, bg, padding=FRAME_PADDING)
        except ValueError:
            return None
        frames_with_positions.append((int(fr), int(fc), out_patch))
        for shape, _ori, _pr, _pc in placements:
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


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    """Internal audit helper used during local validation of the combo behavior."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frames = _collect_frames(g, bg)

    key_to_frames: dict[FrameKey, set[int]] = defaultdict(set)
    nonsym = 0
    for fi, frame in enumerate(frames):
        r0, c0, r1, c1 = map(int, frame["bbox"])
        sub = crop_bbox(g, (r0, c0, r1, c1))
        hole_mask = sub != int(frame["color"])
        if _breaks_all_old_symmetries(hole_mask):
            nonsym += 1
        for key, _pos, _mask in frame["holes"]:
            key_to_frames[key].add(fi)

    shared = sum(1 for owners in key_to_frames.values() if len(owners) >= 2)
    return {
        "frame_count": int(len(frames)),
        "nonsymmetric_frames": int(nonsym),
        "shared_shape_keys": int(shared),
    }
