from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import (  # noqa: E402
    bbox_coords,
    components_mask,
    crop_bbox,
    find_components,
    get_bg,
    paste,
    random_free_location,
    scale_sprite,
)


BG = 0
FRAME_COLOR = 5
MOTIF_COLOR = 2
SPRITE_COLORS = [1, 3, 4, 6, 7, 8, 9]
BASE_SIDE = 3
MAX_GEN_ATTEMPTS = 400

BASE_SHAPES = [
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [1, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
]

MOTIF_MASK = np.array([[1, 1], [1, 0]], dtype=bool)


def _orientations(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for flip in (False, True):
        cur = np.fliplr(mask) if flip else mask
        for k in range(4):
            rot = np.rot90(cur, k)
            key = (rot.shape, rot.tobytes())
            if key not in seen:
                seen.add(key)
                out.append(rot.copy())
    return out


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min((m.shape, m.tobytes()) for m in _orientations(mask))


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _shape_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _largest_frame_component(grid: np.ndarray) -> dict:
    bg = int(get_bg(grid))
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    frame_candidates = [comp for comp in comps if int(comp["color"]) == FRAME_COLOR]
    assert frame_candidates, "expected a gray frame"
    max_area = max(int(comp["area"]) for comp in frame_candidates)
    winners = [comp for comp in frame_candidates if int(comp["area"]) == max_area]
    assert len(winners) == 1, "frame must be unique"
    return winners[0]


def _extract_holes(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> list[dict]:
    sub = crop_bbox(grid, frame_bbox)
    holes = []
    for coords in components_mask(sub == BG, connectivity=4):
        mask = _mask_from_coords(coords)
        r0, c0, r1, c1 = bbox_coords(coords)
        holes.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": mask,
                "coords": coords,
            }
        )
    assert 3 <= len(holes) <= 4, "expected 3 to 4 holes"
    return holes


def _extract_outside_parts(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> tuple[list[dict], np.ndarray]:
    bg = int(get_bg(grid))
    g = np.asarray(grid, dtype=int)
    outside = g.copy()
    r0, c0, r1, c1 = frame_bbox
    outside[r0 : r1 + 1, c0 : c1 + 1] = bg

    comps = find_components(outside, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    motif_candidates = [comp for comp in comps if int(comp["color"]) == MOTIF_COLOR]
    assert len(motif_candidates) == 1, "expected exactly one red motif"
    motif = _mask_from_coords(motif_candidates[0]["coords"])

    sprites = [comp for comp in comps if int(comp["color"]) not in {FRAME_COLOR, MOTIF_COLOR}]
    assert 3 <= len(sprites) <= 4, "expected 3 to 4 outside sprites"
    sprite_parts = []
    for comp in sprites:
        sprite_parts.append(
            {
                "color": int(comp["color"]),
                "mask": _mask_from_coords(comp["coords"]),
            }
        )
    return sprite_parts, motif


def _match_hole_to_sprite(hole_mask: np.ndarray, sprite_mask: np.ndarray) -> int | None:
    hh, hw = hole_mask.shape
    sh, sw = sprite_mask.shape
    for oriented in _orientations(sprite_mask):
        oh, ow = oriented.shape
        if hh % oh != 0 or hw % ow != 0:
            continue
        kr = hh // oh
        kc = hw // ow
        if kr != kc or kr <= 0:
            continue
        scaled = scale_sprite(oriented.astype(int), kr).astype(bool)
        if scaled.shape == hole_mask.shape and np.array_equal(scaled, hole_mask):
            return int(kr)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match outside sprites to interior holes, use the size ratio as a scale factor, and stamp a shared motif into each hole.

    Input structure:
    - Black background with one large gray frame component whose bounding box encloses 3 to 4 black holes.
    - Each hole is a scaled copy of exactly one outside monochrome sprite shape, where matching is rotation/flip invariant.
    - Outside the frame there are 3 to 4 colored sprites plus one red motif component shared by every hole.

    Rule:
    - Find the gray frame, then extract each black hole inside its bounding box as a separate 4-connected component.
    - For each hole, find the unique outside sprite whose shape becomes the hole after an integer nearest-neighbor scale and an optional rotation/flip.
    - Let k be the common scale factor between the sprite bbox and the hole bbox; scale the red motif by k.
    - Align the scaled motif to the hole bbox's top-left corner and paint only motif-on pixels that also lie inside the hole, using the matched sprite's color.
    - Leave the frame, the outside sprites, and the outside motif unchanged.

    Key insight:
    - The outside sprites split the task into two coupled roles: shape decides which hole receives which color, while the hole-to-sprite size ratio decides how large the shared motif must be inside that hole.
    """
    g = np.asarray(grid, dtype=int)
    frame = _largest_frame_component(g)
    frame_bbox = tuple(map(int, frame["bbox"]))
    holes = _extract_holes(g, frame_bbox)
    sprites, motif = _extract_outside_parts(g, frame_bbox)

    assignments = []
    for hole in holes:
        matches = []
        for idx, sprite in enumerate(sprites):
            scale = _match_hole_to_sprite(hole["mask"], sprite["mask"])
            if scale is not None:
                matches.append((idx, scale))
        assert len(matches) == 1, "hole-to-sprite pairing must be unique"
        assignments.append(matches[0])

    assert len({idx for idx, _scale in assignments}) == len(holes), "each sprite must be used once"

    out = g.copy()
    for hole, (sprite_idx, scale) in zip(holes, assignments, strict=False):
        color = int(sprites[sprite_idx]["color"])
        scaled_motif = scale_sprite(motif.astype(int), scale).astype(bool)
        hr0, hc0, hr1, hc1 = hole["bbox"]
        hole_h = hr1 - hr0 + 1
        hole_w = hc1 - hc0 + 1
        ph = min(hole_h, scaled_motif.shape[0])
        pw = min(hole_w, scaled_motif.shape[1])
        patch = np.full((ph, pw), color, dtype=int)
        mask = hole["mask"][:ph, :pw] & scaled_motif[:ph, :pw]
        paste(out, patch, hr0, hc0, mask=mask, clip=False)
    return out


def _make_row_layout(rng: random.Random) -> tuple[np.ndarray, list[np.ndarray]]:
    sizes = rng.choice([[9, 3, 9], [3, 9, 3], [6, 3, 6], [3, 6, 3]])
    shapes = rng.sample(BASE_SHAPES, 3)
    frame_h = max(sizes) + 2
    frame_w = sum(sizes) + len(sizes) + 1
    frame = np.full((frame_h, frame_w), FRAME_COLOR, dtype=int)

    x = 1
    for shape, side in zip(shapes, sizes, strict=False):
        k = side // BASE_SIDE
        hole = scale_sprite(rng.choice(_orientations(shape)).astype(int), k).astype(bool)
        y = 1 + (frame_h - 2 - side) // 2
        frame[y : y + side, x : x + side][hole] = BG
        x += side + 1
    return frame, shapes


def _make_grid_layout(rng: random.Random) -> tuple[np.ndarray, list[np.ndarray]]:
    shapes = rng.sample(BASE_SHAPES, 4)
    size_map = np.array([[3, 6], [6, 3]], dtype=int)
    if rng.random() < 0.5:
        size_map = np.fliplr(size_map)
    if rng.random() < 0.5:
        size_map = np.flipud(size_map)

    cell = 6
    frame_h = 1 + cell + 1 + cell + 1
    frame_w = 1 + cell + 1 + cell + 1
    frame = np.full((frame_h, frame_w), FRAME_COLOR, dtype=int)

    idx = 0
    for rr in range(2):
        for cc in range(2):
            side = int(size_map[rr, cc])
            shape = shapes[idx]
            idx += 1
            k = side // BASE_SIDE
            hole = scale_sprite(rng.choice(_orientations(shape)).astype(int), k).astype(bool)
            r0 = 1 + rr * (cell + 1) + (cell - side) // 2
            c0 = 1 + cc * (cell + 1) + (cell - side) // 2
            frame[r0 : r0 + side, c0 : c0 + side][hole] = BG
    return frame, shapes


def _make_frame_and_shapes(rng: random.Random) -> tuple[np.ndarray, list[np.ndarray]]:
    if rng.random() < 0.62:
        frame, shapes = _make_grid_layout(rng)
    else:
        frame, shapes = _make_row_layout(rng)

    for _ in range(rng.randrange(4)):
        frame = np.rot90(frame)
    if rng.random() < 0.5:
        frame = np.fliplr(frame)
    if rng.random() < 0.5:
        frame = np.flipud(frame)
    return np.asarray(frame, dtype=int), shapes


def _validate_shape_library() -> None:
    assert len(BASE_SHAPES) >= 4
    keys = [_canonical_key(shape) for shape in BASE_SHAPES]
    assert len(keys) == len(set(keys)), "base shapes must be distinct up to rotation/flip"
    for shape in BASE_SHAPES:
        assert shape.shape == (BASE_SIDE, BASE_SIDE)
        assert shape[0, :].any() and shape[-1, :].any() and shape[:, 0].any() and shape[:, -1].any()


def _place_patch(grid: np.ndarray, occ: np.ndarray, patch: np.ndarray, rng: random.Random) -> None:
    r0, c0 = random_free_location(occ, patch, transparent=BG, border=1, padding=1)
    paste(grid, patch, r0, c0, transparent=BG)
    paste(occ, np.where(patch != BG, 11, BG), r0, c0, transparent=BG)


def _semantic_checks(inp: np.ndarray) -> None:
    out = main(inp)
    assert inp.shape == out.shape

    frame = _largest_frame_component(inp)
    frame_bbox = tuple(map(int, frame["bbox"]))
    holes = _extract_holes(inp, frame_bbox)
    sprites, motif = _extract_outside_parts(inp, frame_bbox)
    assert motif.shape == (2, 2)

    matches_per_hole = []
    for hole in holes:
        matches = []
        for idx, sprite in enumerate(sprites):
            scale = _match_hole_to_sprite(hole["mask"], sprite["mask"])
            if scale is not None:
                matches.append((idx, scale))
        assert len(matches) == 1
        matches_per_hole.append(matches[0])

    assert len({idx for idx, _scale in matches_per_hole}) == len(holes)
    scales = [scale for _idx, scale in matches_per_hole]
    assert any(scale > 1 for scale in scales)
    dup_scales = {scale for scale in scales if scales.count(scale) > 1}
    assert dup_scales, "generator must include repeated sizes so size alone is insufficient"

    changed = out != inp
    assert changed.any()
    changed_inside = changed[frame_bbox[0] : frame_bbox[2] + 1, frame_bbox[1] : frame_bbox[3] + 1]
    assert changed_inside.any()


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    _validate_shape_library()

    frame_patch, shapes = _make_frame_and_shapes(rng)
    frame_h, frame_w = frame_patch.shape

    min_h = max(22, frame_h + 6)
    min_w = max(22, frame_w + 8)
    if min_h > 30 or min_w > 30:
        return None
    grid_h = rng.randint(min_h, 30)
    grid_w = rng.randint(min_w, 30)
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    frame_r = rng.randint(1, grid_h - frame_h - 1)
    frame_c = rng.randint(1, grid_w - frame_w - 1)
    paste(grid, frame_patch, frame_r, frame_c)

    occ = np.zeros_like(grid)
    occ[frame_r : frame_r + frame_h, frame_c : frame_c + frame_w] = 11

    sprite_colors = rng.sample(SPRITE_COLORS, len(shapes))
    items = list(zip(shapes, sprite_colors, strict=False))
    rng.shuffle(items)
    for shape, color in items:
        sprite_mask = rng.choice(_orientations(shape))
        _place_patch(grid, occ, _shape_patch(sprite_mask, color), rng)

    motif_mask = rng.choice(_orientations(MOTIF_MASK))
    _place_patch(grid, occ, _shape_patch(motif_mask, MOTIF_COLOR), rng)

    try:
        _semantic_checks(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        seed = random.randrange(1 << 30)
        rng = random.Random(seed)
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
