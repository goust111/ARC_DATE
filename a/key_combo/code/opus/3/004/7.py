from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists() and str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import (  # type: ignore
    bbox_coords,
    components_mask,
    crop_bbox,
    find_components,
    get_bg,
    paste,
    random_free_location,
)


BG = 0
FRAME_COLOR = 3
SPRITE_COLORS = [1, 2, 4, 5, 6, 7, 8, 9]
MAX_ATTEMPTS = 500

GRID_H_RANGE = (24, 30)
GRID_W_RANGE = (24, 30)
FRAME_H_RANGE = (11, 16)
FRAME_W_RANGE = (12, 17)
SMALL_FRAME_COUNT_CHOICES = (1, 2)
HOLE_COUNT_CHOICES = (2, 3, 4)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = map(int, bbox)
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _rectangles_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask, (int(r0), int(c0), int(r1), int(c1))


def _all_transforms(mask: np.ndarray) -> list[np.ndarray]:
    base = np.asarray(mask, dtype=bool)
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for flipped in (False, True):
        cur = np.fliplr(base) if flipped else base
        for k in range(4):
            rotated = np.rot90(cur, k)
            key = (tuple(map(int, rotated.shape)), rotated.tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(rotated.copy())
    return variants


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min((tuple(map(int, item.shape)), item.tobytes()) for item in _all_transforms(mask))


def _find_frame_candidates(grid: np.ndarray, *, bg: int, frame_color: int = FRAME_COLOR) -> list[dict]:
    candidates: list[dict] = []
    comps = find_components(
        grid,
        bg=bg,
        colors=[frame_color],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    for comp in comps:
        bbox = tuple(map(int, comp["bbox"]))
        sub = crop_bbox(grid, bbox)
        if min(sub.shape) < 3:
            continue
        if not np.all((sub == int(frame_color)) | (sub == int(bg))):
            continue
        if not np.all(sub[0, :] == int(frame_color)):
            continue
        if not np.all(sub[-1, :] == int(frame_color)):
            continue
        if not np.all(sub[:, 0] == int(frame_color)):
            continue
        if not np.all(sub[:, -1] == int(frame_color)):
            continue
        item = dict(comp)
        item["bbox"] = bbox
        item["bbox_area"] = _bbox_area(bbox)
        candidates.append(item)
    return candidates


def _extract_holes(crop: np.ndarray, *, bg: int = BG) -> list[dict]:
    holes: list[dict] = []
    for coords in components_mask(crop == int(bg), connectivity=4):
        if coords.size == 0:
            continue
        if (
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == crop.shape[0] - 1)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == crop.shape[1] - 1)
        ):
            continue
        mask, bbox = _mask_from_coords(coords)
        holes.append(
            {
                "bbox": bbox,
                "origin": (int(bbox[0]), int(bbox[1])),
                "mask": mask,
                "key": _canonical_key(mask),
                "area": int(mask.sum()),
            }
        )
    holes.sort(key=lambda hole: (hole["origin"][0], hole["origin"][1]))
    return holes


def _extract_sprites_outside(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    *,
    bg: int = BG,
    frame_color: int = FRAME_COLOR,
) -> list[dict]:
    outside = np.array(grid, dtype=int, copy=True)
    r0, c0, r1, c1 = map(int, frame_bbox)
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    sprites: list[dict] = []
    comps = find_components(
        outside,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    for comp in comps:
        color = int(comp["color"])
        if color in (int(bg), int(frame_color)):
            continue
        mask, bbox = _mask_from_coords(comp["coords"])
        sprites.append(
            {
                "bbox": bbox,
                "color": color,
                "mask": mask,
                "key": _canonical_key(mask),
                "area": int(mask.sum()),
            }
        )
    sprites.sort(key=lambda sprite: (sprite["bbox"][0], sprite["bbox"][1], sprite["color"]))
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the unique largest green frame and fill its internal holes using outside shape-matched sprites.

    Input structure:
    - The background is black, and the grid contains 2 or 3 separate green rectangular frame regions.
    - The unique largest green frame contains 2 to 4 black holes inside its bounding box, while the smaller frames are solid green rectangles.
    - Outside the largest frame bounding box there are several small monochrome sprites in non-green colors.

    Rule:
    - Detect all green rectangular frame components and choose the one with strictly largest bounding-box area.
    - Crop the grid to that frame's bounding box and extract every enclosed black 4-connected hole inside the crop.
    - Find all non-background, non-green connected sprites outside the chosen bounding box; each sprite matches exactly one hole up to rotation and mirror symmetry.
    - Fill each hole with the color of its uniquely shape-matched sprite and return the cropped filled frame.

    Key insight:
    - The largest frame selects the working crop, and the outside sprites act as color keys whose normalized shapes identify which internal hole they must fill.
    """
    g = np.asarray(grid, dtype=int)
    bg = BG

    frames = _find_frame_candidates(g, bg=bg, frame_color=FRAME_COLOR)
    assert frames, "expected at least one green frame candidate"

    max_area = max(int(frame["bbox_area"]) for frame in frames)
    largest = [frame for frame in frames if int(frame["bbox_area"]) == max_area]
    assert len(largest) == 1, "largest frame must be unique"
    frame_bbox = tuple(map(int, largest[0]["bbox"]))

    crop = crop_bbox(g, frame_bbox)
    holes = _extract_holes(crop, bg=bg)
    assert holes, "largest frame must contain holes"

    sprites = _extract_sprites_outside(g, frame_bbox, bg=bg, frame_color=FRAME_COLOR)
    assert len(sprites) == len(holes), "sprite count must match hole count"

    color_by_key: dict[tuple[tuple[int, int], bytes], int] = {}
    for sprite in sprites:
        key = sprite["key"]
        assert key not in color_by_key, "sprite shapes must match uniquely"
        color_by_key[key] = int(sprite["color"])

    out = crop.copy()
    for hole in holes:
        key = hole["key"]
        assert key in color_by_key, "every hole needs one matching sprite"
        patch = np.full(hole["mask"].shape, int(color_by_key[key]), dtype=int)
        paste(out, patch, hole["origin"][0], hole["origin"][1], mask=hole["mask"])
    return out


RAW_SHAPES = [
    [[1, 1, 1]],
    [[1, 0], [1, 1]],
    [[1, 1, 1, 1]],
    [[1, 1], [1, 1]],
    [[1, 0, 0], [1, 1, 1]],
    [[1, 1, 1], [0, 1, 0]],
    [[1, 1, 0], [0, 1, 1]],
    [[1, 1, 1, 1, 1]],
    [[1, 1], [1, 1], [1, 0]],
    [[1, 1, 1], [0, 1, 0], [0, 1, 0]],
    [[1, 0, 1], [1, 1, 1]],
    [[1, 1, 0], [0, 1, 0], [0, 1, 1]],
    [[1, 1, 1], [1, 0, 0], [1, 0, 0]],
    [[1, 1, 1], [1, 0, 1]],
    [[1, 1, 1], [1, 1, 1]],
    [[1, 0, 0], [1, 1, 1], [0, 0, 1]],
    [[1, 1, 1, 1], [1, 0, 0, 0]],
    [[1, 1, 0], [1, 1, 1]],
    [[1, 1, 1], [1, 0, 1], [1, 0, 0]],
    [[1, 1, 1], [0, 1, 1], [1, 1, 0]],
    [[1, 1, 1], [1, 0, 1], [0, 1, 0]],
    [[1, 1, 1, 1], [1, 1, 0, 0]],
    [[1, 1, 1], [1, 1, 0], [1, 0, 0]],
    [[1, 1, 1], [1, 1, 0], [0, 1, 0]],
]


def _build_shape_library() -> list[np.ndarray]:
    shapes: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for raw in RAW_SHAPES:
        mask = np.array(raw, dtype=bool)
        if not (3 <= int(mask.sum()) <= 8):
            continue
        key = _canonical_key(mask)
        if key in seen:
            continue
        seen.add(key)
        shapes.append(mask)
    return shapes


SHAPE_LIBRARY = _build_shape_library()
SHAPES_BY_AREA: dict[int, list[np.ndarray]] = {}
for _shape in SHAPE_LIBRARY:
    SHAPES_BY_AREA.setdefault(int(_shape.sum()), []).append(_shape)


def _pick_hole_bases(n_holes: int) -> list[np.ndarray]:
    pair_areas = [area for area, shapes in SHAPES_BY_AREA.items() if len(shapes) >= 2]
    assert pair_areas, "shape library must contain at least one same-area pair"

    chosen = random.sample(SHAPES_BY_AREA[random.choice(pair_areas)], 2)
    used = {_canonical_key(mask) for mask in chosen}

    pool = [mask for mask in SHAPE_LIBRARY if _canonical_key(mask) not in used]
    random.shuffle(pool)
    while len(chosen) < n_holes and pool:
        chosen.append(pool.pop())

    asym_pool = [mask for mask in SHAPE_LIBRARY if len(_all_transforms(mask)) > 1]
    if all(len(_all_transforms(mask)) == 1 for mask in chosen) and asym_pool:
        chosen[-1] = random.choice(asym_pool)

    if len(chosen) != n_holes:
        raise RuntimeError("failed to choose enough distinct hole shapes")
    return chosen


def _orient_shapes(base_shapes: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
    hole_masks: list[np.ndarray] = []
    sprite_masks: list[np.ndarray] = []
    changed = False

    for base in base_shapes:
        variants = _all_transforms(base)
        hole_mask = random.choice(variants)

        sprite_choices = variants
        if len(variants) > 1:
            different = [item for item in variants if not np.array_equal(item, hole_mask)]
            if different and (not changed or random.random() < 0.75):
                sprite_choices = different

        sprite_mask = random.choice(sprite_choices)
        if not np.array_equal(sprite_mask, hole_mask):
            changed = True

        hole_masks.append(hole_mask)
        sprite_masks.append(sprite_mask)

    if not changed:
        return None
    return hole_masks, sprite_masks


def _try_place_holes(frame_h: int, frame_w: int, hole_masks: list[np.ndarray]) -> list[dict] | None:
    masks = [np.asarray(mask, dtype=bool) for mask in hole_masks]
    order = sorted(
        range(len(masks)),
        key=lambda idx: (-int(masks[idx].sum()), -(masks[idx].shape[0] * masks[idx].shape[1])),
    )

    for _ in range(100):
        blocked = np.zeros((frame_h, frame_w), dtype=bool)
        blocked[0, :] = True
        blocked[-1, :] = True
        blocked[:, 0] = True
        blocked[:, -1] = True
        placements: list[dict | None] = [None] * len(masks)
        failed = False

        for idx in order:
            mask = masks[idx]
            coords = np.argwhere(mask)
            mh, mw = map(int, mask.shape)
            candidates: list[tuple[int, int]] = []
            for r0 in range(1, frame_h - mh):
                for c0 in range(1, frame_w - mw):
                    cells = coords + np.array([r0, c0], dtype=int)
                    if bool(blocked[cells[:, 0], cells[:, 1]].any()):
                        continue
                    candidates.append((int(r0), int(c0)))
            if not candidates:
                failed = True
                break

            r0, c0 = random.choice(candidates)
            placements[idx] = {
                "origin": (int(r0), int(c0)),
                "mask": mask,
                "key": _canonical_key(mask),
                "area": int(mask.sum()),
            }
            for rr, cc in coords + np.array([r0, c0], dtype=int):
                r_lo = max(0, int(rr) - 1)
                r_hi = min(frame_h, int(rr) + 2)
                c_lo = max(0, int(cc) - 1)
                c_hi = min(frame_w, int(cc) + 2)
                blocked[r_lo:r_hi, c_lo:c_hi] = True

        if not failed and all(item is not None for item in placements):
            return [item for item in placements if item is not None]
    return None


def _render_frame_crop(frame_h: int, frame_w: int, holes: list[dict]) -> np.ndarray:
    crop = np.full((frame_h, frame_w), FRAME_COLOR, dtype=int)
    for hole in holes:
        patch = np.full(hole["mask"].shape, BG, dtype=int)
        paste(crop, patch, hole["origin"][0], hole["origin"][1], mask=hole["mask"])
    return crop


def _render_expected_output(frame_h: int, frame_w: int, holes: list[dict], color_by_key: dict) -> np.ndarray:
    out = np.full((frame_h, frame_w), FRAME_COLOR, dtype=int)
    for hole in holes:
        patch = np.full(hole["mask"].shape, int(color_by_key[hole["key"]]), dtype=int)
        paste(out, patch, hole["origin"][0], hole["origin"][1], mask=hole["mask"])
    return out


def _place_largest_frame(grid_h: int, grid_w: int, crop: np.ndarray) -> tuple[int, int]:
    frame_h, frame_w = crop.shape
    margin_r = 2 if grid_h - frame_h >= 5 else 0
    margin_c = 2 if grid_w - frame_w >= 5 else 0
    r_lo = margin_r
    c_lo = margin_c
    r_hi = grid_h - frame_h - margin_r
    c_hi = grid_w - frame_w - margin_c
    if r_hi < r_lo:
        r_lo, r_hi = 0, grid_h - frame_h
    if c_hi < c_lo:
        c_lo, c_hi = 0, grid_w - frame_w
    return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)


def _small_frame_patch(max_area: int) -> np.ndarray | None:
    for _ in range(80):
        h = random.randint(4, 9)
        w = random.randint(4, 10)
        area = h * w
        if area >= max_area:
            continue
        if area > max_area - 8:
            continue
        return np.full((h, w), FRAME_COLOR, dtype=int)
    return None


def _build_instance() -> tuple[np.ndarray, np.ndarray, dict] | None:
    frame_h = random.randint(*FRAME_H_RANGE)
    frame_w = random.randint(*FRAME_W_RANGE)
    grid_h = random.randint(max(GRID_H_RANGE[0], frame_h + 8), GRID_H_RANGE[1])
    grid_w = random.randint(max(GRID_W_RANGE[0], frame_w + 8), GRID_W_RANGE[1])

    n_holes = random.choice(HOLE_COUNT_CHOICES)
    n_small_frames = random.choice(SMALL_FRAME_COUNT_CHOICES)

    base_shapes = _pick_hole_bases(n_holes)
    oriented = _orient_shapes(base_shapes)
    if oriented is None:
        return None
    hole_masks, sprite_masks = oriented

    holes = _try_place_holes(frame_h, frame_w, hole_masks)
    if holes is None:
        return None

    crop = _render_frame_crop(frame_h, frame_w, holes)
    largest_r, largest_c = _place_largest_frame(grid_h, grid_w, crop)
    largest_bbox = (
        int(largest_r),
        int(largest_c),
        int(largest_r + frame_h - 1),
        int(largest_c + frame_w - 1),
    )

    hole_colors = random.sample(SPRITE_COLORS, n_holes)
    color_by_key = {hole["key"]: color for hole, color in zip(holes, hole_colors)}
    expected = _render_expected_output(frame_h, frame_w, holes, color_by_key)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    bbox_canvas = np.full((grid_h, grid_w), BG, dtype=int)
    paste(grid, crop, largest_r, largest_c)
    paste(bbox_canvas, np.full(crop.shape, FRAME_COLOR, dtype=int), largest_r, largest_c)

    small_frame_bboxes: list[tuple[int, int, int, int]] = []
    for _ in range(n_small_frames):
        patch = _small_frame_patch(_bbox_area(largest_bbox))
        if patch is None:
            return None
        try:
            sr, sc = random_free_location(bbox_canvas, patch, transparent=BG, padding=1)
        except ValueError:
            return None
        paste(grid, patch, sr, sc)
        paste(bbox_canvas, patch, sr, sc)
        small_frame_bboxes.append((int(sr), int(sc), int(sr + patch.shape[0] - 1), int(sc + patch.shape[1] - 1)))

    sprite_bboxes: list[tuple[int, int, int, int]] = []
    transform_changed = False
    for hole, sprite_mask in zip(holes, sprite_masks):
        color = int(color_by_key[hole["key"]])
        bbox_patch = np.full(sprite_mask.shape, color, dtype=int)
        sprite_patch = np.full(sprite_mask.shape, BG, dtype=int)
        sprite_patch[sprite_mask] = color
        try:
            sr, sc = random_free_location(bbox_canvas, bbox_patch, transparent=BG, padding=1)
        except ValueError:
            return None
        paste(grid, sprite_patch, sr, sc, transparent=BG)
        paste(bbox_canvas, bbox_patch, sr, sc)
        sprite_bboxes.append((int(sr), int(sc), int(sr + sprite_mask.shape[0] - 1), int(sc + sprite_mask.shape[1] - 1)))
        if not np.array_equal(hole["mask"], sprite_mask):
            transform_changed = True

    if not transform_changed:
        return None

    metadata = {
        "largest_bbox": largest_bbox,
        "small_frame_bboxes": small_frame_bboxes,
        "holes": holes,
        "sprite_masks": sprite_masks,
        "sprite_bboxes": sprite_bboxes,
        "color_by_key": color_by_key,
        "expected": expected,
        "frame_count": 1 + n_small_frames,
    }
    return grid, expected, metadata


def _semantic_checks(grid: np.ndarray, expected: np.ndarray, metadata: dict) -> bool:
    if grid.ndim != 2 or expected.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False

    largest_bbox = metadata["largest_bbox"]
    holes_meta = metadata["holes"]
    sprite_masks_meta = metadata["sprite_masks"]
    frame_count = int(metadata["frame_count"])

    frames = _find_frame_candidates(grid, bg=BG, frame_color=FRAME_COLOR)
    if len(frames) != frame_count:
        return False
    areas = [int(frame["bbox_area"]) for frame in frames]
    if areas.count(max(areas)) != 1:
        return False
    actual_largest = max(frames, key=lambda frame: int(frame["bbox_area"]))
    if tuple(map(int, actual_largest["bbox"])) != tuple(map(int, largest_bbox)):
        return False

    crop = crop_bbox(grid, largest_bbox)
    holes = _extract_holes(crop, bg=BG)
    if len(holes) != len(holes_meta):
        return False
    if sorted(hole["key"] for hole in holes) != sorted(hole["key"] for hole in holes_meta):
        return False

    same_area_keys = {}
    for hole in holes:
        same_area_keys.setdefault(int(hole["area"]), []).append(hole["key"])
    if not any(len(keys) >= 2 for keys in same_area_keys.values()):
        return False

    for frame in frames:
        bbox = tuple(map(int, frame["bbox"]))
        if bbox == tuple(map(int, largest_bbox)):
            continue
        if _extract_holes(crop_bbox(grid, bbox), bg=BG):
            return False

    sprites = _extract_sprites_outside(grid, largest_bbox, bg=BG, frame_color=FRAME_COLOR)
    if len(sprites) != len(holes):
        return False
    if sorted(sprite["key"] for sprite in sprites) != sorted(hole["key"] for hole in holes):
        return False
    for sprite in sprites:
        if _rectangles_intersect(tuple(map(int, sprite["bbox"])), tuple(map(int, largest_bbox))):
            return False

    sprite_by_key = {sprite["key"]: sprite for sprite in sprites}
    if len(sprite_by_key) != len(sprites):
        return False
    if not any(not np.array_equal(sprite_by_key[hole["key"]]["mask"], hole["mask"]) for hole in holes):
        return False

    if not np.array_equal(main(grid), expected):
        return False
    if not np.array_equal(expected, metadata["expected"]):
        return False

    if len(sprite_masks_meta) != len(holes_meta):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        built = _build_instance()
        if built is None:
            continue
        grid, expected, metadata = built
        if _semantic_checks(grid, expected, metadata):
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {MAX_ATTEMPTS} attempts")
