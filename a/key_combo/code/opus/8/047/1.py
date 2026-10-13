from __future__ import annotations

import itertools
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, find_components, get_bg, paste


BG = Color.BLACK
TEMPLATE_COLOR = Color.GREY
TEMPLATE_BBOX = (3, 3)
OBJECT_COLOR_POOL = [
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]
GRID_HEIGHT_RANGE = (17, 24)
GRID_WIDTH_RANGE = (17, 24)
OBJECT_COUNT_RANGE = (3, 4)
OBJECT_SIZE_RANGE = (5, 7)
PLACEMENT_GAP = 1
MAX_GENERATION_ATTEMPTS = 500
HOLE_COUNTS = (0, 1, 2, 3)


def _mask_key(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in mask.astype(np.uint8).ravel())


def _local_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _hole_count(mask: np.ndarray) -> int:
    padded = np.pad(mask.astype(bool), 1, constant_values=False)
    holes = 0
    h, w = padded.shape
    for comp in components_mask(~padded, connectivity=4):
        rr = comp[:, 0]
        cc = comp[:, 1]
        touches_outer = np.any((rr == 0) | (rr == h - 1) | (cc == 0) | (cc == w - 1))
        if not touches_outer:
            holes += 1
    return holes


def _rotate_clockwise(mask: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(mask, -int(quarter_turns) % 4).copy()


def _render_centered_template(slot_shape: tuple[int, int], template_mask: np.ndarray, color: int) -> np.ndarray:
    slot_h, slot_w = map(int, slot_shape)
    tpl_h, tpl_w = map(int, template_mask.shape)
    assert tpl_h <= slot_h and tpl_w <= slot_w
    top = (slot_h - tpl_h) // 2
    left = (slot_w - tpl_w) // 2
    patch = np.full((slot_h, slot_w), BG, dtype=int)
    patch[top : top + tpl_h, left : left + tpl_w][template_mask] = int(color)
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each colored shape by a centered rotated copy of the grey template, where the rotation is determined by hole count.

    Input structure:
    - The grid has a black background.
    - A unique grey 4-connected template sprite occupies a 3x3 bounding box near the lower-right.
    - The other foreground objects are non-grey monochrome 4-connected shapes, each with a distinct number of enclosed black holes.

    Rule:
    - Identify the grey template sprite and crop its 3x3 mask.
    - For each non-grey object, count how many enclosed 4-connected black hole regions it has inside its own bounding box.
    - Rotate the template clockwise by 0, 90, 180, or 270 degrees according to that hole count.
    - Replace the object's entire bounding box by a black patch containing the rotated template recolored to the object's original color and centered in the box.
    - Remove the original grey template from the output and keep all remaining background black.

    Key insight:
    - The objects' colors are preserved, but the geometry comes entirely from the shared grey template; hole topology only selects the template rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = get_bg(g)
    assert bg == BG

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert comps
    template_comps = [comp for comp in comps if int(comp["color"]) == TEMPLATE_COLOR]
    assert len(template_comps) == 1
    object_comps = [comp for comp in comps if int(comp["color"]) != TEMPLATE_COLOR]
    assert len(object_comps) >= 1

    template_mask = _local_mask(template_comps[0]["coords"])
    assert template_mask.shape == TEMPLATE_BBOX
    assert 3 <= int(template_mask.sum()) <= 5

    out = np.full(g.shape, BG, dtype=int)
    for comp in object_comps:
        color = int(comp["color"])
        coords = comp["coords"]
        r0, c0, r1, c1 = bbox_coords(coords)
        obj_mask = _local_mask(coords)
        holes = _hole_count(obj_mask)
        assert holes in HOLE_COUNTS
        rotated = _rotate_clockwise(template_mask, holes)
        patch = _render_centered_template((r1 - r0 + 1, c1 - c0 + 1), rotated, color)
        paste(out, patch, r0, c0, transparent=BG)

    return out


def _enumerate_template_masks() -> tuple[np.ndarray, ...]:
    masks: list[np.ndarray] = []
    for bits in range(1, 1 << 9):
        mask = np.array([(bits >> idx) & 1 for idx in range(9)], dtype=bool).reshape(3, 3)
        area = int(mask.sum())
        if not 3 <= area <= 5:
            continue
        if len(components_mask(mask, connectivity=4)) != 1:
            continue
        if bbox_coords(np.argwhere(mask)) != (0, 0, 2, 2):
            continue
        if len({_mask_key(np.rot90(mask, k)) for k in range(4)}) != 4:
            continue
        masks.append(mask.copy())
    assert masks
    return tuple(masks)


TEMPLATE_MASKS = _enumerate_template_masks()


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    out = np.rot90(out, random.randrange(4))
    return out.copy()


def _has_full_bbox(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    return coords.size > 0 and bbox_coords(coords) == (0, 0, mask.shape[0] - 1, mask.shape[1] - 1)


def _valid_zero_hole_blob(mask: np.ndarray) -> bool:
    if not mask.any():
        return False
    if len(components_mask(mask, connectivity=4)) != 1:
        return False
    if not _has_full_bbox(mask):
        return False
    return _hole_count(mask) == 0


def _generate_zero_hole_mask() -> np.ndarray:
    for _ in range(200):
        h = random.randint(*OBJECT_SIZE_RANGE)
        w = random.randint(*OBJECT_SIZE_RANGE)
        mask = np.ones((h, w), dtype=bool)
        min_area = max(10, h + w - 2)
        max_area = min(20, h * w - 1)
        if min_area > max_area:
            continue
        target_area = random.randint(min_area, max_area)
        stagnation = 0
        while int(mask.sum()) > target_area and stagnation < h * w * 20:
            r = random.randrange(h)
            c = random.randrange(w)
            if not mask[r, c]:
                stagnation += 1
                continue
            mask[r, c] = False
            if _valid_zero_hole_blob(mask):
                stagnation = 0
            else:
                mask[r, c] = True
                stagnation += 1
        area = int(mask.sum())
        if _valid_zero_hole_blob(mask) and min_area <= area <= max_area:
            return _random_transform(mask)
    raise RuntimeError("failed to generate a zero-hole shape")


def _separator_options(width: int, holes: int) -> list[tuple[int, ...]]:
    if holes == 1:
        return [()]
    options: list[tuple[int, ...]] = []
    for combo in itertools.combinations(range(2, width - 2), holes - 1):
        if all(b - a >= 2 for a, b in zip(combo, combo[1:])):
            options.append(tuple(int(v) for v in combo))
    return options


def _generate_holey_mask(holes: int) -> np.ndarray:
    assert holes in {1, 2, 3}
    for _ in range(200):
        h = random.randint(*OBJECT_SIZE_RANGE)
        w_min = 7 if holes == 3 else 5
        w = random.randint(w_min, OBJECT_SIZE_RANGE[1])
        options = _separator_options(w, holes)
        if not options:
            continue
        mask = np.zeros((h, w), dtype=bool)
        mask[[0, -1], :] = True
        mask[:, [0, -1]] = True
        for sep in random.choice(options):
            mask[:, sep] = True
        if _hole_count(mask) == holes:
            return _random_transform(mask)
    raise RuntimeError(f"failed to generate a {holes}-hole shape")


def _generate_object_mask(holes: int) -> np.ndarray:
    if holes == 0:
        return _generate_zero_hole_mask()
    return _generate_holey_mask(holes)


def _colorize(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _bbox_for(top: int, left: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = map(int, shape)
    return int(top), int(left), int(top + h - 1), int(left + w - 1)


def _bboxes_have_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = PLACEMENT_GAP) -> bool:
    g = int(gap)
    return a[2] + g < b[0] or b[2] + g < a[0] or a[3] + g < b[1] or b[3] + g < a[1]


def _place_objects(
    masks: list[np.ndarray], grid_shape: tuple[int, int], template_bbox: tuple[int, int, int, int]
) -> list[tuple[int, int]] | None:
    H, W = map(int, grid_shape)
    order = sorted(range(len(masks)), key=lambda idx: masks[idx].shape[0] * masks[idx].shape[1], reverse=True)
    for _ in range(200):
        placements: dict[int, tuple[int, int]] = {}
        used_bboxes = [template_bbox]
        ok = True
        for idx in order:
            h, w = map(int, masks[idx].shape)
            candidates = []
            for r in range(1, H - h):
                for c in range(1, W - w):
                    bbox = _bbox_for(r, c, masks[idx].shape)
                    if all(_bboxes_have_gap(bbox, prev) for prev in used_bboxes):
                        candidates.append((r, c, bbox))
            if not candidates:
                ok = False
                break
            r, c, bbox = random.choice(candidates)
            placements[idx] = (r, c)
            used_bboxes.append(bbox)
        if ok:
            return [placements[i] for i in range(len(masks))]
    return None


def _template_bbox_for_grid(height: int, width: int) -> tuple[int, int, int, int]:
    margin_bottom = random.randint(0, 1)
    margin_right = random.randint(0, 1)
    top = height - TEMPLATE_BBOX[0] - margin_bottom
    left = width - TEMPLATE_BBOX[1] - margin_right
    return _bbox_for(top, left, TEMPLATE_BBOX)


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    template_bbox = _template_bbox_for_grid(height, width)
    template_mask = random.choice(TEMPLATE_MASKS)

    object_count = random.randint(*OBJECT_COUNT_RANGE)
    hole_counts = random.sample(HOLE_COUNTS, object_count)
    colors = random.sample(OBJECT_COLOR_POOL, object_count)
    masks = [_generate_object_mask(holes) for holes in hole_counts]
    placements = _place_objects(masks, (height, width), template_bbox)
    if placements is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    expected = np.full((height, width), BG, dtype=int)

    tpl_r0, tpl_c0, _tpl_r1, _tpl_c1 = template_bbox
    paste(grid, _colorize(template_mask, TEMPLATE_COLOR), tpl_r0, tpl_c0, transparent=BG)

    for mask, color, holes, (top, left) in zip(masks, colors, hole_counts, placements):
        paste(grid, _colorize(mask, color), top, left, transparent=BG)
        rotated = _rotate_clockwise(template_mask, holes)
        transformed = _render_centered_template(mask.shape, rotated, color)
        paste(expected, transformed, top, left, transparent=BG)

    if not _validate_generated_case(grid, expected):
        return None
    return grid, expected


def _validate_generated_case(grid: np.ndarray, expected: np.ndarray) -> bool:
    if grid.ndim != 2 or expected.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if get_bg(grid) != BG or get_bg(expected) != BG:
        return False

    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=True)
    grey = [comp for comp in comps if int(comp["color"]) == TEMPLATE_COLOR]
    others = [comp for comp in comps if int(comp["color"]) != TEMPLATE_COLOR]
    if len(grey) != 1 or len(others) not in {3, 4}:
        return False

    template_mask = _local_mask(grey[0]["coords"])
    if template_mask.shape != TEMPLATE_BBOX:
        return False
    if not 3 <= int(template_mask.sum()) <= 5:
        return False
    if len({_mask_key(np.rot90(template_mask, k)) for k in range(4)}) != 4:
        return False

    hole_counts = sorted(_hole_count(_local_mask(comp["coords"])) for comp in others)
    if any(h not in HOLE_COUNTS for h in hole_counts):
        return False
    if len(set(hole_counts)) != len(hole_counts):
        return False

    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_case()
        if built is not None:
            grid, _expected = built
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 047")
