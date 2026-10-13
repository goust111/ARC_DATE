import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("empty mask has no bbox")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_mask(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _rotate_cw(grid: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(grid, -int(quarter_turns) % 4).copy()


def _pack_top_aligned(blocks: list[np.ndarray], bg: int) -> np.ndarray:
    max_h = max(block.shape[0] for block in blocks)
    total_w = sum(block.shape[1] for block in blocks) + (len(blocks) - 1)
    out = np.full((max_h, total_w), int(bg), dtype=int)
    c0 = 0
    for block in blocks:
        h, w = block.shape
        out[:h, c0 : c0 + w] = block
        c0 += w + 1
    return out


def _extract_objects(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = sorted(int(color) for color in np.unique(g) if int(color) != bg)
    objects = []
    for color in colors:
        mask = g == color
        if not _is_connected(mask):
            raise AssertionError(f"color {color} is not a single 4-connected object")
        r0, c0, r1, c1 = _bbox_from_mask(mask)
        crop = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        crop[mask[r0 : r1 + 1, c0 : c1 + 1]] = color
        objects.append(
            {
                "color": color,
                "area": int(mask.sum()),
                "bbox": (r0, c0, r1, c1),
                "crop": crop,
            }
        )
    return bg, objects


def _ordered_rotated_blocks(objects: list[dict], bg: int, order: list[int]) -> list[np.ndarray]:
    blocks = []
    for rank, obj_idx in enumerate(order):
        crop = objects[obj_idx]["crop"]
        blocks.append(_rotate_cw(crop, rank))
    return blocks


def _build_output(objects: list[dict], bg: int, order: list[int]) -> np.ndarray:
    return _pack_top_aligned(_ordered_rotated_blocks(objects, bg, order), bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the four cropped shapes by area, rotate each by its sorted rank, then concatenate them left-to-right.

    Input structure:
    - The input is a black-background grid containing exactly four separated 4-connected foreground shapes.
    - Each shape has a unique non-black color, a distinct pixel area, and a non-square minimal bounding box.
    - Shapes are irregular and spaced apart so they do not touch.

    Rule:
    - Crop each colored shape to its own minimal bounding box, keeping black cells inside the crop where needed.
    - Count each shape's non-black pixels and sort the four crops by area in ascending order.
    - Rotate the k-th crop in that sorted order clockwise by (k-1) quarter-turns.
    - Pack the rotated crops from left to right with one black separator column, top-aligned, padding shorter crops with black below.

    Key insight:
    - A shape's final orientation is determined only by its global area rank among all four shapes, so the solver must compare areas before deciding any rotation.
    """
    bg, objects = _extract_objects(np.asarray(grid, dtype=int))
    assert len(objects) == 4, f"expected 4 objects, found {len(objects)}"
    areas = [obj["area"] for obj in objects]
    assert len(set(areas)) == 4, f"areas must be distinct, got {areas}"
    order = [idx for idx, _obj in sorted(enumerate(objects), key=lambda pair: pair[1]["area"])]
    return _build_output(objects, bg, order)


AREA_VALUES = tuple(range(3, 9))
SHAPE_CANVAS = 5
GRID_SIDE_RANGE = (18, 25)
MARGIN = 1
MIN_BBOX_DISTANCE = 4
MAX_SHAPE_ATTEMPTS = 500
MAX_PLACEMENT_ATTEMPTS = 500
MAX_GENERATION_ATTEMPTS = 800


def _bbox_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    dr = max(0, ar0 - br1, br0 - ar1)
    dc = max(0, ac0 - bc1, bc0 - ac1)
    return int(max(dr, dc))


def _mask_signature(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in mask.astype(int).reshape(-1))


def _valid_shape(mask: np.ndarray) -> bool:
    if not _is_connected(mask):
        return False
    crop = _crop_mask(mask)
    h, w = crop.shape
    if min(h, w) < 2 or h == w:
        return False
    if int(crop.sum()) == h * w:
        return False
    if np.unique(np.argwhere(crop)[:, 0]).size == 1 or np.unique(np.argwhere(crop)[:, 1]).size == 1:
        return False
    for turns in (1, 2, 3):
        if np.array_equal(crop, _rotate_cw(crop, turns)):
            return False
    return True


def _random_shape(area: int, used_masks: set[tuple[int, ...]]) -> np.ndarray:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        canvas = np.zeros((SHAPE_CANVAS, SHAPE_CANVAS), dtype=bool)
        start = (random.randrange(SHAPE_CANVAS), random.randrange(SHAPE_CANVAS))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < SHAPE_CANVAS and 0 <= nc < SHAPE_CANVAS and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue
        for r, c in cells:
            canvas[r, c] = True
        if not _valid_shape(canvas):
            continue
        crop = _crop_mask(canvas)
        sig = _mask_signature(crop)
        if sig in used_masks:
            continue
        used_masks.add(sig)
        return crop
    raise RuntimeError(f"failed to build a valid shape of area {area}")


def _random_position(
    shape: np.ndarray,
    grid_shape: tuple[int, int],
    placed_boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    gh, gw = grid_shape
    h, w = shape.shape
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        r0 = random.randint(MARGIN, gh - h - MARGIN)
        c0 = random.randint(MARGIN, gw - w - MARGIN)
        box = (r0, c0, r0 + h - 1, c0 + w - 1)
        if all(_bbox_distance(box, other) >= MIN_BBOX_DISTANCE for other in placed_boxes):
            return box
    return None


def _place_shapes(shapes: list[np.ndarray]) -> tuple[np.ndarray, list[dict]] | None:
    gh = random.randint(*GRID_SIDE_RANGE)
    gw = random.randint(*GRID_SIDE_RANGE)
    colors = random.sample(range(1, 10), k=4)
    indices = list(range(4))
    random.shuffle(indices)
    placed_boxes: list[tuple[int, int, int, int]] = []
    placements: dict[int, dict] = {}
    for obj_idx in indices:
        box = _random_position(shapes[obj_idx], (gh, gw), placed_boxes)
        if box is None:
            return None
        placed_boxes.append(box)
        placements[obj_idx] = {"bbox": box, "color": colors[obj_idx], "shape": shapes[obj_idx]}

    objects = []
    grid = np.zeros((gh, gw), dtype=int)
    for obj_idx in range(4):
        r0, c0, r1, c1 = placements[obj_idx]["bbox"]
        color = int(placements[obj_idx]["color"])
        shape = placements[obj_idx]["shape"]
        patch = np.zeros_like(shape, dtype=int)
        patch[shape] = color
        grid[r0 : r1 + 1, c0 : c1 + 1] = np.where(shape, color, grid[r0 : r1 + 1, c0 : c1 + 1])
        objects.append(
            {
                "color": color,
                "area": int(shape.sum()),
                "bbox": (r0, c0, r1, c1),
                "crop": patch,
            }
        )
    return grid, objects


def _order_from_key(objects: list[dict], key_fn) -> list[int]:
    return [idx for idx, _obj in sorted(enumerate(objects), key=lambda pair: key_fn(pair[1], pair[0]))]


def _bottom_aligned_pack(blocks: list[np.ndarray], bg: int) -> np.ndarray:
    max_h = max(block.shape[0] for block in blocks)
    total_w = sum(block.shape[1] for block in blocks) + (len(blocks) - 1)
    out = np.full((max_h, total_w), int(bg), dtype=int)
    c0 = 0
    for block in blocks:
        h, w = block.shape
        out[max_h - h :, c0 : c0 + w] = block
        c0 += w + 1
    return out


def _validate_semantics(inp: np.ndarray) -> bool:
    try:
        bg, objects = _extract_objects(inp)
        if bg != 0 or len(objects) != 4:
            return False
        areas = [obj["area"] for obj in objects]
        if len(set(areas)) != 4:
            return False
        for obj in objects:
            crop = obj["crop"]
            mask = crop != bg
            h, w = crop.shape
            if h == w or min(h, w) < 2 or int(mask.sum()) < 3 or int(mask.sum()) > 8:
                return False
            if np.array_equal(mask, np.rot90(mask, 2)):
                return False
        boxes = [obj["bbox"] for obj in objects]
        for i in range(4):
            for j in range(i + 1, 4):
                if _bbox_distance(boxes[i], boxes[j]) < MIN_BBOX_DISTANCE:
                    return False

        area_order = _order_from_key(objects, lambda obj, _idx: obj["area"])
        row_order = _order_from_key(objects, lambda obj, idx: (obj["bbox"][0], obj["bbox"][1], idx))
        col_order = _order_from_key(objects, lambda obj, idx: (obj["bbox"][1], obj["bbox"][0], idx))
        color_order = _order_from_key(objects, lambda obj, _idx: obj["color"])
        if area_order in (row_order, col_order, color_order):
            return False

        expected = _build_output(objects, bg, area_order)
        if not np.array_equal(main(inp), expected):
            return False

        rotated_blocks = _ordered_rotated_blocks(objects, bg, area_order)
        if len({block.shape[0] for block in rotated_blocks}) == 1:
            return False
        if any(np.array_equal(block, objects[area_order[idx]]["crop"]) for idx, block in enumerate(rotated_blocks[1:], start=1)):
            return False

        if np.array_equal(expected, _build_output(objects, bg, row_order)):
            return False
        if np.array_equal(expected, _build_output(objects, bg, col_order)):
            return False
        if np.array_equal(expected, _build_output(objects, bg, color_order)):
            return False
        if np.array_equal(expected, _pack_top_aligned([objects[idx]["crop"] for idx in area_order], bg)):
            return False
        if np.array_equal(expected, _bottom_aligned_pack(rotated_blocks, bg)):
            return False
        if expected.ndim != 2 or max(expected.shape) > 30:
            return False
    except Exception:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        areas = random.sample(AREA_VALUES, k=4)
        used_masks: set[tuple[int, ...]] = set()
        try:
            shapes = [_random_shape(area, used_masks) for area in areas]
        except RuntimeError:
            continue
        placed = _place_shapes(shapes)
        if placed is None:
            continue
        grid, _objects = placed
        if _validate_semantics(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
