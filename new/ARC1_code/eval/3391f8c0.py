from __future__ import annotations

import numpy as np

from arc_utils import colors_present, crop_bbox, find_components, get_bg, is_contiguous, paste
import itertools
import random


SpriteCopy = tuple[tuple[int, int, int, int], np.ndarray]


def _sprite_copies_by_color(g: np.ndarray, bg: int, colors: list[int]) -> dict[int, list[SpriteCopy]]:
    copies: dict[int, list[SpriteCopy]] = {color: [] for color in colors}
    for comp in find_components(g, bg=bg, connectivity=8, monochromatic=True):
        color = int(comp["color"])
        assert color in copies, "component color must be one of the two foreground colors"

        b = comp["bbox"]
        region = crop_bbox(g, b)
        assert np.all((region == bg) | (region == color)), "each sprite bbox must contain only its sprite and bg"

        mask = crop_bbox(g == color, b)
        assert int(mask.sum()) == int(comp["area"]), "sprite bbox must not enclose another same-color sprite"
        copies[color].append((b, mask))
    return copies


def _templates(copies: dict[int, list[SpriteCopy]], colors: list[int]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for color in colors:
        masks = [mask for _, mask in copies[color]]
        assert masks, "each foreground color must have at least one sprite"
        shape = masks[0].shape
        assert all(mask.shape == shape for mask in masks), "same-color sprite bboxes must have one size"
        template = masks[0]
        assert all(np.array_equal(mask, template) for mask in masks), "same-color sprites must share one mask"
        out[color] = template
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Exchange the two foreground sprite masks while keeping every copy in its original bounding box.

    Input structure:
    - The dominant color is the background.
    - Exactly two foreground colors are present.
    - Each foreground color forms one or more copies of a single 8-connected sprite mask.
    - The two sprite masks have the same bounding-box size.

    Rule:
    - Clear every sprite bounding box.
    - In each cleared box, draw the other foreground color's template mask in that other color.
    - Apply the replacements simultaneously, so all original boxes are cleared before any template is redrawn.

    Key insight:
    The colors identify two reusable sprite templates, and each template is the replacement shape for the other color.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 2, "expected exactly two foreground colors"

    copies = _sprite_copies_by_color(g, bg, colors)
    templates = _templates(copies, colors)

    color_a, color_b = colors
    assert templates[color_a].shape == templates[color_b].shape, "the two templates must fit the same bboxes"
    other_color = {color_a: color_b, color_b: color_a}

    out = g.copy()
    for color in colors:
        for (r0, c0, r1, c1), _mask in copies[color]:
            out[r0 : r1 + 1, c0 : c1 + 1] = bg

    for color in colors:
        target_color = other_color[color]
        target_mask = templates[target_color]
        patch = np.full(target_mask.shape, bg, dtype=int)
        patch[target_mask] = target_color
        for (r0, c0, _r1, _c1), _mask in copies[color]:
            paste(out, patch, r0, c0)

    return out

# --------------- generator (o5x) ---------------

# Canvas size bounds.
GRID_HEIGHT_RANGE = (9, 11)
GRID_WIDTH_RANGE = (10, 17)
# Sprite mask shape and density constraints.
SPRITE_BOX_SHAPES = ((2, 2), (2, 3), (3, 3))
MASK_MIN_CELLS = 2
MASK_MAX_CELLS = 6
MASK_MIN_DENSITY = (1, 2)
MASK_MAX_DENSITY = (3, 4)
MASK_CONNECTIVITY = 8
# Color and copy-count sampling.
FOREGROUND_COLOR_IDS = range(1, 10)
REPEAT_COUNT_RANGE = (2, 5)
SINGLETON_FIRST_PROB = 0.5
# Placement constraints and retry limits.
EDGE_GAP = 1
PLACEMENT_NODE_LIMIT = 8000
GENERATION_RETRIES = 300


def _gen_spans_full_box(mask: np.ndarray) -> bool:
    return bool(mask.any() and mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _gen_valid_masks(h: int, w: int) -> tuple[np.ndarray, ...]:
    """Return cached connected masks whose bbox is the full sprite box."""
    key = (int(h), int(w))
    cache = getattr(_gen_valid_masks, "_cache", {})
    if key in cache:
        return cache[key]

    h, w = key
    total = h * w
    min_num, min_den = MASK_MIN_DENSITY
    max_num, max_den = MASK_MAX_DENSITY
    min_area = max(MASK_MIN_CELLS, (total * min_num + min_den - 1) // min_den)
    max_area = min(MASK_MAX_CELLS, (total * max_num) // max_den)
    valid = []
    for area in range(min_area, max_area + 1):
        for combo in itertools.combinations(range(total), area):
            mask = np.zeros((h, w), dtype=bool)
            for idx in combo:
                mask[idx // w, idx % w] = True
            if _gen_spans_full_box(mask) and is_contiguous(mask, background=0, connectivity=MASK_CONNECTIVITY):
                valid.append(mask)

    cache[key] = tuple(valid)
    setattr(_gen_valid_masks, "_cache", cache)
    return cache[key]


def _gen_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _gen_rect_overlap(a: tuple[int, int], b: tuple[int, int], shape: tuple[int, int]) -> bool:
    ar, ac = a
    br, bc = b
    h, w = shape
    return not (ar + h <= br or br + h <= ar or ac + w <= bc or bc + w <= ac)


def _gen_cells(mask: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    return [(int(r0 + r), int(c0 + c)) for r, c in np.argwhere(mask)]


def _gen_touching(cells_a: list[tuple[int, int]], cells_b: list[tuple[int, int]]) -> bool:
    for ar, ac in cells_a:
        for br, bc in cells_b:
            if abs(ar - br) <= 1 and abs(ac - bc) <= 1:
                return True
    return False


def _gen_can_place(placed, item, r0: int, c0: int, shape: tuple[int, int], new_cells) -> bool:
    color = item[0]
    for old_color, _old_mask, _old_sep, old_r, old_c, old_cells in placed:
        if _gen_rect_overlap((r0, c0), (old_r, old_c), shape):
            return False
        if old_color == color:
            if _gen_touching(new_cells, old_cells):
                return False
    return True


def _gen_place_boxes(height: int, width: int, items, shape: tuple[int, int]):
    """Place boxes without bbox overlap or same-color 8-touching cells."""
    h, w = shape
    positions = [
        (r, c) for r in range(EDGE_GAP, height - h - EDGE_GAP + 1) for c in range(EDGE_GAP, width - w - EDGE_GAP + 1)
    ]
    random.shuffle(items)
    counts = {color: sum(1 for item_color, _mask, _sep in items if item_color == color) for color, _mask, _sep in items}
    items.sort(key=lambda item: -counts[item[0]])
    placed: list[tuple[int, np.ndarray, np.ndarray, int, int, list[tuple[int, int]]]] = []
    nodes = 0

    def search(idx: int) -> bool:
        nonlocal nodes
        nodes += 1
        if nodes > PLACEMENT_NODE_LIMIT:
            return False
        if idx == len(items):
            return True

        item = items[idx]
        candidates = positions[:]
        random.shuffle(candidates)
        for r0, c0 in candidates:
            color, mask, sep_mask = item
            new_cells = _gen_cells(sep_mask, r0, c0)
            if not _gen_can_place(placed, item, r0, c0, shape, new_cells):
                continue
            placed.append((color, mask, sep_mask, r0, c0, new_cells))
            if search(idx + 1):
                return True
            placed.pop()
        return False

    if not search(0):
        return None
    return [(color, mask, r0, c0) for color, mask, _sep, r0, c0, _cells in placed]


def _gen_copy_counts() -> tuple[int, int]:
    repeat_count = random.randint(*REPEAT_COUNT_RANGE)
    if random.random() < SINGLETON_FIRST_PROB:
        return 1, repeat_count
    return repeat_count, 1


def _gen_items(colors: list[int], masks: tuple[np.ndarray, np.ndarray], counts: tuple[int, int]) -> list[tuple[int, np.ndarray, np.ndarray]]:
    color_a, color_b = colors
    mask_a, mask_b = masks
    count_a, count_b = counts
    sep_mask = mask_a | mask_b
    return [(color_a, mask_a, sep_mask)] * count_a + [(color_b, mask_b, sep_mask)] * count_b


def _gen_render_grids(height: int, width: int, colors, masks, placed):
    """Render the output grid and the swapped-mask input grid."""
    out = np.zeros((height, width), dtype=int)
    inp = np.zeros((height, width), dtype=int)

    color_a, color_b = colors
    mask_a, mask_b = masks
    replacement = {color_a: (color_b, mask_b), color_b: (color_a, mask_a)}

    for color, mask, r0, c0 in placed:
        paste(out, _gen_patch(mask, color), r0, c0)
        other_color, other_mask = replacement[color]
        paste(inp, _gen_patch(other_mask, other_color), r0, c0)

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    shape = random.choice(SPRITE_BOX_SHAPES)
    colors = random.sample(FOREGROUND_COLOR_IDS, 2)
    masks = tuple(random.sample(_gen_valid_masks(*shape), 2))
    counts = _gen_copy_counts()

    items = _gen_items(colors, masks, counts)
    placed = _gen_place_boxes(height, width, items, shape)
    if placed is None:
        return None

    inp, out = _gen_render_grids(height, width, colors, masks, placed)
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 3391f8c0 input")
