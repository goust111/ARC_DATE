from collections import Counter

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg, is_contiguous, paste
import random
from itertools import combinations


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the single 5x5 foreground block whose number of background holes is different from the others.

    Input structure
    - The background is the dominant color.
    - Exactly one foreground color forms several disconnected 5x5 blocks.
    - Each block has a solid foreground border and may contain background holes in its interior.

    Rule
    - Count the background pixels inside each 5x5 block's bounding box.
    - Find the one hole count that occurs for exactly one block.
    - Return that unique block cropped to its 5x5 bounding box.

    Key insight
    - The requested pattern is not determined by position; it is the outlier among the repeated blocks by internal hole count.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    objects = find_components(g, bg=bg, connectivity=4, monochromatic=True)

    assert len(objects) >= 2
    colors = {int(obj["color"]) for obj in objects}
    assert len(colors) == 1
    fg = next(iter(colors))

    crops: list[tuple[int, np.ndarray]] = []
    for obj in objects:
        crop = crop_bbox(g, obj["bbox"])
        assert crop.shape == (5, 5)
        assert int(np.sum(crop == fg)) == int(obj["area"])
        assert np.all((crop == bg) | (crop == fg))
        assert np.all(crop[0, :] == fg)
        assert np.all(crop[-1, :] == fg)
        assert np.all(crop[:, 0] == fg)
        assert np.all(crop[:, -1] == fg)
        holes = int(np.sum(crop == bg))
        crops.append((holes, crop))

    hole_counts = Counter(holes for holes, _ in crops)
    unique_holes = [holes for holes, n in hole_counts.items() if n == 1]
    assert len(unique_holes) == 1

    matches = [crop for holes, crop in crops if holes == unique_holes[0]]
    assert len(matches) == 1
    return matches[0].copy()

# --------------- generator (o5x) ---------------

# Canvas and palette bounds.
CANVAS_HEIGHT_RANGE = (14, 17)
CANVAS_WIDTH_RANGE = (18, 19)
FG_COLOR_RANGE = (1, 9)

# Block and hole structure.
BG = 0
BLOCK_SIZE = 5
INTERIOR_SIZE = 3
INTERIOR_OFFSET = 1
INTERIOR_CENTER = (1, 1)
HOLE_COUNT_RANGE = (1, 4)
COMMON_HOLE_RANGE = (2, 3)
BLOCK_COUNT_RANGE = (4, 5)
HOLE_CONNECTIVITY = 4
INTERIOR_HOLE_CELLS = tuple((r, c) for r in range(INTERIOR_SIZE) for c in range(INTERIOR_SIZE) if (r, c) != INTERIOR_CENTER)

# Layout candidates, separation, and retry policy.
MIN_BLOCK_GAP = 1
BASE_LAYOUTS = ((2, 3),)
FOUR_BLOCK_LAYOUT = (2, 2)
TALL_LAYOUT = (3, 2)
TALL_LAYOUT_MIN_HEIGHT = 17
GEN_ATTEMPTS = 300

def _gen_hole_pattern(count: int) -> tuple[tuple[int, int], ...]:
    options = []
    for combo in combinations(INTERIOR_HOLE_CELLS, count):
        if all(abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1 for a, b in combinations(combo, 2)):
            mask = np.ones((BLOCK_SIZE, BLOCK_SIZE), dtype=int)
            for r, c in combo:
                mask[r + INTERIOR_OFFSET, c + INTERIOR_OFFSET] = BG
            if is_contiguous(mask, background=BG, connectivity=HOLE_CONNECTIVITY):
                options.append(combo)
    return random.choice(options)

def _gen_block(fg: int, holes: int) -> np.ndarray:
    block = np.full((BLOCK_SIZE, BLOCK_SIZE), fg, dtype=int)
    for r, c in _gen_hole_pattern(holes):
        block[r + INTERIOR_OFFSET, c + INTERIOR_OFFSET] = BG
    return block

def _gen_axis_positions(total: int, count: int) -> list[int] | None:
    base = BLOCK_SIZE * count + MIN_BLOCK_GAP * (count - 1)
    if base > total:
        return None
    extras = [0] * (count + 1)
    for _ in range(total - base):
        extras[random.randrange(count + 1)] += 1
    positions = []
    cursor = extras[0]
    for i in range(count):
        positions.append(cursor)
        if i + 1 < count:
            cursor += BLOCK_SIZE + MIN_BLOCK_GAP + extras[i + 1]
    return positions

def _gen_locations(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    layouts = list(BASE_LAYOUTS)
    if count == FOUR_BLOCK_LAYOUT[0] * FOUR_BLOCK_LAYOUT[1]:
        layouts.append(FOUR_BLOCK_LAYOUT)
    if height >= TALL_LAYOUT_MIN_HEIGHT:
        layouts.append(TALL_LAYOUT)
    random.shuffle(layouts)
    for rows, cols in layouts:
        if rows * cols < count:
            continue
        row_positions = _gen_axis_positions(height, rows)
        col_positions = _gen_axis_positions(width, cols)
        if row_positions is None or col_positions is None:
            continue
        slots = [(r, c) for r in row_positions for c in col_positions]
        random.shuffle(slots)
        return slots[:count]
    return None

def _render_grid(height, width, fg, block_count, common_holes, unique_holes, locations):
    """Return the input grid and expected crop for the unique block."""
    unique_index = random.randrange(block_count)
    grid = np.zeros((height, width), dtype=int)
    expected = None
    for i, (r, c) in enumerate(locations):
        holes = unique_holes if i == unique_index else common_holes
        block = _gen_block(fg, holes)
        paste(grid, block, r, c, transparent=None)
        if i == unique_index:
            expected = block
    return grid, expected

def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*CANVAS_HEIGHT_RANGE)
    width = random.randint(*CANVAS_WIDTH_RANGE)
    fg = random.randint(*FG_COLOR_RANGE)
    block_count = random.randint(*BLOCK_COUNT_RANGE)
    common_holes = random.randint(*COMMON_HOLE_RANGE)
    unique_holes = random.choice([n for n in range(HOLE_COUNT_RANGE[0], HOLE_COUNT_RANGE[1] + 1) if n != common_holes])
    locations = _gen_locations(height, width, block_count)
    if locations is None:
        return None
    grid, expected = _render_grid(height, width, fg, block_count, common_holes, unique_holes, locations)
    try:
        if expected is not None and np.array_equal(main(grid), expected):
            return grid
    except AssertionError:
        pass
    return None

def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
