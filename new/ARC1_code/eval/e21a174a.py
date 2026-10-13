import numpy as np

from arc_utils import bbox, bbox_h, colors_present, get_bg, paste
import math
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse the top-to-bottom order of the colored sprites while preserving each sprite's shape and columns.

    Input structure
    A background canvas contains one sprite mask for each foreground color; the sprites' bboxes form a contiguous vertical stack.

    Rule
    Read the foreground color masks by bbox top row, then paste those same sprite masks from the original stack top in reverse order.

    Key insight
    The sprites are defined by color masks and bbox rows, so connectivity is irrelevant; only the vertical stack order changes.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    sprites = []
    for color in colors_present(g, exclude={bg}):
        mask = g == color
        b = bbox(mask)
        assert b is not None
        r0, c0, r1, c1 = b
        crop_mask = mask[r0 : r1 + 1, c0 : c1 + 1]
        sprite = np.full(crop_mask.shape, bg, dtype=int)
        sprite[crop_mask] = color
        sprites.append((b, sprite))

    assert sprites
    sprites.sort(key=lambda item: item[0][0])
    tops = [b[0] for b, _sprite in sprites]
    assert len(tops) == len(set(tops))

    for (prev_b, _prev_sprite), (next_b, _next_sprite) in zip(sprites, sprites[1:]):
        assert prev_b[2] < next_b[0]
        assert next_b[0] == prev_b[2] + 1

    out = np.full_like(g, bg)
    row = sprites[0][0][0]
    for b, sprite in reversed(sprites):
        _r0, c0, _r1, c1 = b
        h = bbox_h(b)
        assert row + h <= g.shape[0]
        out[row : row + h, c0 : c1 + 1] = sprite
        row += h

    assert row == sprites[-1][0][2] + 1
    return out

# --------------- generator (o5x) ---------------
# Canvas and sprite bounds.
BACKGROUND_COLOR = 0
GRID_COLS_MIN, GRID_COLS_MAX = 7, 13
SPRITE_COUNT_MIN, SPRITE_COUNT_MAX = 3, 5
SPRITE_HEIGHT_MIN, SPRITE_HEIGHT_MAX = 1, 5
SPRITE_WIDTH_MIN = 3
SIDE_MARGIN_CELLS = 1
TOP_MARGIN_ROWS = 1
GRID_VERTICAL_MARGIN_ROWS = 2
STACK_HEIGHT_MIN_FLOOR, STACK_HEIGHT_MAX = 5, 11
# Mask and color constraints.
MIN_SPRITE_DENSITY = 0.5
FOREGROUND_COLORS = range(1, 10)
# Rejection limits.
GENERATION_RETRIES = 300

def _gen_sprite_heights(stack_height, count):
    """Split stack height into shuffled per-sprite heights within bounds."""
    vals = []
    remaining = stack_height
    for i in range(count):
        slots = count - i - 1
        min_v = max(SPRITE_HEIGHT_MIN, remaining - SPRITE_HEIGHT_MAX * slots)
        max_v = min(SPRITE_HEIGHT_MAX, remaining - SPRITE_HEIGHT_MIN * slots)
        v = random.randint(min_v, max_v)
        vals.append(v)
        remaining -= v
    random.shuffle(vals)
    return vals

def _gen_centered_widths(cols, count):
    parity = cols % 2
    widths = [w for w in range(SPRITE_WIDTH_MIN, cols - SIDE_MARGIN_CELLS) if w % 2 == parity]
    if not widths:
        widths = [w for w in range(SPRITE_WIDTH_MIN, cols - SIDE_MARGIN_CELLS)]
    return [random.choice(widths) for _ in range(count)]

def _gen_symmetric_mask(height, width):
    """Return an LR-symmetric mask with no empty rows or column pairs."""
    group_count = (width + 1) // 2
    group_sizes = [2] * (width // 2)
    if width % 2:
        group_sizes.append(1)

    chosen = np.zeros((height, group_count), dtype=bool)
    for r in range(height):
        chosen[r, random.randrange(group_count)] = True
    for g in range(group_count):
        chosen[random.randrange(height), g] = True

    area = int(sum(group_sizes[g] for r in range(height) for g in range(group_count) if chosen[r, g]))
    min_area = math.ceil(MIN_SPRITE_DENSITY * height * width)
    target = random.randint(max(area, min_area), height * width)

    open_cells = [(r, g) for r in range(height) for g in range(group_count) if not chosen[r, g]]
    random.shuffle(open_cells)
    for r, g in open_cells:
        if area >= target:
            break
        chosen[r, g] = True
        area += group_sizes[g]

    mask = np.zeros((height, width), dtype=bool)
    for g in range(group_count):
        rows = chosen[:, g]
        mask[rows, g] = True
        mask[rows, width - 1 - g] = True
    return mask

def _sample_stack_plan():
    """Draw dimensions and colors in the generator's sampling order."""
    num_sprites = random.randint(SPRITE_COUNT_MIN, SPRITE_COUNT_MAX)
    cols = random.randint(GRID_COLS_MIN, GRID_COLS_MAX)
    stack_height = random.randint(
        max(STACK_HEIGHT_MIN_FLOOR, num_sprites),
        min(STACK_HEIGHT_MAX, SPRITE_HEIGHT_MAX * num_sprites),
    )
    rows = stack_height + GRID_VERTICAL_MARGIN_ROWS
    heights = _gen_sprite_heights(stack_height, num_sprites)
    widths = _gen_centered_widths(cols, num_sprites)
    colors = random.sample(FOREGROUND_COLORS, num_sprites)
    return rows, cols, heights, widths, colors

def _render_stack_pair(rows, cols, heights, widths, colors):
    """Render the sampled input grid and its reversed-stack oracle output."""
    grid = np.full((rows, cols), BACKGROUND_COLOR, dtype=int)
    expected = np.full_like(grid, BACKGROUND_COLOR)
    sprites = []

    row = TOP_MARGIN_ROWS
    for color, height, width in zip(colors, heights, widths):
        mask = _gen_symmetric_mask(height, width)
        patch = np.zeros((height, width), dtype=int)
        patch[mask] = color
        c0 = (cols - width) // 2
        paste(grid, patch, row, c0, transparent=BACKGROUND_COLOR)
        sprites.append((patch, c0))
        row += height

    row = TOP_MARGIN_ROWS
    for patch, c0 in reversed(sprites):
        paste(expected, patch, row, c0, transparent=BACKGROUND_COLOR)
        row += patch.shape[0]
    return grid, expected

def _gen_attempt():
    rows, cols, heights, widths, colors = _sample_stack_plan()
    grid, expected = _render_stack_pair(rows, cols, heights, widths, colors)
    bg_count = int(np.count_nonzero(grid == BACKGROUND_COLOR))
    fg_counts = [int(np.count_nonzero(grid == color)) for color in colors]
    if bg_count <= max(fg_counts):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid

def generate_input():
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e21a174a input")
