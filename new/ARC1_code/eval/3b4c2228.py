import numpy as np

from arc_utils import Color, colors_present, find_components, get_bg, paste
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the full green 2x2 blocks and show that count as blue cells on the main diagonal of a 3x3 grid.

    Input structure:
    - The background is black.
    - The only foreground colors are red and green.
    - Each monochromatic 4-connected foreground object is either a single pixel or a solid 2x2 block.

    Rule:
    - Count the green objects whose bounding box is exactly 2x2 and whose area is four.
    - Return a black 3x3 grid with that many blue cells filled from the top-left along the main diagonal.

    Key insight:
    - Red objects and isolated green pixels are distractors; the output encodes only the number of complete green squares.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"

    foreground = set(colors_present(g, exclude={bg}))
    assert foreground <= {Color.RED, Color.GREEN}, "expected only red and green foreground"
    assert Color.GREEN in foreground, "expected green foreground objects"

    green_blocks = 0
    for obj in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        r0, c0, r1, c1 = obj["bbox"]
        area = int(obj["area"])
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        assert (area, height, width) in {
            (1, 1, 1),
            (4, 2, 2),
        }, "expected foreground objects to be single pixels or solid 2x2 blocks"

        if int(obj["color"]) == Color.GREEN and area == 4:
            green_blocks += 1

    assert 0 <= green_blocks <= 3, "green block count must fit the 3x3 diagonal"
    out = np.full((3, 3), bg, dtype=int)
    for i in range(green_blocks):
        out[i, i] = Color.BLUE
    return out

# --------------- generator (o5x) ---------------

# Canvas and object sizes.
GRID_SIZE_RANGE = (5, 8)
BLOCK_SIZE = 2
PIXEL_SIZE = 1
OUTPUT_SIZE = 3

# Object count choices.
GREEN_BLOCK_COUNT_RANGE = (1, 3)
RED_BLOCK_COUNTS = range(0, 4)
GREEN_PIXEL_COUNTS = range(0, 5)
RED_PIXEL_COUNTS = range(0, 4)
TOTAL_OBJECT_COUNT_RANGE = (4, 11)

# Colors and retry limits.
BG_COLOR = Color.BLACK
TARGET_COLOR = Color.GREEN
DISTRACTOR_COLOR = Color.RED
ANSWER_COLOR = Color.BLUE
MAX_GENERATION_ATTEMPTS = 300


def _gen_count_options(n_green_blocks, height, width):
    options = []
    for red_blocks in RED_BLOCK_COUNTS:
        for green_pixels in GREEN_PIXEL_COUNTS:
            for red_pixels in RED_PIXEL_COUNTS:
                total_objects = n_green_blocks + red_blocks + green_pixels + red_pixels
                if not TOTAL_OBJECT_COUNT_RANGE[0] <= total_objects <= TOTAL_OBJECT_COUNT_RANGE[1]:
                    continue
                green_cells = BLOCK_SIZE * BLOCK_SIZE * n_green_blocks + green_pixels
                red_cells = BLOCK_SIZE * BLOCK_SIZE * red_blocks + red_pixels
                black_cells = height * width - green_cells - red_cells
                if black_cells > max(green_cells, red_cells):
                    options.append((red_blocks, green_pixels, red_pixels))
    return options


def _gen_can_place(grid, size, r0, c0, color):
    height, width = grid.shape
    r1, c1 = r0 + size, c0 + size
    if r0 < 0 or c0 < 0 or r1 > height or c1 > width:
        return False
    if np.any(grid[r0:r1, c0:c1] != BG_COLOR):
        return False

    color = int(color)
    for rr in range(r0, r1):
        for cc in range(c0, c1):
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                inside_object = r0 <= nr < r1 and c0 <= nc < c1
                if 0 <= nr < height and 0 <= nc < width and not inside_object:
                    if int(grid[nr, nc]) == color:
                        return False
    return True


def _gen_place_object(grid, color, size):
    patch = np.full((size, size), int(color), dtype=int)
    height, width = grid.shape
    locations = [
        (r, c)
        for r in range(height - size + 1)
        for c in range(width - size + 1)
        if _gen_can_place(grid, size, r, c, color)
    ]
    if not locations:
        return False
    r0, c0 = random.choice(locations)
    paste(grid, patch, r0, c0)
    return True


def _gen_object_sequence(n_green_blocks, red_blocks, green_pixels, red_pixels):
    block_objects = (
        [(TARGET_COLOR, BLOCK_SIZE)] * n_green_blocks
        + [(DISTRACTOR_COLOR, BLOCK_SIZE)] * red_blocks
    )
    pixel_objects = (
        [(TARGET_COLOR, PIXEL_SIZE)] * green_pixels
        + [(DISTRACTOR_COLOR, PIXEL_SIZE)] * red_pixels
    )
    random.shuffle(block_objects)
    random.shuffle(pixel_objects)
    return block_objects + pixel_objects


def _gen_attempt():
    n_green_blocks = random.randint(*GREEN_BLOCK_COUNT_RANGE)
    height = random.randint(*GRID_SIZE_RANGE)
    width = random.randint(*GRID_SIZE_RANGE)

    count_options = _gen_count_options(n_green_blocks, height, width)
    if not count_options:
        return None
    red_blocks, green_pixels, red_pixels = random.choice(count_options)

    grid = np.zeros((height, width), dtype=int)
    for color, size in _gen_object_sequence(n_green_blocks, red_blocks, green_pixels, red_pixels):
        if not _gen_place_object(grid, color, size):
            return None

    expected = np.full((OUTPUT_SIZE, OUTPUT_SIZE), int(BG_COLOR), dtype=int)
    for i in range(n_green_blocks):
        expected[i, i] = ANSWER_COLOR
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input():
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
