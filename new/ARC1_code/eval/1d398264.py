import numpy as np

from arc_utils import bbox, bbox_h, bbox_w, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the eight non-center cells of a filled 3x3 block as rays to the grid edge.

    Input structure
    - The background is the dominant color.
    - All foreground pixels form exactly one filled 3x3 block.
    - The middle cell of that block is the center, and the other eight cells are compass-direction ray seeds.

    Rule
    - Keep the input grid unchanged as the base output.
    - For each non-center cell in the 3x3 block, continue its color away from the center in that cell's compass direction.
    - Paint that ray through the starting cell and onward until it leaves the grid.

    Key insight
    - The local position of a cell around the 3x3 center is also its global ray direction.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    box = bbox(g != bg)
    assert box is not None
    assert bbox_h(box) == 3
    assert bbox_w(box) == 3

    r0, c0, r1, c1 = box
    block = g[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(block != bg)

    center_r = r0 + 1
    center_c = c0 + 1
    H, W = g.shape
    out = g.copy()

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            color = int(g[center_r + dr, center_c + dc])
            r = center_r + dr
            c = center_c + dc
            while 0 <= r < H and 0 <= c < W:
                out[r, c] = color
                r += dr
                c += dc

    return out

# --------------- generator (o5x) ---------------

# Canvas and center placement constraints.
GRID_SIZE_RANGE = (10, 21)
CENTER_EDGE_DISTANCES = range(2, 4)
CENTER_FAR_EDGE_DISTANCES = range(6, 17)
# Palette and 3x3 block construction.
BACKGROUND_COLOR = 0
COLOR_VALUES = np.arange(1, 10)
SURROUND_COUNT_RANGE = (3, 5)
BLOCK_CENTER = (1, 1)
ANCHOR_SURROUND_CELLS = ((0, 0), (0, 2), (2, 0))
OPEN_SURROUND_CELLS = ((0, 1), (1, 0), (1, 2), (2, 1), (2, 2))
# Orientation, ray oracle, and retry budget.
ROTATION_COUNT = 4
LR_FLIP_CHOICES = 2
RAY_DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
GEN_RETRY_LIMIT = 300


def _gen_center(n: int) -> tuple[int, int]:
    choices = []
    for r in range(2, n - 2):
        for c in range(2, n - 2):
            distances = (r, c, n - 1 - r, n - 1 - c)
            if min(distances) in CENTER_EDGE_DISTANCES and max(distances) in CENTER_FAR_EDGE_DISTANCES:
                choices.append((r, c))
    return choices[np.random.randint(len(choices))]


def _gen_palette() -> tuple[int, list[int]]:
    n_surround = int(np.random.randint(*SURROUND_COUNT_RANGE))
    colors = np.random.choice(COLOR_VALUES, size=n_surround + 1, replace=False)
    return int(colors[0]), [int(x) for x in colors[1:]]


def _gen_block(center_color: int, surround_colors: list[int]) -> np.ndarray:
    block = np.zeros((3, 3), dtype=int)
    block[BLOCK_CENTER] = center_color

    for (r, c), color in zip(ANCHOR_SURROUND_CELLS, surround_colors):
        block[r, c] = color

    open_cells = list(OPEN_SURROUND_CELLS)
    np.random.shuffle(open_cells)
    if len(surround_colors) == 4:
        r, c = open_cells.pop()
        block[r, c] = surround_colors[3]

    for r, c in open_cells:
        block[r, c] = int(np.random.choice(surround_colors))

    block = np.rot90(block, int(np.random.randint(ROTATION_COUNT)))
    if np.random.randint(LR_FLIP_CHOICES):
        block = np.fliplr(block)
    return block.copy()


def _gen_expected_output(inp: np.ndarray) -> np.ndarray:
    """Independent oracle for the eight ray extensions from the 3x3 block."""
    box = bbox(inp != BACKGROUND_COLOR)
    assert box is not None
    r0, c0, r1, c1 = box
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2
    H, W = inp.shape
    out = inp.copy()

    for dr, dc in RAY_DIRECTIONS:
        color = int(inp[center_r + dr, center_c + dc])
        r = center_r + dr
        c = center_c + dc
        while 0 <= r < H and 0 <= c < W:
            out[r, c] = color
            r += dr
            c += dc
    return out


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.randint(*GRID_SIZE_RANGE))
    center_r, center_c = _gen_center(n)
    center_color, surround_colors = _gen_palette()
    block = _gen_block(center_color, surround_colors)
    inp = np.zeros((n, n), dtype=int)
    inp[center_r - 1 : center_r + 2, center_c - 1 : center_c + 2] = block

    expected = _gen_expected_output(inp)
    return inp if np.array_equal(main(inp), expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 1d398264 input")
