import numpy as np

from arc_utils import bbox


def _complete_frame_bbox(g: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int]] = []
    for color in np.unique(g):
        mask = g == int(color)
        b = bbox(mask)
        assert b is not None
        r0, c0, r1, c1 = b
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if h < 3 or w < 3:
            continue

        frame = np.zeros_like(mask, dtype=bool)
        frame[r0 : r1 + 1, c0] = True
        frame[r0 : r1 + 1, c1] = True
        frame[r0, c0 : c1 + 1] = True
        frame[r1, c0 : c1 + 1] = True
        if np.array_equal(mask, frame):
            candidates.append(b)

    assert len(candidates) == 1
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Crop out the unique region marked by a complete monochrome rectangular frame.

    Input structure
    The input is a larger patterned grid containing exactly one color whose pixels
    form the full perimeter of a rectangle and appear nowhere else.

    Rule
    Find that complete rectangular frame and return the input cells inside its
    bounding box, including the frame itself.

    Key insight
    The wanted output is not rebuilt from the repeated background pattern; it is
    the exact subgrid selected by the only color that forms a complete frame.
    """
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1 = _complete_frame_bbox(g)
    return g[r0 : r1 + 1, c0 : c1 + 1].copy()

# --------------- generator (o5x) ---------------

# Canvas/object size choices.
CANVAS_SIZE_RANGE = (24, 29)
BLOCK_SIZES = [2, 5]
CONTENT_COUNTS = [2, 3]
FRAME_SIDES = [6, 7]
# Palette and basis-vector sampling.
COLOR_POOL = np.arange(1, 10)
BACKGROUND_COLOR = 0
PALETTE_EXTRA_COLORS = 2
ZERO_PREFIX_POSITIONS = (0, 1)
BG_WEIGHT_FLOOR_RANGE = (0.20, 0.45)
FORCED_CONTENT_BLOCK_DRAWS = 2
# Retry budget.
GENERATION_RETRIES = 300

def _gen_sample_palette(content_count: int) -> tuple[list[int], int, int, list[int]]:
    """Return content, separator, frame, and rank-order colors."""
    colors = np.random.choice(COLOR_POOL, size=content_count + PALETTE_EXTRA_COLORS, replace=False)
    content_colors = [int(x) for x in colors[:content_count]]
    sep_color = int(colors[content_count])
    frame_color = int(colors[content_count + 1])

    content_order = content_colors[:]
    np.random.shuffle(content_order)
    order = [BACKGROUND_COLOR] + content_order + [sep_color]
    return content_colors, sep_color, frame_color, order


def _gen_background_from_vector(v: np.ndarray, order: list[int]) -> np.ndarray:
    ranks = np.zeros(len(COLOR_POOL) + 1, dtype=int)
    for i, color in enumerate(order):
        ranks[color] = i

    vr = ranks[v]
    choose_row = vr[:, None] >= vr[None, :]
    return np.where(choose_row, v[:, None], v[None, :]).astype(int)


def _gen_fill_basis_vector(
    n: int, k: int, content_colors: list[int], sep_color: int
) -> np.ndarray:
    v = np.full(n, BACKGROUND_COLOR, dtype=int)
    sep_positions = set(range(k, n, k + 1))
    for pos in sep_positions:
        v[pos] = sep_color

    choices = np.array([BACKGROUND_COLOR] + content_colors, dtype=int)
    weights = np.random.dirichlet(np.ones(len(choices)))
    weights[0] = max(weights[0], np.random.uniform(*BG_WEIGHT_FLOOR_RANGE))
    weights = weights / weights.sum()

    blocks: list[list[int]] = []
    for start in range(0, n, k + 1):
        block = [pos for pos in range(start, min(start + k, n)) if pos not in sep_positions]
        if block:
            blocks.append(block)
            for pos in block:
                if pos not in ZERO_PREFIX_POSITIONS:
                    v[pos] = int(np.random.choice(choices, p=weights))

    usable_blocks = [[pos for pos in block if pos not in ZERO_PREFIX_POSITIONS] for block in blocks]
    usable_blocks = [block for block in usable_blocks if block]
    for color in content_colors:
        replace = len(usable_blocks) < FORCED_CONTENT_BLOCK_DRAWS
        block_ids = np.random.choice(len(usable_blocks), size=FORCED_CONTENT_BLOCK_DRAWS, replace=replace)
        for block_id in np.atleast_1d(block_ids):
            pos = int(np.random.choice(usable_blocks[int(block_id)]))
            v[pos] = color

    for pos in ZERO_PREFIX_POSITIONS:
        v[pos] = BACKGROUND_COLOR
    return v


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.randint(*CANVAS_SIZE_RANGE))
    k = int(np.random.choice(BLOCK_SIZES))
    content_count = int(np.random.choice(CONTENT_COUNTS))
    content_colors, sep_color, frame_color, order = _gen_sample_palette(content_count)
    v = _gen_fill_basis_vector(n, k, content_colors, sep_color)
    grid = _gen_background_from_vector(v, order)

    side = int(np.random.choice(FRAME_SIDES))
    r0 = int(np.random.randint(0, n - side + 1))
    c0 = int(np.random.randint(0, n - side + 1))
    r1 = r0 + side - 1
    c1 = c0 + side - 1
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color
    expected = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GENERATION_RETRIES} attempts")
