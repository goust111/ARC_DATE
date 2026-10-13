import numpy as np

from arc_utils import bbox_coords, get_bg, in_bounds, neighbors4, paste
import itertools
import random


Block = tuple[int, int, int, tuple[int, int, int, int]]


def _solid_3x3_blocks(g: np.ndarray, bg: int) -> list[Block]:
    height, width = map(int, g.shape)
    seen = np.zeros((height, width), dtype=bool)
    blocks: list[Block] = []

    for row in range(height):
        for col in range(width):
            if seen[row, col] or int(g[row, col]) == bg:
                continue

            color = int(g[row, col])
            stack = [(row, col)]
            seen[row, col] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for nr, nc in neighbors4(r, c):
                    if (
                        in_bounds(g.shape, nr, nc)
                        and not seen[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0, r1, c1 = bbox_coords(arr)
            if (
                len(coords) == 9
                and r1 - r0 == 2
                and c1 - c0 == 2
                and np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color)
            ):
                blocks.append((r0 + 1, c0 + 1, color, (r0, c0, r1, c1)))

    return blocks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each solid 3x3 color block into a full-width and full-height cross, clearing crossings between different blocks.

    Input structure:
    - The background is the dominant color.
    - The cross generators are exact 4-connected same-color components shaped as solid 3x3 blocks.
    - These 3x3 block centers have distinct rows and distinct columns, separated enough that their cross intersections are outside every source block.

    Rule:
    - Copy the input grid.
    - For every solid 3x3 block, fill its center row and center column with the block color.
    - Where the center row of one block crosses the center column of another block, reset that pixel to background.

    Key insight:
    - The 3x3 blocks are cross generators; other foreground pixels are noise that is overwritten only if a generated cross passes through them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    blocks = _solid_3x3_blocks(g, bg)
    assert blocks, "expected at least one solid 3x3 foreground block"

    rows = [row for row, _, _, _ in blocks]
    cols = [col for _, col, _, _ in blocks]
    assert len(set(rows)) == len(rows), "expected 3x3 block centers to have unique rows"
    assert len(set(cols)) == len(cols), "expected 3x3 block centers to have unique columns"
    for i, (row_i, col_i, _, _) in enumerate(blocks):
        for row_j, col_j, _, _ in blocks[i + 1 :]:
            assert abs(row_i - row_j) > 2, "expected source blocks to occupy separate row bands"
            assert abs(col_i - col_j) > 2, "expected source blocks to occupy separate column bands"

    out = g.copy()
    for row, col, color, _ in blocks:
        out[row, :] = color
        out[:, col] = color

    for i, (row, _, _, _) in enumerate(blocks):
        for j, (_, col, _, _) in enumerate(blocks):
            if i != j:
                out[row, col] = bg

    return out

# --------------- generator (o5x) ---------------
# Canvas and block layout.
GEN_HEIGHT_RANGE = (14, 19)
GEN_WIDTH_RANGE = (13, 19)
GEN_BLOCK_COUNT_RANGE = (1, 3)
GEN_BLOCK_SIZE = 3
GEN_BLOCK_RADIUS = GEN_BLOCK_SIZE // 2
GEN_ROW_MIN_SEP = 3
GEN_COL_MIN_SEP = 4
# Palette and noise policy.
GEN_BACKGROUND = 0
GEN_COLORS = tuple(range(1, 10))
GEN_NOISE_DENSITY_RANGE = (0.05, 0.12)
GEN_MAX_NOISE_COMPONENT = 2
# Rejection budget.
GEN_MAX_ATTEMPTS = 300


def _gen_axis_positions(length: int, count: int, min_sep: int) -> list[int]:
    positions = range(GEN_BLOCK_RADIUS, int(length) - GEN_BLOCK_RADIUS)
    valid = [
        combo
        for combo in itertools.combinations(positions, int(count))
        if all(abs(a - b) >= int(min_sep) for a, b in itertools.combinations(combo, 2))
    ]
    if not valid:
        return []
    chosen = list(random.choice(valid))
    random.shuffle(chosen)
    return chosen


def _gen_neighbors4(row: int, col: int):
    yield row - 1, col
    yield row + 1, col
    yield row, col - 1
    yield row, col + 1


def _gen_in_bounds(shape: tuple[int, int], row: int, col: int) -> bool:
    height, width = shape
    return 0 <= row < height and 0 <= col < width


def _gen_noise_component(
    noise_color: np.ndarray, row: int, col: int, color: int
) -> set[tuple[int, int]]:
    stack = [(int(row), int(col))]
    seen: set[tuple[int, int]] = set()
    while stack:
        r, c = stack.pop()
        if (r, c) in seen or not _gen_in_bounds(noise_color.shape, r, c):
            continue
        if int(noise_color[r, c]) != int(color):
            continue
        seen.add((r, c))
        stack.extend(_gen_neighbors4(r, c))
    return seen


def _gen_valid_noise_color(
    grid: np.ndarray,
    block_mask: np.ndarray,
    noise_color: np.ndarray,
    row: int,
    col: int,
    color: int,
) -> bool:
    for nr, nc in _gen_neighbors4(int(row), int(col)):
        if not _gen_in_bounds(grid.shape, nr, nc):
            continue
        if block_mask[nr, nc] and int(grid[nr, nc]) == int(color):
            return False

    joined: set[tuple[int, int]] = set()
    for nr, nc in _gen_neighbors4(int(row), int(col)):
        if not _gen_in_bounds(grid.shape, nr, nc):
            continue
        if int(noise_color[nr, nc]) == int(color) and (nr, nc) not in joined:
            joined |= _gen_noise_component(noise_color, nr, nc, color)
    return len(joined) + 1 <= GEN_MAX_NOISE_COMPONENT


def _gen_add_noise(grid: np.ndarray, block_mask: np.ndarray, noise_count: int) -> bool:
    noise_color = np.zeros_like(grid)
    for _ in range(int(noise_count)):
        choices: list[tuple[int, int, int]] = []
        free = np.argwhere((grid == GEN_BACKGROUND) & (~block_mask))
        for row, col in free:
            colors = list(GEN_COLORS)
            random.shuffle(colors)
            for color in colors:
                if _gen_valid_noise_color(grid, block_mask, noise_color, row, col, color):
                    choices.append((int(row), int(col), int(color)))
        if not choices:
            return False
        row, col, color = random.choice(choices)
        grid[row, col] = color
        noise_color[row, col] = color
    return True


def _gen_expected_output(grid: np.ndarray, blocks: list[tuple[int, int, int]]) -> np.ndarray:
    out = grid.copy()
    for row, col, color in blocks:
        out[row, :] = color
        out[:, col] = color
    for i, (row, _, _) in enumerate(blocks):
        for j, (_, col, _) in enumerate(blocks):
            if i != j:
                out[row, col] = 0
    return out


def _gen_place_blocks(
    height: int, width: int, count: int
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int]]] | None:
    rows = _gen_axis_positions(height, count, GEN_ROW_MIN_SEP)
    cols = _gen_axis_positions(width, count, GEN_COL_MIN_SEP)
    if len(rows) != count or len(cols) != count:
        return None

    colors = random.sample(GEN_COLORS, count)
    grid = np.zeros((height, width), dtype=int)
    block_mask = np.zeros((height, width), dtype=bool)
    blocks: list[tuple[int, int, int]] = []

    for row, col, color in zip(rows, cols, colors):
        patch = np.full((GEN_BLOCK_SIZE, GEN_BLOCK_SIZE), int(color), dtype=int)
        paste(grid, patch, row - GEN_BLOCK_RADIUS, col - GEN_BLOCK_RADIUS)
        block_mask[
            row - GEN_BLOCK_RADIUS : row + GEN_BLOCK_RADIUS + 1,
            col - GEN_BLOCK_RADIUS : col + GEN_BLOCK_RADIUS + 1,
        ] = True
        blocks.append((int(row), int(col), int(color)))

    return grid, block_mask, blocks


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GEN_HEIGHT_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)
    count = random.randint(*GEN_BLOCK_COUNT_RANGE)

    placed = _gen_place_blocks(height, width, count)
    if placed is None:
        return None
    grid, block_mask, blocks = placed

    total = height * width
    min_noise = max(1, int(np.ceil(total * GEN_NOISE_DENSITY_RANGE[0])))
    max_noise = int(np.floor(total * GEN_NOISE_DENSITY_RANGE[1]))
    noise_count = random.randint(min_noise, max_noise)
    if not _gen_add_noise(grid, block_mask, noise_count):
        return None

    expected = _gen_expected_output(grid, blocks)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid e9bb6954 input")
