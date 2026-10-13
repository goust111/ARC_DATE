import numpy as np


BACKGROUND = 0
LEFT_COLOR = 1
OUTPUT_COLOR = 2
RIGHT_COLOR = 3
SEPARATOR_COLOR = 5
SEED_SIZES = (3, 4)
FOREGROUND_COUNT_RANGE = (3, 6)
GENERATION_RETRIES = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start_r, start_c = map(int, coords[0])
    seen = {(start_r, start_c)}
    stack = [(start_r, start_c)]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                if (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _grow_connected_mask(size: int, count: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    start = tuple(int(x) for x in np.random.randint(0, size, size=2))
    mask[start] = True
    cells = {start}
    while len(cells) < count:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < size and 0 <= nc < size and not mask[nr, nc]:
                    frontier.add((nr, nc))
        choice = list(frontier)[int(np.random.randint(len(frontier)))]
        mask[choice] = True
        cells.add(choice)
    return mask


def _expand_mask(mask: np.ndarray) -> np.ndarray:
    size = int(mask.shape[0])
    out = np.zeros((size * size, size * size), dtype=bool)
    for r in range(size):
        for c in range(size):
            if mask[r, c]:
                out[r * size : (r + 1) * size, c * size : (c + 1) * size] = mask
    return out


def _render_panel(mask: np.ndarray, color: int) -> np.ndarray:
    panel = np.zeros(mask.shape, dtype=int)
    panel[mask] = int(color)
    return panel


def _validate_seed(panel: np.ndarray, expected_color: int) -> np.ndarray:
    panel = np.asarray(panel, dtype=int)
    colors = set(map(int, np.unique(panel)))
    assert colors <= {BACKGROUND, expected_color}, "panel must be a binary mask with the expected color"
    mask = panel == expected_color
    fg_count = int(mask.sum())
    assert FOREGROUND_COUNT_RANGE[0] <= fg_count <= FOREGROUND_COUNT_RANGE[1], "foreground count out of range"
    assert _is_connected(mask), "foreground must be 4-connected"
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand two seed masks into self-similar fractals and output only their XOR disagreement.

    Input structure:
    - The input is an N x (2N+1) grid with two N x N panels separated by a solid one-column bar.
    - The left panel is a black-and-blue binary seed and the right panel is a black-and-green binary seed.
    - Each seed has 3-6 foreground cells forming one 4-connected shape, and the separator column is grey.

    Rule:
    - Split the grid at the separator into left and right N x N seeds.
    - For each seed, build an N^2 x N^2 block expansion: every foreground cell copies the whole seed into its block, while every background cell leaves its block black.
    - Compare the two expanded masks cell by cell and output red exactly where one expansion has foreground and the other does not; all other cells stay black.

    Key insight:
    - The answer is not either expanded fractal itself, but the symmetric difference between two independently self-stamped expansions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert width % 2 == 1, "expected two panels with a center separator"
    mid = width // 2
    assert height == mid, "expected square panels with width 2N+1"
    assert np.all(g[:, mid] == SEPARATOR_COLOR), "expected a solid grey separator column"

    left = g[:, :mid]
    right = g[:, mid + 1 :]
    assert left.shape == right.shape == (height, height), "panels must be equal N x N squares"

    left_mask = _validate_seed(left, LEFT_COLOR)
    right_mask = _validate_seed(right, RIGHT_COLOR)

    left_expanded = _expand_mask(left_mask)
    right_expanded = _expand_mask(right_mask)
    out = np.zeros(left_expanded.shape, dtype=int)
    out[np.logical_xor(left_expanded, right_expanded)] = OUTPUT_COLOR
    return out


def _gen_attempt() -> np.ndarray | None:
    size = int(SEED_SIZES[int(np.random.randint(len(SEED_SIZES)))])
    left_count = int(np.random.randint(FOREGROUND_COUNT_RANGE[0], FOREGROUND_COUNT_RANGE[1] + 1))
    right_count = int(np.random.randint(FOREGROUND_COUNT_RANGE[0], FOREGROUND_COUNT_RANGE[1] + 1))

    left_mask = _grow_connected_mask(size, left_count)
    right_mask = _grow_connected_mask(size, right_count)
    if np.array_equal(left_mask, right_mask):
        return None

    left_expanded = _expand_mask(left_mask)
    right_expanded = _expand_mask(right_mask)
    overlap = int(np.logical_and(left_expanded, right_expanded).sum())
    xor_mask = np.logical_xor(left_expanded, right_expanded)
    xor_count = int(xor_mask.sum())
    total = int(xor_mask.size)

    if xor_count == 0 or xor_count == total:
        return None
    if overlap == 0:
        return None
    if xor_count < max(4, total // 8):
        return None
    if xor_count > total - max(4, total // 8):
        return None

    left_panel = _render_panel(left_mask, LEFT_COLOR)
    right_panel = _render_panel(right_mask, RIGHT_COLOR)
    separator = np.full((size, 1), SEPARATOR_COLOR, dtype=int)
    inp = np.concatenate([left_panel, separator, right_panel], axis=1)

    expected = np.zeros((size * size, size * size), dtype=int)
    expected[xor_mask] = OUTPUT_COLOR
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
