import numpy as np


BG = 0
COLOR_POOL = np.arange(1, 10, dtype=int)
GEN_RETRIES = 300
_RNG = np.random.default_rng()


def _read_top_ruler(grid: np.ndarray) -> list[tuple[int, int]]:
    entries = [(int(c), int(grid[0, c])) for c in range(1, grid.shape[1]) if int(grid[0, c]) != BG]
    colors = [color for _, color in entries]
    assert entries, "expected colored top-edge ruler pixels"
    assert len(colors) == len(set(colors)), "top ruler colors must be unique"
    return entries


def _read_left_ruler(grid: np.ndarray) -> list[tuple[int, int]]:
    entries = [(int(r), int(grid[r, 0])) for r in range(1, grid.shape[0]) if int(grid[r, 0]) != BG]
    colors = [color for _, color in entries]
    assert entries, "expected colored left-edge ruler pixels"
    assert len(colors) == len(set(colors)), "left ruler colors must be unique"
    return entries


def _segments(starts: list[int], limit: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] - 1 if idx + 1 < len(starts) else limit - 1
        out.append((int(start), int(end)))
    return out


def _crop_block(region: np.ndarray, color: int) -> np.ndarray:
    coords = np.argwhere(region == color)
    assert coords.size, f"missing block for color {color}"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return region[r0 : r1 + 1, c0 : c1 + 1].copy()


def _offsets(lengths: list[int]) -> list[int]:
    out = [0]
    for length in lengths[:-1]:
        out.append(out[-1] + int(length))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop ruler-addressed color blocks and reassemble them in ruler order.

    Input structure:
    - Background is black.
    - The top edge contains distinct colored ruler pixels that mark interior column windows.
    - The left edge contains distinct colored ruler pixels that mark interior row windows.
    - The same color set appears once on each ruler, and each interior solid rectangle uses one of those colors.

    Rule:
    - Read top ruler pixels from left to right and left ruler pixels from top to bottom.
    - For each color, use the matching top and left ruler positions to select that color's interior window, then crop the rectangle to its minimal bounding box.
    - Build the output by stacking row bands in left-ruler order and column bands in top-ruler order, placing each cropped block at the intersection of its matching row and column.
    - Leave every other output cell black.

    Key insight:
    - The edge colors are both the address keys for finding each block and the ordering keys for arranging the cropped pieces in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    top_ruler = _read_top_ruler(g)
    left_ruler = _read_left_ruler(g)

    top_colors = {color for _, color in top_ruler}
    left_colors = {color for _, color in left_ruler}
    assert top_colors == left_colors, "top and left rulers must use the same color set"

    col_segments = _segments([pos for pos, _ in top_ruler], g.shape[1])
    row_segments = _segments([pos for pos, _ in left_ruler], g.shape[0])

    top_index = {color: idx for idx, (_, color) in enumerate(top_ruler)}
    left_index = {color: idx for idx, (_, color) in enumerate(left_ruler)}

    crops: dict[int, np.ndarray] = {}
    for color in top_colors:
        row_idx = left_index[color]
        col_idx = top_index[color]
        r0, r1 = row_segments[row_idx]
        c0, c1 = col_segments[col_idx]
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        crops[color] = _crop_block(region, color)

    row_heights = [int(crops[color].shape[0]) for _, color in left_ruler]
    col_widths = [int(crops[color].shape[1]) for _, color in top_ruler]
    out = np.full((sum(row_heights), sum(col_widths)), BG, dtype=int)

    row_offsets = _offsets(row_heights)
    col_offsets = _offsets(col_widths)
    for color, crop in crops.items():
        rr = row_offsets[left_index[color]]
        cc = col_offsets[top_index[color]]
        h, w = crop.shape
        out[rr : rr + h, cc : cc + w] = crop

    return out


def _sample_segments(count: int) -> list[int]:
    min_size = 3
    max_size = 7 if count <= 3 else 6 if count <= 4 else 5
    total_min = count * min_size
    total_max = min(29, count * max_size)
    total = int(_RNG.integers(total_min, total_max + 1))

    sizes: list[int] = []
    remaining = total
    for idx in range(count):
        slots_left = count - idx - 1
        lo = max(min_size, remaining - slots_left * max_size)
        hi = min(max_size, remaining - slots_left * min_size)
        sizes.append(int(_RNG.integers(lo, hi + 1)))
        remaining -= sizes[-1]

    sizes = list(map(int, _RNG.permutation(np.array(sizes, dtype=int))))
    return sizes


def _expected_output(left_order: list[int], top_order: list[int], block_shapes: dict[int, tuple[int, int]]) -> np.ndarray:
    row_heights = [block_shapes[color][0] for color in left_order]
    col_widths = [block_shapes[color][1] for color in top_order]
    out = np.full((sum(row_heights), sum(col_widths)), BG, dtype=int)

    row_offsets = _offsets(row_heights)
    col_offsets = _offsets(col_widths)
    left_index = {color: idx for idx, color in enumerate(left_order)}
    top_index = {color: idx for idx, color in enumerate(top_order)}

    for color, (height, width) in block_shapes.items():
        rr = row_offsets[left_index[color]]
        cc = col_offsets[top_index[color]]
        out[rr : rr + height, cc : cc + width] = color

    return out


def _gen_attempt() -> np.ndarray:
    count = int(_RNG.integers(2, 7))
    colors = list(map(int, _RNG.choice(COLOR_POOL, size=count, replace=False)))

    left_order = list(map(int, _RNG.permutation(np.array(colors, dtype=int))))
    if _RNG.random() < 0.5:
        top_order = left_order.copy()
    else:
        top_order = list(map(int, _RNG.permutation(np.array(colors, dtype=int))))
        if top_order == left_order:
            top_order[0], top_order[1] = top_order[1], top_order[0]

    row_sizes = _sample_segments(count)
    col_sizes = _sample_segments(count)
    height = 1 + sum(row_sizes)
    width = 1 + sum(col_sizes)
    assert height <= 30 and width <= 30

    grid = np.full((height, width), BG, dtype=int)

    row_starts = [1]
    for size in row_sizes[:-1]:
        row_starts.append(row_starts[-1] + int(size))
    col_starts = [1]
    for size in col_sizes[:-1]:
        col_starts.append(col_starts[-1] + int(size))

    for row, color in zip(row_starts, left_order):
        grid[row, 0] = color
    for col, color in zip(col_starts, top_order):
        grid[0, col] = color

    left_index = {color: idx for idx, color in enumerate(left_order)}
    top_index = {color: idx for idx, color in enumerate(top_order)}
    block_shapes: dict[int, tuple[int, int]] = {}

    for color in colors:
        row_idx = left_index[color]
        col_idx = top_index[color]
        row_size = row_sizes[row_idx]
        col_size = col_sizes[col_idx]

        block_h = int(_RNG.integers(1, row_size - 1))
        block_w = int(_RNG.integers(1, col_size - 1))
        top_pad = int(_RNG.integers(1, row_size - block_h))
        left_pad = int(_RNG.integers(1, col_size - block_w))

        r0 = row_starts[row_idx] + top_pad
        c0 = col_starts[col_idx] + left_pad
        grid[r0 : r0 + block_h, c0 : c0 + block_w] = color
        block_shapes[color] = (block_h, block_w)

    expected = _expected_output(left_order, top_order, block_shapes)
    actual = main(grid)
    assert np.array_equal(actual, expected)
    assert max(actual.shape) <= 30

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            return _gen_attempt()
        except AssertionError:
            continue
    raise RuntimeError("generate_input: exhausted retries")
