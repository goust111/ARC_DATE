import random

import numpy as np


_RNG = random.Random()
_GEN_RETRIES = 400


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _edge_runs(line: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    line = np.asarray(line, dtype=int)
    runs: list[tuple[int, int, int]] = []
    i = 0
    while i < line.size:
        if int(line[i]) == bg:
            i += 1
            continue
        color = int(line[i])
        j = i
        while j + 1 < line.size and int(line[j + 1]) == color:
            j += 1
        runs.append((color, int(i), int(j)))
        i = j + 1
    return runs


def _stack_crops(crops: list[np.ndarray], bg: int) -> np.ndarray:
    total_h = sum(int(crop.shape[0]) for crop in crops)
    max_w = max(int(crop.shape[1]) for crop in crops)
    out = np.full((total_h, max_w), bg, dtype=int)
    row = 0
    for crop in crops:
        h, w = crop.shape
        out[row : row + h, :w] = crop
        row += h
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use matching color rulers on the top and left edges to define crop windows, then
    stack the paired crops vertically in left-ruler order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 2
    bg = _bg_color(g)

    top_runs = _edge_runs(g[0], bg)
    left_runs = _edge_runs(g[:, 0], bg)
    assert top_runs and left_runs

    top_by_color: dict[int, tuple[int, int]] = {}
    for color, c0, c1 in top_runs:
        assert color not in top_by_color
        top_by_color[color] = (c0, c1)

    left_by_color: dict[int, tuple[int, int]] = {}
    for color, r0, r1 in left_runs:
        assert color not in left_by_color
        left_by_color[color] = (r0, r1)

    assert set(top_by_color) == set(left_by_color)

    crops: list[np.ndarray] = []
    for color, r0, r1 in left_runs:
        c0, c1 = top_by_color[color]
        crops.append(g[r0 : r1 + 1, c0 : c1 + 1].copy())

    return _stack_crops(crops, bg)


def _place_runs(total_len: int, sizes: list[int], rng: random.Random) -> list[int] | None:
    if not sizes:
        return []
    min_needed = sum(sizes) + max(0, len(sizes) - 1)
    if min_needed > total_len:
        return None
    extra = total_len - min_needed
    gaps = [0] * (len(sizes) + 1)
    for _ in range(extra):
        gaps[rng.randrange(len(gaps))] += 1

    starts: list[int] = []
    pos = gaps[0]
    for i, size in enumerate(sizes):
        starts.append(pos)
        pos += size
        if i + 1 < len(sizes):
            pos += 1 + gaps[i + 1]
    return starts


def _axis_blocks(length: int, rng: random.Random) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
    pos = 0
    while pos < length:
        max_step = min(4, length - pos)
        step = rng.randint(1, max_step)
        blocks.append((pos, pos + step))
        pos += step
    return blocks


def _mosaic(inner_h: int, inner_w: int, bg: int, rng: random.Random) -> np.ndarray:
    palette = [c for c in range(10) if c != bg]
    color_count = rng.randint(3, 5)
    pattern_colors = rng.sample(palette, color_count)
    non_bg_prob = rng.uniform(0.60, 0.68)

    row_blocks = _axis_blocks(inner_h, rng)
    col_blocks = _axis_blocks(inner_w, rng)
    block_colors: list[list[int]] = [[bg] * len(col_blocks) for _ in row_blocks]

    for i in range(len(row_blocks)):
        for j in range(len(col_blocks)):
            for _ in range(8):
                color = rng.choice(pattern_colors) if rng.random() < non_bg_prob else bg
                if j > 0 and color == block_colors[i][j - 1]:
                    continue
                if i > 0 and color == block_colors[i - 1][j]:
                    continue
                break
            block_colors[i][j] = color

    inner = np.full((inner_h, inner_w), bg, dtype=int)
    for i, (r0, r1) in enumerate(row_blocks):
        for j, (c0, c1) in enumerate(col_blocks):
            inner[r0:r1, c0:c1] = block_colors[i][j]
    return inner


def _interesting_crop(crop: np.ndarray, bg: int) -> bool:
    unique = np.unique(crop)
    return unique.size >= 2 and int(np.count_nonzero(crop != bg)) >= 2


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.randrange(10)
    pair_count = rng.randint(2, 4)
    crop_width = rng.randint(2, 5)
    crop_heights = [rng.randint(2, 4) for _ in range(pair_count)]

    width_slack = rng.randint(1, 7)
    height_slack = rng.randint(1, 7)
    grid_w = 1 + pair_count * crop_width + (pair_count - 1) + width_slack
    grid_h = 1 + sum(crop_heights) + (pair_count - 1) + height_slack

    top_starts = _place_runs(grid_w - 1, [crop_width] * pair_count, rng)
    left_starts = _place_runs(grid_h - 1, crop_heights, rng)
    if top_starts is None or left_starts is None:
        return None

    color_pool = [c for c in range(10) if c != bg]
    pair_colors = rng.sample(color_pool, pair_count)
    top_colors = pair_colors[:]
    left_colors = pair_colors[:]
    rng.shuffle(top_colors)
    rng.shuffle(left_colors)
    if pair_count > 1 and top_colors == left_colors:
        left_colors = left_colors[1:] + left_colors[:1]

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    grid[1:, 1:] = _mosaic(grid_h - 1, grid_w - 1, bg, rng)

    for color, start in zip(top_colors, top_starts):
        c0 = 1 + start
        grid[0, c0 : c0 + crop_width] = color

    for color, start, height in zip(left_colors, left_starts, crop_heights):
        r0 = 1 + start
        grid[r0 : r0 + height, 0] = color

    if _bg_color(grid) != bg:
        return None

    top_map = {color: (1 + start, 1 + start + crop_width - 1) for color, start in zip(top_colors, top_starts)}
    expected_crops = [grid[r0 : r1 + 1, top_map[color][0] : top_map[color][1] + 1] for color, r0, r1 in _edge_runs(grid[:, 0], bg)]
    if len(expected_crops) != pair_count:
        return None
    if not all(_interesting_crop(crop, bg) for crop in expected_crops):
        return None
    if len({crop.tobytes() for crop in expected_crops}) < 2:
        return None

    expected = _stack_crops(expected_crops, bg)
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
