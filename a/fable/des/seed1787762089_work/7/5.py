import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
BLOCK_COLORS = (1, 2, 3, 4, 6, 7)

MAIN_HEIGHT_RANGE = (10, 16)
BLOCK_COUNT_RANGE = (4, 6)
BLOCK_WIDTH_RANGE = (2, 4)
MARKERS_PER_BLOCK_RANGE = (0, 3)
MAX_GENERATION_ATTEMPTS = 300


def _dominant_color(values: np.ndarray) -> int:
    counts = Counter(map(int, values.tolist()))
    return int(max(counts.items(), key=lambda item: item[1])[0])


def _extract_blocks(main_area: np.ndarray) -> list[tuple[int, int, int]]:
    column_colors = [_dominant_color(main_area[:, c]) for c in range(main_area.shape[1])]
    blocks: list[tuple[int, int, int]] = []
    start = 0
    while start < len(column_colors):
        color = column_colors[start]
        end = start + 1
        while end < len(column_colors) and column_colors[end] == color:
            end += 1
        blocks.append((start, end, int(color)))
        start = end
    return blocks


def _recipe_colors(recipe_row: np.ndarray) -> list[int]:
    colors: list[int] = []
    for value in map(int, recipe_row.tolist()):
        if value == BLACK:
            break
        colors.append(value)
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder whole vertical color blocks according to the bottom recipe row.

    The output keeps only the main area above the grey divider. Named blocks move
    left-to-right in recipe order, preserving all internal marker pixels. Blocks
    absent from the recipe are discarded and the remaining columns are black.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3

    main_area = g[:-2, :]
    blocks = _extract_blocks(main_area)
    recipe = _recipe_colors(g[-1, :])

    block_by_color = {
        color: main_area[:, c0:c1].copy()
        for c0, c1, color in blocks
    }

    out = np.full(main_area.shape, BLACK, dtype=int)
    cursor = 0
    for color in recipe:
        block = block_by_color.get(int(color))
        if block is None:
            continue
        width = int(block.shape[1])
        out[:, cursor : cursor + width] = block
        cursor += width

    return out


def _place_markers(main_area: np.ndarray, body_colors: list[int], block_width: int) -> None:
    height = int(main_area.shape[0])
    marker_counts = [
        random.randint(*MARKERS_PER_BLOCK_RANGE)
        for _ in body_colors
    ]
    if sum(marker_counts) == 0:
        marker_counts[random.randrange(len(marker_counts))] = random.randint(1, 3)

    for block_index, body_color in enumerate(body_colors):
        count = marker_counts[block_index]
        if count == 0:
            continue
        c0 = block_index * block_width
        cells = [(r, c) for r in range(height) for c in range(c0, c0 + block_width)]
        for r, c in random.sample(cells, count):
            marker_palette = [color for color in body_colors if color != body_color]
            main_area[r, c] = random.choice(marker_palette)


def _choose_recipe(body_colors: list[int]) -> list[int]:
    if random.random() < 0.5:
        recipe = body_colors.copy()
        while recipe == body_colors:
            random.shuffle(recipe)
        return recipe

    subset_size = random.randint(1, len(body_colors) - 1)
    recipe = random.sample(body_colors, subset_size)
    return recipe


def _gen_attempt() -> np.ndarray | None:
    block_count = random.randint(*BLOCK_COUNT_RANGE)
    block_width = random.randint(*BLOCK_WIDTH_RANGE)
    main_height = random.randint(*MAIN_HEIGHT_RANGE)
    body_colors = random.sample(BLOCK_COLORS, block_count)

    main_area = np.empty((main_height, block_count * block_width), dtype=int)
    for i, color in enumerate(body_colors):
        c0 = i * block_width
        main_area[:, c0 : c0 + block_width] = color

    _place_markers(main_area, body_colors, block_width)
    recipe = _choose_recipe(body_colors)

    divider = np.full((1, main_area.shape[1]), GREY, dtype=int)
    recipe_row = np.full((1, main_area.shape[1]), BLACK, dtype=int)
    recipe_row[0, : len(recipe)] = recipe
    grid = np.vstack([main_area, divider, recipe_row])

    out = main(grid)
    if np.array_equal(out, main_area):
        return None

    blocks = _extract_blocks(main_area)
    if len(blocks) != block_count:
        return None
    if [color for _, _, color in blocks] != body_colors:
        return None
    if any((c1 - c0) != block_width for c0, c1, _ in blocks):
        return None
    if _recipe_colors(grid[-1]) != recipe:
        return None

    named_colors = set(recipe)
    used_width = len(recipe) * block_width
    if len(named_colors) < block_count and not np.all(out[:, used_width:] == BLACK):
        return None

    return grid


def generate_input() -> np.ndarray:
    """
    Generate a valid block-reordering ARC input with balanced keep/remove cases.
    """
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
