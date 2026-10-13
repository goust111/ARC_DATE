import random
from itertools import combinations

import numpy as np


BACKGROUND = 0
SEPARATOR = 9
COLOR_POOL = tuple(range(1, 9))
GEN_RETRIES = 400
ROW_BAND_RANGE = (3, 5)
COL_BAND_RANGE = (4, 6)
DICT_ROW_CHOICES = (1, 2)
DICT_COL_CHOICES = (2, 3, 4)
DICT_CELL_SIZE = (3, 5)
QUERY_ROW_SIZE = (4, 7)
QUERY_COL_SIZE = (4, 8)
CORNERS = ("tl", "tr", "bl", "br")


def _separator_color(grid: np.ndarray) -> int:
    best_color = None
    best_score = -1
    for color in map(int, np.unique(grid)):
        if color == BACKGROUND:
            continue
        score = int(np.all(grid == color, axis=1).sum() + np.all(grid == color, axis=0).sum())
        if score > best_score:
            best_color = color
            best_score = score
    assert best_color is not None and best_score > 0, "expected full-row/full-column separators"
    return int(best_color)


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for sep in separators:
        assert start < sep < length, "separator rows/cols must be interior and non-adjacent"
        spans.append((start, sep))
        start = sep + 1
    assert start < length, "separators must leave a final nonempty span"
    spans.append((start, length))
    return spans


def _parse_cell(tile: np.ndarray, separator: int) -> tuple[str, frozenset[int], int | None]:
    colors, counts = np.unique(tile, return_counts=True)
    color_counts = {int(color): int(count) for color, count in zip(colors, counts)}
    assert separator not in color_counts, "separator color must not appear inside a cell"
    assert tile.size >= 4, "cells must be large enough to distinguish seeds from fill"

    if BACKGROUND in color_counts:
        non_bg = [color for color in color_counts if color != BACKGROUND]
        assert len(color_counts) == 3 and len(non_bg) == 2, "query cell must contain black plus two seed colors"
        assert color_counts[BACKGROUND] == tile.size - 2, "query cell must have exactly two seeds"
        assert all(color_counts[color] == 1 for color in non_bg), "each seed color must appear once"
        return "query", frozenset(non_bg), None

    singles = [color for color, count in color_counts.items() if count == 1]
    fills = [color for color, count in color_counts.items() if count == tile.size - 2]
    assert len(color_counts) == 3 and len(singles) == 2 and len(fills) == 1, "dictionary cell must be two keys plus one fill color"
    return "dict", frozenset(singles), int(fills[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each black partition using the color looked up from the seed-pair dictionary.

    The grid is split by full maroon separator lines into rectangular cells. Cells that
    contain black are query cells: they have exactly two distinct seed pixels. Cells that
    contain no black are dictionary entries: they have exactly two single-pixel key colors
    plus one fill color occupying the rest of the cell. The unordered key-color pair maps
    to the fill color used for every matching query cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    sep = _separator_color(g)

    row_separators = [int(r) for r in range(g.shape[0]) if np.all(g[r, :] == sep)]
    col_separators = [int(c) for c in range(g.shape[1]) if np.all(g[:, c] == sep)]
    assert row_separators and col_separators, "expected both horizontal and vertical separators"

    row_spans = _spans(g.shape[0], row_separators)
    col_spans = _spans(g.shape[1], col_separators)

    mapping: dict[frozenset[int], int] = {}
    queries: list[tuple[int, int, int, int, frozenset[int]]] = []

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            tile = g[r0:r1, c0:c1]
            kind, key, value = _parse_cell(tile, sep)
            if kind == "dict":
                assert value is not None
                assert key not in mapping, "dictionary key pairs must be unique"
                mapping[key] = int(value)
            else:
                queries.append((r0, r1, c0, c1, key))

    assert mapping, "expected at least one dictionary cell"
    assert queries, "expected at least one query cell"

    out = g.copy()
    for r0, r1, c0, c1, key in queries:
        assert key in mapping, "every query key pair must exist in the dictionary"
        patch = out[r0:r1, c0:c1]
        patch[patch == BACKGROUND] = mapping[key]
    return out


def _band_layout(lengths: list[int]) -> tuple[list[tuple[int, int]], list[int]]:
    spans: list[tuple[int, int]] = []
    separators: list[int] = []
    start = 0
    for i, size in enumerate(lengths):
        end = start + size
        spans.append((start, end))
        if i < len(lengths) - 1:
            separators.append(end)
            start = end + 1
    return spans, separators


def _dict_indices(total: int, count: int, at_end: bool) -> set[int]:
    if at_end:
        return set(range(total - count, total))
    return set(range(count))


def _cell_coords(height: int, width: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(height) for c in range(width)]


def _place_two_seeds(tile: np.ndarray, colors: tuple[int, int]) -> None:
    (r0, c0), (r1, c1) = random.sample(_cell_coords(*tile.shape), 2)
    a, b = colors
    if random.random() < 0.5:
        a, b = b, a
    tile[r0, c0] = a
    tile[r1, c1] = b


def _dictionary_shape_options(row_bands: int, col_bands: int) -> list[tuple[int, int]]:
    options = []
    for dr in DICT_ROW_CHOICES:
        for dc in DICT_COL_CHOICES:
            entries = dr * dc
            queries = row_bands * col_bands - entries
            if dr < row_bands and dc < col_bands and entries >= 2 and queries >= entries:
                options.append((dr, dc))
    return options


def _sample_palette(entry_count: int) -> tuple[list[frozenset[int]], dict[frozenset[int], int]]:
    possible_key_counts = [
        k for k in range(3, len(COLOR_POOL) + 1) if len(list(combinations(COLOR_POOL[:k], 2))) >= entry_count
    ]
    if not possible_key_counts:
        raise RuntimeError("too many dictionary entries for palette")
    key_count = random.choice(possible_key_counts)
    palette = random.sample(COLOR_POOL, random.randint(key_count, len(COLOR_POOL)))
    key_colors = random.sample(palette, key_count)
    all_pairs = [frozenset(pair) for pair in combinations(key_colors, 2)]
    chosen_pairs = random.sample(all_pairs, entry_count)

    for _ in range(50):
        mapping = {
            pair: random.choice([color for color in palette if color not in pair])
            for pair in chosen_pairs
        }
        if entry_count == 1 or len(set(mapping.values())) >= 2:
            return chosen_pairs, mapping
    raise RuntimeError("failed to assign dictionary values")


def _transform(grid: np.ndarray, rotation: int, flip_lr: bool) -> np.ndarray:
    out = np.rot90(grid, rotation)
    if flip_lr:
        out = np.fliplr(out)
    return out


def _gen_attempt() -> np.ndarray | None:
    row_bands = random.randint(*ROW_BAND_RANGE)
    col_bands = random.randint(*COL_BAND_RANGE)
    shape_options = _dictionary_shape_options(row_bands, col_bands)
    if not shape_options:
        return None

    dict_rows_count, dict_cols_count = random.choice(shape_options)
    corner = random.choice(CORNERS)
    dict_rows = _dict_indices(row_bands, dict_rows_count, at_end=corner in {"bl", "br"})
    dict_cols = _dict_indices(col_bands, dict_cols_count, at_end=corner in {"tr", "br"})

    row_sizes = [
        random.randint(*(DICT_CELL_SIZE if r in dict_rows else QUERY_ROW_SIZE))
        for r in range(row_bands)
    ]
    col_sizes = [
        random.randint(*(DICT_CELL_SIZE if c in dict_cols else QUERY_COL_SIZE))
        for c in range(col_bands)
    ]

    row_spans, row_separators = _band_layout(row_sizes)
    col_spans, col_separators = _band_layout(col_sizes)
    height = row_spans[-1][1]
    width = col_spans[-1][1]

    grid = np.full((height, width), BACKGROUND, dtype=int)
    for r in row_separators:
        grid[r, :] = SEPARATOR
    for c in col_separators:
        grid[:, c] = SEPARATOR
    expected = grid.copy()

    dict_cells = [
        (r, c)
        for r in range(row_bands)
        for c in range(col_bands)
        if r in dict_rows and c in dict_cols
    ]
    query_cells = [
        (r, c)
        for r in range(row_bands)
        for c in range(col_bands)
        if (r, c) not in set(dict_cells)
    ]
    if len(query_cells) < len(dict_cells):
        return None

    ordered_pairs, mapping = _sample_palette(len(dict_cells))
    random.shuffle(ordered_pairs)

    for (band_r, band_c), pair in zip(dict_cells, ordered_pairs):
        r0, r1 = row_spans[band_r]
        c0, c1 = col_spans[band_c]
        value = mapping[pair]
        tile = np.full((r1 - r0, c1 - c0), value, dtype=int)
        colors = tuple(pair)
        _place_two_seeds(tile, colors)
        grid[r0:r1, c0:c1] = tile
        expected[r0:r1, c0:c1] = tile

    query_pairs = ordered_pairs.copy()
    while len(query_pairs) < len(query_cells):
        query_pairs.append(random.choice(ordered_pairs))
    random.shuffle(query_pairs)

    for (band_r, band_c), pair in zip(query_cells, query_pairs):
        r0, r1 = row_spans[band_r]
        c0, c1 = col_spans[band_c]
        tile = np.full((r1 - r0, c1 - c0), BACKGROUND, dtype=int)
        colors = tuple(pair)
        _place_two_seeds(tile, colors)
        grid[r0:r1, c0:c1] = tile

        filled = tile.copy()
        filled[filled == BACKGROUND] = mapping[pair]
        expected[r0:r1, c0:c1] = filled

    rotation = random.randrange(4)
    flip_lr = random.random() < 0.5
    grid = _transform(grid, rotation, flip_lr)
    expected = _transform(expected, rotation, flip_lr)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
