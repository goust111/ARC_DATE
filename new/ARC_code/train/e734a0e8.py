from __future__ import annotations

from collections import Counter

import numpy as np
import random
from arc_utils import paste


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected a nonempty set of values"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique dominant color"
    return int(ranked[0][0])


def _separator_lines(g: np.ndarray) -> tuple[int, list[int], list[int]]:
    height, width = map(int, g.shape)
    candidates = []
    for color in np.unique(g):
        color = int(color)
        rows = [r for r in range(height) if np.all(g[r, :] == color)]
        cols = [c for c in range(width) if np.all(g[:, c] == color)]
        if rows and cols:
            candidates.append((color, rows, cols))
    assert len(candidates) == 1, "expected one color to form full separator rows and columns"
    return candidates[0]


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for sep in separators:
        assert start < sep, "separator lines must leave nonempty regions"
        spans.append((start, sep - 1))
        start = sep + 1
    assert start < length, "separator lines must leave a final nonempty region"
    spans.append((start, length - 1))
    return spans


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the unique colored sprite tile into every tiled cell marked by an internal separator-colored pixel.

    Input structure
    - Full same-color separator rows and columns partition the grid into equal rectangular tiles.
    - One tile contains the only non-background, non-separator colored sprite.
    - Target tiles are otherwise background-colored but contain at least one internal pixel of the separator color.

    Rule
    - Keep the separator lines and the source sprite tile unchanged.
    - Replace each marked target tile with an exact copy of the source tile.
    - Unmarked background-only tiles remain unchanged.

    Key insight
    - The internal separator-colored pixels are placement markers; the single colored tile is the template to stamp into those marked cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    separator, sep_rows, sep_cols = _separator_lines(g)
    row_spans = _spans(g.shape[0], sep_rows)
    col_spans = _spans(g.shape[1], sep_cols)

    tile_heights = {r1 - r0 + 1 for r0, r1 in row_spans}
    tile_widths = {c1 - c0 + 1 for c0, c1 in col_spans}
    assert len(tile_heights) == 1 and len(tile_widths) == 1, "expected equal-sized tile rows and columns"
    tile_h = tile_heights.pop()
    tile_w = tile_widths.pop()

    bg = _unique_mode(g[g != separator])
    assert bg != separator, "background and separator colors must differ"
    sprite_colors = [int(color) for color in np.unique(g) if int(color) not in {bg, separator}]
    assert len(sprite_colors) == 1, "expected one sprite color"
    sprite_color = sprite_colors[0]

    source_tiles = []
    marked_origins = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            assert tile.shape == (tile_h, tile_w)
            has_sprite = np.any(tile == sprite_color)
            has_marker = np.any(tile == separator)
            if has_sprite:
                assert not has_marker, "source tile should contain only sprite and background"
                assert set(map(int, np.unique(tile))).issubset({bg, sprite_color})
                source_tiles.append(tile.copy())
            elif has_marker:
                assert set(map(int, np.unique(tile))).issubset({bg, separator})
                marked_origins.append((r0, c0))
            else:
                assert np.all(tile == bg), "unmarked non-source tiles should be plain background"

    assert len(source_tiles) == 1, "expected exactly one source sprite tile"
    source_tile = source_tiles[0]

    out = g.copy()
    for r0, c0 in marked_origins:
        out[r0 : r0 + tile_h, c0 : c0 + tile_w] = source_tile
    return out

# --------------- generator (o5x) ---------------

# Palette: separator, background, and sprite are distinct.
COLOR_POOL = range(10)
PALETTE_SIZE = 3
# Tile grid dimensions and odd square tile sizes.
TILE_ROWS_RANGE = (3, 4)
TILE_COLS_RANGE = (3, 4)
TILE_SIZE_CHOICES = [3, 5]
SEPARATOR_WIDTH = 1
# Sprite density and 4-connected component count by tile size.
SPRITE_PIXEL_RANGES = {3: (3, 4), 5: (8, 11)}
SPRITE_COMPONENT_CAPS = {3: 3, 5: 5}
NEIGHBOR_OFFSETS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
ORIENTATION_COUNT = 4
FLIP_PROBABILITY = 0.5
# Marker tiles and retry budget.
MARKER_COUNT_RANGE = (3, 5)
MAX_GENERATION_ATTEMPTS = 300


def _gen_neighbors4(r: int, c: int, size: int):
    for dr, dc in NEIGHBOR_OFFSETS_4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < size and 0 <= cc < size:
            yield rr, cc


def _gen_partition(total: int, parts: int) -> list[int]:
    sizes = [1] * parts
    for _ in range(total - parts):
        sizes[random.randrange(parts)] += 1
    random.shuffle(sizes)
    return sizes


def _gen_separate_from_other(labels: np.ndarray, r: int, c: int, label: int) -> bool:
    size = int(labels.shape[0])
    for rr, cc in _gen_neighbors4(r, c, size):
        if labels[rr, cc] not in (0, label):
            return False
    return True


def _gen_sprite_tile(size: int, bg: int, sprite_color: int) -> np.ndarray | None:
    pixel_count = random.randint(*SPRITE_PIXEL_RANGES[size])
    component_count = random.randint(1, min(SPRITE_COMPONENT_CAPS[size], pixel_count))
    labels = np.zeros((size, size), dtype=int)
    for label, component_size in enumerate(_gen_partition(pixel_count, component_count), start=1):
        seed_candidates = [
            (r, c)
            for r in range(size)
            for c in range(size)
            if labels[r, c] == 0 and _gen_separate_from_other(labels, r, c, label)
        ]
        if not seed_candidates:
            return None

        seed = random.choice(seed_candidates)
        labels[seed] = label
        cells = [seed]
        while len(cells) < component_size:
            growth = []
            for r, c in cells:
                for rr, cc in _gen_neighbors4(r, c, size):
                    if labels[rr, cc] == 0 and _gen_separate_from_other(labels, rr, cc, label):
                        growth.append((rr, cc))
            if not growth:
                return None
            cell = random.choice(growth)
            labels[cell] = label
            cells.append(cell)

    tile = np.full((size, size), bg, dtype=int)
    tile[labels > 0] = sprite_color
    tile = np.rot90(tile, random.randrange(ORIENTATION_COUNT))
    if random.random() < FLIP_PROBABILITY:
        tile = np.fliplr(tile)
    if random.random() < FLIP_PROBABILITY:
        tile = np.flipud(tile)
    return tile.copy()


def _gen_tile_origin(tile_r: int, tile_c: int, tile_size: int) -> tuple[int, int]:
    return tile_r * (tile_size + SEPARATOR_WIDTH), tile_c * (tile_size + SEPARATOR_WIDTH)


def _gen_base_grid(tile_rows: int, tile_cols: int, tile_size: int, bg: int, separator: int) -> np.ndarray:
    height = tile_rows * tile_size + (tile_rows - 1) * SEPARATOR_WIDTH
    width = tile_cols * tile_size + (tile_cols - 1) * SEPARATOR_WIDTH
    grid = np.full((height, width), bg, dtype=int)
    grid[tile_size :: tile_size + SEPARATOR_WIDTH, :] = separator
    grid[:, tile_size :: tile_size + SEPARATOR_WIDTH] = separator
    return grid


def _gen_paste_tile(grid: np.ndarray, pos: tuple[int, int], tile: np.ndarray, tile_size: int) -> None:
    r0, c0 = _gen_tile_origin(pos[0], pos[1], tile_size)
    paste(grid, tile, r0, c0)


def _gen_render_pair(tile_rows, tile_cols, tile_size, separator, bg, sprite_tile, source_pos, marked_positions):
    expected = _gen_base_grid(tile_rows, tile_cols, tile_size, bg, separator)
    for pos in [source_pos, *marked_positions]:
        _gen_paste_tile(expected, pos, sprite_tile, tile_size)

    inp = expected.copy()
    marker_tile = np.full((tile_size, tile_size), bg, dtype=int)
    marker_tile[tile_size // 2, tile_size // 2] = separator
    for pos in marked_positions:
        _gen_paste_tile(inp, pos, marker_tile, tile_size)
    return inp, expected


def _gen_attempt() -> np.ndarray | None:
    separator, bg, sprite_color = random.sample(COLOR_POOL, PALETTE_SIZE)
    tile_rows, tile_cols, tile_size = (
        random.randint(*TILE_ROWS_RANGE),
        random.randint(*TILE_COLS_RANGE),
        random.choice(TILE_SIZE_CHOICES),
    )

    sprite_tile = _gen_sprite_tile(tile_size, bg, sprite_color)
    if sprite_tile is None:
        return None

    tile_positions = [(r, c) for r in range(tile_rows) for c in range(tile_cols)]
    source_pos = random.choice(tile_positions)
    marker_count = random.randint(*MARKER_COUNT_RANGE)
    marked_positions = random.sample([pos for pos in tile_positions if pos != source_pos], marker_count)
    inp, expected = _gen_render_pair(
        tile_rows, tile_cols, tile_size, separator, bg, sprite_tile, source_pos, marked_positions
    )
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e734a0e8 input")
