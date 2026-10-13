from __future__ import annotations

import numpy as np


BG = 0
RED = 2
GREY = 5
INPUT_PALETTE = (1, 3, 4, 6, 7, 8, 9)
COUNT_PATTERNS = ((1, 2, 6), (1, 3, 5), (2, 3, 4))
TILE_SIZES = (3, 4, 5)
MIN_FG_CELLS = 3
MAX_FG_CELLS = 8
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _normalize_cells(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _separator_indices(grid: np.ndarray, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]


def _tile_spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    assert len(separators) == 2, "expected exactly two full separator lines"
    assert separators[0] + 1 < separators[1], "separator lines must be distinct"
    boundaries = [-1, separators[0], separators[1], length]
    spans = [(boundaries[i] + 1, boundaries[i + 1] - 1) for i in range(3)]
    sizes = [end - start + 1 for start, end in spans]
    assert min(sizes) > 0, "every tile band must have positive size"
    assert sizes[0] == sizes[1] == sizes[2], "tile bands must be equal-sized"
    return spans


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    row_separators = _separator_indices(g, axis=0)
    col_separators = _separator_indices(g, axis=1)
    row_spans = _tile_spans(g.shape[0], row_separators)
    col_spans = _tile_spans(g.shape[1], col_separators)

    tile_size = row_spans[0][1] - row_spans[0][0] + 1
    assert tile_size == col_spans[0][1] - col_spans[0][0] + 1, "tiles must be square"

    tiles = []
    for tile_r, (r0, r1) in enumerate(row_spans):
        for tile_c, (c0, c1) in enumerate(col_spans):
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            colors = set(map(int, np.unique(tile)))
            assert GREY not in colors, "grey may only appear on separator rows and columns"
            foreground = sorted(colors - {BG})
            assert len(foreground) == 1, "each tile must contain exactly one foreground color"
            color = int(foreground[0])
            mask = tile == color
            assert int(mask.sum()) > 0, "foreground tile cannot be empty"
            tiles.append(
                {
                    "tile_r": tile_r,
                    "tile_c": tile_c,
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c1,
                    "color": color,
                    "tile": tile.copy(),
                    "mask": mask,
                }
            )

    return {
        "grid": g,
        "row_separators": row_separators,
        "col_separators": col_separators,
        "row_spans": row_spans,
        "col_spans": col_spans,
        "tile_size": tile_size,
        "tiles": tiles,
    }


def _color_frequencies(tiles: list[dict]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for tile in tiles:
        color = int(tile["color"])
        counts[color] = counts.get(color, 0) + 1
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rank tile colors by cross-tile frequency and invert, keep, or clear each tile accordingly.

    Input structure:
    - The grid is divided by two full grey rows and two full grey columns into a 3x3 tile layout.
    - Non-separator cells use black as tile background, and each tile contains exactly one non-black, non-grey foreground color.
    - The same foreground color can appear in multiple tiles, and exactly three foreground colors are used across the nine tiles.

    Rule:
    - Count, for each foreground color, in how many different tiles that color appears.
    - The most frequent color's tiles are occupancy-inverted: their original foreground cells become black, and their original black cells become red.
    - The least frequent color's tiles are kept unchanged, the middle-frequency color's tiles are cleared to black, and all grey separators stay grey.

    Key insight:
    - The deciding signal is not pixel count inside a tile but how often a color is reused across different tiles, which routes each whole tile to one of three distinct actions.
    """
    parsed = _parse_grid(grid)
    g = parsed["grid"]
    counts = _color_frequencies(parsed["tiles"])
    assert len(counts) == 3, "expected exactly three foreground colors"
    ranked = sorted(counts, key=lambda color: counts[color])
    assert counts[ranked[0]] < counts[ranked[1]] < counts[ranked[2]], "tile frequencies must be strictly ordered"

    least_color, middle_color, most_color = ranked
    out = np.full_like(g, BG)
    out[parsed["row_separators"], :] = GREY
    out[:, parsed["col_separators"]] = GREY

    for tile in parsed["tiles"]:
        r0, r1, c0, c1 = tile["r0"], tile["r1"], tile["c0"], tile["c1"]
        src = tile["tile"]
        mask = tile["mask"]
        color = int(tile["color"])

        if color == most_color:
            out[r0 : r1 + 1, c0 : c1 + 1] = np.where(mask, BG, RED)
        elif color == least_color:
            out[r0 : r1 + 1, c0 : c1 + 1] = src
        else:
            assert color == middle_color
            out[r0 : r1 + 1, c0 : c1 + 1] = BG

    return out


def _random_connected_mask(tile_size: int, used_shapes: set[tuple[tuple[int, int], ...]]) -> np.ndarray | None:
    max_area = min(MAX_FG_CELLS, tile_size * tile_size - 1)
    for _ in range(300):
        area = int(np.random.randint(MIN_FG_CELLS, max_area + 1))
        start = (int(np.random.randint(tile_size)), int(np.random.randint(tile_size)))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < tile_size and 0 <= nc < tile_size and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            choice = frontier[int(np.random.randint(len(frontier)))]
            cells.add(choice)
        if len(cells) != area:
            continue

        norm = _normalize_cells(cells)
        if norm in used_shapes:
            continue

        bbox_h = max(r for r, _ in norm) + 1
        bbox_w = max(c for _, c in norm) + 1
        row_offset = int(np.random.randint(tile_size - bbox_h + 1))
        col_offset = int(np.random.randint(tile_size - bbox_w + 1))
        shifted = {(r + row_offset, c + col_offset) for r, c in norm}

        mask = np.zeros((tile_size, tile_size), dtype=bool)
        for r, c in shifted:
            mask[r, c] = True

        if not _mask_connected(mask):
            continue
        used_shapes.add(norm)
        return mask
    return None


def _build_candidate() -> np.ndarray | None:
    tile_size = int(TILE_SIZES[int(np.random.randint(len(TILE_SIZES)))])
    side = 3 * tile_size + 2
    grid = np.zeros((side, side), dtype=int)

    sep_rows = [tile_size, 2 * tile_size + 1]
    sep_cols = [tile_size, 2 * tile_size + 1]
    grid[sep_rows, :] = GREY
    grid[:, sep_cols] = GREY

    colors = np.random.choice(np.array(INPUT_PALETTE, dtype=int), size=3, replace=False).astype(int).tolist()
    pattern = list(COUNT_PATTERNS[int(np.random.randint(len(COUNT_PATTERNS)))])
    np.random.shuffle(pattern)
    color_counts = dict(zip(colors, pattern))

    tile_colors: list[int] = []
    for color in colors:
        tile_colors.extend([int(color)] * int(color_counts[color]))
    np.random.shuffle(tile_colors)
    assert len(tile_colors) == 9

    used_shapes: dict[int, set[tuple[tuple[int, int], ...]]] = {int(color): set() for color in colors}

    index = 0
    for tile_r in range(3):
        for tile_c in range(3):
            color = int(tile_colors[index])
            index += 1
            mask = _random_connected_mask(tile_size, used_shapes[color])
            if mask is None:
                return None
            r0 = tile_r * (tile_size + 1)
            c0 = tile_c * (tile_size + 1)
            tile_view = grid[r0 : r0 + tile_size, c0 : c0 + tile_size]
            tile_view[mask] = color

    return grid


def _validate_input(grid: np.ndarray) -> None:
    parsed = _parse_grid(grid)
    counts = _color_frequencies(parsed["tiles"])
    assert len(counts) == 3
    ranked_counts = sorted(counts.values())
    assert ranked_counts[0] < ranked_counts[1] < ranked_counts[2]

    seen_shapes: dict[int, set[tuple[tuple[int, int], ...]]] = {color: set() for color in counts}
    for tile in parsed["tiles"]:
        mask = tile["mask"]
        assert MIN_FG_CELLS <= int(mask.sum()) <= MAX_FG_CELLS
        assert _mask_connected(mask), "each tile pattern must be 4-connected"
        norm = _normalize_cells({tuple(map(int, rc)) for rc in np.argwhere(mask)})
        color = int(tile["color"])
        assert norm not in seen_shapes[color], "same-color tiles must use distinct shapes"
        seen_shapes[color].add(norm)


def _validate_semantics(inp: np.ndarray, out: np.ndarray) -> None:
    parsed = _parse_grid(inp)
    counts = _color_frequencies(parsed["tiles"])
    ranked = sorted(counts, key=lambda color: counts[color])
    least_color, middle_color, most_color = ranked

    assert np.all(out[parsed["row_separators"], :] == GREY)
    assert np.all(out[:, parsed["col_separators"]] == GREY)

    for tile in parsed["tiles"]:
        r0, r1, c0, c1 = tile["r0"], tile["r1"], tile["c0"], tile["c1"]
        src = tile["tile"]
        dst = out[r0 : r1 + 1, c0 : c1 + 1]
        mask = tile["mask"]
        color = int(tile["color"])

        if color == most_color:
            assert np.array_equal(dst, np.where(mask, BG, RED))
        elif color == least_color:
            assert np.array_equal(dst, src)
        else:
            assert color == middle_color
            assert np.all(dst == BG)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _build_candidate()
        if candidate is None:
            continue
        try:
            _validate_input(candidate)
            result = main(candidate)
            _validate_semantics(candidate, result)
            return candidate
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input")
