from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
TILE_COLOR = 2
SEPARATOR_COLOR = 5
CODE_PALETTE = (1, 3, 4, 6, 7, 8, 9)
RETRY_LIMIT = 400


def _most_common_color(values: np.ndarray) -> int:
    return int(Counter(map(int, values.reshape(-1))).most_common(1)[0][0])


def _content_bands(separator_indices: list[int], limit: int) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for idx in separator_indices:
        if start < idx:
            bands.append((start, idx))
        start = idx + 1
    if start < limit:
        bands.append((start, limit))
    return bands


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]), "expected a unique majority"
    return int(ranked[0][0])


def _parse_upper_tiles(grid: np.ndarray, separator_row: int, separator: int) -> tuple[np.ndarray, int]:
    top = grid[:separator_row, :]
    row_separators = [r for r in range(top.shape[0]) if np.all(top[r, :] == separator)]
    col_separators = [c for c in range(top.shape[1]) if np.all(top[:, c] == separator)]

    row_bands = _content_bands(row_separators, top.shape[0])
    col_bands = _content_bands(col_separators, top.shape[1])
    assert len(row_bands) == 2 and len(col_bands) == 3, "expected a 2x3 tile lattice"

    heights = {r1 - r0 for r0, r1 in row_bands}
    widths = {c1 - c0 for c0, c1 in col_bands}
    assert len(heights) == 1 and len(widths) == 1, "tiles must have equal size"
    tile_h = heights.pop()
    tile_w = widths.pop()

    blocks = np.empty((len(row_bands), len(col_bands), tile_h, tile_w), dtype=int)
    for br, (r0, r1) in enumerate(row_bands):
        for bc, (c0, c1) in enumerate(col_bands):
            blocks[br, bc] = top[r0:r1, c0:c1]

    motif = np.empty((tile_h, tile_w), dtype=int)
    for r in range(tile_h):
        for c in range(tile_w):
            motif[r, c] = _unique_mode(blocks[:, :, r, c])

    diff_mask = blocks != motif
    diff_counts = diff_mask.reshape(len(row_bands), len(col_bands), tile_h * tile_w).sum(axis=2)
    damaged = np.argwhere(diff_counts > 0)
    assert damaged.shape[0] == 1, "expected exactly one damaged tile"

    dr, dc = map(int, damaged[0])
    damaged_mask = diff_mask[dr, dc]
    damage_count = int(damaged_mask.sum())
    assert damage_count >= 2, "damage count must be at least 2"
    assert np.all(blocks[dr, dc][damaged_mask] == BG), "damage must erase motif pixels to background"
    assert np.all(motif[damaged_mask] == TILE_COLOR), "only foreground motif pixels may be damaged"
    return motif, damage_count


def _find_layout(grid: np.ndarray) -> tuple[int, int, int]:
    bg = _most_common_color(grid)
    solid_rows = [
        r for r in range(grid.shape[0]) if np.all(grid[r, :] == grid[r, 0]) and int(grid[r, 0]) != bg
    ]
    assert solid_rows, "expected separator rows"
    separator_row = max(solid_rows)
    separator = int(grid[separator_row, 0])
    assert separator == SEPARATOR_COLOR, "expected the histogram separator color"

    code_rows = [
        r
        for r in range(separator_row + 1, grid.shape[0])
        if np.any(grid[r, :] != bg) and not np.all(grid[r, :] == separator)
    ]
    assert len(code_rows) == 1, "expected exactly one code row"
    code_row = code_rows[0]
    assert code_row - separator_row - 1 >= 2, "expected blank space for bars below the separator"
    return bg, separator_row, code_row


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the damaged-tile count and use it to draw matching histogram bars below the separator.

    Input structure:
    - The upper region is a 2x3 lattice of equal tiles separated by one-cell grey rows and columns.
    - Every tile shows the same red-on-black motif except one damaged copy, where some red cells were erased to black.
    - Below the tile region sits a full-width grey separator row, then a black gap, then a sparse one-row color code.

    Rule:
    - Reconstruct the consensus motif by taking the unique majority color at each local tile coordinate.
    - Find the lone damaged tile and count how many coordinates differ from the consensus; call this count D.
    - In the code row, count each non-black color. For every code color whose frequency is exactly D, fill D cells downward from the row immediately below the grey separator in every column where that color appears.

    Key insight:
    - The upper tile lattice does not need to be repaired for its own sake; it encodes a damage count D that becomes the frequency threshold and bar height used in the lower histogram area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, separator_row, code_row = _find_layout(g)
    motif, damage_count = _parse_upper_tiles(g, separator_row, SEPARATOR_COLOR)
    assert separator_row + damage_count < code_row, "bars must fit between separator and code row"
    assert np.all(g[separator_row + 1 : separator_row + 1 + damage_count, :] != SEPARATOR_COLOR)

    code = g[code_row, :]
    color_counts = Counter(int(color) for color in code if int(color) != bg)
    matching_colors = {color for color, count in color_counts.items() if count == damage_count}
    assert matching_colors, "expected at least one code color with matching frequency"

    out = g.copy()
    for c, color in enumerate(map(int, code)):
        if color not in matching_colors:
            continue
        assert np.all(out[separator_row + 1 : separator_row + 1 + damage_count, c] == bg), "bar path must be empty"
        out[separator_row + 1 : separator_row + 1 + damage_count, c] = color
    return out


def _neighbors4(r: int, c: int, size: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < size and 0 <= nc < size:
            out.append((nr, nc))
    return out


def _grow_connected_mask(size: int, count: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    start = (int(np.random.randint(size)), int(np.random.randint(size)))
    mask[start] = True
    while int(mask.sum()) < count:
        frontier = {
            cell
            for r, c in np.argwhere(mask)
            for cell in _neighbors4(int(r), int(c), size)
            if not mask[cell]
        }
        pick = list(frontier)[int(np.random.randint(len(frontier)))]
        mask[pick] = True
    return mask


def _build_motif(size: int, damage_count: int) -> np.ndarray:
    min_red = max(damage_count + 1, size + 1)
    max_red = min(size * size - 2, size * size // 2 + 2)
    red_count = int(np.random.randint(min_red, max_red + 1))
    motif = np.zeros((size, size), dtype=int)
    motif[_grow_connected_mask(size, red_count)] = TILE_COLOR
    return motif


def _build_upper_lattice(motif: np.ndarray, damage_cells: np.ndarray) -> np.ndarray:
    size = motif.shape[0]
    rows = 2 * size + 1
    cols = 3 * size + 2
    grid = np.full((rows, cols), SEPARATOR_COLOR, dtype=int)
    damaged_slot = (int(np.random.randint(2)), int(np.random.randint(3)))
    for br in range(2):
        for bc in range(3):
            r0 = br * (size + 1)
            c0 = bc * (size + 1)
            grid[r0 : r0 + size, c0 : c0 + size] = motif
            if (br, bc) == damaged_slot:
                grid[r0 + damage_cells[:, 0], c0 + damage_cells[:, 1]] = BG
    return grid


def _sample_code_plan(width: int, damage_count: int) -> tuple[np.ndarray, set[int]] | None:
    max_matching = min(2, max(1, (width - 2) // damage_count))
    if max_matching <= 0:
        return None
    matching_count = int(np.random.randint(1, max_matching + 1))

    remaining = width - matching_count * damage_count
    if remaining < 2:
        return None

    other_color_count = int(np.random.randint(1, min(3, len(CODE_PALETTE) - matching_count) + 1))
    others: list[int] = []
    attempts = 0
    while attempts < 80:
        attempts += 1
        choices = np.random.choice(np.arange(1, max(remaining, 2)), size=other_color_count, replace=True)
        counts = [int(x) for x in choices if int(x) != damage_count]
        if len(counts) != other_color_count:
            continue
        if sum(counts) >= remaining:
            continue
        blacks = remaining - sum(counts)
        if blacks < 1:
            continue
        others = counts
        break
    if not others:
        return None

    colors = list(np.random.choice(CODE_PALETTE, size=matching_count + other_color_count, replace=False))
    matching_colors = set(map(int, colors[:matching_count]))
    values: list[int] = [BG] * (remaining - sum(others))
    for color in colors[:matching_count]:
        values.extend([int(color)] * damage_count)
    for color, count in zip(colors[matching_count:], others):
        values.extend([int(color)] * int(count))
    if len(values) != width:
        return None
    np.random.shuffle(values)
    return np.array(values, dtype=int), matching_colors


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    size = int(np.random.randint(4, 7))
    damage_count = int(np.random.randint(2, min(6, size * size - 2) + 1))
    motif = _build_motif(size, damage_count)

    red_cells = np.argwhere(motif == TILE_COLOR)
    if red_cells.shape[0] <= damage_count:
        return None
    damage_indices = np.random.choice(red_cells.shape[0], size=damage_count, replace=False)
    damage_cells = red_cells[damage_indices]

    upper = _build_upper_lattice(motif, damage_cells)
    width = upper.shape[1]
    plan = _sample_code_plan(width, damage_count)
    if plan is None:
        return None
    code_row, matching_colors = plan

    blank_height = int(np.random.randint(max(6, damage_count), max(6, damage_count) + 4))
    height = upper.shape[0] + 1 + blank_height + 1
    if height > 30 or width > 30:
        return None

    inp = np.zeros((height, width), dtype=int)
    inp[: upper.shape[0], :] = upper
    separator_row = upper.shape[0]
    inp[separator_row, :] = SEPARATOR_COLOR
    code_row_index = height - 1
    inp[code_row_index, :] = code_row

    out = inp.copy()
    for c, color in enumerate(map(int, code_row)):
        if color in matching_colors:
            out[separator_row + 1 : separator_row + 1 + damage_count, c] = color

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        case = _build_case()
        if case is None:
            continue
        inp, expected = case
        try:
            observed = main(inp)
        except Exception:
            continue
        if np.array_equal(observed, expected):
            return inp
    raise RuntimeError(f"generate_input failed after {RETRY_LIMIT} attempts")
