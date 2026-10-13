from __future__ import annotations

from collections import Counter
import random

import numpy as np


SEPARATOR_WIDTH = 1
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_POOL = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 500


def _most_common_color(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected at least one cell"
    return int(ranked[0][0])


def _separator_lines(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    height, width = map(int, grid.shape)
    candidates: list[tuple[int, list[int], list[int]]] = []
    for color in np.unique(grid):
        color = int(color)
        rows = [r for r in range(height) if np.all(grid[r, :] == color)]
        cols = [c for c in range(width) if np.all(grid[:, c] == color)]
        if rows and cols:
            candidates.append((color, rows, cols))
    assert len(candidates) == 1, "expected exactly one separator color"
    return candidates[0]


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for sep in separators:
        assert start < sep, "separator lines must leave non-empty tiles"
        spans.append((start, sep - 1))
        start = sep + 1
    assert start < length, "separator lines must leave a final tile band"
    spans.append((start, length - 1))
    return spans


def _tile_key(tile: np.ndarray) -> tuple[int, ...]:
    return tuple(map(int, tile.reshape(-1)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the hidden sprite by overlaying only the per-row odd-cell differences.

    Input structure:
    - Full separator rows and columns partition the grid into equal rectangular cells.
    - Within each tile row, exactly C-1 cells are identical and one odd cell differs in a few positions.
    - Different tile rows hide their odd-cell differences at disjoint in-cell positions.

    Rule:
    - Split the grid into tiles using the separator lines.
    - For each tile row, identify the repeated common tile and compare the unique odd tile against it cell by cell.
    - Start a blank output tile filled with the background color, then paint every differing position with the odd tile's color.

    Key insight:
    - The answer is not any full odd tile; each row contributes only its changed pixels, and those sparse fragments must be overlaid across rows.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    separator, sep_rows, sep_cols = _separator_lines(g)
    row_spans = _spans(g.shape[0], sep_rows)
    col_spans = _spans(g.shape[1], sep_cols)
    assert len(col_spans) >= 3, "each row must contain at least three tiles"

    tile_heights = {r1 - r0 + 1 for r0, r1 in row_spans}
    tile_widths = {c1 - c0 + 1 for c0, c1 in col_spans}
    assert len(tile_heights) == 1 and len(tile_widths) == 1, "tiles must be equal-sized"
    tile_h = tile_heights.pop()
    tile_w = tile_widths.pop()

    bg = _most_common_color(g[g != separator])
    out = np.full((tile_h, tile_w), bg, dtype=int)
    used_positions: set[tuple[int, int]] = set()

    for r0, r1 in row_spans:
        row_tiles = [g[r0 : r1 + 1, c0 : c1 + 1] for c0, c1 in col_spans]
        counts = Counter(_tile_key(tile) for tile in row_tiles)
        assert len(counts) == 2, "each tile row must contain one odd tile and one repeated tile"
        assert sorted(counts.values()) == [1, len(col_spans) - 1], "expected one odd tile per row"

        common_key = max(counts, key=counts.get)
        common_tile = next(tile for tile in row_tiles if _tile_key(tile) == common_key)
        odd_tile = next(tile for tile in row_tiles if _tile_key(tile) != common_key)

        diff_mask = odd_tile != common_tile
        assert bool(diff_mask.any()), "odd tile must differ from the repeated tile"
        for rr, cc in np.argwhere(diff_mask):
            pos = (int(rr), int(cc))
            assert pos not in used_positions, "row differences must occupy disjoint in-cell positions"
            used_positions.add(pos)
            out[pos] = int(odd_tile[pos])

    assert used_positions, "expected at least one hidden sprite pixel"
    return out


def _sample_layout() -> tuple[int, int, int, int, int]:
    for _ in range(200):
        rows = random.randint(3, 5)
        cols = random.randint(3, 5)
        max_h = min(7, (30 - (rows - 1) * SEPARATOR_WIDTH) // rows)
        max_w = min(7, (30 - (cols - 1) * SEPARATOR_WIDTH) // cols)
        if max_h < 4 or max_w < 4:
            continue

        tile_h = random.randint(4, max_h)
        tile_w = random.randint(4, max_w)
        tile_area = tile_h * tile_w
        max_sprite_area = min(tile_area // 2, 2 * rows + 3)
        if max_sprite_area < rows:
            continue
        min_sprite_area = rows + 1 if max_sprite_area >= rows + 1 else rows
        sprite_area = random.randint(min_sprite_area, max_sprite_area)
        return rows, cols, tile_h, tile_w, sprite_area
    raise RuntimeError("failed to sample a valid layout")


def _neighbors4(r: int, c: int, height: int, width: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < height and 0 <= cc < width:
            out.append((rr, cc))
    return out


def _generate_sprite_cells(tile_h: int, tile_w: int, area: int) -> list[tuple[int, int]] | None:
    for _ in range(200):
        row_margin = 1 if tile_h >= 5 else 0
        col_margin = 1 if tile_w >= 5 else 0
        start = (
            random.randint(row_margin, tile_h - 1 - row_margin),
            random.randint(col_margin, tile_w - 1 - col_margin),
        )
        cells = {start}
        while len(cells) < area:
            frontier = {
                nbr
                for r, c in cells
                for nbr in _neighbors4(r, c, tile_h, tile_w)
                if nbr not in cells
            }
            if not frontier:
                break
            cells.add(random.choice(sorted(frontier)))

        if len(cells) != area:
            continue
        rows = {r for r, _c in cells}
        cols = {c for _r, c in cells}
        if len(rows) < 2 or len(cols) < 2:
            continue
        return sorted(cells)
    return None


def _assign_sprite_colors(
    cells: list[tuple[int, int]], sprite_palette: list[int]
) -> dict[tuple[int, int], int]:
    shuffled = cells.copy()
    random.shuffle(shuffled)
    color_map: dict[tuple[int, int], int] = {}
    forced = shuffled[: min(len(shuffled), len(sprite_palette))]
    for cell, color in zip(forced, sprite_palette):
        color_map[cell] = int(color)
    for cell in shuffled[len(forced) :]:
        color_map[cell] = int(random.choice(sprite_palette))
    return color_map


def _partition_sprite_cells(
    cells: list[tuple[int, int]], row_count: int
) -> list[list[tuple[int, int]]]:
    assert len(cells) >= row_count
    sizes = [1] * row_count
    for _ in range(len(cells) - row_count):
        eligible = [idx for idx, size in enumerate(sizes) if size < 4]
        sizes[random.choice(eligible)] += 1

    shuffled = cells.copy()
    random.shuffle(shuffled)
    groups: list[list[tuple[int, int]]] = []
    start = 0
    for size in sizes:
        groups.append(shuffled[start : start + size])
        start += size
    assert start == len(shuffled)
    return groups


def _paint_horizontal(tile: np.ndarray, row: int, c0: int, c1: int, color: int) -> None:
    tile[row, c0 : c1 + 1] = int(color)


def _paint_vertical(tile: np.ndarray, col: int, r0: int, r1: int, color: int) -> None:
    tile[r0 : r1 + 1, col] = int(color)


def _make_base_template(tile_h: int, tile_w: int, bg: int, palette: list[int]) -> np.ndarray:
    tile = np.full((tile_h, tile_w), bg, dtype=int)
    color_a = int(random.choice(palette))
    color_b = int(random.choice(palette))
    family = random.randrange(4)

    if family == 0:
        row = random.choice([1, tile_h - 2])
        left = 1 if tile_w <= 4 else random.randint(1, 2)
        right = tile_w - 2 if tile_w <= 4 else random.randint(tile_w - 3, tile_w - 2)
        if right - left + 1 < 2:
            left, right = 1, tile_w - 2
        _paint_horizontal(tile, row, left, right, color_a)
        tile[tile_h - 1 - row, 1] = color_b
        tile[tile_h - 1 - row, tile_w - 2] = color_b
        tile[tile_h // 2, random.randint(1, tile_w - 2)] = color_a
    elif family == 1:
        col = random.choice([1, tile_w - 2])
        top = 1 if tile_h <= 4 else random.randint(1, 2)
        bottom = tile_h - 2 if tile_h <= 4 else random.randint(tile_h - 3, tile_h - 2)
        if bottom - top + 1 < 2:
            top, bottom = 1, tile_h - 2
        _paint_vertical(tile, col, top, bottom, color_a)
        tile[1, tile_w - 1 - col] = color_b
        tile[tile_h - 2, tile_w - 1 - col] = color_b
        tile[random.randint(1, tile_h - 2), tile_w // 2] = color_a
    elif family == 2:
        r0 = 1 if tile_h <= 5 else random.randint(1, tile_h - 3)
        c0 = 1 if tile_w <= 5 else random.randint(1, tile_w - 3)
        tile[r0 : r0 + 2, c0 : c0 + 2] = color_a
        tail_options = [
            (r0 - 1, c0),
            (r0 + 2, c0 + 1),
            (r0, c0 - 1),
            (r0 + 1, c0 + 2),
        ]
        valid_tails = [
            (r, c)
            for r, c in tail_options
            if 0 <= r < tile_h and 0 <= c < tile_w and 0 < r < tile_h - 1 and 0 < c < tile_w - 1
        ]
        if not valid_tails:
            valid_tails = [(r, c) for r in range(1, tile_h - 1) for c in range(1, tile_w - 1)]
        tile[random.choice(valid_tails)] = color_b
        tile[random.choice([1, tile_h - 2]), random.choice([1, tile_w - 2])] = color_b
    else:
        row = random.randint(1, tile_h - 2)
        col = random.randint(1, tile_w - 2)
        _paint_horizontal(tile, row, 1, tile_w - 2, color_a)
        _paint_vertical(tile, col, 1, tile_h - 2, color_b)
        tile[random.choice([1, tile_h - 2]), random.randint(1, tile_w - 2)] = color_a
        tile[random.randint(1, tile_h - 2), random.choice([1, tile_w - 2])] = color_b

    for _ in range(random.randint(0, 2)):
        tile[random.randint(1, tile_h - 2), random.randint(1, tile_w - 2)] = int(random.choice(palette))
    return tile


def _enforce_diff_mode(
    tile: np.ndarray,
    diff_cells: list[tuple[int, int]],
    palette: list[int],
    bg: int,
    mode: str,
) -> None:
    if not diff_cells:
        return
    if mode == "accent":
        tile[random.choice(diff_cells)] = int(random.choice(palette))
        return
    if mode == "bg":
        tile[random.choice(diff_cells)] = int(bg)
        return

    accent_cell = random.choice(diff_cells)
    tile[accent_cell] = int(random.choice(palette))
    remaining = [cell for cell in diff_cells if cell != accent_cell]
    if remaining:
        tile[random.choice(remaining)] = int(bg)


def _sample_odd_columns(row_count: int, col_count: int) -> list[int]:
    odd_cols = [random.randrange(col_count) for _ in range(row_count)]
    if len(set(odd_cols)) == 1:
        idx = random.randrange(row_count)
        options = [c for c in range(col_count) if c != odd_cols[idx]]
        odd_cols[idx] = random.choice(options)
    return odd_cols


def _build_input_grid(
    row_templates: list[np.ndarray],
    row_groups: list[list[tuple[int, int]]],
    sprite_colors: dict[tuple[int, int], int],
    odd_cols: list[int],
    rows: int,
    cols: int,
    tile_h: int,
    tile_w: int,
    separator: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    height = rows * tile_h + (rows - 1) * SEPARATOR_WIDTH
    width = cols * tile_w + (cols - 1) * SEPARATOR_WIDTH
    grid = np.full((height, width), separator, dtype=int)
    odd_tiles: list[np.ndarray] = []

    for row_idx in range(rows):
        common_tile = row_templates[row_idx]
        odd_tile = common_tile.copy()
        for cell in row_groups[row_idx]:
            odd_tile[cell] = int(sprite_colors[cell])
        odd_tiles.append(odd_tile)

        for col_idx in range(cols):
            tile = odd_tile if col_idx == odd_cols[row_idx] else common_tile
            r0 = row_idx * (tile_h + SEPARATOR_WIDTH)
            c0 = col_idx * (tile_w + SEPARATOR_WIDTH)
            grid[r0 : r0 + tile_h, c0 : c0 + tile_w] = tile

    return grid, odd_tiles


def _expected_output(
    tile_h: int,
    tile_w: int,
    bg: int,
    sprite_colors: dict[tuple[int, int], int],
) -> np.ndarray:
    out = np.full((tile_h, tile_w), bg, dtype=int)
    for (r, c), color in sprite_colors.items():
        out[r, c] = int(color)
    return out


def _full_odd_overlay(bg: int, odd_tiles: list[np.ndarray]) -> np.ndarray:
    overlaid = np.full_like(odd_tiles[0], bg)
    for tile in odd_tiles:
        mask = tile != bg
        overlaid[mask] = tile[mask]
    return overlaid


def _generate_attempt() -> np.ndarray | None:
    rows, cols, tile_h, tile_w, sprite_area = _sample_layout()

    separator, bg = random.sample(COLOR_POOL, 2)
    remaining = [color for color in COLOR_POOL if color not in {separator, bg}]
    sprite_color_count = random.randint(1, min(3, len(remaining) - 1))
    sprite_palette = random.sample(remaining, sprite_color_count)
    template_palette = [color for color in remaining if color not in sprite_palette]
    if not template_palette:
        return None

    sprite_cells = _generate_sprite_cells(tile_h, tile_w, sprite_area)
    if sprite_cells is None:
        return None
    sprite_colors = _assign_sprite_colors(sprite_cells, sprite_palette)
    row_groups = _partition_sprite_cells(sprite_cells, rows)

    modes = ["accent", "bg"] + [random.choice(["accent", "bg", "mixed"]) for _ in range(rows - 2)]
    random.shuffle(modes)

    row_templates: list[np.ndarray] = []
    for row_idx in range(rows):
        palette_size = random.randint(1, min(3, len(template_palette)))
        row_palette = random.sample(template_palette, palette_size)
        tile = _make_base_template(tile_h, tile_w, bg, row_palette)
        _enforce_diff_mode(tile, row_groups[row_idx], row_palette, bg, modes[row_idx])
        row_templates.append(tile)

    if len({_tile_key(tile) for tile in row_templates}) < 2:
        return None

    odd_cols = _sample_odd_columns(rows, cols)
    grid, odd_tiles = _build_input_grid(
        row_templates,
        row_groups,
        sprite_colors,
        odd_cols,
        rows,
        cols,
        tile_h,
        tile_w,
        separator,
    )
    expected = _expected_output(tile_h, tile_w, bg, sprite_colors)

    if _most_common_color(grid[grid != separator]) != bg:
        return None
    if np.array_equal(_full_odd_overlay(bg, odd_tiles), expected):
        return None
    if any(np.array_equal(tile, expected) for tile in row_templates):
        return None
    if any(np.array_equal(tile, expected) for tile in odd_tiles):
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-difference puzzle")
