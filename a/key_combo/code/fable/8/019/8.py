from __future__ import annotations

import random
from collections import Counter

import numpy as np


SEPARATOR_WIDTH = 1
GRID_ROWS_RANGE = (3, 5)
GRID_COLS_RANGE = (3, 5)
TILE_HEIGHT_RANGE = (3, 5)
TILE_WIDTH_RANGE = (3, 5)
SEED_COLOR_COUNT_RANGE = (1, 3)
SEED_FILL_FRACTION_RANGE = (0.28, 0.5)
EXTRA_MARKER_RANGE = (0, 3)
MAX_GENERATION_ATTEMPTS = 400


def _mode_color(values: np.ndarray) -> int:
    return int(Counter(map(int, values.tolist())).most_common(1)[0][0])


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    prev = -1
    for idx in separators:
        if idx <= prev + 1:
            return []
        spans.append((prev + 1, idx - 1))
        prev = idx
    if prev >= length - 1:
        return []
    spans.append((prev + 1, length - 1))
    return spans


def _tile_origin(tile_r: int, tile_c: int, tile_h: int, tile_w: int) -> tuple[int, int]:
    step_r = tile_h + SEPARATOR_WIDTH
    step_c = tile_w + SEPARATOR_WIDTH
    return tile_r * step_r, tile_c * step_c


def _apply_parity_transform(tile: np.ndarray, row_delta: int, col_delta: int) -> np.ndarray:
    out = tile.copy()
    if col_delta % 2:
        out = np.fliplr(out)
    if row_delta % 2:
        out = np.flipud(out)
    return out


def _find_structure(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]], int]:
    candidates = []
    unique_colors = sorted(int(v) for v in np.unique(grid))
    for separator in unique_colors:
        sep_rows = np.flatnonzero(np.all(grid == separator, axis=1)).tolist()
        sep_cols = np.flatnonzero(np.all(grid == separator, axis=0)).tolist()
        if not sep_rows or not sep_cols:
            continue

        row_spans = _spans(grid.shape[0], sep_rows)
        col_spans = _spans(grid.shape[1], sep_cols)
        if len(row_spans) < 2 or len(col_spans) < 2:
            continue

        row_heights = {r1 - r0 + 1 for r0, r1 in row_spans}
        col_widths = {c1 - c0 + 1 for c0, c1 in col_spans}
        if len(row_heights) != 1 or len(col_widths) != 1:
            continue

        non_sep = grid[grid != separator]
        if non_sep.size == 0:
            continue
        bg = _mode_color(non_sep)

        source_count = 0
        marked_count = 0
        valid = True
        for r0, r1 in row_spans:
            for c0, c1 in col_spans:
                tile = grid[r0 : r1 + 1, c0 : c1 + 1]
                colors = set(map(int, np.unique(tile)))
                motif_colors = colors - {bg, separator}
                has_marker = separator in colors
                if motif_colors:
                    source_count += 1
                elif has_marker:
                    if not colors <= {bg, separator}:
                        valid = False
                        break
                    marked_count += 1
                else:
                    if colors != {bg}:
                        valid = False
                        break
            if not valid:
                break

        if valid and source_count == 1 and marked_count >= 1:
            candidates.append((separator, bg, row_spans, col_spans))

    assert len(candidates) == 1, "expected exactly one separator-grid interpretation"
    separator, bg, row_spans, col_spans = candidates[0]
    return int(separator), int(bg), row_spans, col_spans, int(row_spans[0][1] - row_spans[0][0] + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the seed tile into each marked cell, reflecting it by row and column offset parity.

    Input structure:
    - Full separator-colored rows and columns partition the grid into equal-sized tiles.
    - Exactly one tile contains the only seed motif made of non-background, non-separator colors.
    - Some other tiles contain only background plus one or more separator-colored marker pixels.

    Rule:
    - Keep the separator lattice and the seed tile unchanged.
    - For each marked tile, copy the entire seed tile into that tile.
    - If the marked tile is an odd column offset from the seed, mirror left-right first; if it is an odd row offset from the seed, mirror up-down next.

    Key insight:
    - Marker tiles select where stamping happens, while the row/column parity of the tile displacement independently determines horizontal and vertical reflection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    separator, bg, row_spans, col_spans, _tile_h = _find_structure(g)

    source_pos: tuple[int, int] | None = None
    source_tile: np.ndarray | None = None
    marked_tiles: list[tuple[int, int]] = []

    for tile_r, (r0, r1) in enumerate(row_spans):
        for tile_c, (c0, c1) in enumerate(col_spans):
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            colors = set(map(int, np.unique(tile)))
            motif_colors = colors - {bg, separator}
            if motif_colors:
                assert source_pos is None, "expected exactly one seed tile"
                source_pos = (tile_r, tile_c)
                source_tile = tile.copy()
            elif separator in colors:
                marked_tiles.append((tile_r, tile_c))

    assert source_pos is not None and source_tile is not None, "missing seed tile"

    out = g.copy()
    for tile_r, tile_c in marked_tiles:
        r0, r1 = row_spans[tile_r]
        c0, c1 = col_spans[tile_c]
        stamped = _apply_parity_transform(
            source_tile,
            tile_r - source_pos[0],
            tile_c - source_pos[1],
        )
        out[r0 : r1 + 1, c0 : c1 + 1] = stamped

    return out


def _build_base_grid(
    tile_rows: int,
    tile_cols: int,
    tile_h: int,
    tile_w: int,
    bg: int,
    separator: int,
) -> np.ndarray:
    height = tile_rows * tile_h + (tile_rows - 1) * SEPARATOR_WIDTH
    width = tile_cols * tile_w + (tile_cols - 1) * SEPARATOR_WIDTH
    grid = np.full((height, width), bg, dtype=int)
    grid[tile_h :: tile_h + SEPARATOR_WIDTH, :] = separator
    grid[:, tile_w :: tile_w + SEPARATOR_WIDTH] = separator
    return grid


def _seed_variants(tile: np.ndarray) -> list[np.ndarray]:
    return [
        tile,
        np.fliplr(tile),
        np.flipud(tile),
        np.flipud(np.fliplr(tile)),
    ]


def _make_seed_tile(tile_h: int, tile_w: int, bg: int, palette: list[int]) -> np.ndarray | None:
    area = tile_h * tile_w
    min_fill = max(3, int(np.ceil(area * SEED_FILL_FRACTION_RANGE[0])))
    max_fill = min(area - 1, int(np.floor(area * SEED_FILL_FRACTION_RANGE[1])))
    if min_fill > max_fill:
        return None

    fill_count = random.randint(min_fill, max_fill)
    color_count = random.randint(SEED_COLOR_COUNT_RANGE[0], min(SEED_COLOR_COUNT_RANGE[1], fill_count, len(palette)))
    colors = random.sample(palette, color_count)

    tile = np.full((tile_h, tile_w), bg, dtype=int)
    positions = random.sample([(r, c) for r in range(tile_h) for c in range(tile_w)], fill_count)
    assignments = [colors[i % color_count] for i in range(fill_count)]
    random.shuffle(assignments)
    for (r, c), color in zip(positions, assignments):
        tile[r, c] = color

    variants = _seed_variants(tile)
    signatures = {variant.tobytes() for variant in variants}
    if len(signatures) != 4:
        return None
    return tile


def _group_positions(
    tile_rows: int,
    tile_cols: int,
    source_pos: tuple[int, int],
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    groups = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}
    for tile_r in range(tile_rows):
        for tile_c in range(tile_cols):
            pos = (tile_r, tile_c)
            if pos == source_pos:
                continue
            key = ((tile_r - source_pos[0]) & 1, (tile_c - source_pos[1]) & 1)
            groups[key].append(pos)
    return groups


def _choose_source_pos(tile_rows: int, tile_cols: int) -> tuple[int, int] | None:
    candidates = []
    for tile_r in range(tile_rows):
        for tile_c in range(tile_cols):
            groups = _group_positions(tile_rows, tile_cols, (tile_r, tile_c))
            if all(groups[key] for key in groups):
                candidates.append((tile_r, tile_c))
    if not candidates:
        return None
    return random.choice(candidates)


def _choose_marker_positions(tile_rows: int, tile_cols: int, source_pos: tuple[int, int]) -> list[tuple[int, int]]:
    groups = _group_positions(tile_rows, tile_cols, source_pos)
    chosen = [random.choice(groups[key]) for key in [(0, 0), (0, 1), (1, 0), (1, 1)]]

    remaining = [
        pos
        for key, positions in groups.items()
        for pos in positions
        if pos not in chosen
    ]
    extra_max = min(EXTRA_MARKER_RANGE[1], len(remaining))
    extra_count = random.randint(EXTRA_MARKER_RANGE[0], extra_max)
    if extra_count:
        chosen.extend(random.sample(remaining, extra_count))
    random.shuffle(chosen)
    return chosen


def _marker_offsets(tile_h: int, tile_w: int) -> list[tuple[int, int]]:
    inner_rows = list(range(1, tile_h - 1))
    inner_cols = list(range(1, tile_w - 1))
    if inner_rows and inner_cols:
        return [(r, c) for r in inner_rows for c in inner_cols]
    return [(r, c) for r in range(tile_h) for c in range(tile_w)]


def _gen_attempt() -> np.ndarray | None:
    tile_rows = random.randint(*GRID_ROWS_RANGE)
    tile_cols = random.randint(*GRID_COLS_RANGE)
    tile_h = random.randint(*TILE_HEIGHT_RANGE)
    tile_w = random.randint(*TILE_WIDTH_RANGE)
    if tile_rows * tile_h + tile_rows - 1 > 30 or tile_cols * tile_w + tile_cols - 1 > 30:
        return None

    source_pos = _choose_source_pos(tile_rows, tile_cols)
    if source_pos is None:
        return None

    palette = random.sample(list(range(10)), 5)
    separator = palette[0]
    bg = palette[1]
    seed_palette = palette[2:]

    seed_tile = _make_seed_tile(tile_h, tile_w, bg, seed_palette)
    if seed_tile is None:
        return None

    marker_positions = _choose_marker_positions(tile_rows, tile_cols, source_pos)
    marker_offsets = _marker_offsets(tile_h, tile_w)

    grid = _build_base_grid(tile_rows, tile_cols, tile_h, tile_w, bg, separator)
    src_r0, src_c0 = _tile_origin(source_pos[0], source_pos[1], tile_h, tile_w)
    grid[src_r0 : src_r0 + tile_h, src_c0 : src_c0 + tile_w] = seed_tile

    for tile_r, tile_c in marker_positions:
        marker_r, marker_c = random.choice(marker_offsets)
        r0, c0 = _tile_origin(tile_r, tile_c, tile_h, tile_w)
        grid[r0 + marker_r, c0 + marker_c] = separator

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    if max(out.shape) > 30:
        return None

    expected_variants = {
        ((tile_r - source_pos[0]) & 1, (tile_c - source_pos[1]) & 1)
        for tile_r, tile_c in marker_positions
    }
    if len(expected_variants) < 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid odd-even echo input")
