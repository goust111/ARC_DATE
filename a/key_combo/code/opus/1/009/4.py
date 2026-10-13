from __future__ import annotations

import random
from collections import Counter, defaultdict
import itertools

import numpy as np


BG = 0
CYAN = 8
GRAY = 5
TILE_SIZE = 3
SEP = 1
STEP = TILE_SIZE + SEP
COLUMN_TO_COLOR = {0: 2, 1: 4, 2: 3}
MAX_GENERATION_ATTEMPTS = 500


def _non_separator_spans(is_separator: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for idx, sep in enumerate(is_separator.tolist()):
        if sep:
            if start is not None:
                spans.append((start, idx))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        spans.append((start, len(is_separator)))
    return spans


def _parse_tile_grid(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_spans = _non_separator_spans(np.all(g == BG, axis=1))
    col_spans = _non_separator_spans(np.all(g == BG, axis=0))
    assert row_spans and col_spans, "expected a tiled grid separated by full black rows and columns"
    assert all(r1 - r0 == TILE_SIZE for r0, r1 in row_spans), "expected 3x3 tile rows"
    assert all(c1 - c0 == TILE_SIZE for c0, c1 in col_spans), "expected 3x3 tile columns"
    return row_spans, col_spans


def _is_marker_tile(tile: np.ndarray) -> bool:
    non_bg = tile[tile != BG]
    return non_bg.size == 1 and int(non_bg[0]) == GRAY


def _invert_tile(tile: np.ndarray, fill_color: int) -> np.ndarray:
    colors = set(map(int, np.unique(tile)))
    assert colors <= {BG, CYAN}, "non-marker tiles must contain only black and cyan"
    assert BG in colors and CYAN in colors, "non-marker tiles must contain both black and cyan"
    out = np.full(tile.shape, fill_color, dtype=int)
    out[tile == CYAN] = BG
    return out


def _row_marker_info(row_tiles: list[np.ndarray]) -> tuple[int, int, int]:
    marker_candidates = []
    for col_idx, tile in enumerate(row_tiles):
        if _is_marker_tile(tile):
            marker_candidates.append((col_idx, tile))

    assert len(marker_candidates) == 1, "expected exactly one marker tile per tile row"
    marker_col_idx, marker_tile = marker_candidates[0]
    marker_pos = np.argwhere(marker_tile == GRAY)
    assert marker_pos.shape == (1, 2), "marker tile must contain exactly one grey pixel"
    marker_r, marker_c = map(int, marker_pos[0])
    assert marker_c in COLUMN_TO_COLOR, "marker column must be left, middle, or right"
    return marker_col_idx, marker_r, marker_c


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the grey marker's in-tile column to choose the inversion fill color for every other tile in that tile row.

    Input structure:
    - The grid is partitioned into a regular tile layout by single-cell full black separator rows and columns.
    - Each non-separator tile is 3x3.
    - In every tile row, exactly one special marker tile contains a single grey pixel on black.
    - Every other tile in that tile row contains a cyan-on-black pattern.

    Rule:
    - Detect the separator rows and columns, then split the grid into 3x3 tiles.
    - In each tile row, find the unique marker tile and read the grey pixel's column inside that tile.
    - Map marker column 0 -> red (2), 1 -> yellow (4), 2 -> green (3).
    - Erase the marker tile to black.
    - For every non-marker tile in the same tile row, invert occupancy: cyan pixels become black and black pixels become the mapped color.
    - Keep all separator rows and separator columns black.

    Key insight:
    - The grey pixel does not recolor its own tile; its horizontal position broadcasts one fill color to every other tile in the same tile row, while each target tile contributes only its cyan-vs-black occupancy mask.
    """
    g = np.asarray(grid, dtype=int)
    row_spans, col_spans = _parse_tile_grid(g)
    out = np.zeros_like(g)

    for r0, r1 in row_spans:
        row_tiles = [g[r0:r1, c0:c1] for c0, c1 in col_spans]
        marker_col_idx, _marker_r, marker_c = _row_marker_info(row_tiles)
        fill_color = COLUMN_TO_COLOR[marker_c]

        for col_idx, (c0, c1) in enumerate(col_spans):
            tile = g[r0:r1, c0:c1]
            if col_idx == marker_col_idx:
                continue
            assert not _is_marker_tile(tile), "tile row must have only one marker tile"
            out[r0:r1, c0:c1] = _invert_tile(tile, fill_color)

    return out


def _all_patterns() -> tuple[np.ndarray, ...]:
    patterns: list[np.ndarray] = []
    for bits in itertools.product([0, 1], repeat=TILE_SIZE * TILE_SIZE):
        mask = np.array(bits, dtype=bool).reshape(TILE_SIZE, TILE_SIZE)
        area = int(mask.sum())
        if not 3 <= area <= 6:
            continue
        if np.any(mask.sum(axis=0) == 0) or np.any(mask.sum(axis=1) == 0):
            continue
        patterns.append(mask)
    patterns.sort(key=lambda m: (int(m.sum()), tuple(int(x) for x in m.ravel())))
    return tuple(patterns)


PATTERN_LIBRARY = _all_patterns()


def _sample_control_columns(tile_rows: int) -> list[int]:
    controls = [0, 1, 2]
    controls.extend(random.randrange(TILE_SIZE) for _ in range(tile_rows - 3))
    random.shuffle(controls)
    return controls


def _sample_marker_rows(controls: list[int]) -> list[int]:
    while True:
        marker_rows = [random.randrange(TILE_SIZE) for _ in controls]
        grouped: dict[int, set[int]] = defaultdict(set)
        for code, marker_r in zip(controls, marker_rows):
            grouped[int(code)].add(int(marker_r))
        if any(len(rows) >= 2 for rows in grouped.values()):
            return marker_rows


def _sample_marker_tile_columns(tile_rows: int, tile_cols: int, controls: list[int]) -> list[int]:
    while True:
        marker_tile_cols = [random.randrange(tile_cols) for _ in range(tile_rows)]
        counts = Counter(marker_tile_cols)
        if len(counts) < 2 or max(counts.values()) == tile_rows:
            continue
        if not any(col > 2 for col in marker_tile_cols):
            continue
        if not any(marker_tile_cols[row] != controls[row] for row in range(tile_rows)):
            continue
        if not any(
            marker_tile_cols[i] == marker_tile_cols[j] and controls[i] != controls[j]
            for i in range(tile_rows)
            for j in range(i + 1, tile_rows)
        ):
            continue
        return marker_tile_cols


def _mask_key(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in mask.astype(int).ravel())


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    tile_rows = random.randint(4, 5)
    tile_cols = random.randint(4, 6)
    controls = _sample_control_columns(tile_rows)
    marker_rows = _sample_marker_rows(controls)
    marker_tile_cols = _sample_marker_tile_columns(tile_rows, tile_cols, controls)

    pool_size = random.randint(4, 7)
    pattern_pool = random.sample(list(PATTERN_LIBRARY), k=pool_size)
    shared_pattern = random.choice(pattern_pool)

    tile_masks: dict[tuple[int, int], np.ndarray] = {}
    for row in range(tile_rows):
        non_marker_cols = [col for col in range(tile_cols) if col != marker_tile_cols[row]]
        forced_col = random.choice(non_marker_cols)
        for col in non_marker_cols:
            tile_masks[(row, col)] = random.choice(pattern_pool)
        tile_masks[(row, forced_col)] = shared_pattern

        if len(non_marker_cols) >= 2:
            keys = {_mask_key(tile_masks[(row, col)]) for col in non_marker_cols}
            if len(keys) == 1:
                alt_patterns = [mask for mask in pattern_pool if not np.array_equal(mask, shared_pattern)]
                if not alt_patterns:
                    alt_patterns = [mask for mask in PATTERN_LIBRARY if not np.array_equal(mask, shared_pattern)]
                swap_col = random.choice([col for col in non_marker_cols if col != forced_col])
                tile_masks[(row, swap_col)] = random.choice(alt_patterns)

    height = tile_rows * TILE_SIZE + (tile_rows - 1) * SEP
    width = tile_cols * TILE_SIZE + (tile_cols - 1) * SEP
    inp = np.zeros((height, width), dtype=int)
    expected = np.zeros_like(inp)

    for row in range(tile_rows):
        fill_color = COLUMN_TO_COLOR[controls[row]]
        for col in range(tile_cols):
            r0 = row * STEP
            c0 = col * STEP
            if col == marker_tile_cols[row]:
                marker_tile = np.zeros((TILE_SIZE, TILE_SIZE), dtype=int)
                marker_tile[marker_rows[row], controls[row]] = GRAY
                inp[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE] = marker_tile
                continue

            mask = tile_masks[(row, col)]
            tile = np.zeros((TILE_SIZE, TILE_SIZE), dtype=int)
            tile[mask] = CYAN
            inp[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE] = tile

            out_tile = np.full((TILE_SIZE, TILE_SIZE), fill_color, dtype=int)
            out_tile[mask] = BG
            expected[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE] = out_tile

    return inp, expected


def _transform_fixed_color(grid: np.ndarray, fill_color: int = 2) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    row_spans, col_spans = _parse_tile_grid(g)
    out = np.zeros_like(g)
    for r0, r1 in row_spans:
        row_tiles = [g[r0:r1, c0:c1] for c0, c1 in col_spans]
        marker_col_idx, _marker_r, _marker_c = _row_marker_info(row_tiles)
        for col_idx, (c0, c1) in enumerate(col_spans):
            if col_idx == marker_col_idx:
                continue
            out[r0:r1, c0:c1] = _invert_tile(g[r0:r1, c0:c1], fill_color)
    return out


def _transform_marker_row_mapping(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    row_spans, col_spans = _parse_tile_grid(g)
    out = np.zeros_like(g)
    for r0, r1 in row_spans:
        row_tiles = [g[r0:r1, c0:c1] for c0, c1 in col_spans]
        marker_col_idx, marker_r, _marker_c = _row_marker_info(row_tiles)
        fill_color = COLUMN_TO_COLOR[marker_r]
        for col_idx, (c0, c1) in enumerate(col_spans):
            if col_idx == marker_col_idx:
                continue
            out[r0:r1, c0:c1] = _invert_tile(g[r0:r1, c0:c1], fill_color)
    return out


def _transform_marker_tile_column_mapping(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    row_spans, col_spans = _parse_tile_grid(g)
    out = np.zeros_like(g)
    for r0, r1 in row_spans:
        row_tiles = [g[r0:r1, c0:c1] for c0, c1 in col_spans]
        marker_col_idx, _marker_r, _marker_c = _row_marker_info(row_tiles)
        fill_color = COLUMN_TO_COLOR[marker_col_idx % TILE_SIZE]
        for col_idx, (c0, c1) in enumerate(col_spans):
            if col_idx == marker_col_idx:
                continue
            out[r0:r1, c0:c1] = _invert_tile(g[r0:r1, c0:c1], fill_color)
    return out


def _transform_recolor_without_inversion(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    row_spans, col_spans = _parse_tile_grid(g)
    out = np.zeros_like(g)
    for r0, r1 in row_spans:
        row_tiles = [g[r0:r1, c0:c1] for c0, c1 in col_spans]
        marker_col_idx, _marker_r, marker_c = _row_marker_info(row_tiles)
        fill_color = COLUMN_TO_COLOR[marker_c]
        for col_idx, (c0, c1) in enumerate(col_spans):
            if col_idx == marker_col_idx:
                continue
            tile = g[r0:r1, c0:c1]
            colors = set(map(int, np.unique(tile)))
            assert colors <= {BG, CYAN}
            recolored = np.zeros_like(tile)
            recolored[tile == CYAN] = fill_color
            out[r0:r1, c0:c1] = recolored
    return out


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    out_colors = set(map(int, np.unique(expected)))
    if not {2, 3, 4}.issubset(out_colors):
        return False

    try:
        row_spans, col_spans = _parse_tile_grid(inp)
    except Exception:
        return False

    code_to_marker_rows: dict[int, set[int]] = defaultdict(set)
    shared_patterns: dict[tuple[int, ...], set[int]] = defaultdict(set)
    for row_idx, (r0, r1) in enumerate(row_spans):
        row_tiles = [inp[r0:r1, c0:c1] for c0, c1 in col_spans]
        marker_col_idx, marker_r, marker_c = _row_marker_info(row_tiles)
        code_to_marker_rows[marker_c].add(marker_r)
        for col_idx, tile in enumerate(row_tiles):
            if col_idx == marker_col_idx:
                continue
            shared_patterns[_mask_key(tile == CYAN)].add(marker_c)

        if sum(col_idx != marker_col_idx for col_idx in range(len(col_spans))) < 2:
            return False

    if not any(len(rows) >= 2 for rows in code_to_marker_rows.values()):
        return False
    if not any(len(codes) >= 2 for codes in shared_patterns.values()):
        return False

    wrong_transforms = (
        _transform_fixed_color,
        _transform_marker_row_mapping,
        _transform_marker_tile_column_mapping,
        _transform_recolor_without_inversion,
    )
    for transform in wrong_transforms:
        try:
            if np.array_equal(transform(inp), expected):
                return False
        except Exception:
            continue

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp, expected = _build_case()
        if _semantic_checks(inp, expected):
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 4")
