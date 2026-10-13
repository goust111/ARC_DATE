from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
YELLOW = 4

GEN_ATTEMPTS = 500
TILE_ROW_RANGE = (2, 4)
TILE_COL_RANGE = (2, 4)
TILE_H_RANGE = (3, 5)
TILE_W_RANGE = (3, 5)
MIN_FAKE_DOTS = 2
MAX_FAKE_DOTS = 6


def _has_horizontal_pair(values: np.ndarray) -> bool:
    return bool(np.any((values[:-1] == BLUE) & (values[1:] == BLUE)))


def _has_vertical_pair(values: np.ndarray) -> bool:
    return bool(np.any((values[:-1] == BLUE) & (values[1:] == BLUE)))


def _segments(length: int, dividers: list[int]) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start = 0
    for divider in dividers + [length]:
        assert start < divider, "separator lines must leave non-empty tile spans"
        segments.append((start, divider))
        start = divider + 1
    assert start == length + 1
    return segments


def _detect_true_rows(grid: np.ndarray) -> list[int]:
    return [r for r in range(grid.shape[0]) if _has_horizontal_pair(grid[r, :])]


def _detect_true_cols(grid: np.ndarray) -> list[int]:
    return [c for c in range(grid.shape[1]) if _has_vertical_pair(grid[:, c])]


def _tile_counts(grid: np.ndarray, row_segments, col_segments) -> list[int]:
    counts: list[int] = []
    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            counts.append(int(np.count_nonzero(grid[r0:r1, c0:c1] == YELLOW)))
    return counts


def _render_output(
    shape: tuple[int, int],
    true_rows: list[int],
    true_cols: list[int],
    row_segments: list[tuple[int, int]],
    col_segments: list[tuple[int, int]],
    tiles: list[np.ndarray],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    out[true_rows, :] = BLUE
    out[:, true_cols] = BLUE
    idx = 0
    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            out[r0:r1, c0:c1] = tiles[idx]
            idx += 1
    return out


def _clean_tile(tile: np.ndarray) -> np.ndarray:
    return np.where(tile == YELLOW, YELLOW, BLACK)


def main(grid: np.ndarray) -> np.ndarray:
    """Identify the true broken blue separators, sort the tiles by yellow count, and restore the full grid lines.

    Input structure:
    - A black canvas contains several internal blue separator rows and columns, but each true separator is broken.
    - A row is a true separator iff it contains a horizontally adjacent blue pair; a column is a true separator iff it contains a vertically adjacent blue pair.
    - Inside the rectangular tile cells there are yellow pixels, while extra isolated blue dots may appear on false rows and false columns as distractors.

    Rule:
    - Detect the true separator rows and columns using only the adjacent-pair criterion.
    - Use those separators to partition the canvas into equal-sized tiles, count the yellow pixels in each tile, and sort the whole tile contents by ascending yellow count.
    - Place the sorted tiles back in top-left to bottom-right row-major order, draw every true separator row and column as a complete blue line, and remove all isolated fake blue dots.

    Key insight:
    - Blue pixels inside tile interiors are not tile content: only adjacent blue pairs reveal the real separator lattice, and only that lattice defines which yellow patterns must be sorted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    true_rows = _detect_true_rows(g)
    true_cols = _detect_true_cols(g)
    assert true_rows and true_cols, "expected at least one true separator row and column"

    row_segments = _segments(g.shape[0], true_rows)
    col_segments = _segments(g.shape[1], true_cols)
    assert len(row_segments) >= 2 and len(col_segments) >= 2, "expected a tile matrix"

    row_heights = {r1 - r0 for r0, r1 in row_segments}
    col_widths = {c1 - c0 for c0, c1 in col_segments}
    assert len(row_heights) == 1 and len(col_widths) == 1, "expected equal-sized tiles"

    tiles: list[tuple[int, np.ndarray]] = []
    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            tile = _clean_tile(g[r0:r1, c0:c1])
            count = int(np.count_nonzero(tile == YELLOW))
            tiles.append((count, tile))

    counts = [count for count, _tile in tiles]
    assert len(set(counts)) == len(counts), "yellow counts must be unique"

    ordered_tiles = [tile for _count, tile in sorted(tiles, key=lambda item: item[0])]
    return _render_output(g.shape, true_rows, true_cols, row_segments, col_segments, ordered_tiles)


def _choose_geometry() -> tuple[int, int, int, int]:
    tile_rows = random.randint(*TILE_ROW_RANGE)
    tile_cols = random.randint(*TILE_COL_RANGE)
    num_tiles = tile_rows * tile_cols

    choices: list[tuple[int, int]] = []
    for tile_h in range(TILE_H_RANGE[0], TILE_H_RANGE[1] + 1):
        for tile_w in range(TILE_W_RANGE[0], TILE_W_RANGE[1] + 1):
            height = tile_rows * tile_h + (tile_rows - 1)
            width = tile_cols * tile_w + (tile_cols - 1)
            if height <= 30 and width <= 30 and tile_h * tile_w + 1 >= num_tiles:
                choices.append((tile_h, tile_w))
    assert choices, "at least one geometry must fit the ARC bounds"
    tile_h, tile_w = random.choice(choices)
    return tile_rows, tile_cols, tile_h, tile_w


def _separator_positions(tile_count: int, tile_size: int) -> list[int]:
    return [tile_size + i * (tile_size + 1) for i in range(tile_count - 1)]


def _make_tile(tile_h: int, tile_w: int, yellow_count: int) -> np.ndarray:
    tile = np.zeros((tile_h, tile_w), dtype=int)
    if yellow_count:
        cells = random.sample(range(tile_h * tile_w), yellow_count)
        for idx in cells:
            tile[idx // tile_w, idx % tile_w] = YELLOW
    return tile


def _non_identity_permutation(size: int) -> list[int]:
    order = list(range(size))
    random.shuffle(order)
    if order == list(range(size)):
        i = random.randrange(size - 1)
        order[i], order[i + 1] = order[i + 1], order[i]
    return order


def _pick_pair_inside_segment(segments: list[tuple[int, int]]) -> tuple[int, int]:
    r0, r1 = random.choice(segments)
    start = random.randint(r0, r1 - 2)
    return start, start + 1


def _sample_marked_row(width: int, col_segments: list[tuple[int, int]]) -> list[int]:
    a, b = _pick_pair_inside_segment(col_segments)
    target = random.randint(max(3, width // 3), max(3, (2 * width) // 3))
    cols = {a, b}
    rest = list(range(width))
    random.shuffle(rest)
    for c in rest:
        if len(cols) >= target:
            break
        cols.add(c)
    if len(cols) == width:
        removable = [c for c in cols if c not in (a, b)]
        cols.remove(random.choice(removable))
    return sorted(cols)


def _sample_marked_col(height: int, row_segments: list[tuple[int, int]]) -> list[int]:
    a, b = _pick_pair_inside_segment(row_segments)
    target = random.randint(max(3, height // 3), max(3, (2 * height) // 3))
    rows = {a, b}
    rest = list(range(height))
    random.shuffle(rest)
    for r in rest:
        if len(rows) >= target:
            break
        rows.add(r)
    if len(rows) == height:
        removable = [r for r in rows if r not in (a, b)]
        rows.remove(random.choice(removable))
    return sorted(rows)


def _safe_false_indices(length: int, true_indices: list[int]) -> list[int]:
    return [idx for idx in range(length) if all(abs(idx - mark) > 1 for mark in true_indices)]


def _place_fake_dots(grid: np.ndarray, true_rows: list[int], true_cols: list[int]) -> int:
    safe_rows = _safe_false_indices(grid.shape[0], true_rows)
    safe_cols = _safe_false_indices(grid.shape[1], true_cols)
    candidates = [
        (r, c)
        for r in safe_rows
        for c in safe_cols
        if grid[r, c] == BLACK
    ]
    random.shuffle(candidates)
    if len(candidates) < MIN_FAKE_DOTS:
        return 0

    chosen: list[tuple[int, int]] = []
    target = random.randint(MIN_FAKE_DOTS, min(MAX_FAKE_DOTS, len(candidates)))
    for r, c in candidates:
        if any((r == rr and abs(c - cc) == 1) or (c == cc and abs(r - rr) == 1) for rr, cc in chosen):
            continue
        chosen.append((r, c))
        if len(chosen) == target:
            break
    if len(chosen) < MIN_FAKE_DOTS:
        return 0

    for r, c in chosen:
        grid[r, c] = BLUE
    return len(chosen)


def _build_input_grid(
    height: int,
    width: int,
    true_rows: list[int],
    true_cols: list[int],
    row_segments: list[tuple[int, int]],
    col_segments: list[tuple[int, int]],
    tiles: list[np.ndarray],
) -> tuple[np.ndarray, int]:
    inp = _render_output((height, width), [], [], row_segments, col_segments, tiles)

    line_mask = np.zeros((height, width), dtype=bool)
    for r in true_rows:
        line_mask[r, _sample_marked_row(width, col_segments)] = True
    for c in true_cols:
        line_mask[_sample_marked_col(height, row_segments), c] = True
    inp[line_mask] = BLUE

    fake_dot_count = _place_fake_dots(inp, true_rows, true_cols)
    return inp, fake_dot_count


def _validate_instance(
    inp: np.ndarray,
    expected: np.ndarray,
    true_rows: list[int],
    true_cols: list[int],
    row_segments: list[tuple[int, int]],
    col_segments: list[tuple[int, int]],
    order: list[int],
    fake_dot_count: int,
) -> bool:
    if fake_dot_count < MIN_FAKE_DOTS:
        return False
    if order == list(range(len(order))):
        return False
    if np.array_equal(inp, expected):
        return False

    if _detect_true_rows(inp) != true_rows or _detect_true_cols(inp) != true_cols:
        return False

    true_line_mask = np.zeros(inp.shape, dtype=bool)
    true_line_mask[true_rows, :] = True
    true_line_mask[:, true_cols] = True
    if int(np.count_nonzero((inp == BLUE) & ~true_line_mask)) != fake_dot_count:
        return False

    if any(np.all(inp[r, :] == BLUE) for r in true_rows):
        return False
    if any(np.all(inp[:, c] == BLUE) for c in true_cols):
        return False
    if not all(np.all(expected[r, :] == BLUE) for r in true_rows):
        return False
    if not all(np.all(expected[:, c] == BLUE) for c in true_cols):
        return False
    if np.any(expected[(inp == BLUE) & ~true_line_mask] == BLUE):
        return False

    output_counts = _tile_counts(expected, row_segments, col_segments)
    if output_counts != sorted(output_counts):
        return False
    if len(set(output_counts)) != len(output_counts):
        return False

    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _generate_attempt() -> np.ndarray | None:
    tile_rows, tile_cols, tile_h, tile_w = _choose_geometry()
    height = tile_rows * tile_h + (tile_rows - 1)
    width = tile_cols * tile_w + (tile_cols - 1)
    true_rows = _separator_positions(tile_rows, tile_h)
    true_cols = _separator_positions(tile_cols, tile_w)
    row_segments = _segments(height, true_rows)
    col_segments = _segments(width, true_cols)

    num_tiles = tile_rows * tile_cols
    counts = random.sample(range(tile_h * tile_w + 1), num_tiles)
    sorted_tiles = [_make_tile(tile_h, tile_w, count) for count in sorted(counts)]

    order = _non_identity_permutation(num_tiles)
    shuffled_tiles = [sorted_tiles[idx].copy() for idx in order]
    expected = _render_output((height, width), true_rows, true_cols, row_segments, col_segments, sorted_tiles)
    inp, fake_dot_count = _build_input_grid(
        height, width, true_rows, true_cols, row_segments, col_segments, shuffled_tiles
    )

    if not _validate_instance(
        inp, expected, true_rows, true_cols, row_segments, col_segments, order, fake_dot_count
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
