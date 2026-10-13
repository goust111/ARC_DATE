from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


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
        assert start < sep, "separator lines must leave non-empty cells"
        spans.append((start, sep - 1))
        start = sep + 1
    assert start < length, "separator lines must leave a final non-empty cell"
    spans.append((start, length - 1))
    return spans


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_source_patch(tile: np.ndarray, bg: int) -> tuple[np.ndarray, int] | None:
    mask = tile != bg
    box = _bbox(mask)
    if box is None:
        return None

    r0, c0, r1, c1 = box
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None

    crop = tile[r0 : r1 + 1, c0 : c1 + 1]
    corner_color = int(crop[0, 0])
    if corner_color == bg:
        return None
    if not (
        int(crop[0, -1]) == corner_color
        and int(crop[-1, 0]) == corner_color
        and int(crop[-1, -1]) == corner_color
    ):
        return None

    border = np.zeros_like(crop, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    corners = np.zeros_like(crop, dtype=bool)
    corners[0, 0] = True
    corners[0, -1] = True
    corners[-1, 0] = True
    corners[-1, -1] = True
    if np.any(crop[border & ~corners] != bg):
        return None

    interior = crop[1:-1, 1:-1]
    interior_colors = sorted({int(v) for v in np.unique(interior) if int(v) != bg})
    if len(interior_colors) != 1 or interior_colors[0] == corner_color:
        return None

    patch = interior.copy()
    patch[patch != bg] = corner_color
    return patch, corner_color


def _paste_opaque(tile: np.ndarray, patch: np.ndarray, anchor_r: int, anchor_c: int) -> None:
    h, w = tile.shape
    ph, pw = patch.shape
    rr0 = max(0, anchor_r)
    cc0 = max(0, anchor_c)
    rr1 = min(h, anchor_r + ph)
    cc1 = min(w, anchor_c + pw)
    if rr0 >= rr1 or cc0 >= cc1:
        return
    src_r0 = rr0 - anchor_r
    src_c0 = cc0 - anchor_c
    tile[rr0:rr1, cc0:cc1] = patch[src_r0 : src_r0 + (rr1 - rr0), src_c0 : src_c0 + (cc1 - cc0)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the corner-framed source motif and stamp it into matching marker cells using the marker as the top-left anchor.

    Input structure:
    - Full same-color separator rows and columns partition the grid into equal rectangular cells.
    - One corner cell contains four matching corner-marker pixels that frame a different-colored interior pattern.
    - Other cells are empty or contain exactly one colored marker pixel.

    Rule:
    - Extract the framed interior pattern from the source cell, preserving its background but recoloring every non-background interior pixel to the corner-marker color.
    - Keep the separator grid and the source cell unchanged.
    - In every non-source cell whose lone marker pixel has the same color as the corner markers, stamp the recolored pattern into that cell with the marker pixel aligned to the pattern's top-left pixel; cells with other marker colors stay unchanged.

    Key insight:
    - The source corner color plays three roles at once: it identifies the crop rectangle, gives the motif's output color, and selects which marker cells receive the stamp.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    separator, sep_rows, sep_cols = _separator_lines(g)
    row_spans = _spans(g.shape[0], sep_rows)
    col_spans = _spans(g.shape[1], sep_cols)

    cell_heights = {r1 - r0 + 1 for r0, r1 in row_spans}
    cell_widths = {c1 - c0 + 1 for c0, c1 in col_spans}
    assert len(cell_heights) == 1 and len(cell_widths) == 1, "expected equal-sized cells"

    bg = _get_bg(g[g != separator])
    assert bg != separator, "background must differ from separator"

    source_patch: np.ndarray | None = None
    source_color: int | None = None
    source_pos: tuple[int, int] | None = None
    marker_cells: list[tuple[tuple[int, int], tuple[int, int], int]] = []

    for tile_r, (r0, r1) in enumerate(row_spans):
        for tile_c, (c0, c1) in enumerate(col_spans):
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            assert separator not in set(map(int, np.unique(tile))), "separator cannot appear inside cells"
            non_bg = np.argwhere(tile != bg)
            if len(non_bg) == 0:
                continue

            extracted = _extract_source_patch(tile, bg)
            if extracted is not None:
                assert source_patch is None, "expected exactly one source cell"
                source_patch, source_color = extracted
                source_pos = (tile_r, tile_c)
                continue

            assert len(non_bg) == 1, "non-source cells must be empty or have one marker"
            mr, mc = map(int, non_bg[0])
            marker_cells.append(((tile_r, tile_c), (mr, mc), int(tile[mr, mc])))

    assert source_patch is not None and source_color is not None and source_pos is not None

    out = g.copy()
    for (tile_r, tile_c), (mr, mc), color in marker_cells:
        if color != source_color or (tile_r, tile_c) == source_pos:
            continue
        r0, r1 = row_spans[tile_r]
        c0, c1 = col_spans[tile_c]
        tile_out = out[r0 : r1 + 1, c0 : c1 + 1]
        _paste_opaque(tile_out, source_patch, mr, mc)

    return out


COLOR_POOL = list(range(10))
GEN_RETRIES = 400
TILE_ROWS_RANGE = (3, 4)
TILE_COLS_RANGE = (3, 4)
MIN_CELL_SIZE = 6
MAX_MARKERS = 7


def _random_pattern(height: int, width: int, bg: int, color: int) -> np.ndarray:
    cell_count = height * width
    target = random.randint(max(1, cell_count // 4), max(1, (3 * cell_count) // 5))
    patch = np.full((height, width), bg, dtype=int)
    cells = [(r, c) for r in range(height) for c in range(width)]
    for r, c in random.sample(cells, target):
        patch[r, c] = color
    return patch


def _build_source_tile(
    cell_h: int, cell_w: int, bg: int, corner_color: int, pattern_color: int
) -> tuple[np.ndarray, np.ndarray]:
    pattern_h = random.randint(2, cell_h - 2)
    pattern_w = random.randint(2, cell_w - 2)
    patch = _random_pattern(pattern_h, pattern_w, bg, pattern_color)

    framed_h = pattern_h + 2
    framed_w = pattern_w + 2
    top = random.randint(0, cell_h - framed_h)
    left = random.randint(0, cell_w - framed_w)

    tile = np.full((cell_h, cell_w), bg, dtype=int)
    tile[top, left] = corner_color
    tile[top, left + framed_w - 1] = corner_color
    tile[top + framed_h - 1, left] = corner_color
    tile[top + framed_h - 1, left + framed_w - 1] = corner_color
    tile[top + 1 : top + 1 + pattern_h, left + 1 : left + 1 + pattern_w] = patch

    stamp = patch.copy()
    stamp[stamp != bg] = corner_color
    return tile, stamp


def _empty_grid(tile_rows: int, tile_cols: int, cell_h: int, cell_w: int, bg: int, separator: int) -> np.ndarray:
    height = tile_rows * cell_h + (tile_rows - 1)
    width = tile_cols * cell_w + (tile_cols - 1)
    grid = np.full((height, width), bg, dtype=int)
    for r in range(1, tile_rows):
        grid[r * cell_h + (r - 1), :] = separator
    for c in range(1, tile_cols):
        grid[:, c * cell_w + (c - 1)] = separator
    return grid


def _cell_origin(tile_r: int, tile_c: int, cell_h: int, cell_w: int) -> tuple[int, int]:
    return tile_r * (cell_h + 1), tile_c * (cell_w + 1)


def _corner_positions(tile_rows: int, tile_cols: int) -> list[tuple[int, int]]:
    return [(0, 0), (0, tile_cols - 1), (tile_rows - 1, 0), (tile_rows - 1, tile_cols - 1)]


def _expected_output(
    inp: np.ndarray,
    cell_rows: list[tuple[int, int]],
    cell_cols: list[tuple[int, int]],
    source_pos: tuple[int, int],
    source_color: int,
    stamp: np.ndarray,
    bg: int,
) -> np.ndarray:
    out = inp.copy()
    for tile_r, (r0, r1) in enumerate(cell_rows):
        for tile_c, (c0, c1) in enumerate(cell_cols):
            if (tile_r, tile_c) == source_pos:
                continue
            tile = inp[r0 : r1 + 1, c0 : c1 + 1]
            pts = np.argwhere(tile != bg)
            if len(pts) != 1:
                continue
            mr, mc = map(int, pts[0])
            if int(tile[mr, mc]) != source_color:
                continue
            _paste_opaque(out[r0 : r1 + 1, c0 : c1 + 1], stamp, mr, mc)
    return out


def _gen_attempt() -> np.ndarray | None:
    tile_rows = random.randint(*TILE_ROWS_RANGE)
    tile_cols = random.randint(*TILE_COLS_RANGE)
    max_cell_h = (30 - (tile_rows - 1)) // tile_rows
    max_cell_w = (30 - (tile_cols - 1)) // tile_cols
    if max_cell_h < MIN_CELL_SIZE or max_cell_w < MIN_CELL_SIZE:
        return None

    cell_h = random.randint(MIN_CELL_SIZE, max_cell_h)
    cell_w = random.randint(MIN_CELL_SIZE, max_cell_w)
    separator, bg, corner_color, pattern_color = random.sample(COLOR_POOL, 4)

    grid = _empty_grid(tile_rows, tile_cols, cell_h, cell_w, bg, separator)
    row_spans = _spans(grid.shape[0], [r * cell_h + (r - 1) for r in range(1, tile_rows)])
    col_spans = _spans(grid.shape[1], [c * cell_w + (c - 1) for c in range(1, tile_cols)])

    source_pos = random.choice(_corner_positions(tile_rows, tile_cols))
    source_tile, stamp = _build_source_tile(cell_h, cell_w, bg, corner_color, pattern_color)
    sr0, sc0 = _cell_origin(source_pos[0], source_pos[1], cell_h, cell_w)
    grid[sr0 : sr0 + cell_h, sc0 : sc0 + cell_w] = source_tile

    other_tiles = [(r, c) for r in range(tile_rows) for c in range(tile_cols) if (r, c) != source_pos]
    marker_total = random.randint(2, min(len(other_tiles), MAX_MARKERS))
    same_count = random.randint(1, marker_total - 1)
    diff_count = marker_total - same_count
    chosen_tiles = random.sample(other_tiles, marker_total)
    same_tiles = chosen_tiles[:same_count]
    diff_tiles = chosen_tiles[same_count:]

    distractor_choices = [c for c in COLOR_POOL if c not in {bg, separator, corner_color}]
    distractor_colors = random.sample(distractor_choices, random.randint(1, min(3, len(distractor_choices))))

    stamp_h, stamp_w = stamp.shape
    for tile_r, tile_c in same_tiles:
        r0, c0 = _cell_origin(tile_r, tile_c, cell_h, cell_w)
        mr = random.randint(0, cell_h - stamp_h)
        mc = random.randint(0, cell_w - stamp_w)
        grid[r0 + mr, c0 + mc] = corner_color

    for tile_r, tile_c in diff_tiles:
        r0, c0 = _cell_origin(tile_r, tile_c, cell_h, cell_w)
        mr = random.randint(0, cell_h - stamp_h)
        mc = random.randint(0, cell_w - stamp_w)
        grid[r0 + mr, c0 + mc] = random.choice(distractor_colors)

    try:
        actual = main(grid)
    except Exception:
        return None

    expected = _expected_output(grid, row_spans, col_spans, source_pos, corner_color, stamp, bg)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
