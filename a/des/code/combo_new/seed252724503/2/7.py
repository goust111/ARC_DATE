from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
SEPARATOR = 5
ARROW = 9
PATTERN_COLORS = (1, 2, 3, 4, 6, 7, 8)

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ARROW_TEMPLATES = {
    "up": np.array([[0, 1, 0], [1, 0, 1]], dtype=int),
    "down": np.array([[1, 0, 1], [0, 1, 0]], dtype=int),
    "left": np.array([[0, 1], [1, 0], [0, 1]], dtype=int),
    "right": np.array([[1, 0], [0, 1], [1, 0]], dtype=int),
}

CELL_SHAPES = [
    (tile_rows, tile_cols, cell_h, cell_w)
    for tile_rows in range(3, 6)
    for tile_cols in range(3, 6)
    for cell_h in range(4, 7)
    for cell_w in range(4, 7)
    if tile_rows * cell_h + (tile_rows - 1) <= 30 and tile_cols * cell_w + (tile_cols - 1) <= 30
]
MAX_GENERATION_ATTEMPTS = 400


def _dominant_color(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected a non-empty grid"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique dominant color"
    return int(ranked[0][0])


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for sep in separators:
        assert 0 < sep < length - 1, "separator rows and columns must be internal"
        assert start < sep, "separators must leave a non-empty cell span"
        spans.append((start, sep - 1))
        start = sep + 1
    assert start < length, "separators must leave a final non-empty cell span"
    spans.append((start, length - 1))
    return spans


def _separator_layout(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]]]] = []
    for color in np.unique(g):
        color = int(color)
        sep_rows = [r for r in range(h) if np.all(g[r, :] == color)]
        sep_cols = [c for c in range(w) if np.all(g[:, c] == color)]
        if not sep_rows or not sep_cols:
            continue
        if 0 in sep_rows or h - 1 in sep_rows or 0 in sep_cols or w - 1 in sep_cols:
            continue
        try:
            row_spans = _spans(h, sep_rows)
            col_spans = _spans(w, sep_cols)
        except AssertionError:
            continue
        row_sizes = {r1 - r0 + 1 for r0, r1 in row_spans}
        col_sizes = {c1 - c0 + 1 for c0, c1 in col_spans}
        if len(row_sizes) == 1 and len(col_sizes) == 1:
            candidates.append((color, row_spans, col_spans))
    assert len(candidates) == 1, "expected one separator color forming an equal-cell layout"
    return candidates[0]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _arrow_direction(tile: np.ndarray, bg: int) -> str | None:
    unique = {int(v) for v in np.unique(tile)}
    if not unique.issubset({bg, ARROW}):
        return None
    mask = tile == ARROW
    if int(mask.sum()) != 3:
        return None
    box = _bbox(mask)
    if box is None:
        return None
    r0, c0, r1, c1 = box
    local = mask[r0 : r1 + 1, c0 : c1 + 1].astype(int)
    for name, template in ARROW_TEMPLATES.items():
        if local.shape == template.shape and np.array_equal(local, template):
            return name
    return None


def _cell_slice(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    cell: tuple[int, int],
) -> np.ndarray:
    r0, r1 = row_spans[cell[0]]
    c0, c1 = col_spans[cell[1]]
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull each adjacent pattern into the arrow cell inside a separator-divided board.

    Input structure:
    - The background is black, and full grey separator rows and columns split the grid into equal rectangular cells.
    - Some cells contain a small non-background pattern; some cells are empty.
    - Arrow cells contain exactly three maroon pixels arranged as a small triangular arrow pointing up, down, left, or right.
    - Every arrow points to an orthogonally adjacent cell that contains the pattern it references.

    Rule:
    - For each arrow, read its pointing direction and locate the adjacent source cell in that direction.
    - Move the entire source-cell pattern into the arrow cell, preserving all pattern pixels at the same local coordinates inside the cell.
    - Clear the original source cell to background and remove the arrow.
    - Leave untouched pattern cells and all grey separator lines unchanged.

    Key insight:
    - The arrow marks where the pattern will end up, not where it should travel next: it pulls the neighboring pattern inward from the pointed cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    separator, row_spans, col_spans = _separator_layout(g)
    bg = _dominant_color(g[g != separator])
    assert bg != separator, "background and separator colors must differ"

    patterns: dict[tuple[int, int], np.ndarray] = {}
    arrows: dict[tuple[int, int], str] = {}
    empty: set[tuple[int, int]] = set()
    for rr in range(len(row_spans)):
        for cc in range(len(col_spans)):
            cell = (rr, cc)
            tile = _cell_slice(g, row_spans, col_spans, cell)
            direction = _arrow_direction(tile, bg)
            if direction is not None:
                arrows[cell] = direction
            elif np.all(tile == bg):
                empty.add(cell)
            else:
                patterns[cell] = tile.copy()

    moves: list[tuple[tuple[int, int], tuple[int, int], np.ndarray]] = []
    used_sources: set[tuple[int, int]] = set()
    for cell, direction in arrows.items():
        dr, dc = DIRS[direction]
        source = (cell[0] + dr, cell[1] + dc)
        assert 0 <= source[0] < len(row_spans) and 0 <= source[1] < len(col_spans), "arrow source must be in bounds"
        assert source in patterns, "every arrow must point to a pattern cell"
        assert source not in used_sources, "source cells must be unique"
        used_sources.add(source)
        moves.append((cell, source, patterns[source]))

    out = g.copy()
    for _dest, source, _tile in moves:
        source_view = _cell_slice(out, row_spans, col_spans, source)
        source_view[:] = bg
    for dest, _source, tile in moves:
        dest_view = _cell_slice(out, row_spans, col_spans, dest)
        dest_view[:] = tile
    return out


def _grow_connected_mask(h: int, w: int, area: int) -> np.ndarray | None:
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIRS.values():
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _is_asymmetric(mask: np.ndarray) -> bool:
    return not (
        np.array_equal(mask, np.flipud(mask))
        or np.array_equal(mask, np.fliplr(mask))
        or np.array_equal(mask, np.rot90(mask, 2))
    )


def _pattern_tile(cell_h: int, cell_w: int) -> np.ndarray | None:
    max_h = max(2, cell_h - random.randint(0, 1))
    max_w = max(2, cell_w - random.randint(0, 1))
    for _ in range(120):
        shape_h = random.randint(2, max_h)
        shape_w = random.randint(2, max_w)
        max_area = min(shape_h * shape_w - 1, max(4, (shape_h * shape_w * 2) // 3))
        if max_area < 4:
            continue
        area = random.randint(4, max_area)
        mask = _grow_connected_mask(shape_h, shape_w, area)
        if mask is None or not _is_asymmetric(mask):
            continue
        if np.all(mask) or np.all(mask.sum(axis=0) == 1) or np.all(mask.sum(axis=1) == 1):
            continue
        tile = np.full((cell_h, cell_w), BG, dtype=int)
        r_off = random.randint(0, cell_h - shape_h)
        c_off = random.randint(0, cell_w - shape_w)
        color = random.choice(PATTERN_COLORS)
        tile[r_off : r_off + shape_h, c_off : c_off + shape_w][mask] = color
        if _arrow_direction(tile, BG) is not None:
            continue
        return tile
    return None


def _arrow_tile(cell_h: int, cell_w: int, direction: str) -> np.ndarray:
    template = ARROW_TEMPLATES[direction]
    th, tw = template.shape
    r_off = random.randint(0, cell_h - th)
    c_off = random.randint(0, cell_w - tw)
    tile = np.full((cell_h, cell_w), BG, dtype=int)
    tile[r_off : r_off + th, c_off : c_off + tw][template.astype(bool)] = ARROW
    return tile


def _base_grid(tile_rows: int, tile_cols: int, cell_h: int, cell_w: int) -> np.ndarray:
    h = tile_rows * cell_h + (tile_rows - 1)
    w = tile_cols * cell_w + (tile_cols - 1)
    grid = np.full((h, w), BG, dtype=int)
    for r in range(1, tile_rows):
        grid[r * cell_h + (r - 1), :] = SEPARATOR
    for c in range(1, tile_cols):
        grid[:, c * cell_w + (c - 1)] = SEPARATOR
    return grid


def _cell_origin(cell: tuple[int, int], cell_h: int, cell_w: int) -> tuple[int, int]:
    return cell[0] * (cell_h + 1), cell[1] * (cell_w + 1)


def _paste_cell(grid: np.ndarray, cell: tuple[int, int], tile: np.ndarray, cell_h: int, cell_w: int) -> None:
    r0, c0 = _cell_origin(cell, cell_h, cell_w)
    grid[r0 : r0 + cell_h, c0 : c0 + cell_w] = tile


def _choose_pairs(tile_rows: int, tile_cols: int) -> list[tuple[tuple[int, int], tuple[int, int], str]] | None:
    total = tile_rows * tile_cols
    arrow_count = random.randint(1, min(4, total // 2))
    directions = random.sample(list(DIRS), arrow_count)
    reserved: set[tuple[int, int]] = set()
    pairs: list[tuple[tuple[int, int], tuple[int, int], str]] = []

    for direction in directions:
        dr, dc = DIRS[direction]
        options = []
        for r in range(tile_rows):
            for c in range(tile_cols):
                dest = (r, c)
                source = (r + dr, c + dc)
                if dest in reserved or source in reserved:
                    continue
                if not (0 <= source[0] < tile_rows and 0 <= source[1] < tile_cols):
                    continue
                options.append((dest, source, direction))
        if not options:
            return None
        pair = random.choice(options)
        pairs.append(pair)
        reserved.add(pair[0])
        reserved.add(pair[1])
    return pairs


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    tile_rows, tile_cols, cell_h, cell_w = random.choice(CELL_SHAPES)
    pairs = _choose_pairs(tile_rows, tile_cols)
    if pairs is None:
        return None

    reserved = {cell for pair in pairs for cell in pair[:2]}
    remaining = [(r, c) for r in range(tile_rows) for c in range(tile_cols) if (r, c) not in reserved]
    untouched: list[tuple[int, int]] = []
    if remaining and random.random() < 0.7:
        untouched_count = random.randint(1, min(4, len(remaining)))
        untouched = random.sample(remaining, untouched_count)

    pattern_tiles: dict[tuple[int, int], np.ndarray] = {}
    for dest, _source, _direction in pairs:
        tile = _pattern_tile(cell_h, cell_w)
        if tile is None:
            return None
        pattern_tiles[dest] = tile
    for cell in untouched:
        tile = _pattern_tile(cell_h, cell_w)
        if tile is None:
            return None
        pattern_tiles[cell] = tile

    out = _base_grid(tile_rows, tile_cols, cell_h, cell_w)
    for cell, tile in pattern_tiles.items():
        _paste_cell(out, cell, tile, cell_h, cell_w)

    inp = out.copy()
    for dest, source, direction in pairs:
        _paste_cell(inp, dest, _arrow_tile(cell_h, cell_w, direction), cell_h, cell_w)
        _paste_cell(inp, source, pattern_tiles[dest], cell_h, cell_w)

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _build_case()
        if candidate is None:
            continue
        inp, out = candidate
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, out):
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 7")
