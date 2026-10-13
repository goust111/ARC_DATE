import random
from typing import Optional

import numpy as np


UP_ARROW = np.array(
    [
        [1, 1, 1],
        [0, 1, 0],
        [0, 1, 0],
    ],
    dtype=bool,
)
ARROW_MASKS = {
    "up": UP_ARROW,
    "right": np.rot90(UP_ARROW, 3),
    "down": np.rot90(UP_ARROW, 2),
    "left": np.rot90(UP_ARROW, 1),
}
ARROW_TO_ROT_K = {
    "up": 0,
    "right": 3,
    "down": 2,
    "left": 1,
}

GEN_BG = 0
GEN_SEPARATOR = 5
GEN_ARROW_COLOR = 9
GEN_PATTERN_COLORS = (1, 2, 3, 4, 6, 7, 8)
GEN_TILE_ROW_RANGE = (2, 4)
GEN_TILE_COL_RANGE = (2, 4)
GEN_TILE_HEIGHT_RANGE = (5, 8)
GEN_TILE_WIDTH_RANGE = (5, 8)
GEN_MAX_ATTEMPTS = 400


def _mode(values: np.ndarray) -> int:
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[np.argmax(counts)])


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    seps = sorted(int(idx) for idx in separators)
    assert seps, "expected at least one separator line"
    assert len(set(seps)) == len(seps), "separator lines must be unique"

    spans: list[tuple[int, int]] = []
    start = 0
    for idx in seps + [length]:
        assert start < idx, "separator lines must lie strictly inside the grid"
        spans.append((start, idx))
        start = idx + 1
    assert start == length + 1
    return spans


def _separator_structure(grid: np.ndarray) -> tuple[int, int, int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates = []
    for color in map(int, np.unique(g)):
        rows = [r for r in range(h) if np.all(g[r, :] == color)]
        cols = [c for c in range(w) if np.all(g[:, c] == color)]
        if rows and cols:
            candidates.append((color, rows, cols))

    assert len(candidates) == 1, "expected exactly one separator color"
    separator, sep_rows, sep_cols = candidates[0]
    row_spans = _spans(h, sep_rows)
    col_spans = _spans(w, sep_cols)

    tile_heights = {r1 - r0 for r0, r1 in row_spans}
    tile_widths = {c1 - c0 for c0, c1 in col_spans}
    assert len(tile_heights) == 1 and len(tile_widths) == 1, "tiles must be equal-sized"
    tile_h = tile_heights.pop()
    tile_w = tile_widths.pop()
    assert tile_h >= 3 and tile_w >= 3
    return separator, tile_h, tile_w, row_spans, col_spans


def _crop_non_bg(tile: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    coords = np.argwhere(tile != bg)
    assert len(coords) > 0, "expected non-background pixels"
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1) + 1
    c1 = int(c1) + 1
    return tile[r0:r1, c0:c1].copy(), (r0, c0, r1, c1)


def _arrow_direction(tile: np.ndarray, bg: int) -> Optional[str]:
    coords = np.argwhere(tile != bg)
    if len(coords) != 5:
        return None

    colors = {int(tile[r, c]) for r, c in coords}
    if len(colors) != 1:
        return None

    patch, _ = _crop_non_bg(tile, bg)
    if patch.shape != (3, 3):
        return None

    mask = patch != bg
    for direction, template in ARROW_MASKS.items():
        if np.array_equal(mask, template):
            return direction
    return None


def _signature(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return grid.shape, grid.tobytes()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the unique source pattern into each arrow-marked tile, rotating the
    pattern according to arrow direction while preserving the separator grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    separator, tile_h, tile_w, row_spans, col_spans = _separator_structure(g)
    bg = _mode(g[g != separator])
    assert bg != separator

    source_patch = None
    arrows: list[tuple[int, int, str]] = []
    source_count = 0

    for row_idx, (r0, r1) in enumerate(row_spans):
        for col_idx, (c0, c1) in enumerate(col_spans):
            tile = g[r0:r1, c0:c1]
            values = set(map(int, np.unique(tile)))
            assert separator not in values, "separator color cannot appear inside a tile"

            if values <= {bg}:
                continue

            direction = _arrow_direction(tile, bg)
            if direction is not None:
                arrows.append((row_idx, col_idx, direction))
                continue

            source_count += 1
            assert source_count == 1, "expected exactly one source tile"
            source_patch, _ = _crop_non_bg(tile, bg)

    assert source_patch is not None, "source tile not found"
    assert arrows, "expected at least one arrow tile"

    out = g.copy()
    for row_idx, col_idx, direction in arrows:
        rotated = np.rot90(source_patch, ARROW_TO_ROT_K[direction])
        rh, rw = rotated.shape
        top = (tile_h - rh) // 2
        left = (tile_w - rw) // 2
        assert top >= 0 and left >= 0, "rotated source pattern must fit in the tile"

        tile = np.full((tile_h, tile_w), bg, dtype=int)
        tile[top : top + rh, left : left + rw] = rotated

        r0, r1 = row_spans[row_idx]
        c0, c1 = col_spans[col_idx]
        out[r0:r1, c0:c1] = tile

    return out


def _neighbors(cell: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = set(_neighbors(start, h, w))

    while len(cells) < area:
        if not frontier:
            frontier = {
                nxt
                for cell in list(cells)
                for nxt in _neighbors(cell, h, w)
                if nxt not in cells
            }
        choice = random.choice(tuple(frontier))
        frontier.remove(choice)
        cells.add(choice)
        for nxt in _neighbors(choice, h, w):
            if nxt not in cells:
                frontier.add(nxt)

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_source_patch(tile_h: int, tile_w: int) -> Optional[np.ndarray]:
    span_cap = min(tile_h, tile_w) - 1
    if span_cap < 2:
        return None

    patch_h = random.randint(2, span_cap)
    patch_w = random.randint(2, span_cap)
    max_area = patch_h * patch_w - 1
    min_area = max(4, (patch_h * patch_w + 1) // 2)
    if min_area > max_area:
        return None

    area = random.randint(min_area, max_area)
    mask = _random_connected_mask(patch_h, patch_w, area)
    patch = np.full((patch_h, patch_w), GEN_BG, dtype=int)

    color_count = random.randint(2, min(4, len(GEN_PATTERN_COLORS), area))
    colors = random.sample(GEN_PATTERN_COLORS, color_count)
    cells = list(zip(*np.where(mask)))
    random.shuffle(cells)

    for idx, (r, c) in enumerate(cells):
        if idx < color_count:
            patch[r, c] = colors[idx]
        else:
            patch[r, c] = random.choice(colors)

    rotations = {_signature(np.rot90(patch, k)) for k in range(4)}
    if len(rotations) != 4:
        return None
    return patch


def _place_centered_arrow(tile_h: int, tile_w: int, direction: str) -> np.ndarray:
    tile = np.full((tile_h, tile_w), GEN_BG, dtype=int)
    mask = ARROW_MASKS[direction]
    top = (tile_h - 3) // 2
    left = (tile_w - 3) // 2
    tile[top : top + 3, left : left + 3][mask] = GEN_ARROW_COLOR
    return tile


def _gen_attempt() -> Optional[np.ndarray]:
    tile_rows = random.randint(*GEN_TILE_ROW_RANGE)
    tile_cols = random.randint(*GEN_TILE_COL_RANGE)
    tile_h = random.randint(*GEN_TILE_HEIGHT_RANGE)
    tile_w = random.randint(*GEN_TILE_WIDTH_RANGE)

    total_tiles = tile_rows * tile_cols
    if total_tiles < 3:
        return None

    source_patch = _make_source_patch(tile_h, tile_w)
    if source_patch is None:
        return None
    if source_patch.shape[0] > min(tile_h, tile_w) or source_patch.shape[1] > min(tile_h, tile_w):
        return None

    grid_h = tile_rows * tile_h + (tile_rows - 1)
    grid_w = tile_cols * tile_w + (tile_cols - 1)
    grid = np.full((grid_h, grid_w), GEN_BG, dtype=int)

    for r in range(tile_rows - 1):
        grid[(r + 1) * tile_h + r, :] = GEN_SEPARATOR
    for c in range(tile_cols - 1):
        grid[:, (c + 1) * tile_w + c] = GEN_SEPARATOR

    positions = [(r, c) for r in range(tile_rows) for c in range(tile_cols)]
    source_pos = random.choice(positions)
    remaining = [pos for pos in positions if pos != source_pos]

    min_arrows = 2 if len(remaining) >= 2 else 1
    max_arrows = len(remaining)
    if max_arrows < min_arrows:
        return None
    arrow_count = random.randint(min_arrows, max_arrows)
    arrow_positions = random.sample(remaining, arrow_count)

    distinct_dirs = min(4, arrow_count)
    directions = random.sample(list(ARROW_MASKS), distinct_dirs)
    while len(directions) < arrow_count:
        directions.append(random.choice(tuple(ARROW_MASKS)))
    random.shuffle(directions)

    tile_top = source_pos[0] * (tile_h + 1)
    tile_left = source_pos[1] * (tile_w + 1)
    sr = tile_top + random.randint(0, tile_h - source_patch.shape[0])
    sc = tile_left + random.randint(0, tile_w - source_patch.shape[1])
    grid[sr : sr + source_patch.shape[0], sc : sc + source_patch.shape[1]] = source_patch

    for (tr, tc), direction in zip(arrow_positions, directions):
        r0 = tr * (tile_h + 1)
        c0 = tc * (tile_w + 1)
        grid[r0 : r0 + tile_h, c0 : c0 + tile_w] = _place_centered_arrow(tile_h, tile_w, direction)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
