from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE_CORNERS = ("tl", "tr", "bl", "br")
MIN_COLOR_COUNT = 3
MAX_COLOR_COUNT = 5
MIN_GRID_SIZE = 14
MAX_GRID_SIZE = 21
MAX_ATTEMPTS = 400


def _corner_bbox(shape: tuple[int, int], corner: str, size: int) -> tuple[int, int, int, int]:
    h, w = shape
    if corner == "tl":
        return 0, 0, size - 1, size - 1
    if corner == "tr":
        return 0, w - size, size - 1, w - 1
    if corner == "bl":
        return h - size, 0, h - 1, size - 1
    if corner == "br":
        return h - size, w - size, h - 1, w - 1
    raise ValueError(f"unknown corner {corner}")


def _extract_corner_block(grid: np.ndarray, corner: str, size: int) -> np.ndarray:
    r0, c0, r1, c1 = _corner_bbox(grid.shape, corner, size)
    block = grid[r0 : r1 + 1, c0 : c1 + 1]
    if corner == "tl":
        return block.copy()
    if corner == "tr":
        return np.fliplr(block).copy()
    if corner == "bl":
        return np.flipud(block).copy()
    return np.flipud(np.fliplr(block)).copy()


def _write_corner_block(grid: np.ndarray, corner: str, block: np.ndarray) -> None:
    size = int(block.shape[0])
    r0, c0, r1, c1 = _corner_bbox(grid.shape, corner, size)
    if corner == "tl":
        placed = block
    elif corner == "tr":
        placed = np.fliplr(block)
    elif corner == "bl":
        placed = np.flipud(block)
    else:
        placed = np.flipud(np.fliplr(block))
    grid[r0 : r1 + 1, c0 : c1 + 1] = placed


def _inner_moat_is_clear(grid: np.ndarray, corner: str, size: int) -> bool:
    h, w = grid.shape
    if corner == "tl":
        row = grid[size, :size] if size < h else None
        col = grid[:size, size] if size < w else None
    elif corner == "tr":
        row = grid[size, w - size : w] if size < h else None
        col = grid[:size, w - size - 1] if w - size - 1 >= 0 else None
    elif corner == "bl":
        row = grid[h - size - 1, :size] if h - size - 1 >= 0 else None
        col = grid[h - size : h, size] if size < w else None
    else:
        row = grid[h - size - 1, w - size : w] if h - size - 1 >= 0 else None
        col = grid[h - size : h, w - size - 1] if w - size - 1 >= 0 else None

    if row is not None and not bool(np.all(row == BG)):
        return False
    if col is not None and not bool(np.all(col == BG)):
        return False
    return True


def _looks_like_palette(block: np.ndarray) -> bool:
    size = int(block.shape[0])
    if block.shape[0] != block.shape[1] or not (4 <= size <= 6):
        return False
    if int(block[0, 0]) != BG:
        return False
    if bool(np.any(block[0, 1:] == BG)) or bool(np.any(block[1:, 0] == BG)):
        return False
    if bool(np.any(block[1:, 1:] == BG)):
        return False

    row_headers = [int(x) for x in block[1:, 0]]
    col_headers = [int(x) for x in block[0, 1:]]
    return len(set(row_headers)) == size - 1 and len(set(col_headers)) == size - 1 and set(row_headers) == set(col_headers)


def _detect_palette(grid: np.ndarray) -> tuple[str, np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[int, str, np.ndarray]] = []
    max_size = min(6, h, w)
    for size in range(max_size, 3, -1):
        for corner in PALETTE_CORNERS:
            block = _extract_corner_block(grid, corner, size)
            if _looks_like_palette(block) and _inner_moat_is_clear(grid, corner, size):
                candidates.append((size, corner, block))

    assert candidates, "failed to detect the corner palette"
    best_size = max(size for size, _corner, _block in candidates)
    best = [(corner, block) for size, corner, block in candidates if size == best_size]
    assert len(best) == 1, "expected a unique palette"
    return best[0]


def _palette_mask(shape: tuple[int, int], corner: str, size: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = _corner_bbox(shape, corner, size)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _parse_palette(block: np.ndarray) -> tuple[list[int], dict[frozenset[int], int]]:
    row_headers = [int(x) for x in block[1:, 0]]
    col_headers = [int(x) for x in block[0, 1:]]
    lookup: dict[frozenset[int], int] = {}
    for i, row_color in enumerate(row_headers):
        for j, col_color in enumerate(col_headers):
            key = frozenset((int(row_color), int(col_color)))
            value = int(block[i + 1, j + 1])
            if key in lookup:
                assert lookup[key] == value, "palette must define a consistent symmetric mix"
            else:
                lookup[key] = value
    return row_headers, lookup


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color endpoint pair and recolor crossings by consulting the corner palette.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    corner, palette = _detect_palette(g)
    pair_colors, mix_lookup = _parse_palette(palette)
    palette_mask = _palette_mask(g.shape, corner, int(palette.shape[0]))

    painted: dict[tuple[int, int], list[int]] = {}
    for color in pair_colors:
        coords = np.argwhere((g == int(color)) & ~palette_mask)
        assert coords.shape == (2, 2), "each palette color must appear exactly twice outside the palette"
        (r0, c0), (r1, c1) = [tuple(map(int, rc)) for rc in coords]
        assert (r0 == r1) ^ (c0 == c1), "endpoint pairs must align horizontally or vertically"
        if r0 == r1:
            for c in range(min(c0, c1), max(c0, c1) + 1):
                assert not palette_mask[r0, c], "lines may not pass through the palette"
                painted.setdefault((r0, c), []).append(int(color))
        else:
            for r in range(min(r0, r1), max(r0, r1) + 1):
                assert not palette_mask[r, c0], "lines may not pass through the palette"
                painted.setdefault((r, c0), []).append(int(color))

    out = g.copy()
    for (r, c), colors in painted.items():
        unique = list(dict.fromkeys(colors))
        assert len(unique) == len(colors), "a line cannot intersect itself"
        if len(unique) == 1:
            out[r, c] = unique[0]
        else:
            assert len(unique) == 2, "only pairwise crossings are supported"
            key = frozenset(unique)
            assert key in mix_lookup, "crossing colors must appear in the palette"
            out[r, c] = mix_lookup[key]
    return out


def _usable_interval(length: int, corner: str, palette_size: int, axis: str) -> tuple[int, int]:
    start = 0
    stop = length - 1
    if axis == "row":
        if corner in ("tl", "tr"):
            start = palette_size + 1
        else:
            stop = length - palette_size - 2
    else:
        if corner in ("tl", "bl"):
            start = palette_size + 1
        else:
            stop = length - palette_size - 2
    return start, stop


def _build_palette(
    pair_colors: list[int],
    row_order: list[int],
    col_order: list[int],
    mix_lookup: dict[frozenset[int], int],
) -> np.ndarray:
    size = len(pair_colors) + 1
    block = np.full((size, size), BG, dtype=int)
    block[0, 1:] = np.array(col_order, dtype=int)
    block[1:, 0] = np.array(row_order, dtype=int)
    for i, row_color in enumerate(row_order):
        for j, col_color in enumerate(col_order):
            block[i + 1, j + 1] = int(mix_lookup[frozenset((row_color, col_color))])
    return block


def _random_mix_lookup(pair_colors: list[int]) -> dict[frozenset[int], int]:
    remaining = [color for color in range(1, 10) if color not in pair_colors]
    pool_size = random.randint(2, min(4, len(remaining)))
    mix_pool = random.sample(remaining, pool_size)
    lookup: dict[frozenset[int], int] = {}
    for color in pair_colors:
        lookup[frozenset((int(color),))] = int(color)
    for i, color_a in enumerate(pair_colors):
        for color_b in pair_colors[i + 1 :]:
            lookup[frozenset((int(color_a), int(color_b)))] = int(random.choice(mix_pool))
    return lookup


def _gen_attempt() -> np.ndarray | None:
    color_count = random.randint(MIN_COLOR_COUNT, MAX_COLOR_COUNT)
    pair_colors = random.sample(list(range(1, 10)), color_count)
    palette_size = color_count + 1

    h = random.randint(max(MIN_GRID_SIZE, palette_size + 8), MAX_GRID_SIZE)
    w = random.randint(max(MIN_GRID_SIZE, palette_size + 8), MAX_GRID_SIZE)
    corner = random.choice(PALETTE_CORNERS)

    horizontal_count = random.randint(1, color_count - 1)
    vertical_count = color_count - horizontal_count

    row_lo, row_hi = _usable_interval(h, corner, palette_size, "row")
    col_lo, col_hi = _usable_interval(w, corner, palette_size, "col")
    if row_hi - row_lo + 1 < horizontal_count + 2:
        return None
    if col_hi - col_lo + 1 < vertical_count + 2:
        return None

    rows = sorted(random.sample(range(row_lo, row_hi + 1), horizontal_count + 2))
    cols = sorted(random.sample(range(col_lo, col_hi + 1), vertical_count + 2))
    top_row, bottom_row = rows[0], rows[-1]
    left_col, right_col = cols[0], cols[-1]
    horizontal_rows = rows[1:-1]
    vertical_cols = cols[1:-1]

    shuffled_colors = random.sample(pair_colors, len(pair_colors))
    horizontal_colors = shuffled_colors[:horizontal_count]
    vertical_colors = shuffled_colors[horizontal_count:]

    row_order = random.sample(pair_colors, len(pair_colors))
    col_order = random.sample(pair_colors, len(pair_colors))
    mix_lookup = _random_mix_lookup(pair_colors)

    grid = np.full((h, w), BG, dtype=int)
    palette = _build_palette(pair_colors, row_order, col_order, mix_lookup)
    _write_corner_block(grid, corner, palette)

    for color, row in zip(horizontal_colors, horizontal_rows):
        grid[row, left_col] = int(color)
        grid[row, right_col] = int(color)
    for color, col in zip(vertical_colors, vertical_cols):
        grid[top_row, col] = int(color)
        grid[bottom_row, col] = int(color)

    return grid


def generate_input() -> np.ndarray:
    """
    Generate a random puzzle instance with a corner palette and intersecting orthogonal lines.
    """
    for _ in range(MAX_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is None:
            continue
        try:
            _ = main(candidate)
        except Exception:
            continue
        return candidate
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
