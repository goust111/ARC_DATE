from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_RETRY_LIMIT = 400


def _count_bands(separator_flags: np.ndarray) -> int:
    bands = 0
    in_band = False
    for is_separator in separator_flags.tolist():
        if is_separator:
            in_band = False
        elif not in_band:
            bands += 1
            in_band = True
    return bands


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_single_component(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _random_composition(total: int, parts: int) -> list[int]:
    assert parts >= 1 and total >= parts
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    values = []
    prev = 0
    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut
    return values


def _separator_positions(total: int, band_sizes: list[int], add_start: bool, add_end: bool) -> list[int]:
    positions: list[int] = []
    cursor = 1 if add_start else 0
    if add_start:
        positions.append(0)
    for idx, size in enumerate(band_sizes):
        cursor += size
        if idx < len(band_sizes) - 1:
            positions.append(cursor)
            cursor += 1
    if add_end:
        positions.append(total - 1)
        assert cursor == total - 1
    else:
        assert cursor == total
    return positions


def _choose_border_flags(total: int, bands: int) -> tuple[bool, bool]:
    options = []
    for add_top in (False, True):
        for add_bottom in (False, True):
            sep_count = (bands - 1) + int(add_top) + int(add_bottom)
            if total - sep_count >= bands:
                options.append((add_top, add_bottom))
    assert options
    return random.choice(options)


def _extract_partition_counts(left: np.ndarray) -> tuple[int, int]:
    full_rows = np.all(left == BLUE, axis=1)
    full_cols = np.all(left == BLUE, axis=0)
    assert np.any(full_rows)
    assert np.any(full_cols)
    expected_mask = full_rows[:, None] | full_cols[None, :]
    assert np.array_equal(left == BLUE, expected_mask)
    row_bands = _count_bands(full_rows)
    col_bands = _count_bands(full_cols)
    assert row_bands >= 2 and col_bands >= 2
    return row_bands, col_bands


def _extract_sprite(right: np.ndarray) -> np.ndarray:
    mask = right != BG
    assert np.any(mask)
    assert _is_single_component(mask)
    r0, c0, r1, c1 = _bbox(mask)
    sprite = right[r0 : r1 + 1, c0 : c1 + 1].copy()
    h, w = sprite.shape
    assert h <= 3 and w <= 3
    area = int(np.count_nonzero(sprite))
    assert 3 <= area <= 5
    colors = set(map(int, np.unique(sprite[sprite != BG])))
    assert colors == {GREEN, YELLOW}
    assert _is_single_component(sprite != BG)
    return sprite


def _tile_recolored(sprite: np.ndarray, rows: int, cols: int, target_shape: tuple[int, int]) -> np.ndarray:
    h, w = sprite.shape
    out = np.zeros(target_shape, dtype=int)
    recolored = sprite.copy()
    recolored[recolored == GREEN] = RED
    expected_h = rows * h + (rows - 1)
    expected_w = cols * w + (cols - 1)
    assert target_shape == (expected_h, expected_w)
    for rr in range(rows):
        for cc in range(cols):
            r0 = rr * (h + 1)
            c0 = cc * (w + 1)
            block = out[r0 : r0 + h, c0 : c0 + w]
            mask = recolored != BG
            block[mask] = recolored[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the left-side separator partitions, then tile and recolor the right-side sprite to that many rows and columns.

    Input structure:
    - The background is black, and one full-height grey vertical line splits the grid into a left region and a right region.
    - In the left region, blue full rows and blue full columns form a separator-line partition; these lines may include the region border.
    - In the right region, exactly one small 4-connected sprite appears on black background, using green and yellow and fitting in a bbox of at most 3x3.

    Rule:
    - Count the contiguous non-separator row bands and column bands induced by the blue lines in the left region; call them R and C.
    - Extract the right-region sprite bbox as the base pattern.
    - Rebuild the right region as an R-by-C tiling of that sprite, starting at the right region's top-left corner, with a single black spacer row and column between adjacent copies.
    - Recolor every green sprite pixel to red while leaving yellow unchanged, and keep the left region plus the grey divider exactly as they were.

    Key insight:
    - The left half does not contribute visible output content directly; it only encodes the tiling dimensions that must be transferred to the right-side sprite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    grey_cols = np.where(np.all(g == GREY, axis=0))[0]
    assert len(grey_cols) == 1
    sep_c = int(grey_cols[0])
    assert 0 < sep_c < g.shape[1] - 1

    left = g[:, :sep_c]
    right = g[:, sep_c + 1 :]
    assert np.all((left == BG) | (left == BLUE))
    assert np.all((right == BG) | (right == GREEN) | (right == YELLOW))

    rows, cols = _extract_partition_counts(left)
    sprite = _extract_sprite(right)
    tiled = _tile_recolored(sprite, rows, cols, right.shape)

    out = g.copy()
    out[:, sep_c + 1 :] = tiled
    return out


def _generate_sprite() -> np.ndarray:
    shapes = [(2, 2), (2, 3), (3, 2), (3, 3)]
    for _ in range(200):
        h, w = random.choice(shapes)
        area = random.randint(3, min(5, h * w))
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                        frontier.append((rr, cc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        if min(rs) != 0 or max(rs) != h - 1 or min(cs) != 0 or max(cs) != w - 1:
            continue

        sprite = np.zeros((h, w), dtype=int)
        ordered = list(cells)
        random.shuffle(ordered)
        split = random.randint(1, area - 1)
        for idx, (r, c) in enumerate(ordered):
            sprite[r, c] = GREEN if idx < split else YELLOW
        if {int(v) for v in sprite.flat if v != BG} != {GREEN, YELLOW}:
            continue
        if _is_single_component(sprite != BG):
            return sprite
    raise AssertionError("failed to generate sprite")


def _build_left_region(height: int, width: int, rows: int, cols: int) -> np.ndarray:
    row_start, row_end = _choose_border_flags(height, rows)
    col_start, col_end = _choose_border_flags(width, cols)
    row_sep_count = (rows - 1) + int(row_start) + int(row_end)
    col_sep_count = (cols - 1) + int(col_start) + int(col_end)
    row_bands = _random_composition(height - row_sep_count, rows)
    col_bands = _random_composition(width - col_sep_count, cols)
    row_lines = _separator_positions(height, row_bands, row_start, row_end)
    col_lines = _separator_positions(width, col_bands, col_start, col_end)

    left = np.zeros((height, width), dtype=int)
    left[row_lines, :] = BLUE
    left[:, col_lines] = BLUE

    found_rows, found_cols = _extract_partition_counts(left)
    assert (found_rows, found_cols) == (rows, cols)
    return left


def _construct_expected(left: np.ndarray, sprite: np.ndarray, rows: int, cols: int) -> np.ndarray:
    right = _tile_recolored(sprite, rows, cols, (rows * sprite.shape[0] + rows - 1, cols * sprite.shape[1] + cols - 1))
    out = np.zeros((left.shape[0], left.shape[1] + 1 + right.shape[1]), dtype=int)
    out[:, : left.shape[1]] = left
    out[:, left.shape[1]] = GREY
    out[:, left.shape[1] + 1 :] = right
    return out


def _generate_attempt() -> np.ndarray | None:
    sprite = _generate_sprite()
    sh, sw = sprite.shape

    rows = random.randint(2, 5)
    cols = random.randint(2, 5)
    height = rows * sh + (rows - 1)
    right_w = cols * sw + (cols - 1)
    if height > 30 or right_w >= 30:
        return None

    col_border_flags = _choose_border_flags(max(2 * cols - 1, 2), cols)
    min_left_w = cols + (cols - 1) + int(col_border_flags[0]) + int(col_border_flags[1])
    max_left_w = 29 - right_w
    if max_left_w < min_left_w:
        return None
    left_w = random.randint(min_left_w, max_left_w)

    total_w = left_w + 1 + right_w
    if height < 8 or total_w < 14 or height > 30 or total_w > 30:
        return None

    left = _build_left_region(height, left_w, rows, cols)
    expected = _construct_expected(left, sprite, rows, cols)

    inp = np.zeros_like(expected)
    inp[:, :left_w] = left
    inp[:, left_w] = GREY

    max_off_r = min(1, height - sh)
    max_off_c = min(1, right_w - sw)
    off_r = random.randint(0, max_off_r)
    off_c = random.randint(0, max_off_c)
    inp[off_r : off_r + sh, left_w + 1 + off_c : left_w + 1 + off_c + sw] = sprite

    try:
        out = main(inp)
    except Exception:
        return None

    if not np.array_equal(out, expected):
        return None

    right_in = inp[:, left_w + 1 :]
    if int(np.count_nonzero(right_in)) != int(np.count_nonzero(sprite)):
        return None
    if _extract_partition_counts(inp[:, :left_w]) != (rows, cols):
        return None
    if _extract_sprite(right_in).shape != sprite.shape:
        return None
    if expected.shape != inp.shape:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid input")
