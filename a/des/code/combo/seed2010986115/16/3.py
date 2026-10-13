from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)


def _find_pair_table(g: np.ndarray) -> tuple[dict[int, int], int]:
    h, w = g.shape
    assert w >= 2

    rows: list[int] = []
    r = h - 1
    while r >= 0:
        left = int(g[r, w - 2])
        right = int(g[r, w - 1])
        if left not in (BLACK, GREY) and right not in (BLACK, GREY):
            rows.append(r)
            r -= 1
        else:
            break

    assert rows, "expected a bottom-right color pair table"
    rows.reverse()

    mapping: dict[int, int] = {}
    for row in rows:
        src = int(g[row, w - 2])
        dst = int(g[row, w - 1])
        assert src not in mapping, "pair-table source colors must be unique"
        assert src != dst, "source and shadow colors must differ"
        mapping[src] = dst

    return mapping, rows[0]


def _collect_runs(line: np.ndarray, valid_colors: set[int], offset: int) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    i = 0
    while i < len(line):
        color = int(line[i])
        if color == BLACK:
            i += 1
            continue
        assert color in valid_colors, "edge strips must use colors listed in the pair table"
        j = i + 1
        while j < len(line) and int(line[j]) == color:
            j += 1
        runs.append((offset + i, offset + j, color))
        i = j
    return runs


def _cast_horizontal(
    out: np.ndarray,
    original: np.ndarray,
    obstacle_mask: np.ndarray,
    row: int,
    color: int,
    shadow_color: int,
) -> None:
    for c in range(1, original.shape[1]):
        if obstacle_mask[row, c]:
            out[row, c] = shadow_color
            break
        if int(original[row, c]) == BLACK:
            out[row, c] = color


def _cast_vertical(
    out: np.ndarray,
    original: np.ndarray,
    obstacle_mask: np.ndarray,
    col: int,
    color: int,
    shadow_color: int,
) -> None:
    for r in range(1, original.shape[0]):
        if obstacle_mask[r, col]:
            out[r, col] = shadow_color
            break
        if int(original[r, col]) == BLACK:
            out[r, col] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project edge color bands into the grid and recolor obstacle faces via a bottom-right lookup table.

    Input structure
    - Background is black.
    - The top edge and left edge contain separated monochrome source bands.
    - Interior grey rectangles are obstacles.
    - The bottom-right corner contains an N x 2 source->shadow lookup table.

    Rule
    - Top-edge bands project vertical beams downward.
    - Left-edge bands project horizontal beams rightward.
    - Beams color only background cells and stop at the first grey obstacle cell.
    - The lit obstacle face is recolored with the table's mapped shadow color.
    - Where horizontal and vertical beams cross on background, the vertical color wins.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    mapping, table_r0 = _find_pair_table(g)
    valid_colors = set(mapping)
    obstacle_mask = g == GREY
    out = g.copy()

    left_runs = _collect_runs(g[1:table_r0, 0], valid_colors, 1)
    top_runs = _collect_runs(g[0, 1:-2], valid_colors, 1)

    for r0, r1, color in left_runs:
        shadow = mapping[color]
        for r in range(r0, r1):
            _cast_horizontal(out, g, obstacle_mask, r, color, shadow)

    for c0, c1, color in top_runs:
        shadow = mapping[color]
        for c in range(c0, c1):
            _cast_vertical(out, g, obstacle_mask, c, color, shadow)

    return out


MIN_SIZE = 18
MAX_SIZE = 27
MAX_GENERATION_ATTEMPTS = 400


def _band_containing(point: int, lo: int, hi: int, min_size: int, max_size: int) -> tuple[int, int] | None:
    max_size = min(max_size, hi - lo + 1)
    if max_size < min_size:
        return None
    size = random.randint(min_size, max_size)
    start_lo = max(lo, point - size + 1)
    start_hi = min(point, hi - size + 1)
    if start_lo > start_hi:
        return None
    start = random.randint(start_lo, start_hi)
    return start, start + size - 1


def _place_bands(limit: int, total_count: int, fixed_band: tuple[int, int]) -> list[tuple[int, int]] | None:
    if total_count < 1 or fixed_band[1] > limit:
        return None

    bands = [fixed_band]
    occupied = np.zeros(limit + 3, dtype=bool)

    def reserve(a: int, b: int) -> None:
        occupied[max(1, a - 1) : min(limit, b + 1) + 1] = True

    reserve(*fixed_band)
    attempts = 0
    while len(bands) < total_count and attempts < 200:
        attempts += 1
        size = random.randint(1, 4)
        if size > limit:
            continue
        start = random.randint(1, limit - size + 1)
        end = start + size - 1
        if occupied[start : end + 1].any():
            continue
        if occupied[start - 1] or occupied[end + 1]:
            continue
        bands.append((start, end))
        reserve(start, end)

    if len(bands) != total_count:
        return None
    return sorted(bands)


def _choose_counts(color_count: int) -> tuple[int, int] | None:
    top = random.randint(1, 3)
    left = random.randint(1, 3)
    while top + left < color_count:
        choices = []
        if top < 3:
            choices.append("top")
        if left < 3:
            choices.append("left")
        if not choices:
            return None
        if random.choice(choices) == "top":
            top += 1
        else:
            left += 1
    return top, left


def _assign_band_colors(
    top_bands: list[tuple[int, int]],
    left_bands: list[tuple[int, int]],
    top_fixed: tuple[int, int],
    left_fixed: tuple[int, int],
    source_colors: list[int],
    top_color: int,
    left_color: int,
) -> tuple[list[int], list[int]]:
    top_colors: list[int | None] = [None] * len(top_bands)
    left_colors: list[int | None] = [None] * len(left_bands)

    top_fixed_index = top_bands.index(top_fixed)
    left_fixed_index = left_bands.index(left_fixed)
    top_colors[top_fixed_index] = top_color
    left_colors[left_fixed_index] = left_color

    slots = [("top", i) for i, color in enumerate(top_colors) if color is None]
    slots += [("left", i) for i, color in enumerate(left_colors) if color is None]

    remaining_needed = [c for c in source_colors if c not in {top_color, left_color}]
    fill = remaining_needed[:]
    while len(fill) < len(slots):
        fill.append(random.choice(source_colors))
    random.shuffle(fill)

    for (side, idx), color in zip(slots, fill):
        if side == "top":
            top_colors[idx] = color
        else:
            left_colors[idx] = color

    return [int(c) for c in top_colors], [int(c) for c in left_colors]


def _paint_rect(grid: np.ndarray, occupied: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> bool:
    if h <= 0 or w <= 0:
        return False
    r1 = r0 + h
    c1 = c0 + w
    if r0 < 0 or c0 < 0 or r1 > grid.shape[0] or c1 > grid.shape[1]:
        return False
    if occupied[r0:r1, c0:c1].any():
        return False
    grid[r0:r1, c0:c1] = color
    occupied[r0:r1, c0:c1] = True
    return True


def _crossing_cell(top_band: tuple[int, int], left_band: tuple[int, int]) -> tuple[int, int]:
    return left_band[0], top_band[0]


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(MIN_SIZE, MAX_SIZE)
    w = random.randint(MIN_SIZE, MAX_SIZE)
    pair_count = random.randint(2, 4)
    source_colors = random.sample(list(COLOR_POOL), pair_count)
    shadow_pool = [c for c in COLOR_POOL if c not in source_colors]
    shadow_colors = random.sample(shadow_pool, pair_count)
    pair_rows = list(zip(source_colors, shadow_colors))
    random.shuffle(pair_rows)
    mapping = {src: dst for src, dst in pair_rows}

    table_r0 = h - pair_count
    barrier_w = random.randint(1, 2)
    barrier_c0 = w - 2 - 1 - barrier_w
    top_zone_hi = barrier_c0 - 7
    left_zone_hi = table_r0 - 7
    if top_zone_hi < 6 or left_zone_hi < 6:
        return None

    counts = _choose_counts(pair_count)
    if counts is None:
        return None
    top_count, left_count = counts

    top_color, left_color = random.sample(source_colors, 2)

    cross_col = random.randint(3, top_zone_hi - 1)
    cross_row = random.randint(3, left_zone_hi - 1)
    top_fixed = _band_containing(cross_col, 1, top_zone_hi, 2, 4)
    left_fixed = _band_containing(cross_row, 1, left_zone_hi, 2, 4)
    if top_fixed is None or left_fixed is None:
        return None

    top_bands = _place_bands(top_zone_hi, top_count, top_fixed)
    left_bands = _place_bands(left_zone_hi, left_count, left_fixed)
    if top_bands is None or left_bands is None:
        return None

    top_colors, left_colors = _assign_band_colors(
        top_bands,
        left_bands,
        top_fixed,
        left_fixed,
        source_colors,
        top_color,
        left_color,
    )

    grid = np.full((h, w), BLACK, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    if not _paint_rect(grid, occupied, 1, barrier_c0, table_r0 - 1, barrier_w, GREY):
        return None

    v_h_max = min(4, table_r0 - left_zone_hi - 3)
    if v_h_max < 2:
        return None
    v_h = random.randint(2, v_h_max)
    v_w = random.randint(1, min(3, top_fixed[1] - top_fixed[0] + 1))
    v_r0 = random.randint(left_zone_hi + 2, table_r0 - v_h - 1)
    v_c0 = random.randint(top_fixed[0], top_fixed[1] - v_w + 1)
    if not _paint_rect(grid, occupied, v_r0, v_c0, v_h, v_w, GREY):
        return None

    h_w_max = min(4, barrier_c0 - top_zone_hi - 3)
    if h_w_max < 2:
        return None
    h_h = random.randint(2, min(3, left_fixed[1] - left_fixed[0] + 1))
    h_w = random.randint(2, h_w_max)
    h_r0 = random.randint(left_fixed[0], left_fixed[1] - h_h + 1)
    h_c0 = random.randint(top_zone_hi + 2, barrier_c0 - h_w - 1)
    if not _paint_rect(grid, occupied, h_r0, h_c0, h_h, h_w, GREY):
        return None

    extra_count = random.randint(0, 2)
    for _ in range(extra_count):
        for _attempt in range(40):
            eh = random.randint(2, 4)
            ew = random.randint(2, 4)
            max_r0 = table_r0 - eh - 1
            max_c0 = barrier_c0 - ew - 1
            if max_r0 < left_zone_hi + 1 or max_c0 < top_zone_hi + 1:
                break
            er0 = random.randint(left_zone_hi + 1, max_r0)
            ec0 = random.randint(top_zone_hi + 1, max_c0)
            if _paint_rect(grid, occupied, er0, ec0, eh, ew, GREY):
                break

    for (start, end), color in zip(top_bands, top_colors):
        grid[0, start : end + 1] = color
    for (start, end), color in zip(left_bands, left_colors):
        grid[start : end + 1, 0] = color

    for i, (src, dst) in enumerate(pair_rows):
        grid[table_r0 + i, w - 2] = src
        grid[table_r0 + i, w - 1] = dst

    cross_r, cross_c = _crossing_cell(top_fixed, left_fixed)

    try:
        out = main(grid)
    except Exception:
        return None

    if int(out[cross_r, cross_c]) != top_color:
        return None
    if not np.all(out[v_r0, v_c0 : v_c0 + v_w] == mapping[top_color]):
        return None
    if not np.all(out[h_r0 : h_r0 + h_h, h_c0] == mapping[left_color]):
        return None
    if not np.array_equal(out[table_r0:, w - 2 : w], grid[table_r0:, w - 2 : w]):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
