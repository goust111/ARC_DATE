from __future__ import annotations

import random

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    YELLOW = 4
    GREY = 5


MAX_GENERATION_ATTEMPTS = 400
ROW_PERIODS_A = (1, 2, 3)
ROW_PERIODS_B = (2, 3)
COL_PERIODS = (1, 2, 3, 4)


def _single_foreground_color(region: np.ndarray) -> int:
    colors = sorted(set(map(int, np.unique(region))) - {Color.BLACK})
    assert len(colors) == 1, "expected exactly one foreground color in each fragment"
    return colors[0]


def _infer_periodic_tile(mask: np.ndarray, row_offset: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    h, w = map(int, mask.shape)
    best_key: tuple[int, int, int] | None = None
    best_tile: np.ndarray | None = None

    for period_r in range(1, h + 1):
        for period_c in range(1, w + 1):
            tile = np.zeros((period_r, period_c), dtype=np.int8)
            seen = np.zeros((period_r, period_c), dtype=bool)
            ok = True
            for r in range(h):
                rr = (row_offset + r) % period_r
                for c in range(w):
                    cc = c % period_c
                    value = int(mask[r, c])
                    if seen[rr, cc]:
                        if tile[rr, cc] != value:
                            ok = False
                            break
                    else:
                        seen[rr, cc] = True
                        tile[rr, cc] = value
                if not ok:
                    break
            if ok and np.all(seen):
                key = (period_r * period_c, period_r, period_c)
                if best_key is None or key < best_key:
                    best_key = key
                    best_tile = tile.astype(bool)

    assert best_tile is not None, "failed to infer a periodic tile"
    return best_tile


def _render_mask(tile: np.ndarray, row_start: int, height: int, width: int) -> np.ndarray:
    period_r, period_c = map(int, tile.shape)
    rows = np.arange(row_start, row_start + height) % period_r
    cols = np.arange(width) % period_c
    return tile[rows[:, None], cols[None, :]]


def _render_region(tile: np.ndarray, row_start: int, height: int, width: int, color: int) -> np.ndarray:
    region = np.full((height, width), Color.BLACK, dtype=int)
    region[_render_mask(tile, row_start, height, width)] = int(color)
    return region


def _bottom_intersection(
    tile_a: np.ndarray,
    tile_b: np.ndarray,
    bottom_start: int,
    bottom_height: int,
    width: int,
) -> np.ndarray:
    mask_a = _render_mask(tile_a, bottom_start, bottom_height, width)
    mask_b = _render_mask(tile_b, bottom_start, bottom_height, width)
    return mask_a & mask_b


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the two visible periodic fragments to the lower blank region and keep only their overlapping foreground.

    Input structure:
    - Two solid grey separator rows split the grid into a top fragment, a middle fragment, and a lower black region.
    - The top fragment contains one red-on-black periodic pattern fragment.
    - The middle fragment contains one blue-on-black periodic pattern fragment.
    - The lower region is empty black background.

    Rule:
    - Infer the smallest row/column translation period consistent with the red fragment, using whole-grid row indices.
    - Infer the smallest row/column translation period consistent with the blue fragment, again in the same global coordinates.
    - Extend both periodic foreground masks into the lower region.
    - Recolor only the lower cells covered by both extensions to yellow; keep everything else unchanged.

    Key insight:
    - The answer is not either periodic pattern by itself: each fragment must be completed in the shared absolute coordinate
      system first, and the output is the intersection of those two completed foreground masks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    separator_rows = np.flatnonzero(np.all(g == Color.GREY, axis=1))
    assert len(separator_rows) == 2, "expected exactly two grey separator rows"
    sep1, sep2 = map(int, separator_rows)
    assert 0 < sep1 < sep2 - 1 < g.shape[0] - 1, "separator rows must create three non-empty regions"

    top = g[:sep1]
    middle = g[sep1 + 1 : sep2]
    bottom = g[sep2 + 1 :]
    width = int(g.shape[1])

    top_fg = _single_foreground_color(top)
    middle_fg = _single_foreground_color(middle)

    tile_a = _infer_periodic_tile(top == top_fg, row_offset=0)
    tile_b = _infer_periodic_tile(middle == middle_fg, row_offset=sep1 + 1)

    out = g.copy()
    out[sep2 + 1 :] = Color.BLACK
    overlap = _bottom_intersection(tile_a, tile_b, sep2 + 1, bottom.shape[0], width)
    out_bottom = out[sep2 + 1 :]
    out_bottom[overlap] = Color.YELLOW
    return out


def _make_primitive_tile(period_r: int, period_c: int) -> np.ndarray | None:
    area = period_r * period_c
    if area == 1:
        return None

    min_fg = max(1, int(np.ceil(area * 0.25)))
    max_fg = min(area - 1, int(np.floor(area * 0.60)))
    if min_fg > max_fg:
        return None

    for _ in range(300):
        fg_count = random.randint(min_fg, max_fg)
        chosen = random.sample(range(area), fg_count)
        tile = np.zeros((period_r, period_c), dtype=bool)
        tile.flat[chosen] = True

        if period_r > 1 and len({tuple(tile[r, :]) for r in range(period_r)}) < 2:
            continue
        if period_c > 1 and len({tuple(tile[:, c]) for c in range(period_c)}) < 2:
            continue

        inferred = _infer_periodic_tile(tile, row_offset=0)
        if inferred.shape == tile.shape and np.array_equal(inferred, tile):
            return tile
    return None


def _random_visible_size(period: int, min_repeats: int, max_repeats: int) -> int:
    repeats = random.randint(min_repeats, max_repeats)
    return repeats * period + random.randint(0, max(0, period - 1))


def _naive_gapless_mask(tile: np.ndarray, visible_start: int, visible_height: int, bottom_height: int, width: int) -> np.ndarray:
    return _render_mask(tile, visible_start + visible_height, bottom_height, width)


def _gen_attempt() -> np.ndarray | None:
    period_ra = random.choice(ROW_PERIODS_A)
    period_rb = random.choice(ROW_PERIODS_B)
    period_ca = random.choice(COL_PERIODS)
    period_cb = random.choice(COL_PERIODS)

    tile_a = _make_primitive_tile(period_ra, period_ca)
    tile_b = _make_primitive_tile(period_rb, period_cb)
    if tile_a is None or tile_b is None:
        return None

    top_h = _random_visible_size(period_ra, 2, 3)
    middle_h = _random_visible_size(period_rb, 2, 3)
    bottom_h = _random_visible_size(max(period_ra, period_rb), 2, 3)
    width = _random_visible_size(max(period_ca, period_cb), 2, 4)
    total_h = top_h + middle_h + bottom_h + 2
    if total_h > 30 or width > 30:
        return None

    sep1 = top_h
    sep2 = top_h + 1 + middle_h
    bottom_start = sep2 + 1

    top_region = _render_region(tile_a, 0, top_h, width, Color.RED)
    middle_region = _render_region(tile_b, sep1 + 1, middle_h, width, Color.BLUE)
    if not np.any(top_region == Color.RED) or not np.any(top_region == Color.BLACK):
        return None
    if not np.any(middle_region == Color.BLUE) or not np.any(middle_region == Color.BLACK):
        return None

    grid = np.full((total_h, width), Color.BLACK, dtype=int)
    grid[:top_h] = top_region
    grid[sep1] = Color.GREY
    grid[sep1 + 1 : sep2] = middle_region
    grid[sep2] = Color.GREY

    overlap = _bottom_intersection(tile_a, tile_b, bottom_start, bottom_h, width)
    overlap_count = int(overlap.sum())
    bottom_area = bottom_h * width
    if overlap_count < (1 if bottom_area < 20 else 2):
        return None
    if overlap_count > max(1, bottom_area // 3):
        return None

    wrong_b = _render_mask(tile_a, bottom_start, bottom_h, width) & _naive_gapless_mask(
        tile_b,
        sep1 + 1,
        middle_h,
        bottom_h,
        width,
    )
    if np.array_equal(overlap, wrong_b):
        return None

    wrong_both = _naive_gapless_mask(tile_a, 0, top_h, bottom_h, width) & _naive_gapless_mask(
        tile_b,
        sep1 + 1,
        middle_h,
        bottom_h,
        width,
    )
    if np.array_equal(overlap, wrong_both):
        return None

    inferred_a = _infer_periodic_tile(top_region == Color.RED, row_offset=0)
    inferred_b = _infer_periodic_tile(middle_region == Color.BLUE, row_offset=sep1 + 1)
    if not np.array_equal(inferred_a, tile_a):
        return None
    if not np.array_equal(inferred_b, tile_b):
        return None

    expected = grid.copy()
    expected_bottom = expected[bottom_start:]
    expected_bottom[overlap] = Color.YELLOW
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-intersection input")
