import math
import random

import numpy as np


BG = 0


def _foreground_colors_in_order(grid: np.ndarray) -> list[int]:
    seen: set[int] = set()
    colors: list[int] = []
    for value in np.asarray(grid, dtype=int).ravel():
        color = int(value)
        if color != BG and color not in seen:
            seen.add(color)
            colors.append(color)
    return colors


def _tile_to_shape(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    th, tw = map(int, tile.shape)
    reps_r = math.ceil(height / th)
    reps_c = math.ceil(width / tw)
    return np.tile(tile, (reps_r, reps_c))[:height, :width]


def _minimal_row_period(mask: np.ndarray) -> int:
    height = int(mask.shape[0])
    for period in range(1, height + 1):
        if np.array_equal(_tile_to_shape(mask[:period, :], height, mask.shape[1]), mask):
            return period
    raise AssertionError("row period not found")


def _minimal_col_period(mask: np.ndarray) -> int:
    width = int(mask.shape[1])
    for period in range(1, width + 1):
        if np.array_equal(_tile_to_shape(mask[:, :period], mask.shape[0], width), mask):
            return period
    raise AssertionError("column period not found")


def _extract_minimal_tile(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    assert binary.ndim == 2
    assert binary.any(), "foreground layer must be nonempty"
    row_period = _minimal_row_period(binary)
    col_period = _minimal_col_period(binary)
    tile = binary[:row_period, :col_period].copy()
    assert np.array_equal(_tile_to_shape(tile, *binary.shape), binary)
    return tile


def _stack_tiles(layers: list[tuple[int, np.ndarray]]) -> np.ndarray:
    assert layers
    out_h = sum(int(tile.shape[0]) for _color, tile in layers) + len(layers) - 1
    out_w = max(int(tile.shape[1]) for _color, tile in layers)
    out = np.full((out_h, out_w), BG, dtype=int)

    r0 = 0
    for color, tile in layers:
        h, w = map(int, tile.shape)
        patch = out[r0 : r0 + h, :w]
        patch[tile] = int(color)
        r0 += h + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify each foreground color's periodic layer, recover its smallest repeating
    tile, then stack the tiles vertically in descending pixel-count order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    layers: list[tuple[int, int, int, np.ndarray]] = []
    for color in _foreground_colors_in_order(g):
        mask = g == int(color)
        tile = _extract_minimal_tile(mask)
        count = int(mask.sum())
        first_idx = int(np.flatnonzero(mask)[0])
        layers.append((count, first_idx, int(color), tile))

    assert 2 <= len(layers) <= 4, "expected 2-4 foreground colors"
    layers.sort(key=lambda item: (-item[0], item[1]))
    return _stack_tiles([(color, tile) for _count, _first, color, tile in layers])


# --------------- generator ---------------

COLOR_POOL = tuple(range(1, 10))
COLOR_COUNT_RANGE = (2, 4)
PERIOD_CHOICES = tuple(
    (h, w)
    for h in range(1, 5)
    for w in range(1, 5)
    if (h, w) != (1, 1)
)
GRID_SIZE_RANGE = (8, 18)
MAX_TOTAL_FILL_RATIO = 0.58
GENERATION_ATTEMPTS = 400
TILE_ATTEMPTS = 250


def _sample_grid_dim(periods: list[int]) -> int:
    lo = max(GRID_SIZE_RANGE[0], 2 * max(periods))
    hi = GRID_SIZE_RANGE[1]
    assert lo <= hi
    candidates = list(range(lo, hi + 1))
    if random.random() < 0.5:
        non_multiples = [size for size in candidates if any(size % period != 0 for period in periods)]
        if non_multiples:
            candidates = non_multiples
    return random.choice(candidates)


def _fundamental_shape(tile: np.ndarray) -> tuple[int, int]:
    fundamental = _extract_minimal_tile(tile)
    return tuple(map(int, fundamental.shape))


def _sample_tile(h: int, w: int, fill_band: tuple[float, float]) -> np.ndarray | None:
    area = h * w
    lo, hi = fill_band
    min_fill = max(1, math.ceil(area * lo))
    max_fill = min(area - 1, math.floor(area * hi))
    if max_fill < min_fill:
        min_fill = max_fill = min(area - 1, max(1, round(area * ((lo + hi) / 2))))
    if min_fill > max_fill:
        return None

    cells = list(range(area))
    for _ in range(TILE_ATTEMPTS):
        fill = random.randint(min_fill, max_fill)
        chosen = random.sample(cells, fill)
        tile = np.zeros((h, w), dtype=bool)
        tile.flat[chosen] = True
        if _fundamental_shape(tile) == (h, w):
            return tile
    return None


def _build_mask_from_tile(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    return _tile_to_shape(np.asarray(tile, dtype=bool), height, width)


def _stack_expected(layer_specs: list[dict]) -> np.ndarray:
    ordered = sorted(layer_specs, key=lambda spec: (-spec["count"], spec["first_idx"]))
    return _stack_tiles([(int(spec["color"]), np.asarray(spec["tile"], dtype=bool)) for spec in ordered])


def _gen_attempt() -> np.ndarray | None:
    n_colors = random.randint(*COLOR_COUNT_RANGE)
    periods = random.sample(PERIOD_CHOICES, n_colors)
    colors = random.sample(COLOR_POOL, n_colors)

    height = _sample_grid_dim([h for h, _w in periods])
    width = _sample_grid_dim([w for _h, w in periods])

    lcm_h = math.lcm(*(h for h, _w in periods))
    lcm_w = math.lcm(*(_w for _h, _w in periods))
    super_occupied = np.zeros((lcm_h, lcm_w), dtype=bool)

    fill_bands = [
        (0.24, 0.42),
        (0.14, 0.28),
        (0.09, 0.18),
        (0.05, 0.12),
    ][:n_colors]

    specs: list[dict] = []
    for (tile_h, tile_w), color, fill_band in zip(periods, colors, fill_bands):
        placed = False
        for _ in range(TILE_ATTEMPTS):
            tile = _sample_tile(tile_h, tile_w, fill_band)
            if tile is None:
                break
            super_mask = _build_mask_from_tile(tile, lcm_h, lcm_w)
            if np.any(super_mask & super_occupied):
                continue

            full_mask = _build_mask_from_tile(tile, height, width)
            count = int(full_mask.sum())
            if count == 0:
                continue

            first_idx = int(np.flatnonzero(full_mask)[0])
            specs.append(
                {
                    "color": int(color),
                    "tile": tile.copy(),
                    "mask": full_mask,
                    "count": count,
                    "first_idx": first_idx,
                }
            )
            super_occupied |= super_mask
            placed = True
            break
        if not placed:
            return None

    counts = [int(spec["count"]) for spec in specs]
    if len(set(counts)) != len(counts):
        return None

    total_filled = sum(counts)
    if total_filled >= int(height * width * MAX_TOTAL_FILL_RATIO):
        return None

    grid = np.full((height, width), BG, dtype=int)
    for spec in specs:
        grid[spec["mask"]] = int(spec["color"])

    expected = _stack_expected(specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-layer input")
