from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _candidate_tile(grid: np.ndarray, tile_h: int, tile_w: int) -> np.ndarray | None:
    known = grid != 0
    tile = np.full((tile_h, tile_w), -1, dtype=int)
    for r, c in np.argwhere(known):
        rr = int(r) % tile_h
        cc = int(c) % tile_w
        value = int(grid[r, c])
        if tile[rr, cc] == -1:
            tile[rr, cc] = value
        elif tile[rr, cc] != value:
            return None
    if np.any(tile == -1):
        return None
    return tile


def _infer_minimal_tile(grid: np.ndarray) -> np.ndarray:
    height, width = map(int, grid.shape)
    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for tile_h in range(1, height + 1):
        for tile_w in range(1, width + 1):
            tile = _candidate_tile(grid, tile_h, tile_w)
            if tile is not None:
                candidates.append((tile_h * tile_w, tile_h, tile_w, tile))
    assert candidates, "expected a fully observed periodic tile"

    best_area = min(area for area, *_rest in candidates)
    best = [(tile_h, tile_w, tile) for area, tile_h, tile_w, tile in candidates if area == best_area]
    assert len(best) == 1, "expected a unique minimal periodic tile"
    return best[0][2]


def _repeat_tile(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = map(int, shape)
    tile_h, tile_w = map(int, tile.shape)
    rows = np.arange(height)[:, None] % tile_h
    cols = np.arange(width)[None, :] % tile_w
    return tile[rows, cols]


def _scale(grid: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(grid, np.ones((factor, factor), dtype=int))


def _minimal_period(tile: np.ndarray) -> tuple[int, int]:
    height, width = map(int, tile.shape)
    periods: list[tuple[int, int, int]] = []
    for tile_h in range(1, height + 1):
        for tile_w in range(1, width + 1):
            if _candidate_tile(tile, tile_h, tile_w) is not None:
                periods.append((tile_h * tile_w, tile_h, tile_w))
    best_area = min(area for area, *_rest in periods)
    best = [(tile_h, tile_w) for area, tile_h, tile_w in periods if area == best_area]
    assert best
    return best[0]


def _clean_grid_and_scale(grid: np.ndarray) -> tuple[np.ndarray, int]:
    counts = Counter(int(v) for v in grid.flat if int(v) != 0)
    wallpaper_colors = {color for color, count in counts.items() if count >= 3}
    marker_colors = {color for color, count in counts.items() if count < 3}
    assert wallpaper_colors, "expected wallpaper colors with frequency at least three"
    assert marker_colors, "expected rare pollution markers"

    marker_mask = np.isin(grid, list(marker_colors))
    scale = int(np.count_nonzero(marker_mask))
    assert scale > 0

    cleaned = np.array(grid, dtype=int, copy=True)
    cleaned[marker_mask] = 0
    return cleaned, scale


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the rare pollution markers, infer the latent wallpaper tile, and magnify the repaired wallpaper by the marker count.

    Input structure:
    - The grid is a small periodic wallpaper made from 2-3 non-black colors plus 1-2 genuine black holes.
    - A few additional pixels use rare non-black colors that are not part of the wallpaper; these pollution markers occur fewer than three times per color.
    - After treating both the black holes and the pollution markers as missing, every tile phase of the true wallpaper is still observed.

    Rule:
    - Classify every non-black color with frequency at least three as a wallpaper color, and every rarer non-black pixel as a pollution marker.
    - Count all pollution-marker pixels to get the scale factor K, then replace those marker pixels by black.
    - Ignore black cells while inferring the unique minimal repeating tile, fill every black position from that tile, and finally scale the repaired wallpaper by K in both dimensions.

    Key insight:
    - The same rare pixels play two roles at once: they must be removed as noise before periodic inference, and their total count simultaneously encodes the output magnification factor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    cleaned, scale = _clean_grid_and_scale(g)
    tile = _infer_minimal_tile(cleaned)
    repaired = _repeat_tile(tile, cleaned.shape)
    return _scale(repaired, scale)


WALLPAPER_COLORS = [1, 2, 3, 6, 7, 8, 9]
MARKER_COLORS = [4, 5]
TILE_OPTIONS = [(2, 2), (2, 3), (3, 2)]
GEN_RETRIES = 400


def _sample_tile() -> np.ndarray | None:
    tile_h, tile_w = random.choice(TILE_OPTIONS)
    max_colors = min(3, tile_h * tile_w)
    n_colors = random.randint(2, max_colors)
    palette = random.sample(WALLPAPER_COLORS, n_colors)

    tile = np.array(random.choices(palette, k=tile_h * tile_w), dtype=int).reshape(tile_h, tile_w)
    if len(set(map(int, tile.flat))) != n_colors:
        return None
    if _minimal_period(tile) != (tile_h, tile_w):
        return None
    return tile


def _sample_shape(tile: np.ndarray) -> tuple[int, int]:
    tile_h, tile_w = map(int, tile.shape)
    row_reps = [rep for rep in range(2, 6) if 6 <= tile_h * rep <= 10]
    col_reps = [rep for rep in range(2, 6) if 6 <= tile_w * rep <= 10]
    return tile_h * random.choice(row_reps), tile_w * random.choice(col_reps)


def _choose_marker_payload(total_markers: int) -> list[int]:
    if total_markers == 2:
        if random.random() < 0.5:
            color = random.choice(MARKER_COLORS)
            return [color, color]
        return random.sample(MARKER_COLORS, 2)

    doubled, single = random.sample(MARKER_COLORS, 2)
    payload = [doubled, doubled, single]
    random.shuffle(payload)
    return payload


def _phase_coverage_ok(shape: tuple[int, int], tile_shape: tuple[int, int], missing: list[tuple[int, int]]) -> bool:
    height, width = shape
    tile_h, tile_w = tile_shape
    counts = np.zeros((tile_h, tile_w), dtype=int)
    for r in range(height):
        for c in range(width):
            counts[r % tile_h, c % tile_w] += 1
    for r, c in missing:
        counts[int(r) % tile_h, int(c) % tile_w] -= 1
    return bool(np.all(counts > 0))


def _wallpaper_counts_ok(grid: np.ndarray) -> bool:
    counts = Counter(int(v) for v in grid.flat if int(v) != 0)
    wallpaper_counts = [count for count in counts.values() if count >= 3]
    if not wallpaper_counts:
        return False
    return min(wallpaper_counts) >= 3


def _attempt() -> np.ndarray | None:
    tile = _sample_tile()
    if tile is None:
        return None

    shape = _sample_shape(tile)
    full = _repeat_tile(tile, shape)
    total_markers = random.choice([2, 3])
    marker_payload = _choose_marker_payload(total_markers)
    hole_count = random.choice([1, 2])
    total_missing = total_markers + hole_count

    cells = [(r, c) for r in range(shape[0]) for c in range(shape[1])]
    random.shuffle(cells)
    chosen = cells[:total_missing]
    if len(chosen) != total_missing:
        return None
    if not _phase_coverage_ok(shape, tile.shape, chosen):
        return None

    inp = full.copy()
    marker_cells = chosen[:total_markers]
    hole_cells = chosen[total_markers:]
    for (r, c), color in zip(marker_cells, marker_payload):
        inp[r, c] = color
    for r, c in hole_cells:
        inp[r, c] = 0

    if not _wallpaper_counts_ok(inp):
        return None

    try:
        cleaned, scale = _clean_grid_and_scale(inp)
        _infer_minimal_tile(cleaned)
    except AssertionError:
        return None

    expected = _scale(full, scale)
    if expected.shape[0] > 30 or expected.shape[1] > 30:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid polluted-wallpaper instance")
