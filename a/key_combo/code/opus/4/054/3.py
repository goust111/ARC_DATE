from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox, crop_bbox, get_bg, tile_fill


OBSTACLE = Color.ORANGE
BG = Color.BLACK
NON_SPECIAL_COLORS = tuple(c for c in Color.NOT_BLACK if c != OBSTACLE)
MAX_GENERATION_ATTEMPTS = 300


def _contiguous_runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(mask_1d.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((int(start), int(idx - 1)))
            start = None
    if start is not None:
        runs.append((int(start), int(len(mask_1d) - 1)))
    return runs


def _extract_parameters(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert int(get_bg(g)) == BG, "expected a black background"

    orange_rows = np.where(np.any(g == OBSTACLE, axis=1))[0]
    assert orange_rows.size >= 1, "expected at least one obstacle"
    assert np.unique(orange_rows).size == 1, "expected all obstacles on one row"
    seed_r = int(orange_rows[0])

    seed_cols = np.where((g[seed_r] != BG) & (g[seed_r] != OBSTACLE))[0]
    assert seed_cols.size == 1, "expected exactly one seed pixel on the obstacle row"
    seed_c = int(seed_cols[0])

    obstacle_cols = sorted(int(c) for c in np.where(g[seed_r] == OBSTACLE)[0] if int(c) > seed_c)
    assert obstacle_cols, "expected at least one obstacle to the right of the seed"
    assert np.all(g[seed_r, seed_c + 1 : obstacle_cols[0]] == BG), "expected empty space before the first obstacle"

    d = None
    hits = 0
    cursor = seed_c + 1
    width = int(g.shape[1])
    while cursor < width:
        rel = np.where(g[seed_r, cursor:] == OBSTACLE)[0]
        if rel.size == 0:
            break
        hit_c = int(cursor + rel[0])
        if d is None:
            d = int(hit_c - seed_c - 1)
        hits += 1
        cursor = hit_c + 1

    assert d is not None and d >= 1

    lower_mask = np.zeros_like(g, dtype=bool)
    if seed_r + 1 < g.shape[0]:
        lower_mask[seed_r + 1 :, :] = (g[seed_r + 1 :, :] != BG) & (g[seed_r + 1 :, :] != OBSTACLE)
    tile_box = bbox(lower_mask)
    assert tile_box is not None, "expected a lower template tile"
    r0, c0, r1, c1 = tile_box
    assert c0 == 0, "expected the template to be left-aligned"
    assert r0 > seed_r, "expected the template below the ray row"
    assert np.all(g[seed_r + 1 : r0, :] == BG), "expected a black separator band between ray and template"

    active_rows = np.any(lower_mask, axis=1)
    row_runs = _contiguous_runs(active_rows)
    assert len(row_runs) == 1 and row_runs[0] == (r0, r1), "expected one contiguous template row band"

    tile = crop_bbox(g, tile_box)
    assert tile.shape[0] >= 1 and tile.shape[1] >= 1
    assert not np.any((tile == BG) | (tile == OBSTACLE)), "template tile must be fully non-black and non-orange"
    return int(d), int(hits), tile


def main(grid: np.ndarray) -> np.ndarray:
    """Encode a ray's collision distance and collision count as the size of a tiled template output.

    Input structure:
    - The background is black.
    - One upper row contains a single colored seed pixel and several orange obstacle pixels to its right.
    - A lower rectangular template tile, separated by at least one black row, is filled with non-black non-orange colors.

    Rule:
    - Start from the seed and cast a ray to the right along the obstacle row.
    - Each time the ray hits an orange obstacle, count that collision, stop that segment, skip over the obstacle, and continue rightward.
    - Let `d` be the number of empty cells between the seed and the first obstacle, and let `n` be the total number of collisions.
    - Output a new `n x d` grid obtained by tiling the lower template from the top-left and clipping any excess on the right or bottom.

    Key insight:
    - The upper structure is only a parameter extractor: the first-hit spacing sets output width, the total hit count sets output height, and the lower tile supplies the repeated visual content.
    """
    d, n, tile = _extract_parameters(np.asarray(grid, dtype=int))
    return tile_fill((n, d), tile)


def _horizontal_period(tile: np.ndarray) -> int:
    h, w = tile.shape
    for p in range(1, w + 1):
        if w % p == 0 and np.array_equal(tile, np.tile(tile[:, :p], (1, w // p))):
            return int(p)
    return int(w)


def _vertical_period(tile: np.ndarray) -> int:
    h, w = tile.shape
    for p in range(1, h + 1):
        if h % p == 0 and np.array_equal(tile, np.tile(tile[:p, :], (h // p, 1))):
            return int(p)
    return int(h)


def _sample_tile(tile_h: int, tile_w: int, seed_color: int) -> np.ndarray | None:
    color_count = random.randint(2, min(3, tile_h * tile_w))
    if random.random() < 0.6:
        remaining = [c for c in NON_SPECIAL_COLORS if c != seed_color]
        chosen = [int(seed_color)] + random.sample(remaining, color_count - 1)
    else:
        chosen = random.sample(list(NON_SPECIAL_COLORS), color_count)
    random.shuffle(chosen)

    cells = tile_h * tile_w
    flat = chosen[:]
    flat.extend(random.choice(chosen) for _ in range(cells - len(chosen)))
    random.shuffle(flat)
    tile = np.array(flat, dtype=int).reshape(tile_h, tile_w)

    if len(set(map(int, tile.ravel()))) != color_count:
        return None
    if _horizontal_period(tile) != tile_w:
        return None
    if _vertical_period(tile) != tile_h:
        return None
    return tile


def _build_grid(
    top_h: int,
    sep_h: int,
    seed_r: int,
    seed_color: int,
    obstacle_cols: list[int],
    tile: np.ndarray,
    width: int,
) -> np.ndarray:
    tile_h, tile_w = map(int, tile.shape)
    grid = np.full((top_h + sep_h + tile_h, width), BG, dtype=int)
    grid[seed_r, 0] = int(seed_color)
    for col in obstacle_cols:
        grid[seed_r, int(col)] = OBSTACLE
    grid[top_h + sep_h :, :tile_w] = tile
    return grid


def _expected_from_layout(obstacle_cols: list[int], tile: np.ndarray) -> np.ndarray:
    d = int(obstacle_cols[0] - 1)
    n = int(len(obstacle_cols))
    return tile_fill((n, d), tile)


def _mutate_first_obstacle(grid: np.ndarray) -> np.ndarray | None:
    g = np.array(grid, copy=True)
    ray_rows = np.where(np.any(g == OBSTACLE, axis=1))[0]
    if ray_rows.size != 1:
        return None
    row = int(ray_rows[0])
    seed_c = int(np.where((g[row] != BG) & (g[row] != OBSTACLE))[0][0])
    cols = sorted(int(c) for c in np.where(g[row] == OBSTACLE)[0])
    if len(cols) < 2:
        return None
    first, second = cols[0], cols[1]
    for delta in (-1, 1):
        candidate = first + delta
        new_d = candidate - seed_c - 1
        if not (2 <= new_d <= 5):
            continue
        if candidate <= seed_c or candidate >= second:
            continue
        if second - candidate < 3:
            continue
        if g[row, candidate] != BG:
            continue
        g[row, first] = BG
        g[row, candidate] = OBSTACLE
        return g
    return None


def _mutate_remove_last_obstacle(grid: np.ndarray) -> np.ndarray | None:
    g = np.array(grid, copy=True)
    ray_rows = np.where(np.any(g == OBSTACLE, axis=1))[0]
    if ray_rows.size != 1:
        return None
    row = int(ray_rows[0])
    cols = sorted(int(c) for c in np.where(g[row] == OBSTACLE)[0])
    if len(cols) < 2:
        return None
    g[row, cols[-1]] = BG
    return g


def _semantic_checks(grid: np.ndarray, obstacle_cols: list[int], tile: np.ndarray) -> bool:
    expected = _expected_from_layout(obstacle_cols, tile)
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    shifted = _mutate_first_obstacle(grid)
    if shifted is not None:
        shifted_out = main(shifted)
        if shifted_out.shape == actual.shape and np.array_equal(shifted_out, actual):
            return False

    removed = _mutate_remove_last_obstacle(grid)
    if removed is not None:
        removed_out = main(removed)
        if removed_out.shape[0] != actual.shape[0] - 1:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    top_h = random.randint(2, 3)
    sep_h = random.randint(1, 2)
    seed_r = random.randint(0, top_h - 1)
    seed_color = random.choice(NON_SPECIAL_COLORS)

    d = random.randint(2, 5)
    obstacle_count = random.randint(2, 4)
    obstacle_cols = [d + 1]
    for _ in range(obstacle_count - 1):
        obstacle_cols.append(obstacle_cols[-1] + random.randint(4, 5))

    if random.random() < 0.6 and d >= 3:
        tile_w = random.randint(2, d - 1)
    else:
        tile_w = random.randint(2, min(3, d))

    if random.random() < 0.6 and obstacle_count >= 3:
        tile_h = 2
    else:
        tile_h = random.randint(2, min(3, obstacle_count))

    tile = _sample_tile(tile_h, tile_w, seed_color)
    if tile is None:
        return None

    right_margin = random.randint(1, 3)
    width = max(tile_w, obstacle_cols[-1] + 1 + right_margin)
    if width > 30:
        return None

    grid = _build_grid(top_h, sep_h, seed_r, seed_color, obstacle_cols, tile, width)
    if max(grid.shape) > 30:
        return None

    if not _semantic_checks(grid, obstacle_cols, tile):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid collision-distance tiling input")
