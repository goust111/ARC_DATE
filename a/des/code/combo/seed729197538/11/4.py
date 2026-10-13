from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
MAROON = 9

TOWER_COLORS = (RED, BLUE, YELLOW)
COLOR_TO_BIT = {RED: 1, BLUE: 2, YELLOW: 4}
BIT_TO_MIX = {
    0: BLACK,
    1: RED,
    2: BLUE,
    3: MAGENTA,
    4: YELLOW,
    5: ORANGE,
    6: GREEN,
    7: MAROON,
}
MIX_COLORS = {GREEN, MAGENTA, ORANGE, MAROON}

LAYOUTS = ("vertical_only", "horizontal_only", "cross")
GRID_MIN, GRID_MAX = 10, 20
ONE_ROOM_MIN, ONE_ROOM_MAX = 8, 14
MULTI_ROOM_MIN, MULTI_ROOM_MAX = 2, 6
GEN_RETRIES = 400


def _cast_ray(
    grid: np.ndarray,
    covered: np.ndarray,
    r: int,
    c: int,
    dr: int,
    dc: int,
    bit: int,
) -> None:
    h, w = grid.shape
    while 0 <= r < h and 0 <= c < w:
        if int(grid[r, c]) == GREY:
            break
        if int(grid[r, c]) == BLACK:
            covered[r, c] |= bit
        r += dr
        c += dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast inward rays from edge towers, stop them at grey walls, and recolor each
    black pixel by the set of tower colors that reaches it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 3 and w >= 3, "grid must have an interior"

    covered = np.zeros_like(g, dtype=np.uint8)

    for c in range(w):
        color = int(g[0, c])
        if color in COLOR_TO_BIT:
            _cast_ray(g, covered, 1, c, 1, 0, COLOR_TO_BIT[color])
        color = int(g[h - 1, c])
        if color in COLOR_TO_BIT:
            _cast_ray(g, covered, h - 2, c, -1, 0, COLOR_TO_BIT[color])

    for r in range(h):
        color = int(g[r, 0])
        if color in COLOR_TO_BIT:
            _cast_ray(g, covered, r, 1, 0, 1, COLOR_TO_BIT[color])
        color = int(g[r, w - 1])
        if color in COLOR_TO_BIT:
            _cast_ray(g, covered, r, w - 2, 0, -1, COLOR_TO_BIT[color])

    out = g.copy()
    bg_mask = g == BLACK
    for bits, color in BIT_TO_MIX.items():
        out[bg_mask & (covered == bits)] = color
    return out


def _sample_room_sizes(room_count: int) -> list[int] | None:
    for _ in range(120):
        if room_count == 1:
            sizes = [random.randint(ONE_ROOM_MIN, ONE_ROOM_MAX)]
        else:
            sizes = [random.randint(MULTI_ROOM_MIN, MULTI_ROOM_MAX) for _ in range(room_count)]
        total = sum(sizes) + (room_count - 1) + 2
        if GRID_MIN <= total <= GRID_MAX:
            return sizes
    return None


def _bands_and_walls(room_sizes: list[int]) -> tuple[list[tuple[int, int]], list[int]]:
    bands: list[tuple[int, int]] = []
    walls: list[int] = []
    start = 1
    for i, size in enumerate(room_sizes):
        end = start + size - 1
        bands.append((start, end))
        if i + 1 < len(room_sizes):
            walls.append(end + 1)
            start = end + 2
    return bands, walls


def _edge_positions(band: tuple[int, int]) -> list[int]:
    lo, hi = band
    return list(range(lo, hi + 1))


def _add_tower(towers: dict[str, dict[int, int]], side: str, idx: int, color: int) -> bool:
    if idx in towers[side]:
        return towers[side][idx] == color
    towers[side][idx] = color
    return True


def _random_band_position(
    towers: dict[str, dict[int, int]],
    side: str,
    band: tuple[int, int],
) -> int | None:
    options = [idx for idx in _edge_positions(band) if idx not in towers[side]]
    if not options:
        return None
    return random.choice(options)


def _render_input(
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    row_walls: list[int],
    col_walls: list[int],
    towers: dict[str, dict[int, int]],
) -> np.ndarray:
    h = row_bands[-1][1] + 2
    w = col_bands[-1][1] + 2
    grid = np.full((h, w), BLACK, dtype=int)

    for r in row_walls:
        grid[r, 1 : w - 1] = GREY
    for c in col_walls:
        grid[1 : h - 1, c] = GREY

    for c, color in towers["top"].items():
        grid[0, c] = color
    for c, color in towers["bottom"].items():
        grid[h - 1, c] = color
    for r, color in towers["left"].items():
        grid[r, 0] = color
    for r, color in towers["right"].items():
        grid[r, w - 1] = color
    return grid


def _add_sparse_extras(
    towers: dict[str, dict[int, int]],
    side: str,
    bands: list[tuple[int, int]],
) -> None:
    for band in bands:
        if random.random() >= 0.55:
            continue
        idx = _random_band_position(towers, side, band)
        if idx is None:
            continue
        towers[side][idx] = random.choice(TOWER_COLORS)


def _sample_layout() -> tuple[str, int, int]:
    layout = random.choice(LAYOUTS)
    if layout == "vertical_only":
        return layout, 1, random.randint(2, 4)
    if layout == "horizontal_only":
        return layout, random.randint(2, 4), 1
    return layout, random.randint(2, 3), random.randint(2, 3)


def _forced_towers(
    layout: str,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
) -> dict[str, dict[int, int]] | None:
    towers = {"top": {}, "bottom": {}, "left": {}, "right": {}}

    if layout == "vertical_only":
        side = random.choice(("left", "right"))
        target_band = col_bands[0] if side == "left" else col_bands[-1]
        c = random.choice(_edge_positions(target_band))
        r = random.choice(_edge_positions(row_bands[0]))
        top_color, bottom_color, side_color = random.sample(TOWER_COLORS, 3)
        if not _add_tower(towers, "top", c, top_color):
            return None
        if not _add_tower(towers, "bottom", c, bottom_color):
            return None
        if not _add_tower(towers, side, r, side_color):
            return None
        return towers

    if layout == "horizontal_only":
        side = random.choice(("top", "bottom"))
        target_band = row_bands[0] if side == "top" else row_bands[-1]
        r = random.choice(_edge_positions(target_band))
        c = random.choice(_edge_positions(col_bands[0]))
        left_color, right_color, side_color = random.sample(TOWER_COLORS, 3)
        if not _add_tower(towers, "left", r, left_color):
            return None
        if not _add_tower(towers, "right", r, right_color):
            return None
        if not _add_tower(towers, side, c, side_color):
            return None
        return towers

    vertical_side = random.choice(("top", "bottom"))
    horizontal_side = random.choice(("left", "right"))
    target_row_band = row_bands[0] if vertical_side == "top" else row_bands[-1]
    target_col_band = col_bands[0] if horizontal_side == "left" else col_bands[-1]
    r = random.choice(_edge_positions(target_row_band))
    c = random.choice(_edge_positions(target_col_band))
    color_a, color_b = random.sample(TOWER_COLORS, 2)
    if not _add_tower(towers, vertical_side, c, color_a):
        return None
    if not _add_tower(towers, horizontal_side, r, color_b):
        return None
    return towers


def _valid_sample(grid: np.ndarray, layout: str) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    if np.array_equal(out, grid):
        return False
    if not np.any(np.isin(out, tuple(MIX_COLORS))):
        return False
    if layout in ("vertical_only", "horizontal_only") and not np.any(out == MAROON):
        return False
    if layout == "cross" and np.any(out == MAROON):
        return False
    changed = int(np.count_nonzero(out != grid))
    if changed < 8:
        return False
    return True


def _build_candidate() -> np.ndarray | None:
    layout, row_room_count, col_room_count = _sample_layout()
    row_sizes = _sample_room_sizes(row_room_count)
    col_sizes = _sample_room_sizes(col_room_count)
    if row_sizes is None or col_sizes is None:
        return None

    row_bands, row_walls = _bands_and_walls(row_sizes)
    col_bands, col_walls = _bands_and_walls(col_sizes)
    towers = _forced_towers(layout, row_bands, col_bands)
    if towers is None:
        return None

    _add_sparse_extras(towers, "top", col_bands)
    _add_sparse_extras(towers, "bottom", col_bands)
    _add_sparse_extras(towers, "left", row_bands)
    _add_sparse_extras(towers, "right", row_bands)

    grid = _render_input(row_bands, col_bands, row_walls, col_walls, towers)
    if _valid_sample(grid, layout):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
