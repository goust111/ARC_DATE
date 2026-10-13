from __future__ import annotations

import random

import numpy as np


BLACK = 0
MAROON = 9
PALETTE = tuple(range(1, 9))
DIGITS = {
    1: np.array(
        [
            [0, 9, 0],
            [9, 9, 0],
            [0, 9, 0],
            [0, 9, 0],
            [9, 9, 9],
        ],
        dtype=int,
    ),
    2: np.array(
        [
            [9, 9, 9],
            [0, 0, 9],
            [9, 9, 9],
            [9, 0, 0],
            [9, 9, 9],
        ],
        dtype=int,
    ),
    3: np.array(
        [
            [9, 9, 9],
            [0, 0, 9],
            [0, 9, 9],
            [0, 0, 9],
            [9, 9, 9],
        ],
        dtype=int,
    ),
}


def _divisors(n: int) -> list[int]:
    out: list[int] = []
    for d in range(1, int(n ** 0.5) + 1):
        if n % d != 0:
            continue
        out.append(d)
        if d * d != n:
            out.append(n // d)
    return sorted(out)


def _majority_tile(area: np.ndarray, tile_h: int, tile_w: int) -> tuple[np.ndarray, int]:
    area_h, area_w = area.shape
    n_h = area_h // tile_h
    n_w = area_w // tile_w
    tiles = (
        area.reshape(n_h, tile_h, n_w, tile_w)
        .transpose(0, 2, 1, 3)
        .reshape(n_h * n_w, tile_h, tile_w)
    )

    tile = np.zeros((tile_h, tile_w), dtype=int)
    mismatch = 0
    color_count = max(10, int(area.max()) + 1)
    for r in range(tile_h):
        for c in range(tile_w):
            values = tiles[:, r, c]
            counts = np.bincount(values, minlength=color_count)
            color = int(np.argmax(counts))
            tile[r, c] = color
            mismatch += int(values.size - counts[color])
    return tile, mismatch


def _solve_tiled_area(area: np.ndarray) -> tuple[tuple[int, ...], np.ndarray] | None:
    area_h, area_w = area.shape
    best: tuple[tuple[int, ...], np.ndarray] | None = None
    for tile_h in _divisors(area_h):
        for tile_w in _divisors(area_w):
            n_h = area_h // tile_h
            n_w = area_w // tile_w
            if n_h == 1 and n_w == 1:
                continue

            tile, mismatch = _majority_tile(area, tile_h, tile_w)
            prefer_2d = 0 if n_h >= 2 and n_w >= 2 else 1
            key = (mismatch, prefer_2d, tile_h * tile_w, tile_h, tile_w, -(area_h * area_w))
            if best is None or key < best[0]:
                best = (key, tile)
    return best


def _candidate_areas(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    candidates: list[np.ndarray] = []

    for cut in range(1, w):
        area = grid[:, :cut]
        strip = grid[:, cut:]
        if np.any(area == MAROON):
            continue
        if np.any(strip == MAROON) and np.all((strip == BLACK) | (strip == MAROON)):
            candidates.append(area)

    for cut in range(1, h):
        area = grid[:cut, :]
        strip = grid[cut:, :]
        if np.any(area == MAROON):
            continue
        if np.any(strip == MAROON) and np.all((strip == BLACK) | (strip == MAROON)):
            candidates.append(area)

    if not candidates and not np.any(grid == MAROON):
        candidates.append(grid)
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    best: tuple[tuple[int, ...], np.ndarray] | None = None
    for area in _candidate_areas(g):
        solved = _solve_tiled_area(area)
        if solved is None:
            continue
        if best is None or solved[0] < best[0]:
            best = solved
    assert best is not None
    return best[1]


def _fundamental_period(tile: np.ndarray) -> tuple[int, int]:
    h, w = tile.shape
    for ph in _divisors(h):
        for pw in _divisors(w):
            if ph == h and pw == w:
                return h, w
            base = tile[:ph, :pw]
            repeated = np.tile(base, (h // ph, w // pw))
            if np.array_equal(repeated, tile):
                return ph, pw
    return h, w


def _random_tile(tile_h: int, tile_w: int) -> np.ndarray:
    for _ in range(400):
        colors = random.sample(PALETTE, random.randint(2, 4))
        tile = np.full((tile_h, tile_w), BLACK, dtype=int)
        fill_p = random.uniform(0.45, 0.8)
        for r in range(tile_h):
            for c in range(tile_w):
                if random.random() < fill_p:
                    tile[r, c] = random.choice(colors)

        if not np.any(tile != BLACK):
            continue
        if np.any(np.all(tile == BLACK, axis=1)) or np.any(np.all(tile == BLACK, axis=0)):
            continue
        if len(set(int(v) for v in tile.flat if int(v) != BLACK)) < 2:
            continue
        if _fundamental_period(tile) != (tile_h, tile_w):
            continue
        return tile
    raise RuntimeError("failed to sample a full-period tile")


def _apply_noise(tile: np.ndarray, n_h: int, n_w: int, max_noise: int) -> np.ndarray:
    tile_h, tile_w = tile.shape
    area = np.tile(tile, (n_h, n_w))
    total_noise = 0
    all_colors = (BLACK,) + PALETTE

    for tr in range(n_h):
        for tc in range(n_w):
            count = random.randint(0, max_noise)
            if total_noise == 0 and tr == n_h - 1 and tc == n_w - 1:
                count = max(1, count)
            positions = random.sample(range(tile_h * tile_w), count)
            for pos in positions:
                rr = pos // tile_w
                cc = pos % tile_w
                old = int(tile[rr, cc])
                choices = [color for color in all_colors if color != old]
                area[tr * tile_h + rr, tc * tile_w + cc] = random.choice(choices)
                total_noise += 1
    return area


def _paste_digit(canvas: np.ndarray, digit: np.ndarray, start_r: int, start_c: int) -> None:
    patch = canvas[start_r : start_r + digit.shape[0], start_c : start_c + digit.shape[1]]
    patch[digit == MAROON] = MAROON


def _build_input(tile: np.ndarray, n_h: int, n_w: int, max_noise: int, side: str) -> np.ndarray:
    tiled = _apply_noise(tile, n_h, n_w, max_noise)
    area_h, area_w = tiled.shape
    digit = DIGITS[max_noise]
    digit_h, digit_w = digit.shape

    if side == "right":
        extra_w = digit_w + random.randint(0, 2)
        grid = np.full((area_h, area_w + extra_w), BLACK, dtype=int)
        grid[:, :area_w] = tiled
        start_r = random.randint(0, area_h - digit_h)
        _paste_digit(grid, digit, start_r, area_w)
        return grid

    extra_h = digit_h + random.randint(0, 2)
    grid = np.full((area_h + extra_h, area_w), BLACK, dtype=int)
    grid[:area_h, :] = tiled
    start_c = random.randint(0, area_w - digit_w)
    _paste_digit(grid, digit, area_h, start_c)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        tile_h = random.randint(3, 6)
        tile_w = random.randint(3, 6)
        n_h = random.randint(2, 4)
        n_w = random.randint(2, 4)
        max_noise = random.randint(1, 3)
        side = random.choice(("right", "bottom"))

        tile = _random_tile(tile_h, tile_w)
        grid = _build_input(tile, n_h, n_w, max_noise, side)
        if np.array_equal(main(grid), tile):
            return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
