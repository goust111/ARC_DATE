import random
from collections import deque

import numpy as np


BG = 0
LINE_COLORS = (2, 3, 4, 6)  # red, green, yellow, magenta
GEN_RETRIES = 400
BRANCHES = ((3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (4, 3))


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _line_rows(grid: np.ndarray) -> list[int]:
    rows = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if row[0] != BG and np.all(row == row[0]):
            rows.append(r)
    return rows


def _extract_seed(grid: np.ndarray, line_rows: list[int]) -> tuple[np.ndarray, int]:
    row_mask = np.ones(grid.shape[0], dtype=bool)
    row_mask[line_rows] = False
    coords = np.argwhere((grid != BG) & row_mask[:, None])
    assert coords.size > 0, "expected one seed sprite"
    assert int(coords[:, 0].max()) < int(line_rows[0]), "seed must be in the top band"

    start = tuple(map(int, coords[0]))
    allowed = {(int(r), int(c)) for r, c in coords}
    queue = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for rr, cc in _neighbors4(r, c, grid.shape[0], grid.shape[1]):
            if (rr, cc) in allowed and (rr, cc) not in seen:
                seen.add((rr, cc))
                queue.append((rr, cc))

    assert len(seen) == len(allowed), "expected a single 4-connected seed"
    seed = np.array(sorted(seen), dtype=int)
    colors = {int(grid[r, c]) for r, c in seed}
    assert len(colors) == 1, "seed must be monochrome"
    return seed, int(next(iter(colors)))


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    line_rows = _line_rows(g)
    assert 3 <= len(line_rows) <= 4, "expected 3 or 4 horizontal separator lines"

    seed, seed_color = _extract_seed(g, line_rows)
    out = g.copy()
    current = seed.copy()

    for line_row in line_rows:
        line_color = int(g[line_row, 0])
        if line_color == seed_color:
            break
        reflected = current.copy()
        reflected[:, 0] = 2 * int(line_row) - current[:, 0]
        assert np.all((0 <= reflected[:, 0]) & (reflected[:, 0] < g.shape[0]))
        out[reflected[:, 0], reflected[:, 1]] = line_color
        current = reflected

    return out


def _random_connected_mask(h: int, w: int) -> np.ndarray | None:
    area_lo = max(4, h + w - 1)
    area_hi = min(h * w - 1, h * w)
    if area_lo > area_hi:
        area_lo = max(3, min(h * w, area_lo))
        area_hi = h * w
    target_area = random.randint(area_lo, area_hi)

    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = set(_neighbors4(start[0], start[1], h, w))

    while len(cells) < target_area and frontier:
        r, c = random.choice(tuple(frontier))
        frontier.remove((r, c))
        cells.add((r, c))
        for rr, cc in _neighbors4(r, c, h, w):
            if (rr, cc) not in cells:
                frontier.add((rr, cc))

    if len(cells) != target_area:
        return None

    rows = {r for r, _c in cells}
    cols = {c for _r, c in cells}
    if 0 not in rows or h - 1 not in rows or 0 not in cols or w - 1 not in cols:
        return None

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _choose_band_heights(
    num_lines: int,
    stop_idx: int,
    sprite_h: int,
    dmin0: int,
    dmax0: int,
) -> list[int] | None:
    band_count = num_lines - 1
    max_height = sprite_h + 5

    def dfs(i: int, used: set[int], dmin: int, dmax: int) -> list[int] | None:
        if i == band_count:
            return []

        lower = sprite_h
        if i < stop_idx:
            lower = max(lower, dmax)
        candidates = list(range(lower, max_height + 1))
        random.shuffle(candidates)
        for band_h in candidates:
            if band_h in used:
                continue
            if i < stop_idx:
                next_dmin = band_h + 1 - dmax
                next_dmax = band_h + 1 - dmin
                if next_dmin < 1:
                    continue
                tail = dfs(i + 1, used | {band_h}, next_dmin, next_dmax)
            else:
                tail = dfs(i + 1, used | {band_h}, dmin, dmax)
            if tail is not None:
                return [band_h] + tail
        return None

    return dfs(0, set(), dmin0, dmax0)


def _simulate_reflections(
    seed_coords: np.ndarray, line_rows: list[int], stop_idx: int
) -> list[np.ndarray]:
    copies: list[np.ndarray] = []
    current = seed_coords.copy()
    for i in range(stop_idx):
        reflected = current.copy()
        reflected[:, 0] = 2 * int(line_rows[i]) - current[:, 0]
        copies.append(reflected)
        current = reflected
    return copies


def _gen_attempt() -> np.ndarray | None:
    num_lines, stop_idx = random.choice(BRANCHES)

    max_h = 4 if num_lines == 4 else 5
    sprite_h = random.randint(2, max_h)
    sprite_w = random.randint(2, 5)
    mask = _random_connected_mask(sprite_h, sprite_w)
    if mask is None:
        return None

    top_band = random.randint(sprite_h, sprite_h + 4)
    max_dmin = top_band - sprite_h + 1
    dmin0 = random.randint(1, max_dmin)
    dmax0 = dmin0 + sprite_h - 1

    band_heights = _choose_band_heights(num_lines, stop_idx, sprite_h, dmin0, dmax0)
    if band_heights is None:
        return None

    bottom_band = random.randint(sprite_h, sprite_h + 3)
    height = top_band + num_lines + sum(band_heights) + bottom_band
    if not (18 <= height <= 32):
        return None

    width = random.randint(max(14, sprite_w + 4), 18)
    grid = np.zeros((height, width), dtype=int)

    line_rows = []
    row = top_band
    line_rows.append(row)
    for band_h in band_heights:
        row += band_h + 1
        line_rows.append(row)

    palette = list(random.sample(LINE_COLORS, num_lines))
    seed_color = palette[stop_idx]

    left = random.randint(1, width - sprite_w - 1)
    top = line_rows[0] - dmax0
    seed_coords = np.argwhere(mask).astype(int)
    seed_coords[:, 0] += top
    seed_coords[:, 1] += left

    copies = _simulate_reflections(seed_coords, line_rows, stop_idx)
    for i, coords in enumerate(copies):
        if np.any(coords[:, 0] <= line_rows[i]):
            return None
        if i + 1 < num_lines and np.any(coords[:, 0] >= line_rows[i + 1]):
            return None

    grid[line_rows, :] = np.array(palette, dtype=int)[:, None]
    grid[seed_coords[:, 0], seed_coords[:, 1]] = seed_color

    try:
        out = main(grid)
    except AssertionError:
        return None

    expected = grid.copy()
    for i, coords in enumerate(copies):
        expected[coords[:, 0], coords[:, 1]] = palette[i]
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chained-reflection puzzle")
