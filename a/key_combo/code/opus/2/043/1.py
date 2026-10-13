import random
from collections import deque

import numpy as np

BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _detect_separator_col(grid: np.ndarray) -> int:
    h, w = grid.shape
    best_col = w // 2
    best_hits = -1
    for c in range(w):
        hits = int(np.sum(grid[:, c] == RED))
        if hits > best_hits:
            best_hits = hits
            best_col = c
    return best_col


def _flood_black_region(grid: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = grid.shape
    sr, sc = start
    if not _in_bounds((h, w), sr, sc):
        return []
    q = deque([(sr, sc)])
    visited = set()
    region: list[tuple[int, int]] = []
    while q:
        r, c = q.pop()
        if not _in_bounds((h, w), r, c):
            continue
        if (r, c) in visited:
            continue
        if (r, c) != start and int(grid[r, c]) != BLACK:
            continue
        visited.add((r, c))
        region.append((r, c))
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) not in visited:
                q.append((nr, nc))
    return region


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int, copy=True)
    h, w = grid.shape
    c_sep = _detect_separator_col(grid)
    out = grid.copy()
    flooded = np.zeros((h, w), dtype=bool)

    sources: list[tuple[int, int, int]] = []
    for r in range(h):
        for c in range(c_sep):
            color = int(grid[r, c])
            if color in (BLACK, RED):
                continue
            sources.append((r, c, color))

    sources.sort(key=lambda x: (x[0], x[1]))
    for r, c, color in sources:
        tr, tc = r, 2 * c_sep - c
        if not _in_bounds((h, w), tr, tc):
            continue
        if tc == c_sep:
            continue
        target = int(out[tr, tc])
        if target == GREEN:
            region = _flood_black_region(out, (tr, tc))
            if any(flooded[rr, cc] for rr, cc in region):
                continue
            for rr, cc in region:
                out[rr, cc] = color
                flooded[rr, cc] = True
        elif target == BLACK:
            if not flooded[tr, tc]:
                out[tr, tc] = color

    return out


def _build_left_sources(h: int, w: int, sep: int, target_count: int) -> list[tuple[int, int]]:
    left_cols = list(range(1, max(1, sep - 1)))
    left_rows = list(range(1, h - 1))
    positions: list[tuple[int, int]] = []
    tries = 0
    while len(positions) < target_count and tries < 5000:
        tries += 1
        r = random.choice(left_rows)
        c = random.choice(left_cols)
        if (r, c) in positions:
            continue
        ok = True
        for rr, cc in positions:
            if abs(rr - r) + abs(cc - c) <= 1:
                ok = False
                break
        if not ok:
            continue
        tr = 2 * sep - c
        if not (0 <= tr < w):
            continue
        positions.append((r, c))
    if len(positions) < target_count:
        raise RuntimeError("failed to place unique mirrored sources")
    return positions


def generate_input() -> np.ndarray:
    h = 15
    w = 15
    palette = [BLUE, YELLOW, ORANGE, TEAL, MAROON]
    while True:
        sep = random.randint(4, 10)
        g = np.full((h, w), BLACK, dtype=int)
        g[:, sep] = RED

        src_n = random.randint(3, 5)
        src_positions = _build_left_sources(h, w, sep, src_n)
        random.shuffle(src_positions)

        sources: list[tuple[int, int, int]] = []
        for i, (r, c) in enumerate(src_positions):
            color = random.choice(palette)
            sources.append((r, c, color))
            g[r, c] = color

        target_map: dict[tuple[int, int], int] = {}
        for r, c, color in sources:
            target_map[(r, 2 * sep - c)] = color

        max_seeds = min(3, src_n - 1)
        if max_seeds <= 0:
            continue
        n_seed = random.randint(1, max_seeds)
        seed_positions = set(random.sample(range(len(sources)), n_seed))

        for idx, (r, c, color) in enumerate(sources):
            tr, tc = r, 2 * sep - c
            if not _in_bounds((h, w), tr, tc):
                break
            if idx in seed_positions:
                g[tr, tc] = GREEN
            # non-seed remains black until copy step

        # avoid pathological invalids
        if sum((idx in seed_positions) for idx in range(len(sources))) == 0:
            continue

        if np.sum(g[:, sep + 1 :] == GREEN) != n_seed:
            continue
        return g


_generate_input_single = generate_input


def generate_inputs() -> list[np.ndarray]:
    outputs: list[np.ndarray] = []
    attempts = 0
    while len(outputs) < 20 and attempts < 200:
        attempts += 1
        g = _generate_input_single()
        arr = np.array(g, dtype=int, copy=True)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        if arr.ndim != 2 or arr.shape[0] > 30 or arr.shape[1] > 30:
            continue
        outputs.append(arr)

    while len(outputs) < 20:
        if outputs:
            outputs.append(outputs[-1].copy())
        else:
            outputs.append(np.zeros((1, 1), dtype=int))
    return outputs


if __name__ == "__main__":
    random.seed(0)
    print(main(generate_input()[0]).shape)
