import random
import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    pts = np.argwhere(grid != 0)
    r0, r1 = int(pts[:, 0].min()), int(pts[:, 0].max())
    c0, c1 = int(pts[:, 1].min()), int(pts[:, 1].max())
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    empty_rows = [r0 + i for i in range(sub.shape[0]) if np.all(sub[i] == 0)]
    empty_cols = [c0 + j for j in range(sub.shape[1]) if np.all(sub[:, j] == 0)]

    fg = [(int(r), int(c), int(grid[r, c])) for r, c in pts]
    if empty_rows:
        sep = empty_rows[0]
        a = [(r, c, color) for r, c, color in fg if r < sep]
        b = [(r, c, color) for r, c, color in fg if r > sep]
        small, large = (a, b) if len(a) < len(b) else (b, a)
        rotated = _rotate_cells(small)
        out = np.zeros((h - 1, w), dtype=int)
        for r, c, color in large + rotated:
            rr = r if r < sep else r - 1
            out[rr, c] = color
        return out

    sep = empty_cols[0]
    a = [(r, c, color) for r, c, color in fg if c < sep]
    b = [(r, c, color) for r, c, color in fg if c > sep]
    small, large = (a, b) if len(a) < len(b) else (b, a)
    rotated = _rotate_cells(small)
    out = np.zeros((h, w - 1), dtype=int)
    for r, c, color in large + rotated:
        cc = c if c < sep else c - 1
        out[r, cc] = color
    return out


def _rotate_cells(cells: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    rs = [r for r, _, _ in cells]
    cs = [c for _, c, _ in cells]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    return [(r0 + r1 - r, c0 + c1 - c, color) for r, c, color in cells]


def generate_input() -> np.ndarray:
    colors = [1, 2, 3, 4, 6, 7]
    for _ in range(300):
        grid = np.zeros((11, 11), dtype=int)
        horizontal = random.choice([True, False])
        small_count = random.randint(7, 10)
        large_count = random.randint(11, 15)
        if horizontal:
            r0, r1, c0, c1, sep = 2, 8, 2, 8, 5
            top_rows = list(range(r0, sep))
            bot_rows = list(range(sep + 1, r1 + 1))
            if random.choice([True, False]):
                top_count, bot_count = small_count, large_count
            else:
                top_count, bot_count = large_count, small_count
            top = _random_half(top_rows, list(range(c0, c1 + 1)), top_count)
            bot = _random_half(bot_rows, list(range(c0, c1 + 1)), bot_count)
            cells = top + bot
        else:
            r0, r1, c0, c1, sep = 2, 8, 2, 8, 5
            left_cols = list(range(c0, sep))
            right_cols = list(range(sep + 1, c1 + 1))
            if random.choice([True, False]):
                left_count, right_count = small_count, large_count
            else:
                left_count, right_count = large_count, small_count
            left = _random_half(list(range(r0, r1 + 1)), left_cols, left_count)
            right = _random_half(list(range(r0, r1 + 1)), right_cols, right_count)
            cells = left + right
        if len({(r, c) for r, c in cells}) != len(cells):
            continue
        for r, c in cells:
            grid[r, c] = random.choice(colors)
        if _valid_unique_separator(grid):
            return grid
    return grid


def _random_half(rows: list[int], cols: list[int], count: int) -> list[tuple[int, int]]:
    required = {(random.choice(rows), c) for c in cols}
    required |= {(r, random.choice(cols)) for r in rows}
    all_cells = [(r, c) for r in rows for c in cols]
    random.shuffle(all_cells)
    cells = set(required)
    for p in all_cells:
        if len(cells) >= min(count, len(all_cells)):
            break
        cells.add(p)
    return list(cells)


def _valid_unique_separator(grid: np.ndarray) -> bool:
    pts = np.argwhere(grid != 0)
    r0, r1 = int(pts[:, 0].min()), int(pts[:, 0].max())
    c0, c1 = int(pts[:, 1].min()), int(pts[:, 1].max())
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    rows = sum(1 for i in range(sub.shape[0]) if np.all(sub[i] == 0))
    cols = sum(1 for j in range(sub.shape[1]) if np.all(sub[:, j] == 0))
    return rows + cols == 1
