import random
from collections import deque

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
BG_COLOR = 8
MAX_RETRIES = 500
PATH_ATTEMPTS = 250


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _legend_colors(grid: np.ndarray, bg: int) -> list[int]:
    bottom = grid[-1]
    colors = [int(v) for v in bottom if int(v) != bg]
    assert colors, "expected a bottom-row legend"
    return colors


def _components_of_color(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for sr, sc in np.argwhere(mask):
        r0, c0 = int(sr), int(sc)
        if seen[r0, c0]:
            continue
        q: deque[tuple[int, int]] = deque([(r0, c0)])
        seen[r0, c0] = True
        comp: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each connected path by looking up its total length in the bottom legend.

    Input structure:
    - The background is the dominant azure color.
    - Above the bottom row are one-pixel-wide horizontal and vertical colored path segments.
    - The bottom row is a legend made of isolated color markers read from left to right.

    Rule:
    - Ignore the bottom legend row when finding paths, and erase that row in the output.
    - For each original foreground color, split its cells into 4-connected path components.
    - Count the pixels in each component; lengths 1-2 use the first legend color,
      lengths 3-4 use the second, lengths 5-6 use the third, and so on.
    - Repaint every cell of each component with the looked-up legend color.

    Key insight:
    The path's original color only groups connected cells; the output color is determined
    after connecting the segments and counting the full component length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, _w = g.shape
    assert h >= 2

    bg = _background_color(g)
    legend = _legend_colors(g, bg)
    field = g[:-1]

    out = g.copy()
    out[-1, :] = bg

    path_colors = [int(c) for c in np.unique(field) if int(c) != bg]
    for color in path_colors:
        mask = field == color
        for comp in _components_of_color(mask):
            length = len(comp)
            legend_i = (length - 1) // 2
            assert 0 <= legend_i < len(legend), "path length exceeds the legend"
            new_color = int(legend[legend_i])
            for r, c in comp:
                out[r, c] = new_color

    return out


def _composition(total: int, parts: int) -> list[int]:
    shares = [0] * parts
    for _ in range(total):
        shares[random.randrange(parts)] += 1
    return shares


def _legend_positions(width: int, count: int) -> list[int]:
    min_width = 2 * count - 1
    assert width >= min_width
    gaps = _composition(width - min_width, count + 1)
    cols: list[int] = []
    c = gaps[0]
    for i in range(count):
        cols.append(c)
        c += 1
        if i < count - 1:
            c += 1 + gaps[i + 1]
    return cols


def _clear_of_occupied(r: int, c: int, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    if not (0 <= r < h and 0 <= c < w):
        return False
    r0, r1 = max(0, r - 1), min(h, r + 2)
    c0, c1 = max(0, c - 1), min(w, c + 2)
    return not bool(occupied[r0:r1, c0:c1].any())


def _touches_used_except_current(
    r: int, c: int, used: set[tuple[int, int]], current: tuple[int, int]
) -> bool:
    for dr, dc in DIR4:
        nb = (r + dr, c + dc)
        if nb in used and nb != current:
            return True
    return False


def _ordered_dirs(last_dir: tuple[int, int] | None) -> list[tuple[int, int]]:
    dirs = list(DIR4)
    random.shuffle(dirs)
    if last_dir is not None and random.random() < 0.70:
        dirs.sort(key=lambda d: 0 if d == last_dir else 1)
    return dirs


def _grow_path(
    height: int, width: int, length: int, occupied: np.ndarray
) -> list[tuple[int, int]] | None:
    for _ in range(PATH_ATTEMPTS):
        start = (random.randrange(height), random.randrange(width))
        if not _clear_of_occupied(start[0], start[1], occupied):
            continue

        cells = [start]
        used = {start}
        last_dir: tuple[int, int] | None = None

        while len(cells) < length:
            r, c = cells[-1]
            placed = False
            for dr, dc in _ordered_dirs(last_dir):
                rr, cc = r + dr, c + dc
                if (rr, cc) in used:
                    continue
                if not _clear_of_occupied(rr, cc, occupied):
                    continue
                if _touches_used_except_current(rr, cc, used, (r, c)):
                    continue
                cells.append((rr, cc))
                used.add((rr, cc))
                last_dir = (dr, dc)
                placed = True
                break
            if not placed:
                break

        if len(cells) == length:
            return cells
    return None


def _path_lengths(legend_count: int) -> list[int]:
    lengths = [2 * i + 1 + random.randrange(2) for i in range(legend_count)]
    for _ in range(random.randint(0, 3)):
        i = random.randrange(legend_count)
        lengths.append(2 * i + 1 + random.randrange(2))
    random.shuffle(lengths)
    return lengths


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(14, 24)
    width = random.randint(14, 28)
    legend_count = random.randint(3, 5)
    lengths = _path_lengths(legend_count)

    available = [c for c in range(10) if c != BG_COLOR]
    random.shuffle(available)
    max_path_colors = min(4, len(lengths), len(available) - legend_count)
    if max_path_colors < 2:
        return None
    path_color_count = random.randint(2, max_path_colors)
    legend = available[:legend_count]
    path_colors = available[legend_count : legend_count + path_color_count]

    assignments = path_colors[:] + [random.choice(path_colors) for _ in range(len(lengths) - path_color_count)]
    random.shuffle(assignments)
    jobs = list(zip(lengths, assignments))
    jobs.sort(key=lambda item: item[0], reverse=True)

    grid = np.full((height, width), BG_COLOR, dtype=int)
    expected = grid.copy()

    for col, color in zip(_legend_positions(width, legend_count), legend):
        grid[-1, col] = int(color)
    expected[-1, :] = BG_COLOR

    path_area_h = height - 2
    occupied = np.zeros((path_area_h, width), dtype=bool)

    for length, color in jobs:
        cells = _grow_path(path_area_h, width, int(length), occupied)
        if cells is None:
            return None
        for r, c in cells:
            grid[r, c] = int(color)
            occupied[r, c] = True
            expected[r, c] = int(legend[(int(length) - 1) // 2])

    bg_count = int(np.count_nonzero(grid == BG_COLOR))
    if any(bg_count <= int(np.count_nonzero(grid == color)) for color in set(legend + path_colors)):
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid path-length legend input")
