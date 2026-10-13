import numpy as np

from collections import Counter
from typing import Iterable, Tuple


_RED = 2
_BG = 0
_MAX_SIZE = 30
_MAX_GENERATE_ATTEMPTS = 500


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _seed_coords(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    non_bg = np.argwhere(grid != bg)
    if non_bg.size == 0:
        return []

    colors = grid[non_bg[:, 0], non_bg[:, 1]]
    # 每个种子像素互不相同且非背景色。
    if int(np.any(colors == _RED)):
        raise AssertionError("seed color must not be red")

    unique_colors, counts = np.unique(colors, return_counts=True)
    if not np.all(counts == 1):
        raise AssertionError("each seed color must appear exactly once")
    if len(unique_colors) != len(colors):
        raise AssertionError("seed colors must be unique")

    occupied = {(int(r), int(c)) for r, c in non_bg}
    for r, c in occupied:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (r + dr, c + dc) in occupied:
                raise AssertionError("seeds must be isolated")

    return [(int(r), int(c), int(grid[int(r), int(c)])) for r, c in non_bg]


def _intersections(
    seeds: list[tuple[int, int, int]],
    h: int,
    w: int,
    bg: int,
    grid: np.ndarray,
) -> set[tuple[int, int]]:
    points: set[tuple[int, int]] = set()
    for i, (r1, c1, _c1) in enumerate(seeds):
        slash = r1 + c1
        for j, (r2, c2, _c2) in enumerate(seeds):
            if i == j:
                continue

            backslash = r2 - c2
            s_plus_d = slash + backslash
            if s_plus_d & 1:
                continue

            r = s_plus_d // 2
            c = (slash - backslash) // 2
            if 0 <= r < h and 0 <= c < w and grid[r, c] == bg:
                points.add((r, c))
    return points


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint red points where invisible diagonal rays from different seed pixels intersect.

    Input structure:
    - Background is black.
    - One or more single-cell colored seeds are present.
    - Seed colors are pairwise distinct, none is red, and seeds are isolated.

    Rule:
    - Each seed emits two diagonal ray families: slash (r+c constant) and backslash (r-c constant).
    - For every ordered pair of distinct seeds, compute the intersection between the first seed's slash line and the second seed's backslash line.
    - A cell becomes red iff it is a background cell and the intersection coordinates are integer grid positions.
    - Seed cells stay unchanged.

    Key insight:
    - Only seed pairs with matching row/column parity intersect on integer cells; parity mismatch suppresses intersections.
    """

    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.size == 0:
        raise AssertionError("expected a non-empty 2D grid")

    bg = _background_color(g)
    seeds = _seed_coords(g, bg)
    if not seeds:
        raise AssertionError("expected at least one seed")

    h, w = g.shape
    out = g.copy()
    for r, c in _intersections(seeds, h, w, bg, g):
        out[r, c] = _RED
    return out


def _find_free_position(
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    target_parity: int,
    max_tries: int = 200,
) -> tuple[int, int] | None:
    for _ in range(max_tries):
        r = int(np.random.randint(0, h))
        c = int(np.random.randint(0, w))
        if (r + c) % 2 != target_parity:
            continue
        if (r, c) in occupied:
            continue

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (r + dr, c + dc) in occupied:
                break
        else:
            return r, c
    return None


def _parity_mode(rng: np.random.Generator, branch: int, seed_count: int) -> list[int]:
    if branch == 0:
        return [int(rng.integers(0, 2))]

    if branch == 1:
        p = int(rng.integers(0, 2))
        return [p, p]

    if branch == 2:
        return [0, 1]

    if seed_count == 2:
        p = int(rng.integers(0, 2))
        return [p, p]

    if rng.random() < 0.55:
        p = int(rng.integers(0, 2))
        return [p] * seed_count

    return [int(rng.integers(0, 2)) for _ in range(seed_count)]


def _random_input() -> np.ndarray:
    rng = np.random.default_rng()

    h = int(rng.integers(10, _MAX_SIZE + 1))
    w = int(rng.integers(10, _MAX_SIZE + 1))

    branch = int(rng.random() * 4)
    seed_count = (
        1 if branch == 0 else 2 if branch in (1, 2) else int(rng.integers(3, 6))
    )

    # 只使用非红色前景色，且颜色互异。
    available_colors = [1, 3, 4, 5, 6, 7, 8, 9]
    if seed_count > len(available_colors):
        return _random_input()
    colors = list(rng.choice(available_colors, size=seed_count, replace=False))

    parities = _parity_mode(rng, branch, seed_count)

    occupied: set[tuple[int, int]] = set()
    seeds: list[tuple[int, int, int]] = []
    for idx, parity in enumerate(parities):
        pos = _find_free_position(h, w, occupied, parity)
        if pos is None:
            raise RuntimeError("position sampling failed")
        r, c = pos
        occupied.add((r, c))
        seeds.append((r, c, colors[idx]))

    grid = np.full((h, w), _BG, dtype=int)
    for r, c, color in seeds:
        grid[r, c] = color

    return grid


def generate_input() -> np.ndarray:
    """Generate diverse random inputs for the diagonal-ray intersection rule."""
    for _ in range(_MAX_GENERATE_ATTEMPTS):
        try:
            inp = _random_input()
            _ = main(inp)
            return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate valid input after many attempts")
