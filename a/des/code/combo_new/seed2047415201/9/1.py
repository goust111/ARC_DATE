from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
SEED_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _foreground_colors(g: np.ndarray) -> list[int]:
    """Return non-black colors in first-seen order."""
    colors: list[int] = []
    seen: set[int] = set()
    for value in g.reshape(-1):
        color = int(value)
        if color == BLACK or color in seen:
            continue
        seen.add(color)
        colors.append(color)
    return colors


def _voronoi_fill(g: np.ndarray, colors: list[int]) -> np.ndarray:
    h, w = g.shape
    rows, cols = np.indices((h, w))

    best_dist = np.full((h, w), h + w + 1, dtype=int)
    best_color = np.full((h, w), BLACK, dtype=int)
    tied = np.zeros((h, w), dtype=bool)

    for color in colors:
        seeds = np.argwhere(g == int(color))
        assert seeds.size > 0, "each foreground color must have at least one seed"

        color_dist = np.full((h, w), h + w + 1, dtype=int)
        for sr, sc in seeds:
            dist = np.abs(rows - int(sr)) + np.abs(cols - int(sc))
            color_dist = np.minimum(color_dist, dist)

        closer = color_dist < best_dist
        equal = color_dist == best_dist
        tied[closer] = False
        tied[equal] = True
        best_color[closer] = int(color)
        best_dist[closer] = color_dist[closer]

    out = best_color
    out[tied] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow scattered color seeds into Voronoi regions and draw grey separators on equal-distance boundaries.

    Input structure:
    - The background is black.
    - Several non-black, non-grey foreground colors appear as isolated single-pixel seeds.
    - Each foreground color may have one to three seeds, and different colors never share a cell.

    Rule:
    - Treat every foreground pixel as a seed expanding one orthogonal step per tick.
    - Same-colored seeds merge into one region by using the nearest seed of that color.
    - A cell reached first by exactly one color becomes that color; a cell reached at the same tick by two or more different colors stays unclaimed.
    - After all cells have been considered, repaint every unclaimed boundary cell grey.

    Key insight:
    - The simultaneous expansion is equivalent to comparing Manhattan distance to each color's nearest seed; ties between colors are the separator lines.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must not exceed 30 in either dimension"

    colors = _foreground_colors(g)
    assert len(colors) >= 2, "expected at least two seed colors"
    assert GREY not in colors, "grey is reserved for output separator lines"

    return _voronoi_fill(g, colors)


# --------------- generator ---------------

GRID_H_RANGE = (9, 24)
GRID_W_RANGE = (9, 24)
COLOR_COUNT_RANGE = (2, 4)
GEN_RETRIES = 500
PLACE_RETRIES = 250


def _seed_count_pattern(n_colors: int, rng: random.Random) -> list[int]:
    """Balance singleton-only, one-merge, and multi-merge examples."""
    mode = rng.choice(("singletons", "one_multi", "many_multi"))
    counts = [1] * n_colors

    if mode == "one_multi":
        counts[rng.randrange(n_colors)] = rng.choice((2, 3))
    elif mode == "many_multi":
        k = rng.randint(2, n_colors)
        for idx in rng.sample(range(n_colors), k):
            counts[idx] = rng.choice((2, 3))

    return counts


def _isolated_from(existing: list[tuple[int, int]], r: int, c: int) -> bool:
    return all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in existing)


def _place_seeds(
    h: int,
    w: int,
    colors: list[int],
    counts: list[int],
    rng: random.Random,
) -> np.ndarray | None:
    grid = np.full((h, w), BLACK, dtype=int)
    occupied: list[tuple[int, int]] = []

    margin = 1 if h >= 7 and w >= 7 else 0
    candidates = [
        (r, c)
        for r in range(margin, h - margin)
        for c in range(margin, w - margin)
    ]
    rng.shuffle(candidates)

    for color, count in zip(colors, counts, strict=False):
        for _ in range(count):
            placed = False
            for _attempt in range(PLACE_RETRIES):
                if not candidates:
                    return None
                idx = rng.randrange(len(candidates))
                r, c = candidates[idx]
                if not _isolated_from(occupied, r, c):
                    continue
                grid[r, c] = int(color)
                occupied.append((r, c))
                candidates.pop(idx)
                placed = True
                break
            if not placed:
                return None

    return grid


def _grey_has_color_contacts(out: np.ndarray) -> bool:
    h, w = out.shape
    for r, c in zip(*np.where(out == GREY), strict=False):
        adjacent: set[int] = set()
        for dr, dc in DIR4:
            rr, cc = int(r) + dr, int(c) + dc
            if 0 <= rr < h and 0 <= cc < w:
                color = int(out[rr, cc])
                if color not in (BLACK, GREY):
                    adjacent.add(color)
        if len(adjacent) >= 2:
            return True
    return False


def _valid_generated(grid: np.ndarray, colors: list[int]) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    if out.shape != grid.shape or np.array_equal(out, grid):
        return False

    seed_mask = grid != BLACK
    if not bool(np.all(out[seed_mask] == grid[seed_mask])):
        return False

    area = int(grid.size)
    grey_count = int(np.sum(out == GREY))
    min_grey = max(4, min(grid.shape) // 2)
    if grey_count < min_grey or grey_count > int(area * 0.45):
        return False
    if not _grey_has_color_contacts(out):
        return False

    for color in colors:
        seed_count = int(np.sum(grid == int(color)))
        region_count = int(np.sum(out == int(color)))
        if seed_count == 0 or region_count < max(seed_count + 2, 5):
            return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    n_colors = rng.randint(*COLOR_COUNT_RANGE)
    colors = rng.sample(list(SEED_PALETTE), n_colors)
    counts = _seed_count_pattern(n_colors, rng)

    grid = _place_seeds(h, w, colors, counts, rng)
    if grid is None:
        return None
    if not _valid_generated(grid, colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
