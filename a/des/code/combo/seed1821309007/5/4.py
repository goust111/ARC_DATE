import random

import numpy as np


ORANGE = 7
BLUE = 1
GREY = 5
MAROON = 9
GEN_RETRIES = 300


def _find_cross(grid: np.ndarray) -> tuple[int, int, int, int]:
    h, w = grid.shape
    row_candidates = []
    for r in range(h):
        if np.all(grid[r] == grid[r, 0]):
            row_candidates.append((r, int(grid[r, 0])))

    col_candidates = []
    for c in range(w):
        if np.all(grid[:, c] == grid[0, c]):
            col_candidates.append((c, int(grid[0, c])))

    for r, row_color in row_candidates:
        for c, col_color in col_candidates:
            if row_color != col_color:
                continue
            cross_color = row_color
            non_cross = grid[grid != cross_color]
            if non_cross.size == 0:
                continue
            colors, counts = np.unique(non_cross, return_counts=True)
            bg = int(colors[np.argmax(counts)])
            return r, c, cross_color, bg

    raise AssertionError("failed to identify center cross")


def _draw_triangle(out: np.ndarray, corner: str, size: int, color: int) -> None:
    if size <= 0:
        return

    h, w = out.shape
    for dr in range(size):
        span = size - dr
        if corner == "tl":
            out[dr, :span] = color
        elif corner == "tr":
            out[dr, w - span :] = color
        elif corner == "bl":
            out[h - 1 - dr, :span] = color
        elif corner == "br":
            out[h - 1 - dr, w - span :] = color
        else:
            raise ValueError(f"unknown corner: {corner}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the scattered marker pixels in each quadrant and encode each count as
    a solid corner triangle in that quadrant.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    center_r, center_c, cross_color, bg = _find_cross(g)
    out = g.copy()

    quadrants = [
        ((slice(0, center_r), slice(0, center_c)), "tl"),
        ((slice(0, center_r), slice(center_c + 1, g.shape[1])), "tr"),
        ((slice(center_r + 1, g.shape[0]), slice(0, center_c)), "bl"),
        ((slice(center_r + 1, g.shape[0]), slice(center_c + 1, g.shape[1])), "br"),
    ]

    for (rs, cs), corner in quadrants:
        sub = g[rs, cs]
        count = int(np.count_nonzero((sub != bg) & (sub != cross_color)))
        out[rs, cs] = np.where(sub == cross_color, cross_color, bg)
        _draw_triangle(out[rs, cs], corner, count, MAROON)

    out[center_r, :] = cross_color
    out[:, center_c] = cross_color
    return out


def _random_points(
    rng: random.Random, row_range: range, col_range: range, count: int
) -> list[tuple[int, int]]:
    cells = [(r, c) for r in row_range for c in col_range]
    if count == 0:
        return []
    return rng.sample(cells, count)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randrange(9, 22, 2)
    w = rng.randrange(9, 22, 2)
    center_r = h // 2
    center_c = w // 2
    max_size = min(center_r, center_c)
    if max_size <= 0:
        return None

    grid = np.full((h, w), ORANGE, dtype=int)
    grid[center_r, :] = GREY
    grid[:, center_c] = GREY

    counts = []
    for _ in range(4):
        if rng.random() < 0.5:
            counts.append(rng.randint(0, max_size // 2))
        else:
            counts.append(rng.randint(max(0, max_size // 2), max_size))

    if len(set(counts)) == 1:
        idx = rng.randrange(4)
        counts[idx] = (counts[idx] + 1) % (max_size + 1)

    placements = [
        _random_points(rng, range(0, center_r), range(0, center_c), counts[0]),
        _random_points(rng, range(0, center_r), range(center_c + 1, w), counts[1]),
        _random_points(rng, range(center_r + 1, h), range(0, center_c), counts[2]),
        _random_points(rng, range(center_r + 1, h), range(center_c + 1, w), counts[3]),
    ]

    for pts in placements:
        for r, c in pts:
            grid[r, c] = BLUE

    if not np.any(grid == BLUE):
        q = rng.randrange(4)
        r, c = placements[q][0] if placements[q] else {
            0: (rng.randrange(0, center_r), rng.randrange(0, center_c)),
            1: (rng.randrange(0, center_r), rng.randrange(center_c + 1, w)),
            2: (rng.randrange(center_r + 1, h), rng.randrange(0, center_c)),
            3: (rng.randrange(center_r + 1, h), rng.randrange(center_c + 1, w)),
        }[q]
        grid[r, c] = BLUE

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
