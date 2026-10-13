import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
SEED_COLORS = (3, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300


def _validate_input(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 3 and w >= 4
    assert np.all(np.isin(g[:, -1], (RED, BLUE))), "filter column must be red/blue"
    assert np.all(g[1:, :-1] == BLACK), "only the seed row may occupy the interior"

    seed = g[0, :-1]
    assert np.all(seed != BLACK), "seed row must be fully colored"
    assert np.unique(seed).size >= 2, "seed row must use multiple colors"
    return g, seed.copy()


def _propagate(seed: np.ndarray, filters: np.ndarray) -> list[np.ndarray]:
    rows = [seed.copy()]
    prev = seed.copy()
    for value in filters[1:]:
        if int(value) == RED:
            curr = prev.copy()
        else:
            curr = prev[::-1].copy()
        rows.append(curr)
        prev = curr
    return rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the seed-row pattern downward, copying or mirroring it according
    to the red/blue filter column while keeping that filter column unchanged.
    """
    g, seed = _validate_input(grid)
    out = g.copy()
    rows = _propagate(seed, g[:, -1])
    for r in range(1, g.shape[0]):
        out[r, :-1] = rows[r]
    return out


def _make_seed(rng: random.Random, length: int) -> np.ndarray:
    color_count = rng.randint(2, min(5, length, len(SEED_COLORS)))
    palette = rng.sample(SEED_COLORS, color_count)

    for _ in range(100):
        seed = [rng.choice(palette) for _ in range(length)]
        if len(set(seed)) < 2:
            continue
        if seed == seed[::-1]:
            continue
        return np.array(seed, dtype=int)
    raise RuntimeError("failed to build a non-degenerate seed row")


def _make_filters(rng: random.Random, height: int) -> np.ndarray:
    lower = [RED] * (height - 1)
    blue_count = rng.randint(1, height - 2)
    blue_indices = rng.sample(range(height - 1), blue_count)
    for idx in blue_indices:
        lower[idx] = BLUE
    rng.shuffle(lower)

    if RED not in lower or BLUE not in lower:
        raise RuntimeError("filter column must expose both rule branches")

    return np.array([rng.choice((RED, BLUE)), *lower], dtype=int)


def _gen_attempt(rng: random.Random) -> np.ndarray:
    height = rng.randint(3, 12)
    width = rng.randint(4, 12)

    grid = np.full((height, width), BLACK, dtype=int)
    grid[0, :-1] = _make_seed(rng, width - 1)
    grid[:, -1] = _make_filters(rng, height)

    # Keep the two behaviors visibly distinct in the generated tasks.
    assert not np.array_equal(grid[0, :-1], grid[0, :-1][::-1])
    _validate_input(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            return _gen_attempt(rng)
        except RuntimeError:
            continue
    raise RuntimeError("failed to generate a valid input")
