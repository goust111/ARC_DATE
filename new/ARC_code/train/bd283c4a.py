from collections import Counter

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Pack all pixels into color-frequency runs along bottom-up columns.

    Input structure
    The input is a rectangular grid whose colors are all foreground material;
    only the number of pixels of each color matters. In the observed task
    instances, every present color has a distinct pixel count.

    Rule
    Count the pixels of each color and order the colors from most frequent to
    least frequent. Rewrite a same-size grid by placing each color's full count
    as one contiguous run while scanning columns from left to right and, within
    each column, from bottom to top.

    Key insight
    The output preserves the multiset of colors but discards all original
    positions, turning the grid into a frequency-sorted column packing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    counts = Counter(map(int, g.ravel()))
    assert len(set(counts.values())) == len(counts), "color counts must have a unique rank"

    colors = sorted(counts, key=lambda color: -counts[color])
    out = np.empty_like(g)
    height, width = g.shape

    scan = ((r, c) for c in range(width) for r in range(height - 1, -1, -1))
    for color in colors:
        for _ in range(counts[color]):
            r, c = next(scan)
            out[r, c] = color

    return out

# --------------- generator (att1) ---------------

GRID_SHAPE = (10, 10)
GRID_AREA = GRID_SHAPE[0] * GRID_SHAPE[1]

# Palette: four distinct foreground colors, no background.
COLOR_POOL = np.arange(1, 10)
N_COLORS = 4

# Counts: largest, second, third are sampled; smallest is residual.
COUNT_RANGES = ((40, 55), (25, 35), (10, 20))
RESIDUAL_COUNT_RANGE = (2, 10)
COUNT_SAMPLE_RETRIES = 100
GENERATION_RETRIES = 300


def _sample_ranked_counts(rng: np.random.Generator) -> list[int] | None:
    """Return four distinct counts in descending rank ranges, summing to the grid area."""
    for _ in range(COUNT_SAMPLE_RETRIES):
        sampled = [int(rng.integers(lo, hi + 1)) for lo, hi in COUNT_RANGES]
        residual = GRID_AREA - sum(sampled)
        lo, hi = RESIDUAL_COUNT_RANGE
        counts = sampled + [residual]
        if lo <= residual <= hi and len(set(counts)) == N_COLORS:
            return counts
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    colors = rng.choice(COLOR_POOL, size=N_COLORS, replace=False).tolist()
    counts = _sample_ranked_counts(rng)
    if counts is None:
        return None

    pixels = np.repeat(colors, counts)
    rng.shuffle(pixels)
    return pixels.astype(int, copy=False).reshape(GRID_SHAPE)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        result = _gen_attempt()
        if result is not None:
            try:
                out = main(result)
            except Exception:
                continue
            if out is not None:
                return result
    raise RuntimeError(f"Failed to generate valid input after {GENERATION_RETRIES} attempts")
