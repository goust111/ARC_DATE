import numpy as np


def _unique_most_frequent_color(g: np.ndarray) -> int:
    values, counts = np.unique(g, return_counts=True)
    winners = values[counts == counts.max()]
    assert winners.size == 1, "expected a unique most frequent color"
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp the whole input pattern into output block slots marked by its unique most frequent color.

    Input structure
    - The input is a 3x3 grid of non-black colors.
    - Exactly one color appears more often than every other color.

    Rule
    - Treat the input as both a source sprite and a 3x3 placement mask.
    - Create a 9x9 black output divided into 3x3 block slots.
    - For each input cell whose color is the unique most frequent color, copy
      the entire input grid into the corresponding output block slot.

    Key insight
    - The dominant color does not get expanded by itself; its positions select
      where full copies of the original 3x3 pattern are stamped.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert (h, w) == (3, 3), "expected a 3x3 input grid"
    assert not np.any(g == 0), "input pattern should not contain black"

    marker_color = _unique_most_frequent_color(g)
    out = np.zeros((h * h, w * w), dtype=int)
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) == marker_color:
                out[r * h : (r + 1) * h, c * w : (c + 1) * w] = g
    return out

# --------------- generator (o5x) ---------------

# Grid, palette, and retry bounds.
_GEN_GRID_SIZE = 3
_GEN_CELL_COUNT = _GEN_GRID_SIZE * _GEN_GRID_SIZE
_GEN_COLOR_VALUES = np.arange(1, 10)
_GEN_MAX_ATTEMPTS = 300

# Distinct-color and dominant-count mixture.
_GEN_KD_PAIRS = ((3, 4), (4, 3), (4, 4), (5, 3), (5, 4))
_GEN_KD_WEIGHTS = np.array((4, 3, 1, 3, 1), dtype=float) / 12.0

# Non-dominant color count allocation.
_GEN_NON_DOM_MIN = 1
_GEN_NON_DOM_CAP = 2


def _gen_non_dominant_counts(n_colors: int, remaining: int, dom_count: int) -> list[int]:
    max_count = _GEN_NON_DOM_CAP if n_colors * _GEN_NON_DOM_CAP >= remaining else dom_count - 1
    counts = [_GEN_NON_DOM_MIN] * n_colors
    extra = remaining - n_colors * _GEN_NON_DOM_MIN
    while extra > 0:
        available = [i for i, count in enumerate(counts) if count < max_count]
        if not available:
            return []
        idx = int(np.random.choice(available))
        counts[idx] += 1
        extra -= 1
    np.random.shuffle(counts)
    return counts


def _gen_expected_output(inp: np.ndarray) -> np.ndarray:
    s = _GEN_GRID_SIZE
    values, counts = np.unique(inp, return_counts=True)
    marker = int(values[np.argmax(counts)])
    out = np.zeros((_GEN_CELL_COUNT, _GEN_CELL_COUNT), dtype=int)
    for r in range(s):
        for c in range(s):
            if int(inp[r, c]) == marker:
                out[r * s : (r + 1) * s, c * s : (c + 1) * s] = inp
    return out


def _gen_attempt() -> np.ndarray | None:
    pair_idx = int(np.random.choice(len(_GEN_KD_PAIRS), p=_GEN_KD_WEIGHTS))
    distinct_count, dom_count = _GEN_KD_PAIRS[pair_idx]

    colors = np.random.choice(_GEN_COLOR_VALUES, size=distinct_count, replace=False)
    dom_idx = int(np.random.randint(distinct_count))
    dom_color = int(colors[dom_idx])
    other_colors = [int(color) for i, color in enumerate(colors) if i != dom_idx]

    other_counts = _gen_non_dominant_counts(
        len(other_colors), _GEN_CELL_COUNT - dom_count, dom_count
    )
    if not other_counts:
        return None

    values = [dom_color] * dom_count
    for color, count in zip(other_colors, other_counts):
        values.extend([color] * count)
    np.random.shuffle(values)

    inp = np.array(values, dtype=int).reshape(_GEN_GRID_SIZE, _GEN_GRID_SIZE)
    if not np.array_equal(main(inp), _gen_expected_output(inp)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError(f"generate_input: failed after {_GEN_MAX_ATTEMPTS} attempts")
