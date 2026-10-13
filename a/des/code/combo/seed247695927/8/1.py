import random

import numpy as np


TEAL = 8
RED = 2
GREY = 5


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bar_height(grid: np.ndarray, col: int, bg: int) -> tuple[int, int] | None:
    column = grid[:, col]
    non_bg = np.flatnonzero(column != bg)
    if non_bg.size == 0:
        return None

    top = int(non_bg[0])
    color = int(column[top])
    if color not in (TEAL, RED):
        raise AssertionError("operand bars must be teal or red")
    if not np.all(column[top:] == color):
        raise AssertionError("bars must be contiguous vertical suffixes")
    if top > 0 and np.any(column[:top] != bg):
        raise AssertionError("cells above a bar must be background")
    return color, grid.shape[0] - top


def main(grid: np.ndarray) -> np.ndarray:
    """Draw a grey result bar whose height is teal total minus red total."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = map(int, g.shape)
    assert width >= 3

    bg = _background_color(g)
    result_col = width - 2
    assert np.all(g[:, result_col] == bg), "penultimate column must be the empty result slot"

    total = 0
    teal_seen = False
    for col in range(width):
        if col == result_col:
            continue
        info = _bar_height(g, col, bg)
        if info is None:
            continue
        color, bar_height = info
        if color == TEAL:
            total += bar_height
            teal_seen = True
        else:
            total -= bar_height

    assert teal_seen, "at least one teal operand is required"
    assert 0 < total <= height, "result height must fit in the grid"

    out = g.copy()
    out[height - total :, result_col] = GREY
    return out


GEN_RETRIES = 500
HEIGHT_RANGE = (6, 13)
WIDTH_RANGE = (7, 17)
BAR_COUNT_RANGE = (2, 6)


def _bounded_composition(total: int, parts: int, upper: int) -> list[int] | None:
    if parts == 0:
        return [] if total == 0 else None
    if not (parts <= total <= parts * upper):
        return None
    values = []
    remaining = total
    for i in range(parts):
        left = parts - i - 1
        lo = max(1, remaining - left * upper)
        hi = min(upper, remaining - left)
        values.append(random.randint(lo, hi))
        remaining -= values[-1]
    random.shuffle(values)
    return values


def _sample_specs(bar_count: int, max_height: int) -> list[tuple[int, int]] | None:
    include_red = random.choice([False, True])
    if include_red:
        teal_count = random.randint(1, bar_count - 1)
    else:
        teal_count = bar_count
    red_count = bar_count - teal_count

    result_height = random.randint(1, max_height)
    red_sum = 0 if red_count == 0 else random.randint(red_count, red_count * max_height)
    teal_sum = result_height + red_sum
    teal_heights = _bounded_composition(teal_sum, teal_count, max_height)
    red_heights = _bounded_composition(red_sum, red_count, max_height)
    if teal_heights is None or red_heights is None:
        return None

    specs = [(TEAL, h) for h in teal_heights] + [(RED, h) for h in red_heights]
    random.shuffle(specs)
    return specs


def _candidate_columns(width: int) -> list[int]:
    start = random.randint(0, 1)
    return list(range(start, width - 2, 2))


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    candidates = _candidate_columns(width)
    if len(candidates) < BAR_COUNT_RANGE[0]:
        return None

    bar_count = random.randint(BAR_COUNT_RANGE[0], min(BAR_COUNT_RANGE[1], len(candidates)))
    max_bar_height = height - 1
    specs = _sample_specs(bar_count, max_bar_height)
    if specs is None:
        return None

    bg_choices = [c for c in range(10) if c not in (TEAL, RED, GREY)]
    bg = random.choice(bg_choices)
    grid = np.full((height, width), bg, dtype=int)

    cols = sorted(random.sample(candidates, bar_count))
    for col, (color, bar_height) in zip(cols, specs):
        grid[height - bar_height :, col] = color

    try:
        out = main(grid)
    except AssertionError:
        return None

    expected = grid.copy()
    total = sum(h for color, h in specs if color == TEAL) - sum(h for color, h in specs if color == RED)
    expected[height - total :, width - 2] = GREY
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
