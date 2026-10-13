import random

import numpy as np


BG = 0
NEG = 2
GREEN = 3
SEP = 5
POS = 8

MAX_GENERATE_ATTEMPTS = 500
REPEAT_CHOICES = [1, 2, 3]
REPEAT_WEIGHTS = [2, 5, 3]


def _separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEP)]
    assert len(cols) == 1, "expected exactly one full-height separator column"
    return int(cols[0])


def _extract_bars(grid: np.ndarray, sep_col: int) -> list[tuple[int, int, int]]:
    h = grid.shape[0]
    bars: list[tuple[int, int, int]] = []
    for c in range(sep_col):
        col = grid[:, c]
        nz = np.flatnonzero(col != BG)
        if nz.size == 0:
            continue
        color = int(col[nz[0]])
        assert color in {POS, NEG}, "left bars must be teal or red"
        assert np.all(col[nz] == color), "each bar must be monochromatic"
        assert np.array_equal(nz, np.arange(int(nz[0]), h)), "bars must be contiguous and bottom-aligned"
        bars.append((int(c), color, int(nz.size)))
    assert bars, "expected at least one bar"
    for (c0, _color0, _h0), (c1, _color1, _h1) in zip(bars, bars[1:]):
        assert c1 - c0 == 2, "adjacent bars must be separated by one background column"
    return bars


def _period_from_bars(bars: list[tuple[int, int, int]]) -> int:
    pos_sum = sum(height for _c, color, height in bars if color == POS)
    neg_sum = sum(height for _c, color, height in bars if color == NEG)
    period = int(pos_sum - neg_sum)
    assert period >= 3 and period % 2 == 1, "period must be an odd integer >= 3"
    return period


def _rise_fall_heights(period: int) -> list[int]:
    peak = (period + 1) // 2
    return list(range(1, peak + 1)) + list(range(peak - 1, 0, -1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn the signed bar-height difference into a repeated rise-fall column pattern.

    Input structure:
    - The background is black.
    - Left of a single full-height gray separator column are bottom-aligned, one-cell-wide
      vertical bars separated by one background column.
    - Teal bars are positive operands and red bars are negative operands; the region
      right of the separator is empty background.

    Rule:
    - Compute P as the sum of teal bar heights minus the sum of red bar heights.
    - Build the height sequence [1, 2, ..., (P+1)/2, ..., 2, 1], whose length is exactly P.
    - Starting in the first column right of the separator, draw bottom-aligned green bars
      with those heights across P columns, then leave one blank background column.
    - Repeat that P-column rise-fall block plus one-column gap while a full P-column block
      still fits; keep the original left bars and separator unchanged.

    Key insight:
    The arithmetic result is not drawn as a single bar: it simultaneously determines the
    shape of one rise-fall run and the period with which that run is tiled to the right.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sep_col = _separator_col(g)
    bars = _extract_bars(g, sep_col)
    period = _period_from_bars(bars)
    heights = _rise_fall_heights(period)

    out = g.copy()
    out[:, sep_col + 1 :] = BG

    start = sep_col + 1
    while start + period <= w:
        for dc, height in enumerate(heights):
            out[h - height : h, start + dc] = GREEN
        start += period + 1

    return out


def _random_partition(total: int, count: int, lo: int, hi: int) -> list[int] | None:
    for _ in range(100):
        parts: list[int] = []
        remain = total
        for i in range(count):
            slots_left = count - i - 1
            low = max(lo, remain - slots_left * hi)
            high = min(hi, remain - slots_left * lo)
            if low > high:
                break
            parts.append(random.randint(low, high))
            remain -= parts[-1]
        if len(parts) == count and remain == 0:
            random.shuffle(parts)
            return parts
    return None


def _run_count(right_width: int, period: int) -> int:
    runs = 0
    start = 0
    while start + period <= right_width:
        runs += 1
        start += period + 1
    return runs


def _pick_tail(max_tail: int) -> int:
    if max_tail <= 0:
        return 0
    if max_tail == 1:
        return random.choice([0, 1])
    tail_choices = [0] + list(range(1, max_tail + 1)) + list(range(1, max_tail + 1))
    return random.choice(tail_choices)


def _expected_output(inp: np.ndarray) -> np.ndarray:
    sep_col = _separator_col(inp)
    bars = _extract_bars(inp, sep_col)
    period = _period_from_bars(bars)
    heights = _rise_fall_heights(period)
    h, w = inp.shape
    out = inp.copy()
    out[:, sep_col + 1 :] = BG
    start = sep_col + 1
    while start + period <= w:
        for dc, height in enumerate(heights):
            out[h - height : h, start + dc] = GREEN
        start += period + 1
    return out


def _validate_generated(inp: np.ndarray) -> bool:
    if not isinstance(inp, np.ndarray) or inp.ndim != 2:
        return False
    h, w = inp.shape
    if h > 30 or w > 30:
        return False

    try:
        sep_col = _separator_col(inp)
        if not np.all(inp[:, sep_col + 1 :] == BG):
            return False

        bars = _extract_bars(inp, sep_col)
        pos_heights = [height for _c, color, height in bars if color == POS]
        neg_heights = [height for _c, color, height in bars if color == NEG]
        if len(pos_heights) < 2 or len(neg_heights) < 1:
            return False

        period = _period_from_bars(bars)
        if period in {len(bars), len(pos_heights), max(pos_heights)}:
            return False
        if len(set(pos_heights + neg_heights)) < 2:
            return False

        right_width = w - sep_col - 1
        run_count = _run_count(right_width, period)
        if run_count < 1:
            return False

        out = main(inp)
        expected = _expected_output(inp)
        if not np.array_equal(out, expected):
            return False
        if not np.array_equal(out[:, : sep_col + 1], inp[:, : sep_col + 1]):
            return False
        heights = _rise_fall_heights(period)
        start = sep_col + 1
        for _ in range(run_count):
            for dc, height in enumerate(heights):
                col = out[:, start + dc]
                if not np.all(col[: h - height] == BG):
                    return False
                if not np.all(col[h - height :] == GREEN):
                    return False
            start += period
            if start < w:
                if start + period <= w:
                    if not np.all(out[:, start] == BG):
                        return False
                else:
                    if not np.all(out[:, start:] == BG):
                        return False
                start += 1
    except AssertionError:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(9, 13)
    max_bar_height = h - 2
    max_peak = h - 3
    period_choices = [p for p in (3, 5, 7, 9, 11) if (p + 1) // 2 <= max_peak]
    if not period_choices:
        return None

    pos_count = random.randint(2, 4)
    neg_count = random.randint(1, 3)
    bar_count = pos_count + neg_count
    left_pad = random.randint(0, 2)
    sep_gap = random.randint(1, 3)
    sep_col = left_pad + 2 * (bar_count - 1) + sep_gap + 1

    configs: list[tuple[int, int, int]] = []
    for period in period_choices:
        for repeats, weight in zip(REPEAT_CHOICES, REPEAT_WEIGHTS):
            min_right = repeats * period + (repeats - 1)
            max_tail = min(period, 30 - (sep_col + 1 + min_right))
            if max_tail >= 0:
                configs.extend([(period, repeats, max_tail)] * weight)
    if not configs:
        return None

    period, repeats, max_tail = random.choice(configs)
    tail = _pick_tail(max_tail)
    width = sep_col + 1 + repeats * period + (repeats - 1) + tail
    if width > 30:
        return None

    neg_sum_min = max(2, neg_count)
    neg_sum_max = min(neg_count * max_bar_height, pos_count * max_bar_height - period)
    if neg_sum_min > neg_sum_max:
        return None
    neg_sum = random.randint(neg_sum_min, neg_sum_max)
    pos_sum = neg_sum + period

    pos_heights = _random_partition(pos_sum, pos_count, 1, max_bar_height)
    neg_heights = _random_partition(neg_sum, neg_count, 1, max_bar_height)
    if pos_heights is None or neg_heights is None:
        return None
    if period in {bar_count, pos_count, max(pos_heights)}:
        return None

    bars = [(POS, height) for height in pos_heights] + [(NEG, height) for height in neg_heights]
    random.shuffle(bars)

    grid = np.zeros((h, width), dtype=int)
    grid[:, sep_col] = SEP
    c = left_pad
    for color, height in bars:
        grid[h - height : h, c] = color
        c += 2

    if _run_count(width - sep_col - 1, period) != repeats:
        return None
    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
