import random

import numpy as np


BLACK = 0
GRAY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

HEIGHT_RANGE = (10, 15)
WIDTH_RANGE = (20, 25)
SEGMENT_COUNT_BY_PARITY = {
    0: (2, 4),
    1: (3, 5),
}
SEGMENT_LEN_MAX = 4
GAP_LEN_MAX = 4
GENERATION_RETRIES = 300
ROW_RETRIES = 200


def _count_gray_segments(data: np.ndarray) -> int:
    mask = np.asarray(data == GRAY, dtype=bool)
    if mask.size == 0 or not bool(mask.any()):
        return 0
    starts = mask.copy()
    starts[1:] &= ~mask[:-1]
    return int(starts.sum())


def _bounded_composition(
    total: int,
    parts: int,
    min_each: int,
    max_each: int,
    rng: random.Random,
) -> list[int] | None:
    if parts == 0:
        return [] if total == 0 else None
    if total < parts * min_each or total > parts * max_each:
        return None

    values = [min_each] * parts
    remaining = total - parts * min_each
    for _ in range(remaining):
        choices = [i for i, value in enumerate(values) if value < max_each]
        if not choices:
            return None
        values[rng.choice(choices)] += 1
    rng.shuffle(values)
    return values


def _split_outside_black(total: int, rng: random.Random) -> tuple[int, int]:
    if total <= 0:
        return 0, 0
    if total == 1:
        return (1, 0) if rng.random() < 0.5 else (0, 1)
    if rng.random() < 0.7:
        lead = rng.randint(1, total - 1)
    else:
        lead = rng.randint(0, total)
    return lead, total - lead


def _build_data_row(
    data_width: int,
    parity: int,
    rng: random.Random,
) -> tuple[np.ndarray, int, int, int] | None:
    count_options = [
        count
        for count in SEGMENT_COUNT_BY_PARITY[parity]
        if data_width >= 2 * count - 1
    ]
    if not count_options:
        return None

    for _ in range(ROW_RETRIES):
        segment_count = rng.choice(count_options)

        max_gray_total = min(SEGMENT_LEN_MAX * segment_count, data_width - (segment_count - 1))
        gray_total = rng.randint(segment_count, max_gray_total)
        lengths = _bounded_composition(gray_total, segment_count, 1, SEGMENT_LEN_MAX, rng)
        if lengths is None:
            continue

        black_total = data_width - gray_total
        max_internal_black = min(GAP_LEN_MAX * (segment_count - 1), black_total)
        if max_internal_black < segment_count - 1:
            continue

        internal_black = rng.randint(segment_count - 1, max_internal_black)
        if black_total > segment_count - 1 and rng.random() < 0.75:
            capped = min(max_internal_black, black_total - 1)
            if capped >= segment_count - 1:
                internal_black = rng.randint(segment_count - 1, capped)

        gaps = _bounded_composition(internal_black, segment_count - 1, 1, GAP_LEN_MAX, rng)
        if gaps is None:
            continue

        lead, tail = _split_outside_black(black_total - internal_black, rng)
        row = np.zeros(data_width, dtype=int)

        pos = lead
        for i, length in enumerate(lengths):
            row[pos : pos + length] = GRAY
            pos += length
            if i < segment_count - 1:
                pos += gaps[i]

        if pos + tail != data_width:
            continue

        return row, segment_count, lead, tail

    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill row-wise gaps between gray segments using one of two label colors.

    Input structure:
    - A black rectangular grid whose leftmost two columns contain one colored label pixel each per row.
    - Column 2 is a black separator, and the data area starts at column 3.
    - In each row's data area, gray pixels form several horizontal segments separated by black gaps.

    Rule:
    - Count how many gray segments appear in each row's data area.
    - If the count is odd, select the color in column 0; if it is even, select the color in column 1.
    - Fill every black pixel between the first gray segment and the last gray segment of that row with the selected color.
    - Keep the gray pixels, the label pixels, and the black area before the first segment and after the last segment unchanged.

    Key insight:
    The two colored cells at the start of each row form a local two-color legend, and the parity of the row's gray-segment count decides which legend color fills the internal gaps.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    if g.ndim != 2 or g.shape[1] <= 3:
        return out

    for r in range(g.shape[0]):
        data = g[r, 3:]
        gray_positions = np.flatnonzero(data == GRAY)
        if gray_positions.size == 0:
            continue

        segment_count = _count_gray_segments(data)
        fill_color = int(g[r, 0]) if segment_count % 2 == 1 else int(g[r, 1])

        start = int(gray_positions[0])
        end = int(gray_positions[-1])
        span = out[r, 3 + start : 3 + end + 1]
        span[span == BLACK] = fill_color

    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*HEIGHT_RANGE)
    width = rng.randint(*WIDTH_RANGE)
    data_width = width - 3

    odd_rows = height // 2
    if height % 2 == 1 and rng.random() < 0.5:
        odd_rows += 1
    even_rows = height - odd_rows
    if odd_rows == 0 or even_rows == 0:
        return None

    parities = [1] * odd_rows + [0] * even_rows
    rng.shuffle(parities)

    rows = []
    segment_counts = []
    has_leading_black = False
    has_trailing_black = False

    for parity in parities:
        built = _build_data_row(data_width, parity, rng)
        if built is None:
            return None
        data_row, segment_count, lead, tail = built

        label_a, label_b = rng.sample(COLOR_POOL, 2)
        full_row = np.zeros(width, dtype=int)
        full_row[0] = label_a
        full_row[1] = label_b
        full_row[3:] = data_row

        rows.append(full_row)
        segment_counts.append(segment_count)
        has_leading_black |= lead > 0
        has_trailing_black |= tail > 0

    if not has_leading_black or not has_trailing_black:
        return None
    if len(set(segment_counts)) < 2:
        return None

    grid = np.array(rows, dtype=int)
    out = main(grid)

    if np.array_equal(grid, out):
        return None

    used_parities = {count % 2 for count in segment_counts}
    if used_parities != {0, 1}:
        return None

    for r, segment_count in enumerate(segment_counts):
        fill_color = int(grid[r, 0]) if segment_count % 2 == 1 else int(grid[r, 1])
        data_in = grid[r, 3:]
        data_out = out[r, 3:]
        gray_positions = np.flatnonzero(data_in == GRAY)
        start = int(gray_positions[0])
        end = int(gray_positions[-1])

        if not np.array_equal(data_in[:start], data_out[:start]):
            return None
        if not np.array_equal(data_in[end + 1 :], data_out[end + 1 :]):
            return None

        expected = data_in.copy()
        span = expected[start : end + 1]
        span[span == BLACK] = fill_color
        if not np.array_equal(expected, data_out):
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated retries")
