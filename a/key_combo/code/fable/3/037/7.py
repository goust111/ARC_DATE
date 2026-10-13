import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
TEAL = 8

GEN_RETRY_LIMIT = 200


def _extract_valid_segments(grid: np.ndarray) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    segment_width = None

    for row in grid:
        blue_cols = np.flatnonzero(row == BLUE)
        teal_cols = np.flatnonzero(row == TEAL)
        if blue_cols.size != 1 or teal_cols.size != 1:
            continue

        blue_col = int(blue_cols[0])
        teal_col = int(teal_cols[0])
        if blue_col > teal_col:
            continue

        segment = row[blue_col + 1 : teal_col].copy()
        assert segment.size > 0, "valid bracket pair must enclose at least one cell"
        if segment_width is None:
            segment_width = int(segment.size)
        else:
            assert int(segment.size) == segment_width, "all valid segments must share one width"
        segments.append(segment)

    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """Extract blue-teal bracketed row segments, discard reversed rows, and stack the kept segments by increasing red count.

    Input structure:
    - The grid is a black-and-red base surface with some rows marked by one blue control pixel and one teal control pixel.
    - A marked row is valid only when the blue control sits left of the teal control; teal-left-blue-right rows are decoys.
    - The cells strictly between a valid control pair form a horizontal segment, and every valid segment has the same width.

    Rule:
    - Scan the grid row by row and keep only rows containing exactly one blue control and one teal control in blue-left-teal-right order.
    - For each kept row, extract the cells strictly between the two controls and count how many red pixels the extracted segment contains.
    - Sort the extracted segments from fewest red pixels to most red pixels, then stack them top to bottom as the output.

    Key insight:
    - The colored pixels act as row brackets that say where to crop; their vertical positions do not set the answer order, and reversed brackets are deliberate distractors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    segments = _extract_valid_segments(g)
    assert segments, "expected at least one valid blue-left-teal row"

    red_counts = [int(np.count_nonzero(segment == RED)) for segment in segments]
    assert len(set(red_counts)) == len(red_counts), "valid segments must have unique red counts"

    order = np.argsort(red_counts)
    return np.stack([segments[int(i)] for i in order], axis=0)


def _count_order_is_informative(top_to_bottom_counts: list[int]) -> bool:
    if len(top_to_bottom_counts) <= 1:
        return True
    ascending = top_to_bottom_counts == sorted(top_to_bottom_counts)
    if len(top_to_bottom_counts) == 2:
        return not ascending
    descending = top_to_bottom_counts == sorted(top_to_bottom_counts, reverse=True)
    return (not ascending) and (not descending)


def _random_segment(length: int, red_count: int) -> np.ndarray:
    segment = np.full(length, BLACK, dtype=int)
    assert 0 <= red_count <= length
    if red_count == 0:
        return segment

    mode = random.randrange(3)
    if mode == 0 or red_count == length:
        red_cols = random.sample(range(length), red_count)
    elif mode == 1:
        start = random.randint(0, length - red_count)
        red_cols = list(range(start, start + red_count))
    else:
        pivot = random.randint(0, length - 1)
        red_cols = sorted(range(length), key=lambda col: (abs(col - pivot), random.random()))[:red_count]

    segment[np.array(red_cols, dtype=int)] = RED
    return segment


def _sparse_red_noise(width: int, protected: set[int], density: float) -> np.ndarray:
    row = np.full(width, BLACK, dtype=int)
    free_cols = [c for c in range(width) if c not in protected]
    if free_cols and density > 0.0:
        mask = np.random.random(len(free_cols)) < density
        for idx, col in enumerate(free_cols):
            if mask[idx]:
                row[col] = RED
    return row


def _build_marked_row(width: int, left: int, segment: np.ndarray, valid: bool) -> np.ndarray:
    right = left + int(segment.size) + 1
    protected = set(range(left, right + 1))
    row = _sparse_red_noise(width, protected, density=random.uniform(0.03, 0.16))
    row[left + 1 : right] = segment
    if valid:
        row[left] = BLUE
        row[right] = TEAL
    else:
        row[left] = TEAL
        row[right] = BLUE
    return row


def _gen_attempt() -> np.ndarray | None:
    segment_len = random.randint(4, 11)
    valid_count = random.randint(2, min(6, segment_len - 1))
    decoy_count = random.randint(1, 3)
    min_height = valid_count + decoy_count
    height = random.randint(min_height + 1, min(30, min_height + 10))
    width = random.randint(segment_len + 2, min(30, segment_len + 14))
    left_choices = list(range(width - segment_len - 1))

    top_to_bottom_counts = random.sample(range(1, segment_len), valid_count)
    for _ in range(50):
        if _count_order_is_informative(top_to_bottom_counts):
            break
        top_to_bottom_counts = random.sample(range(1, segment_len), valid_count)
    if not _count_order_is_informative(top_to_bottom_counts):
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    background_density = random.uniform(0.01, 0.10)
    background_mask = np.random.random((height, width)) < background_density
    grid[background_mask] = RED

    all_rows = list(range(height))
    valid_rows = sorted(random.sample(all_rows, valid_count))
    remaining_rows = [r for r in all_rows if r not in valid_rows]
    decoy_rows = random.sample(remaining_rows, decoy_count)

    left_positions: dict[int, int] = {}
    for row_idx in valid_rows + decoy_rows:
        left_positions[row_idx] = random.choice(left_choices)
    if len(left_choices) > 1 and len(set(left_positions.values())) == 1 and len(left_positions) > 1:
        bumped_row = random.choice(valid_rows + decoy_rows)
        alternatives = [x for x in left_choices if x != left_positions[bumped_row]]
        left_positions[bumped_row] = random.choice(alternatives)

    for row_idx, red_count in zip(valid_rows, top_to_bottom_counts):
        segment = _random_segment(segment_len, red_count)
        grid[row_idx] = _build_marked_row(width, left_positions[row_idx], segment, valid=True)

    for row_idx in decoy_rows:
        red_count = random.randint(0, segment_len)
        segment = _random_segment(segment_len, red_count)
        grid[row_idx] = _build_marked_row(width, left_positions[row_idx], segment, valid=False)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != (valid_count, segment_len):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
