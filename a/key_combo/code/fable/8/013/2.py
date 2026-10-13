import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress identical row runs, then reorder the kept rows by run length from longest to shortest.

    Input structure:
    - The grid is a vertical stack of row segments.
    - Inside each segment, every row is exactly identical to the others in that same segment.
    - Adjacent segments use different row patterns, and the width is unchanged across the whole grid.

    Rule:
    - Scan from top to bottom and compress each maximal run of identical adjacent rows into one representative row.
    - Record the original length of every run.
    - Output the representative rows sorted by run length in descending order; if lengths tie, keep their original top-to-bottom order.

    Key insight:
    - The repeated copies are not discarded noise: the size of each repeated run is the sorting key for the compressed rows.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, _width = g.shape
    assert height >= 1, "grid must contain at least one row"

    runs: list[tuple[int, int, np.ndarray]] = []
    start = 0
    while start < height:
        end = start + 1
        while end < height and np.array_equal(g[end], g[start]):
            end += 1
        runs.append((end - start, start, g[start].copy()))
        start = end

    ordered = sorted(runs, key=lambda item: (-item[0], item[1]))
    return np.stack([row for _length, _start, row in ordered], axis=0)


def _sample_run_lengths(segment_count: int) -> list[int]:
    pool = list(range(2, 11))
    for _ in range(200):
        lengths = random.sample(pool, segment_count)
        if sum(lengths) <= 30:
            random.shuffle(lengths)
            return lengths
    lengths = list(range(2, 2 + segment_count))
    random.shuffle(lengths)
    return lengths


def _sample_cut_points(width: int, block_count: int) -> list[int]:
    interior = sorted(random.sample(range(1, width), block_count - 1))
    return [0, *interior, width]


def _build_row_pattern(width: int, used: set[bytes]) -> np.ndarray:
    for _ in range(300):
        row = np.zeros(width, dtype=int)
        block_count = random.randint(2, min(5, width))
        cuts = _sample_cut_points(width, block_count)

        palette = random.sample(range(1, 10), random.randint(2, min(6, block_count + 1)))
        prev_color = None
        for left, right in zip(cuts, cuts[1:]):
            choices = [color for color in palette if color != prev_color]
            color = random.choice(choices)
            row[left:right] = color
            prev_color = color

        if width >= 8 and random.random() < 0.6:
            accent_color = random.choice([color for color in range(1, 10) if color != row[0]])
            accent_count = random.randint(1, min(3, width // 4))
            accent_positions = random.sample(range(width), accent_count)
            for pos in accent_positions:
                row[pos] = accent_color

        key = row.tobytes()
        if key not in used:
            used.add(key)
            return row

    raise RuntimeError("failed to generate a distinct row pattern")


def generate_input() -> np.ndarray:
    segment_count = random.randint(3, 6)
    run_lengths = _sample_run_lengths(segment_count)
    width = random.randint(8, 20)

    used: set[bytes] = set()
    rows = [_build_row_pattern(width, used) for _ in range(segment_count)]

    stacked_rows = []
    for row, run_length in zip(rows, run_lengths):
        stacked_rows.extend([row] * run_length)

    grid = np.stack(stacked_rows, axis=0)
    assert grid.ndim == 2 and max(grid.shape) <= 30
    return grid
