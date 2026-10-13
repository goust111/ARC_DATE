import random

import numpy as np


BLACK = 0
ORANGE = 7
STRIPE_COLORS = [1, 2, 3, 4, 5, 6, 8, 9]


def main(grid: np.ndarray) -> np.ndarray:
    """Sample each top probe's downward stripe-color sequence and draw it as a compact bottom column.

    Input structure:
    The grid has a black background, several colored horizontal stripes spanning contiguous ranges of
    columns, and one or more single orange probe pixels on the top row. Stripes may be 1 to 3 rows
    thick, may have black gaps between them, and may cover different horizontal intervals. The lower
    part of the grid contains enough black rows to receive the sampled color columns.

    Rule:
    For every orange probe in the top row, scan downward in that same column, excluding the probe
    cell. Record each non-black color encountered, collapsing vertically adjacent cells of the same
    color into one stripe hit. In the output, remove the orange probes, keep all original horizontal
    stripes unchanged, and draw the recorded sequence in the probe column from the bottom upward:
    the first color encountered is placed on the last row, the second on the row above it, and so on.

    Key insight:
    The orange pixels are sampling probes, not decoration. Each probe records only the stripes that
    cross its column, stripe thickness counts once, and the resulting bottom marker reverses vertical
    position while preserving encounter order from bottom to top.
    """
    out = np.array(grid, copy=True)
    if out.ndim != 2 or out.shape[0] == 0 or out.shape[1] == 0:
        return out

    rows, _ = out.shape
    probe_cols = np.flatnonzero(grid[0] == ORANGE)
    out[0, probe_cols] = BLACK

    for col in probe_cols:
        seen = []
        last = BLACK
        for row in range(1, rows):
            color = int(grid[row, col])
            if color == BLACK:
                last = BLACK
            elif color != last:
                seen.append(color)
                last = color

        for offset, color in enumerate(seen):
            target_row = rows - 1 - offset
            if target_row <= 0:
                break
            out[target_row, col] = color

    return out


def _choose_stripe_layout(height: int, stripe_count: int, bottom_rows: int):
    usable_rows = list(range(2, height - bottom_rows))
    layout = []
    cursor = 2 + random.randint(0, 1)

    for index in range(stripe_count):
        remaining = stripe_count - index
        max_start = height - bottom_rows - remaining
        if cursor > max_start:
            break

        top = random.randint(cursor, max_start)
        max_thick = min(3, height - bottom_rows - top - (remaining - 1))
        thick = random.randint(1, max(1, max_thick))
        layout.append((top, thick))
        cursor = top + thick + random.randint(0, 2)

    if len(layout) < stripe_count:
        free = usable_rows[:]
        random.shuffle(free)
        layout = sorted((row, 1) for row in free[:stripe_count])

    return layout[:stripe_count]


def generate_input() -> np.ndarray:
    height = random.randint(12, 30)
    width = random.randint(8, 30)
    stripe_count = random.randint(3, min(7, max(3, height // 3)))
    bottom_rows = random.randint(stripe_count, min(height - 5, stripe_count + 5))

    min_needed = 2 + stripe_count + bottom_rows
    if height < min_needed:
        height = min(30, min_needed)
        bottom_rows = min(bottom_rows, height - 2 - stripe_count)

    grid = np.zeros((height, width), dtype=int)

    colors = random.sample(STRIPE_COLORS, stripe_count)
    row_layout = _choose_stripe_layout(height, stripe_count, bottom_rows)

    for color, (top, thick) in zip(colors, row_layout):
        span_min = max(2, width // 4)
        span_len = random.randint(span_min, width)
        start = random.randint(0, width - span_len)
        end = start + span_len
        grid[top : top + thick, start:end] = color

    probe_count = random.randint(2, min(width, max(3, width // 2)))
    probe_cols = random.sample(range(width), probe_count)
    for col in probe_cols:
        grid[0, col] = ORANGE

    return grid
