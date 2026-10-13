from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
NON_BLACK_COLORS = tuple(range(1, 10))
GEN_ATTEMPTS = 200


def _mode_info(values: np.ndarray) -> tuple[int | None, int]:
    counts = Counter(map(int, values.tolist()))
    best_count = max(counts.values())
    best_colors = [color for color, count in counts.items() if count == best_count]
    if len(best_colors) != 1:
        return None, best_count
    return int(best_colors[0]), int(best_count)


def _orientation_score(grid: np.ndarray, *, by_rows: bool) -> int:
    total = 0
    if by_rows:
        for row in range(grid.shape[0]):
            _color, count = _mode_info(grid[row, :])
            total += count
    else:
        for col in range(grid.shape[1]):
            _color, count = _mode_info(grid[:, col])
            total += count
    return int(total)


def _extract_stripes(grid: np.ndarray, *, by_rows: bool) -> list[tuple[int, int]]:
    stripes: list[tuple[int, int]] = []
    line_count = grid.shape[0] if by_rows else grid.shape[1]
    line_len = grid.shape[1] if by_rows else grid.shape[0]
    for index in range(line_count):
        values = grid[index, :] if by_rows else grid[:, index]
        color, count = _mode_info(values)
        if color is None or color == BLACK:
            continue
        if count * 2 <= line_len:
            continue
        stripes.append((int(index), int(color)))
    return stripes


def _neighbors8(r: int, c: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    height, width = shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < height and 0 <= cc < width:
                out.append((rr, cc))
    return out


def _singleton_pixels(grid: np.ndarray) -> list[tuple[int, int, int]]:
    singletons: list[tuple[int, int, int]] = []
    height, width = grid.shape
    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if color == BLACK:
                continue
            if all(int(grid[rr, cc]) == BLACK for rr, cc in _neighbors8(r, c, grid.shape)):
                singletons.append((r, c, color))
    return singletons


def _target_line(source_index: int, candidates: list[int]) -> int:
    return min(candidates, key=lambda index: (abs(index - source_index), index))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Denoise stripe columns, then slide each loose pixel until it first touches the nearest same-color stripe.

    Input structure:
    - The grid contains sparse full-height stripes separated by black gap columns, plus a few loose single pixels.
    - Each true stripe column has 1-2 noisy replacements, but its real color remains the strict unique column mode.
    - Loose pixels sit only in non-stripe columns; some match a stripe's true color and at least one does not.

    Rule:
    - Compare row-mode and column-mode support totals; the larger one gives the stripe orientation.
    - Repaint every detected stripe line to its unique dominant color, removing stripe noise.
    - Ignore every nonzero pixel that lies on a stripe line, and treat each isolated non-stripe pixel as a loose pixel.
    - If a loose pixel's color matches a denoised stripe, slide it along the perpendicular gap direction to the nearest
      stripe of that color and place it in the last gap cell before contact; otherwise delete it.

    Key insight:
    - Stripe denoising must happen first because the recovered stripe colors, not the visible noisy outliers, determine
      which loose pixels are valid and where they are allowed to settle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_score = _orientation_score(g, by_rows=True)
    col_score = _orientation_score(g, by_rows=False)
    by_rows = row_score > col_score

    stripes = _extract_stripes(g, by_rows=by_rows)
    assert stripes, "expected at least one stripe line"

    stripe_by_color: dict[int, list[int]] = {}
    for line_index, color in stripes:
        stripe_by_color.setdefault(int(color), []).append(int(line_index))

    out = np.zeros_like(g)
    stripe_lines = {line_index for line_index, _color in stripes}
    for line_index, color in stripes:
        if by_rows:
            out[line_index, :] = int(color)
        else:
            out[:, line_index] = int(color)

    residual = g.copy()
    for line_index in stripe_lines:
        if by_rows:
            residual[line_index, :] = BLACK
        else:
            residual[:, line_index] = BLACK

    for r, c, color in _singleton_pixels(residual):
        if color not in stripe_by_color:
            continue
        target = _target_line(r if by_rows else c, stripe_by_color[color])
        if by_rows:
            dest_r = target - 1 if r < target else target + 1
            dest_c = c
        else:
            dest_r = r
            dest_c = target - 1 if c < target else target + 1
        if 0 <= dest_r < out.shape[0] and 0 <= dest_c < out.shape[1]:
            out[dest_r, dest_c] = int(color)

    return out


def _sample_spaced_rows(height: int, count: int) -> list[int] | None:
    for _ in range(80):
        rows = sorted(random.sample(range(height), count))
        if all(rows[i + 1] - rows[i] > 1 for i in range(len(rows) - 1)):
            return rows
    return None


def _stripe_columns(stripe_count: int, gaps: list[int], left_margin: int) -> list[int]:
    cols: list[int] = []
    cur = left_margin
    for index in range(stripe_count):
        cols.append(cur)
        cur += 1
        if index < stripe_count - 1:
            cur += gaps[index]
    return cols


def _matched_options(stripe_cols: list[int], width: int, stripe_index: int) -> list[tuple[int, list[int]]]:
    col = stripe_cols[stripe_index]
    prev_stripe = None if stripe_index == 0 else stripe_cols[stripe_index - 1]
    next_stripe = None if stripe_index == len(stripe_cols) - 1 else stripe_cols[stripe_index + 1]
    options: list[tuple[int, list[int]]] = []
    left_start = 0 if prev_stripe is None else prev_stripe + 2
    left_end = col - 2
    if left_start <= left_end:
        options.append((col - 1, list(range(left_start, left_end + 1))))
    right_start = col + 2
    right_end = width - 1 if next_stripe is None else next_stripe - 2
    if right_start <= right_end:
        options.append((col + 1, list(range(right_start, right_end + 1))))
    return options


def _build_clean_grid(height: int, width: int, stripe_cols: list[int], stripe_colors: list[int]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for col, color in zip(stripe_cols, stripe_colors):
        grid[:, col] = int(color)
    return grid


def _add_noise(
    grid: np.ndarray,
    stripe_cols: list[int],
    stripe_colors: list[int],
    forced_noisy_stripe: int,
    forced_noise_color: int,
) -> None:
    height = grid.shape[0]
    for stripe_index, (col, true_color) in enumerate(zip(stripe_cols, stripe_colors)):
        rows = random.sample(range(height), random.randint(1, 2))
        noise_colors: list[int] = []
        if stripe_index == forced_noisy_stripe:
            noise_colors.append(int(forced_noise_color))
        while len(noise_colors) < len(rows):
            pool = [BLACK] + [color for color in stripe_colors if color != true_color]
            noise_colors.append(int(random.choice(pool)))
        random.shuffle(noise_colors)
        for row, noise_color in zip(rows, noise_colors):
            grid[row, col] = int(noise_color)


def _choose_unmatched_colors(stripe_colors: list[int], count: int) -> list[int]:
    pool = [color for color in NON_BLACK_COLORS if color not in stripe_colors]
    random.shuffle(pool)
    return pool[:count]


def _safe_nonstripe_columns(stripe_cols: list[int], width: int) -> list[int]:
    return [
        col
        for col in range(width)
        if all(abs(col - stripe_col) > 1 for stripe_col in stripe_cols)
    ]


def _generate_case() -> np.ndarray | None:
    height = random.randint(11, 14)
    stripe_count = random.randint(3, 4)
    gaps = [random.randint(2, 3) for _ in range(stripe_count - 1)]
    if 3 not in gaps:
        return None
    left_margin = 2
    right_margin = 2
    width = left_margin + right_margin + stripe_count + sum(gaps)
    if width > 18:
        return None

    stripe_cols = _stripe_columns(stripe_count, gaps, left_margin)
    stripe_colors = random.sample(list(NON_BLACK_COLORS), stripe_count)
    safe_cols = _safe_nonstripe_columns(stripe_cols, width)
    if not safe_cols:
        return None

    total_loose = random.randint(3, 5)
    unmatched_count = 1 if total_loose == 3 else random.randint(1, 2)
    matched_count = total_loose - unmatched_count
    if matched_count < 2:
        return None

    rows = _sample_spaced_rows(height, total_loose)
    if rows is None:
        return None

    base_grid = _build_clean_grid(height, width, stripe_cols, stripe_colors)

    forced_target = random.choice([index for index, gap in enumerate(gaps) if gap == 3])
    forced_noisy_stripe = forced_target + 1
    forced_color = stripe_colors[forced_target]
    forced_row = rows[0]
    forced_source_col = stripe_cols[forced_target] + 2

    input_grid = base_grid.copy()
    _add_noise(input_grid, stripe_cols, stripe_colors, forced_noisy_stripe, forced_color)

    loose_sources: list[tuple[int, int, int]] = [(forced_row, forced_source_col, forced_color)]
    for row in rows[1:matched_count]:
        stripe_index = random.randrange(stripe_count)
        options = _matched_options(stripe_cols, width, stripe_index)
        if not options:
            return None
        _dest_col, candidates = random.choice(options)
        source_col = int(random.choice(candidates))
        color = stripe_colors[stripe_index]
        loose_sources.append((row, source_col, color))

    unmatched_colors = _choose_unmatched_colors(stripe_colors, unmatched_count)
    if len(unmatched_colors) != unmatched_count:
        return None
    for row, color in zip(rows[matched_count:], unmatched_colors):
        loose_sources.append((row, int(random.choice(safe_cols)), color))

    for r, c, color in loose_sources:
        if input_grid[r, c] != BLACK:
            return None
        input_grid[r, c] = int(color)

    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 7")
