from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _active_width(grid: np.ndarray) -> int:
    cols = np.where(np.any(grid != 0, axis=0))[0]
    assert cols.size > 0, "expected a nonempty foreground pattern"
    return int(cols[-1]) + 1


def _minimal_column_period(observed: np.ndarray) -> int:
    _, width = observed.shape
    for period in range(1, width + 1):
        if all(np.array_equal(observed[:, c], observed[:, c % period]) for c in range(width)):
            return period
    raise AssertionError("column period must exist")


def _tile_right(observed: np.ndarray, full_width: int, period: int) -> np.ndarray:
    tile = observed[:, :period]
    reps = (full_width + period - 1) // period
    return np.tile(tile, (1, reps))[:, :full_width].astype(int)


def _nonzero_counts(grid: np.ndarray) -> Counter[int]:
    cells = grid[grid != 0]
    return Counter(map(int, cells.ravel()))


def _least_frequent_color(grid: np.ndarray) -> int:
    counts = _nonzero_counts(grid)
    assert counts, "tiled grid must contain foreground colors"
    minimum = min(counts.values())
    winners = [color for color, count in counts.items() if count == minimum]
    assert len(winners) == 1, "least frequent foreground color must be unique"
    return int(winners[0])


def _first_occurrence_column(grid: np.ndarray, color: int) -> int:
    cols = np.where(np.any(grid == int(color), axis=0))[0]
    assert cols.size > 0, "phase color must appear in the grid"
    return int(cols[0])


def _striped_output(full: np.ndarray, parity: int) -> np.ndarray:
    out = full.copy()
    keep = (np.arange(full.shape[1]) % 2) == int(parity)
    out[:, ~keep] = 0
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the observed periodic left pattern across the grid, then keep only columns matching a phase set by the rarest color.

    Input structure:
    - Black (0) is the background.
    - A multi-color foreground pattern occupies a left prefix of the grid, while the remaining right-side columns are blank.
    - The visible pattern already follows an exact column-periodic structure, but only a partial prefix of the full tiling is shown.

    Rule:
    - Find the rightmost foreground column and treat the grid prefix up to that column as the observed pattern.
    - Infer the smallest positive column period whose repeated columns exactly reproduce that observed prefix, then repeat that period to fill the whole grid width.
    - In the fully tiled grid, find the unique least frequent foreground color and read the first column where that color appears.
    - Keep every column whose parity matches that first-occurrence column, and erase every opposite-parity column to black.

    Key insight:
    - The task is not just completion or just striping: the final stripes come from a full periodic extension, and the kept parity is anchored by the earliest column of the rarest tiled color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    observed_width = _active_width(g)
    observed = g[:, :observed_width]
    period = _minimal_column_period(observed)
    full = _tile_right(observed, g.shape[1], period)

    phase_color = _least_frequent_color(full)
    phase_parity = _first_occurrence_column(full, phase_color) % 2
    return _striped_output(full, phase_parity)


HEIGHT_RANGE = (5, 8)
WIDTH_RANGE = (10, 18)
PERIOD_RANGE = (2, 4)
VISIBLE_COLOR_COUNT_CHOICES = (2, 3)
MAX_TILE_ATTEMPTS = 400
MAX_GENERATION_ATTEMPTS = 600


def _build_period_tile(
    height: int,
    period: int,
    palette: list[int],
    phase_color: int,
    phase_col: int,
) -> np.ndarray | None:
    other_colors = [color for color in palette if color != phase_color]

    for _ in range(MAX_TILE_ATTEMPTS):
        tile = np.zeros((height, period), dtype=int)

        phase_rows = random.sample(range(height), random.randint(1, min(2, height)))
        for r in phase_rows:
            tile[r, phase_col] = phase_color

        later_cells = [(r, c) for c in range(phase_col, period) for r in range(height) if tile[r, c] == 0]
        random.shuffle(later_cells)
        for r, c in later_cells[: random.randint(0, min(1, len(later_cells)))]:
            tile[r, c] = phase_color

        failed = False
        for color in other_colors:
            need = random.randint(max(2, len(phase_rows) + 1), max(3, height))
            free = [(r, c) for r in range(height) for c in range(period) if tile[r, c] == 0]
            if len(free) < need:
                failed = True
                break
            for r, c in random.sample(free, need):
                tile[r, c] = color
        if failed:
            continue

        for r in range(height):
            for c in range(period):
                if tile[r, c] != 0:
                    continue
                roll = random.random()
                if roll < 0.45:
                    tile[r, c] = random.choice(other_colors)
                elif roll < 0.55 and c >= phase_col:
                    tile[r, c] = phase_color

        counts = _nonzero_counts(tile)
        if set(counts) != set(palette):
            continue
        if _first_occurrence_column(tile, phase_color) != phase_col:
            continue
        if counts[phase_color] != min(counts.values()):
            continue
        if sum(count == counts[phase_color] for count in counts.values()) != 1:
            continue
        if any(counts[color] <= counts[phase_color] for color in other_colors):
            continue
        if _minimal_column_period(tile) != period:
            continue

        foreground = int(np.count_nonzero(tile))
        if not (height * period * 0.25 <= foreground <= height * period * 0.75):
            continue
        if not np.any(tile[:, ::2] != 0):
            continue
        if not np.any(tile[:, 1::2] != 0):
            continue
        return tile

    return None


def _candidate_visible_widths(width: int, period: int, phase_col: int) -> list[int]:
    min_visible = max(period + 2, (width + 2) // 3)
    max_visible = min(width - 2, width // 2 + 1)
    return [
        visible
        for visible in range(min_visible, max_visible + 1)
        if visible > phase_col and visible % period != 0
    ]


def _validate_sample(
    inp: np.ndarray,
    *,
    full: np.ndarray,
    visible_width: int,
    period: int,
    phase_color: int,
) -> bool:
    observed = inp[:, :visible_width]
    if _minimal_column_period(observed) != period:
        return False

    input_counts = _nonzero_counts(observed)
    if phase_color not in input_counts:
        return False
    input_min = min(input_counts.values())
    if input_counts[phase_color] != input_min:
        return False
    if sum(count == input_min for count in input_counts.values()) != 1:
        return False

    if _least_frequent_color(full) != phase_color:
        return False
    phase_parity = _first_occurrence_column(full, phase_color) % 2
    expected = _striped_output(full, phase_parity)

    if np.array_equal(expected, full):
        return False
    if not np.any(full[:, visible_width:] != 0):
        return False
    if not np.any(expected[:, visible_width:] != 0):
        return False
    if not np.any(full[:, :visible_width][:, (np.arange(visible_width) % 2) != phase_parity] != 0):
        return False
    if not np.any(full[:, :visible_width][:, (np.arange(visible_width) % 2) == phase_parity] != 0):
        return False
    if not np.array_equal(main(inp), expected):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height = random.randint(*HEIGHT_RANGE)
        width = random.randint(*WIDTH_RANGE)
        max_period = min(PERIOD_RANGE[1], width - 2)
        if max_period < PERIOD_RANGE[0]:
            continue
        period = random.randint(PERIOD_RANGE[0], max_period)

        color_count = random.choice(VISIBLE_COLOR_COUNT_CHOICES)
        palette = random.sample(range(1, 10), color_count)
        phase_color = random.choice(palette)

        desired_parity = random.randint(0, 1)
        phase_candidates = [c for c in range(period) if c % 2 == desired_parity]
        if not phase_candidates:
            continue
        phase_col = random.choice(phase_candidates)

        tile = _build_period_tile(height, period, palette, phase_color, phase_col)
        if tile is None:
            continue

        full = _tile_right(tile, width, period)
        if _least_frequent_color(full) != phase_color:
            continue

        visible_candidates = _candidate_visible_widths(width, period, phase_col)
        if not visible_candidates:
            continue
        visible_width = random.choice(visible_candidates)

        inp = np.zeros((height, width), dtype=int)
        inp[:, :visible_width] = full[:, :visible_width]

        if _validate_sample(
            inp,
            full=full,
            visible_width=visible_width,
            period=period,
            phase_color=phase_color,
        ):
            return inp

    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
