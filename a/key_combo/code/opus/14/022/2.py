from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
HIGHLIGHT = 2
FOREGROUND_CHOICES = (1, 3, 4, 5, 6, 7, 8, 9)
HEIGHT_RANGE = (7, 10)
PERIOD_RANGE = (3, 6)
REPEATS_RANGE = (4, 6)
MAX_GENERATION_ATTEMPTS = 600
MAX_TILE_ATTEMPTS = 200


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _majority_value(values: np.ndarray) -> int:
    counts = Counter(map(int, values.tolist()))
    ranked = counts.most_common()
    assert ranked, "expected at least one value"
    if len(ranked) > 1:
        assert ranked[0][1] != ranked[1][1], "majority must be unique"
    return int(ranked[0][0])


def _build_consensus(blocks: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(blocks, axis=0)
    height, width = blocks[0].shape
    consensus = np.empty((height, width), dtype=int)
    for r in range(height):
        for c in range(width):
            consensus[r, c] = _majority_value(stack[:, r, c])
    return consensus


def _enumerate_candidates(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape

    bg = _get_bg(g)
    fg_colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert len(fg_colors) == 1, "expected exactly one foreground color"

    candidates: list[dict] = []
    for period in range(1, width + 1):
        if width % period != 0:
            continue
        repeats = width // period
        if repeats < 4:
            continue

        blocks = [g[:, idx * period : (idx + 1) * period] for idx in range(repeats)]
        row_ok = True
        for r in range(height):
            row_patterns = [tuple(map(int, block[r, :])) for block in blocks]
            if Counter(row_patterns).most_common(1)[0][1] < repeats - 1:
                row_ok = False
                break
        if not row_ok:
            continue

        consensus = _build_consensus(blocks)
        mismatch_counts = [int(np.count_nonzero(block != consensus)) for block in blocks]
        bad_indices = [idx for idx, count in enumerate(mismatch_counts) if count > 0]
        if len(bad_indices) != 1:
            continue
        if not 1 <= mismatch_counts[bad_indices[0]] <= 3:
            continue

        candidates.append(
            {
                "period": period,
                "repeats": repeats,
                "consensus": consensus,
                "blocks": blocks,
                "bad_index": bad_indices[0],
                "mismatch_count": mismatch_counts[bad_indices[0]],
            }
        )
    return candidates


def _solve_instance(grid: np.ndarray) -> dict:
    candidates = _enumerate_candidates(grid)
    assert candidates, "expected at least one valid period candidate"
    best = min(candidates, key=lambda item: item["period"])
    best_period = int(best["period"])
    assert sum(int(item["period"]) == best_period for item in candidates) == 1
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the unique defective horizontal period cell and mark its differing pixels in red.

    Input structure:
    - The background is dominant and exactly one other color forms a horizontal repeated motif.
    - The full width is split into 4-6 equal period cells, all with the same unknown minimal column period.
    - Exactly one period cell is defective, differing from the consensus motif in 1-3 pixels.

    Rule:
    - Find the smallest column period whose width divides the grid and for which each row has at least all-but-one
      identical period-cell slices.
    - Split the grid into period cells and take the per-pixel majority value across cells as the consensus template.
    - Locate the unique period cell that disagrees with that template and crop that full-height cell.
    - Recolor every mismatching pixel inside the cropped cell to red (2), leaving matching pixels unchanged.

    Key insight:
    - The repeated copies define what "normal" looks like, so periodic consensus must be inferred before the single
      defective copy can be isolated and highlighted.
    """
    g = np.asarray(grid, dtype=int)
    solution = _solve_instance(g)
    block = solution["blocks"][solution["bad_index"]].copy()
    consensus = solution["consensus"]
    block[block != consensus] = HIGHLIGHT
    return block


def _exact_period(tile: np.ndarray) -> int:
    width = tile.shape[1]
    for period in range(1, width + 1):
        if width % period != 0:
            continue
        if np.array_equal(tile, np.tile(tile[:, :period], (1, width // period))):
            return period
    return width


def _column_library(height: int) -> list[tuple[int, ...]]:
    patterns: set[tuple[int, ...]] = set()

    for size in range(2, height):
        col = [0] * height
        for r in range(size):
            col[r] = 1
        patterns.add(tuple(col))

        col = [0] * height
        for r in range(height - size, height):
            col[r] = 1
        patterns.add(tuple(col))

    for length in range(2, height - 1):
        for start in range(1, height - length):
            col = [0] * height
            for r in range(start, start + length):
                col[r] = 1
            patterns.add(tuple(col))

    max_cap = min(3, height - 1)
    for top in range(1, max_cap + 1):
        for bottom in range(1, max_cap + 1):
            if top + bottom >= height:
                continue
            col = [0] * height
            for r in range(top):
                col[r] = 1
            for r in range(height - bottom, height):
                col[r] = 1
            patterns.add(tuple(col))

    usable = [
        pattern
        for pattern in patterns
        if 1 < sum(pattern) < height and 0 in pattern and 1 in pattern
    ]
    return sorted(usable)


def _gen_tile(height: int, period: int, fg: int) -> np.ndarray | None:
    library = _column_library(height)
    if len(library) < period:
        return None

    for _ in range(MAX_TILE_ATTEMPTS):
        chosen = random.sample(library, period)
        random.shuffle(chosen)

        tile = np.zeros((height, period), dtype=int)
        for c, pattern in enumerate(chosen):
            tile[:, c] = np.array(pattern, dtype=int) * fg

        density = float(np.count_nonzero(tile)) / tile.size
        row_activity = np.count_nonzero(tile == fg, axis=1)
        if not (0.22 <= density <= 0.45):
            continue
        if _exact_period(tile) != period:
            continue
        if len({tuple(int(v) for v in tile[:, c]) for c in range(period)}) != period:
            continue
        if np.count_nonzero(row_activity > 0) < 4:
            continue
        if np.count_nonzero(row_activity < period) < 4:
            continue
        if len(set(map(int, row_activity.tolist()))) < 3:
            continue
        if not np.any(tile[0, :] == fg):
            continue
        if not np.any(tile[-1, :] == fg):
            continue
        return tile
    return None


def _repeat_tile(tile: np.ndarray, repeats: int) -> np.ndarray:
    return np.tile(tile, (1, repeats))


def _anomaly_options(tile: np.ndarray, fg: int) -> list[tuple[str, int, int]]:
    fg_count = int(np.count_nonzero(tile == fg))
    bg_count = int(np.count_nonzero(tile == BACKGROUND))
    options: list[tuple[str, int, int]] = []

    for count in (1, 2, 3):
        if fg_count >= count:
            options.extend([("remove", count, 0)] * 2)
        if bg_count >= count:
            options.extend([("add", 0, count)] * 2)

    if fg_count >= 1 and bg_count >= 1:
        options.extend([("mixed", 1, 1)] * 4)
    if fg_count >= 2 and bg_count >= 1:
        options.extend([("mixed", 2, 1)] * 2)
    if fg_count >= 1 and bg_count >= 2:
        options.extend([("mixed", 1, 2)] * 2)
    return options


def _apply_anomaly(tile: np.ndarray, fg: int) -> tuple[np.ndarray, str, set[tuple[int, int]]]:
    options = _anomaly_options(tile, fg)
    assert options, "expected at least one anomaly option"
    kind, removes, adds = random.choice(options)

    defective = tile.copy()
    mismatch_coords: set[tuple[int, int]] = set()
    fg_coords = [tuple(map(int, rc)) for rc in np.argwhere(tile == fg)]
    bg_coords = [tuple(map(int, rc)) for rc in np.argwhere(tile == BACKGROUND)]

    if removes:
        for r, c in random.sample(fg_coords, removes):
            defective[r, c] = BACKGROUND
            mismatch_coords.add((r, c))
    if adds:
        choices = [coord for coord in bg_coords if coord not in mismatch_coords]
        for r, c in random.sample(choices, adds):
            defective[r, c] = fg
            mismatch_coords.add((r, c))

    assert 1 <= len(mismatch_coords) <= 3
    assert np.count_nonzero(defective != tile) == len(mismatch_coords)
    return defective, kind, mismatch_coords


def _validate_instance(
    grid: np.ndarray,
    tile: np.ndarray,
    period: int,
    repeats: int,
    bad_index: int,
    mismatches: set[tuple[int, int]],
) -> bool:
    try:
        solution = _solve_instance(grid)
    except AssertionError:
        return False

    if int(solution["period"]) != period or int(solution["repeats"]) != repeats:
        return False
    if int(solution["bad_index"]) != bad_index:
        return False
    if int(solution["mismatch_count"]) != len(mismatches):
        return False
    if not np.array_equal(solution["consensus"], tile):
        return False

    smaller = [
        item
        for item in _enumerate_candidates(grid)
        if int(item["period"]) < period
    ]
    if smaller:
        return False

    if not mismatches:
        return False

    out = main(grid)
    if out.shape != (grid.shape[0], period):
        return False
    if not np.array_equal(out == HIGHLIGHT, solution["blocks"][bad_index] != tile):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    period = random.randint(*PERIOD_RANGE)
    repeats_choices = [count for count in range(REPEATS_RANGE[0], REPEATS_RANGE[1] + 1) if count * period <= 30]
    if not repeats_choices:
        return None
    repeats = random.choice(repeats_choices)
    fg = random.choice(FOREGROUND_CHOICES)

    tile = _gen_tile(height, period, fg)
    if tile is None:
        return None

    grid = _repeat_tile(tile, repeats)
    bad_index = random.randrange(repeats)
    defective, bad_kind, mismatches = _apply_anomaly(tile, fg)
    grid[:, bad_index * period : (bad_index + 1) * period] = defective

    if not _validate_instance(grid, tile, period, repeats, bad_index, mismatches):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-outlier instance")
