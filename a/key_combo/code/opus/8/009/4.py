from __future__ import annotations

import numpy as np


BG = 0
BLUE = 1
MARKER_COLORS = (2, 3)
GRID_SIZE = 10
MAX_GENERATION_ATTEMPTS = 500


def _shift_row(row: np.ndarray, dc: int) -> np.ndarray:
    out = np.full(row.shape, BG, dtype=int)
    cols = np.flatnonzero(row != BG)
    dst = cols + int(dc)
    keep = (0 <= dst) & (dst < row.size)
    out[dst[keep]] = row[cols[keep]]
    return out


def _observed_height(grid: np.ndarray) -> int:
    blue_rows = np.flatnonzero(np.any(grid == BLUE, axis=1))
    assert blue_rows.size > 0, "expected at least one blue row"
    return int(blue_rows[-1] + 1)


def _infer_period_and_shift(observed: np.ndarray) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for period in (2, 3):
        if observed.shape[0] <= period:
            continue
        for shift in (1, 2):
            if all(
                np.array_equal(_shift_row(observed[r - period], shift), observed[r])
                for r in range(period, observed.shape[0])
            ):
                candidates.append((period, shift))
    assert candidates, "expected a valid translated row period"
    min_period = min(period for period, _shift in candidates)
    best = [cand for cand in candidates if cand[0] == min_period]
    assert len(best) == 1, f"expected a unique minimal candidate, got {candidates}"
    return best[0]


def _complete_blue_pattern(grid: np.ndarray) -> tuple[np.ndarray, int]:
    h = _observed_height(grid)
    observed = (grid[:h] == BLUE).astype(int)
    period, shift = _infer_period_and_shift(observed)

    blue = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    blue[:h] = observed
    for r in range(h, GRID_SIZE):
        blue[r] = _shift_row(blue[r - period], shift)
    return blue, h


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the translated periodic blue rows and keep only markers that land on new pixels.

    Input structure:
    - The grid is 10x10 with black background.
    - The top observed rows contain blue pixels whose row pattern repeats after a
      vertical period of 2 or 3, shifting right by 1 or 2 columns each cycle.
    - Lower rows contain only isolated red or green marker pixels on otherwise
      blank rows.

    Rule:
    - Infer the observed blue height, the minimal vertical period, and the
      associated horizontal shift.
    - Extend the blue row pattern downward to fill the missing rows by copying
      the row one period above and shifting it right, clipping anything that
      leaves the grid.
    - Preserve the original observed blue rows unchanged.
    - In the extrapolated region, replace a blue completion pixel with a red or
      green marker when a marker sits exactly on that completed location; erase
      every marker that does not coincide with a completed blue pixel.

    Key insight:
    - The markers are not independent objects: they matter only after the blue
      periodic completion is inferred, because overlap with the newly completed
      pattern decides which markers survive and which are erased.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected a 10x10 grid"

    blue_pattern, observed_height = _complete_blue_pattern(g)
    out = blue_pattern.copy()
    for r, c in np.argwhere(np.isin(g, MARKER_COLORS)):
        r = int(r)
        c = int(c)
        if r >= observed_height and out[r, c] == BLUE:
            out[r, c] = int(g[r, c])
    return out


def _candidate_pairs(observed: np.ndarray) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for period in (2, 3):
        if observed.shape[0] <= period:
            continue
        for shift in (1, 2):
            if all(
                np.array_equal(_shift_row(observed[r - period], shift), observed[r])
                for r in range(period, observed.shape[0])
            ):
                pairs.append((period, shift))
    return pairs


def _sample_base_row(rng: np.random.Generator, anchor_max: int) -> np.ndarray:
    row = np.zeros(GRID_SIZE, dtype=int)
    anchor = int(rng.integers(0, anchor_max + 1))
    count = int(rng.integers(2, 5))

    if rng.random() < 0.5:
        start_lo = max(0, anchor - count + 1)
        start_hi = min(anchor, GRID_SIZE - count)
        if start_lo <= start_hi:
            start = int(rng.integers(start_lo, start_hi + 1))
            cols = list(range(start, start + count))
        else:
            cols = [anchor]
    else:
        pool = list(range(GRID_SIZE))
        cols = [anchor]
        extras = [c for c in pool if c != anchor]
        rng.shuffle(extras)
        cols.extend(extras[: count - 1])
        cols.sort()

    row[cols] = BLUE
    return row


def _build_blue_output(base_rows: np.ndarray, period: int, shift: int) -> np.ndarray:
    out = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    out[:period] = base_rows
    for r in range(period, GRID_SIZE):
        out[r] = _shift_row(out[r - period], shift)
    return out


def _marker_positions_with_spacing(
    rng: np.random.Generator,
    candidates: list[tuple[int, int]],
    count: int,
    blocked: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    blocked = set() if blocked is None else set(blocked)
    shuffled = candidates.copy()
    rng.shuffle(shuffled)
    picked: list[tuple[int, int]] = []
    for pos in shuffled:
        if pos in blocked:
            continue
        if any(max(abs(pos[0] - q[0]), abs(pos[1] - q[1])) <= 1 for q in picked):
            continue
        picked.append(pos)
        if len(picked) == count:
            return picked
    if len(candidates) < count:
        return None
    remaining = [pos for pos in shuffled if pos not in blocked and pos not in picked]
    for pos in remaining:
        picked.append(pos)
        if len(picked) == count:
            return picked
    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    period = int(rng.choice([2, 3]))
    shift = int(rng.choice([1, 2]))
    min_height = max(5, 2 * period + 1)
    observed_height = int(rng.integers(min_height, 8))

    base_rows = np.zeros((period, GRID_SIZE), dtype=int)
    for k in range(period):
        max_observed_cycles = (observed_height - 1 - k) // period
        anchor_max = GRID_SIZE - 1 - max_observed_cycles * shift
        if anchor_max < 0:
            return None
        base_rows[k] = _sample_base_row(rng, anchor_max)

    blue_output = _build_blue_output(base_rows, period, shift)
    if any(not np.any(blue_output[r] == BLUE) for r in range(observed_height)):
        return None
    if not np.any(blue_output[observed_height:] == BLUE):
        return None

    observed = blue_output[:observed_height].copy()
    if _candidate_pairs(observed) != [(period, shift)]:
        return None

    input_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    input_grid[:observed_height] = observed

    lower_rows = np.arange(observed_height, GRID_SIZE)
    overlap_positions = [(int(r), int(c)) for r, c in np.argwhere(blue_output[observed_height:] == BLUE)]
    overlap_positions = [(r + observed_height, c) for r, c in overlap_positions]
    empty_positions = [(r, c) for r in lower_rows for c in range(GRID_SIZE) if blue_output[r, c] == BG]

    total_markers = int(rng.integers(3, 7))
    max_hits = min(len(overlap_positions), total_markers - 1, 3)
    if max_hits < 1 or len(empty_positions) < total_markers - 1:
        return None
    hit_count = int(rng.integers(1, max_hits + 1))
    miss_count = total_markers - hit_count

    hit_positions = _marker_positions_with_spacing(rng, overlap_positions, hit_count)
    if hit_positions is None:
        return None
    miss_positions = _marker_positions_with_spacing(
        rng, empty_positions, miss_count, blocked=set(hit_positions)
    )
    if miss_positions is None:
        return None

    marker_positions = hit_positions + miss_positions
    rng.shuffle(marker_positions)
    colors = rng.choice(MARKER_COLORS, size=len(marker_positions), replace=True)
    for (r, c), color in zip(marker_positions, colors):
        input_grid[r, c] = int(color)

    expected = blue_output.copy()
    hit_set = set(hit_positions)
    for (r, c), color in zip(marker_positions, colors):
        if (r, c) in hit_set:
            expected[r, c] = int(color)

    try:
        actual = main(input_grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-completion marker puzzle")
