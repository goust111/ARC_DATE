from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
MAGENTA = 6

_GEN_MAX_ATTEMPTS = 400
_PATTERN_MAX_ATTEMPTS = 200


def _repeat_columns(tile: np.ndarray, width: int) -> np.ndarray:
    """Repeat a column tile until the requested width is filled."""
    h, p = tile.shape
    reps = (int(width) + p - 1) // p
    return np.tile(tile, (1, reps))[:, :width].reshape(h, width)


def _minimal_period(grid: np.ndarray) -> int:
    """Return the smallest column period consistent with the observed columns."""
    g = np.asarray(grid, dtype=int)
    w = int(g.shape[1])
    if w == 0:
        return 1
    for p in range(1, w + 1):
        if np.array_equal(g, g[:, np.arange(w) % p]):
            return p
    return w


def _xor_result(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    half = h // 2
    upper = g[:half] != BLACK
    lower = g[half:] != BLACK
    out = np.zeros((half, w), dtype=int)
    out[np.logical_xor(upper, lower)] = MAGENTA
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the hidden periodic mask by XORing the aligned half-grids, then complete it across all columns.

    Input structure:
    - The input is an even-height black-background grid split into equal upper and lower halves.
    - The upper half uses blue foreground cells and the lower half uses red foreground cells.
    - Some aligned foreground cells overlap across the halves while others appear in exactly one half.

    Rule:
    - Compare the two halves cell-by-cell after aligning them by row and column.
    - Produce a magenta intermediate grid where exactly one half has foreground and both-background / both-foreground stay black.
    - Restrict attention to the non-empty left prefix of that intermediate grid, find its smallest column period, and repeat that period to the full output width.

    Key insight:
    - Shared clutter cancels under the half-grid XOR, revealing a shorter periodic prefix that can then be tiled horizontally.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] % 2 == 0

    xor_grid = _xor_result(g)
    active_cols = np.where(np.any(xor_grid != BLACK, axis=0))[0]
    if active_cols.size == 0:
        return xor_grid

    observed_width = int(active_cols[-1]) + 1
    observed = xor_grid[:, :observed_width]
    period = _minimal_period(observed)
    tile = observed[:, :period]
    return _repeat_columns(tile, g.shape[1]).astype(int)


def _sample_base_pattern(rng: np.random.Generator, height: int, period: int, tail: int) -> np.ndarray:
    visible_width = period + tail
    for _ in range(_PATTERN_MAX_ATTEMPTS):
        density = float(rng.uniform(0.25, 0.55))
        base = rng.random((height, period)) < density
        if int(base.sum()) < 2:
            continue
        if np.any(base.sum(axis=0) == 0):
            continue
        if not np.any(~base):
            continue
        if _minimal_period(base.astype(int)) != period:
            continue
        visible = _repeat_columns(base.astype(int), visible_width)
        if _minimal_period(visible) != period:
            continue
        return base
    raise RuntimeError("failed to sample a non-degenerate periodic base pattern")


def _sample_common_noise(
    rng: np.random.Generator, signal_mask: np.ndarray, visible_width: int
) -> np.ndarray:
    height, width = signal_mask.shape
    noise = np.zeros_like(signal_mask, dtype=bool)

    available = np.argwhere(~signal_mask)
    if available.size == 0:
        return noise

    chosen: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()

    right_candidates = np.argwhere(~signal_mask[:, visible_width:])
    if right_candidates.size > 0:
        rr, cc = right_candidates[int(rng.integers(0, len(right_candidates)))]
        cell = (int(rr), int(cc + visible_width))
        chosen.append(cell)
        used.add(cell)

    left_candidates = np.argwhere(~signal_mask[:, :visible_width])
    if left_candidates.size > 0:
        rr, cc = left_candidates[int(rng.integers(0, len(left_candidates)))]
        cell = (int(rr), int(cc))
        if cell not in used:
            chosen.append(cell)
            used.add(cell)

    target = int(rng.integers(max(2, len(chosen)), max(3, min(len(available), (height * width) // 5)) + 1))
    remaining = [tuple(map(int, coord)) for coord in available if tuple(map(int, coord)) not in used]
    if target > len(chosen) and remaining:
        pick = min(target - len(chosen), len(remaining))
        idx = rng.choice(len(remaining), size=pick, replace=False)
        for k in np.atleast_1d(idx):
            chosen.append(remaining[int(k)])

    for r, c in chosen:
        noise[r, c] = True
    return noise


def _split_signal(rng: np.random.Generator, signal_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(signal_mask)
    assert len(coords) >= 2

    assign_upper = rng.integers(0, 2, size=len(coords)).astype(bool)
    if assign_upper.all():
        assign_upper[0] = False
    if (~assign_upper).all():
        assign_upper[0] = True

    upper = np.zeros_like(signal_mask, dtype=bool)
    lower = np.zeros_like(signal_mask, dtype=bool)
    for (r, c), to_upper in zip(coords, assign_upper):
        if bool(to_upper):
            upper[int(r), int(c)] = True
        else:
            lower[int(r), int(c)] = True
    return upper, lower


def _build_example(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    width = int(rng.integers(12, 25))
    half_height = int(rng.integers(4, 11))
    max_period = min(6, (width // 2 + 1) // 2)
    if max_period < 2:
        raise RuntimeError("grid too small for the required period")

    period = int(rng.integers(2, max_period + 1))
    tail = int(rng.integers(1, period))
    visible_width = period + tail
    assert visible_width <= width // 2

    base = _sample_base_pattern(rng, half_height, period, tail)
    signal_visible = _repeat_columns(base.astype(bool), visible_width)
    signal_mask = np.zeros((half_height, width), dtype=bool)
    signal_mask[:, :visible_width] = signal_visible

    common_noise = _sample_common_noise(rng, signal_mask, visible_width)
    if not common_noise.any():
        raise RuntimeError("shared clutter is required")

    upper_signal, lower_signal = _split_signal(rng, signal_mask)
    upper_mask = np.logical_or(common_noise, upper_signal)
    lower_mask = np.logical_or(common_noise, lower_signal)

    grid = np.zeros((half_height * 2, width), dtype=int)
    grid[:half_height][upper_mask] = BLUE
    grid[half_height:][lower_mask] = RED

    expected = np.zeros((half_height, width), dtype=int)
    expected[_repeat_columns(base.astype(bool), width)] = MAGENTA
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        rng = np.random.default_rng()
        try:
            grid, expected = _build_example(rng)
        except RuntimeError:
            continue

        try:
            actual = main(grid)
        except Exception:
            continue

        if not np.array_equal(actual, expected):
            continue
        if len(np.unique(grid)) < 3:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input grid")
