from __future__ import annotations

import numpy as np


MIN_PERIOD = 3
MAX_PERIOD = 5
GEN_ATTEMPTS = 300
ORTH_LEN_RANGE = (6, 20)
REPEAT_COUNT_RANGE = (2, 6)


def _infer_orientation(grid: np.ndarray) -> str:
    row_counts = np.array([np.unique(grid[r]).size for r in range(grid.shape[0])], dtype=int)
    col_counts = np.array([np.unique(grid[:, c]).size for c in range(grid.shape[1])], dtype=int)

    if int(row_counts.max()) <= 2 < int(col_counts.max()):
        return "horizontal"
    if int(col_counts.max()) <= 2 < int(row_counts.max()):
        return "vertical"
    return "horizontal" if float(row_counts.mean()) <= float(col_counts.mean()) else "vertical"


def _line_color_sets(lines: np.ndarray) -> list[set[int]]:
    return [set(map(int, np.unique(line))) for line in lines]


def _assign_period_colors(line_sets: list[set[int]], period: int) -> list[int] | None:
    residue_candidates: list[list[int]] = []
    for residue in range(period):
        group = [line_sets[i] for i in range(residue, len(line_sets), period)]
        singletons = {next(iter(colors)) for colors in group if len(colors) == 1}
        if len(singletons) > 1:
            return None
        if singletons:
            candidates = sorted(singletons)
        else:
            common = set.intersection(*(set(colors) for colors in group))
            candidates = sorted(common if common else set.union(*(set(colors) for colors in group)))
        residue_candidates.append([int(color) for color in candidates])

    order = sorted(range(period), key=lambda residue: len(residue_candidates[residue]))
    assignment: list[int | None] = [None] * period
    used: set[int] = set()

    def backtrack(k: int) -> bool:
        if k == len(order):
            return True
        residue = order[k]
        group = [line_sets[i] for i in range(residue, len(line_sets), period)]
        for color in residue_candidates[residue]:
            if color in used:
                continue
            if any(color not in colors for colors in group):
                continue
            assignment[residue] = int(color)
            used.add(int(color))
            if backtrack(k + 1):
                return True
            used.remove(int(color))
            assignment[residue] = None
        return False

    if not backtrack(0):
        return None
    return [int(color) for color in assignment if color is not None]


def _infer_cycle(lines: np.ndarray) -> tuple[int, list[int]]:
    line_sets = _line_color_sets(lines)
    impure = [i for i, colors in enumerate(line_sets) if len(colors) > 1]
    best: tuple[int, int, list[int]] | None = None

    for period in range(MIN_PERIOD, min(MAX_PERIOD, lines.shape[0]) + 1):
        colors = _assign_period_colors(line_sets, period)
        if colors is None or len(colors) != period or len(set(colors)) != period:
            continue
        if any(colors[i % period] not in line_sets[i] for i in range(len(line_sets))):
            continue
        if impure:
            if len(impure) != 2 or impure[1] != impure[0] + 1:
                continue
            first, second = impure
            pair = {int(colors[first % period]), int(colors[second % period])}
            if line_sets[first] != pair or line_sets[second] != pair:
                continue
        score = sum(
            len(line_sets[i]) == 1 and next(iter(line_sets[i])) == int(colors[i % period])
            for i in range(len(line_sets))
        )
        candidate = (int(score), int(period), [int(color) for color in colors])
        if best is None or candidate[:2] > best[:2]:
            best = candidate

    if best is None:
        raise AssertionError("failed to infer stripe period and colors")
    return int(best[1]), best[2]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore a stripe grid by undoing a local color swap between two adjacent stripes.

    Input structure:
    - The grid is made of 1-cell-wide horizontal or vertical monochrome stripes.
    - Stripe colors follow a repeating cycle of 3 to 5 distinct colors.
    - A connected run of cells has been swapped between two adjacent stripes, so those two stripes are the only non-uniform ones.

    Rule:
    - Detect whether the stripes run horizontally or vertically.
    - Infer the repeating stripe period and each stripe's intended color from the mostly clean stripes.
    - Repaint every stripe to its inferred solid color, which restores both sides of the swapped run.

    Key insight:
    - The visible defect is not a new object; it is a local exchange between two neighboring stripes, so the clean periodic stripe pattern tells you how to undo it.
    """
    g = np.asarray(grid, dtype=int)
    orientation = _infer_orientation(g)
    lines = g if orientation == "horizontal" else g.T
    period, colors = _infer_cycle(lines)

    restored = np.empty_like(lines)
    for i in range(lines.shape[0]):
        restored[i, :] = int(colors[i % period])
    return restored if orientation == "horizontal" else restored.T


def _clean_grid(
    rng: np.random.Generator,
    orientation: str,
    period: int,
    axis_len: int,
    orth_len: int,
) -> tuple[np.ndarray, list[int]]:
    palette = [int(x) for x in rng.choice(np.arange(10), size=period, replace=False)]
    if orientation == "horizontal":
        grid = np.empty((axis_len, orth_len), dtype=int)
        for r in range(axis_len):
            grid[r, :] = int(palette[r % period])
    else:
        grid = np.empty((orth_len, axis_len), dtype=int)
        for c in range(axis_len):
            grid[:, c] = int(palette[c % period])
    return grid, palette


def _corrupt_grid(
    rng: np.random.Generator,
    clean: np.ndarray,
    orientation: str,
    axis_len: int,
    orth_len: int,
) -> np.ndarray:
    dirty = clean.copy()
    stripe = int(rng.integers(0, axis_len - 1))
    seg_len = int(rng.integers(1, orth_len))
    start = int(rng.integers(0, orth_len - seg_len + 1))

    if orientation == "horizontal":
        seg_a = dirty[stripe, start : start + seg_len].copy()
        seg_b = dirty[stripe + 1, start : start + seg_len].copy()
        dirty[stripe, start : start + seg_len] = seg_b
        dirty[stripe + 1, start : start + seg_len] = seg_a
    else:
        seg_a = dirty[start : start + seg_len, stripe].copy()
        seg_b = dirty[start : start + seg_len, stripe + 1].copy()
        dirty[start : start + seg_len, stripe] = seg_b
        dirty[start : start + seg_len, stripe + 1] = seg_a
    return dirty


def _sample_axis_len(rng: np.random.Generator, period: int) -> int:
    max_repeats = min(REPEAT_COUNT_RANGE[1], 30 // period)
    min_repeats = min(REPEAT_COUNT_RANGE[0], max_repeats)
    repeats = int(rng.integers(min_repeats, max_repeats + 1))
    base = repeats * period
    extra_cap = min(period - 1, 30 - base)
    extra = int(rng.integers(0, extra_cap + 1))
    return int(base + extra)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    orientation = "horizontal" if bool(rng.integers(0, 2)) else "vertical"
    period = int(rng.integers(MIN_PERIOD, MAX_PERIOD + 1))
    axis_len = _sample_axis_len(rng, period)
    orth_len = int(rng.integers(ORTH_LEN_RANGE[0], ORTH_LEN_RANGE[1] + 1))

    clean, _palette = _clean_grid(rng, orientation, period, axis_len, orth_len)
    dirty = _corrupt_grid(rng, clean, orientation, axis_len, orth_len)

    if not np.array_equal(main(dirty), clean):
        return None
    return dirty


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stripe-swap input")
