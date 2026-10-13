from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_GEN_ATTEMPTS = 400


def _line_candidates(grid: np.ndarray) -> list[dict]:
    candidates: list[dict] = []
    h, w = grid.shape

    for r in range(h):
        cols = np.flatnonzero(grid[r] != BG)
        if len(cols) < 5:
            continue
        diffs = np.diff(cols)
        if len(set(map(int, diffs))) != 1:
            continue
        step = int(diffs[0])
        if step <= 0:
            continue
        coords = [(r, int(c)) for c in cols]
        colors = [int(grid[rr, cc]) for rr, cc in coords]
        candidates.append(
            {
                "orientation": "horizontal",
                "step": step,
                "coords": coords,
                "colors": colors,
            }
        )

    for c in range(w):
        rows = np.flatnonzero(grid[:, c] != BG)
        if len(rows) < 5:
            continue
        diffs = np.diff(rows)
        if len(set(map(int, diffs))) != 1:
            continue
        step = int(diffs[0])
        if step <= 0:
            continue
        coords = [(int(r), c) for r in rows]
        colors = [int(grid[rr, cc]) for rr, cc in coords]
        candidates.append(
            {
                "orientation": "vertical",
                "step": step,
                "coords": coords,
                "colors": colors,
            }
        )

    return candidates


def _infer_defect(colors: list[int]) -> tuple[int, int, int]:
    n = len(colors)
    solutions: list[tuple[int, int, int, int]] = []

    for period in range(2, n):
        mismatch_index = None
        mismatch_count = 0
        residue_colors: list[int] = []
        valid = True

        for residue in range(period):
            indices = list(range(residue, n, period))
            group = [colors[i] for i in indices]
            counts = Counter(group)
            if len(counts) > 2:
                valid = False
                break

            major_color, major_count = counts.most_common(1)[0]
            if sum(1 for v in counts.values() if v == major_count) > 1 and len(counts) > 1:
                valid = False
                break

            residue_colors.append(int(major_color))
            wrong_here = [i for i in indices if colors[i] != major_color]
            mismatch_count += len(wrong_here)
            if wrong_here:
                if len(wrong_here) != 1 or mismatch_index is not None:
                    valid = False
                    break
                mismatch_index = wrong_here[0]

        if not valid or mismatch_count != 1 or mismatch_index is None:
            continue

        expected_color = residue_colors[mismatch_index % period]
        wrong_color = colors[mismatch_index]
        if expected_color == wrong_color:
            continue
        solutions.append((period, mismatch_index, expected_color, wrong_color))

    assert solutions, "failed to infer a unique periodic defect"
    best_period = min(period for period, *_ in solutions)
    best = [sol for sol in solutions if sol[0] == best_period]
    assert len(best) == 1, "period inference should be unique"
    _, defect_index, expected_color, wrong_color = best[0]
    return defect_index, expected_color, wrong_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the one wrong color on a periodic dot track by consuming the off-track pixel whose color points to the mistake.

    Input structure:
    - The background is black.
    - One horizontal or vertical line contains evenly spaced single-pixel track dots.
    - The track colors follow a short repeating cycle except for exactly one wrong dot.
    - Additional off-track pixels are isolated single-pixel scatter dots with distinct colors; one matches the wrong track color and another matches the color that should be at the defect.

    Rule:
    - Detect the unique evenly spaced straight track and read its colors in order.
    - Infer the shortest repeating color cycle that explains every track dot except one, locating the defect and its expected color.
    - Find the off-track scatter pixel whose color equals the wrong track color, erase that scatter, and repaint the defect cell with the expected periodic color.
    - Leave all other scatter pixels, including the decoy scatter in the expected color, unchanged.

    Key insight:
    - The wrong track color is a pointer to which scatter disappears, while the replacement color must be inferred from the periodic structure of the rest of the track.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    candidates = _line_candidates(g)
    assert candidates, "expected at least one line candidate"
    max_len = max(len(c["coords"]) for c in candidates)
    candidates = [c for c in candidates if len(c["coords"]) == max_len]
    assert len(candidates) == 1, "expected a unique track"
    track = candidates[0]

    defect_index, expected_color, wrong_color = _infer_defect(track["colors"])
    defect_r, defect_c = track["coords"][defect_index]

    track_mask = np.zeros_like(g, dtype=bool)
    for r, c in track["coords"]:
        track_mask[r, c] = True

    pointer_coords = np.argwhere((g == wrong_color) & ~track_mask)
    assert pointer_coords.shape == (1, 2), "expected a unique wrong-color scatter"

    decoy_coords = np.argwhere((g == expected_color) & ~track_mask)
    assert decoy_coords.shape[0] >= 1, "expected at least one expected-color scatter decoy"

    out = g.copy()
    out[defect_r, defect_c] = expected_color
    pr, pc = map(int, pointer_coords[0])
    out[pr, pc] = BG
    return out


def _choose_track_params() -> tuple[int, int]:
    period = random.randint(2, 4)
    repeats = 3 if period == 4 else random.randint(3, 4)
    return period, repeats


def _place_scatters(
    grid: np.ndarray,
    track_row: int | None,
    track_col: int | None,
    scatter_colors: list[int],
) -> bool:
    h, w = grid.shape
    count = len(scatter_colors)

    if track_row is not None:
        allowed_rows = [r for r in range(h) if abs(r - track_row) > 1]
        allowed_cols = list(range(w))
    else:
        allowed_rows = list(range(h))
        allowed_cols = [c for c in range(w) if abs(c - track_col) > 1]

    if len(allowed_rows) < count or len(allowed_cols) < count:
        return False

    for _ in range(200):
        rows = random.sample(allowed_rows, count)
        cols = random.sample(allowed_cols, count)
        random.shuffle(cols)
        coords = list(zip(rows, cols))
        if any(grid[r, c] != BG for r, c in coords):
            continue
        if any(max(abs(r1 - r2), abs(c1 - c2)) <= 1 for i, (r1, c1) in enumerate(coords) for r2, c2 in coords[i + 1 :]):
            continue
        for (r, c), color in zip(coords, scatter_colors):
            grid[r, c] = color
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 28)
    w = random.randint(12, 28)
    grid = np.zeros((h, w), dtype=int)

    orientation = random.choice(["horizontal", "vertical"])
    period, repeats = _choose_track_params()
    track_len = period * repeats
    axis_len = w if orientation == "horizontal" else h
    max_step = min(3, (axis_len - 1) // (track_len - 1))
    if max_step < 1:
        return None
    step = random.randint(1, max_step)
    span = step * (track_len - 1) + 1

    period_colors = random.sample(PALETTE, period)
    expected_sequence = [period_colors[i % period] for i in range(track_len)]

    defect_index = random.randrange(track_len)
    expected_color = expected_sequence[defect_index]
    wrong_color = random.choice([c for c in period_colors if c != expected_color])
    track_colors = expected_sequence[:]
    track_colors[defect_index] = wrong_color

    if orientation == "horizontal":
        track_row = random.randint(1, h - 2)
        start_col = random.randint(0, w - span)
        coords = [(track_row, start_col + i * step) for i in range(track_len)]
        for (r, c), color in zip(coords, track_colors):
            grid[r, c] = color
        line_row, line_col = track_row, None
    else:
        track_col = random.randint(1, w - 2)
        start_row = random.randint(0, h - span)
        coords = [(start_row + i * step, track_col) for i in range(track_len)]
        for (r, c), color in zip(coords, track_colors):
            grid[r, c] = color
        line_row, line_col = None, track_col

    extra_pool = [c for c in PALETTE if c not in period_colors]
    extra_count = random.randint(1, min(4, len(extra_pool)))
    scatter_colors = [wrong_color, expected_color] + random.sample(extra_pool, extra_count)
    if not _place_scatters(grid, line_row, line_col, scatter_colors):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-track repair input")
