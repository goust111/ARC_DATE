from __future__ import annotations

import random

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
PALETTE = (BLUE, RED, GREEN)
PALETTE_SET = set(PALETTE)
THIRD_COLOR = {
    frozenset((BLUE, RED)): GREEN,
    frozenset((BLUE, GREEN)): RED,
    frozenset((RED, GREEN)): BLUE,
}

# --------------- solver ---------------


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    winners = values[counts == counts.max()]
    assert winners.size == 1, "background must be unique"
    return int(winners[0])


def _combine_colors(a: int, b: int) -> int:
    a = int(a)
    b = int(b)
    assert a in PALETTE_SET and b in PALETTE_SET, "ticks must use the three rule colors"
    if a == b:
        return a
    return THIRD_COLOR[frozenset((a, b))]


def _read_ticks(grid: np.ndarray, bg: int) -> tuple[dict[int, int], dict[int, int]]:
    assert int(grid[0, 0]) == bg, "top-left corner must stay background"

    row_ticks = {r: int(grid[r, 0]) for r in range(1, grid.shape[0]) if int(grid[r, 0]) != bg}
    col_ticks = {c: int(grid[0, c]) for c in range(1, grid.shape[1]) if int(grid[0, c]) != bg}
    assert row_ticks and col_ticks, "expected ticks on both axes"

    for color in row_ticks.values():
        assert color in PALETTE_SET
    for color in col_ticks.values():
        assert color in PALETTE_SET

    return row_ticks, col_ticks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Place a colored pixel at every intersection of the left-side row ticks and
    top-side column ticks.

    If the two tick colors match, the intersection keeps that color. If they
    differ, the intersection takes the third color from {blue, red, green}.
    Existing interior example points must already agree with this rule.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] >= 3 and g.shape[1] >= 3

    bg = _background_color(g)
    row_ticks, col_ticks = _read_ticks(g, bg)
    valid_rows = set(row_ticks)
    valid_cols = set(col_ticks)

    out = g.copy()
    for r in range(1, g.shape[0]):
        for c in range(1, g.shape[1]):
            color = int(g[r, c])
            if color == bg:
                continue
            assert r in valid_rows and c in valid_cols, "example pixels must sit on tick intersections"
            assert color == _combine_colors(row_ticks[r], col_ticks[c]), "example pixel conflicts with rule"

    for r, row_color in row_ticks.items():
        for c, col_color in col_ticks.items():
            out[r, c] = _combine_colors(row_color, col_color)

    return out


# --------------- generator ---------------

BG = 0
MIN_SIZE = 8
MAX_SIZE = 18
MIN_GAP = 2
MAX_ROW_TICKS = 4
MAX_COL_TICKS = 4
MIN_INTERSECTIONS = 6
MAX_GENERATION_ATTEMPTS = 400


def _max_ticks(length: int, min_gap: int) -> int:
    usable = length - 1
    if usable <= 0:
        return 0
    return 1 + (usable - 1) // min_gap


def _sample_spaced_positions(length: int, count: int, min_gap: int) -> list[int]:
    usable = length - 1
    adjusted = usable - (count - 1) * (min_gap - 1)
    assert adjusted >= count
    base = sorted(random.sample(range(1, adjusted + 1), count))
    return [pos + i * (min_gap - 1) for i, pos in enumerate(base)]


def _sample_tick_counts(height: int, width: int) -> tuple[int, int]:
    row_max = min(MAX_ROW_TICKS, _max_ticks(height, MIN_GAP))
    col_max = min(MAX_COL_TICKS, _max_ticks(width, MIN_GAP))
    choices = [
        (n_rows, n_cols)
        for n_rows in range(2, row_max + 1)
        for n_cols in range(2, col_max + 1)
        if n_rows * n_cols >= MIN_INTERSECTIONS
    ]
    assert choices, "canvas too small for a clear instance"
    return random.choice(choices)


def _sample_axis_colors(n_rows: int, n_cols: int) -> tuple[list[int], list[int]]:
    for _ in range(300):
        row_colors = [random.choice(PALETTE) for _ in range(n_rows)]
        col_colors = [random.choice(PALETTE) for _ in range(n_cols)]

        if len(set(row_colors)) < 2 or len(set(col_colors)) < 2:
            continue
        if set(row_colors) | set(col_colors) != PALETTE_SET:
            continue

        same_count = sum(row_colors.count(color) * col_colors.count(color) for color in PALETTE)
        if same_count < 2:
            continue

        outputs = {_combine_colors(r, c) for r in row_colors for c in col_colors}
        if outputs != PALETTE_SET:
            continue

        return row_colors, col_colors
    raise RuntimeError("failed to sample non-degenerate axis colors")


def _intersection_entries(
    row_positions: list[int],
    col_positions: list[int],
    row_colors: list[int],
    col_colors: list[int],
) -> list[dict[str, int | bool]]:
    entries: list[dict[str, int | bool]] = []
    for r, row_color in zip(row_positions, row_colors):
        for c, col_color in zip(col_positions, col_colors):
            out_color = _combine_colors(row_color, col_color)
            entries.append(
                {
                    "r": int(r),
                    "c": int(c),
                    "out_color": int(out_color),
                    "same_branch": bool(row_color == col_color),
                }
            )
    return entries


def _choose_examples(entries: list[dict[str, int | bool]]) -> list[dict[str, int | bool]]:
    total = len(entries)
    indices = list(range(total))
    all_outputs = {int(entry["out_color"]) for entry in entries}
    min_count = max(3, total // 3)
    max_count = total - 2
    assert min_count <= max_count

    for _ in range(400):
        count = random.randint(min_count, max_count)
        chosen = set(random.sample(indices, count))
        examples = [entries[i] for i in indices if i in chosen]
        missing = [entries[i] for i in indices if i not in chosen]

        if not any(bool(entry["same_branch"]) for entry in examples):
            continue
        if not any(not bool(entry["same_branch"]) for entry in examples):
            continue
        if not any(bool(entry["same_branch"]) for entry in missing):
            continue
        if not any(not bool(entry["same_branch"]) for entry in missing):
            continue
        if {int(entry["out_color"]) for entry in examples} != all_outputs:
            continue

        return examples

    raise RuntimeError("failed to choose informative example intersections")


def _render_input(
    height: int,
    width: int,
    row_positions: list[int],
    col_positions: list[int],
    row_colors: list[int],
    col_colors: list[int],
    examples: list[dict[str, int | bool]],
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)

    for r, color in zip(row_positions, row_colors):
        grid[r, 0] = color
    for c, color in zip(col_positions, col_colors):
        grid[0, c] = color
    for entry in examples:
        grid[int(entry["r"]), int(entry["c"])] = int(entry["out_color"])

    return grid


def _render_expected(
    inp: np.ndarray,
    row_positions: list[int],
    col_positions: list[int],
    row_colors: list[int],
    col_colors: list[int],
) -> np.ndarray:
    out = inp.copy()
    for r, row_color in zip(row_positions, row_colors):
        for c, col_color in zip(col_positions, col_colors):
            out[r, c] = _combine_colors(row_color, col_color)
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(MIN_SIZE, MAX_SIZE)
    width = random.randint(MIN_SIZE, MAX_SIZE)

    try:
        n_rows, n_cols = _sample_tick_counts(height, width)
    except AssertionError:
        return None

    row_positions = _sample_spaced_positions(height, n_rows, MIN_GAP)
    col_positions = _sample_spaced_positions(width, n_cols, MIN_GAP)

    try:
        row_colors, col_colors = _sample_axis_colors(n_rows, n_cols)
    except RuntimeError:
        return None

    entries = _intersection_entries(row_positions, col_positions, row_colors, col_colors)
    try:
        examples = _choose_examples(entries)
    except RuntimeError:
        return None

    inp = _render_input(height, width, row_positions, col_positions, row_colors, col_colors, examples)
    expected = _render_expected(inp, row_positions, col_positions, row_colors, col_colors)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
