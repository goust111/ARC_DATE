from __future__ import annotations

import random
from typing import Sequence

import numpy as np


SEED_COLORS = (6, 7)
SEPARATOR_COLOR = 5
BASE_PALETTE = (1, 2, 3, 4, 8, 9)

BASE_ROWS_RANGE = (6, 8)
BASE_COLS_RANGE = (6, 8)
RUN_CHOICES = (2, 3)
GEN_ATTEMPTS = 500


def _is_seed_row(row: np.ndarray) -> bool:
    return bool(np.all(row == row[0]) and int(row[0]) in SEED_COLORS)


def _run_starts(lines: np.ndarray) -> tuple[list[int], list[int]]:
    starts = [0]
    runs: list[int] = []
    for i in range(1, lines.shape[0]):
        if not np.array_equal(lines[i], lines[i - 1]):
            runs.append(i - starts[-1])
            starts.append(i)
    runs.append(lines.shape[0] - starts[-1])
    return starts, runs


def _compress_columns(grid: np.ndarray) -> tuple[np.ndarray, list[int]]:
    starts, runs = _run_starts(grid.T)
    return grid[:, starts].copy(), runs


def _compress_rows_with_seed_groups(grid: np.ndarray) -> tuple[np.ndarray, list[int]]:
    rows: list[np.ndarray] = []
    runs: list[int] = []
    h = grid.shape[0]
    i = 0
    while i < h:
        row = grid[i]
        assert not _is_seed_row(row), "seed rows must be internal to a repeated row group"

        if i + 2 < h and _is_seed_row(grid[i + 1]) and np.array_equal(grid[i + 2], row):
            rows.append(grid[i + 1].copy())
            runs.append(3)
            i += 3
            continue

        j = i + 1
        while j < h and np.array_equal(grid[j], row):
            j += 1
        rows.append(row.copy())
        runs.append(j - i)
        i = j

    out = np.stack(rows, axis=0)
    return out, runs


def _nearest_fill_rows(comp: np.ndarray) -> np.ndarray:
    h, w = comp.shape
    seed_rows = [(r, int(comp[r, 0])) for r in range(h) if _is_seed_row(comp[r])]
    assert 2 <= len(seed_rows) <= 3, "expected 2-3 seed rows after compression"

    out = np.empty_like(comp)
    for r in range(h):
        if _is_seed_row(comp[r]):
            out[r, :] = comp[r, 0]
            continue

        distances = [abs(r - sr) for sr, _color in seed_rows]
        best = min(distances)
        nearest_colors = [color for (sr, color), dist in zip(seed_rows, distances) if dist == best]
        if len(set(nearest_colors)) == 1:
            out[r, :] = nearest_colors[0]
        else:
            out[r, :] = SEPARATOR_COLOR
    assert out.shape == (h, w)
    return out


def _expand(grid: np.ndarray, row_runs: Sequence[int], col_runs: Sequence[int]) -> np.ndarray:
    return np.repeat(np.repeat(grid, row_runs, axis=0), col_runs, axis=1)


def _solve_core(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    cols_compressed, col_runs = _compress_columns(grid)
    comp, row_runs = _compress_rows_with_seed_groups(cols_compressed)
    filled = _nearest_fill_rows(comp)
    return comp, filled, row_runs, col_runs


def main(grid: np.ndarray) -> np.ndarray:
    """Compress duplicated row/column runs, fill rows from the nearest seed strip, then re-expand by the original run lengths.

    Input structure:
    - The grid has no black background and is formed by repeating each row and each column of a smaller color pattern 2-3 times.
    - Some repeated row groups have one internal row replaced by a full-width seed strip in magenta(6) or orange(7).
    - Adjacent equal rows/columns encode the hidden compression factors; seed strips interrupt only row groups.

    Rule:
    - Collapse identical adjacent column runs to one column each, and collapse row groups by treating a `pattern, seed, pattern` triple as one repeated row group of length 3.
    - In the compressed grid, keep each seed strip as its row and recolor every other row by the nearest seed-strip row using compressed-row distance.
    - If a row is equally close to nearest seed strips of different colors, recolor the whole row grey(5); equal-distance same-color ties keep that shared seed color.
    - Re-expand the filled compressed grid by the recorded row and column run lengths.

    Key insight:
    - Visual pixel distance is misleading because row groups were stretched before one copy was overwritten; the correct nearest-seed partition only appears after undoing the hidden row/column duplication.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] > 0 and g.shape[1] > 0
    assert max(g.shape) <= 30

    _comp, filled, row_runs, col_runs = _solve_core(g)
    out = _expand(filled, row_runs, col_runs)
    assert out.shape == g.shape
    return out


def _generate_base_pattern(rows: int, cols: int, n_colors: int) -> np.ndarray | None:
    palette = random.sample(BASE_PALETTE, n_colors)
    for _ in range(200):
        base = np.array([[random.choice(palette) for _ in range(cols)] for _ in range(rows)], dtype=int)
        if len(set(map(int, base.flat))) != n_colors:
            continue
        if any(len(set(map(int, base[r]))) < 2 for r in range(rows)):
            continue
        if any(len(set(map(int, base[:, c]))) < 2 for c in range(cols)):
            continue
        if any(np.array_equal(base[r], base[r + 1]) for r in range(rows - 1)):
            continue
        if any(np.array_equal(base[:, c], base[:, c + 1]) for c in range(cols - 1)):
            continue
        return base
    return None


def _choose_seed_layout(rows: int) -> tuple[list[int], list[int]] | None:
    scenario = random.choice(("two_diff", "three_alt", "three_pair"))

    if scenario == "two_diff":
        gap = random.choice((3, 4, 5))
        start_choices = [r for r in range(rows - gap)]
        if not start_choices:
            return None
        r0 = random.choice(start_choices)
        positions = [r0, r0 + gap]
        colors = [6, 7] if random.random() < 0.5 else [7, 6]
        return positions, colors

    if rows < 7:
        scenario = "two_diff"

    if scenario == "three_alt":
        for _ in range(100):
            positions = sorted(random.sample(range(rows), 3))
            gaps = [positions[1] - positions[0], positions[2] - positions[1]]
            if min(gaps) < 2:
                continue
            if any(g < 3 for g, a, b in zip(gaps, (6, 7), (7, 6)) if a != b):
                continue
            colors = random.choice(([6, 7, 6], [7, 6, 7]))
            return positions, list(colors)
        return None

    for _ in range(100):
        positions = sorted(random.sample(range(rows), 3))
        gaps = [positions[1] - positions[0], positions[2] - positions[1]]
        color_triplets = ([6, 6, 7], [7, 7, 6], [6, 7, 7], [7, 6, 6])
        colors = list(random.choice(color_triplets))
        ok = True
        for gap, c0, c1 in zip(gaps, colors, colors[1:]):
            if c0 != c1 and gap < 3:
                ok = False
        if ok and min(gaps) >= 2:
            return positions, colors
    return None


def _row_runs_for_seed_positions(rows: int, seed_rows: Sequence[int]) -> list[int]:
    seed_set = set(seed_rows)
    runs = []
    for r in range(rows):
        if r in seed_set:
            runs.append(3)
        else:
            runs.append(random.choice(RUN_CHOICES))
    if len(set(runs)) == 1:
        idx = random.choice([r for r in range(rows) if r not in seed_set])
        runs[idx] = 2 if runs[idx] == 3 else 3
    return runs


def _col_runs(cols: int) -> list[int]:
    runs = [random.choice(RUN_CHOICES) for _ in range(cols)]
    if len(set(runs)) == 1:
        runs[random.randrange(cols)] = 2 if runs[0] == 3 else 3
    return runs


def _render_input(
    base: np.ndarray,
    row_runs: Sequence[int],
    col_runs: Sequence[int],
    seed_rows: Sequence[int],
    seed_colors: Sequence[int],
) -> np.ndarray:
    inp = _expand(base, row_runs, col_runs)
    row_offsets = np.cumsum([0, *row_runs[:-1]])
    for base_r, color in zip(seed_rows, seed_colors):
        expanded_r = int(row_offsets[base_r] + 1)
        inp[expanded_r, :] = int(color)
    return inp


def _naive_pixel_distance_fill(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    seed_rows = [(r, int(g[r, 0])) for r in range(g.shape[0]) if _is_seed_row(g[r])]
    out = np.empty_like(g)
    for r in range(g.shape[0]):
        if _is_seed_row(g[r]):
            out[r, :] = g[r, 0]
            continue
        distances = [abs(r - sr) for sr, _color in seed_rows]
        best = min(distances)
        nearest_colors = [color for (sr, color), dist in zip(seed_rows, distances) if dist == best]
        out[r, :] = nearest_colors[0] if len(set(nearest_colors)) == 1 else SEPARATOR_COLOR
    return out


def _simple_row_compress_count(grid: np.ndarray) -> int:
    starts, _runs = _run_starts(grid)
    return len(starts)


def _semantic_checks(
    inp: np.ndarray,
    base: np.ndarray,
    row_runs: Sequence[int],
    col_runs: Sequence[int],
    seed_rows: Sequence[int],
    seed_colors: Sequence[int],
) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False

    cols_compressed, detected_col_runs = _compress_columns(inp)
    if list(detected_col_runs) != list(col_runs):
        return False

    comp, filled, detected_row_runs, _ = _solve_core(inp)
    if comp.shape != base.shape:
        return False
    if list(detected_row_runs) != list(row_runs):
        return False

    if [r for r in range(comp.shape[0]) if _is_seed_row(comp[r])] != list(seed_rows):
        return False
    if [int(comp[r, 0]) for r in seed_rows] != list(seed_colors):
        return False

    expected = _expand(filled, row_runs, col_runs)
    if not np.array_equal(main(inp), expected):
        return False

    if np.array_equal(_naive_pixel_distance_fill(inp), expected):
        return False

    if _simple_row_compress_count(cols_compressed) == len(row_runs):
        return False

    if len(set(map(int, expected.flat))) < 2:
        return False

    if np.array_equal(inp, expected):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*BASE_ROWS_RANGE)
    cols = random.randint(*BASE_COLS_RANGE)

    seed_layout = _choose_seed_layout(rows)
    if seed_layout is None:
        return None
    seed_rows, seed_colors = seed_layout

    base = _generate_base_pattern(rows, cols, random.choice((2, 3)))
    if base is None:
        return None

    row_runs = _row_runs_for_seed_positions(rows, seed_rows)
    col_runs = _col_runs(cols)
    inp = _render_input(base, row_runs, col_runs, seed_rows, seed_colors)

    if not _semantic_checks(inp, base, row_runs, col_runs, seed_rows, seed_colors):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
