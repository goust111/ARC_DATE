from __future__ import annotations

from collections import Counter

import numpy as np


BACKGROUND = 0
SEED_SHAPES = (
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 1)),
    ((0, 1), (1, 0)),
)
SEED_COUNTS = (3, 4)
CELL_SIZES = (5, 6)
MAX_GENERATION_ATTEMPTS = 400


def _runs_between_stops(length: int, stops: list[int]) -> list[tuple[int, int]]:
    stop_set = set(map(int, stops))
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx in range(int(length)):
        if idx in stop_set:
            if start is not None:
                runs.append((start, idx - 1))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        runs.append((start, int(length) - 1))
    return runs


def _cell_index(runs: list[tuple[int, int]], value: int) -> int:
    for idx, (lo, hi) in enumerate(runs):
        if lo <= int(value) <= hi:
            return int(idx)
    raise AssertionError("coordinate does not belong to any open cell")


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _most_common(values: np.ndarray) -> int:
    return int(Counter(map(int, values.ravel())).most_common(1)[0][0])


def _parse_layout(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    candidates: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]], np.ndarray]] = []
    for color in map(int, np.unique(g)):
        sep_rows = np.flatnonzero(np.all(g == color, axis=1)).tolist()
        sep_cols = np.flatnonzero(np.all(g == color, axis=0)).tolist()
        if not sep_rows or not sep_cols:
            continue
        row_runs = _runs_between_stops(h, sep_rows)
        col_runs = _runs_between_stops(w, sep_cols)
        if len(row_runs) < 2 or len(col_runs) < 2:
            continue
        row_sizes = {hi - lo + 1 for lo, hi in row_runs}
        col_sizes = {hi - lo + 1 for lo, hi in col_runs}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue
        sep_mask = np.zeros_like(g, dtype=bool)
        sep_mask[sep_rows, :] = True
        sep_mask[:, sep_cols] = True
        if not np.array_equal(g == color, sep_mask):
            continue
        candidates.append((color, row_runs, col_runs, sep_mask))
    assert len(candidates) == 1, "expected one separator color"
    return candidates[0]


def _parse_seeds(
    grid: np.ndarray,
    separator_color: int,
    row_runs: list[tuple[int, int]],
    col_runs: list[tuple[int, int]],
    sep_mask: np.ndarray,
) -> tuple[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    open_values = g[~sep_mask]
    bg = _most_common(open_values)
    assert bg != separator_color, "background and separator must differ"

    seeds: list[dict[str, object]] = []
    for color in map(int, np.unique(open_values)):
        if color == bg:
            continue
        coords = np.argwhere(g == color)
        assert len(coords) > 0, "seed color must appear"
        row_hits = {_cell_index(row_runs, int(r)) for r, _ in coords}
        col_hits = {_cell_index(col_runs, int(c)) for _, c in coords}
        assert len(row_hits) == 1 and len(col_hits) == 1, "each seed must stay within one cell"
        row_idx = row_hits.pop()
        col_idx = col_hits.pop()
        cell_r0, cell_c0 = row_runs[row_idx][0], col_runs[col_idx][0]
        local = np.array([(int(r - cell_r0), int(c - cell_c0)) for r, c in coords], dtype=int)
        r0, c0, r1, c1 = _bbox(local)
        shape = tuple(sorted((int(r - r0), int(c - c0)) for r, c in local))
        seeds.append(
            {
                "color": int(color),
                "row": int(row_idx),
                "col": int(col_idx),
                "top": int(r0),
                "left": int(c0),
                "height": int(r1 - r0 + 1),
                "width": int(c1 - c0 + 1),
                "shape": shape,
            }
        )

    assert len(seeds) >= 2, "expected at least two seeds"
    return bg, sorted(seeds, key=lambda seed: (int(seed["row"]), int(seed["col"])))


def _stamp(grid: np.ndarray, cell_r0: int, cell_c0: int, seed_shape: tuple[tuple[int, int], ...], top: int, left: int, color: int) -> None:
    for dr, dc in seed_shape:
        grid[int(cell_r0) + int(top) + int(dr), int(cell_c0) + int(left) + int(dc)] = int(color)


def _compose_output(
    base: np.ndarray,
    seeds: list[dict[str, object]],
    row_runs: list[tuple[int, int]],
    col_runs: list[tuple[int, int]],
) -> np.ndarray:
    out = np.asarray(base, dtype=int).copy()
    by_row = {int(seed["row"]): seed for seed in seeds}
    by_col = {int(seed["col"]): seed for seed in seeds}
    for row_idx, row_seed in by_row.items():
        cell_r0 = row_runs[int(row_idx)][0]
        for col_idx, col_seed in by_col.items():
            cell_c0 = col_runs[int(col_idx)][0]
            _stamp(
                out,
                cell_r0,
                cell_c0,
                row_seed["shape"],
                int(row_seed["top"]),
                int(col_seed["left"]),
                int(col_seed["color"]),
            )
    return out


def _validate_semantics(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        separator_color, row_runs, col_runs, sep_mask = _parse_layout(inp)
        bg, seeds = _parse_seeds(inp, separator_color, row_runs, col_runs, sep_mask)
    except AssertionError:
        return False

    n = len(seeds)
    if n not in SEED_COUNTS:
        return False
    if len(set(int(seed["row"]) for seed in seeds)) != n:
        return False
    if len(set(int(seed["col"]) for seed in seeds)) != n:
        return False
    if len(set(int(seed["color"]) for seed in seeds)) != n:
        return False
    if len(set(seed["shape"] for seed in seeds)) != n:
        return False
    if len(set(int(seed["top"]) for seed in seeds)) != n:
        return False
    if len(set(int(seed["left"]) for seed in seeds)) != n:
        return False

    expected = _compose_output(inp, seeds, row_runs, col_runs)
    if not np.array_equal(expected, out):
        return False

    by_row = {int(seed["row"]): seed for seed in seeds}
    by_col = {int(seed["col"]): seed for seed in seeds}
    horizontal_evidence = False
    vertical_evidence = False
    shape_evidence = False
    cell_h = int(row_runs[0][1] - row_runs[0][0] + 1)
    cell_w = int(col_runs[0][1] - col_runs[0][0] + 1)
    for row_idx, row_seed in by_row.items():
        cell_r0 = row_runs[row_idx][0]
        for col_idx, col_seed in by_col.items():
            if row_idx == int(col_seed["row"]):
                continue
            cell_c0 = col_runs[col_idx][0]
            actual = np.zeros((cell_h, cell_w), dtype=int)
            _stamp(actual, 0, 0, row_seed["shape"], int(row_seed["top"]), int(col_seed["left"]), 1)

            wrong_full_row = np.zeros_like(actual)
            _stamp(wrong_full_row, 0, 0, row_seed["shape"], int(row_seed["top"]), int(row_seed["left"]), 1)
            if not np.array_equal(actual, wrong_full_row):
                horizontal_evidence = True

            wrong_full_col = np.zeros_like(actual)
            _stamp(wrong_full_col, 0, 0, row_seed["shape"], int(col_seed["top"]), int(col_seed["left"]), 1)
            if not np.array_equal(actual, wrong_full_col):
                vertical_evidence = True

            wrong_shape = np.zeros_like(actual)
            _stamp(wrong_shape, 0, 0, col_seed["shape"], int(row_seed["top"]), int(col_seed["left"]), 1)
            if not np.array_equal(actual, wrong_shape):
                shape_evidence = True

            cell = out[cell_r0 : row_runs[row_idx][1] + 1, cell_c0 : col_runs[col_idx][1] + 1]
            if np.count_nonzero(cell != bg) == 0:
                return False

    return horizontal_evidence and vertical_evidence and shape_evidence


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compose motifs in each seed-row × seed-column cell by splitting inherited attributes across axes.

    Input structure:
    - Full separator rows and columns divide the canvas into equal open cells.
    - A few cells contain monochrome seed motifs; their colors are all distinct, and no two seeds share a cell row or a cell column.
    - Each seed contributes a unique shape and a unique in-cell bounding-box position, while all other open cells are empty background.

    Rule:
    - Parse the separator lattice and locate every seed cell.
    - For each ordered pair of seeds (A, B), use A's cell row and B's cell column to choose a target cell.
    - Stamp A's normalized shape into that cell, keep A's bounding-box top offset, replace the color with B's color, and use B's bounding-box left offset.
    - Leave the separator lines and the original seed cells unchanged.

    Key insight:
    - The seed motif encodes four visible attributes, and the output is the full row-column recombination where shape plus vertical offset travel with rows while color plus horizontal offset travel with columns.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator_color, row_runs, col_runs, sep_mask = _parse_layout(g)
    _bg, seeds = _parse_seeds(g, separator_color, row_runs, col_runs, sep_mask)
    return _compose_output(g, seeds, row_runs, col_runs)


def _grid_size(cell_size: int, count: int) -> int:
    return int(count) * int(cell_size) + int(count) - 1


def _separator_lines(cell_size: int, count: int) -> np.ndarray:
    return np.array([cell_size + i * (cell_size + 1) for i in range(count - 1)], dtype=int)


def _sample_seed_spec(count: int, cell_size: int) -> tuple[int, list[dict[str, object]]]:
    palette = np.random.permutation(np.arange(1, 10))
    separator_color = int(palette[0])
    seed_colors = [int(c) for c in palette[1 : 1 + count]]
    column_perm = [int(x) for x in np.random.permutation(count)]
    top_offsets = [int(x) for x in np.random.permutation(cell_size - 1)[:count]]
    left_offsets = [int(x) for x in np.random.permutation(cell_size - 1)[:count]]
    shape_ids = [int(x) for x in np.random.permutation(len(SEED_SHAPES))[:count]]

    seeds: list[dict[str, object]] = []
    for row_idx in range(count):
        seeds.append(
            {
                "color": seed_colors[row_idx],
                "row": row_idx,
                "col": column_perm[row_idx],
                "top": top_offsets[row_idx],
                "left": left_offsets[row_idx],
                "shape": SEED_SHAPES[shape_ids[row_idx]],
            }
        )
    return separator_color, seeds


def _render_input_and_output(cell_size: int, count: int, separator_color: int, seeds: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    side = _grid_size(cell_size, count)
    separators = _separator_lines(cell_size, count)
    row_runs = _runs_between_stops(side, separators.tolist())
    col_runs = _runs_between_stops(side, separators.tolist())

    output = np.full((side, side), BACKGROUND, dtype=int)
    output[separators, :] = separator_color
    output[:, separators] = separator_color
    output = _compose_output(output, seeds, row_runs, col_runs)

    inp = np.full((side, side), BACKGROUND, dtype=int)
    inp[separators, :] = separator_color
    inp[:, separators] = separator_color
    for seed in seeds:
        cell_r0 = row_runs[int(seed["row"])][0]
        cell_c0 = col_runs[int(seed["col"])][0]
        _stamp(inp, cell_r0, cell_c0, seed["shape"], int(seed["top"]), int(seed["left"]), int(seed["color"]))
    return inp, output


def _gen_attempt() -> np.ndarray | None:
    count = int(np.random.choice(SEED_COUNTS))
    cell_size = int(np.random.choice(CELL_SIZES))
    separator_color, seeds = _sample_seed_spec(count, cell_size)
    inp, expected = _render_input_and_output(cell_size, count, separator_color, seeds)

    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _validate_semantics(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
