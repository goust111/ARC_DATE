import random
from itertools import combinations

import numpy as np


BACKGROUND = 0
SEPARATOR = 7
BRIDGE = 8
PAIR_COLORS = (1, 2, 3, 4, 5, 6, 9)
GENERATION_ATTEMPTS = 400
PAIR_BLUEPRINTS = (
    (("H", True), ("V", False), ("H", False)),
    (("V", True), ("H", False), ("V", False)),
    (("H", True), ("V", False), ("V", True), ("H", False)),
    (("V", True), ("H", False), ("H", True), ("V", False)),
)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _open_runs(length: int, lines: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = 0
    for idx in sorted(lines):
        if start <= idx - 1:
            runs.append((start, idx - 1))
        start = idx + 1
    if start <= length - 1:
        runs.append((start, length - 1))
    return runs


def _run_index(runs: list[tuple[int, int]], pos: int) -> int:
    for idx, (start, end) in enumerate(runs):
        if start <= pos <= end:
            return idx
    raise AssertionError(f"position {pos} does not lie in any open run")


def _separator_structure(grid: np.ndarray) -> tuple[int, list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    candidates: list[tuple[int, list[int], list[int]]] = []
    for color in np.unique(g):
        color = int(color)
        if color == bg or color == BRIDGE:
            continue
        row_lines = np.flatnonzero(np.all(g == color, axis=1)).tolist()
        col_lines = np.flatnonzero(np.all(g == color, axis=0)).tolist()
        if row_lines and col_lines:
            candidates.append((color, row_lines, col_lines))

    assert len(candidates) == 1, "expected exactly one separator color"
    separator, row_lines, col_lines = candidates[0]
    row_runs = _open_runs(g.shape[0], row_lines)
    col_runs = _open_runs(g.shape[1], col_lines)
    assert row_runs and col_runs, "separator grid must leave open cells"
    assert len({r1 - r0 for r0, r1 in row_runs}) == 1, "open row cells must have equal heights"
    assert len({c1 - c0 for c0, c1 in col_runs}) == 1, "open col cells must have equal widths"
    return separator, row_lines, col_lines, row_runs, col_runs


def _straight_gap_path(
    grid: np.ndarray, start: tuple[int, int], end: tuple[int, int], bg: int, separator: int
) -> list[tuple[int, int]] | None:
    r1, c1 = map(int, start)
    r2, c2 = map(int, end)
    if r1 == r2:
        c_lo, c_hi = sorted((c1, c2))
        path = [(r1, c) for c in range(c_lo + 1, c_hi)]
    elif c1 == c2:
        r_lo, r_hi = sorted((r1, r2))
        path = [(r, c1) for r in range(r_lo + 1, r_hi)]
    else:
        return None
    if not path:
        return None
    if any(int(grid[r, c]) not in (bg, separator) for r, c in path):
        return None
    return path


def _pair_info(grid: np.ndarray) -> tuple[int, int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    separator, _row_lines, _col_lines, row_runs, col_runs = _separator_structure(g)

    pair_infos: list[dict] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) not in (bg, separator, BRIDGE)):
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "each foreground color must appear exactly twice"
        p1 = tuple(map(int, coords[0]))
        p2 = tuple(map(int, coords[1]))
        cell1 = (_run_index(row_runs, p1[0]), _run_index(col_runs, p1[1]))
        cell2 = (_run_index(row_runs, p2[0]), _run_index(col_runs, p2[1]))
        same_parity = (cell1[0] % 2, cell1[1] % 2) == (cell2[0] % 2, cell2[1] % 2)

        candidate_paths: list[list[tuple[int, int]]] = []
        path = _straight_gap_path(g, p1, p2, bg, separator)
        if path is not None:
            candidate_paths.append(path)
        shortest = min(candidate_paths, key=len) if candidate_paths else None

        pair_infos.append(
            {
                "color": color,
                "coords": (p1, p2),
                "cells": (cell1, cell2),
                "same_parity": same_parity,
                "path": shortest,
            }
        )
    return bg, separator, pair_infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Bridge only the same-color pixel pairs whose separator-grid cells share the same parity class.

    Input structure:
    - The grid has a black background and one orange separator color that fills complete rows and columns.
    - Those separator lines partition the board into equal open cells; some cells contain one isolated foreground pixel.
    - Every non-separator foreground color appears exactly twice, and teal is absent from the input.

    Rule:
    - Use the full orange rows and columns to index the open cells by `(row_idx, col_idx)`, starting at `0`.
    - For each same-color pair, locate the cells containing its two pixels and compare the cell parities `(row_idx mod 2, col_idx mod 2)`.
    - If and only if the two cells have identical parity, draw the shortest clear horizontal or vertical gap path between the pixels in teal, repainting any crossed separator cells but not the endpoints themselves.

    Key insight:
    - Color matching alone is insufficient: the hidden gate is whether the two occupied cells belong to the same parity class of the separator-defined lattice.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    _bg, _separator, pair_infos = _pair_info(g)
    out = g.copy()
    for info in pair_infos:
        if not info["same_parity"]:
            continue
        assert info["path"] is not None, "same-parity pairs must expose a clear straight bridge path"
        for r, c in info["path"]:
            out[r, c] = BRIDGE
    return out


def _make_base_grid(n_cells: int, cell_h: int, cell_w: int) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    height = n_cells * cell_h + (n_cells - 1)
    width = n_cells * cell_w + (n_cells - 1)
    grid = np.zeros((height, width), dtype=int)

    row_lines = [cell_h + k * (cell_h + 1) for k in range(n_cells - 1)]
    col_lines = [cell_w + k * (cell_w + 1) for k in range(n_cells - 1)]
    grid[row_lines, :] = SEPARATOR
    grid[:, col_lines] = SEPARATOR
    return grid, _open_runs(height, row_lines), _open_runs(width, col_lines)


def _cell_coord(run: tuple[int, int], offset: int) -> int:
    start, end = run
    coord = start + int(offset)
    assert start <= coord <= end
    return coord


def _interior_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r1, c1 = start
    r2, c2 = end
    if r1 == r2:
        c_lo, c_hi = sorted((c1, c2))
        return [(r1, c) for c in range(c_lo + 1, c_hi)]
    r_lo, r_hi = sorted((r1, r2))
    return [(r, c1) for r in range(r_lo + 1, r_hi)]


def _sample_pair(
    grid: np.ndarray,
    row_runs: list[tuple[int, int]],
    col_runs: list[tuple[int, int]],
    orientation: str,
    bridgeable: bool,
    used_cells: set[tuple[int, int]],
    used_h_rows: set[int],
    used_v_cols: set[int],
    reserved: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]], tuple[int, int], tuple[int, int]] | None:
    n_rows = len(row_runs)
    n_cols = len(col_runs)
    cell_h = row_runs[0][1] - row_runs[0][0] + 1
    cell_w = col_runs[0][1] - col_runs[0][0] + 1

    for _ in range(200):
        if orientation == "H":
            row_choices = [r for r in range(n_rows) if r not in used_h_rows]
            if not row_choices:
                return None
            row_idx = random.choice(row_choices)
            col_pairs = [
                (c1, c2)
                for c1, c2 in combinations(range(n_cols), 2)
                if ((c1 - c2) % 2 == 0) == bridgeable
                and (row_idx, c1) not in used_cells
                and (row_idx, c2) not in used_cells
            ]
            if not col_pairs:
                continue
            c1, c2 = random.choice(col_pairs)
            local_r = random.randrange(cell_h)
            p1 = (_cell_coord(row_runs[row_idx], local_r), _cell_coord(col_runs[c1], random.randrange(cell_w)))
            p2 = (_cell_coord(row_runs[row_idx], local_r), _cell_coord(col_runs[c2], random.randrange(cell_w)))
            cell1, cell2 = (row_idx, c1), (row_idx, c2)
        else:
            col_choices = [c for c in range(n_cols) if c not in used_v_cols]
            if not col_choices:
                return None
            col_idx = random.choice(col_choices)
            row_pairs = [
                (r1, r2)
                for r1, r2 in combinations(range(n_rows), 2)
                if ((r1 - r2) % 2 == 0) == bridgeable
                and (r1, col_idx) not in used_cells
                and (r2, col_idx) not in used_cells
            ]
            if not row_pairs:
                continue
            r1, r2 = random.choice(row_pairs)
            local_c = random.randrange(cell_w)
            p1 = (_cell_coord(row_runs[r1], random.randrange(cell_h)), _cell_coord(col_runs[col_idx], local_c))
            p2 = (_cell_coord(row_runs[r2], random.randrange(cell_h)), _cell_coord(col_runs[col_idx], local_c))
            cell1, cell2 = (r1, col_idx), (r2, col_idx)

        path = _interior_path(p1, p2)
        if not path:
            continue
        touched = set(path) | {p1, p2}
        if touched & reserved:
            continue
        if any(int(grid[r, c]) not in (BACKGROUND, SEPARATOR) for r, c in path):
            continue
        return p1, p2, path, cell1, cell2

    return None


def _attempt_generate() -> np.ndarray | None:
    n_cells = random.choice((4, 5))
    cell_h = random.choice((2, 3))
    cell_w = random.choice((2, 3))
    grid, row_runs, col_runs = _make_base_grid(n_cells, cell_h, cell_w)

    blueprint = random.choice(PAIR_BLUEPRINTS)
    colors = random.sample(PAIR_COLORS, len(blueprint))

    used_cells: set[tuple[int, int]] = set()
    used_h_rows: set[int] = set()
    used_v_cols: set[int] = set()
    reserved: set[tuple[int, int]] = set()

    for color, (orientation, bridgeable) in zip(colors, blueprint):
        placed = _sample_pair(grid, row_runs, col_runs, orientation, bridgeable, used_cells, used_h_rows, used_v_cols, reserved)
        if placed is None:
            return None
        p1, p2, path, cell1, cell2 = placed
        grid[p1] = color
        grid[p2] = color
        used_cells.update((cell1, cell2))
        reserved.update(path)
        reserved.add(p1)
        reserved.add(p2)
        if orientation == "H":
            used_h_rows.add(cell1[0])
        else:
            used_v_cols.add(cell1[1])

    try:
        _validate_generated(grid)
    except AssertionError:
        return None
    return grid


def _validate_generated(grid: np.ndarray) -> None:
    bg, separator, pair_infos = _pair_info(grid)
    assert bg == BACKGROUND
    assert separator == SEPARATOR
    assert len(pair_infos) in (3, 4)

    bridgeable_count = 0
    blocked_count = 0
    blocked_with_clear_gap = 0
    expected = grid.copy()

    for info in pair_infos:
        path = info["path"]
        assert path is not None, "generator must create visually plausible straight corridors for every pair"
        if info["same_parity"]:
            bridgeable_count += 1
            for r, c in path:
                expected[r, c] = BRIDGE
        else:
            blocked_count += 1
            blocked_with_clear_gap += 1

    assert bridgeable_count >= 1, "need at least one actual bridge"
    assert blocked_count >= 1, "need at least one same-color pair that stays unchanged"
    assert blocked_with_clear_gap >= 1, "need a counterexample to the color-only shortcut"
    out = main(grid)
    assert np.array_equal(out, expected), "main() output disagrees with the intended parity rule"
    assert np.any(out == BRIDGE), "output must visibly add at least one bridge"


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid same-parity bridge puzzle")
