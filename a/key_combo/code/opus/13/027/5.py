from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
OVERLAP_COLOR = 4
SEPARATOR_COLOR = 5
MOTIF_COLORS = (1, 2, 3, 6, 7, 8, 9)
CELL_SIZE_OPTIONS = (3, 4)
CELL_COUNT_OPTIONS = (5, 6)
MOTIF_SIZE_RANGE = (2, 4)
GENERATION_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, grid.reshape(-1)))
    if exclude is not None:
        colors -= {int(color) for color in exclude}
    return sorted(colors)


def _open_runs(length: int, separators: np.ndarray) -> list[tuple[int, int]]:
    separator_set = set(map(int, separators))
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx in range(length):
        if idx in separator_set:
            if start is not None:
                runs.append((start, idx - 1))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        runs.append((start, length - 1))
    return runs


def _run_index(runs: list[tuple[int, int]], value: int) -> int:
    hits = [idx for idx, (lo, hi) in enumerate(runs) if lo <= value <= hi]
    assert len(hits) == 1, "every motif pixel must lie inside exactly one open cell"
    return hits[0]


def _run_length(run: tuple[int, int]) -> int:
    return int(run[1] - run[0] + 1)


def _full_cell_size(runs: list[tuple[int, int]]) -> int:
    lengths = [_run_length(run) for run in runs]
    counts = Counter(lengths)
    best_count = max(counts.values())
    best_lengths = [length for length, count in counts.items() if count == best_count]
    return max(best_lengths)


def _extract_sources(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int, int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    row_lines = np.flatnonzero(np.all(g == SEPARATOR_COLOR, axis=1))
    col_lines = np.flatnonzero(np.all(g == SEPARATOR_COLOR, axis=0))
    assert len(row_lines) >= 1 and len(col_lines) >= 1, "expected full separator lines"

    row_runs = _open_runs(g.shape[0], row_lines)
    col_runs = _open_runs(g.shape[1], col_lines)
    assert row_runs and col_runs, "separator grid must leave open cells"

    full_row_size = _full_cell_size(row_runs)
    full_col_size = _full_cell_size(col_runs)
    first_full_row = next(idx for idx, run in enumerate(row_runs) if _run_length(run) == full_row_size)
    first_full_col = next(idx for idx, run in enumerate(col_runs) if _run_length(run) == full_col_size)

    bg = _get_bg(g)
    motif_colors = _colors_present(g, exclude={bg, SEPARATOR_COLOR})
    assert len(motif_colors) == 2, "expected exactly two motif colors"

    sources: list[dict] = []
    for color in motif_colors:
        coords = np.argwhere(g == color)
        assert 2 <= len(coords) <= 4, "each seed motif should contain 2-4 pixels"
        row_ids = {_run_index(row_runs, int(r)) for r, _c in coords}
        col_ids = {_run_index(col_runs, int(c)) for _r, c in coords}
        assert len(row_ids) == 1 and len(col_ids) == 1, "each motif must occupy one cell"

        raw_row = row_ids.pop()
        raw_col = col_ids.pop()
        r0 = row_runs[raw_row][0]
        c0 = col_runs[raw_col][0]
        offsets = sorted((int(r - r0), int(c - c0)) for r, c in coords)
        sources.append(
            {
                "color": int(color),
                "raw_row": int(raw_row),
                "raw_col": int(raw_col),
                "row": int(raw_row - first_full_row),
                "col": int(raw_col - first_full_col),
                "offsets": offsets,
            }
        )

    assert (sources[0]["raw_row"], sources[0]["raw_col"]) != (sources[1]["raw_row"], sources[1]["raw_col"])
    return row_runs, col_runs, first_full_row, first_full_col, sources


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp two seed motifs into parity-matched scaffold cells and resolve coincident pixels with a yellow XOR-style overlap color.

    Input structure:
    - Full gray separator rows and columns divide the grid into open cells, with edge cells possibly truncated by cropping.
    - Exactly two open cells contain seed motifs made of 2-4 connected non-black, non-gray pixels, and the two motif colors are different.
    - All other open-cell pixels are black, and the separator lines remain intact across the whole grid.

    Rule:
    - Split the canvas into open row-cells and column-cells between the full gray separator lines, and treat the first full cell in the top-left direction as logical cell (0,0).
    - For each seed motif, record its pixel offsets from its source cell's top-left corner, then stamp that motif into every cell whose logical row parity and logical column parity match the source cell.
    - When both motifs stamp into the same target cell, color motif-A-only pixels with motif A's color, motif-B-only pixels with motif B's color, and pixels claimed by both motifs with yellow; clip any offsets that fall outside truncated edge cells.

    Key insight:
    - The scaffold defines a checkerboard of parity classes, and whether yellow overlap appears is controlled implicitly by whether the two source cells belong to the same parity class.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    row_runs, col_runs, first_full_row, first_full_col, sources = _extract_sources(g)
    out = np.full_like(g, BACKGROUND)

    row_lines = np.flatnonzero(np.all(g == SEPARATOR_COLOR, axis=1))
    col_lines = np.flatnonzero(np.all(g == SEPARATOR_COLOR, axis=0))
    out[row_lines, :] = SEPARATOR_COLOR
    out[:, col_lines] = SEPARATOR_COLOR

    for raw_row, (r0, r1) in enumerate(row_runs):
        logical_row = raw_row - first_full_row
        cell_h = r1 - r0 + 1
        for raw_col, (c0, c1) in enumerate(col_runs):
            logical_col = raw_col - first_full_col
            cell_w = c1 - c0 + 1

            masks: list[np.ndarray] = []
            for source in sources:
                if logical_row % 2 != source["row"] % 2 or logical_col % 2 != source["col"] % 2:
                    masks.append(np.zeros((cell_h, cell_w), dtype=bool))
                    continue
                mask = np.zeros((cell_h, cell_w), dtype=bool)
                for dr, dc in source["offsets"]:
                    if dr < cell_h and dc < cell_w:
                        mask[dr, dc] = True
                masks.append(mask)

            patch = out[r0 : r1 + 1, c0 : c1 + 1]
            a_mask, b_mask = masks
            a_color = int(sources[0]["color"])
            b_color = int(sources[1]["color"])
            patch[a_mask & ~b_mask] = a_color
            patch[b_mask & ~a_mask] = b_color
            patch[a_mask & b_mask] = OVERLAP_COLOR

    return out


def _neighbors4(cell: tuple[int, int], size: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= nr < size and 0 <= nc < size:
            out.append((nr, nc))
    return out


def _sample_connected_offsets(size: int) -> list[tuple[int, int]]:
    target = random.randint(*MOTIF_SIZE_RANGE)
    cells = {(random.randrange(size), random.randrange(size))}
    while len(cells) < target:
        frontier = []
        for cell in cells:
            for nxt in _neighbors4(cell, size):
                if nxt not in cells:
                    frontier.append(nxt)
        cells.add(random.choice(frontier))
    return sorted(cells)


def _sample_motif_pair(size: int, same_parity: bool) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    for _ in range(200):
        first = _sample_connected_offsets(size)
        second = _sample_connected_offsets(size)
        if not same_parity:
            return first, second
        first_set = set(first)
        second_set = set(second)
        if first_set & second_set and first_set != second_set and first_set - second_set and second_set - first_set:
            return first, second
    raise RuntimeError("failed to sample a suitable motif pair")


def _build_scaffold(size: int, row_cells: int, col_cells: int) -> np.ndarray:
    height = row_cells * size + (row_cells - 1)
    width = col_cells * size + (col_cells - 1)
    grid = np.full((height, width), BACKGROUND, dtype=int)
    for r in range(size, height, size + 1):
        grid[r, :] = SEPARATOR_COLOR
    for c in range(size, width, size + 1):
        grid[:, c] = SEPARATOR_COLOR
    return grid


def _crop_scaffold(grid: np.ndarray, size: int) -> np.ndarray:
    if random.random() < 0.35:
        top = bottom = left = right = 0
    else:
        top = random.randint(0, size - 1)
        bottom = random.randint(0, size - 1)
        left = random.randint(0, size - 1)
        right = random.randint(0, size - 1)
        if top == bottom == left == right == 0:
            top = random.randint(1, size - 1)
    h, w = grid.shape
    return grid[top : h - bottom, left : w - right].copy()


def _candidate_source_cells(
    row_runs: list[tuple[int, int]],
    col_runs: list[tuple[int, int]],
    full_size: int,
) -> tuple[int, int, list[tuple[int, int]]]:
    full_rows = [idx for idx, run in enumerate(row_runs) if _run_length(run) == full_size]
    full_cols = [idx for idx, run in enumerate(col_runs) if _run_length(run) == full_size]
    first_full_row = full_rows[0]
    first_full_col = full_cols[0]
    candidates = [(r, c) for r in full_rows for c in full_cols]
    return first_full_row, first_full_col, candidates


def _choose_source_pair(
    candidates: list[tuple[int, int]],
    first_full_row: int,
    first_full_col: int,
    same_parity: bool,
) -> tuple[tuple[int, int], tuple[int, int]]:
    valid = []
    for idx, first in enumerate(candidates):
        fr, fc = first[0] - first_full_row, first[1] - first_full_col
        for second in candidates[idx + 1 :]:
            sr, sc = second[0] - first_full_row, second[1] - first_full_col
            parity_equal = (fr % 2, fc % 2) == (sr % 2, sc % 2)
            if parity_equal != same_parity:
                continue
            if abs(first[0] - second[0]) + abs(first[1] - second[1]) < 2:
                continue
            valid.append((first, second))
    assert valid, "expected at least one source-cell pair for the selected branch"
    return random.choice(valid)


def _place_motif(
    grid: np.ndarray,
    row_runs: list[tuple[int, int]],
    col_runs: list[tuple[int, int]],
    cell: tuple[int, int],
    offsets: list[tuple[int, int]],
    color: int,
) -> None:
    r0 = row_runs[cell[0]][0]
    c0 = col_runs[cell[1]][0]
    for dr, dc in offsets:
        grid[r0 + dr, c0 + dc] = int(color)


def _nondegenerate_output(out: np.ndarray, a_color: int, b_color: int, same_parity: bool) -> bool:
    if int(np.count_nonzero(out == a_color)) == 0 or int(np.count_nonzero(out == b_color)) == 0:
        return False
    if same_parity:
        if int(np.count_nonzero(out == OVERLAP_COLOR)) == 0:
            return False
    else:
        if int(np.count_nonzero(out == OVERLAP_COLOR)) != 0:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    size = random.choice(CELL_SIZE_OPTIONS)
    row_cells = random.choice(CELL_COUNT_OPTIONS)
    col_cells = random.choice(CELL_COUNT_OPTIONS)
    same_parity = bool(random.getrandbits(1))

    base = _build_scaffold(size, row_cells, col_cells)
    grid = _crop_scaffold(base, size)
    row_runs = _open_runs(grid.shape[0], np.flatnonzero(np.all(grid == SEPARATOR_COLOR, axis=1)))
    col_runs = _open_runs(grid.shape[1], np.flatnonzero(np.all(grid == SEPARATOR_COLOR, axis=0)))
    first_full_row, first_full_col, candidates = _candidate_source_cells(row_runs, col_runs, size)
    source_a, source_b = _choose_source_pair(candidates, first_full_row, first_full_col, same_parity)

    color_a, color_b = random.sample(MOTIF_COLORS, 2)
    offsets_a, offsets_b = _sample_motif_pair(size, same_parity)
    _place_motif(grid, row_runs, col_runs, source_a, offsets_a, color_a)
    _place_motif(grid, row_runs, col_runs, source_b, offsets_b, color_b)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not _nondegenerate_output(out, color_a, color_b, same_parity):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
