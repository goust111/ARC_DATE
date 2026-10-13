from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
MARKER = 1
TEMPLATE_COLORS = (2, 3, 4)
SEPARATOR = 5

CELL_COUNT = 4
CELL_SIZE_RANGE = (2, 4)
GEN_ATTEMPTS = 400

CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CORNER_TEMPLATE_OFFSETS = [
    [(-1, 0), (0, -1)],
    [(-1, 0), (0, 1)],
    [(1, 0), (0, -1)],
    [(1, 0), (0, 1)],
]


Span = tuple[int, int]
Cell = tuple[int, int]
Offset = tuple[int, int, int]


def _groups(indices: Iterable[int]) -> list[Span]:
    values = sorted(int(idx) for idx in indices)
    assert values, "expected separator lines"
    groups: list[Span] = []
    start = end = values[0]
    for idx in values[1:]:
        if idx == end + 1:
            end = idx
        else:
            groups.append((start, end))
            start = end = idx
    groups.append((start, end))
    return groups


def _spans_between(line_groups: list[Span], limit: int) -> list[Span]:
    spans: list[Span] = []
    start = 0
    for line_start, line_end in line_groups:
        if start < line_start:
            spans.append((start, line_start - 1))
        start = line_end + 1
    if start < limit:
        spans.append((start, limit - 1))
    assert spans, "expected lattice cells between separator lines"
    sizes = {end - start + 1 for start, end in spans}
    assert len(sizes) == 1, "all lattice cells must share one size"
    return spans


def _separator_color(grid: np.ndarray) -> int:
    row_colors = {
        int(grid[r, 0])
        for r in range(grid.shape[0])
        if np.all(grid[r, :] == grid[r, 0])
    }
    col_colors = {
        int(grid[0, c])
        for c in range(grid.shape[1])
        if np.all(grid[:, c] == grid[0, c])
    }
    candidates = row_colors & col_colors
    assert len(candidates) == 1, "expected one full-row/full-column separator color"
    return candidates.pop()


def _compress_lattice(grid: np.ndarray) -> tuple[list[Span], list[Span], np.ndarray, int]:
    separator = _separator_color(grid)
    row_lines = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == separator)]
    col_lines = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == separator)]
    row_spans = _spans_between(_groups(row_lines), grid.shape[0])
    col_spans = _spans_between(_groups(col_lines), grid.shape[1])

    cells = np.zeros((len(row_spans), len(col_spans)), dtype=int)
    for r, (r0, r1) in enumerate(row_spans):
        for c, (c0, c1) in enumerate(col_spans):
            block = grid[r0 : r1 + 1, c0 : c1 + 1]
            values = np.unique(block)
            assert len(values) == 1, "each lattice cell must be uniformly colored"
            cells[r, c] = int(values[0])
    return row_spans, col_spans, cells, separator


def _expand_lattice(cells: np.ndarray, cell_size: int, separator: int = SEPARATOR) -> np.ndarray:
    n_rows, n_cols = cells.shape
    out_h = n_rows * cell_size + (n_rows - 1)
    out_w = n_cols * cell_size + (n_cols - 1)
    grid = np.full((out_h, out_w), int(separator), dtype=int)
    for r in range(n_rows):
        r0 = r * (cell_size + 1)
        for c in range(n_cols):
            c0 = c * (cell_size + 1)
            grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = int(cells[r, c])
    return grid


def _restore_full_grid(
    cells: np.ndarray,
    row_spans: list[Span],
    col_spans: list[Span],
    shape: tuple[int, int],
    separator: int,
) -> np.ndarray:
    grid = np.full(shape, int(separator), dtype=int)
    for r, (r0, r1) in enumerate(row_spans):
        for c, (c0, c1) in enumerate(col_spans):
            grid[r0 : r1 + 1, c0 : c1 + 1] = int(cells[r, c])
    return grid


def _cardinal_non_background(cells: np.ndarray, r: int, c: int) -> list[Offset]:
    neighbors: list[Offset] = []
    for dr, dc in CARDINALS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < cells.shape[0] and 0 <= cc < cells.shape[1]:
            value = int(cells[rr, cc])
            if value != BACKGROUND:
                neighbors.append((dr, dc, value))
    return neighbors


def _find_exemplar_and_template(cells: np.ndarray) -> tuple[Cell, list[Offset]]:
    blue_positions = [tuple(map(int, rc)) for rc in np.argwhere(cells == MARKER)]
    assert len(blue_positions) >= 3, "expected one decorated blue marker and at least two isolated ones"
    decorated: list[tuple[Cell, list[Offset]]] = []
    for r, c in blue_positions:
        offsets = _cardinal_non_background(cells, r, c)
        if offsets:
            decorated.append(((r, c), offsets))
    assert len(decorated) == 1, "expected a unique decorated blue marker"
    center, offsets = decorated[0]
    assert 2 <= len(offsets) <= 3, "decorated exemplar must expose a 2-3 neighbor template"
    colors = [color for _dr, _dc, color in offsets]
    assert len(colors) == len(set(colors)), "template colors must be pairwise distinct"
    assert all(color in TEMPLATE_COLORS for color in colors), "unexpected template color"
    return center, offsets


def _stamp_and_erase(cells: np.ndarray) -> np.ndarray:
    exemplar, offsets = _find_exemplar_and_template(cells)
    stamped = cells.copy()
    for r, c in [tuple(map(int, rc)) for rc in np.argwhere(cells == MARKER)]:
        if (r, c) == exemplar:
            continue
        for dr, dc, color in offsets:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < stamped.shape[0] and 0 <= cc < stamped.shape[1]):
                continue
            assert stamped[rr, cc] == BACKGROUND, "template stamp conflict"
            stamped[rr, cc] = int(color)

    out = stamped.copy()
    for r in range(stamped.shape[0]):
        for c in range(stamped.shape[1]):
            if stamped[r, c] == BACKGROUND:
                continue
            count = 0
            for dr, dc in CARDINALS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < stamped.shape[0] and 0 <= cc < stamped.shape[1] and stamped[rr, cc] != BACKGROUND:
                    count += 1
            if count == 0:
                out[r, c] = BACKGROUND
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the decorated blue marker's cardinal template onto all other blue markers, then erase isolated lattice cells.

    Input structure:
    - A 4x4 separator lattice is embedded in a larger pixel grid; each lattice cell is uniformly colored and separators are full rows/columns.
    - Exactly one blue marker cell is decorated by 2-3 colored cardinal neighbors chosen from red, green, and yellow.
    - The remaining blue marker cells are isolated in the compressed lattice and sit on an otherwise black background.

    Rule:
    - Compress the separator lattice to one abstract color per cell.
    - Find the unique decorated blue marker and record its non-background up/down/left/right offsets with colors.
    - Stamp those offsets around every other blue marker, clipping any offset that leaves the lattice and painting only background cells.
    - On the stamped lattice, count each non-background cell's non-background 4-neighbors; erase cells whose count is zero.
    - Expand the resulting lattice back to the original separator-grid scale.

    Key insight:
    - The separator grid is only a carrier: the real task is first copying the exemplar's cardinal neighborhood in lattice coordinates, then applying the erase rule to the stamped abstract grid.
    """
    grid = np.asarray(grid, dtype=int)
    assert grid.ndim == 2 and grid.size > 0, "expected a non-empty 2D grid"
    row_spans, col_spans, cells, separator = _compress_lattice(grid)
    out_cells = _stamp_and_erase(cells)
    return _restore_full_grid(out_cells, row_spans, col_spans, grid.shape, separator)


def _in_bounds(pos: Cell) -> bool:
    r, c = pos
    return 0 <= r < CELL_COUNT and 0 <= c < CELL_COUNT


def _valid_targets(pos: Cell, offsets: list[Offset]) -> list[Cell]:
    r, c = pos
    targets: list[Cell] = []
    for dr, dc, _color in offsets:
        rr, cc = r + dr, c + dc
        if _in_bounds((rr, cc)):
            targets.append((rr, cc))
    return targets


def _manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _category(valid_count: int, total_offsets: int) -> str:
    if valid_count == 0:
        return "erased"
    if valid_count == total_offsets:
        return "full"
    return "partial"


def _place_marker(
    wanted: str,
    cells: np.ndarray,
    offsets: list[Offset],
    input_nonbg: set[Cell],
    output_reserved: set[Cell],
) -> tuple[Cell, list[Cell]] | None:
    candidates: list[tuple[Cell, list[Cell]]] = []
    positions = [(r, c) for r in range(CELL_COUNT) for c in range(CELL_COUNT)]
    random.shuffle(positions)
    for pos in positions:
        if int(cells[pos]) != BACKGROUND:
            continue
        if any(_manhattan(pos, other) == 1 for other in input_nonbg):
            continue
        valid = _valid_targets(pos, offsets)
        if _category(len(valid), len(offsets)) != wanted:
            continue
        if pos in output_reserved:
            continue
        if any(target in output_reserved for target in valid):
            continue
        candidates.append((pos, valid))
    if not candidates:
        return None
    return random.choice(candidates)


def _stamped_only(cells: np.ndarray) -> np.ndarray:
    exemplar, offsets = _find_exemplar_and_template(cells)
    stamped = cells.copy()
    for r, c in [tuple(map(int, rc)) for rc in np.argwhere(cells == MARKER)]:
        if (r, c) == exemplar:
            continue
        for dr, dc, color in offsets:
            rr, cc = r + dr, c + dc
            if 0 <= rr < stamped.shape[0] and 0 <= cc < stamped.shape[1]:
                stamped[rr, cc] = int(color)
    return stamped


def _semantic_checks(inp_cells: np.ndarray, out_cells: np.ndarray) -> bool:
    blue_in = {tuple(map(int, rc)) for rc in np.argwhere(inp_cells == MARKER)}
    blue_out = {tuple(map(int, rc)) for rc in np.argwhere(out_cells == MARKER)}
    if len(blue_in) < 4:
        return False
    exemplar, offsets = _find_exemplar_and_template(inp_cells)
    for pos in blue_in:
        neighbors = _cardinal_non_background(inp_cells, *pos)
        if pos == exemplar:
            if len(neighbors) != len(offsets):
                return False
        elif neighbors:
            return False

    if not (blue_in - blue_out):
        return False
    if len(blue_out - {exemplar}) < 1:
        return False

    stamped = _stamped_only(inp_cells)
    if np.array_equal(stamped, out_cells):
        return False
    if np.array_equal(inp_cells, out_cells):
        return False

    added_nonbg = {
        tuple(map(int, rc))
        for rc in np.argwhere((out_cells != BACKGROUND) & (inp_cells == BACKGROUND))
    }
    if not added_nonbg:
        return False
    return True


def _generate_cells() -> np.ndarray | None:
    offsets2 = random.choice(CORNER_TEMPLATE_OFFSETS)
    colors = random.sample(list(TEMPLATE_COLORS), 2)
    offsets = [(dr, dc, color) for (dr, dc), color in zip(offsets2, colors)]

    exemplar_candidates = []
    for r in range(CELL_COUNT):
        for c in range(CELL_COUNT):
            if all(_in_bounds((r + dr, c + dc)) for dr, dc, _color in offsets):
                exemplar_candidates.append((r, c))
    random.shuffle(exemplar_candidates)

    for exemplar in exemplar_candidates:
        cells = np.zeros((CELL_COUNT, CELL_COUNT), dtype=int)
        cells[exemplar] = MARKER
        input_nonbg = {exemplar}
        output_reserved = {exemplar}
        for dr, dc, color in offsets:
            rr, cc = exemplar[0] + dr, exemplar[1] + dc
            cells[rr, cc] = int(color)
            input_nonbg.add((rr, cc))
            output_reserved.add((rr, cc))

        placed_markers: list[Cell] = []
        for wanted in ("erased", "partial", "full"):
            result = _place_marker(wanted, cells, offsets, input_nonbg, output_reserved)
            if result is None:
                break
            pos, valid = result
            cells[pos] = MARKER
            input_nonbg.add(pos)
            output_reserved.add(pos)
            output_reserved.update(valid)
            placed_markers.append(pos)
        else:
            if random.random() < 0.5:
                extra_kind = random.choice(["partial", "full"])
                extra = _place_marker(extra_kind, cells, offsets, input_nonbg, output_reserved)
                if extra is not None:
                    pos, valid = extra
                    cells[pos] = MARKER
                    input_nonbg.add(pos)
                    output_reserved.add(pos)
                    output_reserved.update(valid)
                    placed_markers.append(pos)

            if len(placed_markers) >= 3:
                out_cells = _stamp_and_erase(cells)
                if _semantic_checks(cells, out_cells):
                    return cells
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        cells = _generate_cells()
        if cells is None:
            continue
        cell_size = random.randint(*CELL_SIZE_RANGE)
        grid = _expand_lattice(cells, cell_size, separator=SEPARATOR)
        try:
            out = main(grid)
        except AssertionError:
            continue
        out_cells = _stamp_and_erase(cells)
        expected = _expand_lattice(out_cells, cell_size, separator=SEPARATOR)
        if np.array_equal(out, expected):
            return grid
    raise RuntimeError("failed to generate a valid input for task 8")
