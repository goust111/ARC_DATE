from __future__ import annotations

from collections import Counter
import random

import numpy as np


Span = tuple[int, int]
Cell = tuple[int, int]
Instruction = tuple[Cell, Cell, int]


def _groups(indices: list[int]) -> list[Span]:
    assert indices, "expected separator lines"
    groups: list[Span] = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
        else:
            groups.append((start, prev))
            start = prev = idx
    groups.append((start, prev))
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
    assert len(sizes) == 1, "all cells must share one side length"
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
    assert len(candidates) == 1, "expected one separator color"
    return candidates.pop()


def _parse_lattice(grid: np.ndarray) -> tuple[list[Span], list[Span], int, int, list[Instruction]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator = _separator_color(g)
    row_lines = [r for r in range(g.shape[0]) if np.all(g[r, :] == separator)]
    col_lines = [c for c in range(g.shape[1]) if np.all(g[:, c] == separator)]
    row_groups = _groups(row_lines)
    col_groups = _groups(col_lines)
    assert all(r0 == r1 for r0, r1 in row_groups), "separator lines must be one pixel thick"
    assert all(c0 == c1 for c0, c1 in col_groups), "separator lines must be one pixel thick"

    row_spans = _spans_between(row_groups, g.shape[0])
    col_spans = _spans_between(col_groups, g.shape[1])
    assert len(row_spans) == len(col_spans), "expected an N x N cell lattice"
    n = len(row_spans)
    assert n >= 2, "expected at least a 2x2 lattice"

    heights = {r1 - r0 + 1 for r0, r1 in row_spans}
    widths = {c1 - c0 + 1 for c0, c1 in col_spans}
    assert len(heights) == 1 and len(widths) == 1, "cell sizes must be uniform"
    cell_h = next(iter(heights))
    cell_w = next(iter(widths))
    assert cell_h == cell_w == n, "each cell must be N x N where the lattice is N x N"

    cell_pixels: list[int] = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            block = g[r0 : r1 + 1, c0 : c1 + 1]
            assert not np.any(block == separator), "separator color cannot appear inside a cell"
            cell_pixels.extend(int(v) for v in block.ravel())
    bg = Counter(cell_pixels).most_common(1)[0][0]

    instructions: list[Instruction] = []
    for cr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            block = g[r0 : r1 + 1, c0 : c1 + 1]
            mask = block != bg
            count = int(mask.sum())
            if count == 0:
                continue
            assert count == 1, "each instruction cell must contain exactly one colored pixel"
            rr, cc2 = map(int, np.argwhere(mask)[0])
            color = int(block[rr, cc2])
            assert color not in (bg, separator)
            instructions.append(((cr, cc), (rr, cc2), color))

    assert instructions, "expected at least one instruction pixel"
    return row_spans, col_spans, separator, int(bg), instructions


def _uniform_cell_grid(
    shape: tuple[int, int],
    row_spans: list[Span],
    col_spans: list[Span],
    separator: int,
    cells: np.ndarray,
) -> np.ndarray:
    out = np.full(shape, separator, dtype=int)
    for r, (r0, r1) in enumerate(row_spans):
        for c, (c0, c1) in enumerate(col_spans):
            out[r0 : r1 + 1, c0 : c1 + 1] = int(cells[r, c])
    return out


def _apply_instructions(n: int, bg: int, instructions: list[Instruction]) -> np.ndarray:
    cells = np.full((n, n), bg, dtype=int)
    for _source, target, color in instructions:
        tr, tc = target
        prev = int(cells[tr, tc])
        assert prev in (bg, color), "two instructions cannot assign different colors to one target"
        cells[tr, tc] = color
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each instruction pixel's local coordinates as a whole-lattice cell address and paint that cell solid.

    Input structure:
    - The grid is partitioned by one-color separator lines into an N x N lattice of cells.
    - Every cell interior is N x N pixels, with one dominant background color.
    - Most cells are pure background; each non-empty cell contains exactly one colored instruction pixel.

    Rule:
    - For each instruction pixel, take its local cell coordinates (r, c) inside its own N x N cell.
    - Interpret that local position as the address of the target cell at lattice row r and lattice column c.
    - Fill the entire target cell interior with the instruction pixel's color.
    - After all instructions are applied, erase the original instruction pixels; any instruction cell not targeted by any instruction becomes pure background.
    - Preserve all separator lines unchanged.

    Key insight:
    - A pixel's color says what to write, while its position inside the miniature N x N cell says where to write at the coarser cell-grid scale.
    """
    row_spans, col_spans, separator, bg, instructions = _parse_lattice(grid)
    n = len(row_spans)
    painted_cells = _apply_instructions(n, bg, instructions)
    return _uniform_cell_grid(np.asarray(grid).shape, row_spans, col_spans, separator, painted_cells)


GEN_ATTEMPTS = 500
N_RANGE = (3, 5)
BG = 0
NON_BG_COLORS = list(range(1, 10))


def _cell_origin(n: int, cell: Cell) -> tuple[int, int]:
    r, c = cell
    return r * (n + 1), c * (n + 1)


def _empty_lattice(n: int, separator: int, fill: int) -> np.ndarray:
    side = n * n + (n - 1)
    grid = np.full((side, side), separator, dtype=int)
    for r in range(n):
        for c in range(n):
            r0, c0 = _cell_origin(n, (r, c))
            grid[r0 : r0 + n, c0 : c0 + n] = fill
    return grid


def _render_input(n: int, separator: int, instructions: list[Instruction]) -> np.ndarray:
    grid = _empty_lattice(n, separator, BG)
    for source, target, color in instructions:
        sr, sc = source
        tr, tc = target
        r0, c0 = _cell_origin(n, (sr, sc))
        grid[r0 + tr, c0 + tc] = color
    return grid


def _render_uniform_output(n: int, separator: int, cells: np.ndarray) -> np.ndarray:
    grid = _empty_lattice(n, separator, BG)
    for r in range(n):
        for c in range(n):
            if cells[r, c] == BG:
                continue
            r0, c0 = _cell_origin(n, (r, c))
            grid[r0 : r0 + n, c0 : c0 + n] = int(cells[r, c])
    return grid


def _choose_layout(n: int) -> tuple[list[Cell], list[Cell], list[Cell]]:
    cells = [(r, c) for r in range(n) for c in range(n)]
    options: list[tuple[int, int]] = []
    for k in range(3, min(5, n * n - 1) + 1):
        for overlap in range(max(1, 2 * k - n * n), k):
            options.append((k, overlap))
    k, overlap = random.choice(options)

    chosen = random.sample(cells, 2 * k - overlap)
    overlap_cells = chosen[:overlap]
    source_only = chosen[overlap : overlap + (k - overlap)]
    target_only = chosen[overlap + (k - overlap) :]
    sources = overlap_cells + source_only
    targets = overlap_cells + target_only
    random.shuffle(sources)
    random.shuffle(targets)
    return sources, targets, overlap_cells


def _pair_sources_to_targets(sources: list[Cell], targets: list[Cell], overlap_cells: list[Cell]) -> dict[Cell, Cell]:
    mapping = {source: target for source, target in zip(sources, random.sample(targets, len(targets)))}
    if overlap_cells and random.random() < 0.5:
        self_cell = random.choice(overlap_cells)
        current_target = mapping[self_cell]
        if current_target != self_cell:
            other_source = next(source for source, target in mapping.items() if target == self_cell)
            mapping[self_cell], mapping[other_source] = self_cell, current_target
    return mapping


def _assign_target_colors(targets: list[Cell], separator: int) -> dict[Cell, int]:
    num_colors = random.randint(2, min(4, len(targets)))
    palette = random.sample([c for c in NON_BG_COLORS if c != separator], num_colors)
    shuffled = random.sample(targets, len(targets))
    target_colors: dict[Cell, int] = {}
    for cell, color in zip(shuffled[:num_colors], palette):
        target_colors[cell] = color
    for cell in shuffled[num_colors:]:
        target_colors[cell] = random.choice(palette)
    return target_colors


def _instructions_from_mapping(
    source_to_target: dict[Cell, Cell], target_colors: dict[Cell, int]
) -> list[Instruction]:
    return [
        (source, target, target_colors[target])
        for source, target in source_to_target.items()
    ]


def _wrong_fill_sources(n: int, separator: int, instructions: list[Instruction]) -> np.ndarray:
    cells = np.full((n, n), BG, dtype=int)
    for source, _target, color in instructions:
        sr, sc = source
        cells[sr, sc] = color
    return _render_uniform_output(n, separator, cells)


def _wrong_swap_coords(n: int, separator: int, instructions: list[Instruction]) -> np.ndarray:
    cells = np.full((n, n), BG, dtype=int)
    for _source, target, color in instructions:
        tr, tc = target
        prev = int(cells[tc, tr])
        if prev not in (BG, color):
            return np.full((1, 1), -1, dtype=int)
        cells[tc, tr] = color
    return _render_uniform_output(n, separator, cells)


def _wrong_keep_source_pixels(inp: np.ndarray, n: int, instructions: list[Instruction]) -> np.ndarray:
    wrong = inp.copy()
    for _source, target, color in instructions:
        r0, c0 = _cell_origin(n, target)
        wrong[r0 : r0 + n, c0 : c0 + n] = color
    return wrong


def _wrong_single_pixels(n: int, separator: int, instructions: list[Instruction]) -> np.ndarray:
    wrong = _empty_lattice(n, separator, BG)
    for _source, target, color in instructions:
        tr, tc = target
        r0, c0 = _cell_origin(n, target)
        wrong[r0 + tr, c0 + tc] = color
    return wrong


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        row_spans, col_spans, separator, bg, instructions = _parse_lattice(inp)
    except AssertionError:
        return False

    n = len(row_spans)
    if bg != BG or len(instructions) < 3:
        return False

    sources = [source for source, _target, _color in instructions]
    targets = [target for _source, target, _color in instructions]
    colors = {color for _source, _target, color in instructions}
    source_set = set(sources)
    target_set = set(targets)

    if len(source_set) != len(instructions) or len(target_set) != len(instructions):
        return False
    if len(colors) < 2:
        return False
    if not (source_set - target_set):
        return False
    if not (target_set - source_set):
        return False
    if not (source_set & target_set):
        return False
    if not any(tr != tc for tr, tc in targets):
        return False
    if len({tr for tr, _tc in targets}) < 2 or len({tc for _tr, tc in targets}) < 2:
        return False

    correct = main(inp)
    if not np.array_equal(correct, expected):
        return False

    cells = _apply_instructions(n, bg, instructions)
    if int(np.count_nonzero(cells != bg)) < 3:
        return False

    wrongs = [
        _wrong_fill_sources(n, separator, instructions),
        _wrong_swap_coords(n, separator, instructions),
        _wrong_keep_source_pixels(inp, n, instructions),
        _wrong_single_pixels(n, separator, instructions),
    ]
    if any(np.array_equal(correct, wrong) for wrong in wrongs):
        return False

    output_cells = np.zeros((n, n), dtype=int)
    for r, (r0, r1) in enumerate(row_spans):
        for c, (c0, c1) in enumerate(col_spans):
            block = correct[r0 : r1 + 1, c0 : c1 + 1]
            values = np.unique(block)
            if len(values) != 1:
                return False
            output_cells[r, c] = int(values[0])
    if not np.array_equal(output_cells, cells):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*N_RANGE)
    sources, targets, overlap_cells = _choose_layout(n)

    separator = random.choice(NON_BG_COLORS)
    target_colors = _assign_target_colors(targets, separator)
    source_to_target = _pair_sources_to_targets(sources, targets, overlap_cells)
    instructions = _instructions_from_mapping(source_to_target, target_colors)

    inp = _render_input(n, separator, instructions)
    expected = _render_uniform_output(n, separator, _apply_instructions(n, BG, instructions))
    if _semantic_checks(inp, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid micro-address stamp input")
