from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

DATA_COLORS = (BLUE, RED, GREEN, YELLOW)
CELL_SIZE = 6
SEPARATOR = 1
GRID_SIZE = CELL_SIZE * 3 + SEPARATOR * 2
MAX_ATTEMPTS = 600


def _separator_slices(length: int, separators: list[int]) -> list[slice]:
    prev = 0
    parts: list[slice] = []
    for sep in separators:
        parts.append(slice(prev, sep))
        prev = sep + 1
    parts.append(slice(prev, length))
    return parts


def _grid_partition(grid: np.ndarray) -> tuple[list[slice], list[slice]]:
    row_seps = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    col_seps = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(row_seps) == 2 and len(col_seps) == 2, "expected a 3x3 grey lattice"

    row_parts = _separator_slices(int(grid.shape[0]), row_seps)
    col_parts = _separator_slices(int(grid.shape[1]), col_seps)
    assert len(row_parts) == 3 and len(col_parts) == 3

    heights = [part.stop - part.start for part in row_parts]
    widths = [part.stop - part.start for part in col_parts]
    assert len(set(heights)) == 1 and len(set(widths)) == 1, "cells must be equal-sized"
    assert heights[0] == widths[0], "cells must be square"
    return row_parts, col_parts


def _decode_instruction_mask(cell: np.ndarray) -> np.ndarray:
    h, w = map(int, cell.shape)
    assert h % 3 == 0 and w % 3 == 0, "instruction cell must split into a 3x3 logical grid"
    bh, bw = h // 3, w // 3
    mask = np.zeros((3, 3), dtype=bool)
    for rr in range(3):
        for cc in range(3):
            block = cell[rr * bh : (rr + 1) * bh, cc * bw : (cc + 1) * bw]
            if np.all(block == ORANGE):
                mask[rr, cc] = True
            else:
                assert np.all(block == BLACK), "instruction blocks must be solid orange or solid black"
    return mask


def _color_counts(cell: np.ndarray) -> Counter[int]:
    return Counter(int(v) for v in cell.reshape(-1) if int(v) in DATA_COLORS)


def _hole_position(cell: np.ndarray) -> tuple[int, int]:
    holes = np.argwhere(cell == BLACK)
    assert len(holes) == 1, "each candidate cell must contain exactly one black hole"
    return tuple(map(int, holes[0]))


def _deficit_for_target(counts: Counter[int], target: Counter[int]) -> dict[int, int]:
    deficit: dict[int, int] = {}
    for color, need in target.items():
        have = int(counts.get(color, 0))
        if have < need:
            deficit[int(color)] = int(need - have)
    return deficit


def _completion_info(cells: dict[tuple[int, int], np.ndarray], positions: list[tuple[int, int]]) -> tuple[Counter[int], dict[tuple[int, int], int]] | None:
    area = next(iter(cells.values())).size
    target: Counter[int] = Counter()
    for pos in positions:
        counts = _color_counts(cells[pos])
        for color, count in counts.items():
            target[color] = max(int(target.get(color, 0)), int(count))
    if sum(target.values()) != area:
        return None

    fills: dict[tuple[int, int], int] = {}
    for pos in positions:
        deficit = _deficit_for_target(_color_counts(cells[pos]), target)
        if sum(deficit.values()) != 1 or len(deficit) != 1:
            return None
        fills[pos] = int(next(iter(deficit)))
    return target, fills


def _extract_cells(grid: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    rows, cols = _grid_partition(grid)
    cells: dict[tuple[int, int], np.ndarray] = {}
    for rr, rs in enumerate(rows):
        for cc, cs in enumerate(cols):
            cells[(rr, cc)] = grid[rs, cs]
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the instruction mask and fill only the masked lattice cells so their color multisets become equal.

    Input structure:
    - Grey separator rows and columns split the canvas into a 3x3 lattice of equal square cells.
    - The top-left cell is an instruction region made only of black pixels and orange solid sub-blocks that encode a 3x3 mask.
    - Every other cell is a candidate region containing colors from blue/red/green/yellow plus exactly one black hole.

    Rule:
    - Decode the top-left instruction cell into a 3x3 boolean mask by reading each logical sub-block as on iff it is solid orange.
    - Map that mask onto the 3x3 lattice positions and select the masked candidate cells, always excluding the instruction cell itself.
    - For the selected cells, compute the target color multiset by taking the per-color maximum count across the selected cells.
    - Fill each selected cell's unique black hole with the one color that makes its multiset match the target; leave unselected cells, the instruction cell, and the grey lattice unchanged.

    Key insight:
    - The orange code does not mark pixels to recolor inside a cell; it chooses which cells participate in the shared multiset balancing, and ignoring that mask lets decoy cells either stay wrongly filled or break the inferred multiset.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rows, cols = _grid_partition(g)
    cell_shape = (rows[0].stop - rows[0].start, cols[0].stop - cols[0].start)
    assert cell_shape[0] == cell_shape[1] >= 3

    cells = _extract_cells(g)
    mask = _decode_instruction_mask(cells[(0, 0)])
    selected = [(r, c) for r in range(3) for c in range(3) if mask[r, c] and (r, c) != (0, 0)]
    assert len(selected) >= 3, "need at least three selected candidate cells"

    for pos, cell in cells.items():
        if pos == (0, 0):
            continue
        assert not np.any(cell == GREY), "grey must appear only on separator lines"
        _hole_position(cell)

    completion = _completion_info(cells, selected)
    assert completion is not None, "selected cells must determine a unique shared multiset"
    _target, fills = completion

    out = g.copy()
    for r, c in selected:
        rr = rows[r]
        cc = cols[c]
        local_r, local_c = _hole_position(g[rr, cc])
        out[rr.start + local_r, cc.start + local_c] = int(fills[(r, c)])
    return out


def _random_composition(total: int, parts: int, min_each: int) -> list[int]:
    assert total >= parts * min_each
    remaining = total - parts * min_each
    cuts = sorted(random.sample(range(remaining + parts - 1), parts - 1))
    extras: list[int] = []
    prev = -1
    for cut in cuts + [remaining + parts - 1]:
        extras.append(cut - prev - 1)
        prev = cut
    counts = [min_each + extra for extra in extras]
    random.shuffle(counts)
    return counts


def _build_counter(colors: list[int], counts: list[int]) -> Counter[int]:
    return Counter({int(color): int(count) for color, count in zip(colors, counts)})


def _counter_to_values(counter: Counter[int]) -> list[int]:
    values: list[int] = []
    for color in sorted(counter):
        values.extend([int(color)] * int(counter[color]))
    return values


def _render_data_cell(counts: Counter[int], hole: tuple[int, int]) -> np.ndarray:
    cell = np.zeros((CELL_SIZE, CELL_SIZE), dtype=int)
    values = _counter_to_values(counts)
    assert len(values) == CELL_SIZE * CELL_SIZE - 1
    random.shuffle(values)
    slots = [(r, c) for r in range(CELL_SIZE) for c in range(CELL_SIZE) if (r, c) != hole]
    for (r, c), value in zip(slots, values):
        cell[r, c] = int(value)
    return cell


def _render_instruction_cell(mask_on: set[tuple[int, int]]) -> np.ndarray:
    cell = np.zeros((CELL_SIZE, CELL_SIZE), dtype=int)
    for rr in range(3):
        for cc in range(3):
            if (rr, cc) in mask_on:
                cell[rr * 2 : rr * 2 + 2, cc * 2 : cc * 2 + 2] = ORANGE
    return cell


def _choose_target_counts() -> tuple[list[int], Counter[int]]:
    color_count = random.choice((2, 3))
    colors = random.sample(list(DATA_COLORS), color_count)
    min_each = 8 if color_count == 2 else 4
    counts = _random_composition(CELL_SIZE * CELL_SIZE, color_count, min_each)
    return colors, _build_counter(colors, counts)


def _make_disruptive_counts(target: Counter[int], colors: list[int]) -> Counter[int]:
    counts = target.copy()
    variant = "two_missing_one_surplus"
    if len(colors) == 3 and random.random() < 0.5:
        variant = "split_missing_one_surplus"

    if variant == "two_missing_one_surplus":
        miss = random.choice(colors)
        add = random.choice([color for color in colors if color != miss])
        assert counts[miss] >= 2
        counts[miss] -= 2
        counts[add] += 1
        return counts

    miss_a, miss_b = random.sample(colors, 2)
    add = random.choice([color for color in colors if color not in (miss_a, miss_b)])
    counts[miss_a] -= 1
    counts[miss_b] -= 1
    counts[add] += 1
    return counts


def _assemble_grid(cells: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), GREY, dtype=int)
    for rr in range(3):
        for cc in range(3):
            r0 = rr * (CELL_SIZE + SEPARATOR)
            c0 = cc * (CELL_SIZE + SEPARATOR)
            grid[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE] = cells[(rr, cc)]
    return grid


def _compatibility_positions(grid: np.ndarray, target: Counter[int]) -> set[tuple[int, int]]:
    cells = _extract_cells(grid)
    compatible = set()
    for pos, cell in cells.items():
        if pos == (0, 0):
            continue
        deficit = _deficit_for_target(_color_counts(cell), target)
        if sum(deficit.values()) == 1 and len(deficit) == 1:
            compatible.add(pos)
    return compatible


def _validate_instance(
    inp: np.ndarray,
    expected: np.ndarray,
    selected: set[tuple[int, int]],
    decoys: set[tuple[int, int]],
    disruptive: set[tuple[int, int]],
    selected_target: Counter[int],
) -> bool:
    if inp.shape != (GRID_SIZE, GRID_SIZE) or expected.shape != (GRID_SIZE, GRID_SIZE):
        return False

    cells = _extract_cells(inp)
    mask = _decode_instruction_mask(cells[(0, 0)])
    decoded_selected = {(r, c) for r in range(3) for c in range(3) if mask[r, c] and (r, c) != (0, 0)}
    if decoded_selected != selected:
        return False
    if len(decoded_selected) < 3 or not decoys or not disruptive:
        return False

    compatible = _compatibility_positions(inp, selected_target)
    if compatible != selected | decoys:
        return False

    all_candidates = [pos for pos in cells if pos != (0, 0)]
    if _completion_info(cells, all_candidates) is not None:
        return False
    if _completion_info(cells, sorted(selected)) is None:
        return False

    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    changed = np.argwhere(inp != expected)
    allowed_changes = set()
    for pos in selected:
        rr = pos[0] * (CELL_SIZE + SEPARATOR)
        cc = pos[1] * (CELL_SIZE + SEPARATOR)
        local_r, local_c = _hole_position(cells[pos])
        allowed_changes.add((rr + local_r, cc + local_c))
    if {tuple(map(int, rc)) for rc in changed} != allowed_changes:
        return False

    fill_colors = []
    for pos in selected:
        rr = pos[0] * (CELL_SIZE + SEPARATOR)
        cc = pos[1] * (CELL_SIZE + SEPARATOR)
        local_r, local_c = _hole_position(cells[pos])
        if inp[rr + local_r, cc + local_c] != BLACK:
            return False
        if expected[rr + local_r, cc + local_c] == BLACK:
            return False
        fill_colors.append(int(expected[rr + local_r, cc + local_c]))
    if len(set(fill_colors)) < 2:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    candidate_positions = [(r, c) for r in range(3) for c in range(3) if (r, c) != (0, 0)]
    selected_count = random.randint(3, 6)
    selected = set(random.sample(candidate_positions, selected_count))
    unselected = [pos for pos in candidate_positions if pos not in selected]

    decoy_count = random.randint(1, len(unselected) - 1)
    decoys = set(random.sample(unselected, decoy_count))
    disruptive = set(pos for pos in unselected if pos not in decoys)

    colors, target = _choose_target_counts()

    selected_missing = [random.choice(colors) for _ in range(selected_count)]
    if len(set(selected_missing)) < 2:
        alt = random.choice([color for color in colors if color != selected_missing[0]])
        selected_missing[-1] = alt

    cells: dict[tuple[int, int], np.ndarray] = {}
    mask_on = set(selected)
    if random.random() < 0.5:
        mask_on.add((0, 0))
    cells[(0, 0)] = _render_instruction_cell(mask_on)

    expected_cells: dict[tuple[int, int], np.ndarray] = {(0, 0): cells[(0, 0)].copy()}
    selected_target = target.copy()

    selected_list = sorted(selected)
    for pos, missing_color in zip(selected_list, selected_missing):
        counts = target.copy()
        counts[missing_color] -= 1
        hole = (random.randrange(CELL_SIZE), random.randrange(CELL_SIZE))
        cell = _render_data_cell(counts, hole)
        out_cell = cell.copy()
        out_cell[hole] = int(missing_color)
        cells[pos] = cell
        expected_cells[pos] = out_cell

    for pos in sorted(decoys):
        missing_color = random.choice(colors)
        counts = target.copy()
        counts[missing_color] -= 1
        hole = (random.randrange(CELL_SIZE), random.randrange(CELL_SIZE))
        cell = _render_data_cell(counts, hole)
        cells[pos] = cell
        expected_cells[pos] = cell.copy()

    for pos in sorted(disruptive):
        counts = _make_disruptive_counts(target, colors)
        if any(value < 0 for value in counts.values()) or sum(counts.values()) != CELL_SIZE * CELL_SIZE - 1:
            return None
        hole = (random.randrange(CELL_SIZE), random.randrange(CELL_SIZE))
        cell = _render_data_cell(counts, hole)
        cells[pos] = cell
        expected_cells[pos] = cell.copy()

    inp = _assemble_grid(cells)
    out = _assemble_grid(expected_cells)
    if not _validate_instance(inp, out, selected, decoys, disruptive, selected_target):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid masked multiset-balancing ARC input")
