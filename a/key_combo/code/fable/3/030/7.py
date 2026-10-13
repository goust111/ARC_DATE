from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
GREY = 5
DIFF_DIAGONAL = "diff"  # r - c is constant: down-right diagonal
SUM_DIAGONAL = "sum"  # r + c is constant: down-left diagonal
LINE_KINDS = (DIFF_DIAGONAL, SUM_DIAGONAL)
DRAW_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATION_ATTEMPTS = 1000


def _line_cells(shape: tuple[int, int], kind: str, value: int) -> tuple[tuple[int, int], ...]:
    height, width = map(int, shape)
    cells: list[tuple[int, int]] = []
    if kind == DIFF_DIAGONAL:
        for r in range(height):
            c = r - int(value)
            if 0 <= c < width:
                cells.append((r, c))
    elif kind == SUM_DIAGONAL:
        for r in range(height):
            c = int(value) - r
            if 0 <= c < width:
                cells.append((r, c))
    else:
        raise ValueError("unknown diagonal kind")
    return tuple(cells)


def _infer_line(coords: np.ndarray) -> tuple[str, int]:
    assert coords.ndim == 2 and coords.shape[1] == 2, "expected an array of (r, c) coords"
    assert coords.shape[0] >= 3, "each color must have at least three residue pixels"

    diffs = coords[:, 0] - coords[:, 1]
    sums = coords[:, 0] + coords[:, 1]
    same_diff = bool(np.all(diffs == diffs[0]))
    same_sum = bool(np.all(sums == sums[0]))
    assert same_diff ^ same_sum, "residue pixels must determine one 45-degree diagonal"

    if same_diff:
        return DIFF_DIAGONAL, int(diffs[0])
    return SUM_DIAGONAL, int(sums[0])


def _detect_window_rows(g: np.ndarray) -> tuple[int, ...]:
    rows = tuple(int(r) for r in np.flatnonzero(np.any(g == GREY, axis=1)))
    assert rows, "expected at least one gray window row"

    window_set = set(rows)
    for r in range(g.shape[0]):
        if r in window_set:
            assert np.all(g[r, :] != BLACK), "a window row must be a full gray stripe with colored residues"
        else:
            assert np.all(g[r, :] == BLACK), "colored residues may appear only on gray window rows"
    return rows


def _line_owners(
    shape: tuple[int, int],
    line_specs: list[tuple[int, str, int]],
) -> dict[tuple[int, int], list[int]]:
    owners: dict[tuple[int, int], list[int]] = defaultdict(list)
    for color, kind, value in line_specs:
        cells = _line_cells(shape, kind, value)
        assert len(cells) >= 3, "a reconstructed diagonal must span at least three cells"
        for cell in cells:
            owners[cell].append(int(color))
    return owners


def _render_from_lines(shape: tuple[int, int], line_specs: list[tuple[int, str, int]]) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    for (r, c), colors in _line_owners(shape, line_specs).items():
        out[r, c] = GREY if len(colors) > 1 else int(colors[0])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild full 45-degree diagonals from colored residues sampled on gray window rows, with crossings gray.

    Input structure:
    - The grid has a black background and several full gray horizontal window rows.
    - Non-gray colored pixels occur only on those gray rows, replacing gray cells as line residues.
    - Each non-gray residue color appears at least three times and its pixels lie on exactly one 45-degree diagonal.

    Rule:
    - Erase every gray window row back to black.
    - For each residue color, infer the unique 45-degree diagonal through all of its pixels and extend it to the grid boundary.
    - If two or more different reconstructed diagonals cover the same cell, color that cell gray instead of any line color.

    Key insight:
    - The gray rows are sampling windows, not separators; the sparse colored samples reveal hidden whole diagonals, and line crossings take the window color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30x30"

    window_rows = _detect_window_rows(g)
    window_set = set(window_rows)
    colors = sorted(int(c) for c in np.unique(g) if int(c) not in (BLACK, GREY))
    assert colors, "expected at least one residue color"

    line_specs: list[tuple[int, str, int]] = []
    seen_lines: set[tuple[str, int]] = set()
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.shape[0] >= 3, "each residue color needs at least three samples"
        assert all(int(r) in window_set for r, _c in coords), "residues must lie on gray window rows"

        kind, value = _infer_line(coords)
        key = (kind, value)
        assert key not in seen_lines, "different colors must imply different diagonals"
        seen_lines.add(key)

        cell_set = set(_line_cells(g.shape, kind, value))
        assert all((int(r), int(c)) in cell_set for r, c in coords), "all residues must sit on the inferred diagonal"
        line_specs.append((color, kind, value))

    return _render_from_lines(g.shape, line_specs)


def _candidate_lines(shape: tuple[int, int], kind: str, window_rows: tuple[int, ...]) -> list[tuple[str, int]]:
    height, width = map(int, shape)
    if kind == DIFF_DIAGONAL:
        values = range(-(width - 1), height)
    else:
        values = range(height + width - 1)

    candidates: list[tuple[str, int]] = []
    window_set = set(window_rows)
    for value in values:
        cells = _line_cells(shape, kind, int(value))
        visible = [cell for cell in cells if cell[0] in window_set]
        if len(cells) >= 6 and len(visible) >= 4:
            candidates.append((kind, int(value)))
    return candidates


def _compatible_with_chosen(
    key: tuple[str, int],
    chosen: list[tuple[str, int]],
) -> bool:
    kind, value = key
    for other_kind, other_value in chosen:
        if kind == other_kind and abs(value - other_value) < 3:
            return False
        if kind == other_kind and value == other_value:
            return False
    return True


def _intersection(
    shape: tuple[int, int],
    first: tuple[str, int],
    second: tuple[str, int],
) -> tuple[int, int] | None:
    if first[0] == second[0]:
        return None
    if first[0] == DIFF_DIAGONAL:
        diff_value, sum_value = first[1], second[1]
    else:
        diff_value, sum_value = second[1], first[1]

    total = sum_value + diff_value
    delta = sum_value - diff_value
    if total % 2 or delta % 2:
        return None

    r = total // 2
    c = delta // 2
    height, width = map(int, shape)
    if 0 <= r < height and 0 <= c < width:
        return int(r), int(c)
    return None


def _choose_line_keys(
    shape: tuple[int, int],
    window_rows: tuple[int, ...],
    line_count: int,
) -> list[tuple[str, int]] | None:
    kinds = [DIFF_DIAGONAL, SUM_DIAGONAL]
    kinds.extend(random.choice(LINE_KINDS) for _ in range(line_count - 2))
    random.shuffle(kinds)

    chosen: list[tuple[str, int]] = []
    for kind in kinds:
        candidates = [
            key
            for key in _candidate_lines(shape, kind, window_rows)
            if _compatible_with_chosen(key, chosen)
        ]
        if not candidates:
            return None
        chosen.append(random.choice(candidates))
    return chosen


def _owners_for_keys(
    shape: tuple[int, int],
    keys: list[tuple[str, int]],
) -> dict[tuple[int, int], list[int]]:
    owners: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, (kind, value) in enumerate(keys):
        for cell in _line_cells(shape, kind, value):
            owners[cell].append(idx)
    return owners


def _residue_cells_by_line(
    shape: tuple[int, int],
    keys: list[tuple[str, int]],
    window_rows: tuple[int, ...],
) -> list[list[tuple[int, int]]] | None:
    owners = _owners_for_keys(shape, keys)
    window_set = set(window_rows)
    residues: list[list[tuple[int, int]]] = []
    for idx, (kind, value) in enumerate(keys):
        cells = [
            cell
            for cell in _line_cells(shape, kind, value)
            if cell[0] in window_set and owners[cell] == [idx]
        ]
        if len(cells) < 3:
            return None
        residues.append(cells)
    return residues


def _crossing_cells(
    shape: tuple[int, int],
    keys: list[tuple[str, int]],
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            cell = _intersection(shape, keys[i], keys[j])
            if cell is not None:
                cells.add(cell)
    return cells


def _build_input(
    shape: tuple[int, int],
    window_rows: tuple[int, ...],
    keys: list[tuple[str, int]],
    colors: list[int],
    residues: list[list[tuple[int, int]]],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for r in window_rows:
        grid[int(r), :] = GREY
    for color, cells in zip(colors, residues):
        for r, c in cells:
            grid[r, c] = int(color)
    return grid


def _semantic_check(
    inp: np.ndarray,
    keys: list[tuple[str, int]],
    colors: list[int],
    window_rows: tuple[int, ...],
) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if len(set(colors)) != len(colors) or GREY in colors or BLACK in colors:
        return False
    if len({kind for kind, _value in keys}) != 2:
        return False

    detected_rows = _detect_window_rows(inp)
    if detected_rows != window_rows:
        return False

    line_specs = [(color, kind, value) for color, (kind, value) in zip(colors, keys)]
    expected = _render_from_lines(inp.shape, line_specs)
    crossings = _crossing_cells(inp.shape, keys)
    if not crossings:
        return False

    crossing_rows = {r for r, _c in crossings}
    window_set = set(window_rows)
    if not (crossing_rows & window_set) or not (crossing_rows - window_set):
        return False

    owners = _owners_for_keys(inp.shape, keys)
    true_crossings = {cell for cell, owner_list in owners.items() if len(owner_list) > 1}
    if true_crossings != crossings:
        return False
    if not all(expected[r, c] == GREY for r, c in true_crossings):
        return False

    for idx, color in enumerate(colors):
        coords = np.argwhere(inp == color)
        if coords.shape[0] < 3:
            return False
        if _infer_line(coords) != keys[idx]:
            return False
        line_cells = set(_line_cells(inp.shape, *keys[idx]))
        if not any(r not in window_set for r, c in line_cells if expected[r, c] == color):
            return False

    colored_rows = {int(r) for r, _c in np.argwhere((inp != BLACK) & (inp != GREY))}
    if not set(window_rows).issubset(colored_rows):
        return False
    if np.array_equal(inp, expected):
        return False

    try:
        got = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(got, expected))


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(13, 24)
    width = random.randint(13, 24)
    shape = (height, width)

    max_windows = min(8, height - 2)
    window_count = random.randint(4, max_windows)
    window_rows = tuple(sorted(random.sample(range(1, height - 1), window_count)))

    line_count = random.randint(3, 5)
    keys = _choose_line_keys(shape, window_rows, line_count)
    if keys is None:
        return None

    residues = _residue_cells_by_line(shape, keys, window_rows)
    if residues is None:
        return None

    colors = random.sample(DRAW_COLORS, line_count)
    inp = _build_input(shape, window_rows, keys, colors, residues)
    if not _semantic_check(inp, keys, colors, window_rows):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid striped-window diagonal input")
