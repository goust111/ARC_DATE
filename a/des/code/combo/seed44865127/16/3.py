from __future__ import annotations

import random

import numpy as np


SYMMETRY_BOTH = "both"
SYMMETRY_HORIZONTAL = "horizontal"
SYMMETRY_VERTICAL = "vertical"
SYMMETRY_NONE = "none"
SYMMETRY_KINDS = (
    SYMMETRY_BOTH,
    SYMMETRY_HORIZONTAL,
    SYMMETRY_VERTICAL,
    SYMMETRY_NONE,
)

GRID_ROWS_RANGE = (2, 4)
GRID_COLS_RANGE = (2, 4)
CELL_SIZE_RANGE = (5, 7)
MAX_PATTERN_ATTEMPTS = 300
MAX_GENERATION_ATTEMPTS = 300
PALETTE = tuple(range(10))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _regular_intervals(separators: list[int], length: int) -> list[tuple[int, int]]:
    assert separators, "expected at least one separator line"
    bounds = [-1, *sorted(separators), int(length)]
    intervals = [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]
    sizes = [end - start for start, end in intervals]
    assert min(sizes) > 0, "separators must leave non-empty cells"
    assert len(set(sizes)) == 1, "all cells must have the same size"
    return intervals


def _find_structure(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]], int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)

    row_candidates = [
        (r, int(g[r, 0]))
        for r in range(h)
        if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != bg
    ]
    col_candidates = [
        (c, int(g[0, c]))
        for c in range(w)
        if np.all(g[:, c] == g[0, c]) and int(g[0, c]) != bg
    ]

    sep_colors = {color for _idx, color in row_candidates} & {color for _idx, color in col_candidates}
    assert len(sep_colors) == 1, "expected one separator color shared by rows and columns"
    sep = next(iter(sep_colors))

    row_separators = [r for r, color in row_candidates if color == sep]
    col_separators = [c for c, color in col_candidates if color == sep]
    row_intervals = _regular_intervals(row_separators, h)
    col_intervals = _regular_intervals(col_separators, w)

    row_sizes = {r1 - r0 for r0, r1 in row_intervals}
    col_sizes = {c1 - c0 for c0, c1 in col_intervals}
    assert row_sizes == col_sizes, "cells must be square"
    cell_size = next(iter(row_sizes))
    return int(bg), int(sep), row_intervals, col_intervals, int(cell_size)


def _classify_symmetry(block: np.ndarray) -> str:
    horizontal = np.array_equal(block, np.flipud(block))
    vertical = np.array_equal(block, np.fliplr(block))
    if horizontal and vertical:
        return SYMMETRY_BOTH
    if horizontal:
        return SYMMETRY_HORIZONTAL
    if vertical:
        return SYMMETRY_VERTICAL
    return SYMMETRY_NONE


def _dominant_foreground_color(block: np.ndarray, bg: int) -> int:
    fg = block[block != bg]
    assert fg.size > 0, "each cell must contain some foreground"
    colors, counts = np.unique(fg, return_counts=True)
    max_count = int(counts.max())
    winners = colors[counts == max_count]
    assert winners.size == 1, "cell foreground must have a unique dominant color"
    return int(winners[0])


def _texture(size: int, kind: str, main_color: int, bg: int) -> np.ndarray:
    rr, cc = np.indices((size, size))
    if kind == SYMMETRY_BOTH:
        return np.full((size, size), int(main_color), dtype=int)
    if kind == SYMMETRY_HORIZONTAL:
        return np.where(rr % 2 == 0, int(main_color), int(bg)).astype(int)
    if kind == SYMMETRY_VERTICAL:
        return np.where(cc % 2 == 0, int(main_color), int(bg)).astype(int)
    return np.where((rr + cc) % 2 == 0, int(main_color), int(bg)).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each cell pattern with a macro texture determined by its reflection symmetry.

    Input structure:
    - The dominant color is the background.
    - Full separator rows and columns partition the grid into equal square cells.
    - Each cell contains a small multi-color pattern.

    Rule:
    - Classify each cell pattern by its exact horizontal/vertical reflection symmetry.
    - Both symmetries: fill the whole cell with the pattern's dominant foreground color.
    - Only horizontal symmetry: fill horizontal stripes.
    - Only vertical symmetry: fill vertical stripes.
    - No symmetry: fill a checkerboard.
    - Keep separator lines unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg, _sep, row_intervals, col_intervals, cell_size = _find_structure(g)

    out = g.copy()
    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            block = g[r0:r1, c0:c1]
            kind = _classify_symmetry(block)
            main_color = _dominant_foreground_color(block, bg)
            out[r0:r1, c0:c1] = _texture(cell_size, kind, main_color, bg)
    return out


def _symmetry_orbits(size: int, kind: str) -> list[list[tuple[int, int]]]:
    use_horizontal = kind in (SYMMETRY_BOTH, SYMMETRY_HORIZONTAL)
    use_vertical = kind in (SYMMETRY_BOTH, SYMMETRY_VERTICAL)
    seen: set[tuple[int, int]] = set()
    orbits: list[list[tuple[int, int]]] = []
    for r in range(size):
        for c in range(size):
            if (r, c) in seen:
                continue
            orbit = {(r, c)}
            if use_horizontal:
                orbit.add((size - 1 - r, c))
            if use_vertical:
                orbit.add((r, size - 1 - c))
            if use_horizontal and use_vertical:
                orbit.add((size - 1 - r, size - 1 - c))
            orbit_list = sorted(orbit)
            seen.update(orbit_list)
            orbits.append(orbit_list)
    return orbits


def _row_col_mixed(mask: np.ndarray) -> bool:
    size = mask.shape[0]
    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)
    return bool(
        np.all((row_counts > 0) & (row_counts < size))
        and np.all((col_counts > 0) & (col_counts < size))
    )


def _template_valid(template: np.ndarray, kind: str) -> bool:
    size = template.shape[0]
    area = size * size
    if _classify_symmetry(template) != kind:
        return False
    if not set(np.unique(template)).issubset({0, 1, 2}):
        return False

    mask = template != 0
    if not _row_col_mixed(mask):
        return False

    main_count = int(np.count_nonzero(template == 1))
    accent_count = int(np.count_nonzero(template == 2))
    total_fg = main_count + accent_count
    return bool(
        main_count > accent_count >= 1
        and total_fg >= size
        and total_fg <= area // 2
    )


def _make_template(size: int, kind: str) -> np.ndarray | None:
    orbits = _symmetry_orbits(size, kind)
    orbit_indices = list(range(len(orbits)))

    for _ in range(MAX_PATTERN_ATTEMPTS):
        fill_prob = random.uniform(0.35, 0.65)
        chosen = [idx for idx in orbit_indices if random.random() < fill_prob]
        if len(chosen) < 2:
            continue

        mask = np.zeros((size, size), dtype=bool)
        for idx in chosen:
            for r, c in orbits[idx]:
                mask[r, c] = True

        if _classify_symmetry(mask.astype(int)) != kind:
            continue
        if not _row_col_mixed(mask):
            continue

        for _accent_attempt in range(40):
            accent_orbit_count = random.randint(1, min(3, len(chosen) - 1))
            accent_orbits = set(random.sample(chosen, accent_orbit_count))
            template = np.zeros((size, size), dtype=int)
            for idx in chosen:
                value = 2 if idx in accent_orbits else 1
                for r, c in orbits[idx]:
                    template[r, c] = value
            if _template_valid(template, kind):
                return template
    return None


def _paint_template(template: np.ndarray, bg: int, main_color: int, accent_color: int) -> np.ndarray:
    out = np.full(template.shape, int(bg), dtype=int)
    out[template == 1] = int(main_color)
    out[template == 2] = int(accent_color)
    return out


def _balanced_kinds(count: int) -> list[str]:
    kinds = [SYMMETRY_KINDS[i % len(SYMMETRY_KINDS)] for i in range(count)]
    random.shuffle(kinds)
    return kinds


def _separator_indices(count: int, cell_size: int) -> list[int]:
    return [(i + 1) * cell_size + i for i in range(count - 1)]


def _cell_origin(cell_row: int, cell_col: int, cell_size: int) -> tuple[int, int]:
    return cell_row * (cell_size + 1), cell_col * (cell_size + 1)


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.randint(*GRID_ROWS_RANGE)
    n_cols = random.randint(*GRID_COLS_RANGE)
    cell_size = random.randint(*CELL_SIZE_RANGE)

    bg, separator = random.sample(PALETTE, 2)
    available_colors = [color for color in PALETTE if color not in {bg, separator}]
    if len(available_colors) < 2:
        return None

    height = n_rows * cell_size + (n_rows - 1)
    width = n_cols * cell_size + (n_cols - 1)
    grid = np.full((height, width), int(bg), dtype=int)
    expected = np.full((height, width), int(bg), dtype=int)

    row_separators = _separator_indices(n_rows, cell_size)
    col_separators = _separator_indices(n_cols, cell_size)
    for r in row_separators:
        grid[r, :] = int(separator)
        expected[r, :] = int(separator)
    for c in col_separators:
        grid[:, c] = int(separator)
        expected[:, c] = int(separator)

    kinds = _balanced_kinds(n_rows * n_cols)
    templates: dict[str, list[np.ndarray]] = {kind: [] for kind in SYMMETRY_KINDS}

    for kind in SYMMETRY_KINDS:
        template = _make_template(cell_size, kind)
        if template is None:
            return None
        templates[kind].append(template)

    for cell_index, kind in enumerate(kinds):
        if random.random() < 0.5:
            template = templates[kind][0]
        else:
            template = _make_template(cell_size, kind)
            if template is None:
                return None

        main_color, accent_color = random.sample(available_colors, 2)
        block = _paint_template(template, int(bg), int(main_color), int(accent_color))

        cell_row, cell_col = divmod(cell_index, n_cols)
        r0, c0 = _cell_origin(cell_row, cell_col, cell_size)
        grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = block
        expected[r0 : r0 + cell_size, c0 : c0 + cell_size] = _texture(cell_size, kind, int(main_color), int(bg))

    if len(set(kinds)) != len(SYMMETRY_KINDS):
        return None
    if np.array_equal(grid, expected):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-texture grid")
