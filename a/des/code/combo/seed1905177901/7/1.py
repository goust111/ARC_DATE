from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND_COLOR = 0
SEPARATOR_COLOR = 5
MARKER_COLOR = 2
TEMPLATE_COUNT = 4
MAX_GRID_SIZE = 30
MAX_GENERATION_ATTEMPTS = 300
TEMPLATE_PALETTE = (1, 3, 4, 6, 7, 8, 9)


def _infer_geometry(grid: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Return (bg, cell_size, separator-row indices, separator-col indices)."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    constant_rows = [r for r in range(h) if np.all(g[r] == int(g[r, 0]))]
    constant_cols = [c for c in range(w) if np.all(g[:, c] == int(g[0, c]))]
    assert constant_rows and constant_cols, "expected full separator rows and columns"

    colors = [int(g[r, 0]) for r in constant_rows] + [int(g[0, c]) for c in constant_cols]
    sep = int(Counter(colors).most_common(1)[0][0])

    row_lines = np.array([r for r in constant_rows if int(g[r, 0]) == sep], dtype=int)
    col_lines = np.array([c for c in constant_cols if int(g[0, c]) == sep], dtype=int)
    assert len(row_lines) >= 3 and len(col_lines) >= 5, "need legend row and at least four columns"
    assert int(row_lines[0]) == 0 and int(row_lines[-1]) == h - 1
    assert int(col_lines[0]) == 0 and int(col_lines[-1]) == w - 1

    row_steps = np.diff(row_lines)
    col_steps = np.diff(col_lines)
    assert np.all(row_steps == row_steps[0]) and np.all(col_steps == col_steps[0]), "grid must be regular"
    cell_size = int(row_steps[0] - 1)
    assert cell_size >= 2 and int(col_steps[0] - 1) == cell_size, "cells must be square"

    bg_counts = Counter(int(v) for v in g.ravel() if int(v) != sep)
    assert bg_counts, "expected non-separator pixels"
    bg = int(bg_counts.most_common(1)[0][0])
    return bg, cell_size, row_lines, col_lines


def _cell_bounds(row_lines: np.ndarray, col_lines: np.ndarray, r: int, c: int) -> tuple[int, int, int, int]:
    return (
        int(row_lines[r] + 1),
        int(row_lines[r + 1]),
        int(col_lines[c] + 1),
        int(col_lines[c + 1]),
    )


def _extract_template(cell: np.ndarray, bg: int) -> np.ndarray | None:
    pts = np.argwhere(cell != bg)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    template = np.array(cell[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=int)
    assert template.shape[0] == template.shape[1], "legend template must be square"
    assert template.shape[0] in (2, 3), "legend template must be 2x2 or 3x3"
    return template


def _scale_template(template: np.ndarray, size: int) -> np.ndarray:
    rows = (np.arange(size) * template.shape[0]) // size
    cols = (np.arange(size) * template.shape[1]) // size
    return np.array(template[np.ix_(rows, cols)], dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count red markers in each lower cell, select the corresponding top-row legend template,
    and scale that template to fill the full cell while keeping the grey separators intact.
    """
    g = np.asarray(grid, dtype=int)
    bg, cell_size, row_lines, col_lines = _infer_geometry(g)
    n_rows = len(row_lines) - 1
    n_cols = len(col_lines) - 1

    templates: list[np.ndarray] = []
    for c in range(n_cols):
        r0, r1, c0, c1 = _cell_bounds(row_lines, col_lines, 0, c)
        template = _extract_template(g[r0:r1, c0:c1], bg)
        if template is not None:
            templates.append(template)

    assert templates, "expected at least one legend template"
    out = g.copy()

    for r in range(1, n_rows):
        for c in range(n_cols):
            r0, r1, c0, c1 = _cell_bounds(row_lines, col_lines, r, c)
            cell = g[r0:r1, c0:c1]
            k = int(np.count_nonzero(cell == MARKER_COLOR))
            if k == 0:
                continue
            assert 1 <= k <= len(templates), "marker count must index an existing template"
            out[r0:r1, c0:c1] = _scale_template(templates[k - 1], cell_size)

    return out


def _make_template(size: int, palette: tuple[int, ...]) -> np.ndarray:
    n_colors = random.randint(2, min(3, len(palette)))
    colors = random.sample(list(palette), n_colors)
    while True:
        arr = np.array([random.choice(colors) for _ in range(size * size)], dtype=int).reshape(size, size)
        if len(set(map(int, arr.ravel()))) >= 2:
            return arr


def _family_spec() -> tuple[int, list[int]]:
    family = random.choice(["compact-2", "expanded-2", "all-3", "mixed"])
    if family == "compact-2":
        return 4, [2, 2, 2, 2]
    if family == "expanded-2":
        return 6, [2, 2, 2, 2]
    if family == "all-3":
        return 6, [3, 3, 3, 3]
    sizes = [random.choice([2, 3]) for _ in range(TEMPLATE_COUNT)]
    if len(set(sizes)) == 1:
        sizes[0] = 2
        sizes[1] = 3
    return 6, sizes


def _grid_shape(cell_size: int, target_rows: int, total_cols: int) -> tuple[int, int, np.ndarray, np.ndarray]:
    total_rows = target_rows + 1
    h = total_rows * cell_size + total_rows + 1
    w = total_cols * cell_size + total_cols + 1
    row_lines = np.arange(0, h, cell_size + 1, dtype=int)
    col_lines = np.arange(0, w, cell_size + 1, dtype=int)
    return h, w, row_lines, col_lines


def _ensure_count_coverage(target_rows: int, total_cols: int) -> np.ndarray:
    total_cells = target_rows * total_cols
    counts = [random.randint(1, TEMPLATE_COUNT) for _ in range(total_cells)]
    for idx, value in enumerate([1, 2, 3, 4]):
        counts[idx] = value
    random.shuffle(counts)
    return np.array(counts, dtype=int).reshape(target_rows, total_cols)


def _place_templates(
    grid: np.ndarray,
    row_lines: np.ndarray,
    col_lines: np.ndarray,
    cell_size: int,
    templates: list[np.ndarray],
    legend_cols: list[int],
) -> None:
    for template, col in zip(templates, legend_cols):
        r0, r1, c0, c1 = _cell_bounds(row_lines, col_lines, 0, col)
        size = template.shape[0]
        off_r = random.randint(0, cell_size - size)
        off_c = random.randint(0, cell_size - size)
        grid[r0 + off_r : r0 + off_r + size, c0 + off_c : c0 + off_c + size] = template


def _place_markers(
    grid: np.ndarray,
    row_lines: np.ndarray,
    col_lines: np.ndarray,
    counts: np.ndarray,
    cell_size: int,
) -> None:
    for rr in range(counts.shape[0]):
        for cc in range(counts.shape[1]):
            r0, r1, c0, c1 = _cell_bounds(row_lines, col_lines, rr + 1, cc)
            picks = random.sample(range(cell_size * cell_size), int(counts[rr, cc]))
            for idx in picks:
                pr = idx // cell_size
                pc = idx % cell_size
                grid[r0 + pr, c0 + pc] = MARKER_COLOR


def _expected_output(
    inp: np.ndarray,
    row_lines: np.ndarray,
    col_lines: np.ndarray,
    counts: np.ndarray,
    cell_size: int,
    templates: list[np.ndarray],
) -> np.ndarray:
    out = inp.copy()
    for rr in range(counts.shape[0]):
        for cc in range(counts.shape[1]):
            r0, r1, c0, c1 = _cell_bounds(row_lines, col_lines, rr + 1, cc)
            out[r0:r1, c0:c1] = _scale_template(templates[int(counts[rr, cc]) - 1], cell_size)
    return out


def _gen_attempt() -> np.ndarray | None:
    cell_size, template_sizes = _family_spec()
    max_cells = (MAX_GRID_SIZE - 1) // (cell_size + 1)
    total_cols = random.randint(TEMPLATE_COUNT, max_cells)
    max_target_rows = max_cells - 1
    if max_target_rows < 2:
        return None
    target_rows = random.randint(2, max_target_rows)

    h, w, row_lines, col_lines = _grid_shape(cell_size, target_rows, total_cols)
    if h > MAX_GRID_SIZE or w > MAX_GRID_SIZE:
        return None

    palette = tuple(random.sample(TEMPLATE_PALETTE, random.randint(3, min(5, len(TEMPLATE_PALETTE)))))
    templates: list[np.ndarray] = []
    seen: set[tuple[int, tuple[int, ...]]] = set()
    for size in template_sizes:
        for _ in range(50):
            template = _make_template(size, palette)
            key = (size, tuple(int(v) for v in template.ravel()))
            if key not in seen:
                seen.add(key)
                templates.append(template)
                break
        else:
            return None

    legend_cols = sorted(random.sample(range(total_cols), TEMPLATE_COUNT))
    counts = _ensure_count_coverage(target_rows, total_cols)

    grid = np.full((h, w), BACKGROUND_COLOR, dtype=int)
    grid[row_lines, :] = SEPARATOR_COLOR
    grid[:, col_lines] = SEPARATOR_COLOR
    _place_templates(grid, row_lines, col_lines, cell_size, templates, legend_cols)
    _place_markers(grid, row_lines, col_lines, counts, cell_size)

    expected = _expected_output(grid, row_lines, col_lines, counts, cell_size, templates)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
