from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import Color, colors_present


SEPARATOR_COLOR = Color.GREY
BLACK = Color.BLACK
PAIRABLE_COLORS = [
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]
MAX_GENERATION_ATTEMPTS = 400


def _detect_separator_color(grid: np.ndarray) -> int:
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
    candidates = (row_colors & col_colors) - {BLACK}
    assert len(candidates) == 1, "expected exactly one full-row/full-col separator color"
    return candidates.pop()


def _bands_from_separators(indices: list[int], limit: int) -> list[tuple[int, int]]:
    assert indices and indices[0] == 0 and indices[-1] == limit - 1
    assert all(b > a for a, b in zip(indices, indices[1:]))
    bands = [(a + 1, b) for a, b in zip(indices, indices[1:]) if b - a > 1]
    assert bands, "expected at least one non-separator band"
    return bands


def _extract_cell_matrix(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], np.ndarray]:
    sep = _detect_separator_color(grid)
    row_seps = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == sep)]
    col_seps = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == sep)]
    row_bands = _bands_from_separators(row_seps, grid.shape[0])
    col_bands = _bands_from_separators(col_seps, grid.shape[1])

    cells = np.empty((len(row_bands), len(col_bands)), dtype=int)
    for r_idx, (r0, r1) in enumerate(row_bands):
        for c_idx, (c0, c1) in enumerate(col_bands):
            patch = grid[r0:r1, c0:c1]
            colors = np.unique(patch)
            assert len(colors) == 1, "each compressed cell must be uniformly filled"
            color = int(colors[0])
            assert color != sep, "separator color must not appear inside a cell"
            cells[r_idx, c_idx] = color
    return sep, row_bands, col_bands, cells


def _infer_pair_mapping(cells: np.ndarray) -> dict[int, int]:
    rows, cols = map(int, cells.shape)
    mapping: dict[int, int] = {}
    observed_pairs = 0

    for r in range(rows):
        for c in range(cols):
            twin = (rows - 1 - r, cols - 1 - c)
            if (r, c) > twin:
                continue
            if (r, c) == twin:
                assert int(cells[r, c]) == BLACK, "self-symmetric center cell must stay black"
                continue

            a = int(cells[r, c])
            b = int(cells[twin])
            if a == BLACK or b == BLACK:
                continue

            assert a != b, "paired symmetric cells must swap to a different color"
            if a in mapping:
                assert mapping[a] == b, "inconsistent observed color mapping"
            if b in mapping:
                assert mapping[b] == a, "inconsistent observed color mapping"
            mapping[a] = b
            mapping[b] = a
            observed_pairs += 1

    assert observed_pairs > 0, "need at least one intact colored symmetric pair"
    for color in colors_present(cells, exclude={BLACK}):
        assert color in mapping, "every visible non-black color must belong to the inferred pairing table"
    return mapping


def _repair_cells(cells: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    rows, cols = map(int, cells.shape)
    out = cells.copy()
    for r in range(rows):
        for c in range(cols):
            if int(cells[r, c]) != BLACK:
                continue
            rr = rows - 1 - r
            cc = cols - 1 - c
            twin = int(cells[rr, cc])
            if twin != BLACK:
                assert twin in mapping, "repair color must be inferable from intact pairs"
                out[r, c] = mapping[twin]
    return out


def _render_cells(
    template: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    cells: np.ndarray,
) -> np.ndarray:
    out = template.copy()
    for r_idx, (r0, r1) in enumerate(row_bands):
        for c_idx, (c0, c1) in enumerate(col_bands):
            out[r0:r1, c0:c1] = int(cells[r_idx, c_idx])
    return out


def _cell_orbits(rows: int, cols: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    orbits = []
    for r in range(rows):
        for c in range(cols):
            twin = (rows - 1 - r, cols - 1 - c)
            if (r, c) <= twin:
                orbits.append(((r, c), twin))
    return orbits


def _sample_axis_bands(count: int) -> list[int]:
    for _ in range(200):
        widths = [random.randint(1, 4) for _ in range(count)]
        if len(set(widths)) == 1:
            continue
        if widths == widths[::-1]:
            continue
        total = sum(widths) + count + 1
        if total <= 30:
            return widths
    raise RuntimeError("failed to sample a non-palindromic axis layout")


def _build_canvas(
    cells: np.ndarray,
    row_heights: list[int],
    col_widths: list[int],
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    height = sum(row_heights) + len(row_heights) + 1
    width = sum(col_widths) + len(col_widths) + 1
    grid = np.full((height, width), SEPARATOR_COLOR, dtype=int)

    row_bands: list[tuple[int, int]] = []
    col_bands: list[tuple[int, int]] = []

    cursor = 1
    for span in row_heights:
        row_bands.append((cursor, cursor + span))
        cursor += span + 1

    cursor = 1
    for span in col_widths:
        col_bands.append((cursor, cursor + span))
        cursor += span + 1

    for r_idx, (r0, r1) in enumerate(row_bands):
        for c_idx, (c0, c1) in enumerate(col_bands):
            grid[r0:r1, c0:c1] = int(cells[r_idx, c_idx])
    return grid, row_bands, col_bands


def _assign_colored_orbit(
    cells: np.ndarray,
    orbit: tuple[tuple[int, int], tuple[int, int]],
    pair: tuple[int, int],
) -> None:
    (r0, c0), (r1, c1) = orbit
    a, b = pair
    if random.random() < 0.5:
        a, b = b, a
    cells[r0, c0] = int(a)
    cells[r1, c1] = int(b)


def _mask_orbit(
    cells: np.ndarray,
    orbit: tuple[tuple[int, int], tuple[int, int]],
    mode: str,
) -> None:
    (r0, c0), (r1, c1) = orbit
    if mode == "single_first":
        cells[r0, c0] = BLACK
    elif mode == "single_second":
        cells[r1, c1] = BLACK
    elif mode == "double":
        cells[r0, c0] = BLACK
        cells[r1, c1] = BLACK


def _mapping_from_pairs(pairs: list[tuple[int, int]]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for a, b in pairs:
        mapping[int(a)] = int(b)
        mapping[int(b)] = int(a)
    return mapping


def _generate_case() -> dict[str, np.ndarray]:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rows = random.randint(3, 5)
        cols = random.randint(3, 5)
        row_heights = _sample_axis_bands(rows)
        col_widths = _sample_axis_bands(cols)

        all_orbits = _cell_orbits(rows, cols)
        pair_orbits = [orbit for orbit in all_orbits if orbit[0] != orbit[1]]
        max_pair_count = min(3, len(pair_orbits) // 2)
        if max_pair_count < 2:
            continue

        pair_count = random.randint(2, max_pair_count)
        pair_colors = random.sample(PAIRABLE_COLORS, 2 * pair_count)
        color_pairs = [
            (pair_colors[2 * i], pair_colors[2 * i + 1])
            for i in range(pair_count)
        ]
        mapping = _mapping_from_pairs(color_pairs)

        random.shuffle(pair_orbits)
        exemplar_orbits = pair_orbits[:pair_count]
        work_orbits = pair_orbits[pair_count:2 * pair_count]
        remaining_orbits = pair_orbits[2 * pair_count:]

        full_cells = np.full((rows, cols), BLACK, dtype=int)
        for orbit, pair in zip(exemplar_orbits, color_pairs):
            _assign_colored_orbit(full_cells, orbit, pair)
        for orbit, pair in zip(work_orbits, color_pairs):
            _assign_colored_orbit(full_cells, orbit, pair)

        for orbit in remaining_orbits:
            if random.random() < 0.65:
                _assign_colored_orbit(full_cells, orbit, random.choice(color_pairs))

        input_cells = full_cells.copy()
        forced_single = random.randrange(len(work_orbits))
        for idx, orbit in enumerate(work_orbits):
            if idx == forced_single:
                mode = random.choice(("single_first", "single_second"))
            else:
                mode = random.choices(
                    ("intact", "single_first", "single_second", "double"),
                    weights=(2, 3, 3, 2),
                    k=1,
                )[0]
            _mask_orbit(input_cells, orbit, mode)

        for orbit in remaining_orbits:
            (r0, c0), (r1, c1) = orbit
            if int(full_cells[r0, c0]) == BLACK or int(full_cells[r1, c1]) == BLACK:
                continue
            mode = random.choices(
                ("intact", "single_first", "single_second", "double"),
                weights=(6, 2, 2, 2),
                k=1,
            )[0]
            _mask_orbit(input_cells, orbit, mode)

        expected_cells = _repair_cells(input_cells, mapping)
        inp, row_bands, col_bands = _build_canvas(input_cells, row_heights, col_widths)
        expected = _render_cells(inp, row_bands, col_bands, expected_cells)

        recoverable = 0
        unresolved = 0
        for r in range(rows):
            for c in range(cols):
                if int(input_cells[r, c]) != BLACK:
                    continue
                rr = rows - 1 - r
                cc = cols - 1 - c
                if int(input_cells[rr, cc]) == BLACK:
                    unresolved += 1
                else:
                    recoverable += 1
        if recoverable == 0:
            continue

        if not np.array_equal(main(inp), expected):
            continue

        return {
            "input": inp,
            "expected": expected,
            "input_cells": input_cells,
            "expected_cells": expected_cells,
            "full_cells": full_cells,
            "recoverable": np.array([[recoverable]], dtype=int),
            "unresolved": np.array([[unresolved]], dtype=int),
        }

    raise RuntimeError("failed to generate a valid puzzle instance")


def main(grid: np.ndarray) -> np.ndarray:
    """Restore blacked-out lattice cells using center-symmetric color-swap pairs after compressing irregular separators.

    Input structure:
    - Full-width and full-height separator lines partition the canvas into rectangular cells with unequal row-band heights and column-band widths.
    - Every cell is uniformly filled, either black or one visible foreground color, while separator lines stay a single separator color.
    - In compressed cell coordinates, center-symmetric twin cells swap through one fixed bidirectional color-pair table rather than matching exactly; some cells are blacked out.

    Rule:
    - Detect the separator rows and columns and compress the image into a regular matrix of cell colors.
    - Infer the color-pair swap table from intact center-symmetric cell pairs where both twins are non-black.
    - For each black cell, inspect its center-symmetric twin: if the twin is colored, fill the black cell with that twin color's paired color; if the twin is also black, leave it black.

    Key insight:
    - Pixel-level symmetry is misleading because the separator spacing is irregular, and even cell-level equality fails because symmetry acts through a hidden color swap table that must first be induced from intact twin pairs.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    _sep, row_bands, col_bands, cells = _extract_cell_matrix(g)
    mapping = _infer_pair_mapping(cells)
    repaired = _repair_cells(cells, mapping)
    return _render_cells(g, row_bands, col_bands, repaired)


def generate_input() -> np.ndarray:
    return _generate_case()["input"]
