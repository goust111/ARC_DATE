from __future__ import annotations

import random

import numpy as np


GREEN = 3
SEPARATOR_COLORS = (0, 5, 8)
MATRIX_COLORS = (1, 2, 4, 6, 7, 9)
MAX_GENERATION_ATTEMPTS = 400


def _transpose(r: int, c: int, n: int) -> tuple[int, int]:
    return c, r


def _anti_transpose(r: int, c: int, n: int) -> tuple[int, int]:
    return n - 1 - c, n - 1 - r


def _rot180(r: int, c: int, n: int) -> tuple[int, int]:
    return n - 1 - r, n - 1 - c


def _mirror_h(r: int, c: int, n: int) -> tuple[int, int]:
    return r, n - 1 - c


def _mirror_v(r: int, c: int, n: int) -> tuple[int, int]:
    return n - 1 - r, c


SYMMETRIES = (
    ("transpose", _transpose),
    ("anti_transpose", _anti_transpose),
    ("rot180", _rot180),
    ("mirror_h", _mirror_h),
    ("mirror_v", _mirror_v),
)

GENERATOR_SYMMETRIES = ("transpose", "anti_transpose", "rot180")


def _extract_hidden_matrix(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, int, int, int, np.ndarray]] = []

    for row_step in range(2, 7):
        if h % row_step != 0:
            continue
        for col_step in range(2, 7):
            if w % col_step != 0:
                continue
            n_rows = h // row_step
            n_cols = w // col_step
            if n_rows != n_cols or n_rows < 2:
                continue

            variable: list[tuple[int, int, np.ndarray]] = []
            for rr in range(row_step):
                for cc in range(col_step):
                    patch = g[rr::row_step, cc::col_step]
                    if np.unique(patch).size > 1:
                        variable.append((rr, cc, patch))

            if len(variable) != 1:
                continue

            rr, cc, matrix = variable[0]
            if GREEN not in matrix:
                continue
            if np.unique(matrix).size < 2:
                continue
            candidates.append((row_step, col_step, rr, cc, matrix.copy()))

    assert len(candidates) == 1, "expected exactly one editable lattice"
    row_step, col_step, rr, cc, matrix = candidates[0]
    return matrix, (row_step, col_step), (rr, cc)


def _valid_symmetry(matrix: np.ndarray, transform) -> bool:
    n = matrix.shape[0]
    saw_known_pair = False
    for r in range(n):
        for c in range(n):
            rr, cc = transform(r, c, n)
            a = int(matrix[r, c])
            b = int(matrix[rr, cc])
            if a == GREEN and b == GREEN:
                continue
            if a == GREEN or b == GREEN:
                continue
            saw_known_pair = True
            if a != b:
                return False
    return saw_known_pair


def _detect_symmetry(matrix: np.ndarray) -> tuple[str, callable]:
    matches = [(name, transform) for name, transform in SYMMETRIES if _valid_symmetry(matrix, transform)]
    assert len(matches) == 1, "hidden matrix symmetry must be unique"
    return matches[0]


def _repair_matrix(matrix: np.ndarray, transform) -> np.ndarray:
    out = np.array(matrix, copy=True)
    n = out.shape[0]
    for r, c in np.argwhere(out == GREEN):
        rr, cc = transform(int(r), int(c), n)
        value = int(out[rr, cc])
        assert value != GREEN, "damaged cell must have a visible symmetric partner"
        out[int(r), int(c)] = value
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Repair green placeholders inside a hidden lattice matrix by decoding its abstract symmetry.

    Input structure:
    - The visible grid is a repeated separator texture with one editable residue class in a fixed row/column lattice.
    - The editable cells form a hidden square color matrix, and some of those cells are green placeholders.
    - Ignoring green placeholders, the hidden matrix satisfies exactly one abstract symmetry such as transpose or 180-degree rotation.

    Rule:
    - Infer the lattice period and extract the hidden square matrix from the unique non-constant residue class.
    - Determine which symmetry is uniquely consistent with all non-green cells in that hidden matrix.
    - Replace every green cell by the color at its symmetric counterpart and write the repaired matrix back into the original lattice positions.

    Key insight:
    - The symmetry lives only in the decoded sparse matrix, not in the pixel-level canvas, because the separator texture destroys any direct visible symmetry.
    """
    g = np.asarray(grid, dtype=int)
    matrix, (row_step, col_step), (row_offset, col_offset) = _extract_hidden_matrix(g)
    _name, transform = _detect_symmetry(matrix)
    repaired = _repair_matrix(matrix, transform)

    out = np.array(g, copy=True)
    out[row_offset::row_step, col_offset::col_step] = repaired
    return out


def _orbit_cells(n: int, transform) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    orbits: list[list[tuple[int, int]]] = []
    for r in range(n):
        for c in range(n):
            if (r, c) in seen:
                continue
            rr, cc = transform(r, c, n)
            orbit = sorted({(r, c), (rr, cc)})
            for cell in orbit:
                seen.add(cell)
            orbits.append(orbit)
    return orbits


def _build_symmetric_matrix(n: int, transform) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=int)
    for orbit in _orbit_cells(n, transform):
        color = random.choice(MATRIX_COLORS)
        for r, c in orbit:
            matrix[r, c] = color
    return matrix


def _is_unique_full_symmetry(matrix: np.ndarray, chosen_name: str) -> bool:
    matches = []
    for name, transform in SYMMETRIES:
        ok = True
        n = matrix.shape[0]
        for r in range(n):
            for c in range(n):
                rr, cc = transform(r, c, n)
                if int(matrix[r, c]) != int(matrix[rr, cc]):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            matches.append(name)
    return matches == [chosen_name]


def _damage_matrix(matrix: np.ndarray, transform) -> np.ndarray | None:
    out = np.array(matrix, copy=True)
    non_fixed_orbits = [orbit for orbit in _orbit_cells(matrix.shape[0], transform) if len(orbit) > 1]
    if not non_fixed_orbits:
        return None

    damage_count = random.randint(1, min(len(non_fixed_orbits), max(2, matrix.shape[0])))
    chosen_orbits = random.sample(non_fixed_orbits, damage_count)
    for orbit in chosen_orbits:
        r, c = random.choice(orbit)
        out[r, c] = GREEN
    return out


def _random_texture(row_step: int, col_step: int, anchor: tuple[int, int]) -> np.ndarray:
    tile = np.empty((row_step, col_step), dtype=int)
    for r in range(row_step):
        for c in range(col_step):
            if (r, c) == anchor:
                tile[r, c] = SEPARATOR_COLORS[0]
            else:
                choices = [color for color in SEPARATOR_COLORS if not (r > 0 and tile[r - 1, c] == color)]
                choices = [color for color in choices if not (c > 0 and tile[r, c - 1] == color)]
                tile[r, c] = random.choice(choices or list(SEPARATOR_COLORS))
    return tile


def _render_grid(matrix: np.ndarray, row_step: int, col_step: int, anchor: tuple[int, int], tile: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    grid = np.tile(tile, (n, n))
    ar, ac = anchor
    grid[ar::row_step, ac::col_step] = matrix
    return grid


def _generate_attempt() -> np.ndarray | None:
    symmetry_name = random.choice(GENERATOR_SYMMETRIES)
    transform = dict(SYMMETRIES)[symmetry_name]
    n = random.randint(3, 6)
    row_step = random.randint(3, min(5, 30 // n))
    col_step = random.randint(3, min(5, 30 // n))
    anchor = (random.randrange(row_step), random.randrange(col_step))

    for _ in range(80):
        matrix = _build_symmetric_matrix(n, transform)
        if np.unique(matrix).size < 3:
            continue
        if not _is_unique_full_symmetry(matrix, symmetry_name):
            continue

        damaged = _damage_matrix(matrix, transform)
        if damaged is None:
            continue

        tile = _random_texture(row_step, col_step, anchor)
        grid = _render_grid(damaged, row_step, col_step, anchor, tile)

        try:
            repaired = main(grid)
        except Exception:
            continue
        expected = _render_grid(matrix, row_step, col_step, anchor, tile)
        if not np.array_equal(repaired, expected):
            continue
        detected_name, _ = _detect_symmetry(damaged)
        if detected_name != symmetry_name:
            continue
        if np.array_equal(grid, repaired):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-matrix symmetry task")
