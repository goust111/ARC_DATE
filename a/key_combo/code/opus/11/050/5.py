from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Iterable

import numpy as np


BG = 0
OUT_COLOR = 1
SHAPE_COLOR = 2
GRID_COLOR = 8
CELL_SIZE = 4
ROW_COUNT = 3
COL_COUNT = 4

_RNG = np.random.default_rng()


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _is_connected(coords: Iterable[tuple[int, int]]) -> bool:
    pts = set(coords)
    if not pts:
        return False
    stack = [next(iter(pts))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in pts and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(pts)


def _centroid_floor(coords: Iterable[tuple[int, int]]) -> tuple[int, int]:
    pts = list(coords)
    return (
        sum(r for r, _ in pts) // len(pts),
        sum(c for _, c in pts) // len(pts),
    )


def _normalize_shape(coords: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    pts = list(coords)
    if not pts:
        return ()
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _shape_to_array(coords: Iterable[tuple[int, int]]) -> np.ndarray:
    cell = np.full((CELL_SIZE, CELL_SIZE), BG, dtype=int)
    for r, c in coords:
        cell[r, c] = SHAPE_COLOR
    return cell


def _extract_red_coords(cell: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = tuple(sorted(map(tuple, np.argwhere(cell == SHAPE_COLOR))))
    assert 3 <= len(coords) <= 5, "each cell must contain a 3-5 pixel shape"
    assert _is_connected(coords), "each shape must be 4-connected"
    return coords


def _relative_to_centroid(coords: Iterable[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    pts = list(coords)
    cr, cc = _centroid_floor(pts)
    return frozenset((r - cr, c - cc) for r, c in pts)


def _render_output(common: Iterable[tuple[int, int]]) -> np.ndarray:
    pts = list(common)
    r0 = min(r for r, _ in pts)
    r1 = max(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    c1 = max(c for _, c in pts)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in pts:
        out[r - r0, c - c0] = OUT_COLOR
    return out


def _line_slices(line_indices: list[int]) -> list[tuple[int, int]]:
    return [(a + 1, b) for a, b in zip(line_indices, line_indices[1:]) if b - a > 1]


def _extract_cells(grid: np.ndarray) -> list[list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    row_lines = [r for r in range(g.shape[0]) if np.all(g[r, :] == GRID_COLOR)]
    col_lines = [c for c in range(g.shape[1]) if np.all(g[:, c] == GRID_COLOR)]
    row_slices = _line_slices(row_lines)
    col_slices = _line_slices(col_lines)
    assert len(row_slices) == ROW_COUNT and len(col_slices) == COL_COUNT, "unexpected lattice layout"
    assert all(rb - ra == CELL_SIZE for ra, rb in row_slices), "row cell size must be 4"
    assert all(cb - ca == CELL_SIZE for ca, cb in col_slices), "col cell size must be 4"
    cells: list[list[np.ndarray]] = []
    for ra, rb in row_slices:
        row = []
        for ca, cb in col_slices:
            cell = g[ra:rb, ca:cb]
            assert set(np.unique(cell)).issubset({BG, SHAPE_COLOR}), "cell contains unexpected colors"
            row.append(cell.copy())
        cells.append(row)
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the odd sprite from each lattice row, align the extracted sprites by centroid, and keep only their shared pixels.

    Input structure:
    - A teal gridline lattice splits the canvas into 3 rows by 4 columns of 4x4 cells.
    - Each cell interior is black except for one red 4-connected shape of 3-5 pixels.
    - Within each lattice row, exactly three cells contain the same red shape and one cell contains a different red shape.

    Rule:
    - Detect the lattice and isolate the 12 inner 4x4 cells.
    - In each row, identify the unique cell by comparing the full red shape pattern across the four cells.
    - For the three extracted odd shapes, compute the floor of the red-pixel centroid inside each 4x4 cell and translate the shapes into centroid-relative coordinates.
    - Take the pixelwise intersection of those aligned coordinate sets and output the tight bounding box of the shared pixels in blue on black.

    Key insight:
    - The answer is not any one odd cell itself; it is the hidden common substructure that appears only after the row-wise odd cells are isolated and all three are aligned to the same centroid.
    """
    rows = _extract_cells(np.asarray(grid, dtype=int))
    odd_rel_sets: list[frozenset[tuple[int, int]]] = []
    for row in rows:
        sigs = [cell.tobytes() for cell in row]
        counts = Counter(sigs)
        assert sorted(counts.values()) == [1, 3], "each row must have exactly one odd cell and one 3-copy majority"
        odd_index = next(i for i, sig in enumerate(sigs) if counts[sig] == 1)
        odd_coords = _extract_red_coords(row[odd_index])
        odd_rel_sets.append(_relative_to_centroid(odd_coords))
    common = set(odd_rel_sets[0])
    for rel in odd_rel_sets[1:]:
        common &= set(rel)
    assert common, "aligned odd shapes must share at least one pixel"
    return _render_output(sorted(common))


def _enumerate_shapes() -> list[dict]:
    shapes: list[dict] = []
    flats = list(range(CELL_SIZE * CELL_SIZE))
    for size in (3, 4, 5):
        for combo in combinations(flats, size):
            coords = tuple(sorted((idx // CELL_SIZE, idx % CELL_SIZE) for idx in combo))
            if not _is_connected(coords):
                continue
            rel = _relative_to_centroid(coords)
            shapes.append(
                {
                    "coords": coords,
                    "coords_set": frozenset(coords),
                    "rel": rel,
                    "centroid": _centroid_floor(coords),
                    "array": _shape_to_array(coords),
                }
            )
    return shapes


_SHAPES = _enumerate_shapes()


def _common_output_from_shapes(shapes: list[dict]) -> frozenset[tuple[int, int]]:
    common = set(shapes[0]["rel"])
    for shape in shapes[1:]:
        common &= set(shape["rel"])
    return frozenset(common)


def _good_odd_triple(shapes: list[dict]) -> bool:
    common = _common_output_from_shapes(shapes)
    if not (1 <= len(common) <= 3):
        return False
    if len({shape["coords"] for shape in shapes}) != 3:
        return False
    if len({shape["rel"] for shape in shapes}) != 3:
        return False
    if len({shape["centroid"] for shape in shapes}) < 2:
        return False
    if any(shape["rel"] == common for shape in shapes):
        return False
    if any(not (set(shape["rel"]) - set(common)) for shape in shapes):
        return False
    raw_common = set(shapes[0]["coords_set"])
    for shape in shapes[1:]:
        raw_common &= set(shape["coords_set"])
    if _normalize_shape(raw_common) == _normalize_shape(common):
        return False
    return True


def _choose_odd_triple(rng: np.random.Generator) -> list[dict]:
    for _ in range(1000):
        idxs = rng.choice(len(_SHAPES), size=3, replace=False)
        shapes = [_SHAPES[int(i)] for i in idxs]
        if _good_odd_triple(shapes):
            return shapes
    raise RuntimeError("failed to pick a valid odd triple")


def _choose_majorities(
    rng: np.random.Generator,
    odd_shapes: list[dict],
    target_common: frozenset[tuple[int, int]],
) -> list[dict]:
    target_norm = _normalize_shape(target_common)
    odd_coord_set = {shape["coords"] for shape in odd_shapes}
    valid_per_row: list[list[dict]] = []
    reuse_rows: list[list[dict]] = []
    for row_idx in range(ROW_COUNT):
        valid = []
        reuse = []
        for candidate in _SHAPES:
            if candidate["coords"] == odd_shapes[row_idx]["coords"]:
                continue
            trial = list(odd_shapes)
            trial[row_idx] = candidate
            if _normalize_shape(_common_output_from_shapes(trial)) != target_norm:
                valid.append(candidate)
                if candidate["coords"] in odd_coord_set:
                    reuse.append(candidate)
        if not valid:
            raise RuntimeError("no valid majority candidate for one row")
        valid_per_row.append(valid)
        reuse_rows.append(reuse)

    reusable_rows = [idx for idx, reuse in enumerate(reuse_rows) if reuse]
    for _ in range(1000):
        majorities = [valid[int(rng.integers(len(valid)))] for valid in valid_per_row]
        if reusable_rows and rng.random() < 0.7:
            row_idx = int(rng.choice(reusable_rows))
            majorities[row_idx] = reuse_rows[row_idx][int(rng.integers(len(reuse_rows[row_idx])))]
        if _normalize_shape(_common_output_from_shapes(majorities)) == target_norm:
            continue
        return majorities
    raise RuntimeError("failed to pick valid majority shapes")


def _assemble_grid(
    odd_shapes: list[dict],
    majority_shapes: list[dict],
    odd_columns: list[int],
) -> np.ndarray:
    height = ROW_COUNT * CELL_SIZE + (ROW_COUNT + 1)
    width = COL_COUNT * CELL_SIZE + (COL_COUNT + 1)
    grid = np.full((height, width), GRID_COLOR, dtype=int)
    for row in range(ROW_COUNT):
        for col in range(COL_COUNT):
            r0 = 1 + row * (CELL_SIZE + 1)
            c0 = 1 + col * (CELL_SIZE + 1)
            grid[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE] = BG
            shape = odd_shapes[row] if col == odd_columns[row] else majority_shapes[row]
            grid[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE] = shape["array"]
    return grid


def _validate_generated_grid(grid: np.ndarray) -> np.ndarray:
    rows = _extract_cells(grid)
    odd_shapes: list[dict] = []
    majority_shapes: list[dict] = []
    for row in rows:
        sigs = [cell.tobytes() for cell in row]
        counts = Counter(sigs)
        assert sorted(counts.values()) == [1, 3]
        odd_sig = next(sig for sig, count in counts.items() if count == 1)
        maj_sig = next(sig for sig, count in counts.items() if count == 3)
        odd_index = sigs.index(odd_sig)
        maj_index = sigs.index(maj_sig)
        odd_coords = _extract_red_coords(row[odd_index])
        maj_coords = _extract_red_coords(row[maj_index])
        odd_shapes.append(
            {
                "coords_set": frozenset(odd_coords),
                "rel": _relative_to_centroid(odd_coords),
                "centroid": _centroid_floor(odd_coords),
            }
        )
        majority_shapes.append(
            {
                "coords_set": frozenset(maj_coords),
                "rel": _relative_to_centroid(maj_coords),
            }
        )

    assert len({shape["rel"] for shape in odd_shapes}) == 3, "odd shapes must all differ"
    assert len({shape["centroid"] for shape in odd_shapes}) >= 2, "alignment must matter"

    common = _common_output_from_shapes(odd_shapes)
    assert 1 <= len(common) <= 3, "shared aligned structure must be small but non-empty"
    assert all(set(shape["rel"]) - set(common) for shape in odd_shapes), "each odd shape needs extra pixels"

    raw_common = set(odd_shapes[0]["coords_set"])
    for shape in odd_shapes[1:]:
        raw_common &= set(shape["coords_set"])
    assert _normalize_shape(raw_common) != _normalize_shape(common), "direct overlap must not match aligned overlap"

    target_norm = _normalize_shape(common)
    for row_idx in range(ROW_COUNT):
        trial = list(odd_shapes)
        trial[row_idx] = majority_shapes[row_idx]
        assert _normalize_shape(_common_output_from_shapes(trial)) != target_norm, "replacing an odd cell with the majority must change the output"
    assert _normalize_shape(_common_output_from_shapes(majority_shapes)) != target_norm, "using all majority shapes must not reproduce the target"

    expected = _render_output(common)
    actual = main(grid)
    assert np.array_equal(actual, expected), "main() must match the validated target"
    return expected


def generate_input() -> np.ndarray:
    for _ in range(600):
        odd_shapes = _choose_odd_triple(_RNG)
        target_common = _common_output_from_shapes(odd_shapes)
        majority_shapes = _choose_majorities(_RNG, odd_shapes, target_common)
        odd_columns = list(map(int, _RNG.integers(0, COL_COUNT, size=ROW_COUNT)))
        if len(set(odd_columns)) == 1 and _RNG.random() < 0.5:
            odd_columns[int(_RNG.integers(ROW_COUNT))] = int((odd_columns[0] + 1 + _RNG.integers(COL_COUNT - 1)) % COL_COUNT)
        grid = _assemble_grid(odd_shapes, majority_shapes, odd_columns)
        _validate_generated_grid(grid)
        return grid
    raise RuntimeError("failed to generate a valid puzzle input")
