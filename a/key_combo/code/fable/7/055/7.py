import random
from typing import Iterable

import numpy as np


MAX_GENERATION_ATTEMPTS = 400
PALETTE = tuple(range(10))


def _mask_sig(mask: np.ndarray) -> bytes:
    return mask.astype(np.uint8).tobytes()


def _group_consecutive(indices: Iterable[int]) -> list[tuple[int, int]]:
    values = sorted(int(v) for v in indices)
    if not values:
        return []
    groups = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        groups.append((start, prev))
        start = prev = value
    groups.append((start, prev))
    return groups


def _bands(length: int, separators: list[int]) -> list[tuple[int, int]]:
    groups = _group_consecutive(separators)
    bands = []
    cursor = 0
    for start, end in groups:
        if cursor < start:
            bands.append((cursor, start - 1))
        cursor = end + 1
    if cursor < length:
        bands.append((cursor, length - 1))
    return bands


def _crop(grid: np.ndarray, band_r: tuple[int, int], band_c: tuple[int, int]) -> np.ndarray:
    r0, r1 = band_r
    c0, c1 = band_c
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _full_line_color_map(grid: np.ndarray, axis: int) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    limit = grid.shape[axis]
    for idx in range(limit):
        line = grid[idx, :] if axis == 0 else grid[:, idx]
        color = int(line[0])
        if np.all(line == color):
            result.setdefault(color, []).append(idx)
    return result


def _parse_lattice(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    row_map = _full_line_color_map(grid, axis=0)
    col_map = _full_line_color_map(grid, axis=1)
    sep_candidates = [
        color
        for color in row_map
        if color in col_map and len(row_map[color]) >= 4 and len(col_map[color]) >= 4
    ]
    assert len(sep_candidates) == 1, "expected exactly one separator color"
    sep = sep_candidates[0]
    row_bands = _bands(grid.shape[0], row_map[sep])
    col_bands = _bands(grid.shape[1], col_map[sep])
    assert len(row_bands) >= 3 and len(col_bands) >= 3, "table needs headers plus interior"
    row_heights = {r1 - r0 + 1 for r0, r1 in row_bands}
    col_widths = {c1 - c0 + 1 for c0, c1 in col_bands}
    assert len(row_heights) == 1 and len(col_widths) == 1, "all cells must be equal-sized"
    return int(sep), row_bands, col_bands


def _cell_mask_and_colors(cell: np.ndarray, bg: int, sep: int) -> tuple[np.ndarray, list[int]]:
    colors = sorted(int(v) for v in np.unique(cell) if int(v) not in {bg, sep})
    mask = ~np.isin(cell, [bg, sep])
    return mask, colors


def _analyze_with_bg(
    grid: np.ndarray,
    sep: int,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    bg: int,
) -> dict | None:
    cells = [[_crop(grid, rb, cb) for cb in col_bands] for rb in row_bands]
    if not np.all(cells[0][0] == bg):
        return None

    row_header_masks = []
    col_header_masks = []

    for j in range(1, len(col_bands)):
        mask, colors = _cell_mask_and_colors(cells[0][j], bg, sep)
        if len(colors) != 1 or not mask.any():
            return None
        col_header_masks.append(mask)

    for i in range(1, len(row_bands)):
        mask, colors = _cell_mask_and_colors(cells[i][0], bg, sep)
        if len(colors) != 1 or not mask.any():
            return None
        row_header_masks.append(mask)

    expected_masks: list[list[np.ndarray]] = []
    actual_masks: list[list[np.ndarray]] = []
    mismatch_positions: list[tuple[int, int]] = []
    matching_inner_colors = set()

    for i in range(1, len(row_bands)):
        expected_row = []
        actual_row = []
        for j in range(1, len(col_bands)):
            expected = row_header_masks[i - 1] & col_header_masks[j - 1]
            actual, colors = _cell_mask_and_colors(cells[i][j], bg, sep)
            if len(colors) > 1:
                return None
            if np.array_equal(actual, expected):
                if expected.any():
                    if len(colors) != 1:
                        return None
                    matching_inner_colors.add(colors[0])
                elif colors:
                    return None
            else:
                mismatch_positions.append((i - 1, j - 1))
            expected_row.append(expected)
            actual_row.append(actual)
        expected_masks.append(expected_row)
        actual_masks.append(actual_row)

    if not matching_inner_colors or len(matching_inner_colors) != 1:
        return None

    return {
        "bg": int(bg),
        "sep": int(sep),
        "row_bands": row_bands,
        "col_bands": col_bands,
        "row_header_masks": row_header_masks,
        "col_header_masks": col_header_masks,
        "expected_masks": expected_masks,
        "actual_masks": actual_masks,
        "mismatch_positions": mismatch_positions,
        "inner_color": int(next(iter(matching_inner_colors))),
    }


def _analyze_grid(grid: np.ndarray, required_mismatches: int | None = 1) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep, row_bands, col_bands = _parse_lattice(g)
    corner = _crop(g, row_bands[0], col_bands[0])
    candidates = [int(v) for v in np.unique(corner) if int(v) != sep]
    for value in np.unique(g):
        ivalue = int(value)
        if ivalue != sep and ivalue not in candidates:
            candidates.append(ivalue)

    valid = []
    for bg in candidates:
        info = _analyze_with_bg(g, sep, row_bands, col_bands, int(bg))
        if info is None:
            continue
        mismatch_count = len(info["mismatch_positions"])
        if required_mismatches is None or mismatch_count == required_mismatches:
            valid.append(info)

    assert len(valid) == 1, "grid should admit exactly one consistent table interpretation"
    return valid[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the single wrong interior cell in a separator-lattice table by recomputing its row/column mask intersection.

    Input structure:
    - A single-color separator lattice splits the grid into equal rectangular cells with a blank top-left corner cell.
    - The first row (excluding the corner) contains monochrome column-header masks, and the first column contains monochrome row-header masks.
    - Each interior cell is intended to show the pixelwise intersection of its row header and column header using one dedicated interior color.

    Rule:
    - Recover the lattice, extract each header mask as a boolean pattern inside its cell, and compute every expected row/column intersection mask.
    - Compare each expected mask with the actual interior cell contents and locate the unique violating interior cell.
    - Output the original grid with only that one cell redrawn to the correct intersection mask, preserving the separator, background, and all other cells.

    Key insight:
    - The interior is a boolean multiplication table: every data cell is generated from two visible header masks, so the wrong cell is identified by inconsistency with its row and column generators rather than by its own shape.
    """
    info = _analyze_grid(grid, required_mismatches=1)
    out = np.array(grid, dtype=int, copy=True)
    wrong_r, wrong_c = info["mismatch_positions"][0]
    band_r = info["row_bands"][wrong_r + 1]
    band_c = info["col_bands"][wrong_c + 1]
    cell = _crop(out, band_r, band_c)
    expected = info["expected_masks"][wrong_r][wrong_c]
    cell[:] = info["bg"]
    cell[expected] = info["inner_color"]
    return out


def _random_segment_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    primitive_count = random.randint(2, 4)
    for _ in range(primitive_count):
        kind = random.choice(["row", "col", "rect", "point"])
        if kind == "row":
            r = random.randrange(h)
            c0 = random.randrange(w)
            c1 = random.randrange(c0, w)
            mask[r, c0 : c1 + 1] = True
        elif kind == "col":
            c = random.randrange(w)
            r0 = random.randrange(h)
            r1 = random.randrange(r0, h)
            mask[r0 : r1 + 1, c] = True
        elif kind == "rect":
            r0 = random.randrange(h)
            r1 = random.randrange(r0, h)
            c0 = random.randrange(w)
            c1 = random.randrange(c0, w)
            mask[r0 : r1 + 1, c0 : c1 + 1] = True
        else:
            mask[random.randrange(h), random.randrange(w)] = True

    if not mask.any():
        mask[random.randrange(h), random.randrange(w)] = True
    if mask.all():
        rr = random.randrange(h)
        cc = random.randrange(w)
        mask[rr, cc] = False
    return mask


def _sample_distinct_masks(count: int, h: int, w: int) -> list[np.ndarray] | None:
    masks = []
    seen = set()
    for _ in range(200):
        mask = _random_segment_mask(h, w)
        area = int(mask.sum())
        if area == 0 or area == h * w:
            continue
        sig = _mask_sig(mask)
        if sig in seen:
            continue
        seen.add(sig)
        masks.append(mask)
        if len(masks) == count:
            return masks
    return None


def _mask_grid_from_headers(row_masks: list[np.ndarray], col_masks: list[np.ndarray]) -> list[list[np.ndarray]]:
    return [[rm & cm for cm in col_masks] for rm in row_masks]


def _shape_count(masks: list[list[np.ndarray]]) -> dict[bytes, int]:
    counts: dict[bytes, int] = {}
    for row in masks:
        for mask in row:
            sig = _mask_sig(mask)
            counts[sig] = counts.get(sig, 0) + 1
    return counts


def _semantic_checks(grid: np.ndarray) -> None:
    info = _analyze_grid(grid, required_mismatches=1)
    expected = info["expected_masks"]
    actual = info["actual_masks"]
    row_count = len(expected)
    col_count = len(expected[0])
    assert row_count >= 2 and col_count >= 2

    all_expected = [mask for row in expected for mask in row]
    assert any(mask.any() for mask in all_expected), "need visible intersections"
    assert any(not mask.any() for mask in all_expected), "need empty intersections too"
    assert len({_mask_sig(mask) for mask in all_expected}) >= 3, "need several distinct results"

    for i in range(row_count):
        assert len({_mask_sig(expected[i][j]) for j in range(col_count)}) >= 2, "each row must depend on columns"
    for j in range(col_count):
        assert len({_mask_sig(expected[i][j]) for i in range(row_count)}) >= 2, "each column must depend on rows"

    wrong_r, wrong_c = info["mismatch_positions"][0]
    wrong_actual = actual[wrong_r][wrong_c]
    assert not np.array_equal(wrong_actual, expected[wrong_r][wrong_c])
    donor_exists = any(
        (i, j) != (wrong_r, wrong_c) and np.array_equal(expected[i][j], wrong_actual)
        for i in range(row_count)
        for j in range(col_count)
    )
    assert donor_exists, "wrong mask should reuse another valid intersection"

    repaired = main(grid)
    repaired_info = _analyze_grid(repaired, required_mismatches=0)
    assert not repaired_info["mismatch_positions"], "repaired grid must satisfy the full table rule"


def _gen_attempt() -> np.ndarray | None:
    interior_rows = random.randint(2, 4)
    interior_cols = random.randint(2, 4)
    total_row_cells = interior_rows + 1
    total_col_cells = interior_cols + 1

    feasible_h = [size for size in range(3, 6) if total_row_cells * size + total_row_cells + 1 <= 30]
    feasible_w = [size for size in range(3, 6) if total_col_cells * size + total_col_cells + 1 <= 30]
    if not feasible_h or not feasible_w:
        return None

    cell_h = random.choice(feasible_h)
    cell_w = random.choice(feasible_w)
    row_masks = _sample_distinct_masks(interior_rows, cell_h, cell_w)
    col_masks = _sample_distinct_masks(interior_cols, cell_h, cell_w)
    if row_masks is None or col_masks is None:
        return None

    expected = _mask_grid_from_headers(row_masks, col_masks)
    all_expected = [mask for row in expected for mask in row]
    if not any(mask.any() for mask in all_expected):
        return None
    if not any(not mask.any() for mask in all_expected):
        return None
    if len({_mask_sig(mask) for mask in all_expected}) < 3:
        return None

    for i in range(interior_rows):
        if len({_mask_sig(expected[i][j]) for j in range(interior_cols)}) < 2:
            return None
    for j in range(interior_cols):
        if len({_mask_sig(expected[i][j]) for i in range(interior_rows)}) < 2:
            return None

    donor_pairs = [
        ((ti, tj), (di, dj))
        for ti in range(interior_rows)
        for tj in range(interior_cols)
        for di in range(interior_rows)
        for dj in range(interior_cols)
        if (ti, tj) != (di, dj) and not np.array_equal(expected[ti][tj], expected[di][dj])
    ]
    if not donor_pairs:
        return None
    (wrong_r, wrong_c), (donor_r, donor_c) = random.choice(donor_pairs)

    bg, sep, header_color, inner_color = random.sample(PALETTE, 4)
    height = total_row_cells * cell_h + total_row_cells + 1
    width = total_col_cells * cell_w + total_col_cells + 1
    grid = np.full((height, width), sep, dtype=int)

    row_bands = [(1 + i * (cell_h + 1), 1 + i * (cell_h + 1) + cell_h - 1) for i in range(total_row_cells)]
    col_bands = [(1 + j * (cell_w + 1), 1 + j * (cell_w + 1) + cell_w - 1) for j in range(total_col_cells)]

    for band_r in row_bands:
        for band_c in col_bands:
            cell = _crop(grid, band_r, band_c)
            cell[:] = bg

    for j, mask in enumerate(col_masks, start=1):
        cell = _crop(grid, row_bands[0], col_bands[j])
        cell[mask] = header_color
    for i, mask in enumerate(row_masks, start=1):
        cell = _crop(grid, row_bands[i], col_bands[0])
        cell[mask] = header_color

    for i in range(interior_rows):
        for j in range(interior_cols):
            cell = _crop(grid, row_bands[i + 1], col_bands[j + 1])
            mask = expected[i][j]
            if i == wrong_r and j == wrong_c:
                mask = expected[donor_r][donor_c]
            cell[mask] = inner_color

    try:
        _semantic_checks(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid separator-table puzzle")
