from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
LOOKUP_PALETTE = (1, 2, 3, 4, 7)
DRAW_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _foreground_mask(cell: np.ndarray, divider: int = GREY) -> np.ndarray:
    return (cell != BLACK) & (cell != int(divider))


def _crop_foreground(cell: np.ndarray, divider: int = GREY) -> np.ndarray:
    box = _bbox(_foreground_mask(cell, divider))
    assert box is not None, "every cell must contain foreground pixels"
    return _crop_bbox(cell, box)


def _signature(arr: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, arr.shape)), arr.tobytes()


def _divider_color(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    row_colors = [int(g[r, 0]) for r in range(H) if np.all(g[r] == g[r, 0])]
    col_colors = [int(g[0, c]) for c in range(W) if np.all(g[:, c] == g[0, c])]
    shared = set(row_colors) & set(col_colors)
    assert shared, "expected a divider color forming full rows and columns"
    best = max(shared, key=lambda color: (row_colors.count(color) + col_colors.count(color), color == GREY))
    return int(best)


def _bands(limit: int, dividers: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start = 0
    for idx in sorted(dividers) + [int(limit)]:
        if start <= idx - 1:
            out.append((int(start), int(idx - 1)))
        start = int(idx + 1)
    return out


def _split_cells(grid: np.ndarray) -> tuple[list[list[np.ndarray]], int]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    divider = _divider_color(g)
    row_dividers = [r for r in range(H) if np.all(g[r] == divider)]
    col_dividers = [c for c in range(W) if np.all(g[:, c] == divider)]
    row_bands = _bands(H, row_dividers)
    col_bands = _bands(W, col_dividers)
    assert len(row_bands) == 3 and len(col_bands) == 3, "expected a 3x3 divider grid"
    row_heights = {r1 - r0 + 1 for r0, r1 in row_bands}
    col_widths = {c1 - c0 + 1 for c0, c1 in col_bands}
    assert len(row_heights) == 1 and len(col_widths) == 1, "all cells must be equal-sized"
    assert row_heights == col_widths, "cells must be square"
    cell_size = next(iter(row_heights))
    cells = [
        [_crop_bbox(g, (r0, c0, r1, c1)) for c0, c1 in col_bands]
        for r0, r1 in row_bands
    ]
    return cells, int(divider)


def _find_row_outlier(row_cells: list[np.ndarray], divider: int) -> int:
    crops = [_crop_foreground(cell, divider) for cell in row_cells]
    sigs = [_signature(crop) for crop in crops]
    counts = Counter(sigs)
    assert sorted(counts.values()) == [1, 2], "each row must contain one outlier and one duplicated crop"
    target = next(sig for sig, cnt in counts.items() if cnt == 1)
    return int(sigs.index(target))


def _lookup_from_sparse(cells: list[list[np.ndarray]], divider: int) -> tuple[tuple[int, int], list[int], list[list[int]]]:
    counts: list[list[int]] = []
    sparse_index: tuple[int, int] | None = None
    lookup_colors: list[int] | None = None
    for r in range(3):
        row_counts: list[int] = []
        for c in range(3):
            cell = cells[r][c]
            mask = _foreground_mask(cell, divider)
            count = int(mask.sum())
            row_counts.append(count)
            if count == 3:
                assert sparse_index is None, "sparse cell must be unique"
                sparse_index = (int(r), int(c))
                points = [(int(rr), int(cc), int(cell[rr, cc])) for rr, cc in np.argwhere(mask)]
                points.sort(key=lambda item: (item[0], item[1]))
                assert len({rr for rr, _cc, _col in points}) == 3, "sparse cell must occupy three distinct rows"
                lookup_colors = [int(col) for _rr, _cc, col in points]
        counts.append(row_counts)
    assert sparse_index is not None and lookup_colors is not None, "expected one sparse lookup cell"
    return sparse_index, lookup_colors, counts


def _parse_case(grid: np.ndarray) -> dict:
    cells, divider = _split_cells(grid)
    sparse_index, lookup_colors, counts = _lookup_from_sparse(cells, divider)
    anomaly_cols = [_find_row_outlier(cells[r], divider) for r in range(3)]
    return {
        "cells": cells,
        "divider": int(divider),
        "sparse_index": sparse_index,
        "lookup_colors": lookup_colors,
        "counts": counts,
        "anomaly_cols": anomaly_cols,
    }


def _recolor_cell(cell: np.ndarray, color: int, divider: int = GREY) -> np.ndarray:
    out = np.zeros_like(cell)
    out[_foreground_mask(cell, divider)] = int(color)
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Recolor the per-row outlier cells using the sparse cell's top-to-bottom color lookup, then stack them vertically.

    Input structure:
    - Grey divider lines split the grid into a 3x3 array of equal square cells.
    - Each cell has a black background plus a small colored pattern made of non-black, non-grey pixels.
    - In every row of three cells, two cells have identical tight-cropped foreground patterns and one cell is different.
    - Exactly one cell is globally sparse: it contains exactly three foreground pixels, one in each of three distinct occupied rows.

    Rule:
    - Remove the grey divider lines and treat the nine square regions as cells with coordinates (R, C).
    - For each row R, compare the three cells by the tight crop of their foreground pixels and select the unique outlier cell.
    - Find the unique sparse cell, sort its three foreground pixels by row from top to bottom, and use their colors as a row-to-color table.
    - Recolor every foreground pixel of the row-R outlier cell to the sparse-table color for row R, keep black background unchanged, and stack the three recolored outlier cells top-to-bottom.

    Key insight:
    - The output shape comes from per-row outlier detection, while the output colors come from the globally sparse lookup cell; both mechanisms must be solved independently and then crossed.
    """
    parsed = _parse_case(np.asarray(input_grid, dtype=int))
    blocks = []
    for r in range(3):
        cell = parsed["cells"][r][parsed["anomaly_cols"][r]]
        blocks.append(_recolor_cell(cell, parsed["lookup_colors"][r], parsed["divider"]))
    return np.vstack(blocks)


def _mask_signature(arr: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, arr.shape)), (arr != BLACK).tobytes()


def _random_crop(cell_size: int, n_pixels: int, used: set[tuple[tuple[int, int], bytes]]) -> np.ndarray:
    for _ in range(400):
        h = random.randint(3, cell_size)
        w = random.randint(3, cell_size)
        if n_pixels > h * w:
            continue
        coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_pixels)
        rows = {r for r, _c in coords}
        cols = {c for _r, c in coords}
        if len(rows) < 2 or len(cols) < 2:
            continue
        n_colors = random.randint(2, min(4, n_pixels))
        colors = random.sample(DRAW_PALETTE, n_colors)
        random.shuffle(coords)
        crop = np.zeros((h, w), dtype=int)
        for (r, c), color in zip(coords[:n_colors], colors):
            crop[r, c] = int(color)
        for r, c in coords[n_colors:]:
            crop[r, c] = int(random.choice(colors))
        crop = _crop_foreground(crop)
        sig = _signature(crop)
        if sig in used:
            continue
        return crop
    raise RuntimeError("failed to sample a dense crop")


def _random_sparse_crop(
    cell_size: int,
    lookup_colors: list[int],
    used: set[tuple[tuple[int, int], bytes]],
) -> np.ndarray:
    for _ in range(300):
        h = random.randint(3, cell_size)
        w = random.randint(3, cell_size)
        rows = sorted(random.sample(range(h), 3))
        cols = [random.randrange(w) for _ in range(3)]
        if len(set(cols)) < 2:
            continue
        crop = np.zeros((h, w), dtype=int)
        for rr, cc, color in zip(rows, cols, lookup_colors):
            crop[rr, cc] = int(color)
        crop = _crop_foreground(crop)
        sig = _signature(crop)
        if sig in used:
            continue
        return crop
    raise RuntimeError("failed to sample a sparse crop")


def _embed_crop(cell_size: int, crop: np.ndarray) -> np.ndarray:
    h, w = map(int, crop.shape)
    top = random.randint(0, cell_size - h)
    left = random.randint(0, cell_size - w)
    cell = np.zeros((cell_size, cell_size), dtype=int)
    cell[top : top + h, left : left + w] = crop
    return cell


def _assemble_grid(cells: list[list[np.ndarray]], divider: int = GREY) -> np.ndarray:
    cell_size = int(cells[0][0].shape[0])
    size = 3 * cell_size + 2
    grid = np.full((size, size), int(divider), dtype=int)
    stride = cell_size + 1
    for r in range(3):
        for c in range(3):
            r0 = r * stride
            c0 = c * stride
            grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = cells[r][c]
    return grid


def _expected_output(cells: list[list[np.ndarray]], anomaly_cols: list[int], lookup_colors: list[int]) -> np.ndarray:
    return np.vstack([_recolor_cell(cells[r][anomaly_cols[r]], lookup_colors[r]) for r in range(3)])


def _build_case() -> tuple[np.ndarray, np.ndarray, dict]:
    cell_size = random.randint(5, 7)
    sparse_row = random.randint(0, 2)
    sparse_col = random.randint(0, 2)
    lookup_colors = random.sample(list(LOOKUP_PALETTE), 3)
    used_signatures: set[tuple[tuple[int, int], bytes]] = set()
    cells: list[list[np.ndarray]] = [[np.zeros((cell_size, cell_size), dtype=int) for _ in range(3)] for _ in range(3)]
    anomaly_cols: list[int] = []

    for r in range(3):
        majority_pixels = random.choice((5, 6))
        majority_crop = _random_crop(cell_size, majority_pixels, used_signatures)
        used_signatures.add(_signature(majority_crop))

        if r == sparse_row:
            anomaly_col = sparse_col
            anomaly_crop = _random_sparse_crop(cell_size, lookup_colors, used_signatures)
        else:
            anomaly_col = random.randint(0, 2)
            for _ in range(200):
                anomaly_crop = _random_crop(cell_size, majority_pixels, used_signatures)
                if _mask_signature(anomaly_crop) != _mask_signature(majority_crop):
                    break
            else:
                raise RuntimeError("failed to find a non-equivalent outlier crop")
        used_signatures.add(_signature(anomaly_crop))
        anomaly_cols.append(int(anomaly_col))

        for c in range(3):
            crop = anomaly_crop if c == anomaly_col else majority_crop
            cells[r][c] = _embed_crop(cell_size, crop)

    grid = _assemble_grid(cells)
    expected = _expected_output(cells, anomaly_cols, lookup_colors)
    meta = {
        "cell_size": int(cell_size),
        "sparse_index": (int(sparse_row), int(sparse_col)),
        "lookup_colors": [int(color) for color in lookup_colors],
        "anomaly_cols": [int(col) for col in anomaly_cols],
        "cells": cells,
    }
    return grid, expected, meta


def _validate_case(grid: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    try:
        parsed = _parse_case(grid)
    except AssertionError:
        return False

    if parsed["sparse_index"] != meta["sparse_index"]:
        return False
    if parsed["lookup_colors"] != meta["lookup_colors"]:
        return False
    if parsed["anomaly_cols"] != meta["anomaly_cols"]:
        return False
    if not np.array_equal(main(grid), expected):
        return False

    all_counts = [count for row in parsed["counts"] for count in row]
    if all_counts.count(3) != 1:
        return False
    if any(count not in (3, 5, 6) for count in all_counts):
        return False
    sr, sc = meta["sparse_index"]
    if parsed["counts"][sr][sc] != 3:
        return False

    for r in range(3):
        if r == sr:
            continue
        if len(set(parsed["counts"][r])) != 1:
            return False

    for r in range(3):
        anomaly = meta["cells"][r][meta["anomaly_cols"][r]]
        majority = meta["cells"][r][(meta["anomaly_cols"][r] + 1) % 3]
        recolored_anomaly = _recolor_cell(anomaly, meta["lookup_colors"][r])
        recolored_majority = _recolor_cell(majority, meta["lookup_colors"][r])
        if np.array_equal(recolored_anomaly, recolored_majority):
            return False

    if expected.shape != (3 * meta["cell_size"], meta["cell_size"]):
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid, expected, meta = _build_case()
        if _validate_case(grid, expected, meta):
            return grid
    raise RuntimeError("failed to generate a valid input")
