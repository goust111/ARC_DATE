from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
AZURE = 8
ARC_MAX = 30
MAX_GENERATION_ATTEMPTS = 300
REGION_PLACEMENT_ATTEMPTS = 200
COUNTER_PLACEMENT_ATTEMPTS = 120


BASE_SHAPES = [
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
]


def _bands(length: int, separators: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for cut in sorted(set(int(x) for x in separators)):
        if start <= cut - 1:
            bands.append((start, cut - 1))
        start = cut + 1
    if start <= length - 1:
        bands.append((start, length - 1))
    return bands


def _find_separator(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    h, w = map(int, grid.shape)
    best: tuple[int, int, list[int], list[int]] | None = None
    for color in np.unique(grid):
        color = int(color)
        if color == BLACK:
            continue
        row_ids = [r for r in range(h) if np.all(grid[r, :] == color)]
        col_ids = [c for c in range(w) if np.all(grid[:, c] == color)]
        if not row_ids or not col_ids:
            continue
        score = len(row_ids) + len(col_ids)
        cand = (score, color, row_ids, col_ids)
        if best is None or cand[0] > best[0]:
            best = cand
    assert best is not None, "expected a full-row/full-column separator color"
    _, color, row_ids, col_ids = best
    return color, row_ids, col_ids


def _crop_nonzero(masked: np.ndarray) -> np.ndarray:
    coords = np.argwhere(masked != BLACK)
    assert len(coords) > 0, "expected a non-empty template"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return masked[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _scale_template(template: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(template, np.ones((factor, factor), dtype=int))


def _compose_output(cells: list[list[np.ndarray]]) -> np.ndarray:
    n_rows = len(cells)
    n_cols = len(cells[0])
    row_heights = [max(cells[r][c].shape[0] for c in range(n_cols)) for r in range(n_rows)]
    col_widths = [max(cells[r][c].shape[1] for r in range(n_rows)) for c in range(n_cols)]

    out_h = sum(row_heights) + (n_rows - 1)
    out_w = sum(col_widths) + (n_cols - 1)
    out = np.zeros((out_h, out_w), dtype=int)

    r_off = 0
    for r in range(n_rows):
        c_off = 0
        for c in range(n_cols):
            cell = cells[r][c]
            h, w = cell.shape
            out[r_off : r_off + h, c_off : c_off + w] = cell
            c_off += col_widths[c] + 1
        r_off += row_heights[r] + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale each region's template by the number of red counter pixels in that region.

    Input structure:
    - Black background with one full-row/full-column separator lattice in Azure, forming a rectangular grid of regions.
    - Each region contains exactly two disconnected colored objects: a non-red single-color template shape and a red counter made of 1-4 pixels.
    - The template may have different color, orientation, and bounding-box size in different regions.

    Rule:
    - Split the grid into regions using the complete separator rows and columns.
    - In each region, count the Red pixels to get a scale factor N.
    - Crop the non-red template to its tight bounding box, then replace each of its pixels with an N x N block of the same color.
    - Arrange the scaled templates in the same region row/column order, inserting one Black row/column between neighboring output regions.

    Key insight:
    - The red object is not another template part; its pixel count controls the geometric magnification of the other object.
    """
    g = np.asarray(grid, dtype=int)
    sep, sep_rows, sep_cols = _find_separator(g)
    row_bands = _bands(g.shape[0], sep_rows)
    col_bands = _bands(g.shape[1], sep_cols)
    assert row_bands and col_bands, "separator lattice must create at least one region"

    cells: list[list[np.ndarray]] = []
    for r0, r1 in row_bands:
        row_cells: list[np.ndarray] = []
        for c0, c1 in col_bands:
            region = g[r0 : r1 + 1, c0 : c1 + 1]
            factor = int(np.count_nonzero(region == RED))
            assert 1 <= factor <= 4, "counter size must be between 1 and 4"

            colors = {int(x) for x in np.unique(region)}
            template_colors = [color for color in colors if color not in {BLACK, RED, sep}]
            assert len(template_colors) == 1, "each region must contain exactly one template color"
            template_color = int(template_colors[0])

            template = np.where(region == template_color, template_color, BLACK)
            cropped = _crop_nonzero(template)
            row_cells.append(_scale_template(cropped, factor))
        cells.append(row_cells)

    return _compose_output(cells)


def _shape_variants(max_h: int, max_w: int) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for base in BASE_SHAPES:
        for raw in (base, np.fliplr(base)):
            for k in range(4):
                shape = np.rot90(raw, k)
                h, w = map(int, shape.shape)
                key = ((h, w), shape.tobytes())
                if h > max_h or w > max_w or key in seen:
                    continue
                seen.add(key)
                variants.append(shape.copy())
    assert variants, "expected at least one admissible template shape"
    return variants


def _counts_for_layout(n_cells: int) -> list[int]:
    counts = [1, 2, 3, 4]
    while len(counts) < n_cells:
        counts.append(random.randint(1, 4))
    random.shuffle(counts)
    return counts[:n_cells]


def _pick_counter_positions(candidates: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    for _ in range(COUNTER_PLACEMENT_ATTEMPTS):
        random.shuffle(candidates)
        chosen: list[tuple[int, int]] = []
        for r, c in candidates:
            if all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in chosen):
                chosen.append((r, c))
                if len(chosen) == count:
                    return chosen
    return None


def _place_region(shape: np.ndarray, color: int, cell_h: int, cell_w: int, count: int) -> np.ndarray | None:
    sh, sw = map(int, shape.shape)
    for _ in range(REGION_PLACEMENT_ATTEMPTS):
        region = np.zeros((cell_h, cell_w), dtype=int)
        top = random.randint(0, cell_h - sh)
        left = random.randint(0, cell_w - sw)
        template_pixels: list[tuple[int, int]] = []
        for r in range(sh):
            for c in range(sw):
                if int(shape[r, c]) == 1:
                    rr = top + r
                    cc = left + c
                    region[rr, cc] = color
                    template_pixels.append((rr, cc))

        forbidden = np.zeros_like(region, dtype=bool)
        for rr, cc in template_pixels:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < cell_h and 0 <= nc < cell_w:
                        forbidden[nr, nc] = True

        candidates = [
            (r, c)
            for r in range(cell_h)
            for c in range(cell_w)
            if region[r, c] == BLACK and not forbidden[r, c]
        ]
        if len(candidates) < count:
            continue
        counters = _pick_counter_positions(candidates, count)
        if counters is None:
            continue
        for rr, cc in counters:
            region[rr, cc] = RED
        return region
    return None


def _scale_expected(shape: np.ndarray, color: int, factor: int) -> np.ndarray:
    template = (shape.astype(int) * color).astype(int)
    return _scale_template(template, factor)


def _build_grid(
    row_heights: list[int],
    col_widths: list[int],
    regions: list[list[np.ndarray]],
) -> np.ndarray:
    grid_h = sum(row_heights) + len(row_heights) + 1
    grid_w = sum(col_widths) + len(col_widths) + 1
    grid = np.zeros((grid_h, grid_w), dtype=int)

    row_starts: list[int] = []
    cursor = 1
    for h in row_heights:
        row_starts.append(cursor)
        cursor += h + 1

    col_starts: list[int] = []
    cursor = 1
    for w in col_widths:
        col_starts.append(cursor)
        cursor += w + 1

    grid[0, :] = AZURE
    grid[-1, :] = AZURE
    grid[:, 0] = AZURE
    grid[:, -1] = AZURE
    for start, h in zip(row_starts, row_heights):
        grid[start - 1, :] = AZURE
        grid[start + h, :] = AZURE
    for start, w in zip(col_starts, col_widths):
        grid[:, start - 1] = AZURE
        grid[:, start + w] = AZURE

    for r, top in enumerate(row_starts):
        for c, left in enumerate(col_starts):
            region = regions[r][c]
            h, w = region.shape
            grid[top : top + h, left : left + w] = region
    return grid


def _gen_attempt() -> np.ndarray | None:
    n_rows, n_cols = random.choice([(2, 2), (2, 3), (3, 2)])
    max_shape_h = 2 if n_rows == 3 else 3
    max_shape_w = 2 if n_cols == 3 else 3
    shapes = _shape_variants(max_shape_h, max_shape_w)
    palette = [color for color in range(1, 10) if color not in {RED, AZURE}]

    n_cells = n_rows * n_cols
    counts = _counts_for_layout(n_cells)
    shape_grid: list[list[np.ndarray]] = []
    color_grid: list[list[int]] = []
    expected_cells: list[list[np.ndarray]] = []

    idx = 0
    for _r in range(n_rows):
        shape_row: list[np.ndarray] = []
        color_row: list[int] = []
        expected_row: list[np.ndarray] = []
        for _c in range(n_cols):
            shape = random.choice(shapes).copy()
            color = random.choice(palette)
            factor = counts[idx]
            idx += 1
            shape_row.append(shape)
            color_row.append(color)
            expected_row.append(_scale_expected(shape, color, factor))
        shape_grid.append(shape_row)
        color_grid.append(color_row)
        expected_cells.append(expected_row)

    row_heights = [
        max(shape_grid[r][c].shape[0] for c in range(n_cols)) + random.randint(3, 4)
        for r in range(n_rows)
    ]
    col_widths = [
        max(shape_grid[r][c].shape[1] for r in range(n_rows)) + random.randint(3, 4)
        for c in range(n_cols)
    ]

    if sum(row_heights) + n_rows + 1 > ARC_MAX or sum(col_widths) + n_cols + 1 > ARC_MAX:
        return None

    out_h = _compose_output(expected_cells).shape[0]
    out_w = _compose_output(expected_cells).shape[1]
    if out_h > ARC_MAX or out_w > ARC_MAX:
        return None

    regions: list[list[np.ndarray]] = []
    idx = 0
    for r in range(n_rows):
        row_regions: list[np.ndarray] = []
        for c in range(n_cols):
            region = _place_region(
                shape_grid[r][c],
                color_grid[r][c],
                row_heights[r],
                col_widths[c],
                counts[idx],
            )
            idx += 1
            if region is None:
                return None
            row_regions.append(region)
        regions.append(row_regions)

    grid = _build_grid(row_heights, col_widths, regions)
    expected = _compose_output(expected_cells)
    if grid.shape[0] > ARC_MAX or grid.shape[1] > ARC_MAX:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
