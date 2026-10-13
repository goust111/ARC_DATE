from __future__ import annotations

import numpy as np


BACKGROUND = 0
LATTICE_COLOR = 7
FILL_COLORS = (1, 2, 3, 4, 5, 6)
CELL_SIZE_OPTIONS = (3, 4)
LINE_COUNT_OPTIONS = (4, 5)
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _is_regular_spacing(positions: list[int]) -> bool:
    if len(positions) < 3:
        return False
    gaps = np.diff(np.asarray(positions, dtype=int))
    return bool(np.all(gaps == gaps[0]) and gaps[0] > 1)


def _find_lattice(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, list[int], list[int]]] = []

    for color in sorted(set(map(int, np.unique(g))) - {bg}):
        rows = [r for r in range(h) if np.count_nonzero(g[r, :] == color) >= w - 2]
        cols = [c for c in range(w) if np.count_nonzero(g[:, c] == color) >= h - 2]
        if not (_is_regular_spacing(rows) and _is_regular_spacing(cols)):
            continue
        if len(rows) != len(cols) and abs(len(rows) - len(cols)) > 2:
            continue
        if len(rows) >= 2 and len(cols) >= 2:
            row_gap = rows[1] - rows[0]
            col_gap = cols[1] - cols[0]
            if row_gap != col_gap:
                continue
        holes = [(r, c) for r in rows for c in cols if int(g[r, c]) == bg]
        if len(holes) != 4:
            continue
        hole_rows = sorted({r for r, _ in holes})
        hole_cols = sorted({c for _, c in holes})
        if len(hole_rows) != 2 or len(hole_cols) != 2:
            continue
        if set(holes) != {
            (hole_rows[0], hole_cols[0]),
            (hole_rows[0], hole_cols[1]),
            (hole_rows[1], hole_cols[0]),
            (hole_rows[1], hole_cols[1]),
        }:
            continue
        candidates.append((color, rows, cols))

    assert len(candidates) == 1, "expected a unique regular lattice color"
    return candidates[0]


def _crop_box_from_holes(
    grid: np.ndarray, rows: list[int], cols: list[int], bg: int
) -> tuple[int, int, int, int]:
    holes = [(r, c) for r in rows for c in cols if int(grid[r, c]) == bg]
    hole_rows = sorted({r for r, _ in holes})
    hole_cols = sorted({c for _, c in holes})
    assert len(hole_rows) == 2 and len(hole_cols) == 2
    return hole_rows[0], hole_rows[1], hole_cols[0], hole_cols[1]


def _line_mask(shape: tuple[int, int], line_rows: list[int], line_cols: list[int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[np.asarray(line_rows, dtype=int), :] = True
    mask[:, np.asarray(line_cols, dtype=int)] = True
    return mask


def _depth_grid(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    return np.minimum(
        np.minimum(rows, h - 1 - rows),
        np.minimum(cols, w - 1 - cols),
    )


def _occupied_layers(crop: np.ndarray, line_rows: list[int], line_cols: list[int]) -> tuple[list[int], list[int]]:
    depth = _depth_grid(crop.shape)
    mask = _line_mask(crop.shape, line_rows, line_cols)
    layers: list[int] = []
    colors: list[int] = []
    for d in range(int(depth.max()) + 1):
        layer_mask = (depth == d) & (~mask)
        if not np.any(layer_mask):
            continue
        layer_colors = np.unique(crop[layer_mask])
        assert layer_colors.size == 1, "expected each occupied depth layer to be monochrome"
        layers.append(d)
        colors.append(int(layer_colors[0]))
    return layers, colors


def _reverse_crop_layers(crop: np.ndarray, line_rows: list[int], line_cols: list[int]) -> np.ndarray:
    out = crop.copy()
    depth = _depth_grid(crop.shape)
    mask = _line_mask(crop.shape, line_rows, line_cols)
    layers, colors = _occupied_layers(crop, line_rows, line_cols)
    for d, color in zip(layers, reversed(colors), strict=False):
        out[(depth == d) & (~mask)] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the rectangle marked by four missing lattice intersections and reverse its concentric non-line layer colors.

    Input structure:
    - A black background contains one regular one-pixel-thick lattice in a single non-background color.
    - Exactly four lattice intersections are missing and therefore show the background color; those four holes are the corners of one axis-aligned rectangle.
    - Cell interiors contain non-lattice colors arranged as concentric rectangular layers relative to that hole-defined crop.

    Rule:
    - Detect the unique lattice color by its evenly spaced near-complete horizontal and vertical lines.
    - Use the four missing background intersections to define the inclusive crop rectangle.
    - Inside that crop, keep lattice pixels and missing-corner holes unchanged, compute each non-lattice pixel's depth from the crop boundary, collect the monochrome color at every occupied depth, reverse that color order, and repaint the same depth masks with the reversed sequence.

    Key insight:
    - The missing intersections do double duty: they define the crop, and that crop boundary becomes the reference frame for the concentric layer depths whose colors must be reversed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = BACKGROUND
    _line_color, rows, cols = _find_lattice(g, bg)
    r0, r1, c0, c1 = _crop_box_from_holes(g, rows, cols, bg)
    crop = g[r0 : r1 + 1, c0 : c1 + 1].copy()

    crop_rows = [r - r0 for r in rows if r0 <= r <= r1]
    crop_cols = [c - c0 for c in cols if c0 <= c <= c1]
    return _reverse_crop_layers(crop, crop_rows, crop_cols)


def _all_cell_boxes(height: int, width: int, line_rows: list[int], line_cols: list[int]) -> list[tuple[int, int, int, int]]:
    row_breaks = [-1, *line_rows, height]
    col_breaks = [-1, *line_cols, width]
    boxes: list[tuple[int, int, int, int]] = []
    for i in range(len(row_breaks) - 1):
        r0 = row_breaks[i] + 1
        r1 = row_breaks[i + 1] - 1
        if r0 > r1:
            continue
        for j in range(len(col_breaks) - 1):
            c0 = col_breaks[j] + 1
            c1 = col_breaks[j + 1] - 1
            if c0 <= c1:
                boxes.append((r0, r1, c0, c1))
    return boxes


def _relative_lines(size: int, spacing: int) -> list[int]:
    return list(range(0, size, spacing))


def _occupied_depth_indices(height: int, width: int, spacing: int) -> list[int]:
    line_rows = _relative_lines(height, spacing)
    line_cols = _relative_lines(width, spacing)
    mask = _line_mask((height, width), line_rows, line_cols)
    depth = _depth_grid((height, width))
    return [d for d in range(int(depth.max()) + 1) if np.any((depth == d) & (~mask))]


def _paint_crop_pattern(
    grid: np.ndarray,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    spacing: int,
    layer_colors: list[int],
) -> None:
    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = crop.shape
    line_rows = _relative_lines(h, spacing)
    line_cols = _relative_lines(w, spacing)
    mask = _line_mask((h, w), line_rows, line_cols)
    depth = _depth_grid((h, w))
    occupied = _occupied_depth_indices(h, w, spacing)
    assert len(occupied) == len(layer_colors)
    for d, color in zip(occupied, layer_colors, strict=False):
        crop[(depth == d) & (~mask)] = color


def _paint_outside_cells(
    grid: np.ndarray,
    rng: np.random.Generator,
    line_rows: list[int],
    line_cols: list[int],
    crop_box: tuple[int, int, int, int],
) -> None:
    r0, r1, c0, c1 = crop_box
    palette = np.asarray(FILL_COLORS, dtype=int)
    for br0, br1, bc0, bc1 in _all_cell_boxes(grid.shape[0], grid.shape[1], line_rows, line_cols):
        rr = slice(br0, br1 + 1)
        cc = slice(bc0, bc1 + 1)
        if br0 >= r0 and br1 <= r1 and bc0 >= c0 and bc1 <= c1:
            continue
        if rng.random() < 0.35:
            continue
        color = int(rng.choice(palette))
        grid[rr, cc] = color


def _expected_output_from_params(
    grid: np.ndarray,
    line_rows: list[int],
    line_cols: list[int],
    hole_rows: tuple[int, int],
    hole_cols: tuple[int, int],
) -> np.ndarray:
    r0, r1 = hole_rows
    c0, c1 = hole_cols
    crop = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    crop_rows = [r - r0 for r in line_rows if r0 <= r <= r1]
    crop_cols = [c - c0 for c in line_cols if c0 <= c <= c1]
    return _reverse_crop_layers(crop, crop_rows, crop_cols)


def _build_attempt(rng: np.random.Generator) -> np.ndarray | None:
    cell_size = int(rng.choice(CELL_SIZE_OPTIONS))
    spacing = cell_size + 1
    num_h_lines = int(rng.choice(LINE_COUNT_OPTIONS))
    num_v_lines = int(rng.choice(LINE_COUNT_OPTIONS))

    if cell_size == 3:
        span_options = ((2, 2), (2, 3), (3, 2), (3, 3))
    else:
        span_options = ((2, 2), (2, 3), (3, 2))

    span_r, span_c = span_options[int(rng.integers(0, len(span_options)))]
    if num_h_lines <= span_r or num_v_lines <= span_c:
        return None

    trailing_rows = int(rng.integers(0, cell_size + 1))
    trailing_cols = int(rng.integers(0, cell_size + 1))
    height = cell_size + (num_h_lines - 1) * spacing + 1 + trailing_rows
    width = cell_size + (num_v_lines - 1) * spacing + 1 + trailing_cols
    if max(height, width) > 30:
        return None

    line_rows = [cell_size + i * spacing for i in range(num_h_lines)]
    line_cols = [cell_size + i * spacing for i in range(num_v_lines)]
    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[np.asarray(line_rows, dtype=int), :] = LATTICE_COLOR
    grid[:, np.asarray(line_cols, dtype=int)] = LATTICE_COLOR

    start_r = int(rng.integers(0, num_h_lines - span_r))
    start_c = int(rng.integers(0, num_v_lines - span_c))
    hole_rows = (line_rows[start_r], line_rows[start_r + span_r])
    hole_cols = (line_cols[start_c], line_cols[start_c + span_c])
    for r in hole_rows:
        for c in hole_cols:
            grid[r, c] = BACKGROUND

    crop_h = hole_rows[1] - hole_rows[0] + 1
    crop_w = hole_cols[1] - hole_cols[0] + 1
    occupied = _occupied_depth_indices(crop_h, crop_w, spacing)
    if not (3 <= len(occupied) <= 5):
        return None

    layer_colors = [int(c) for c in rng.choice(FILL_COLORS, size=len(occupied), replace=False)]
    _paint_outside_cells(grid, rng, line_rows, line_cols, (hole_rows[0], hole_rows[1], hole_cols[0], hole_cols[1]))
    _paint_crop_pattern(grid, hole_rows[0], hole_rows[1], hole_cols[0], hole_cols[1], spacing, layer_colors)

    expected = _expected_output_from_params(grid, line_rows, line_cols, hole_rows, hole_cols)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    detected_color, detected_rows, detected_cols = _find_lattice(grid, BACKGROUND)
    if detected_color != LATTICE_COLOR or detected_rows != line_rows or detected_cols != line_cols:
        return None

    crop = grid[hole_rows[0] : hole_rows[1] + 1, hole_cols[0] : hole_cols[1] + 1]
    crop_rows = [r - hole_rows[0] for r in line_rows if hole_rows[0] <= r <= hole_rows[1]]
    crop_cols = [c - hole_cols[0] for c in line_cols if hole_cols[0] <= c <= hole_cols[1]]
    occupied_layers, observed_colors = _occupied_layers(crop, crop_rows, crop_cols)
    if occupied_layers != occupied:
        return None
    if observed_colors != layer_colors:
        return None
    if len(set(observed_colors)) != len(observed_colors):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
