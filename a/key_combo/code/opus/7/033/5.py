from __future__ import annotations

import random

import numpy as np


BG = 0
MARKER = 2
FRAME_PALETTE = (1, 3, 4, 8)

# --------------- generator config ---------------

GEN_GRID_MIN = 18
GEN_GRID_MAX = 24
GEN_RECT_COUNT_CHOICES = (3, 4)
GEN_RECT_MIN_SIDE = 5
GEN_RECT_MAX_SIDE = 9
GEN_OUTER_MARGIN = 1
GEN_REGION_GAP_MIN = 1
GEN_REGION_GAP_MAX = 3
GEN_MAX_RETRIES = 400


def _bbox_of_color(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == color)
    assert coords.size > 0, f"missing color {color}"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _side_inner_coords(
    bbox: tuple[int, int, int, int], side: str
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    if side == "right":
        return [(r, c1) for r in range(r0 + 1, r1)]
    raise ValueError(f"unknown side: {side}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each incomplete rectangle and carve its interior checkerboard phase from the count of red markers.

    Input structure:
    - The grid has a black background plus several incomplete monochrome rectangular frames in distinct colors.
    - Each frame color appears only on the border of its own axis-aligned bounding box, with some border cells missing.
    - Red marker pixels lie strictly inside the frame bounding boxes and nowhere on the borders.

    Rule:
    - For each non-black, non-red frame color, infer its full rectangle from the color's bounding box and complete the entire border.
    - Count the red markers inside the strict interior of that box.
    - Fill the strict interior with a checkerboard of the frame color, keeping parity `(local_row + local_col) % 2 == count % 2` and setting the opposite parity to black.
    - Erase all red markers to black, even if a marker lies on a parity that would otherwise keep the frame color.

    Key insight:
    - The colored border pixels reveal the missing rectangle extent, while only the parity of the interior red-marker count decides how the checkerboard aligns.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    frame_colors = [int(color) for color in np.unique(g) if int(color) not in {BG, MARKER}]
    assert frame_colors, "expected at least one frame color"

    covered_markers = np.zeros(g.shape, dtype=bool)

    for color in frame_colors:
        r0, c0, r1, c1 = _bbox_of_color(g, color)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height >= GEN_RECT_MIN_SIDE and width >= GEN_RECT_MIN_SIDE

        color_coords = np.argwhere(g == color)
        for r, c in color_coords:
            rr, cc = int(r), int(c)
            assert rr in (r0, r1) or cc in (c0, c1), "frame color must stay on the bbox border"

        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[:, :] = BG
        region[0, :] = color
        region[-1, :] = color
        region[:, 0] = color
        region[:, -1] = color

        interior_input = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_mask = interior_input == MARKER
        covered_markers[r0 + 1 : r1, c0 + 1 : c1] |= marker_mask
        marker_count = int(marker_mask.sum())

        interior_output = region[1:-1, 1:-1]
        rows, cols = np.indices(interior_output.shape)
        keep_mask = (rows + cols) % 2 == (marker_count % 2)
        interior_output[:, :] = BG
        interior_output[keep_mask] = color
        interior_output[marker_mask] = BG

    assert np.all((g != MARKER) | covered_markers), "all red markers must be inside some rectangle interior"
    return out


def _sample_marker_counts(rect_count: int) -> list[int]:
    while True:
        counts = [random.randint(1, 4) for _ in range(rect_count)]
        if len(set(counts)) < 2:
            continue
        if len({count % 2 for count in counts}) < 2:
            continue
        return counts


def _partition_regions(height: int, width: int) -> list[tuple[int, int, int, int]]:
    min_region_span = GEN_RECT_MIN_SIDE + 2 * GEN_OUTER_MARGIN
    gap_r = random.randint(GEN_REGION_GAP_MIN, GEN_REGION_GAP_MAX)
    gap_c = random.randint(GEN_REGION_GAP_MIN, GEN_REGION_GAP_MAX)

    top_h = random.randint(min_region_span, height - gap_r - min_region_span)
    left_w = random.randint(min_region_span, width - gap_c - min_region_span)

    top_rows = (0, top_h - 1)
    bottom_rows = (top_h + gap_r, height - 1)
    left_cols = (0, left_w - 1)
    right_cols = (left_w + gap_c, width - 1)

    return [
        (top_rows[0], left_cols[0], top_rows[1], left_cols[1]),
        (top_rows[0], right_cols[0], top_rows[1], right_cols[1]),
        (bottom_rows[0], left_cols[0], bottom_rows[1], left_cols[1]),
        (bottom_rows[0], right_cols[0], bottom_rows[1], right_cols[1]),
    ]


def _sample_box_in_region(region: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    rr0, cc0, rr1, cc1 = region
    available_h = rr1 - rr0 + 1 - 2 * GEN_OUTER_MARGIN
    available_w = cc1 - cc0 + 1 - 2 * GEN_OUTER_MARGIN
    if available_h < GEN_RECT_MIN_SIDE or available_w < GEN_RECT_MIN_SIDE:
        return None

    rect_h = random.randint(GEN_RECT_MIN_SIDE, min(GEN_RECT_MAX_SIDE, available_h))
    rect_w = random.randint(GEN_RECT_MIN_SIDE, min(GEN_RECT_MAX_SIDE, available_w))

    r0 = random.randint(rr0 + GEN_OUTER_MARGIN, rr1 - GEN_OUTER_MARGIN - rect_h + 1)
    c0 = random.randint(cc0 + GEN_OUTER_MARGIN, cc1 - GEN_OUTER_MARGIN - rect_w + 1)
    return r0, c0, r0 + rect_h - 1, c0 + rect_w - 1


def _draw_incomplete_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color

    chosen_sides = random.sample(("top", "bottom", "left", "right"), random.randint(1, 3))
    for side in chosen_sides:
        cells = _side_inner_coords(bbox, side)
        gap_len = random.randint(1, len(cells))
        start = random.randint(0, len(cells) - gap_len)
        for r, c in cells[start : start + gap_len]:
            grid[r, c] = BG


def _marker_parity_cells(
    bbox: tuple[int, int, int, int], parity: int
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            local_parity = (r - (r0 + 1) + c - (c0 + 1)) % 2
            if local_parity == parity:
                cells.append((r, c))
    return cells


def _all_interior_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _place_markers(grid: np.ndarray, bbox: tuple[int, int, int, int], count: int) -> None:
    keep_parity = count % 2
    visible_cells = _marker_parity_cells(bbox, keep_parity)
    all_cells = _all_interior_cells(bbox)

    chosen = [random.choice(visible_cells)]
    remaining = [cell for cell in all_cells if cell not in chosen]
    random.shuffle(remaining)
    chosen.extend(remaining[: count - 1])

    for r, c in chosen:
        grid[r, c] = MARKER


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    width = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    rect_count = random.choice(GEN_RECT_COUNT_CHOICES)
    marker_counts = _sample_marker_counts(rect_count)

    regions = _partition_regions(height, width)
    chosen_regions = random.sample(regions, rect_count)
    boxes = [_sample_box_in_region(region) for region in chosen_regions]
    if any(box is None for box in boxes):
        return None

    colors = random.sample(list(FRAME_PALETTE), rect_count)
    grid = np.zeros((height, width), dtype=int)

    for bbox, color, marker_count in zip(boxes, colors, marker_counts):
        assert bbox is not None
        _draw_incomplete_frame(grid, bbox, color)
        _place_markers(grid, bbox, marker_count)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 5")
