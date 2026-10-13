from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg, paste, random_free_location
import random


def _x_mask(height: int, width: int) -> np.ndarray:
    assert height > 0 and width > 0, "interior must be nonempty"
    if height > width:
        return _x_mask(width, height).T

    mask = np.zeros((height, width), dtype=bool)
    center_rows = {(height - 1) // 2, height // 2}
    for r in range(height):
        d = min(r, height - 1 - r)
        left = d
        right = width - 1 - d
        assert left <= right, "x arms must not cross outside the interior"
        if r in center_rows:
            mask[r, left : right + 1] = True
        else:
            mask[r, left] = True
            mask[r, right] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn each filled marked rectangle into a frame containing an X drawn in the marker color.

    Input structure:
    - The background is the most common color.
    - Each 4-connected foreground object is a solid rectangular block with one
      embedded marker pixel of a different color.
    - The rectangle border is all one frame color, so the marker is strictly inside.

    Rule:
    - Replace the full canvas with background, then redraw each rectangle's border
      in its frame color.
    - Clear the rectangle interior except for an X in the marker color.
    - The X uses 45-degree arms from the interior corners and fills any center
      connector where opposite arms would otherwise be separated.

    Key insight:
    - The marker does not mark a position to preserve; it supplies the color for
      the canonical X pattern inside its containing frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    out = np.full(g.shape, bg, dtype=int)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    assert components, "expected at least one marked rectangle"

    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        block = g[r0 : r1 + 1, c0 : c1 + 1]
        height, width = map(int, block.shape)
        assert height >= 3 and width >= 3, "rectangle must have an interior"
        assert np.all(block != bg), "foreground object must fill its bbox"

        border = np.concatenate(
            [block[0, :], block[-1, :], block[1:-1, 0], block[1:-1, -1]]
        )
        frame_colors = np.unique(border)
        assert frame_colors.size == 1, "rectangle border must have one frame color"
        frame = int(frame_colors[0])

        marker_colors = [int(color) for color in np.unique(block) if int(color) != frame]
        assert len(marker_colors) == 1, "expected one marker color in each rectangle"
        marker = marker_colors[0]
        assert int(np.sum(block == marker)) == 1, "marker color must appear once per rectangle"

        out[r0, c0 : c1 + 1] = frame
        out[r1, c0 : c1 + 1] = frame
        out[r0 : r1 + 1, c0] = frame
        out[r0 : r1 + 1, c1] = frame

        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[_x_mask(height - 2, width - 2)] = marker

    return out

# --------------- generator (o5x) ---------------

# Rectangle count and count-conditioned canvas bounds.
RECT_COUNT_CHOICES, RECT_COUNT_WEIGHTS = [1, 2, 3], [1, 3, 10]
GRID_ROWS_BY_RECT_COUNT = {1: (12, 19), 2: (13, 19), 3: (16, 19)}
GRID_COLS_BY_RECT_COUNT = {1: (14, 24), 2: (15, 24), 3: (18, 24)}

# Rectangle dimensions and area gates.
RECT_HEIGHT_BOUNDS, RECT_WIDTH_BOUNDS = (6, 13), (6, 11)
GRID_EDGE_PADDING, SINGLE_RECT_TOTAL_AREA_MARGIN = 1, 2
MIN_RECT_AREA = RECT_HEIGHT_BOUNDS[0] * RECT_WIDTH_BOUNDS[0]
MULTI_RECT_TOTAL_AREA_FRACTION = 0.56
SINGLE_RECT_AREA_FRACTION_BY_COUNT = {2: 0.40, 3: 0.36}

# Packing, colors, and retry budget.
PLACEMENT_BORDER, PLACEMENT_PADDING, PLACEMENT_PADDING_CONNECTIVITY = 1, 1, 4
BG_COLOR, COLOR_BINCOUNT_MIN_LENGTH, GEN_RETRY_LIMIT = 0, 10, 300
PALETTE_COLORS = list(range(1, 10))

def _gen_expected_output(shape: tuple[int, int], rects: list[tuple[int, int, int, int, int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r0, c0, h, w, frame, marker in rects:
        r1 = r0 + h - 1
        c1 = c0 + w - 1
        out[r0, c0 : c1 + 1] = frame
        out[r1, c0 : c1 + 1] = frame
        out[r0 : r1 + 1, c0] = frame
        out[r0 : r1 + 1, c1] = frame
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[_x_mask(h - 2, w - 2)] = marker
    return out

def _gen_choose_sizes(H: int, W: int, n_rects: int) -> list[tuple[int, int]] | None:
    """Choose rectangle sizes while preserving the original area rejection gates."""
    min_h, max_h_bound = RECT_HEIGHT_BOUNDS
    min_w, max_w_bound = RECT_WIDTH_BOUNDS
    max_h = min(max_h_bound, H - 2 * GRID_EDGE_PADDING)
    max_w = min(max_w_bound, W - 2 * GRID_EDGE_PADDING)
    if max_h < min_h or max_w < min_w:
        return None

    grid_area = H * W
    if n_rects == 1:
        total_limit = (grid_area - SINGLE_RECT_TOTAL_AREA_MARGIN) // 2
        single_limit = total_limit
    else:
        total_limit = int(grid_area * MULTI_RECT_TOTAL_AREA_FRACTION)
        single_limit = int(grid_area * SINGLE_RECT_AREA_FRACTION_BY_COUNT[n_rects])

    sizes: list[tuple[int, int]] = []
    used_area = 0
    for idx in range(n_rects):
        remaining_min_area = MIN_RECT_AREA * (n_rects - idx - 1)
        choices = [
            (h, w)
            for h in range(min_h, max_h + 1)
            for w in range(min_w, max_w + 1)
            if h * w <= single_limit and used_area + h * w + remaining_min_area <= total_limit
        ]
        if not choices:
            return None
        h, w = random.choice(choices)
        sizes.append((h, w))
        used_area += h * w

    random.shuffle(sizes)
    return sizes

def _gen_place_rectangles(shape: tuple[int, int], sizes: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    """Pack rectangles with the same padding and random placement policy."""
    occupancy = np.zeros(shape, dtype=int)
    placed: list[tuple[int, int, int, int]] = []
    for h, w in sorted(sizes, key=lambda size: size[0] * size[1], reverse=True):
        patch = np.ones((h, w), dtype=int)
        try:
            r0, c0 = random_free_location(
                occupancy,
                patch,
                transparent=BG_COLOR,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_PADDING_CONNECTIVITY,
            )
        except ValueError:
            return None
        paste(occupancy, patch, r0, c0)
        placed.append((r0, c0, h, w))
    return placed

def _gen_render_marked_rectangles(
    shape: tuple[int, int], placed: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, list[tuple[int, int, int, int, int, int]]]:
    """Render the input grid and matching rectangle descriptors for oracle checks."""
    frame_colors = random.sample(PALETTE_COLORS, len(placed))
    grid = np.zeros(shape, dtype=int)
    rects: list[tuple[int, int, int, int, int, int]] = []
    for (r0, c0, h, w), frame in zip(placed, frame_colors):
        marker = random.choice([color for color in PALETTE_COLORS if color != frame])
        patch = np.full((h, w), frame, dtype=int)
        patch[random.randint(1, h - 2), random.randint(1, w - 2)] = marker
        paste(grid, patch, r0, c0)
        rects.append((r0, c0, h, w, frame, marker))
    return grid, rects

def _gen_matches_solver(shape: tuple[int, int], grid: np.ndarray, rects: list[tuple[int, int, int, int, int, int]]) -> bool:
    expected = _gen_expected_output(shape, rects)
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)

def _gen_attempt() -> np.ndarray | None:
    n_rects = random.choices(RECT_COUNT_CHOICES, weights=RECT_COUNT_WEIGHTS, k=1)[0]
    H = random.randint(*GRID_ROWS_BY_RECT_COUNT[n_rects])
    W = random.randint(*GRID_COLS_BY_RECT_COUNT[n_rects])

    sizes = _gen_choose_sizes(H, W, n_rects)
    if sizes is None:
        return None
    areas = [h * w for h, w in sizes]
    bg_area = H * W - sum(areas)
    if bg_area <= max(areas) + n_rects:
        return None

    shape = (H, W)
    placed = _gen_place_rectangles(shape, sizes)
    if placed is None:
        return None

    grid, rects = _gen_render_marked_rectangles(shape, placed)
    counts = np.bincount(grid.ravel(), minlength=COLOR_BINCOUNT_MIN_LENGTH)
    if counts[BG_COLOR] <= int(counts[1:].max()):
        return None
    if not _gen_matches_solver(shape, grid, rects):
        return None
    return grid

def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 8cb8642d input")
