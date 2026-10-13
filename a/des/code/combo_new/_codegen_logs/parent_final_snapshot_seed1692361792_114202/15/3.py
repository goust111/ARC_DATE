import random

import numpy as np


_BG = 0
_COLOR_POOL = tuple(range(1, 10))
_OUTER_WIDTH = 13
_GEN_RETRIES = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.asarray(coords, dtype=int)
    return (
        int(pts[:, 0].min()),
        int(pts[:, 1].min()),
        int(pts[:, 0].max()),
        int(pts[:, 1].max()),
    )


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _draw_rect_ring(
    grid: np.ndarray, top: int, left: int, height: int, width: int, color: int
) -> None:
    grid[top, left : left + width] = int(color)
    grid[top + height - 1, left : left + width] = int(color)
    grid[top : top + height, left] = int(color)
    grid[top : top + height, left + width - 1] = int(color)


def _extract_structure(
    grid: np.ndarray,
) -> tuple[int, list[tuple[int, int, int, int]], list[int]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    remaining = g.copy()
    ring_bboxes: list[tuple[int, int, int, int]] = []

    for _ in range(3):
        coords = np.argwhere(remaining != bg)
        assert coords.size > 0, "expected remaining foreground pixels"
        r0, c0, r1, c1 = bbox_coords(coords)
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        assert height >= 3 and width >= 3, "ring bbox must be at least 3x3"

        window = remaining[r0 : r1 + 1, c0 : c1 + 1]
        border = _border_mask(height, width)
        border_vals = np.unique(window[border])
        border_vals = border_vals[border_vals != bg]
        assert border_vals.size == 1, "bbox border must be a single-color ring"
        assert np.all(window[border] == int(border_vals[0])), "ring border must be complete"

        ring_bboxes.append((r0, c0, r1, c1))
        window[border] = bg

    markers = np.argwhere(remaining != bg)
    assert markers.shape[0] == 3, "expected exactly three marker pixels"
    marker_cols = [int(c) for _, c in markers]
    assert len(set(marker_cols)) == 3, "marker columns must be distinct"

    ordered_colors = [
        int(remaining[r, c])
        for r, c in sorted(
            ((int(r), int(c)) for r, c in markers), key=lambda rc: (rc[1], rc[0])
        )
    ]
    return bg, ring_bboxes, ordered_colors


def _assemble_output(
    bg: int, ring_bboxes: list[tuple[int, int, int, int]], recolors: list[int]
) -> np.ndarray:
    heights = [int(r1 - r0 + 1) for r0, _c0, r1, _c1 in ring_bboxes]
    widths = [int(c1 - c0 + 1) for _r0, c0, _r1, c1 in ring_bboxes]
    outer_height = heights[0]
    out_width = int(sum(widths) + len(widths) - 1)
    out = np.full((outer_height, out_width), int(bg), dtype=int)

    left = 0
    for idx, ((r0, c0, r1, c1), color) in enumerate(zip(ring_bboxes, recolors)):
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        top = int((outer_height - height) // 2)
        _draw_rect_ring(out, top, left, height, width, int(color))
        left += width
        if idx != len(ring_bboxes) - 1:
            left += 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split concentric rectangular rings into a horizontal row and recolor them by the left-to-right order of interior marker dots.

    Input structure:
    - The background is black.
    - Three one-pixel-thick concentric rectangular rings sit near the grid center, with one background cell of spacing between successive rings.
    - Inside the innermost ring's cavity there are exactly three single-pixel markers, one in each ring color, and their column coordinates are all different.

    Rule:
    - Identify the three nested rectangular ring layers from outermost to innermost and keep each ring's height and width.
    - Read the three interior marker colors from left to right.
    - Build the output by placing the outer, middle, and inner rings horizontally from left to right with one background column between neighbors.
    - Repaint the outer ring with the first marker color, the middle ring with the second marker color, and the inner ring with the third marker color.

    Key insight:
    - The geometric transformation and the recoloring cue are separate: the rings are spatially unpacked, while the new colors come from the markers' horizontal ordering inside the smallest enclosed region.
    """
    bg, ring_bboxes, marker_order = _extract_structure(grid)
    return _assemble_output(bg, ring_bboxes, marker_order)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    outer_height = rng.randint(13, 19)
    margin_y_max = min(5, (30 - outer_height) // 2)
    margin_x_max = min(8, (30 - _OUTER_WIDTH) // 2)
    if margin_y_max < 1 or margin_x_max < 1:
        return None

    margin_y = rng.randint(1, margin_y_max)
    margin_x = rng.randint(1, margin_x_max)
    height = outer_height + 2 * margin_y
    width = _OUTER_WIDTH + 2 * margin_x
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), _BG, dtype=int)
    ring_colors = rng.sample(_COLOR_POOL, 3)

    top = margin_y
    left = margin_x
    ring_heights = [outer_height, outer_height - 4, outer_height - 8]
    ring_widths = [_OUTER_WIDTH, _OUTER_WIDTH - 4, _OUTER_WIDTH - 8]

    for layer, color in enumerate(ring_colors):
        offset = 2 * layer
        _draw_rect_ring(
            grid,
            top + offset,
            left + offset,
            ring_heights[layer],
            ring_widths[layer],
            color,
        )

    inner_top = top + 4
    inner_left = left + 4
    marker_rows = list(range(inner_top + 1, inner_top + ring_heights[2] - 1))
    marker_cols = [inner_left + 1, inner_left + 2, inner_left + 3]
    if len(marker_rows) < 3:
        return None

    chosen_rows = rng.sample(marker_rows, 3)
    rng.shuffle(chosen_rows)
    marker_colors = ring_colors[:]
    rng.shuffle(marker_colors)

    for row, col, color in zip(chosen_rows, marker_cols, marker_colors):
        grid[row, col] = int(color)

    bg, ring_bboxes, ordered = _extract_structure(grid)
    expected = _assemble_output(bg, ring_bboxes, ordered)
    actual = main(grid)
    assert np.array_equal(actual, expected)
    assert set(ordered) == set(ring_colors), "marker colors must match ring colors"
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries")
