from __future__ import annotations

import numpy as np

from arc_utils import get_bg, in_bounds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw an outward square spiral from the lone marker using the two upper-left palette colors.

    Input structure:
    - The background is otherwise empty.
    - The two non-background pixels at the upper-left are palette controls:
      the left color is for horizontal spiral segments, and the right color is
      for vertical spiral segments.
    - The only other foreground pixel is the spiral's center marker, which is
      preserved in the output.

    Rule:
    - Start at the center marker and walk left, down, right, up, repeating.
    - Segment lengths are 2, 3, 4, ... cells.
    - Paint horizontal moves with the left palette color and vertical moves with
      the right palette color.
    - If the next requested step would leave the grid, stop without painting it.

    Key insight:
    The corner pixels are not part of the final picture; they encode the colors
    of a growing spiral whose geometry is fixed by the marker position and grid
    boundary.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape

    assert h > 0 and w > 1, "expected a grid with a two-pixel palette"
    assert int(g[0, 0]) != bg and int(g[0, 1]) != bg, "expected upper-left palette pixels"

    fg = np.argwhere(g != bg)
    assert fg.shape[0] == 3, "expected exactly two palette pixels and one marker"

    marker_coords = [
        (int(r), int(c))
        for r, c in fg
        if (int(r), int(c)) not in {(0, 0), (0, 1)}
    ]
    assert len(marker_coords) == 1, "expected exactly one non-palette marker"
    marker_r, marker_c = marker_coords[0]

    horizontal_color = int(g[0, 0])
    vertical_color = int(g[0, 1])
    marker_color = int(g[marker_r, marker_c])
    assert len({horizontal_color, vertical_color, marker_color}) == 3, "expected three distinct foreground roles"

    out = np.full_like(g, bg)
    out[marker_r, marker_c] = marker_color

    directions = [
        (0, -1, horizontal_color),
        (1, 0, vertical_color),
        (0, 1, horizontal_color),
        (-1, 0, vertical_color),
    ]
    r, c = marker_r, marker_c
    length = 2
    while True:
        dr, dc, color = directions[(length - 2) % len(directions)]
        for _ in range(length):
            nr, nc = r + dr, c + dc
            if not in_bounds(g.shape, nr, nc):
                return out
            out[nr, nc] = color
            r, c = nr, nc
        length += 1

# --------------- generator (o5x) ---------------

# Canvas and fixed colors.
GRID_SIZE_CHOICES = np.arange(11, 14)
BACKGROUND_COLOR, MARKER_COLOR = 0, 1
PALETTE_COLOR_CHOICES = np.arange(2, 10)
VALIDATION_COLORS = (2, 3)

# Spiral geometry constraints.
FIRST_SEGMENT_LENGTH = 2
SPIRAL_DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))
MIN_MARKER_EDGE_DISTANCE = 3
VALID_COMPLETE_SEGMENTS = range(4, 12)
VALID_TOTAL_SEGMENTS = range(5, 13)

# Retry budget.
MAX_GENERATION_ATTEMPTS = 300


def _valid_segment_counts(complete_segments: int, total_segments: int) -> bool:
    return complete_segments in VALID_COMPLETE_SEGMENTS and total_segments in VALID_TOTAL_SEGMENTS


def _gen_spiral_output(
    n: int,
    marker_r: int,
    marker_c: int,
    horizontal_color: int,
    vertical_color: int,
    marker_color: int,
) -> tuple[np.ndarray, int, int]:
    out = np.full((n, n), BACKGROUND_COLOR, dtype=int)
    out[marker_r, marker_c] = marker_color

    r, c = marker_r, marker_c
    length = FIRST_SEGMENT_LENGTH
    complete_segments = 0
    total_segments = 0

    while True:
        direction_index = (length - FIRST_SEGMENT_LENGTH) % len(SPIRAL_DIRECTIONS)
        dr, dc = SPIRAL_DIRECTIONS[direction_index]
        color = horizontal_color if direction_index % 2 == 0 else vertical_color
        painted = 0
        for _ in range(length):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n and 0 <= nc < n):
                if painted:
                    total_segments += 1
                return out, complete_segments, total_segments
            out[nr, nc] = color
            r, c = nr, nc
            painted += 1
        complete_segments += 1
        total_segments += 1
        length += 1


def _gen_marker_positions(n: int) -> list[tuple[int, int]]:
    positions = []
    marker_range = range(MIN_MARKER_EDGE_DISTANCE, n - MIN_MARKER_EDGE_DISTANCE)
    for r in marker_range:
        for c in marker_range:
            out, complete_segments, total_segments = _gen_spiral_output(
                n, r, c, VALIDATION_COLORS[0], VALIDATION_COLORS[1], MARKER_COLOR
            )
            if out[0, 0] == BACKGROUND_COLOR and out[0, 1] == BACKGROUND_COLOR:
                if _valid_segment_counts(complete_segments, total_segments):
                    positions.append((r, c))
    return positions


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice(GRID_SIZE_CHOICES))
    marker_positions = _gen_marker_positions(n)
    marker_r, marker_c = marker_positions[int(np.random.randint(len(marker_positions)))]
    horizontal_color, vertical_color = [
        int(x) for x in np.random.choice(PALETTE_COLOR_CHOICES, size=2, replace=False)
    ]

    out, complete_segments, total_segments = _gen_spiral_output(
        n,
        marker_r,
        marker_c,
        horizontal_color,
        vertical_color,
        MARKER_COLOR,
    )
    if not _valid_segment_counts(complete_segments, total_segments):
        return None

    inp = np.full_like(out, BACKGROUND_COLOR)
    inp[0, 0] = horizontal_color
    inp[0, 1] = vertical_color
    inp[marker_r, marker_c] = MARKER_COLOR

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 08573cc6 input")
