from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, coords_where, get_bg, in_bounds


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    length = max(abs(dr), abs(dc))

    assert length > 0
    assert abs(dr) in (0, length)
    assert abs(dc) in (0, length)

    sr = _sign(dr)
    sc = _sign(dc)
    return [(r0 + i * sr, c0 + i * sc) for i in range(length + 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """Draw the green perimeter of the parallelogram whose opposite corners are the two marker pixels.

    Input structure
    - The input is a same-size canvas with a dominant background and exactly two foreground marker pixels of one color.
    - The marker-to-marker offset has both row and column components, with unequal magnitudes.

    Rule
    - Decompose the vector between markers into a 45-degree diagonal vector plus the remaining horizontal/vertical vector.
    - Treat those two vectors as adjacent sides of a parallelogram, with the markers as opposite corners.
    - Paint the four sides green while preserving the marker pixels.

    Key insight
    - The outline is not a direct line interpolation; it is the perimeter of the parallelogram induced by the diagonal and axis-aligned parts of the marker offset.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    marker_colors = colors_present(g, exclude={bg})
    assert len(marker_colors) == 1

    marker_color = marker_colors[0]
    marker_coords = coords_where(g == marker_color)
    assert marker_coords.shape == (2, 2)
    p = tuple(map(int, marker_coords[0]))
    q = tuple(map(int, marker_coords[1]))

    dr = q[0] - p[0]
    dc = q[1] - p[1]
    abs_dr = abs(dr)
    abs_dc = abs(dc)
    assert abs_dr > 0
    assert abs_dc > 0
    assert abs_dr != abs_dc

    diag_len = min(abs_dr, abs_dc)
    straight_len = abs(abs_dr - abs_dc)
    sr = _sign(dr)
    sc = _sign(dc)

    if abs_dr > abs_dc:
        straight_vec = (sr * straight_len, 0)
    else:
        straight_vec = (0, sc * straight_len)
    diag_vec = (sr * diag_len, sc * diag_len)

    straight_corner = (p[0] + straight_vec[0], p[1] + straight_vec[1])
    diag_corner = (p[0] + diag_vec[0], p[1] + diag_vec[1])

    out = g.copy()
    for a, b in ((p, straight_corner), (p, diag_corner), (straight_corner, q), (diag_corner, q)):
        for r, c in _segment_cells(a, b):
            assert in_bounds(g.shape, r, c)
            if g[r, c] != marker_color:
                out[r, c] = Color.GREEN

    return out

# --------------- generator (o5x) ---------------

# Canvas randint ranges and colors.
ROW_RAND_RANGE, COL_RAND_RANGE = (11, 14), (13, 15)
BACKGROUND_COLOR, MARKER_COLOR = Color.BLACK, Color.TEAL

# Offset geometry constraints; ranges are inclusive.
ABS_DR_RANGE, ABS_DC_RANGE = (2, 11), (6, 10)
DIAG_LEN_RANGE, STRAIGHT_LEN_RANGE = (2, 7), (2, 8)
MANHATTAN_RANGE, EDGE_PADDING = (12, 18), 2
SIGN_CHOICES = (-1, 1)

# Rejection-sampling guardrail.
MAX_GENERATION_ATTEMPTS = 300


def _gen_valid_offsets(h: int, w: int) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    max_abs_dr = min(ABS_DR_RANGE[1], h - EDGE_PADDING)
    max_abs_dc = min(ABS_DC_RANGE[1], w - EDGE_PADDING)
    for abs_dr in range(ABS_DR_RANGE[0], max_abs_dr + 1):
        for abs_dc in range(ABS_DC_RANGE[0], max_abs_dc + 1):
            if abs_dr == abs_dc:
                continue
            diag_len = min(abs_dr, abs_dc)
            straight_len = abs(abs_dr - abs_dc)
            manhattan = abs_dr + abs_dc
            if (
                DIAG_LEN_RANGE[0] <= diag_len <= DIAG_LEN_RANGE[1]
                and STRAIGHT_LEN_RANGE[0] <= straight_len <= STRAIGHT_LEN_RANGE[1]
                and MANHATTAN_RANGE[0] <= manhattan <= MANHATTAN_RANGE[1]
            ):
                offsets.append((abs_dr, abs_dc))
    return offsets


def _gen_marker_grid(h: int, w: int, dr: int, dc: int) -> np.ndarray:
    r_lo, r_hi = max(0, -dr), min(h - 1, h - 1 - dr)
    c_lo, c_hi = max(0, -dc), min(w - 1, w - 1 - dc)
    r0 = int(np.random.randint(r_lo, r_hi + 1))
    c0 = int(np.random.randint(c_lo, c_hi + 1))
    grid = np.full((h, w), BACKGROUND_COLOR, dtype=int)
    grid[r0, c0] = grid[r0 + dr, c0 + dc] = MARKER_COLOR
    return grid


def _gen_attempt() -> np.ndarray | None:
    h, w = int(np.random.randint(*ROW_RAND_RANGE)), int(np.random.randint(*COL_RAND_RANGE))
    offsets = _gen_valid_offsets(h, w)
    if not offsets:
        return None
    abs_dr, abs_dc = offsets[int(np.random.randint(len(offsets)))]
    dr = abs_dr * int(np.random.choice(SIGN_CHOICES))
    dc = abs_dc * int(np.random.choice(SIGN_CHOICES))
    grid = _gen_marker_grid(h, w, dr, dc)
    main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid = _gen_attempt()
            if grid is not None:
                return grid
        except (AssertionError, ValueError):
            pass
    raise RuntimeError("failed to generate valid cb227835 input")
