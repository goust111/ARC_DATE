import numpy as np

from arc_utils import bbox, colors_present, get_bg
import random


def _rectangular_frame_bbox(mask: np.ndarray):
    b = bbox(mask)
    if b is None:
        return None

    r0, c0, r1, c1 = b
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None

    expected = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if not np.array_equal(mask[r0 : r1 + 1, c0 : c1 + 1], expected):
        return None
    return b


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the inside of the rectangular frame with nested color bands read from the diagonal markers.

    Input structure
    - The background is empty, with one one-pixel-thick rectangular foreground frame.
    - Every other foreground color appears as a singleton marker pixel.
    - The markers form a contiguous down-right diagonal ending one pixel above-left of the frame.

    Rule
    - Read the marker colors in diagonal order from upper-left to lower-right.
    - Fill the frame interior with rectangular bands in that order: each color
      paints the next inward ring, and the final color fills the remaining center.
    - Preserve the frame, markers, and exterior background.

    Key insight
    - The diagonal singleton markers are an outer-to-inner palette for the empty framed region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    frame_candidates = []
    for color in colors_present(g, exclude={bg}):
        candidate = _rectangular_frame_bbox(g == color)
        if candidate is not None:
            frame_candidates.append((int(color), candidate))
    assert len(frame_candidates) == 1, "expected one rectangular frame color"

    frame_color, frame_bbox = frame_candidates[0]
    r0, c0, r1, c1 = frame_bbox
    assert np.all(g[r0 + 1 : r1, c0 + 1 : c1] == bg), "expected empty frame interior"

    markers = []
    for color in colors_present(g, exclude={bg, frame_color}):
        coords = np.argwhere(g == color)
        assert coords.shape == (1, 2), "expected each palette color to be one marker pixel"
        markers.append((int(coords[0, 0]), int(coords[0, 1]), int(color)))
    assert markers, "expected at least one marker"

    markers.sort(key=lambda item: (item[0], item[1]))
    for (r, c, _color), (next_r, next_c, _next_color) in zip(markers, markers[1:]):
        assert (next_r, next_c) == (r + 1, c + 1), "expected a contiguous diagonal marker run"
    assert (markers[-1][0], markers[-1][1]) == (r0 - 1, c0 - 1), (
        "expected marker diagonal to end above-left of the frame"
    )

    out = g.copy()
    marker_colors = [color for _r, _c, color in markers]
    for i, color in enumerate(marker_colors):
        rr0 = r0 + 1 + i
        cc0 = c0 + 1 + i
        rr1 = r1 - 1 - i
        cc1 = c1 - 1 - i
        assert rr0 <= rr1 and cc0 <= cc1, "expected enough interior for all marker colors"

        if i == len(marker_colors) - 1:
            assert np.all(out[rr0 : rr1 + 1, cc0 : cc1 + 1] == bg)
            out[rr0 : rr1 + 1, cc0 : cc1 + 1] = color
        else:
            assert np.all(out[rr0, cc0 : cc1 + 1] == bg)
            assert np.all(out[rr1, cc0 : cc1 + 1] == bg)
            assert np.all(out[rr0 : rr1 + 1, cc0] == bg)
            assert np.all(out[rr0 : rr1 + 1, cc1] == bg)
            out[rr0, cc0 : cc1 + 1] = color
            out[rr1, cc0 : cc1 + 1] = color
            out[rr0 : rr1 + 1, cc0] = color
            out[rr0 : rr1 + 1, cc1] = color

    return out

# --------------- generator (o5x) ---------------
# Bounds and palette.
GRID_MIN, GRID_MAX = 10, 20
INTERIOR_MIN, INTERIOR_MAX = 4, 12
PADDING_MIN, PADDING_MAX = 1, 3
MARKER_COUNT_MIN, MARKER_COUNT_MAX = 2, 5
MARKER_COLOR_MIN, MARKER_COLOR_MAX = 2, 9
BG_COLOR, FRAME_COLOR = 0, 1

# Frame geometry and retry budget.
FRAME_CELLS = 2
MAX_GENERATION_ATTEMPTS = 300


def _sample_inner_size(n_markers, pad):
    min_inner = max(INTERIOR_MIN, 2 * n_markers - 1, GRID_MIN - n_markers - FRAME_CELLS - pad)
    max_inner = min(INTERIOR_MAX, GRID_MAX - n_markers - FRAME_CELLS - pad)
    return None if min_inner > max_inner else random.randint(min_inner, max_inner)


def _draw_markers_and_bands(out, marker_colors, frame_bbox):
    r0, c0, r1, c1 = frame_bbox
    for i, color in enumerate(marker_colors):
        out[i, i] = color
        rr0 = r0 + 1 + i
        cc0 = c0 + 1 + i
        rr1 = r1 - 1 - i
        cc1 = c1 - 1 - i
        if i == len(marker_colors) - 1:
            out[rr0 : rr1 + 1, cc0 : cc1 + 1] = color
        else:
            out[rr0, cc0 : cc1 + 1] = color
            out[rr1, cc0 : cc1 + 1] = color
            out[rr0 : rr1 + 1, cc0] = color
            out[rr0 : rr1 + 1, cc1] = color


def _gen_attempt():
    n_markers = random.randint(MARKER_COUNT_MIN, MARKER_COUNT_MAX)
    pad_bottom = random.randint(PADDING_MIN, PADDING_MAX)
    pad_right = random.randint(PADDING_MIN, PADDING_MAX)

    inner_h = _sample_inner_size(n_markers, pad_bottom)
    if inner_h is None:
        return None
    inner_w = _sample_inner_size(n_markers, pad_right)
    if inner_w is None:
        return None
    height = n_markers + inner_h + FRAME_CELLS + pad_bottom
    width = n_markers + inner_w + FRAME_CELLS + pad_right
    out = np.zeros((height, width), dtype=int)
    marker_colors = random.sample(range(MARKER_COLOR_MIN, MARKER_COLOR_MAX + 1), n_markers)
    r0 = c0 = n_markers
    r1, c1 = r0 + inner_h + 1, c0 + inner_w + 1

    out[r0, c0 : c1 + 1] = FRAME_COLOR
    out[r1, c0 : c1 + 1] = FRAME_COLOR
    out[r0 : r1 + 1, c0] = FRAME_COLOR
    out[r0 : r1 + 1, c1] = FRAME_COLOR
    _draw_markers_and_bands(out, marker_colors, (r0, c0, r1, c1))

    inp = out.copy()
    inp[r0 + 1 : r1, c0 + 1 : c1] = BG_COLOR

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input():
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 99306f82 input")
