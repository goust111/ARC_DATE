from __future__ import annotations

import numpy as np

from arc_utils import get_bg
import random


def _find_crop_marker(g: np.ndarray, bg: int) -> tuple[int, int, int]:
    candidates: list[tuple[int, int, int]] = []
    height, width = map(int, g.shape)

    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        coords = np.argwhere(g == color)
        if coords.shape != (2, 2):
            continue

        same_row = int(coords[0, 0]) == int(coords[1, 0])
        same_col = int(coords[0, 1]) == int(coords[1, 1])
        if same_row == same_col:
            continue

        if same_row:
            if width % 2 != 0:
                continue
            half = width // 2
            positions = coords[:, 1]
            crop_axis = 1
        else:
            if height % 2 != 0:
                continue
            half = height // 2
            positions = coords[:, 0]
            crop_axis = 0

        in_first_half = bool(np.all(positions < half))
        in_second_half = bool(np.all(positions >= half))
        if in_first_half or in_second_half:
            side = 0 if in_first_half else 1
            candidates.append((color, crop_axis, side))

    assert len(candidates) == 1, "expected a unique two-pixel aligned crop marker"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop out the grid half containing the two aligned marker pixels, then erase the markers to background.

    Input structure
    - The grid has a dominant background color and one non-background marker color
      that appears as exactly two cells.
    - The marker cells are aligned either in one row or one column and both lie
      in the same half along that direction.
    - Other non-background pixels are content inside one or both halves.

    Rule
    - If the markers share a row, keep the left or right half whose columns
      contain them.
    - If the markers share a column, keep the top or bottom half whose rows
      contain them.
    - In the kept half, recolor the marker pixels to the background color.

    Key insight
    - The two marker pixels are not part of the picture; their alignment and
      half-position identify which half of the doubled canvas is the answer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    marker, crop_axis, side = _find_crop_marker(g, bg)
    height, width = map(int, g.shape)

    if crop_axis == 0:
        assert height % 2 == 0, "vertical crop requires an even height"
        half = height // 2
        out = g[side * half : (side + 1) * half, :].copy()
    else:
        assert width % 2 == 0, "horizontal crop requires an even width"
        half = width // 2
        out = g[:, side * half : (side + 1) * half].copy()

    out[out == marker] = bg
    return out

# --------------- generator (o5x) ---------------

# Colors.
BACKGROUND_COLOR, MARKER_COLOR = 8, 5
CONTENT_COLORS = (1, 2, 3, 4, 7, 9)

# Crop axes and canvas dimensions.
AXIS_VERTICAL, AXIS_HORIZONTAL = 0, 1
HORIZONTAL_HEIGHTS = tuple(range(4, 21, 2))
HORIZONTAL_HALF_WIDTHS = (4, 6, 8, 10, 12, 14)
VERTICAL_HALF_HEIGHTS = (2, 4, 6, 8, 10)
VERTICAL_WIDTHS = tuple(range(8, 31, 2))

# Sparse dot content.
MIN_CONTENT_DOTS = 3
MIN_CONTENT_DENSITY, MAX_CONTENT_DENSITY = 0.08, 0.25
ROW_GAP_MIN, ROW_GAP_MAX = 1, 3
COLUMN_SPAN_TRIES, MAX_COLUMN_SPANS = 30, 4

# Marker placement and retry policy.
MARKER_GAP, GENERATION_TRIES = 2, 300

Coord = tuple[int, int]
MarkerPair = tuple[Coord, Coord]


def _gen_content_rows(size: int, n_rows: int) -> list[int]:
    gaps: list[int] = []
    remaining_span = size - 1
    for i in range(n_rows - 1):
        remaining_gaps = n_rows - 2 - i
        hi = min(ROW_GAP_MAX, remaining_span - remaining_gaps)
        gap = random.randint(ROW_GAP_MIN, hi)
        gaps.append(gap)
        remaining_span -= gap
    span = sum(gaps)
    start = random.randint(0, size - 1 - span)
    rows = [start]
    for gap in gaps:
        rows.append(rows[-1] + gap)
    return rows


def _gen_columns(width: int, n_cols: int) -> list[int]:
    parity = random.randint(0, 1)
    track = list(range(parity, width, 2))
    if n_cols >= len(track):
        return track

    for _ in range(COLUMN_SPAN_TRIES):
        n_spans = random.randint(1, min(MAX_COLUMN_SPANS, n_cols))
        cuts = sorted(random.sample(range(1, n_cols), n_spans - 1))
        lengths = [b - a for a, b in zip([0] + cuts, cuts + [n_cols])]
        spans: list[tuple[int, int]] = []
        used: set[int] = set()
        ok = True
        for length in lengths:
            possible = [i for i in range(0, len(track) - length + 1)
                        if not any(j in used for j in range(i, i + length))]
            if not possible:
                ok = False
                break
            start = random.choice(possible)
            spans.append((start, length))
            used.update(range(start, start + length))
        if ok:
            cols = [track[i] for start, length in spans for i in range(start, start + length)]
            return sorted(cols)
    return sorted(random.sample(track, n_cols))


def _gen_sparse_half(height: int, width: int, color: int) -> np.ndarray | None:
    """Build one half with a randomized every-other-column sparse dot pattern."""
    area = height * width
    max_cols = (width + 1) // 2
    candidates = []
    for n_rows in range(1, height):
        for n_cols in range(1, max_cols + 1):
            count = n_rows * n_cols
            density = count / area
            if count >= MIN_CONTENT_DOTS and MIN_CONTENT_DENSITY <= density <= MAX_CONTENT_DENSITY:
                candidates.append((n_rows, n_cols))
    if not candidates:
        return None
    n_rows, n_cols = random.choice(candidates)
    rows = _gen_content_rows(height, n_rows)
    cols = _gen_columns(width, n_cols)
    half = np.full((height, width), BACKGROUND_COLOR, dtype=int)
    half[np.ix_(rows, cols)] = color
    return half


def _gen_marker_pairs(shape: tuple[int, int], crop_axis: int, side: int) -> list[MarkerPair]:
    """Return shuffled boundary marker pairs for the kept half."""
    height, width = shape
    pairs: list[MarkerPair] = []
    if crop_axis == AXIS_HORIZONTAL:
        half = width // 2
        c0, c1 = (0, half - 1) if side == 0 else (half, width - 1)
        for r in (0, height - 1):
            for c in range(c0, c1 - MARKER_GAP + 1):
                pairs.append(((r, c), (r, c + MARKER_GAP)))
    else:
        half = height // 2
        r0, r1 = (0, half - 1) if side == 0 else (half, height - 1)
        for c in (0, width - 1):
            for r in range(r0, r1 - MARKER_GAP + 1):
                pairs.append(((r, c), (r + MARKER_GAP, c)))
    random.shuffle(pairs)
    return pairs


def _choose_shapes(crop_axis: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return full input shape and output-half shape for the selected crop axis."""
    if crop_axis == AXIS_HORIZONTAL:
        height = random.choice(HORIZONTAL_HEIGHTS)
        half_width = random.choice(HORIZONTAL_HALF_WIDTHS)
        return (height, half_width * 2), (height, half_width)

    half_height = random.choice(VERTICAL_HALF_HEIGHTS)
    width = random.choice(VERTICAL_WIDTHS)
    return (half_height * 2, width), (half_height, width)


def _try_place_marker(inp: np.ndarray, expected_output: np.ndarray, full_shape: tuple[int, int],
                      crop_axis: int, side: int) -> np.ndarray | None:
    """Place the first valid marker pair whose crop matches the generated output."""
    for a, b in _gen_marker_pairs(full_shape, crop_axis, side):
        if inp[a] == BACKGROUND_COLOR and inp[b] == BACKGROUND_COLOR:
            inp[a] = MARKER_COLOR
            inp[b] = MARKER_COLOR
            if np.array_equal(main(inp), expected_output):
                return inp
            inp[a] = BACKGROUND_COLOR
            inp[b] = BACKGROUND_COLOR
    return None


def _gen_attempt() -> np.ndarray | None:
    crop_axis = random.randint(AXIS_VERTICAL, AXIS_HORIZONTAL)
    side = random.randint(0, 1)
    color = random.choice(CONTENT_COLORS)
    full_shape, out_shape = _choose_shapes(crop_axis)

    output = _gen_sparse_half(*out_shape, color)
    if output is None:
        return None

    other = _gen_sparse_half(*out_shape, color)
    if other is None:
        return None

    halves = (output, other) if side == 0 else (other, output)
    inp = np.concatenate(halves, axis=crop_axis)
    return _try_place_marker(inp, output, full_shape, crop_axis, side)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GENERATION_TRIES} attempts")
