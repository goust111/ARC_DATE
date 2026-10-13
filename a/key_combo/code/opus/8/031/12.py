from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
TEAL = 8

INPUT_COLORS = (BLACK, BLUE, RED, TEAL)
MAX_GENERATION_ATTEMPTS = 800
_SEEN_OUTPUT_SIGNATURES: set[tuple[tuple[int, int], bytes]] = set()


def _extract_segments(grid: np.ndarray) -> tuple[list[int], list[int], list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert np.isin(g, INPUT_COLORS).all(), "input may only use black, blue, red, and teal"

    marker_rows: list[int] = []
    blue_cols: list[int] = []
    segments: list[np.ndarray] = []

    for r in range(g.shape[0]):
        left_cols = np.flatnonzero(g[r] == BLUE)
        right_cols = np.flatnonzero(g[r] == TEAL)
        assert (len(left_cols) == 0) == (len(right_cols) == 0), "marker rows must contain both controls"
        if len(left_cols) == 0:
            continue

        assert len(left_cols) == 1 and len(right_cols) == 1, "each marker row must have one blue and one teal marker"
        left = int(left_cols[0])
        right = int(right_cols[0])
        assert left < right - 1, "control pixels must bracket at least one cell"

        row = g[r]
        keep = np.ones(row.shape[0], dtype=bool)
        keep[[left, right]] = False
        assert np.isin(row[keep], (BLACK, RED)).all(), "only the control pixels may use blue or teal"

        segment = row[left + 1 : right].copy()
        assert np.isin(segment, (BLACK, RED)).all(), "bracketed segments may only contain black or red"

        marker_rows.append(int(r))
        blue_cols.append(left)
        segments.append(segment)

    assert len(segments) >= 3, "expected at least three marked rows"
    width = int(segments[0].size)
    assert width > 0, "segments must be non-empty"
    assert all(int(seg.size) == width for seg in segments), "all bracketed segments must have equal width"
    return marker_rows, blue_cols, segments


def _stack_and_shear(segments: list[np.ndarray]) -> np.ndarray:
    n_rows = len(segments)
    width = int(segments[0].size)
    out = np.full((n_rows, width + n_rows - 1), GREY, dtype=int)
    for r, segment in enumerate(segments):
        shift = n_rows - 1 - r
        visible = width - shift
        if visible > 0:
            out[r, :visible] = segment[shift:]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract bracketed row segments, stack them, and shear the stack left over a grey background.

    Input structure:
    - The input is a black field with red/black content plus several marked rows.
    - Each marked row contains exactly one blue control pixel to the left of exactly one teal control pixel.
    - The cells strictly between the two controls form same-width red/black segments; unmarked rows contain no control pixels.

    Rule:
    - Scan rows from top to bottom and keep only the cells strictly between each blue/teal pair.
    - Stack those extracted segments in the same top-to-bottom order into a compact rectangle.
    - Treat the bottom stacked row as the anchor, shift each row above it left by its distance from the bottom row, clip anything that leaves the left edge, and fill every uncovered cell with grey.

    Key insight:
    - The blue/teal markers first define which horizontal slices matter and in what order; only after that stacked intermediate block is built does the bottom-anchored shear determine the final columns.
    """
    _marker_rows, _blue_cols, segments = _extract_segments(grid)
    return _stack_and_shear(segments)


def _sample_binary_row(width: int, red_lo: int, red_hi: int) -> np.ndarray:
    red_count = random.randint(red_lo, red_hi)
    row = np.zeros(width, dtype=int)
    cols = random.sample(range(width), red_count)
    row[cols] = RED
    return row


def _sample_segments(count: int, width: int) -> list[np.ndarray] | None:
    target_unique = min(3, count)
    for _ in range(200):
        segments = [_sample_binary_row(width, 1, width - 1) for _ in range(count)]
        unique_rows = {tuple(int(v) for v in seg) for seg in segments}
        if len(unique_rows) < target_unique:
            continue
        out = _stack_and_shear(segments)
        out_unique = {tuple(int(v) for v in row) for row in out}
        if len(out_unique) < target_unique:
            continue
        if RED not in out or BLACK not in out or GREY not in out:
            continue
        return segments
    return None


def _sample_marker_rows(height: int, count: int) -> list[int] | None:
    candidates = list(range(height - 1))
    for _ in range(200):
        rows = sorted(random.sample(candidates, count))
        if any(b - a > 1 for a, b in zip(rows, rows[1:])):
            return rows
    return None


def _sample_blue_cols(count: int, seg_width: int, grid_width: int) -> list[int] | None:
    max_blue = grid_width - seg_width - 2
    choices = list(range(max_blue + 1))
    if not choices:
        return None
    for _ in range(200):
        cols = [random.choice(choices) for _ in range(count)]
        if len(set(cols)) >= 2:
            return cols
    return None


def _build_input(
    height: int,
    width: int,
    marker_rows: list[int],
    blue_cols: list[int],
    segments: list[np.ndarray],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for r in range(height):
        grid[r] = _sample_binary_row(width, max(1, width // 4), min(width - 1, (3 * width) // 5))

    for out_r, (grid_r, blue_col) in enumerate(zip(marker_rows, blue_cols)):
        teal_col = blue_col + 1 + segments[out_r].size
        row = _sample_binary_row(width, max(1, width // 5), min(width - 1, width // 2 + 1))
        row[blue_col] = BLUE
        row[blue_col + 1 : teal_col] = segments[out_r]
        row[teal_col] = TEAL
        grid[grid_r] = row
    return grid


def _output_signature(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(int(x) for x in grid.shape), grid.tobytes()


def _validate_case(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        marker_rows, blue_cols, segments = _extract_segments(inp)
    except AssertionError:
        return False

    if not (4 <= len(marker_rows) <= 6):
        return False
    if len(set(blue_cols)) < 2:
        return False
    if not any(b - a > 1 for a, b in zip(marker_rows, marker_rows[1:])):
        return False

    seg_width = int(segments[0].size)
    if seg_width < len(segments):
        return False

    expected = _stack_and_shear(segments)
    if not np.array_equal(out, expected):
        return False
    if out.shape != (len(segments), seg_width + len(segments) - 1):
        return False
    if not np.all(out[:, seg_width:] == GREY):
        return False
    if not np.all(out[-1, :seg_width] != GREY):
        return False
    if not np.any(out[0] == GREY):
        return False
    if RED not in out or BLACK not in out or GREY not in out:
        return False

    in_sig = _output_signature(inp)
    out_sig = _output_signature(out)
    if in_sig == out_sig:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    marker_count = random.randint(4, 6)
    seg_width = random.randint(max(4, marker_count), 7)
    grid_width = random.randint(max(10, seg_width + 3), 14)
    grid_height = random.randint(max(10, marker_count + 3), 14)

    marker_rows = _sample_marker_rows(grid_height, marker_count)
    if marker_rows is None:
        return None

    blue_cols = _sample_blue_cols(marker_count, seg_width, grid_width)
    if blue_cols is None:
        return None

    segments = _sample_segments(marker_count, seg_width)
    if segments is None:
        return None

    inp = _build_input(grid_height, grid_width, marker_rows, blue_cols, segments)
    out = main(inp)
    if not _validate_case(inp, out):
        return None

    out_sig = _output_signature(out)
    if out_sig in _SEEN_OUTPUT_SIGNATURES:
        return None
    _SEEN_OUTPUT_SIGNATURES.add(out_sig)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 12")
