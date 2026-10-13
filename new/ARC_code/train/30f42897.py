from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
import random


def _border_cycle(shape: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = map(int, shape)
    assert height >= 2 and width >= 2, "expected a rectangular grid with a border cycle"

    coords: list[tuple[int, int]] = []
    for c in range(width):
        coords.append((0, c))
    for r in range(1, height):
        coords.append((r, width - 1))
    for c in range(width - 2, -1, -1):
        coords.append((height - 1, c))
    for r in range(height - 2, 0, -1):
        coords.append((r, 0))

    assert len(coords) == 2 * height + 2 * width - 4
    return coords


def _cyclic_contiguous_run(indices: list[int], period: int) -> tuple[int, int]:
    assert indices, "expected foreground border pixels"
    ordered = sorted(map(int, indices))
    assert len(ordered) == len(set(ordered)), "border indices must be unique"
    assert all(0 <= idx < int(period) for idx in ordered), "border index out of range"

    run_len = len(ordered)
    if run_len == 1:
        return ordered[0], 1

    gaps = [
        (ordered[(i + 1) % run_len] - ordered[i]) % int(period)
        for i in range(run_len)
    ]
    wide_gaps = [i for i, gap in enumerate(gaps) if gap != 1]
    assert len(wide_gaps) == 1, "foreground must form one contiguous cyclic border segment"

    start = ordered[(wide_gaps[0] + 1) % run_len]
    expected = {(start + k) % int(period) for k in range(run_len)}
    assert expected == set(ordered), "foreground segment is not contiguous"
    return int(start), int(run_len)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat the single colored border segment around the rectangular border, leaving one segment-length gap between copies.

    Input structure:
    - The background is the dominant color.
    - There is exactly one foreground color.
    - All foreground pixels form one contiguous segment on the rectangular border, possibly crossing the cycle start.

    Rule:
    - Unwrap the grid border clockwise into a cyclic path.
    - Let n be the length of the foreground segment.
    - Copy the segment with the same color every 2*n border positions around the cycle.

    Key insight:
    - The foreground segment is not extended by rows or columns directly; it is a motif on the perimeter cycle, repeated with an equal-sized gap.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 1, "expected exactly one foreground color"
    color = int(colors[0])

    border = _border_cycle(g.shape)
    border_index = {coord: i for i, coord in enumerate(border)}

    foreground = [tuple(map(int, coord)) for coord in np.argwhere(g == color)]
    assert foreground, "expected foreground pixels"
    assert all(coord in border_index for coord in foreground), "foreground must lie on the border"

    start, run_len = _cyclic_contiguous_run(
        [border_index[coord] for coord in foreground],
        len(border),
    )
    stride = 2 * run_len
    assert len(border) % stride == 0, "border cycle must tile by segment plus equal gap"

    out = g.copy()
    for offset in range(0, len(border), stride):
        for k in range(run_len):
            r, c = border[(start + offset + k) % len(border)]
            out[r, c] = color
    return out

# --------------- generator (c) ---------------

# Canvas size bounds observed in the task distribution.
GRID_HEIGHT_RANGE = (3, 10)
GRID_WIDTH_RANGE = (3, 15)

# Border tiling constraints: one painted segment, one equal gap, repeated.
MIN_SEG_LEN = 2
MIN_COPIES = 2
SEGMENT_GAP_FACTOR = 2

# Color policy: fixed background with one non-background foreground color.
BACKGROUND_COLOR = 8
FOREGROUND_COLORS = tuple(c for c in range(1, 10) if c != BACKGROUND_COLOR)

# Retry limit for rejection sampling.
MAX_GENERATION_ATTEMPTS = 300


def _valid_segment_lengths(perimeter: int) -> list[int]:
    """Return segment lengths whose segment+gap stride tiles the perimeter."""
    max_seg_len = perimeter // (SEGMENT_GAP_FACTOR * MIN_COPIES)
    valid = []
    for seg_len in range(MIN_SEG_LEN, max_seg_len + 1):
        stride = SEGMENT_GAP_FACTOR * seg_len
        if perimeter % stride == 0 and perimeter // stride >= MIN_COPIES:
            valid.append(seg_len)
    return valid


def _render_single_segment(
    height: int,
    width: int,
    bg: int,
    fg: int,
    start: int,
    seg_len: int,
) -> np.ndarray:
    """Build the source grid with only the original border segment visible."""
    border = _border_cycle((height, width))
    grid = np.full((height, width), bg, dtype=int)
    for j in range(seg_len):
        r, c = border[(start + j) % len(border)]
        grid[r, c] = fg
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    perimeter = 2 * height + 2 * width - 4

    valid_seg_lens = _valid_segment_lengths(perimeter)
    if not valid_seg_lens:
        return None

    seg_len = random.choice(valid_seg_lens)
    bg = BACKGROUND_COLOR
    fg = random.choice(FOREGROUND_COLORS)
    start = random.randint(0, perimeter - 1)
    grid = _render_single_segment(height, width, bg, fg, start, seg_len)

    try:
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(
        f"Failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts"
    )
