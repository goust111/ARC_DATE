from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

SEGMENT_COLORS = (GREEN, BLUE, RED)
X_PALETTE = (BLUE, RED, GREEN, YELLOW)

# 5x5 X positions, using the canonical (r, c) convention from arc_utils.py.
X_MASK = np.zeros((5, 5), dtype=bool)
for _i in range(5):
    X_MASK[_i, _i] = True
    X_MASK[_i, 4 - _i] = True

# Folding map from the 5x5 X source positions to the 3x3 folded tile.
TILE_TO_SOURCE = {
    (0, 0): (0, 0),
    (0, 1): (1, 1),
    (0, 2): (0, 4),
    (1, 0): (3, 1),
    (1, 1): (2, 2),
    (1, 2): (1, 3),
    (2, 0): (4, 0),
    (2, 1): (3, 3),
    (2, 2): (4, 4),
}

TILE_COLUMN_SOURCES = (
    ((0, 0), (3, 1), (4, 0)),
    ((1, 1), (2, 2), (3, 3)),
    ((0, 4), (1, 3), (4, 4)),
)

VALID_COUNT_TRIPLES = [
    counts
    for counts in itertools.product((1, 2, 3), repeat=3)
    if sum(counts) >= 5
]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Inclusive bbox in the same `(r0, c0, r1, c1)` format as arc_utils.py."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _scan_horizontal_runs(grid: np.ndarray, color: int, min_len: int) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    height, width = map(int, grid.shape)
    for r in range(height):
        c = 0
        while c < width:
            if int(grid[r, c]) != int(color):
                c += 1
                continue
            start = c
            while c < width and int(grid[r, c]) == int(color):
                c += 1
            end = c - 1
            if end - start + 1 >= min_len:
                runs.append((r, start, end))
    return runs


def _find_segments(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    segments: list[tuple[int, int, int, int]] = []
    for color in SEGMENT_COLORS:
        runs = _scan_horizontal_runs(grid, color, min_len=3)
        assert len(runs) == 1, f"expected exactly one long run for color {color}, got {runs}"
        row, c0, c1 = runs[0]
        segments.append((int(row), int(c0), int(c1 - c0 + 1), int(color)))

    rows = [row for row, _c0, _length, _color in segments]
    starts = [c0 for _row, c0, _length, _color in segments]
    lengths = [length for _row, _c0, length, _color in segments]

    assert rows == sorted(rows), "segments must appear from top to bottom"
    assert len(set(starts)) == 1, "segments must share one left anchor"
    assert len(set(lengths)) == 1, "segments must share one length"
    return segments


def _fold_column_counts(window: np.ndarray) -> tuple[int, int, int] | None:
    if window.shape != (5, 5):
        return None
    if int(window[2, 2]) == BLACK:
        return None
    if np.count_nonzero(window) < 5:
        return None
    if np.any(window[~X_MASK] != BLACK):
        return None

    counts = []
    for src_coords in TILE_COLUMN_SOURCES:
        count = sum(int(window[r, c]) != BLACK for r, c in src_coords)
        counts.append(int(count))
    if any(count < 1 or count > 3 for count in counts):
        return None
    return tuple(counts)


def _find_x_region(
    grid: np.ndarray, segments: list[tuple[int, int, int, int]]
) -> tuple[tuple[int, int, int, int], tuple[int, int, int]]:
    stripped = np.array(grid, copy=True)
    for row, c0, length, _color in segments:
        stripped[row, c0 : c0 + length] = BLACK

    height, width = map(int, stripped.shape)
    candidates: list[tuple[int, int, tuple[int, int, int], int]] = []
    for r0 in range(height - 4):
        for c0 in range(width - 4):
            window = stripped[r0 : r0 + 5, c0 : c0 + 5]
            counts = _fold_column_counts(window)
            if counts is None:
                continue
            candidates.append((r0, c0, counts, int(np.count_nonzero(window))))

    assert candidates, "failed to locate the 5x5 X region"
    candidates.sort(key=lambda item: (-item[3], item[0], item[1]))
    best_r0, best_c0, best_counts, _best_area = candidates[0]
    return (best_r0, best_c0, best_r0 + 4, best_c0 + 4), best_counts


def _parse_input(
    grid: np.ndarray,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int], list[tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    segments = _find_segments(g)
    x_bbox, counts = _find_x_region(g, segments)
    return x_bbox, counts, segments


def _paint_wedge(
    out: np.ndarray,
    row: int,
    c0: int,
    length: int,
    extent: int,
    color: int,
) -> None:
    height, width = map(int, out.shape)

    for dist in range(1, extent + 1):
        rr = row - dist
        if rr < 0:
            break
        fill_width = length + dist
        out[rr, c0 : min(c0 + fill_width, width)] = color

    for dist in range(1, extent + 1):
        fill_width = length - dist
        if fill_width <= 0:
            break
        rr = row + dist
        if rr >= height:
            break
        out[rr, c0 : min(c0 + fill_width, width)] = color


def _render_output(
    grid: np.ndarray,
    x_bbox: tuple[int, int, int, int],
    counts: tuple[int, int, int],
    segments: list[tuple[int, int, int, int]],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    r0, c0, r1, c1 = x_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = BLACK

    for (row, seg_c0, length, _seg_color), extent in zip(segments, counts):
        _paint_wedge(out, row, seg_c0, length, extent, ORANGE)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold a partial 5x5 X into column counts and use those counts to gate three orange wedges around horizontal segments.

    Input structure:
    - The background is black.
    - One 5x5 region contains foreground pixels only on the two diagonals of an X; some of those
      diagonal cells may be missing, but the center is present and at least five diagonal cells remain.
    - Elsewhere, three horizontal line segments share one left anchor and one length, with colors
      green, blue, and red from top to bottom.

    Rule:
    - Fold the 5x5 X into a 3x3 tile by mapping each diagonal arm onto the matching square edge and
      keeping the center at the tile center.
    - Count the non-black cells in folded tile columns 0, 1, and 2, yielding `c0`, `c1`, and `c2`.
    - For the green, blue, and red segments respectively, fill `c` rows above with an orange
      left-aligned wedge whose width grows by one per row, and fill `c` rows below with an orange
      left-aligned wedge whose width shrinks by one per row until it would become non-positive.
    - Keep the original three segments unchanged, and erase the entire 5x5 X region to black.

    Key insight:
    - The 5x5 X is not output directly; it must first be folded into a 3x3 control tile, because only
      the folded tile's per-column occupancy determines how tall each segment's wedge should be.
    """
    x_bbox, counts, segments = _parse_input(grid)
    return _render_output(np.asarray(grid, dtype=int), x_bbox, counts, segments)


def _sample_tile_mask(counts: tuple[int, int, int]) -> np.ndarray:
    tile_mask = np.zeros((3, 3), dtype=bool)

    rows0 = random.sample((0, 1, 2), counts[0])
    for rr in rows0:
        tile_mask[rr, 0] = True

    tile_mask[1, 1] = True
    for rr in random.sample((0, 2), counts[1] - 1):
        tile_mask[rr, 1] = True

    rows2 = random.sample((0, 1, 2), counts[2])
    for rr in rows2:
        tile_mask[rr, 2] = True

    return tile_mask


def _tile_mask_to_x_window(tile_mask: np.ndarray) -> np.ndarray:
    window = np.zeros((5, 5), dtype=int)
    for (tr, tc), (sr, sc) in TILE_TO_SOURCE.items():
        if bool(tile_mask[tr, tc]):
            window[sr, sc] = random.choice(X_PALETTE)
    return window


def _required_gap(upper_count: int, lower_count: int, length: int) -> int:
    """Row gap that keeps one segment's lower wedge disjoint from the next segment's upper wedge."""
    return min(upper_count, length - 1) + lower_count + 1


def _segments_from_spec(
    top_row: int, gap1: int, gap2: int, length: int, seg_c0: int
) -> list[tuple[int, int, int, int]]:
    rows = (top_row, top_row + gap1, top_row + gap1 + gap2)
    colors = SEGMENT_COLORS
    return [(rows[i], seg_c0, length, colors[i]) for i in range(3)]


def _matches_spec(
    grid: np.ndarray,
    expected_bbox: tuple[int, int, int, int],
    expected_counts: tuple[int, int, int],
    expected_segments: list[tuple[int, int, int, int]],
) -> bool:
    try:
        x_bbox, counts, segments = _parse_input(grid)
    except AssertionError:
        return False
    return x_bbox == expected_bbox and counts == expected_counts and segments == expected_segments


def _output_matches_rule(
    out: np.ndarray,
    x_bbox: tuple[int, int, int, int],
    counts: tuple[int, int, int],
    segments: list[tuple[int, int, int, int]],
) -> bool:
    r0, c0, r1, c1 = x_bbox
    if np.any(out[r0 : r1 + 1, c0 : c1 + 1] != BLACK):
        return False

    height, width = map(int, out.shape)
    for (row, seg_c0, length, color), extent in zip(segments, counts):
        if not np.all(out[row, seg_c0 : seg_c0 + length] == color):
            return False

        for dist in range(1, extent + 1):
            rr = row - dist
            if rr < 0:
                break
            fill_width = length + dist
            if not np.all(out[rr, seg_c0 : min(seg_c0 + fill_width, width)] == ORANGE):
                return False

        for dist in range(1, extent + 1):
            fill_width = length - dist
            if fill_width <= 0:
                break
            rr = row + dist
            if rr >= height:
                break
            if not np.all(out[rr, seg_c0 : min(seg_c0 + fill_width, width)] == ORANGE):
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    counts = random.choice(VALID_COUNT_TRIPLES)
    tile_mask = _sample_tile_mask(counts)
    x_window = _tile_mask_to_x_window(tile_mask)

    x_r0 = random.randint(0, 2)
    x_c0 = random.randint(0, 2)

    length = random.choice((3, 3, 4, 5))
    gap1 = _required_gap(counts[0], counts[1], length) + random.randint(0, 1)
    gap2 = _required_gap(counts[1], counts[2], length) + random.randint(0, 1)

    top_row = max(x_r0 + 6, counts[0]) + random.randint(0, 1)

    max_count = max(counts)
    min_width = max(15, x_c0 + 5 + 6 + length + max_count)
    if min_width > 30:
        return None

    width = None
    seg_c0 = None
    for _ in range(60):
        candidate_width = random.randint(min_width, 30)
        lo = max(candidate_width // 2, x_c0 + 8)
        hi = candidate_width - (length + max_count)
        if lo <= hi:
            width = candidate_width
            seg_c0 = random.randint(lo, hi)
            break
    if width is None or seg_c0 is None:
        return None

    segments = _segments_from_spec(top_row, gap1, gap2, length, seg_c0)
    bottom_row = segments[-1][0]
    bottom_extent = min(counts[2], length - 1)
    height = bottom_row + bottom_extent + 1 + random.randint(0, 2)
    if height < 15 or height > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[x_r0 : x_r0 + 5, x_c0 : x_c0 + 5] = x_window
    for row, c0, seg_len, color in segments:
        grid[row, c0 : c0 + seg_len] = color

    x_bbox = (x_r0, x_c0, x_r0 + 4, x_c0 + 4)
    if not _matches_spec(grid, x_bbox, counts, segments):
        return None

    expected = _render_output(grid, x_bbox, counts, segments)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if not _output_matches_rule(actual, x_bbox, counts, segments):
        return None

    out_bbox = _bbox(actual == ORANGE)
    assert out_bbox is not None, "output should contain orange wedge cells"
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
