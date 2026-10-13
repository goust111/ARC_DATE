from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
GREEN = 3
MARK = 8

Segment = tuple[int, int, int, str]  # (row, red_start, red_end, marker_side)


def _find_segments(grid: np.ndarray) -> list[Segment]:
    """Return all marked horizontal red segments, one per occupied row."""
    h, w = grid.shape
    segments: list[Segment] = []

    for r in range(h):
        c = 0
        while c < w:
            if int(grid[r, c]) != RED:
                c += 1
                continue

            c0 = c
            while c + 1 < w and int(grid[r, c + 1]) == RED:
                c += 1
            c1 = c

            left_mark = c0 > 0 and int(grid[r, c0 - 1]) == MARK
            right_mark = c1 + 1 < w and int(grid[r, c1 + 1]) == MARK
            if left_mark or right_mark:
                assert left_mark != right_mark, "each segment must have exactly one marked end"
                segments.append((r, c0, c1, "left" if left_mark else "right"))
            c += 1

    assert segments, "expected at least one marked segment"
    rows = [segment[0] for segment in segments]
    assert len(rows) == len(set(rows)), "segments must occupy different rows"
    assert int(np.count_nonzero(grid == MARK)) == len(segments), "unexpected extra marker pixels"
    return sorted(segments, key=lambda item: item[0])


def _segment_span(segment: Segment, width: int) -> tuple[int, int]:
    """Return the inclusive full horizontal span after marker merge and ray extension."""
    _row, c0, c1, side = segment
    if side == "left":
        return c0 - 1, width - 1
    return 0, c1 + 1


def _row_interval(span_left: int, span_right: int, side: str, shrink: int) -> tuple[int, int] | None:
    total = span_right - span_left + 1
    width = total - shrink
    if width <= 0:
        return None
    if side == "left":
        return span_left, span_left + width - 1
    return span_right - width + 1, span_right


def _paint_interval(row: np.ndarray, left: int, right: int, color: int) -> None:
    view = row[left : right + 1]
    mask = view == BG
    view[mask] = int(color)


def _paint_wedge(out: np.ndarray, segment: Segment, color: int, direction: int) -> None:
    h, w = out.shape
    row, _c0, _c1, side = segment
    span_left, span_right = _segment_span(segment, w)

    for shrink in range(1, span_right - span_left + 1):
        rr = row + direction * shrink
        if rr < 0 or rr >= h:
            break
        interval = _row_interval(span_left, span_right, side, shrink)
        if interval is None:
            break
        left, right = interval
        _paint_interval(out[rr], left, right, color)


def _render_output_from_segments(shape: tuple[int, int], segments: list[Segment]) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), BG, dtype=int)

    for segment in segments:
        row, c0, c1, side = segment
        span_left, span_right = _segment_span(segment, w)
        if side == "left":
            out[row, c0 - 1 : c1 + 1] = RED
            out[row, c1 + 1 : span_right + 1] = GREEN
        else:
            out[row, span_left:c0] = GREEN
            out[row, c0 : c1 + 2] = RED

    for segment in sorted(segments, key=lambda item: item[0]):
        _paint_wedge(out, segment, RED, -1)
        _paint_wedge(out, segment, GREEN, 1)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Extend each marked horizontal segment away from its marker and use the resulting span as the base of aligned red/green wedges.

    Input structure:
    - The grid is a black background with 2 to 3 horizontal red segments on distinct rows.
    - Each segment has one cyan marker pixel immediately outside exactly one endpoint on the same row.
    - The marker can be on either the left or the right end, and that side encodes the segment's orientation choice.

    Rule:
    - For each red segment, detect whether its cyan marker sits just left of the segment or just right of it.
    - Replace the marker with red so the segment grows by one cell toward the marker, then extend a green ray from the opposite endpoint to the grid edge.
    - Treat the merged red part plus the green ray as one full horizontal span; above it paint a red wedge whose width shrinks by one per row and stays flush with the marker side, and below it paint the analogous green wedge.

    Key insight:
    - The cyan marker is erased in the output, but it determines both which endpoint launches the ray and which side of each wedge remains aligned while the opposite side recedes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    segments = _find_segments(g)
    return _render_output_from_segments(g.shape, segments)


def _sample_rows(height: int, n_segments: int) -> list[int] | None:
    candidates = list(range(3, height - 3))
    min_gap = 5 if n_segments == 2 else 4
    for _ in range(120):
        rows = sorted(random.sample(candidates, n_segments))
        if all(b - a >= min_gap for a, b in zip(rows, rows[1:])):
            return rows
    return None


def _sample_sides(n_segments: int) -> list[str]:
    sides = [random.choice(("left", "right")) for _ in range(n_segments)]
    if n_segments > 1 and len(set(sides)) == 1 and random.random() < 0.7:
        sides[random.randrange(n_segments)] = "left" if sides[0] == "right" else "right"
    return sides


def _sample_segment(row: int, width: int, side: str, compact: bool) -> Segment | None:
    if compact:
        length = random.choice((3, 3, 4))
        ray = random.choice((2, 2, 3))
    else:
        length = random.choice((3, 3, 4, 4, 5, 6))
        ray = random.choice((2, 2, 3, 3, 4))

    if length + ray + 1 > width:
        return None

    if side == "left":
        c1 = width - ray - 1
        c0 = c1 - length + 1
    else:
        c0 = ray
        c1 = c0 + length - 1

    if c0 - 1 < 0 or c1 + 1 >= width:
        return None
    return (row, c0, c1, side)


def _segment_output_cells(shape: tuple[int, int], segment: Segment) -> set[tuple[int, int]]:
    h, w = shape
    row, _c0, _c1, side = segment
    span_left, span_right = _segment_span(segment, w)
    cells = {(row, c) for c in range(span_left, span_right + 1)}

    for shrink in range(1, span_right - span_left + 1):
        interval = _row_interval(span_left, span_right, side, shrink)
        if interval is None:
            break
        left, right = interval
        up = row - shrink
        down = row + shrink
        if up >= 0:
            cells.update((up, c) for c in range(left, right + 1))
        if down < h:
            cells.update((down, c) for c in range(left, right + 1))
        if up < 0 and down >= h:
            break
    return cells


def _segments_overlap(shape: tuple[int, int], segments: list[Segment]) -> bool:
    occupied: set[tuple[int, int]] = set()
    for segment in segments:
        cells = _segment_output_cells(shape, segment)
        if occupied & cells:
            return True
        occupied |= cells
    return False


def _render_input(shape: tuple[int, int], segments: list[Segment]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    for row, c0, c1, side in segments:
        grid[row, c0 : c1 + 1] = RED
        marker_col = c0 - 1 if side == "left" else c1 + 1
        grid[row, marker_col] = MARK
    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        n_segments = 3 if random.random() < 0.35 else 2
        height = random.randint(18, 24) if n_segments == 3 else random.randint(14, 22)
        width = random.randint(12, 18)

        rows = _sample_rows(height, n_segments)
        if rows is None:
            continue

        segments: list[Segment] = []
        for idx, (row, side) in enumerate(zip(rows, _sample_sides(n_segments))):
            compact = n_segments == 3 and idx == 1
            segment = _sample_segment(row, width, side, compact=compact)
            if segment is None:
                segments = []
                break
            segments.append(segment)
        if len(segments) != n_segments:
            continue

        if _segments_overlap((height, width), segments):
            continue

        grid = _render_input((height, width), segments)
        try:
            parsed = _find_segments(grid)
            out = main(grid)
        except AssertionError:
            continue

        if parsed != sorted(segments, key=lambda item: item[0]):
            continue
        if np.array_equal(grid, out):
            continue
        return grid

    raise RuntimeError("failed to generate a valid marked-ray wedge input")
