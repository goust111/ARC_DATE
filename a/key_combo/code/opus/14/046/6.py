import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5

GEN_ATTEMPTS = 400


def _analyze_input(grid: np.ndarray) -> tuple[list[dict], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    rows: list[dict] = []
    segment_lengths: set[int] = set()
    for r in range(h):
        row = g[r]
        non_black = np.flatnonzero(row != BLACK)
        if non_black.size == 0:
            continue

        assert non_black.size == w, "active rows must span the full width"
        assert set(map(int, np.unique(row))) <= {GREY, GREEN, RED}, "active rows use only grey/green/red"

        segment_cols = np.flatnonzero((row == GREEN) | (row == RED))
        assert segment_cols.size >= 2, "each active row must contain a colored middle segment"
        start = int(segment_cols[0])
        end = int(segment_cols[-1])
        assert segment_cols.size == end - start + 1, "segment cells must stay contiguous"

        left = row[:start]
        right = row[end + 1 :]
        if left.size:
            assert np.all(left == GREY), "left side must be grey"
        if right.size:
            assert np.all(right == GREY), "right side must be grey"

        red_cols = np.flatnonzero(row == RED)
        assert red_cols.size <= 1, "each row may contain at most one marker"
        if red_cols.size:
            marker_col = int(red_cols[0])
            assert start <= marker_col <= end
        else:
            marker_col = None

        rows.append(
            {
                "row": r,
                "start": start,
                "end": end,
                "marker_col": marker_col,
            }
        )
        segment_lengths.add(end - start + 1)

    assert rows, "expected at least one active row"
    assert len(segment_lengths) == 1, "all middle segments must share one length"

    marker_rows = [info for info in rows if info["marker_col"] is not None]
    assert 2 <= len(marker_rows) <= 4, "expected 2 to 4 markers"
    assert len({info["row"] for info in marker_rows}) == len(marker_rows)

    min_right = min(info["end"] for info in rows)
    return rows, min_right


def _align_rows(grid: np.ndarray, rows: list[dict], min_right: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    for info in rows:
        r = int(info["row"])
        shift = int(info["end"]) - int(min_right)
        assert shift >= 0
        row = g[r]
        if shift == 0:
            out[r] = row
            continue
        shifted = np.full(w, GREY, dtype=int)
        shifted[: w - shift] = row[shift:]
        out[r] = shifted
    return out


def _marker_positions(grid: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, pos)) for pos in np.argwhere(np.asarray(grid, dtype=int) == RED)]


def _cross_coverage(grid: np.ndarray, markers: list[tuple[int, int]]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    non_black = g != BLACK
    coverage = np.zeros(g.shape, dtype=int)
    for r, c in markers:
        coverage[r, non_black[r]] += 1
        coverage[non_black[:, c], c] += 1
    return coverage


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align each row's green segment by its right edge, then generate red crosses from the shifted markers and clear cross-over cells back to green.

    Input structure:
    - Black cells are background rows above and/or below one horizontal band of active rows.
    - Every active row spans the full width using only grey side cells plus one contiguous green segment; some rows replace exactly one green cell with a red marker.
    - All green segments have the same length, but their starting columns differ, so their right edges are misaligned.

    Rule:
    - Find the minimum right-edge column among the active rows' green/red segments.
    - Shift each active row left so its segment right edge lands on that column; pixels move with the row, left overflow is discarded, and newly exposed right cells become grey.
    - In the aligned grid, treat every red marker as a cross generator: repaint every non-black cell in its full row and full column red.
    - If a non-black cell belongs to the crosses of two or more different markers, recolor that cell green instead; black background never changes.

    Key insight:
    - The row-alignment step changes the absolute marker columns, so the later cross geometry and the green reset intersections can only be determined after shifting the rows first.
    """
    g = np.asarray(grid, dtype=int)
    rows, min_right = _analyze_input(g)
    aligned = _align_rows(g, rows, min_right)
    markers = _marker_positions(aligned)
    assert 2 <= len(markers) <= 4

    coverage = _cross_coverage(aligned, markers)
    out = aligned.copy()
    non_black = aligned != BLACK
    out[non_black & (coverage >= 1)] = RED
    out[non_black & (coverage >= 2)] = GREEN
    return out


def _build_aligned_row(width: int, start: int, length: int, marker_offset: int | None) -> np.ndarray:
    row = np.full(width, GREY, dtype=int)
    row[start : start + length] = GREEN
    if marker_offset is not None:
        row[start + marker_offset] = RED
    return row


def _validate_semantics(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        rows, min_right = _analyze_input(inp)
        active_rows = [info["row"] for info in rows]
        if active_rows != list(range(active_rows[0], active_rows[-1] + 1)):
            return False
        if active_rows[0] == 0 or active_rows[-1] == inp.shape[0] - 1:
            return False

        if len({info["end"] for info in rows}) < 2:
            return False
        if not any(info["end"] == min_right for info in rows):
            return False
        if not any(info["end"] > min_right for info in rows):
            return False

        aligned = _align_rows(inp, rows, min_right)
        aligned_markers = _marker_positions(aligned)
        aligned_cols = [c for _, c in aligned_markers]
        if len(set(aligned_cols)) != len(aligned_cols):
            return False

        coverage = _cross_coverage(aligned, aligned_markers)
        if not np.any((aligned != BLACK) & (coverage >= 2)):
            return False

        expected = aligned.copy()
        non_black = aligned != BLACK
        expected[non_black & (coverage >= 1)] = RED
        expected[non_black & (coverage >= 2)] = GREEN
        if not np.array_equal(out, expected):
            return False

        if np.any(out[aligned == BLACK] != BLACK):
            return False

        for r, c in aligned_markers:
            row_mask = non_black[r]
            for cc in np.flatnonzero(row_mask):
                want = GREEN if coverage[r, cc] >= 2 else RED
                if int(out[r, cc]) != want:
                    return False
            for rr in np.flatnonzero(non_black[:, c]):
                want = GREEN if coverage[rr, c] >= 2 else RED
                if int(out[rr, c]) != want:
                    return False
        return True
    except AssertionError:
        return False


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(10, 16)
    width = random.randint(10, 16)
    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    active_count = height - top_margin - bottom_margin
    if active_count < 5:
        return None

    mark_count = random.randint(2, min(4, active_count - 1))
    seg_len = random.randint(max(mark_count, 4), min(6, width - 3))

    suffix_len = random.randint(2, min(5, width - seg_len))
    anchor_end = width - suffix_len - 1
    start = anchor_end - seg_len + 1
    if start < 1:
        return None

    max_shift = suffix_len
    if max_shift < 1:
        return None

    marker_rows_rel = sorted(random.sample(range(active_count), mark_count))
    marker_offsets = random.sample(range(seg_len), mark_count)
    row_to_marker = {r: marker_offsets[i] for i, r in enumerate(marker_rows_rel)}

    possible_shifts = list(range(max_shift + 1))
    if len(possible_shifts) < 2:
        return None
    shift_values = [0] * active_count
    shifted_rows = random.sample(
        [r for r in range(active_count) if r not in marker_rows_rel[:1]],
        random.randint(1, max(1, active_count - 2)),
    )
    for r in range(active_count):
        if r in shifted_rows:
            shift_values[r] = random.randint(1, max_shift)
        else:
            shift_values[r] = 0
    if len(set(shift_values)) < 2:
        idx = random.choice([r for r in range(active_count) if shift_values[r] == 0])
        shift_values[idx] = random.randint(1, max_shift)

    grid = np.zeros((height, width), dtype=int)
    for rel_r in range(active_count):
        marker_offset = row_to_marker.get(rel_r)
        aligned_row = _build_aligned_row(width, start, seg_len, marker_offset)
        shift = shift_values[rel_r]
        input_row = np.full(width, GREY, dtype=int)
        input_row[shift:] = aligned_row[: width - shift]
        grid[top_margin + rel_r] = input_row

    out = main(grid)
    if not _validate_semantics(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
