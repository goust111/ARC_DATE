from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

SEED_COLORS = np.array([BLUE, RED, GREEN, ORANGE], dtype=int)
GEN_RNG = np.random.default_rng()
GEN_ATTEMPTS = 400

HEIGHT_RANGE = (11, 19)
WIDTH_RANGE = (12, 19)
SEGMENT_COUNT_RANGE = (1, 4)
SEGMENT_WIDTH_RANGE = (1, 4)


def _paired_marker_columns(grid: np.ndarray) -> list[int]:
    h, w = grid.shape
    cols = [c for c in range(w) if int(grid[0, c]) == YELLOW and int(grid[h - 1, c]) == YELLOW]
    assert cols, "expected at least one paired yellow marker column"

    actual = {(int(r), int(c)) for r, c in np.argwhere(grid == YELLOW)}
    expected = {(0, int(c)) for c in cols} | {(h - 1, int(c)) for c in cols}
    assert actual == expected, "yellow pixels must appear only as paired top/bottom markers"
    return cols


def _contiguous_segments(cols: list[int]) -> list[tuple[int, int]]:
    assert cols
    ordered = sorted(int(c) for c in cols)
    segments: list[tuple[int, int]] = []
    start = ordered[0]
    end = ordered[0]
    for c in ordered[1:]:
        if c == end + 1:
            end = c
        else:
            segments.append((start, end))
            start = end = c
    segments.append((start, end))
    return segments


def _seed_pixels(grid: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    assert h >= 3 and w >= 3, "expected an interior area"
    assert not np.any(grid[1 : h - 1, :] == YELLOW), "yellow markers belong only on the top/bottom edges"
    assert not np.any(grid == GREY), "grey is reserved for output"

    seeds: list[tuple[int, int, int]] = []
    for r in range(1, h - 1):
        cols = np.where((grid[r] != BLACK) & (grid[r] != YELLOW))[0]
        assert len(cols) <= 1, "expected at most one internal seed per row"
        if len(cols) == 1:
            c = int(cols[0])
            seeds.append((r, c, int(grid[r, c])))
    assert seeds, "expected at least one internal seed"
    return seeds


def _phase_painted_color(seed_color: int, offset: int, positive_phase: bool) -> int:
    use_seed = (offset % 2 == 0)
    if not positive_phase:
        use_seed = not use_seed
    return int(seed_color) if use_seed else GREY


def _apply_rule(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    marker_cols = _paired_marker_columns(g)
    in_segment = np.zeros(w, dtype=bool)
    in_segment[np.asarray(marker_cols, dtype=int)] = True

    out = g.copy()
    for r, c0, seed_color in _seed_pixels(g):
        assert not in_segment[c0], "seed cannot lie on a yellow marker segment"
        positive_phase = True
        for c in range(c0, w):
            if c > c0 and bool(in_segment[c]) != bool(in_segment[c - 1]):
                positive_phase = not positive_phase
            out[r, c] = _phase_painted_color(seed_color, c - c0, positive_phase)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each internal seed rightward with an alternating ray whose phase flips inside yellow edge-bracketed column segments.

    Input structure:
    - The grid is mostly black.
    - Yellow pixels appear only on the top and bottom edges, aligned by column; contiguous aligned columns form 1-3 vertical marker segments.
    - The interior contains isolated non-black, non-yellow seed pixels, at most one per row, and every seed has only black cells to its right in the input.

    Rule:
    - Treat every contiguous run of paired yellow columns as a vertical marker segment spanning the full height.
    - For each interior seed, paint from the seed cell to the right edge of its row.
    - In positive phase, even offsets from the seed use the seed color and odd offsets use grey; whenever the ray crosses a segment boundary, the phase flips, so the color/grey alternation is swapped until the next boundary.
    - Preserve rows without seeds, and never overwrite the top or bottom yellow marker pixels.

    Key insight:
    - The yellow edge markers do not paint anything directly; they modulate the parity phase of each horizontal alternating ray whenever the ray enters or leaves a marked column strip.
    """
    return _apply_rule(grid)


def _sample_segments(width: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    count = int(rng.integers(*SEGMENT_COUNT_RANGE))
    widths = rng.integers(*SEGMENT_WIDTH_RANGE, size=count).astype(int)
    if count == 1 and int(widths[0]) == 1:
        widths[0] = 2

    min_total = 1 + 2 + int(widths.sum()) + 2 * (count - 1)
    if min_total > width:
        return None

    extra = int(width - min_total)
    extras = rng.multinomial(extra, np.full(count + 1, 1.0 / (count + 1)))
    left_margin = 1 + int(extras[0])
    internal_gaps = [2 + int(extras[i]) for i in range(1, count)]

    col = left_margin
    segments: list[tuple[int, int]] = []
    for i, seg_width in enumerate(widths):
        start = int(col)
        end = int(col + int(seg_width) - 1)
        segments.append((start, end))
        col = end + 1
        if i < count - 1:
            col += internal_gaps[i]

    assert width - col >= 2
    return segments


def _cross_count_interval(segments: list[tuple[int, int]], width: int, cross_count: int) -> tuple[int, int]:
    count = len(segments)
    assert 0 <= cross_count <= count
    if cross_count == count:
        return 0, int(segments[0][0] - 1)
    if cross_count == 0:
        return int(segments[-1][1] + 1), int(width - 2)
    idx = count - cross_count
    return int(segments[idx - 1][1] + 1), int(segments[idx][0] - 1)


def _count_crossed_segments(segments: list[tuple[int, int]], seed_col: int) -> int:
    crossed = 0
    for start, end in segments:
        assert not (start <= seed_col <= end), "seed cannot start inside a marker segment"
        if seed_col < start:
            crossed += 1
    return crossed


def _simple_alternating_row(seed_color: int, start_col: int, width: int) -> np.ndarray:
    row = np.zeros(width, dtype=int)
    for c in range(start_col, width):
        row[c] = seed_color if (c - start_col) % 2 == 0 else GREY
    return row


def _validate_input(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert g.ndim == 2 and max(g.shape) <= 30
    assert int(g.min()) >= 0 and int(g.max()) <= 9
    assert np.any(g == BLACK), "background must be black"

    marker_cols = _paired_marker_columns(g)
    segments = _contiguous_segments(marker_cols)
    assert 1 <= len(segments) <= 3
    for (s0, e0), (s1, e1) in zip(segments, segments[1:]):
        assert s1 - e0 >= 3, "marker segments must be separated by at least two blank columns"

    seed_rows = _seed_pixels(g)
    assert 2 <= len(seed_rows) <= len(SEED_COLORS)
    assert len({r for r, _c, _color in seed_rows}) == len(seed_rows)
    assert len({color for _r, _c, color in seed_rows}) == len(seed_rows)
    assert all(color in set(SEED_COLORS.tolist()) for _r, _c, color in seed_rows)

    crossed_counts: list[int] = []
    segment_changes = 0
    for r, c0, color in seed_rows:
        assert np.all(g[r, c0 + 1 :] == BLACK), "seed must have only black cells to its right in the input"
        crossed = _count_crossed_segments(segments, c0)
        crossed_counts.append(crossed)

        simple = _simple_alternating_row(color, c0, w)
        modulated = main(g)[r]
        if np.any(simple != modulated):
            segment_changes += 1

    assert 0 in crossed_counts, "need one ray that crosses zero marker segments"
    assert any(count > 0 for count in crossed_counts), "need one ray that crosses at least one marker segment"
    assert len(set(crossed_counts)) >= 2, "active rows must demonstrate different crossing counts"
    assert segment_changes >= 1, "marker segments must change at least one ray relative to simple alternation"

    out = main(g)
    assert out.shape == g.shape
    assert np.array_equal(out[0], g[0]) and np.array_equal(out[h - 1], g[h - 1]), "top/bottom markers must remain unchanged"
    assert any(np.any(out[r] != g[r]) for r, _c, _color in seed_rows), "output must change at least one active row"


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = int(rng.integers(*HEIGHT_RANGE))
    w = int(rng.integers(*WIDTH_RANGE))
    segments = _sample_segments(w, rng)
    if segments is None:
        return None

    segment_count = len(segments)
    max_rows = min(4, segment_count + 1)
    row_count = int(rng.integers(2, max_rows + 1))

    desired_counts = {0, segment_count}
    middle_counts = list(range(1, segment_count))
    rng.shuffle(middle_counts)
    for count in middle_counts:
        if len(desired_counts) >= row_count:
            break
        desired_counts.add(int(count))
    desired_count_list = list(desired_counts)
    rng.shuffle(desired_count_list)

    rows = rng.choice(np.arange(1, h - 1), size=len(desired_count_list), replace=False)
    colors = rng.choice(SEED_COLORS, size=len(desired_count_list), replace=False)

    grid = np.zeros((h, w), dtype=int)
    for start, end in segments:
        grid[0, start : end + 1] = YELLOW
        grid[h - 1, start : end + 1] = YELLOW

    for row_raw, color_raw, cross_count in zip(rows, colors, desired_count_list):
        row = int(row_raw)
        color = int(color_raw)
        lo, hi = _cross_count_interval(segments, w, int(cross_count))
        if lo > hi:
            return None
        seed_col = int(rng.integers(lo, hi + 1))
        grid[row, seed_col] = color

    try:
        _validate_input(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
