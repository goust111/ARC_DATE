from __future__ import annotations

import numpy as np

from arc_utils import colors_present, draw_line, draw_line as _gen_draw_line, get_bg


def _segments(g: np.ndarray, bg: int) -> list[tuple[str, int, int, int, int, int]]:
    segments: list[tuple[str, int, int, int, int, int]] = []
    horizontal_owner = np.full(g.shape, -1, dtype=int)
    vertical_owner = np.full(g.shape, -1, dtype=int)

    for color in colors_present(g, exclude={bg}):
        coords = np.argwhere(g == int(color))
        assert coords.shape == (2, 2), "each foreground color must appear as exactly two marker pixels"

        (r0, c0), (r1, c1) = coords.tolist()
        assert r0 == r1 or c0 == c1, "same-color marker pixels must be row- or column-aligned"

        if r0 == r1:
            c_start, c_end = sorted((int(c0), int(c1)))
            for c in range(c_start, c_end + 1):
                assert horizontal_owner[int(r0), c] == -1, "horizontal segments must not overlap each other"
                horizontal_owner[int(r0), c] = int(color)
            segments.append(("horizontal", int(color), int(r0), c_start, int(r1), c_end))
            continue

        r_start, r_end = sorted((int(r0), int(r1)))
        for r in range(r_start, r_end + 1):
            assert vertical_owner[r, int(c0)] == -1, "vertical segments must not overlap each other"
            vertical_owner[r, int(c0)] = int(color)
        segments.append(("vertical", int(color), r_start, int(c0), r_end, int(c1)))

    assert segments, "expected at least one marker pair"
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect every aligned pair of same-colored marker pixels with a straight one-cell-thick segment.

    Input structure
    - The background is the dominant color.
    - Every foreground color occurs exactly twice.
    - The two pixels of each foreground color share either a row or a column.

    Rule
    - Fill the inclusive horizontal or vertical interval between each same-colored pair.
    - When a horizontal segment and a vertical segment cross, keep the vertical segment's color at the crossing.

    Key insight
    - The sparse input pixels are endpoints of colored line segments; segment orientation, not color id order, resolves crossings.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    segments = _segments(g, bg)

    out = g.copy()
    for orient, color, r0, c0, r1, c1 in segments:
        if orient == "horizontal":
            draw_line(out, r0, c0, end_r=r1, end_c=c1, color=color)

    for orient, color, r0, c0, r1, c1 in segments:
        if orient == "vertical":
            draw_line(out, r0, c0, end_r=r1, end_c=c1, color=color)

    return out

# --------------- generator (o5x) ---------------

Segment = dict[str, int]

# Canvas bounds; upper bounds are exclusive for np.random.randint.
ROWS_MIN, ROWS_MAX_EXCL = 10, 31
COLS_MIN, COLS_MAX_EXCL = 10, 21
INTERIOR_MIN, INTERIOR_PAD = 1, 1

# Segment count, orientation mix, span length, and crossing limits.
SEGMENT_COUNT = 5
H_COUNT_MIN, H_COUNT_MAX_EXCL = 2, 4
SPAN_MIN, SPAN_MAX = 4, 18
CROSSINGS_MIN, CROSSINGS_MAX = 2, 3

# Color sampling and retry budget.
COLOR_POOL = np.arange(1, 10)
GEN_ATTEMPTS = 300


def _gen_span(limit: int, include: int | None = None) -> tuple[int, int]:
    max_len = min(SPAN_MAX, limit - 2 * INTERIOR_PAD)
    length = int(np.random.randint(SPAN_MIN, max_len + 1))
    if include is None:
        start = int(np.random.randint(INTERIOR_MIN, limit - length))
    else:
        lo = max(INTERIOR_MIN, int(include) - length + 1)
        hi = min(int(include), limit - length - INTERIOR_PAD)
        start = int(np.random.randint(lo, hi + 1))
    return start, start + length - 1


def _gen_crossings(
    h_segments: list[Segment], v_segments: list[Segment]
) -> list[tuple[int, int]]:
    found: list[tuple[int, int]] = []
    for hi, h_seg in enumerate(h_segments):
        for vi, v_seg in enumerate(v_segments):
            if (
                h_seg["c0"] <= v_seg["c"] <= h_seg["c1"]
                and v_seg["r0"] <= h_seg["r"] <= v_seg["r1"]
            ):
                found.append((hi, vi))
    return found


def _sample_segments(rows: int, cols: int, n_h: int, n_v: int) -> tuple[list[Segment], list[Segment]]:
    colors = np.random.choice(COLOR_POOL, size=SEGMENT_COUNT, replace=False).astype(int).tolist()
    np.random.shuffle(colors)
    h_rows = np.random.choice(np.arange(INTERIOR_MIN, rows - INTERIOR_PAD), size=n_h, replace=False).astype(int).tolist()
    v_cols = np.random.choice(np.arange(INTERIOR_MIN, cols - INTERIOR_PAD), size=n_v, replace=False).astype(int).tolist()

    h_segments: list[Segment] = []
    v_segments: list[Segment] = []
    color_i = 0
    for r in h_rows:
        c0, c1 = _gen_span(cols)
        h_segments.append({"color": colors[color_i], "r": r, "c0": c0, "c1": c1})
        color_i += 1
    for c in v_cols:
        r0, r1 = _gen_span(rows)
        v_segments.append({"color": colors[color_i], "c": c, "r0": r0, "r1": r1})
        color_i += 1
    return h_segments, v_segments


def _align_crossing(h_seg: Segment, v_seg: Segment, rows: int, cols: int) -> None:
    h_seg["c0"], h_seg["c1"] = _gen_span(cols, include=v_seg["c"])
    v_seg["r0"], v_seg["r1"] = _gen_span(rows, include=h_seg["r"])


def _constrain_crossings(h_segments: list[Segment], v_segments: list[Segment], rows: int, cols: int) -> bool:
    """Force the original 2-3 crossing range, returning False on over-dense layouts."""
    if not _gen_crossings(h_segments, v_segments):
        hi = int(np.random.randint(0, len(h_segments)))
        vi = int(np.random.randint(0, len(v_segments)))
        _align_crossing(h_segments[hi], v_segments[vi], rows, cols)

    crossings = _gen_crossings(h_segments, v_segments)
    if len(crossings) == CROSSINGS_MIN - 1:
        used_h = {crossings[0][0]}
        used_v = {crossings[0][1]}
        hi_choices = [i for i in range(len(h_segments)) if i not in used_h]
        vi_choices = [i for i in range(len(v_segments)) if i not in used_v]
        hi = int(np.random.choice(hi_choices))
        vi = int(np.random.choice(vi_choices))
        _align_crossing(h_segments[hi], v_segments[vi], rows, cols)
    return len(_gen_crossings(h_segments, v_segments)) <= CROSSINGS_MAX


def _render_output(rows: int, cols: int, h_segments: list[Segment], v_segments: list[Segment]) -> np.ndarray:
    out = np.zeros((rows, cols), dtype=int)
    for h_seg in h_segments:
        _gen_draw_line(out, h_seg["r"], h_seg["c0"], end_r=h_seg["r"], end_c=h_seg["c1"], color=h_seg["color"])
    for v_seg in v_segments:
        _gen_draw_line(out, v_seg["r0"], v_seg["c"], end_r=v_seg["r1"], end_c=v_seg["c"], color=v_seg["color"])
    return out


def _endpoint_grid(rows: int, cols: int, h_segments: list[Segment], v_segments: list[Segment]) -> np.ndarray | None:
    """Render only segment endpoints, rejecting overlapping endpoint pixels."""
    inp = np.zeros((rows, cols), dtype=int)
    for h_seg in h_segments:
        for c in (h_seg["c0"], h_seg["c1"]):
            if inp[h_seg["r"], c] != 0:
                return None
            inp[h_seg["r"], c] = h_seg["color"]
    for v_seg in v_segments:
        for r in (v_seg["r0"], v_seg["r1"]):
            if inp[r, v_seg["c"]] != 0:
                return None
            inp[r, v_seg["c"]] = v_seg["color"]
    return inp


def _gen_attempt() -> np.ndarray | None:
    rows = int(np.random.randint(ROWS_MIN, ROWS_MAX_EXCL))
    cols = int(np.random.randint(COLS_MIN, COLS_MAX_EXCL))
    n_h = int(np.random.randint(H_COUNT_MIN, H_COUNT_MAX_EXCL))
    n_v = SEGMENT_COUNT - n_h
    h_segments, v_segments = _sample_segments(rows, cols, n_h, n_v)

    if not _constrain_crossings(h_segments, v_segments, rows, cols):
        return None

    out = _render_output(rows, cols, h_segments, v_segments)
    inp = _endpoint_grid(rows, cols, h_segments, v_segments)
    if inp is None:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 40853293 input")
