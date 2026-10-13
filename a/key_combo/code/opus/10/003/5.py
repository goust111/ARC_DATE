import itertools
import random

import numpy as np


BG = 0
MARKER = 8
GREY = 5
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 9)

MIN_H, MAX_H = 12, 18
MIN_W, MAX_W = 12, 18
MIN_SEGMENTS, MAX_SEGMENTS = 4, 6
MIN_SPAN, MAX_SPAN = 3, 7
MAX_ATTEMPTS = 500

THREE_CYCLE_TEMPLATES = (
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 0)),
    ((0, 1), (0, 2), (1, 0), (1, 1), (2, 2), (2, 0)),
    ((0, 0), (0, 2), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (0, 2), (1, 1), (1, 2), (2, 0), (2, 1)),
)


def _pair_combos(values: list[int], pair_count: int, min_span: int, max_span: int) -> list[tuple[tuple[int, int], ...]]:
    if pair_count == 0:
        return [()]
    vals = sorted(int(v) for v in values)
    pairs = [(a, b) for a, b in itertools.combinations(vals, 2) if min_span <= b - a <= max_span]
    combos: list[tuple[tuple[int, int], ...]] = []
    for combo in itertools.combinations(pairs, pair_count):
        used = [x for pair in combo for x in pair]
        if len(set(used)) == len(used):
            combos.append(tuple((int(a), int(b)) for a, b in combo))
    return combos


def _segment_key(seg: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = seg
    return int(r0 + r1), int(c0 + c1)


def _paint_segment(out: np.ndarray, seg: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, seg)
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        out[r0, lo + 1 : hi] = int(color)
    else:
        lo, hi = sorted((r0, r1))
        out[lo + 1 : hi, c0] = int(color)


def _interior_cells(seg: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, seg)
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        return [(r0, c) for c in range(lo + 1, hi)]
    lo, hi = sorted((r0, r1))
    return [(r, c0) for r in range(lo + 1, hi)]


def _ordered_segments(segments: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    return sorted(segments, key=_segment_key)


def _markers_are_isolated(coords: list[tuple[int, int]]) -> bool:
    for (r0, c0), (r1, c1) in itertools.combinations(coords, 2):
        if abs(int(r0) - int(r1)) <= 1 and abs(int(c0) - int(c1)) <= 1:
            return False
    return True


def _extract_palette(grid: np.ndarray) -> tuple[int, list[int], list[int], int]:
    h, w = grid.shape
    palette_cols_rev: list[int] = []
    palette_rev: list[int] = []
    c = w - 1
    while c >= 0:
        column = grid[:, c]
        if np.all(column == column[0]) and int(column[0]) not in (BG, GREY, MARKER):
            palette_cols_rev.append(int(c))
            palette_rev.append(int(column[0]))
            c -= 1
            continue
        break
    assert 2 <= len(palette_cols_rev) <= 3, "expected 2-3 right-edge palette columns"
    palette_cols = list(reversed(palette_cols_rev))
    palette = list(reversed(palette_rev))
    grey_col = palette_cols[0] - 1
    assert grey_col >= 0, "expected a grey run-length column left of the palette"
    grey_column = grid[:, grey_col]
    run_len = 0
    while run_len < h and int(grey_column[run_len]) == GREY:
        run_len += 1
    assert 1 <= run_len <= 3, "grey run length must be 1-3"
    assert np.all(grey_column[:run_len] == GREY), "grey marker must be top-aligned"
    assert np.all(grey_column[run_len:] == BG), "grey column below the marker must be black"
    return int(grey_col), palette_cols, palette, int(run_len)


def _extract_segments(grid: np.ndarray, stop_col: int) -> list[tuple[int, int, int, int]]:
    h, _ = grid.shape
    marker_area = grid[:, :stop_col]
    row_counts = np.count_nonzero(marker_area == MARKER, axis=1)
    col_counts = np.count_nonzero(marker_area == MARKER, axis=0)
    assert set(map(int, row_counts)).issubset({0, 2}), "each marker row must contain 0 or 2 markers"
    assert set(map(int, col_counts)).issubset({0, 2}), "each marker column must contain 0 or 2 markers"

    segments: list[tuple[int, int, int, int]] = []
    for r in range(h):
        cols = np.flatnonzero(marker_area[r] == MARKER)
        if len(cols) == 2:
            c0, c1 = map(int, cols)
            segments.append((int(r), c0, int(r), c1))
    for c in range(stop_col):
        rows = np.flatnonzero(marker_area[:, c] == MARKER)
        if len(rows) == 2:
            r0, r1 = map(int, rows)
            segments.append((r0, int(c), r1, int(c)))
    return segments


def _render_output(
    inp: np.ndarray,
    ordered_segments: list[tuple[int, int, int, int]],
    palette: list[int],
    run_len: int,
    grey_col: int,
    palette_cols: list[int],
) -> np.ndarray:
    out = inp.copy()
    for idx, seg in enumerate(ordered_segments):
        color = int(palette[(idx // run_len) % len(palette)])
        _paint_segment(out, seg, color)
    out[:, grey_col] = BG
    out[:, palette_cols] = BG
    return out


def _color_sequence(segment_count: int, palette: list[int], run_len: int) -> list[int]:
    return [int(palette[(idx // run_len) % len(palette)]) for idx in range(segment_count)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect every aligned marker pair, then recolor the resulting line segments by sorted order using the right-edge palette and grey run length.

    Input structure:
    - The grid uses black as background, isolated cyan marker pixels as segment endpoints, one top-aligned grey run-length column, and 2-3 full-height palette columns at the far right.
    - Every marker row contains either 0 or 2 cyan pixels, and every marker column contains either 0 or 2 cyan pixels.
    - The palette columns are solid colors read left-to-right; the grey column immediately to their left encodes the run length.

    Rule:
    - Find every row pair and column pair of cyan markers, treating each pair as one horizontal or vertical segment.
    - Sort those segments by midpoint row, then midpoint column.
    - Assign colors in that sorted order: the first K segments use the first palette color, the next K use the second palette color, and so on cyclically, where K is the grey segment height.
    - Fill each segment strictly between its endpoints with its assigned color, keep the cyan endpoints unchanged, and erase the grey column plus all palette columns to black.

    Key insight:
    - The visible line color is not stored on the line itself; it is determined jointly by the global segment ordering and the palette/run-length controls on the right edge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    grey_col, palette_cols, palette, run_len = _extract_palette(g)
    segments = _extract_segments(g, grey_col)
    assert segments, "expected at least one aligned marker pair"
    return _render_output(g, _ordered_segments(segments), palette, run_len, grey_col, palette_cols)


def _choose_run_len(segment_count: int, palette_count: int) -> int:
    candidates = [k for k in range(1, min(3, segment_count - 1) + 1)]
    wrap = [k for k in candidates if segment_count > palette_count * k]
    if wrap and random.random() < 0.5:
        return int(random.choice(wrap))
    if len(candidates) > 1 and random.random() < 0.45:
        non_one = [k for k in candidates if k != 1]
        if non_one:
            return int(random.choice(non_one))
    return int(random.choice(candidates))


def _choose_positions(limit: int, count: int, prefer_equal: bool) -> list[int] | None:
    if count == 2:
        spans = [span for span in range(MIN_SPAN, min(MAX_SPAN, limit - 1) + 1)]
        if not spans:
            return None
        span = int(random.choice(spans))
        start = random.randint(0, limit - span - 1)
        return [int(start), int(start + span)]

    gap_options = [(3, 3), (3, 4), (4, 3)]
    if prefer_equal:
        gap_options = [(3, 3), (4, 4), (3, 4), (4, 3)]
    feasible = [(g1, g2) for g1, g2 in gap_options if g1 + g2 <= limit - 1 and g1 + g2 <= MAX_SPAN + 1]
    if not feasible:
        return None
    g1, g2 = random.choice(feasible)
    start = random.randint(0, limit - g1 - g2 - 1)
    return [int(start), int(start + g1), int(start + g1 + g2)]


def _build_cycle_markers(rows: list[int], cols: list[int]) -> list[tuple[int, int]]:
    if len(rows) == 2:
        return [(rows[r], cols[c]) for r in range(2) for c in range(2)]
    template = random.choice(THREE_CYCLE_TEMPLATES)
    return [(int(rows[r]), int(cols[c])) for r, c in template]


def _build_input(
    height: int,
    width: int,
    active_count: int,
    palette: list[int],
    run_len: int,
    grey_col: int,
    palette_cols: list[int],
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    row_positions = _choose_positions(height, active_count, prefer_equal=active_count == 3 and random.random() < 0.7)
    col_positions = _choose_positions(grey_col, active_count, prefer_equal=active_count == 3 and random.random() < 0.7)
    if row_positions is None or col_positions is None:
        return None

    coords = _build_cycle_markers(row_positions, col_positions)
    if not _markers_are_isolated(coords):
        return None

    inp = np.full((height, width), BG, dtype=int)
    inp[:run_len, grey_col] = GREY
    for idx, color in enumerate(palette):
        inp[:, palette_cols[idx]] = int(color)
    for r, c in coords:
        inp[r, c] = MARKER

    segments = _extract_segments(inp, grey_col)
    if len(segments) != 2 * active_count:
        return None
    return inp, segments


def _semantic_checks(
    inp: np.ndarray,
    segments: list[tuple[int, int, int, int]],
    palette: list[int],
    run_len: int,
    grey_col: int,
    palette_cols: list[int],
) -> bool:
    ordered = _ordered_segments(segments)
    expected = _render_output(inp, ordered, palette, run_len, grey_col, palette_cols)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    if len(segments) <= run_len:
        return False

    used_fill_colors = {actual[r, c] for seg in ordered for r, c in _interior_cells(seg)}
    if len(used_fill_colors) < 2:
        return False
    if np.any(actual[:, grey_col] != BG) or np.any(actual[:, palette_cols] != BG):
        return False

    correct_seq = _color_sequence(len(ordered), palette, run_len)
    alt_orders = [
        sorted(segments, key=lambda s: (s[1] + s[3], s[0] + s[2])),
        list(reversed(ordered)),
    ]

    if not any(
        alt != ordered and not np.array_equal(_render_output(inp, alt, palette, run_len, grey_col, palette_cols), expected)
        for alt in alt_orders
    ):
        return False

    alt_run_lens = [k for k in range(1, min(3, len(ordered) - 1) + 1) if k != run_len]
    if alt_run_lens and not any(
        not np.array_equal(_render_output(inp, ordered, palette, alt_k, grey_col, palette_cols), expected)
        for alt_k in alt_run_lens
    ):
        return False

    reversed_palette = list(reversed(palette))
    if reversed_palette != palette and np.array_equal(
        _render_output(inp, ordered, reversed_palette, run_len, grey_col, palette_cols), expected
    ):
        return False

    if len(set(correct_seq)) < 2:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(MIN_H, MAX_H)
    palette_count = random.randint(2, 3)
    active_count = random.choice((2, 3))
    total_segments = 2 * active_count
    run_len = _choose_run_len(total_segments, palette_count)
    width = random.randint(MIN_W, MAX_W)
    marker_width = width - 1 - palette_count
    if marker_width < 6:
        return None

    grey_col = marker_width
    palette_cols = list(range(grey_col + 1, width))
    palette = random.sample(PALETTE_COLORS, palette_count)

    built = _build_input(
        height,
        width,
        active_count,
        palette,
        run_len,
        grey_col,
        palette_cols,
    )
    if built is None:
        return None
    inp, segments = built

    if not _semantic_checks(inp, segments, palette, run_len, grey_col, palette_cols):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input")
