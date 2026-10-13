from __future__ import annotations

from itertools import combinations
import hashlib

import numpy as np


BACKGROUND = 0
CROSS_COLOR = 4
INPUT_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
MIN_SEGMENT_LEN = 3
MAX_SEGMENT_LEN = 5
MIN_OBJECTS = 2
MAX_OBJECTS = 4
MIN_GRID_SIZE = 13
MAX_GRID_SIZE = 17
MAX_ATTEMPTS = 800

_SEEN_INPUT_SIGS: set[str] = set()
_SEEN_OUTPUT_SIGS: set[str] = set()


def _grid_sig(grid: np.ndarray) -> str:
    return hashlib.sha1(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _is_contiguous(values: np.ndarray) -> bool:
    vals = np.sort(np.asarray(values, dtype=int))
    return bool(vals.size > 0 and np.array_equal(vals, np.arange(vals[0], vals[-1] + 1)))


def _covers(interval: tuple[int, int], value: int) -> bool:
    return int(interval[0]) <= int(value) <= int(interval[1])


def _segment_record(color: int, coords: np.ndarray) -> dict:
    rows = np.unique(coords[:, 0])
    cols = np.unique(coords[:, 1])
    area = int(coords.shape[0])
    assert area >= 2, "each object must be a segment, not a single pixel"

    if len(rows) == 1:
        assert len(cols) == area and _is_contiguous(cols), "horizontal segment must be contiguous"
        return {
            "kind": "h",
            "color": int(color),
            "row": int(rows[0]),
            "col0": int(cols.min()),
            "col1": int(cols.max()),
            "length": area,
        }

    if len(cols) == 1:
        assert len(rows) == area and _is_contiguous(rows), "vertical segment must be contiguous"
        return {
            "kind": "v",
            "color": int(color),
            "col": int(cols[0]),
            "row0": int(rows.min()),
            "row1": int(rows.max()),
            "length": area,
        }

    raise AssertionError("foreground object must be a straight 1-pixel-thick segment")


def _analyze_grid(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.all((0 <= g) & (g <= 9)), "expected ARC color ids"

    colors = [int(color) for color in np.unique(g) if int(color) != BACKGROUND]
    horizontals: list[dict] = []
    verticals: list[dict] = []
    for color in colors:
        coords = np.argwhere(g == int(color))
        rec = _segment_record(color, coords)
        if rec["kind"] == "h":
            horizontals.append(rec)
        else:
            verticals.append(rec)

    assert MIN_OBJECTS <= len(horizontals) <= MAX_OBJECTS, "expected 2-4 horizontal segments"
    assert MIN_OBJECTS <= len(verticals) <= MAX_OBJECTS, "expected 2-4 vertical segments"
    assert len({seg["row"] for seg in horizontals}) == len(horizontals), "horizontal rows must be unique"
    assert len({seg["col"] for seg in verticals}) == len(verticals), "vertical cols must be unique"
    assert set(seg["color"] for seg in horizontals).isdisjoint(seg["color"] for seg in verticals), "color sets must be disjoint"

    for hseg in horizontals:
        for vseg in verticals:
            overlap_now = _covers((vseg["row0"], vseg["row1"]), hseg["row"]) and _covers((hseg["col0"], hseg["col1"]), vseg["col"])
            assert not overlap_now, "segments must not overlap before extension"

    horizontals.sort(key=lambda seg: seg["row"])
    verticals.sort(key=lambda seg: seg["col"])
    return horizontals, verticals


def _build_output(horizontals: list[dict], verticals: list[dict]) -> np.ndarray:
    rows = []
    for hseg in horizontals:
        for vseg in verticals:
            rows.append([int(hseg["color"]), CROSS_COLOR, int(vseg["color"])])
    return np.asarray(rows, dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """Extract all ray-crossing events and stack them as sorted color triplets.

    Input structure:
    - The background is black.
    - Foreground objects are several short monochromatic 1-pixel-thick segments.
    - Exactly 2-4 segments are horizontal on distinct rows, and exactly 2-4 segments are vertical on distinct columns.
    - Horizontal colors and vertical colors are disjoint, and no horizontal/vertical pair overlaps in the input.

    Rule:
    - Extend each horizontal segment across its full row while keeping its color.
    - Extend each vertical segment across its full column while keeping its color.
    - For every horizontal/vertical pair, record the crossing event using the horizontal color, yellow, and the vertical color.
    - Sort crossings by row first and then by column, which is equivalent to sorting horizontal rows ascending and vertical columns ascending.
    - Return the sorted crossings as a compact Kx3 table, one event per row.

    Key insight:
    - The output is not the extended cross picture; it is a lossily compressed event list whose row order is determined entirely by the crossing coordinates created after extension.
    """
    horizontals, verticals = _analyze_grid(grid)
    return _build_output(horizontals, verticals)


def _sample_spaced_positions(low: int, high: int, count: int, min_gap: int = 2) -> tuple[int, ...] | None:
    if low > high:
        return None
    choices = [
        combo
        for combo in combinations(range(low, high + 1), count)
        if all(combo[i + 1] - combo[i] >= min_gap for i in range(len(combo) - 1))
    ]
    if not choices:
        return None
    return tuple(map(int, choices[int(np.random.randint(len(choices)))]))


def _all_spans(limit: int) -> list[tuple[int, int]]:
    spans = []
    for length in range(MIN_SEGMENT_LEN, MAX_SEGMENT_LEN + 1):
        for start in range(0, limit - length + 1):
            spans.append((int(start), int(start + length - 1)))
    return spans


def _spans_within(low: int, high: int) -> list[tuple[int, int]]:
    width = int(high - low + 1)
    if width < MIN_SEGMENT_LEN:
        return []
    return [(low + start, low + end) for start, end in _all_spans(width)]


def _pair_relation(hseg: dict, vseg: dict) -> str:
    row_in = _covers((vseg["row0"], vseg["row1"]), hseg["row"])
    col_in = _covers((hseg["col0"], hseg["col1"]), vseg["col"])
    if row_in and col_in:
        return "invalid"
    if (not row_in) and col_in:
        return "row"
    if row_in and (not col_in):
        return "col"
    return "both"


def _render_grid(
    height: int,
    width: int,
    h_rows: tuple[int, ...],
    h_spans: list[tuple[int, int]],
    h_colors: list[int],
    v_cols: tuple[int, ...],
    v_spans: list[tuple[int, int]],
    v_colors: list[int],
) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    for row, span, color in zip(h_rows, h_spans, h_colors):
        c0, c1 = span
        if np.any(grid[int(row), int(c0) : int(c1) + 1] != BACKGROUND):
            return None
        grid[int(row), int(c0) : int(c1) + 1] = int(color)
    for col, span, color in zip(v_cols, v_spans, v_colors):
        r0, r1 = span
        if np.any(grid[int(r0) : int(r1) + 1, int(col)] != BACKGROUND):
            return None
        grid[int(r0) : int(r1) + 1, int(col)] = int(color)
    return grid


def _pick_colors(n_h: int, n_v: int) -> tuple[list[int], list[int]]:
    picked = list(map(int, np.random.choice(INPUT_COLORS, size=n_h + n_v, replace=False)))
    np.random.shuffle(picked)
    return picked[:n_h], picked[n_h:]


def _pick_dimensions() -> tuple[int, int]:
    height = int(np.random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE + 1))
    width = int(np.random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE + 1))
    return height, width


def _choose_h_spans(width: int, v_cols: tuple[int, ...], count: int, require_cover: bool) -> list[tuple[int, int]] | None:
    all_spans = _all_spans(width)
    if not all_spans:
        return None
    chosen = [all_spans[int(np.random.randint(len(all_spans)))] for _ in range(count)]
    if require_cover and not any(_covers(span, col) for span in chosen for col in v_cols):
        cover_options = [span for span in all_spans if any(_covers(span, col) for col in v_cols)]
        if not cover_options:
            return None
        chosen[int(np.random.randint(count))] = cover_options[int(np.random.randint(len(cover_options)))]
    return [(int(a), int(b)) for a, b in chosen]


def _choose_v_spans_from_options(
    options_by_col: list[list[tuple[int, int]]],
    require_include_row: bool,
    h_rows: tuple[int, ...],
) -> list[tuple[int, int]] | None:
    picked = [options[int(np.random.randint(len(options)))] for options in options_by_col]
    if require_include_row and not any(any(_covers(span, row) for row in h_rows) for span in picked):
        candidates = [
            (i, span)
            for i, options in enumerate(options_by_col)
            for span in options
            if any(_covers(span, row) for row in h_rows)
        ]
        if not candidates:
            return None
        idx, span = candidates[int(np.random.randint(len(candidates)))]
        picked[idx] = span
    return [(int(a), int(b)) for a, b in picked]


def _candidate_row_disjoint() -> tuple[np.ndarray, str] | None:
    n_h = int(np.random.randint(MIN_OBJECTS, MAX_OBJECTS + 1))
    n_v = int(np.random.randint(MIN_OBJECTS, MAX_OBJECTS + 1))
    height, width = _pick_dimensions()

    band_choices = []
    for cut in range(4, height - 4):
        top_rows = _sample_spaced_positions(1, cut - 1, n_h)
        bot_rows = _sample_spaced_positions(cut + 1, height - 2, n_h)
        top_spans = _spans_within(1, cut - 1)
        bot_spans = _spans_within(cut + 1, height - 2)
        if top_rows is not None and bot_spans:
            band_choices.append(("top", cut))
        if bot_rows is not None and top_spans:
            band_choices.append(("bottom", cut))
    if not band_choices:
        return None

    side, cut = band_choices[int(np.random.randint(len(band_choices)))]
    if side == "top":
        h_rows = _sample_spaced_positions(1, cut - 1, n_h)
        v_range_options = _spans_within(cut + 1, height - 2)
    else:
        h_rows = _sample_spaced_positions(cut + 1, height - 2, n_h)
        v_range_options = _spans_within(1, cut - 1)
    v_cols = _sample_spaced_positions(1, width - 2, n_v)
    if h_rows is None or v_cols is None or not v_range_options:
        return None

    h_spans = _choose_h_spans(width, v_cols, n_h, require_cover=True)
    if h_spans is None:
        return None
    v_spans = [v_range_options[int(np.random.randint(len(v_range_options)))] for _ in range(n_v)]
    h_colors, v_colors = _pick_colors(n_h, n_v)
    grid = _render_grid(height, width, h_rows, h_spans, h_colors, v_cols, v_spans, v_colors)
    if grid is None:
        return None
    return grid, "row_disjoint"


def _candidate_col_disjoint() -> tuple[np.ndarray, str] | None:
    n_h = int(np.random.randint(MIN_OBJECTS, MAX_OBJECTS + 1))
    n_v = int(np.random.randint(MIN_OBJECTS, MAX_OBJECTS + 1))
    height, width = _pick_dimensions()

    split_choices = []
    for cut in range(4, width - 4):
        left_cols = _sample_spaced_positions(1, cut - 1, n_v)
        right_cols = _sample_spaced_positions(cut + 1, width - 2, n_v)
        left_spans = _spans_within(1, cut - 1)
        right_spans = _spans_within(cut + 1, width - 2)
        if left_cols is not None and right_spans:
            split_choices.append(("left", cut))
        if right_cols is not None and left_spans:
            split_choices.append(("right", cut))
    if not split_choices:
        return None

    side, cut = split_choices[int(np.random.randint(len(split_choices)))]
    h_rows = _sample_spaced_positions(1, height - 2, n_h)
    if side == "left":
        v_cols = _sample_spaced_positions(1, cut - 1, n_v)
        h_span_options = _spans_within(cut + 1, width - 2)
    else:
        v_cols = _sample_spaced_positions(cut + 1, width - 2, n_v)
        h_span_options = _spans_within(1, cut - 1)
    if h_rows is None or v_cols is None or not h_span_options:
        return None

    h_spans = [h_span_options[int(np.random.randint(len(h_span_options)))] for _ in range(n_h)]
    all_v_spans = _all_spans(height)
    options_by_col = [all_v_spans[:] for _ in range(n_v)]
    v_spans = _choose_v_spans_from_options(options_by_col, require_include_row=True, h_rows=h_rows)
    if v_spans is None:
        return None
    h_colors, v_colors = _pick_colors(n_h, n_v)
    grid = _render_grid(height, width, h_rows, h_spans, h_colors, v_cols, v_spans, v_colors)
    if grid is None:
        return None
    return grid, "col_disjoint"


def _candidate_mixed() -> tuple[np.ndarray, str] | None:
    n_h = int(np.random.randint(MIN_OBJECTS, MAX_OBJECTS + 1))
    n_v = int(np.random.randint(MIN_OBJECTS, MAX_OBJECTS + 1))
    height, width = _pick_dimensions()
    h_rows = _sample_spaced_positions(1, height - 2, n_h)
    v_cols = _sample_spaced_positions(1, width - 2, n_v)
    if h_rows is None or v_cols is None:
        return None

    all_h_spans = _all_spans(width)
    all_v_spans = _all_spans(height)
    if not all_h_spans or not all_v_spans:
        return None

    for _ in range(200):
        h_spans = [all_h_spans[int(np.random.randint(len(all_h_spans)))] for _ in range(n_h)]
        if not any(_covers(span, col) for span in h_spans for col in v_cols):
            continue

        options_by_col: list[list[tuple[int, int]]] = []
        include_safe_options: list[list[tuple[int, int]]] = []
        has_forbidden_col = False
        for v_col in v_cols:
            col_options = []
            safe_options = []
            forbidden_here = False
            for span in all_v_spans:
                ok = True
                includes_safe_row = False
                for row, h_span in zip(h_rows, h_spans):
                    row_in = _covers(span, row)
                    col_in = _covers(h_span, v_col)
                    if row_in and col_in:
                        ok = False
                        forbidden_here = True
                        break
                    if row_in and not col_in:
                        includes_safe_row = True
                if ok:
                    col_options.append(span)
                    if includes_safe_row:
                        safe_options.append(span)
            if not col_options:
                break
            has_forbidden_col = has_forbidden_col or any(_covers(h_span, v_col) for h_span in h_spans)
            options_by_col.append(col_options)
            include_safe_options.append(safe_options)
        if len(options_by_col) != n_v or not has_forbidden_col:
            continue

        special = [i for i, options in enumerate(include_safe_options) if options]
        if not special:
            continue
        chosen_special = int(special[int(np.random.randint(len(special)))])
        v_spans = [options[int(np.random.randint(len(options)))] for options in options_by_col]
        special_options = include_safe_options[chosen_special]
        v_spans[chosen_special] = special_options[int(np.random.randint(len(special_options)))]

        h_colors, v_colors = _pick_colors(n_h, n_v)
        grid = _render_grid(height, width, h_rows, h_spans, h_colors, v_cols, v_spans, v_colors)
        if grid is None:
            continue
        return grid, "mixed"
    return None


def _validate_candidate(grid: np.ndarray, mode: str) -> np.ndarray:
    horizontals, verticals = _analyze_grid(grid)
    assert all(MIN_SEGMENT_LEN <= seg["length"] <= MAX_SEGMENT_LEN for seg in horizontals), "horizontal length out of range"
    assert all(MIN_SEGMENT_LEN <= seg["length"] <= MAX_SEGMENT_LEN for seg in verticals), "vertical length out of range"

    out = main(grid)
    expected = _build_output(horizontals, verticals)
    assert np.array_equal(out, expected), "solver output disagrees with geometric event list"
    assert out.shape == (len(horizontals) * len(verticals), 3), "wrong compact table size"
    assert len({tuple(map(int, row)) for row in out}) == out.shape[0], "crossing rows must stay unique"

    relations = [_pair_relation(hseg, vseg) for hseg in horizontals for vseg in verticals]
    assert "invalid" not in relations, "input contains a pre-extension overlap"

    if mode == "row_disjoint":
        assert all(rel in {"row", "both"} for rel in relations), "row-disjoint case leaked column-only overlaps"
        assert "row" in relations, "row-disjoint case must include at least one row-separated pair"
    elif mode == "col_disjoint":
        assert all(rel in {"col", "both"} for rel in relations), "col-disjoint case leaked row-only overlaps"
        assert "col" in relations, "col-disjoint case must include at least one col-separated pair"
    else:
        assert "row" in relations and "col" in relations, "mixed case must exhibit both separation mechanisms"

    return out


def generate_input() -> np.ndarray:
    builders = (_candidate_row_disjoint, _candidate_col_disjoint, _candidate_mixed)
    for _ in range(MAX_ATTEMPTS):
        builder = builders[int(np.random.randint(len(builders)))]
        result = builder()
        if result is None:
            continue
        grid, mode = result
        out = _validate_candidate(grid, mode)
        in_sig = _grid_sig(grid)
        out_sig = _grid_sig(out)
        if in_sig in _SEEN_INPUT_SIGS or out_sig in _SEEN_OUTPUT_SIGS:
            continue
        _SEEN_INPUT_SIGS.add(in_sig)
        _SEEN_OUTPUT_SIGS.add(out_sig)
        return grid
    raise RuntimeError("failed to generate a valid ray-crossing compression input")
