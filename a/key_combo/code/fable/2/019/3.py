import numpy as np


BACKGROUND = 0
MARKER_COLOR = 2
SEGMENT_COLORS = np.array([1, 3, 4, 5, 6, 7, 8, 9], dtype=int)

_RNG = np.random.default_rng()
_GEN_CALL_COUNT = 0

MAX_GENERATION_ATTEMPTS = 400
HORIZONTAL_SIZE_RANGE = ((10, 18), (15, 25))
VERTICAL_SIZE_RANGE = ((15, 25), (10, 18))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _segments_from_line(line: np.ndarray, bg: int) -> list[dict]:
    line = np.asarray(line, dtype=int)
    segments = []
    start = None
    color = None
    for idx, value in enumerate(line):
        value = int(value)
        is_segment = value not in (bg, MARKER_COLOR)
        if is_segment:
            if start is None:
                start = idx
                color = value
            elif value != color:
                segments.append(
                    {
                        "start": int(start),
                        "end": int(idx - 1),
                        "length": int(idx - start),
                        "color": int(color),
                    }
                )
                start = idx
                color = value
        elif start is not None:
            segments.append(
                {
                    "start": int(start),
                    "end": int(idx - 1),
                    "length": int(idx - start),
                    "color": int(color),
                }
            )
            start = None
            color = None
    if start is not None:
        segments.append(
            {
                "start": int(start),
                "end": int(len(line) - 1),
                "length": int(len(line) - start),
                "color": int(color),
            }
        )
    return segments


def _order_segments_for_marker(segments: list[dict], side: str) -> list[dict]:
    ordered = sorted(segments, key=lambda seg: int(seg["start"]))
    if side in {"right", "bottom"}:
        ordered.reverse()
    return ordered


def _relay_destinations(segments: list[dict], line_len: int, side: str) -> list[tuple[int, int]]:
    moved = []
    for idx, seg in enumerate(segments):
        length = int(seg["length"])
        if idx == len(segments) - 1:
            if side in {"left", "top"}:
                start = line_len - length
            else:
                start = 0
        else:
            successor = segments[idx + 1]
            if side in {"left", "top"}:
                start = int(successor["start"]) - length
            else:
                start = int(successor["end"]) + 1
        end = start + length - 1
        moved.append((int(start), int(end)))
    return moved


def _transform_line(line: np.ndarray, side: str, bg: int) -> np.ndarray:
    line = np.asarray(line, dtype=int)
    marker_index = 0 if side in {"left", "top"} else len(line) - 1
    assert int(line[marker_index]) == MARKER_COLOR, "missing edge marker"
    assert int(np.count_nonzero(line == MARKER_COLOR)) == 1, "each active line needs exactly one marker"

    segments = _segments_from_line(line, bg)
    assert segments, "marked line must contain at least one segment"

    ordered = _order_segments_for_marker(segments, side)
    moved = _relay_destinations(ordered, len(line), side)

    out = np.full(line.shape, int(bg), dtype=int)
    out[marker_index] = MARKER_COLOR
    for seg, (start, end) in zip(ordered, moved):
        assert 0 <= start <= end < len(line), "segment moved out of bounds"
        out[start : end + 1] = int(seg["color"])
    return out


def _compressed_line(line: np.ndarray, side: str, bg: int) -> np.ndarray:
    line = np.asarray(line, dtype=int)
    segments = sorted(_segments_from_line(line, bg), key=lambda seg: int(seg["start"]))
    out = np.full(line.shape, int(bg), dtype=int)
    marker_index = 0 if side in {"left", "top"} else len(line) - 1
    out[marker_index] = MARKER_COLOR

    if side in {"left", "top"}:
        cursor = len(line)
        for seg in reversed(segments):
            start = cursor - int(seg["length"])
            out[start:cursor] = int(seg["color"])
            cursor = start
    else:
        cursor = 0
        for seg in segments:
            end = cursor + int(seg["length"])
            out[cursor:end] = int(seg["color"])
            cursor = end
    return out


def _find_markers(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    assert int(grid[0, 0]) != MARKER_COLOR
    assert int(grid[0, w - 1]) != MARKER_COLOR
    assert int(grid[h - 1, 0]) != MARKER_COLOR
    assert int(grid[h - 1, w - 1]) != MARKER_COLOR

    markers = []
    used_rows = set()
    used_cols = set()

    for r in range(1, h - 1):
        left = int(grid[r, 0]) == MARKER_COLOR
        right = int(grid[r, w - 1]) == MARKER_COLOR
        assert not (left and right), "a row cannot have two markers"
        if left or right:
            side = "left" if left else "right"
            assert r not in used_rows, "marker rows must be unique"
            used_rows.add(r)
            markers.append({"axis": "row", "index": int(r), "side": side})

    for c in range(1, w - 1):
        top = int(grid[0, c]) == MARKER_COLOR
        bottom = int(grid[h - 1, c]) == MARKER_COLOR
        assert not (top and bottom), "a column cannot have two markers"
        if top or bottom:
            side = "top" if top else "bottom"
            assert c not in used_cols, "marker columns must be unique"
            used_cols.add(c)
            markers.append({"axis": "col", "index": int(c), "side": side})

    marker_mask = np.zeros_like(grid, dtype=bool)
    marker_mask[0, :] = True
    marker_mask[h - 1, :] = True
    marker_mask[:, 0] = True
    marker_mask[:, w - 1] = True
    assert not np.any((grid == MARKER_COLOR) & ~marker_mask), "marker color may only appear on the border"
    return markers


def _line_view(grid: np.ndarray, axis: str, index: int) -> np.ndarray:
    if axis == "row":
        return np.asarray(grid[index, :], dtype=int)
    return np.asarray(grid[:, index], dtype=int)


def _write_line(assignments: dict, axis: str, index: int, values: np.ndarray) -> None:
    for offset, value in enumerate(np.asarray(values, dtype=int)):
        key = (index, offset) if axis == "row" else (offset, index)
        previous = assignments.get(key)
        value = int(value)
        if previous is not None:
            assert previous == value, "transformed lines overlap inconsistently"
        assignments[key] = value


def main(grid: np.ndarray) -> np.ndarray:
    """
    Relay-push each marked row or column so every segment stops beside its successor's input position and only the last one reaches the wall.

    Input structure:
    - The grid has a dominant background, a fixed marker color on selected border cells, and monochrome 1D segments lying on those marked rows or columns.
    - A row marker on the left or right pushes horizontally inward; a column marker on the top or bottom pushes vertically inward.
    - Within any marked line, the segments are separated by background cells in the input.

    Rule:
    - For each marked line, order the segments by distance from the marker.
    - Move the first segment along the push direction until it becomes adjacent to the second segment's input position, then move the second until it becomes adjacent to the third segment's input position, and continue this relay.
    - The last segment keeps sliding until it touches the opposite wall; unmarked rows and columns remain unchanged, and marker pixels stay in place.

    Key insight:
    - This is not a full compression toward the wall: every segment except the last stops relative to the next segment's original location, so multi-segment examples preserve rule-specific gaps in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    markers = _find_markers(g)
    assert markers, "expected at least one edge marker"

    assignments: dict[tuple[int, int], int] = {}
    for marker in markers:
        line = _line_view(g, marker["axis"], int(marker["index"]))
        transformed = _transform_line(line, str(marker["side"]), bg)
        _write_line(assignments, str(marker["axis"]), int(marker["index"]), transformed)

    out = g.copy()
    for (r, c), value in assignments.items():
        out[r, c] = int(value)
    return out


def _sample_lengths(count: int, capacity: int) -> list[int] | None:
    max_total = min(4 * count, capacity)
    if max_total < count:
        return None
    total = int(_RNG.integers(count, max_total + 1))
    lengths = np.ones(count, dtype=int)
    remaining = total - count
    while remaining > 0:
        idx = int(_RNG.integers(0, count))
        if lengths[idx] < 4:
            lengths[idx] += 1
            remaining -= 1
    return [int(x) for x in lengths]


def _intervals_with_marker(line_len: int, segment_count: int, mirror: bool, showcase: bool) -> list[tuple[int, int]] | None:
    usable = line_len - 1
    gap_mins = np.ones(segment_count + 1, dtype=int)
    if showcase and usable - gap_mins.sum() >= segment_count + 1:
        gap_mins[-1] = 2
    capacity = usable - int(gap_mins.sum())
    lengths = _sample_lengths(segment_count, capacity)
    if lengths is None:
        return None

    extra_gap = usable - int(gap_mins.sum()) - int(sum(lengths))
    extras = _RNG.multinomial(extra_gap, np.full(segment_count + 1, 1.0 / (segment_count + 1)))
    gaps = gap_mins + extras

    intervals = []
    cursor = 1 + int(gaps[0])
    for idx, length in enumerate(lengths):
        start = cursor
        end = start + int(length) - 1
        intervals.append((int(start), int(end)))
        cursor = end + 1 + int(gaps[idx + 1])
    assert cursor == line_len

    if not mirror:
        return intervals
    mirrored = [(line_len - 1 - end, line_len - 1 - start) for start, end in intervals]
    return sorted((int(start), int(end)) for start, end in mirrored)


def _intervals_without_marker(line_len: int, segment_count: int) -> list[tuple[int, int]] | None:
    gap_mins = np.ones(segment_count + 1, dtype=int)
    capacity = line_len - int(gap_mins.sum())
    lengths = _sample_lengths(segment_count, capacity)
    if lengths is None:
        return None

    extra_gap = line_len - int(gap_mins.sum()) - int(sum(lengths))
    extras = _RNG.multinomial(extra_gap, np.full(segment_count + 1, 1.0 / (segment_count + 1)))
    gaps = gap_mins + extras

    intervals = []
    cursor = int(gaps[0])
    for idx, length in enumerate(lengths):
        start = cursor
        end = start + int(length) - 1
        intervals.append((int(start), int(end)))
        cursor = end + 1 + int(gaps[idx + 1])
    assert cursor == line_len

    if bool(_RNG.integers(0, 2)):
        mirrored = [(line_len - 1 - end, line_len - 1 - start) for start, end in intervals]
        intervals = sorted((int(start), int(end)) for start, end in mirrored)
    return intervals


def _spaced_slots(size: int) -> list[int]:
    start = int(_RNG.integers(1, 3))
    slots = list(range(start, size - 1, 2))
    if len(slots) < 3:
        slots = list(range(1, size - 1, 2))
    return slots


def _paint_row(grid: np.ndarray, row: int, side: str | None, intervals: list[tuple[int, int]]) -> None:
    colors = _RNG.choice(SEGMENT_COLORS, size=len(intervals), replace=False)
    if side == "left":
        grid[row, 0] = MARKER_COLOR
    elif side == "right":
        grid[row, -1] = MARKER_COLOR
    for (start, end), color in zip(intervals, colors):
        grid[row, start : end + 1] = int(color)


def _paint_col(grid: np.ndarray, col: int, side: str | None, intervals: list[tuple[int, int]]) -> None:
    colors = _RNG.choice(SEGMENT_COLORS, size=len(intervals), replace=False)
    if side == "top":
        grid[0, col] = MARKER_COLOR
    elif side == "bottom":
        grid[-1, col] = MARKER_COLOR
    for (start, end), color in zip(intervals, colors):
        grid[start : end + 1, col] = int(color)


def _validate_generated(grid: np.ndarray, require_single: bool, require_inert: bool) -> None:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    markers = _find_markers(g)
    assert markers, "generator must place markers"

    axes = {marker["axis"] for marker in markers}
    assert len(axes) == 1, "generator keeps one orientation per instance to avoid ambiguous crossings"
    axis = next(iter(axes))
    active_indices = {int(marker["index"]) for marker in markers}
    transformed = main(g)

    relay_showcase_count = 0
    single_segment_count = 0
    inert_count = 0

    for marker in markers:
        line = _line_view(g, marker["axis"], int(marker["index"]))
        expected = _transform_line(line, str(marker["side"]), bg)
        actual = _line_view(transformed, marker["axis"], int(marker["index"]))
        assert np.array_equal(expected, actual), "active line output is incorrect"

        segments = _segments_from_line(line, bg)
        if len(segments) == 1:
            single_segment_count += 1
        if len(segments) >= 2:
            compressed = _compressed_line(line, str(marker["side"]), bg)
            if not np.array_equal(expected, compressed):
                relay_showcase_count += 1

    if axis == "row":
        for row in range(g.shape[0]):
            if row in active_indices or int(g[row, 0]) == MARKER_COLOR or int(g[row, -1]) == MARKER_COLOR:
                continue
            line = g[row, :]
            if _segments_from_line(line, bg):
                inert_count += 1
                assert np.array_equal(line, transformed[row, :]), "unmarked row changed unexpectedly"
    else:
        for col in range(g.shape[1]):
            if col in active_indices or int(g[0, col]) == MARKER_COLOR or int(g[-1, col]) == MARKER_COLOR:
                continue
            line = g[:, col]
            if _segments_from_line(line, bg):
                inert_count += 1
                assert np.array_equal(line, transformed[:, col]), "unmarked column changed unexpectedly"

    assert relay_showcase_count >= 1, "every generated puzzle must expose relay motion against naive compression"
    if require_single:
        assert single_segment_count >= 1, "single-segment branch must be represented in alternating modes"
    if require_inert:
        assert inert_count >= 1, "static unmarked line branch must be represented in alternating modes"


def _gen_attempt(orientation: str, require_single: bool, require_inert: bool) -> np.ndarray | None:
    size_range = HORIZONTAL_SIZE_RANGE if orientation == "horizontal" else VERTICAL_SIZE_RANGE
    h = int(_RNG.integers(size_range[0][0], size_range[0][1] + 1))
    w = int(_RNG.integers(size_range[1][0], size_range[1][1] + 1))
    grid = np.full((h, w), BACKGROUND, dtype=int)

    span = h if orientation == "horizontal" else w
    line_len = w if orientation == "horizontal" else h
    slots = _spaced_slots(span)
    if len(slots) < 2:
        return None

    min_active = 2
    max_active = min(4, len(slots) - (1 if require_inert and len(slots) >= 3 else 0))
    if max_active < min_active:
        return None

    active_count = int(_RNG.integers(min_active, max_active + 1))
    inert_count = 1 if require_inert and len(slots) > active_count else 0

    chosen = _RNG.permutation(slots)
    active_slots = [int(x) for x in chosen[:active_count]]
    inert_slots = [int(x) for x in chosen[active_count : active_count + inert_count]]

    if orientation == "horizontal":
        sides = ["left", "right"]
    else:
        sides = ["top", "bottom"]

    active_sides = []
    if active_count >= 2:
        active_sides.extend(sides)
    while len(active_sides) < active_count:
        active_sides.append(str(_RNG.choice(sides)))
    _RNG.shuffle(active_sides)

    showcase_idx = int(_RNG.integers(0, active_count))
    single_idx = None
    if require_single:
        candidates = [idx for idx in range(active_count) if idx != showcase_idx]
        if not candidates:
            return None
        single_idx = int(_RNG.choice(candidates))

    for idx, slot in enumerate(active_slots):
        side = active_sides[idx]
        segment_count = int(_RNG.integers(1, 4))
        showcase = idx == showcase_idx
        if showcase:
            segment_count = int(_RNG.integers(2, 5))
        elif single_idx is not None and idx == single_idx:
            segment_count = 1

        mirror = side in {"right", "bottom"}
        intervals = _intervals_with_marker(line_len, segment_count, mirror, showcase)
        if intervals is None:
            return None
        if orientation == "horizontal":
            _paint_row(grid, slot, side, intervals)
        else:
            _paint_col(grid, slot, side, intervals)

    for slot in inert_slots:
        segment_count = int(_RNG.integers(1, 3))
        intervals = _intervals_without_marker(line_len, segment_count)
        if intervals is None:
            return None
        if orientation == "horizontal":
            _paint_row(grid, slot, None, intervals)
        else:
            _paint_col(grid, slot, None, intervals)

    try:
        _validate_generated(grid, require_single, require_inert)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    mode = _GEN_CALL_COUNT % 4
    _GEN_CALL_COUNT += 1

    orientation = "horizontal" if mode in (0, 2) else "vertical"
    require_single = mode in (2, 3)
    require_inert = mode in (1, 2)

    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(orientation, require_single, require_inert)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid relay-slide instance")
