from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _background(grid: np.ndarray) -> int:
    """Return the dominant ARC color."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _runs(cols: np.ndarray) -> list[tuple[int, int]]:
    """Convert sorted column coordinates to inclusive contiguous runs."""
    if len(cols) == 0:
        return []
    start = previous = int(cols[0])
    result: list[tuple[int, int]] = []
    for value in cols[1:]:
        col = int(value)
        if col != previous + 1:
            result.append((start, previous))
            start = col
        previous = col
    result.append((start, previous))
    return result


def _nearest_outside(col: int, left: int, right: int, width: int) -> int:
    """Choose the nearer exterior column, breaking exact ties to the right."""
    choices: list[tuple[int, int, int]] = []
    if left > 0:
        choices.append((col - left + 1, 0, left - 1))
    if right + 1 < width:
        choices.append((right - col + 1, 1, right + 1))
    assert choices, "an obstacle must have an exterior side inside the grid"
    # Negating the side flag makes right win when distances are equal.
    return min(choices, key=lambda item: (item[0], -item[1]))[2]


def _parse(grid: np.ndarray):
    """Identify the visible source row, obstacle segments, and palette row."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape
    assert height >= 5 and width >= 5
    bg = _background(g)

    source_cols = np.flatnonzero(g[0] != bg)
    assert len(source_cols) >= 1, "expected source pixels on the top row"
    source_colors = set(map(int, g[0, source_cols]))
    assert len(source_colors) == 1, "all top-row sources must be grey"
    source_color = next(iter(source_colors))
    assert source_color == 5, "the source pixels are identified by visible grey"

    palette_cols = np.flatnonzero(g[-1] != bg)
    palette_runs = _runs(palette_cols)
    assert len(palette_runs) == 1, "the bottom palette must be one contiguous row"
    p_left, p_right = palette_runs[0]
    palette = [int(x) for x in g[-1, p_left : p_right + 1]]
    assert len(palette) >= 2 and len(set(palette)) >= 2
    assert source_color not in palette and bg not in palette

    middle_colors = set(map(int, np.unique(g[1:-1]))) - {bg} - set(palette)
    assert len(middle_colors) == 1, "expected one common obstacle color"
    obstacle_color = next(iter(middle_colors))
    assert obstacle_color != source_color

    obstacles_by_row: dict[int, list[tuple[int, int]]] = {}
    for row in range(1, height - 1):
        row_runs = _runs(np.flatnonzero(g[row] == obstacle_color))
        if row_runs:
            obstacles_by_row[row] = row_runs

    return bg, source_color, obstacle_color, source_cols, palette, obstacles_by_row


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend grey top sources as palette-colored rays that recolor after every obstacle detour.

    Input structure:
    - The dominant color is background; isolated grey source pixels lie on the top row.
    - Mutually separate horizontal segments of one color are obstacles in the middle.
    - One contiguous multicolor run on the bottom row gives the palette from left to right.

    Rule:
    - Each source descends vertically from row 1 using the first palette color, stopping
      above the palette row.  When its current column meets an obstacle segment, it moves
      on the row above to that segment's nearer exterior column (right on an exact tie).
    - A completed detour advances the palette cyclically; both the horizontal detour and
      the following vertical portion use this new color.  Input clues remain unchanged.

    Key insight:
    - The ray's color is indexed by its own obstacle-event count, so geometry and cyclic
      recoloring must be tracked together rather than solved as independent decorations.
    """
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    bg, _source_color, obstacle_color, source_cols, palette, obstacles = _parse(g)
    out = g.copy()

    for source_col in source_cols:
        col = int(source_col)
        palette_index = 0
        row = 1
        while row < height - 1:
            hit = next(
                ((left, right) for left, right in obstacles.get(row, []) if left <= col <= right),
                None,
            )
            if hit is None:
                assert g[row, col] != obstacle_color
                out[row, col] = palette[palette_index]
                row += 1
                continue

            left, right = hit
            new_col = _nearest_outside(col, left, right, width)
            palette_index = (palette_index + 1) % len(palette)
            lo, hi = sorted((col, new_col))
            assert row > 0 and np.all(g[row - 1, lo : hi + 1] != obstacle_color)
            out[row - 1, lo : hi + 1] = palette[palette_index]
            col = new_col
            # The exterior cell on the obstacle row is the first cell after the detour.
            assert g[row, col] == bg
            out[row, col] = palette[palette_index]
            row += 1

    # Preserve all three kinds of visible clues even if a path approaches one closely.
    out[g == obstacle_color] = obstacle_color
    out[0] = g[0]
    out[-1] = g[-1]
    return out


_ARC_COLORS = tuple(range(10))
_SOURCE_COLOR = 5
_MAX_GENERATION_ATTEMPTS = 300


def _outside_for_segment(col: int, segment: tuple[int, int], width: int) -> int:
    left, right = segment
    left_distance = col - (left - 1)
    right_distance = (right + 1) - col
    if left == 0:
        return right + 1
    if right == width - 1:
        return left - 1
    return right + 1 if right_distance <= left_distance else left - 1


def _segment_candidates(
    col: int, lane_left: int, lane_right: int, direction: str, width: int
) -> list[tuple[int, int]]:
    """Enumerate visible segments producing a requested nearest-side branch."""
    candidates: list[tuple[int, int]] = []
    for left in range(lane_left, col + 1):
        for right in range(col, lane_right + 1):
            if not 2 <= right - left + 1 <= 5:
                continue
            left_distance = col - (left - 1)
            right_distance = (right + 1) - col
            destination = _outside_for_segment(col, (left, right), width)
            if not lane_left <= destination <= lane_right:
                continue
            if direction == "tie":
                valid = left_distance == right_distance and destination == right + 1
            elif direction == "left":
                valid = destination == left - 1 and left_distance < right_distance
            else:
                valid = destination == right + 1 and right_distance < left_distance
            if valid:
                candidates.append((left, right))
    return candidates


def _choose_event_rows(height: int, count: int) -> list[int] | None:
    """Choose rows with two clear vertical cells between successive events."""
    candidates = list(range(3, height - 2))
    for _ in range(80):
        rows = sorted(random.sample(candidates, count))
        if all(b - a >= 3 for a, b in zip(rows, rows[1:])):
            return rows
    return None


def _independent_expected(
    inp: np.ndarray,
    bg: int,
    obstacle_color: int,
    palette: list[int],
    plans: list[dict[str, object]],
) -> np.ndarray:
    """Generator oracle driven by planned events rather than solver parsing."""
    height, _ = inp.shape
    expected = inp.copy()
    for plan in plans:
        col = int(plan["source"])
        event_map = {int(row): tuple(segment) for row, segment in plan["events"]}
        color_index = 0
        for row in range(1, height - 1):
            if row in event_map:
                segment = event_map[row]
                new_col = _outside_for_segment(col, segment, inp.shape[1])
                color_index = (color_index + 1) % len(palette)
                lo, hi = sorted((col, new_col))
                expected[row - 1, lo : hi + 1] = palette[color_index]
                col = new_col
            expected[row, col] = palette[color_index]
    expected[inp == obstacle_color] = obstacle_color
    expected[0] = inp[0]
    expected[-1] = inp[-1]
    assert np.count_nonzero(expected == bg) > expected.size // 2
    return expected


def _generate_candidate() -> np.ndarray | None:
    palette_size = random.randint(2, 4)
    featured_count = palette_size + 1  # guarantees a visible cyclic wrap
    source_count = random.randint(3, 4)
    lane_width = 7
    width = source_count * lane_width + 2
    min_height = 3 * featured_count + 5
    height = random.randint(min_height, min(27, min_height + 5))
    if width > 30 or height > 30:
        return None

    available = [c for c in _ARC_COLORS if c != _SOURCE_COLOR]
    bg = random.choice(available)
    available.remove(bg)
    obstacle_color = random.choice(available)
    available.remove(obstacle_color)
    palette = random.sample(available, palette_size)

    inp = np.full((height, width), bg, dtype=int)
    event_rows = _choose_event_rows(height, featured_count)
    if event_rows is None:
        return None

    plans: list[dict[str, object]] = []
    all_segments: dict[int, list[tuple[int, int]]] = {}
    # The first ray supplies tie-right, both strict sides, and enough events to wrap.
    featured_directions = ["tie", "left", "right", "left", "right"][:featured_count]

    for index in range(source_count):
        lane_left = 1 + index * lane_width
        lane_right = lane_left + lane_width - 1
        source = lane_left + lane_width // 2
        inp[0, source] = _SOURCE_COLOR

        if index == 0:
            rows = event_rows
            directions = featured_directions
        elif index == source_count - 1:
            rows = []  # explicit counterexample to "every source must detour"
            directions = []
        else:
            count = random.randint(1, min(3, featured_count))
            rows = sorted(random.sample(event_rows, count))
            directions = [random.choice(("left", "right", "tie")) for _ in rows]

        col = source
        events: list[tuple[int, tuple[int, int]]] = []
        for row, direction in zip(rows, directions):
            choices = _segment_candidates(col, lane_left, lane_right, direction, width)
            if not choices:
                # A strict branch is still useful if the randomly requested one cannot fit.
                fallback = "left" if col > source else "right"
                choices = _segment_candidates(col, lane_left, lane_right, fallback, width)
            if not choices:
                return None
            segment = random.choice(choices)
            events.append((row, segment))
            all_segments.setdefault(row, []).append(segment)
            col = _outside_for_segment(col, segment, width)
        plans.append({"source": source, "events": events})

    # Paint all obstacle clues after planning all independent lanes.
    for row, segments in all_segments.items():
        for left, right in segments:
            inp[row, left : right + 1] = obstacle_color

    palette_left = random.randint(1, width - palette_size - 1)
    inp[-1, palette_left : palette_left + palette_size] = palette

    # Structural and anti-shortcut checks performed independently of main().
    obstacle_cells = np.argwhere(inp == obstacle_color)
    if len(obstacle_cells) < 3 or _background(inp) != bg:
        return None
    segments_flat = [(row, left, right) for row, segs in all_segments.items() for left, right in segs]
    for i, (row_a, left_a, right_a) in enumerate(segments_flat):
        for row_b, left_b, right_b in segments_flat[i + 1 :]:
            if row_a == row_b and (left_a, right_a) == (left_b, right_b):
                return None
            if abs(row_a - row_b) <= 1 and not (right_a + 1 < left_b or right_b + 1 < left_a):
                return None

    expected = _independent_expected(inp, bg, obstacle_color, palette, plans)
    actual = main(inp)
    if not np.array_equal(actual, expected) or np.array_equal(actual, inp):
        return None
    if not np.array_equal(actual[inp != bg], inp[inp != bg]):
        return None
    if not set(palette).issubset(set(map(int, np.unique(actual)))):
        return None

    # The event-free ray rules out treating obstacles as a global row event.
    straight_source = int(plans[-1]["source"])
    if not np.all(actual[1:-1, straight_source] == palette[0]):
        return None

    # Every planned event must change both the route and the color on its horizontal leg.
    for plan in plans:
        col = int(plan["source"])
        for event_index, (row, segment) in enumerate(plan["events"], start=1):
            new_col = _outside_for_segment(col, tuple(segment), width)
            lo, hi = sorted((col, new_col))
            if not np.all(actual[int(row) - 1, lo : hi + 1] == palette[event_index % palette_size]):
                return None
            if new_col == col:
                return None
            col = new_col

    first_events = plans[0]["events"]
    if len(first_events) <= palette_size:
        return None
    first_source = int(plans[0]["source"])
    first_segment = tuple(first_events[0][1])
    left, right = first_segment
    if first_source - (left - 1) != (right + 1) - first_source:
        return None
    if _outside_for_segment(first_source, first_segment, width) != right + 1:
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate varied, noncrossing examples covering every semantic branch."""
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        candidate = _generate_candidate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid绕行换色射线 input")
