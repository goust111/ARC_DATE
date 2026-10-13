import numpy as np

from arc_utils import Color, get_bg
import random


def _find_divider(g: np.ndarray) -> tuple[str, int, int]:
    bg = get_bg(g)
    h, w = g.shape
    candidates: list[tuple[str, int, int]] = []

    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        mask = g == color
        count = int(mask.sum())
        if count == w:
            for row in np.flatnonzero(np.all(mask, axis=1)):
                candidates.append(("row", int(row), color))
        if count == h:
            for col in np.flatnonzero(np.all(mask, axis=0)):
                candidates.append(("col", int(col), color))

    assert len(candidates) == 1, f"expected one full-line divider, got {candidates}"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each yellow object lane toward the full divider, then continue through the divider side while packing red pixels against the far edge.

    Input structure:
    - A black grid contains one complete horizontal or vertical divider line.
    - Yellow pixels are all on one side of the divider, and red pixels are all on the opposite side.
    - Lanes are columns for a horizontal divider and rows for a vertical divider.

    Rule:
    - Only process lanes that contain yellow pixels.
    - Recolor the original yellow pixels green.
    - Fill the gap from the nearest yellow pixel to the divider with yellow.
    - On the red side of the divider, fill the lane with the divider color and pack the same number of red pixels against the far edge.

    Key insight:
    - The full divider defines independent one-dimensional lanes; each yellow lane preserves its red count but compresses those red pixels to the far boundary.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    orientation, divider, divider_color = _find_divider(g)

    yellow = Color.YELLOW
    red = Color.RED
    green = Color.GREEN
    assert np.any(g == yellow)
    assert np.any(g == red)

    out[g == yellow] = green

    if orientation == "row":
        yellow_above = bool(np.any(g[:divider, :] == yellow))
        yellow_below = bool(np.any(g[divider + 1 :, :] == yellow))
        red_above = bool(np.any(g[:divider, :] == red))
        red_below = bool(np.any(g[divider + 1 :, :] == red))
        assert yellow_above ^ yellow_below
        assert red_above ^ red_below
        assert yellow_above != red_above

        if yellow_above:
            for col in range(w):
                yellow_rows = np.flatnonzero(g[:divider, col] == yellow)
                if yellow_rows.size == 0:
                    continue
                nearest = int(yellow_rows.max())
                out[nearest + 1 : divider, col] = yellow

                red_count = int(np.sum(g[divider + 1 :, col] == red))
                out[divider + 1 : h - red_count, col] = divider_color
                if red_count:
                    out[h - red_count : h, col] = red
        else:
            for col in range(w):
                yellow_rows = np.flatnonzero(g[divider + 1 :, col] == yellow) + divider + 1
                if yellow_rows.size == 0:
                    continue
                nearest = int(yellow_rows.min())
                out[divider + 1 : nearest, col] = yellow

                red_count = int(np.sum(g[:divider, col] == red))
                if red_count:
                    out[:red_count, col] = red
                out[red_count:divider, col] = divider_color
    else:
        assert orientation == "col"
        yellow_left = bool(np.any(g[:, :divider] == yellow))
        yellow_right = bool(np.any(g[:, divider + 1 :] == yellow))
        red_left = bool(np.any(g[:, :divider] == red))
        red_right = bool(np.any(g[:, divider + 1 :] == red))
        assert yellow_left ^ yellow_right
        assert red_left ^ red_right
        assert yellow_left != red_left

        if yellow_left:
            for row in range(h):
                yellow_cols = np.flatnonzero(g[row, :divider] == yellow)
                if yellow_cols.size == 0:
                    continue
                nearest = int(yellow_cols.max())
                out[row, nearest + 1 : divider] = yellow

                red_count = int(np.sum(g[row, divider + 1 :] == red))
                out[row, divider + 1 : w - red_count] = divider_color
                if red_count:
                    out[row, w - red_count : w] = red
        else:
            for row in range(h):
                yellow_cols = np.flatnonzero(g[row, divider + 1 :] == yellow) + divider + 1
                if yellow_cols.size == 0:
                    continue
                nearest = int(yellow_cols.min())
                out[row, divider + 1 : nearest] = yellow

                red_count = int(np.sum(g[row, :divider] == red))
                if red_count:
                    out[row, :red_count] = red
                out[row, red_count:divider] = divider_color

    return out

# --------------- generator (o5x) ---------------

# Grid geometry.
GRID_SIZE = 30
GRID_LAST_INDEX = GRID_SIZE - 1
ORIENTATIONS = ["row", "col"]

# Divider and side depths.
YELLOW_DEPTH_OPTIONS = [9, 10]
MIN_DIVIDER_GAP = 2

# Yellow-side component and pixel counts.
YELLOW_LANE_COUNT_RANGE = (8, 15)
YELLOW_COMPONENT_COUNT_RANGE = (3, 5)
YELLOW_COMPONENT_MAX_LENGTH = 7
YELLOW_TOTAL_RANGE = (13, 23)
YELLOW_SEGMENT_LENGTH_RANGE = (1, 3)

# Red-side contiguous run counts.
RED_TOTAL_RANGE = (46, 73)
RED_SEGMENT_LENGTH_RANGE = (1, 5)

# Generation retry limit.
MAX_GENERATION_ATTEMPTS = 300

Layout = tuple[str, bool, int, int, int]
YellowSegment = tuple[int, int, int]
RedSegment = tuple[int, int, int]


def _gen_bounded_composition(total: int, count: int, lo: int, hi: int) -> list[int]:
    """Sample count integers in [lo, hi] that sum to total."""
    vals: list[int] = []
    remaining = total
    for i in range(count):
        left = count - i - 1
        vmin = max(lo, remaining - left * hi)
        vmax = min(hi, remaining - left * lo)
        val = random.randint(vmin, vmax)
        vals.append(val)
        remaining -= val
    return vals


def _gen_lane_intervals(lengths: list[int], lane_count: int = GRID_SIZE) -> list[tuple[int, int]]:
    """Place separated lane intervals with the given lengths."""
    filled = sum(lengths)
    free = lane_count - filled - (len(lengths) - 1)
    spaces = _gen_bounded_composition(free, len(lengths) + 1, 0, free)
    intervals: list[tuple[int, int]] = []
    lane = spaces[0]
    for i, length in enumerate(lengths):
        intervals.append((lane, lane + length - 1))
        lane += length
        if i + 1 < len(lengths):
            lane += 1 + spaces[i + 1]
    return intervals


def _sample_layout() -> Layout:
    """Return orientation, yellow side, yellow depth, red depth, and divider."""
    orientation = random.choice(ORIENTATIONS)
    yellow_before = bool(random.getrandbits(1))
    yellow_depth = random.choice(YELLOW_DEPTH_OPTIONS)
    red_depth = GRID_LAST_INDEX - yellow_depth
    divider = yellow_depth if yellow_before else GRID_LAST_INDEX - yellow_depth
    return orientation, yellow_before, yellow_depth, red_depth, divider


def _sample_yellow_segments(yellow_depth: int) -> tuple[list[YellowSegment], set[int]]:
    """Sample connected yellow blobs as (lane, start_depth, end_depth)."""
    yellow_lane_count = random.randint(*YELLOW_LANE_COUNT_RANGE)
    component_count = random.randint(*YELLOW_COMPONENT_COUNT_RANGE)
    component_lengths = _gen_bounded_composition(
        yellow_lane_count,
        component_count,
        1,
        min(YELLOW_COMPONENT_MAX_LENGTH, yellow_lane_count),
    )
    random.shuffle(component_lengths)
    intervals = _gen_lane_intervals(component_lengths)

    yellow_total = random.randint(
        max(YELLOW_TOTAL_RANGE[0], yellow_lane_count),
        min(YELLOW_TOTAL_RANGE[1], yellow_lane_count * YELLOW_SEGMENT_LENGTH_RANGE[1]),
    )
    yellow_lengths = _gen_bounded_composition(
        yellow_total,
        yellow_lane_count,
        *YELLOW_SEGMENT_LENGTH_RANGE,
    )
    random.shuffle(yellow_lengths)

    yellow_lanes: set[int] = set()
    yellow_segments: list[YellowSegment] = []
    length_i = 0
    for lane_start, lane_end in intervals:
        prev_start: int | None = None
        prev_end: int | None = None
        for lane in range(lane_start, lane_end + 1):
            seg_len = yellow_lengths[length_i]
            length_i += 1
            max_start = yellow_depth - MIN_DIVIDER_GAP - seg_len
            if prev_start is None or prev_end is None:
                start = random.randint(0, max_start)
            else:
                lo = max(0, prev_start - seg_len + 1)
                hi = min(max_start, prev_end)
                start = random.randint(lo, hi)
            end = start + seg_len - 1
            yellow_segments.append((lane, start, end))
            yellow_lanes.add(lane)
            prev_start, prev_end = start, end

    return yellow_segments, yellow_lanes


def _sample_red_segments(red_depth: int) -> list[RedSegment]:
    """Sample one contiguous red run for each lane."""
    red_total = random.randint(*RED_TOTAL_RANGE)
    red_counts = _gen_bounded_composition(red_total, GRID_SIZE, *RED_SEGMENT_LENGTH_RANGE)
    random.shuffle(red_counts)
    return [
        (lane, random.randint(0, red_depth - red_count), red_count)
        for lane, red_count in enumerate(red_counts)
    ]


def _new_grid_with_divider(orientation: str, divider: int) -> np.ndarray:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    if orientation == "row":
        grid[divider, :] = Color.TEAL
    else:
        grid[:, divider] = Color.TEAL
    return grid


def _paint_yellow(
    grid: np.ndarray,
    orientation: str,
    yellow_before: bool,
    lane: int,
    start: int,
    stop: int,
    color: int,
) -> None:
    if yellow_before:
        depth_slice = slice(start, stop)
    else:
        depth_slice = slice(GRID_SIZE - stop, GRID_SIZE - start)

    if orientation == "row":
        grid[depth_slice, lane] = color
    else:
        grid[lane, depth_slice] = color


def _paint_red(
    grid: np.ndarray,
    orientation: str,
    yellow_before: bool,
    divider: int,
    lane: int,
    start: int,
    stop: int,
    color: int,
) -> None:
    if yellow_before:
        depth_slice = slice(divider + 1 + start, divider + 1 + stop)
    else:
        depth_slice = slice(start, stop)

    if orientation == "row":
        grid[depth_slice, lane] = color
    else:
        grid[lane, depth_slice] = color


def _render_input(
    layout: Layout,
    yellow_segments: list[YellowSegment],
    red_segments: list[RedSegment],
) -> np.ndarray:
    """Render the sampled input grid from yellow and red lane segments."""
    orientation, yellow_before, _, _, divider = layout
    grid = _new_grid_with_divider(orientation, divider)
    for lane, start, end in yellow_segments:
        _paint_yellow(grid, orientation, yellow_before, lane, start, end + 1, Color.YELLOW)
    for lane, red_start, red_count in red_segments:
        red_stop = red_start + red_count
        _paint_red(grid, orientation, yellow_before, divider, lane, red_start, red_stop, Color.RED)
    return grid


def _render_expected_output(
    layout: Layout,
    yellow_segments: list[YellowSegment],
    yellow_lanes: set[int],
    red_segments: list[RedSegment],
) -> np.ndarray:
    """Render the oracle output that main(inp) must match."""
    orientation, yellow_before, yellow_depth, red_depth, divider = layout
    grid = _new_grid_with_divider(orientation, divider)
    for lane, start, end in yellow_segments:
        _paint_yellow(grid, orientation, yellow_before, lane, start, end + 1, Color.GREEN)
        _paint_yellow(grid, orientation, yellow_before, lane, end + 1, yellow_depth, Color.YELLOW)
    for lane, red_start, red_count in red_segments:
        if lane in yellow_lanes:
            if yellow_before:
                teal_start, teal_stop = 0, red_depth - red_count
                packed_start, packed_stop = red_depth - red_count, red_depth
            else:
                packed_start, packed_stop = 0, red_count
                teal_start, teal_stop = red_count, red_depth
            _paint_red(
                grid, orientation, yellow_before, divider, lane, teal_start, teal_stop, Color.TEAL
            )
            _paint_red(
                grid, orientation, yellow_before, divider, lane, packed_start, packed_stop, Color.RED
            )
        else:
            red_stop = red_start + red_count
            _paint_red(
                grid, orientation, yellow_before, divider, lane, red_start, red_stop, Color.RED
            )
    return grid


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    yellow_segments, yellow_lanes = _sample_yellow_segments(layout[2])
    red_segments = _sample_red_segments(layout[3])

    out = _render_expected_output(layout, yellow_segments, yellow_lanes, red_segments)
    inp = _render_input(layout, yellow_segments, red_segments)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 05a7bcf2 input")
