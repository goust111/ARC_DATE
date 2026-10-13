import random
from collections import deque

import numpy as np


BG = 0
BASELINE = 5
FRAME = 6
MARKER = 8
BAR_COLORS = (1, 2, 3, 4, 7)

GRID_HEIGHT_RANGE = (17, 24)
GRID_WIDTH_RANGE = (20, 29)
BAR_HEIGHTS = tuple(range(2, 8))
FRAME_SIDE_RANGE = (3, 5)
COUNT_VALUES = (1, 2, 3, 4, 5)
GEN_ATTEMPTS = 400

_CASE_COUNT_BAG: list[int] = []


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components of a boolean mask."""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            queue = deque([(row, col)])
            seen[row, col] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(coords)
    return out


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for coords in _components(grid == FRAME):
        rows = [r for r, _c in coords]
        cols = [c for _r, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 3 or width < 3:
            raise AssertionError("frame too small")

        border = np.zeros((height, width), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if int(np.count_nonzero(border)) != len(coords):
            raise AssertionError("frame component is not a one-pixel border")
        if not np.all(patch[border] == FRAME):
            raise AssertionError("frame border must be solid")
        if np.any(patch[1:-1, 1:-1] == FRAME):
            raise AssertionError("frame interior must exclude frame color")

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "center_x_twice": c0 + c1,
            }
        )

    frames.sort(key=lambda item: item["center_x_twice"])
    return frames


def _find_baseline_row(grid: np.ndarray) -> int:
    rows = np.where(np.all(grid == BASELINE, axis=1))[0]
    assert len(rows) == 1, "expected exactly one full-width baseline"
    return int(rows[0])


def _find_bars(grid: np.ndarray, baseline_row: int) -> list[dict]:
    bars: list[dict] = []
    for col in range(grid.shape[1]):
        if baseline_row == 0:
            continue
        color = int(grid[baseline_row - 1, col])
        if color in (BG, BASELINE, FRAME, MARKER):
            continue

        row = baseline_row - 1
        while row >= 0 and int(grid[row, col]) == color:
            row -= 1
        height = baseline_row - 1 - row
        bars.append({"x": col, "color": color, "height": height})

    bars.sort(key=lambda item: item["x"])
    return bars


def _render_output(shape: tuple[int, int], baseline_row: int, bars: list[dict], counts: list[int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    out[baseline_row, :] = BASELINE
    sorted_heights = [height for _count, height in sorted((count, bar["height"]) for count, bar in zip(counts, bars))]
    for bar, new_height in zip(bars, sorted_heights):
        out[baseline_row - new_height : baseline_row, bar["x"]] = bar["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Redistribute bar heights by sorting each slot's original height with the count of cyan markers inside its corresponding magenta frame.

    Input structure:
    - The grid has black background, one full-width gray baseline near the bottom, and 4 or 5 single-column colored bars rising from that baseline.
    - Above the bar region are the same number of one-pixel-thick magenta rectangular frames, with cyan markers both inside frames and elsewhere as distractors.
    - Frame order is determined by left-to-right frame-center x position, and bar-slot order is determined by left-to-right bar x position.

    Rule:
    - Count only cyan pixels strictly inside each magenta frame, ignoring cyan pixels outside frames.
    - Pair the left-to-right frame counts with the left-to-right bar heights.
    - Sort those `(count, original_height)` pairs by count ascending, then assign the resulting heights back to the left-to-right bar slots while keeping each slot's original bar color.
    - Remove every magenta frame and cyan marker; keep only the gray baseline and the recolored-slot bars with their redistributed heights.

    Key insight:
    - The sorting key comes from accurate inside-frame counting, not from the bars themselves, and the colors stay attached to slots rather than to moved heights.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    baseline_row = _find_baseline_row(g)
    frames = _find_frames(g)
    bars = _find_bars(g, baseline_row)

    assert 4 <= len(frames) <= 5, "expected 4 or 5 frames"
    assert len(frames) == len(bars), "frame and bar counts must match"

    counts: list[int] = []
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        counts.append(int(np.count_nonzero(interior == MARKER)))

    heights = [bar["height"] for bar in bars]
    assert len(set(counts)) == len(counts), "frame counts must be unique"
    assert len(set(heights)) == len(heights), "bar heights must be unique"

    return _render_output(g.shape, baseline_row, bars, counts)


def _draw_case_count() -> int:
    global _CASE_COUNT_BAG
    if not _CASE_COUNT_BAG:
        _CASE_COUNT_BAG = [4, 5]
        random.shuffle(_CASE_COUNT_BAG)
    return _CASE_COUNT_BAG.pop()


def _choose_counts(n_bars: int) -> list[int]:
    while True:
        counts = random.sample(COUNT_VALUES, n_bars)
        if counts != sorted(counts):
            return counts


def _choose_heights(n_bars: int) -> list[int]:
    while True:
        heights = random.sample(BAR_HEIGHTS, n_bars)
        if heights != sorted(heights):
            return heights


def _pick_frame_size(count: int) -> tuple[int, int]:
    options = []
    for height in range(FRAME_SIDE_RANGE[0], FRAME_SIDE_RANGE[1] + 1):
        for width in range(FRAME_SIDE_RANGE[0], FRAME_SIDE_RANGE[1] + 1):
            if (height - 2) * (width - 2) >= count:
                options.append((height, width))
    return random.choice(options)


def _scatter_slack(total: int, parts: int) -> list[int]:
    slots = [0] * parts
    for _ in range(total):
        slots[random.randrange(parts)] += 1
    return slots


def _place_segments(width: int, lengths: list[int], inner_gap: int, outer_margin: int) -> list[int] | None:
    needed = sum(lengths) + inner_gap * (len(lengths) - 1) + 2 * outer_margin
    if needed > width:
        return None

    extras = _scatter_slack(width - needed, len(lengths) + 1)
    starts: list[int] = []
    cursor = outer_margin + extras[0]
    for index, length in enumerate(lengths):
        starts.append(cursor)
        cursor += length
        if index < len(lengths) - 1:
            cursor += inner_gap + extras[index + 1]
    return starts


def _render_input(case: dict) -> np.ndarray:
    height, width = case["shape"]
    grid = np.zeros((height, width), dtype=int)
    baseline_row = case["baseline_row"]
    grid[baseline_row, :] = BASELINE

    for slot in case["slots"]:
        grid[baseline_row - slot["height"] : baseline_row, slot["x"]] = slot["color"]

    for frame in case["frames"]:
        r0, c0, r1, c1 = frame["bbox"]
        grid[r0 : r1 + 1, c0] = FRAME
        grid[r0 : r1 + 1, c1] = FRAME
        grid[r0, c0 : c1 + 1] = FRAME
        grid[r1, c0 : c1 + 1] = FRAME
        for row, col in frame["inside_markers"]:
            grid[row, col] = MARKER

    for row, col in case["outside_markers"]:
        grid[row, col] = MARKER

    return grid


def _expected_from_case(case: dict) -> np.ndarray:
    counts = [frame["count"] for frame in case["frames"]]
    return _render_output(case["shape"], case["baseline_row"], case["slots"], counts)


def _frame_ring(box: tuple[int, int, int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = shape
    r0, c0, r1, c1 = box
    coords: list[tuple[int, int]] = []
    for row in range(r0 - 1, r1 + 2):
        for col in range(c0 - 1, c1 + 2):
            if not (0 <= row < height and 0 <= col < width):
                continue
            if r0 <= row <= r1 and c0 <= col <= c1:
                continue
            coords.append((row, col))
    random.shuffle(coords)
    return coords


def _free_mask(case: dict) -> np.ndarray:
    height, width = case["shape"]
    free = np.ones((height, width), dtype=bool)
    free[case["baseline_row"], :] = False
    for slot in case["slots"]:
        free[case["baseline_row"] - slot["height"] : case["baseline_row"], slot["x"]] = False
    for frame in case["frames"]:
        r0, c0, r1, c1 = frame["bbox"]
        free[r0 : r1 + 1, c0 : c1 + 1] = False
    return free


def _build_case(n_bars: int) -> dict | None:
    counts = _choose_counts(n_bars)
    heights = _choose_heights(n_bars)
    colors = random.sample(BAR_COLORS, n_bars)
    frame_sizes = [_pick_frame_size(count) for count in counts]

    min_bar_width = n_bars + 2 * (n_bars - 1) + 2
    min_frame_width = sum(width for _height, width in frame_sizes) + (n_bars - 1)
    width_lo = max(GRID_WIDTH_RANGE[0], min_bar_width, min_frame_width)
    if width_lo > GRID_WIDTH_RANGE[1]:
        return None
    width = random.randint(width_lo, GRID_WIDTH_RANGE[1])

    bar_cols = _place_segments(width, [1] * n_bars, inner_gap=2, outer_margin=1)
    frame_cols = _place_segments(width, [width_ for _height, width_ in frame_sizes], inner_gap=1, outer_margin=0)
    if bar_cols is None or frame_cols is None:
        return None

    max_height = max(heights)
    clear_gap = random.randint(1, 3)
    baseline_lo = max(max_height + clear_gap + 5, GRID_HEIGHT_RANGE[0] - 3)
    baseline_hi = min(GRID_HEIGHT_RANGE[1] - 1, baseline_lo + 5)
    if baseline_lo > baseline_hi:
        return None
    baseline_row = random.randint(baseline_lo, baseline_hi)

    bottom_margin = random.randint(0, min(2, GRID_HEIGHT_RANGE[1] - baseline_row - 1))
    height = baseline_row + 1 + bottom_margin
    if not (GRID_HEIGHT_RANGE[0] <= height <= GRID_HEIGHT_RANGE[1]):
        return None

    allowed_bottom = baseline_row - max_height - clear_gap - 1
    if allowed_bottom < 2:
        return None

    frames: list[dict] = []
    for index, (count, (frame_h, frame_w), c0) in enumerate(zip(counts, frame_sizes, frame_cols)):
        top_max = allowed_bottom - frame_h + 1
        if top_max < 0:
            return None
        r0 = random.randint(0, top_max)
        c1 = c0 + frame_w - 1
        r1 = r0 + frame_h - 1
        interior = [(row, col) for row in range(r0 + 1, r1) for col in range(c0 + 1, c1)]
        inside_markers = random.sample(interior, count)
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "count": count,
                "inside_markers": inside_markers,
                "index": index,
            }
        )

    slots = [
        {"x": col, "color": color, "height": height_}
        for col, color, height_ in zip(bar_cols, colors, heights)
    ]

    case = {
        "shape": (height, width),
        "baseline_row": baseline_row,
        "slots": slots,
        "frames": frames,
        "outside_markers": [],
    }

    free = _free_mask(case)
    outside_markers: list[tuple[int, int]] = []

    for frame in frames:
        ring = [coord for coord in _frame_ring(frame["bbox"], case["shape"]) if free[coord]]
        if not ring:
            return None
        need = random.randint(1, min(2, len(ring)))
        picked = random.sample(ring, need)
        for coord in picked:
            free[coord] = False
        outside_markers.extend(picked)

    upper_candidates = [(row, col) for row in range(height) for col in range(width) if free[row, col] and row < baseline_row]
    random.shuffle(upper_candidates)
    extra_upper = random.randint(1, min(4, len(upper_candidates))) if upper_candidates else 0
    for coord in upper_candidates[:extra_upper]:
        free[coord] = False
        outside_markers.append(coord)

    lower_candidates = [
        (row, col)
        for row in range(max(0, allowed_bottom + 1), baseline_row)
        for col in range(width)
        if free[row, col]
    ]
    random.shuffle(lower_candidates)
    extra_lower = random.randint(0, min(2, len(lower_candidates))) if lower_candidates else 0
    for coord in lower_candidates[:extra_lower]:
        free[coord] = False
        outside_markers.append(coord)

    if not outside_markers:
        return None

    case["outside_markers"] = outside_markers
    return case


def _case_is_semantic(case: dict, inp: np.ndarray, expected: np.ndarray) -> bool:
    counts = [frame["count"] for frame in case["frames"]]
    heights = [slot["height"] for slot in case["slots"]]
    if len(set(counts)) != len(counts):
        return False
    if len(set(heights)) != len(heights):
        return False
    if counts == sorted(counts):
        return False
    if heights == sorted(heights):
        return False
    if not case["outside_markers"]:
        return False
    if inp.shape != expected.shape or inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False

    input_counts = []
    for frame in case["frames"]:
        r0, c0, r1, c1 = frame["bbox"]
        input_counts.append(int(np.count_nonzero(inp[r0 + 1 : r1, c0 + 1 : c1] == MARKER)))
    if input_counts != counts:
        return False

    bars_out = _find_bars(expected, case["baseline_row"])
    if len(bars_out) != len(case["slots"]):
        return False
    expected_heights = [height for _count, height in sorted(zip(counts, heights))]
    if [bar["height"] for bar in bars_out] != expected_heights:
        return False
    return not np.array_equal(inp, expected)


def generate_input() -> np.ndarray:
    primary_count = _draw_case_count()
    count_order = [primary_count, 9 - primary_count]

    for n_bars in count_order:
        for _ in range(GEN_ATTEMPTS):
            case = _build_case(n_bars)
            if case is None:
                continue
            inp = _render_input(case)
            expected = _expected_from_case(case)
            if not _case_is_semantic(case, inp, expected):
                continue
            try:
                if np.array_equal(main(inp), expected):
                    return inp
            except Exception:
                continue
    raise RuntimeError("failed to generate a valid puzzle instance")
