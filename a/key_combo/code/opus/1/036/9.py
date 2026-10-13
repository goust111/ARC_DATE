from __future__ import annotations

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

CONTROL_COLORS = (BLUE, RED, GREEN, YELLOW)
DEPTH_BY_COLOR = {
    BLUE: 2,
    RED: 1,
    GREEN: 3,
    YELLOW: 0,
}
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Generator controls.
WIDTH_CHOICES = (4, 5, 6)
HEIGHT_CHOICES = (2, 3, 4)
LANE_BASES = (2, 8, 14)
LANE_BAND = 4
PROFILE_CYCLE = (
    {"n_choices": (4,), "lane_choices": (2,), "require_overlap": False, "force_same_width_pair": False},
    {"n_choices": (5,), "lane_choices": (3,), "require_overlap": True, "force_same_width_pair": False},
    {"n_choices": (6,), "lane_choices": (2, 3), "require_overlap": True, "force_same_width_pair": True},
    {"n_choices": (5, 6), "lane_choices": (2, 3), "require_overlap": False, "force_same_width_pair": True},
)
MAX_GENERATION_ATTEMPTS = 400

_RNG = np.random.default_rng()
_GEN_CALL_COUNT = 0


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            components.append(
                {
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(coords.shape[0]),
                }
            )

    return components


def _control_markers(grid: np.ndarray) -> list[tuple[int, int]]:
    markers = []
    for col, value in enumerate(grid[0]):
        color = int(value)
        if color == BG:
            continue
        assert color in CONTROL_COLORS, "top-row markers must use control colors only"
        markers.append((col, color))
    assert markers, "expected at least one control marker"
    return markers


def _grey_rectangles(grid: np.ndarray) -> list[dict]:
    rectangles = []
    for comp in _components_of_color(grid, GREY):
        r0, c0, r1, c1 = comp["bbox"]
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert int(comp["area"]) == height * width, "grey objects must be solid rectangles"
        assert r0 > 0, "grey rectangles must stay below the top row"
        rectangles.append(
            {
                "top": r0,
                "left": c0,
                "bottom": r1,
                "right": c1,
                "height": height,
                "width": width,
            }
        )
    return rectangles


def _marker_color_for_rect(rect: dict, markers: list[tuple[int, int]]) -> int:
    matches = [color for col, color in markers if rect["left"] <= col <= rect["right"]]
    assert len(matches) == 1, "each rectangle must contain exactly one marker column"
    return int(matches[0])


def main(grid: np.ndarray) -> np.ndarray:
    """Crop each grey rectangle from the left by a control-color depth and repaint the remainder with that same control color.

    Input structure:
    - The grid has a black background, singleton colored control pixels in the top row, and several solid grey 4-connected rectangles below.
    - Each grey rectangle stays away from the top row and its bbox column span contains exactly one control-pixel column.
    - Control pixels use the ARC colors blue, red, green, or yellow.

    Rule:
    - For each grey rectangle, find the unique top-row control pixel whose column lies inside the rectangle's bbox column span.
    - Use that control color both as the repaint color and as the crop-depth code:
      blue -> erase 2 leftmost local columns, red -> erase 1, green -> erase 3, yellow -> erase 0.
    - Recolor the remaining rectangle cells with the same control color.
    - Leave the background and the control pixels unchanged.

    Key insight:
    - The same visible marker color carries two coupled instructions at once: how many columns to delete in the rectangle's own local frame, and which color to paint onto the surviving cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.all((g[1:] == BG) | (g[1:] == GREY)), "only top-row markers may use non-grey colors"

    markers = _control_markers(g)
    rectangles = _grey_rectangles(g)
    assert 4 <= len(rectangles) <= 6, "expected 4-6 grey rectangles"

    out = g.copy()
    for rect in rectangles:
        color = _marker_color_for_rect(rect, markers)
        depth = DEPTH_BY_COLOR[color]
        top = rect["top"]
        left = rect["left"]
        height = rect["height"]
        width = rect["width"]

        cut = min(depth, width)
        if cut:
            out[top : top + height, left : left + cut] = BG
        if cut < width:
            out[top : top + height, left + cut : left + width] = color

    return out


def _oracle_from_spec(inp: np.ndarray, rects: list[dict]) -> np.ndarray:
    out = inp.copy()
    for rect in rects:
        depth = DEPTH_BY_COLOR[int(rect["color"])]
        top = int(rect["top"])
        left = int(rect["left"])
        height = int(rect["height"])
        width = int(rect["width"])
        cut = min(depth, width)
        if cut:
            out[top : top + height, left : left + cut] = BG
        if cut < width:
            out[top : top + height, left + cut : left + width] = int(rect["color"])
    return out


def _fixed_depth_oracle(inp: np.ndarray, depth: int) -> np.ndarray:
    markers = _control_markers(inp)
    rects = _grey_rectangles(inp)
    out = inp.copy()
    for rect in rects:
        color = _marker_color_for_rect(rect, markers)
        top = rect["top"]
        left = rect["left"]
        height = rect["height"]
        width = rect["width"]
        cut = min(depth, width)
        if cut:
            out[top : top + height, left : left + cut] = BG
        if cut < width:
            out[top : top + height, left + cut : left + width] = color
    return out


def _sample_colors(n_rects: int) -> list[int]:
    colors = list(CONTROL_COLORS)
    extra = n_rects - len(CONTROL_COLORS)
    if extra > 0:
        extras = _RNG.choice(np.array(CONTROL_COLORS, dtype=int), size=extra, replace=False)
        colors.extend(int(x) for x in extras.tolist())
    _RNG.shuffle(colors)
    return colors


def _sample_markers(n_rects: int) -> tuple[list[int], int] | None:
    lead = int(_RNG.integers(1, 4))
    markers = [lead]
    for _ in range(n_rects - 1):
        gap = int(_RNG.integers(4, 7))
        markers.append(markers[-1] + gap)
    width = markers[-1] + int(_RNG.integers(2, 5))
    if width > 30:
        return None
    return markers, width


def _interval_candidates(marker: int, prev_marker: int, next_marker: int, canvas_w: int) -> list[tuple[int, int]]:
    candidates = []
    for width in WIDTH_CHOICES:
        lo = max(0, prev_marker + 1, marker - width + 1)
        hi = min(marker, next_marker - width, canvas_w - width)
        if lo > hi:
            continue
        starts = list(range(lo, hi + 1))
        _RNG.shuffle(starts)
        for left in starts:
            candidates.append((left, width))
    _RNG.shuffle(candidates)
    return candidates


def _sample_intervals(markers: list[int], canvas_w: int, require_overlap: bool) -> list[tuple[int, int, int]] | None:
    n_rects = len(markers)
    all_candidates = []
    for i, marker in enumerate(markers):
        prev_marker = markers[i - 1] if i > 0 else -1
        next_marker = markers[i + 1] if i + 1 < n_rects else canvas_w
        candidates = _interval_candidates(marker, prev_marker, next_marker, canvas_w)
        if not candidates:
            return None
        all_candidates.append(candidates)

    intervals: list[tuple[int, int, int]] = []

    def backtrack(i: int, overlap_count: int) -> list[tuple[int, int, int]] | None:
        if i == n_rects:
            if require_overlap and overlap_count == 0:
                return None
            return intervals.copy()

        for left, width in all_candidates[i]:
            right = left + width - 1
            new_overlap = overlap_count
            if intervals and left <= intervals[-1][1]:
                new_overlap += 1
            intervals.append((left, right, width))
            result = backtrack(i + 1, new_overlap)
            if result is not None:
                return result
            intervals.pop()
        return None

    return backtrack(0, 0)


def _assign_lanes(intervals: list[tuple[int, int, int]], lane_count: int, min_used_lanes: int) -> list[int] | None:
    n_rects = len(intervals)
    conflicts = [[False] * n_rects for _ in range(n_rects)]
    for i in range(n_rects):
        left_i, right_i, _ = intervals[i]
        for j in range(i + 1, n_rects):
            left_j, right_j, _ = intervals[j]
            same_lane_bad = not (right_i + 1 < left_j or right_j + 1 < left_i)
            conflicts[i][j] = same_lane_bad
            conflicts[j][i] = same_lane_bad

    assignment = [-1] * n_rects

    def backtrack(i: int) -> list[int] | None:
        if i == n_rects:
            if len(set(assignment)) < min_used_lanes:
                return None
            return assignment.copy()

        lanes = list(range(lane_count))
        _RNG.shuffle(lanes)
        for lane in lanes:
            if any(assignment[j] == lane and conflicts[i][j] for j in range(i)):
                continue
            assignment[i] = lane
            result = backtrack(i + 1)
            if result is not None:
                return result
            assignment[i] = -1
        return None

    return backtrack(0)


def _build_rects(profile: dict) -> tuple[np.ndarray, list[dict], np.ndarray] | None:
    n_rects = int(_RNG.choice(profile["n_choices"]))
    lane_count = int(_RNG.choice(profile["lane_choices"]))
    markers_and_width = _sample_markers(n_rects)
    if markers_and_width is None:
        return None
    marker_cols, canvas_w = markers_and_width

    intervals = _sample_intervals(marker_cols, canvas_w, profile["require_overlap"])
    if intervals is None:
        return None

    min_used_lanes = min(2, lane_count)
    if n_rects >= 6:
        min_used_lanes = min(3, lane_count)
    lanes = _assign_lanes(intervals, lane_count, min_used_lanes)
    if lanes is None:
        return None

    colors = _sample_colors(n_rects)
    top_shift = int(_RNG.integers(0, 2))
    lane_bases = [base + top_shift for base in LANE_BASES[:lane_count]]
    canvas_h = lane_bases[-1] + LANE_BAND + int(_RNG.integers(0, 3))
    if canvas_h > 30:
        return None

    rects = []
    for i, (marker_col, (left, right, width), lane, color) in enumerate(zip(marker_cols, intervals, lanes, colors)):
        height = int(_RNG.choice(HEIGHT_CHOICES))
        top_lo = lane_bases[lane]
        top_hi = lane_bases[lane] + LANE_BAND - height
        if top_lo > top_hi:
            return None
        top = int(_RNG.integers(top_lo, top_hi + 1))
        rects.append(
            {
                "id": i,
                "top": top,
                "left": left,
                "height": height,
                "width": width,
                "bottom": top + height - 1,
                "right": right,
                "lane": lane,
                "marker_col": marker_col,
                "color": color,
            }
        )

    inp = np.full((canvas_h, canvas_w), BG, dtype=int)
    for rect in rects:
        inp[0, int(rect["marker_col"])] = int(rect["color"])
        t, l = int(rect["top"]), int(rect["left"])
        h, w = int(rect["height"]), int(rect["width"])
        inp[t : t + h, l : l + w] = GREY

    out = _oracle_from_spec(inp, rects)
    return inp, rects, out


def _rectangles_are_separated(rects: list[dict]) -> bool:
    for i, a in enumerate(rects):
        for b in rects[i + 1 :]:
            rows_touch = not (a["bottom"] + 1 < b["top"] or b["bottom"] + 1 < a["top"])
            cols_touch = not (a["right"] + 1 < b["left"] or b["right"] + 1 < a["left"])
            if rows_touch and cols_touch:
                return False
    return True


def _quality_checks(inp: np.ndarray, rects: list[dict], expected: np.ndarray, profile: dict) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if not _rectangles_are_separated(rects):
        return False

    colors = {int(rect["color"]) for rect in rects}
    if colors != set(CONTROL_COLORS):
        return False

    widths = [int(rect["width"]) for rect in rects]
    heights = [int(rect["height"]) for rect in rects]
    offsets = [int(rect["marker_col"]) - int(rect["left"]) for rect in rects]
    if len(set(widths)) < 2 or len(set(heights)) < 2:
        return False
    if len(set(offsets)) < (3 if len(rects) >= 5 else 2):
        return False
    if sum(0 < off < width - 1 for off, width in zip(offsets, widths)) < 2:
        return False

    if profile["force_same_width_pair"]:
        if not any(
            widths[i] == widths[j] and rects[i]["color"] != rects[j]["color"]
            for i in range(len(rects))
            for j in range(i + 1, len(rects))
        ):
            return False

    overlap_count = sum(
        int(rects[i + 1]["left"]) <= int(rects[i]["right"])
        for i in range(len(rects) - 1)
    )
    if profile["require_overlap"] and overlap_count == 0:
        return False

    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    for fixed_depth in (0, 1, 2, 3):
        if np.array_equal(_fixed_depth_oracle(inp, fixed_depth), expected):
            return False

    return True


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    profile = PROFILE_CYCLE[_GEN_CALL_COUNT % len(PROFILE_CYCLE)]
    _GEN_CALL_COUNT += 1

    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_rects(profile)
        if built is None:
            continue
        inp, rects, expected = built
        if _quality_checks(inp, rects, expected, profile):
            return inp

    raise RuntimeError("failed to generate a valid input for puzzle 9")
