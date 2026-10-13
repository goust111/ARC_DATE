from __future__ import annotations

import numpy as np

from arc_utils import Color, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the red start pixel into a recursive rectilinear ray network that turns at colored marker pixels.

    Input structure
    - The canvas has black background, one red start pixel on the left edge, and isolated colored marker pixels.
    - Original non-red, non-background marker pixels are blockers and are preserved.
    - Marker colors are green, pink, orange, and teal; orange markers on the lower border are terminal when hit from above.

    Rule
    - Start with a horizontal red ray from the red edge pixel.
    - A ray paints background cells red until it reaches the first original marker in each direction of its row or column.
    - The first time a marker opens, emit the perpendicular row or column from the red cell immediately before that marker; later hits on the same marker only stop the ray.
    - Process emitted rows/columns depth-first, with right/down emissions taking priority over left/up emissions when both are discovered together.
    - If a downward vertical ray reaches an orange marker on the lower border, stop there without opening a new horizontal ray.

    Key insight
    - The output is not a global fill: it is the depth-first closure of marker-triggered perpendicular ray turns, with marker pixels acting as one-time controls.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"

    starts = np.argwhere(g == Color.RED)
    assert len(starts) == 1, "expected exactly one red start pixel"
    start_r, start_c = map(int, starts[0])
    assert start_c == 0, "red start pixel must be on the left edge"

    marker_mask = (g != bg) & (g != Color.RED)
    assert np.any(marker_mask), "expected colored marker pixels"
    marker_colors = set(map(int, np.unique(g[marker_mask])))
    assert marker_colors <= {Color.GREEN, Color.PINK, Color.ORANGE, Color.TEAL}

    H, W = g.shape
    out = g.copy()
    blockers = marker_mask
    pending: list[tuple[int, int, str]] = [(start_r, start_c, "h")]
    seen_segments: set[tuple[int, int, str]] = set()
    opened_markers: set[tuple[int, int]] = set()

    while pending:
        r, c, axis = pending.pop()
        segment_key = (r, c, axis)
        if segment_key in seen_segments:
            continue
        seen_segments.add(segment_key)
        out[r, c] = Color.RED

        if axis == "h":
            directions = [(0, -1), (0, 1)]
            next_axis = "v"
        else:
            directions = [(-1, 0), (1, 0)]
            next_axis = "h"

        new_segments: list[tuple[int, int, str]] = []
        for dr, dc in directions:
            prev_r, prev_c = r, c
            rr, cc = r + dr, c + dc
            while 0 <= rr < H and 0 <= cc < W:
                if blockers[rr, cc]:
                    marker = (int(rr), int(cc))
                    terminal_orange = (
                        axis == "v"
                        and dr == 1
                        and rr == H - 1
                        and int(g[rr, cc]) == Color.ORANGE
                    )
                    if marker not in opened_markers and not terminal_orange:
                        opened_markers.add(marker)
                        new_segments.append((prev_r, prev_c, next_axis))
                    break

                out[rr, cc] = Color.RED
                prev_r, prev_c = rr, cc
                rr += dr
                cc += dc

        pending.extend(new_segments)

    return out

# --------------- generator (o5x) ---------------

# Colors and marker palette.
_GEN_BLACK, _GEN_RED, _GEN_ORANGE, _GEN_MARKER_COLORS = 0, 2, 7, (3, 6, 7, 8)

# Canvas and start placement bounds.
_GEN_HEIGHT_RANGE, _GEN_WIDTH_RANGE = (4, 12), (6, 12)
_GEN_START_ROW_LO_DIVISOR, _GEN_START_ROW_HI_DIVISOR, _GEN_START_ROW_HI_OFFSET = 5, 2, 1

# Ray-network construction limits.
_GEN_DEPTH_RANGE, _GEN_OPENINGS_RANGE, _GEN_SEGMENT_COUNT_RANGE = (1, 6), (1, 10), (2, 11)
_GEN_MAX_SEEN_SEGMENTS, _GEN_DEPTH_BRANCH_CAP = 11, 4
_GEN_BRANCH_TWO_PROB, _GEN_MIN_OPENING_DIST = 0.45, 2

# Marker and output constraints.
_GEN_COLOR_COUNT_RANGE, _GEN_MARKER_COUNT_RANGE = (1, 4), (1, 19)
_GEN_MAX_DISTRACTORS, _GEN_RED_RATIO_RANGE, _GEN_MAX_ATTEMPTS = 12, (0.20, 0.42), 300


def _gen_init_canvas():
    H = random.randint(*_GEN_HEIGHT_RANGE)
    W = random.randint(*_GEN_WIDTH_RANGE)
    grid = np.zeros((H, W), dtype=int)
    start_lo = H // _GEN_START_ROW_LO_DIVISOR
    start_hi = max(start_lo, H // _GEN_START_ROW_HI_DIVISOR - _GEN_START_ROW_HI_OFFSET)
    start_r = random.randint(start_lo, start_hi)
    grid[start_r, 0] = _GEN_RED
    return grid, start_r


def _gen_dirs(axis):
    if axis == "h":
        return ((0, -1), (0, 1)), "v"
    return ((-1, 0), (1, 0)), "h"


def _gen_first_blocker(grid, r, c, dr, dc):
    H, W = grid.shape
    prev_r, prev_c = r, c
    rr, cc = r + dr, c + dc
    while 0 <= rr < H and 0 <= cc < W:
        if grid[rr, cc] not in (_GEN_BLACK, _GEN_RED):
            return int(rr), int(cc), int(prev_r), int(prev_c)
        prev_r, prev_c = rr, cc
        rr += dr
        cc += dc
    return None


def _gen_opening_candidates(grid, r, c, axis, dr, dc, color_pool):
    H, W = grid.shape
    ans = []
    rr, cc = r + dr, c + dc
    dist = 1
    while 0 <= rr < H and 0 <= cc < W:
        if grid[rr, cc] not in (_GEN_BLACK, _GEN_RED):
            break
        if dist >= _GEN_MIN_OPENING_DIST and grid[rr, cc] == _GEN_BLACK:
            valid_colors = list(color_pool)
            if axis == "v" and dr == 1 and rr == H - 1:
                valid_colors = [x for x in valid_colors if x != _GEN_ORANGE]
            if valid_colors:
                ans.append((int(rr), int(cc), int(rr - dr), int(cc - dc), valid_colors))
        rr += dr
        cc += dc
        dist += 1
    return ans


def _gen_apply_available_openings(grid, available, remaining, opened, next_axis, depth):
    new_segments = []
    added = 0
    if remaining <= 0 or not available:
        return new_segments, added

    max_add = min(remaining, len(available))
    add_count = 1
    if max_add >= 2 and random.random() < _GEN_BRANCH_TWO_PROB:
        add_count = 2
    selected = set(random.sample([x[0] for x in available], add_count))

    for dir_i, dr, dc, candidates in available:
        if dir_i not in selected:
            continue
        mr, mc, prev_r, prev_c, valid_colors = random.choice(candidates)
        grid[mr, mc] = random.choice(valid_colors)
        opened.add((mr, mc))
        added += 1
        new_segments.append((prev_r, prev_c, next_axis, depth + 1))
    return new_segments, added


def _gen_build_network(grid, color_pool, target_depth, target_openings):
    start_r = int(np.argwhere(grid == _GEN_RED)[0, 0])
    pending = [(start_r, 0, "h", 0)]
    seen = set()
    opened = set()
    placed = 0
    max_depth = 0

    while pending and len(seen) <= _GEN_MAX_SEEN_SEGMENTS:
        r, c, axis, depth = pending.pop()
        key = (r, c, axis)
        if key in seen:
            continue
        seen.add(key)
        max_depth = max(max_depth, depth)

        dirs, next_axis = _gen_dirs(axis)
        new_segments = []
        available = []
        for dir_i, (dr, dc) in enumerate(dirs):
            hit = _gen_first_blocker(grid, r, c, dr, dc)
            if hit is not None:
                mr, mc, prev_r, prev_c = hit
                marker = (mr, mc)
                terminal = axis == "v" and dr == 1 and mr == grid.shape[0] - 1 and grid[mr, mc] == _GEN_ORANGE
                if marker not in opened and not terminal:
                    opened.add(marker)
                    placed += 1
                    new_segments.append((prev_r, prev_c, next_axis, depth + 1))
                continue

            if depth < target_depth and placed < target_openings:
                candidates = _gen_opening_candidates(grid, r, c, axis, dr, dc, color_pool)
                if candidates:
                    available.append((dir_i, dr, dc, candidates))

        emitted, added = _gen_apply_available_openings(grid, available, target_openings - placed, opened, next_axis, depth)
        placed += added
        new_segments.extend(emitted)

        pending.extend(new_segments)

    return placed, len(seen), max_depth


def _gen_add_distractors(grid, active_out, color_pool):
    marker_count = int(np.count_nonzero((grid != _GEN_BLACK) & (grid != _GEN_RED)))
    safe = np.argwhere((active_out == _GEN_BLACK) & (grid == _GEN_BLACK))
    if marker_count > _GEN_MARKER_COUNT_RANGE[1]:
        return False
    max_distractors = min(_GEN_MAX_DISTRACTORS, _GEN_MARKER_COUNT_RANGE[1] - marker_count, len(safe))
    distractor_count = random.randint(0, max_distractors)
    if distractor_count:
        order = np.random.permutation(len(safe))[:distractor_count]
        for idx in order:
            rr, cc = map(int, safe[idx])
            grid[rr, cc] = random.choice(color_pool)
    return True


def _gen_attempt() -> np.ndarray | None:
    grid, start_r = _gen_init_canvas()
    H, W = grid.shape
    color_count = random.randint(*_GEN_COLOR_COUNT_RANGE)
    color_pool = tuple(random.sample(_GEN_MARKER_COLORS, color_count))
    target_depth = random.randint(*_GEN_DEPTH_RANGE)
    max_openings = min(_GEN_OPENINGS_RANGE[1], 2 ** min(target_depth, _GEN_DEPTH_BRANCH_CAP) - 1)
    target_openings = random.randint(_GEN_OPENINGS_RANGE[0], max_openings)

    opened_count, segment_count, max_depth = _gen_build_network(
        grid, color_pool, target_depth, target_openings
    )
    if not (_GEN_OPENINGS_RANGE[0] <= opened_count <= _GEN_OPENINGS_RANGE[1]):
        return None
    if not (_GEN_SEGMENT_COUNT_RANGE[0] <= segment_count <= _GEN_SEGMENT_COUNT_RANGE[1]):
        return None
    if not (_GEN_DEPTH_RANGE[0] <= max_depth <= _GEN_DEPTH_RANGE[1]):
        return None

    try:
        active_out = main(grid)
    except Exception:
        return None
    if not _gen_add_distractors(grid, active_out, color_pool):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if not np.array_equal(main(grid), out):
        return None
    red_ratio = float(np.count_nonzero(out == _GEN_RED)) / float(H * W)
    if not (_GEN_RED_RATIO_RANGE[0] <= red_ratio <= _GEN_RED_RATIO_RANGE[1]):
        return None

    total_markers = int(np.count_nonzero((grid != _GEN_BLACK) & (grid != _GEN_RED)))
    if not (_GEN_MARKER_COUNT_RANGE[0] <= total_markers <= _GEN_MARKER_COUNT_RANGE[1]):
        return None
    if not np.any(grid[start_r, 1:] != _GEN_BLACK):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate b942fd60 input")
