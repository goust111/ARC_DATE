from __future__ import annotations

import numpy as np

from arc_utils import get_bg


def _single_divider_column(g: np.ndarray, frame: int) -> int:
    dividers = [c for c in range(1, g.shape[1] - 1) if np.all(g[:, c] == frame)]
    assert len(dividers) == 1, "expected exactly one full-height interior divider"
    return int(dividers[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the colored runs from the left reference panel into the blank right panel.

    Input structure:
    - A uniform rectangular frame surrounds the grid.
    - One full-height divider of the frame color splits a left reference panel
      from a blank right panel.
    - Both panels have one background margin column on each side of their active
      columns; the left active columns contain two-column endpoint slots, possibly
      connected by same-color horizontal runs.

    Rule:
    - Keep the original grid and paint only in the right panel's active columns.
    - A source run exactly matching the left endpoint slot is copied to the left
      two columns of the right active span.
    - A source run exactly matching the right endpoint slot is copied to the right
      two columns of the right active span.
    - A source run connecting the endpoint slots is stretched across the entire
      right active span.

    Key insight:
    The left panel is a compressed legend for the right panel: endpoint markers
    stay two pixels wide, while connectors between endpoints expand to fill the
    distance between them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    H, W = g.shape
    assert H >= 3 and W >= 7, "grid too small for framed two-panel structure"

    frame = int(g[0, 0])
    assert np.all(g[0, :] == frame) and np.all(g[-1, :] == frame), "bad horizontal frame"
    assert np.all(g[:, 0] == frame) and np.all(g[:, -1] == frame), "bad vertical frame"

    divider = _single_divider_column(g, frame)
    assert divider >= 5 and W - divider >= 6, "panels must contain margins and active slots"

    left_panel = g[1 : H - 1, 1:divider]
    left_bg = get_bg(left_panel)
    assert np.all(left_panel[:, 0] == left_bg), "left panel must have a left margin"
    assert np.all(left_panel[:, -1] == left_bg), "left panel must have a right margin"

    mini_start = 2
    mini_end = divider - 2
    assert mini_end - mini_start + 1 >= 4, "left active region must contain two endpoint slots"
    left_slot = (mini_start, mini_start + 1)
    right_slot = (mini_end - 1, mini_end)

    target_start = divider + 2
    target_end = W - 3
    assert target_end - target_start + 1 >= 4, "right active region must contain endpoint slots"
    right_bg = int(g[1, divider + 1])
    right_panel = g[1 : H - 1, divider + 1 : W - 1]
    assert np.all(right_panel == right_bg), "right panel must be blank before expansion"

    out = g.copy()
    painted_any = False
    for r in range(1, H - 1):
        c = mini_start
        while c <= mini_end:
            color = int(g[r, c])
            if color == left_bg:
                c += 1
                continue

            c0 = c
            while c <= mini_end and int(g[r, c]) == color:
                c += 1
            c1 = c - 1

            in_left_slot = (c0, c1) == left_slot
            in_right_slot = (c0, c1) == right_slot
            if in_left_slot:
                d0, d1 = target_start, target_start + 1
            elif in_right_slot:
                d0, d1 = target_end - 1, target_end
            else:
                assert c0 == left_slot[0] and c1 >= right_slot[0], "run must connect endpoint slots"
                d0, d1 = target_start, target_end

            assert np.all(out[r, d0 : d1 + 1] == right_bg), "expanded runs must not collide"
            out[r, d0 : d1 + 1] = color
            painted_any = True

    assert painted_any, "expected at least one colored source run"
    return out

# --------------- generator (att2) ---------------
# Color and retry controls.
LEFT_BG = 1
COLOR_IDS = tuple(range(10))
FRAME_COLORS = (0, 2, 3, 4, 5, 6, 7, 8, 9)
MAX_RUN_COLORS = 4
GENERATION_RETRIES = 300

# Canvas, group, and row layout bounds.
LEFT_ACTIVE_WIDTHS = range(4, 7)
RIGHT_ACTIVE_WIDTHS = range(11, 14)
TOTAL_HEIGHTS = (11, 15)
RIGHT_TO_LEFT_MIN_RATIO = 2
FIXED_WIDTH_CELLS = 7
NON_GROUP_ROWS = 4
GROUP_COUNTS = (3, 5)
GROUP_HEIGHT_BOUNDS = (1, 2)
GAP_BOUNDS = (1, 2)
FIRST_ACTIVE_ROW = 2

# Run pattern geometry.
RUN_WIDTH = 2
LEFT_ACTIVE_START = 2
RIGHT_TARGET_OFFSET = 2
BASE_PATTERNS = ("LEFT", "RIGHT", "FULL")
DOUBLE_PATTERN = "LEFT_RIGHT"

def _gen_sequences_with_sum(length: int, lo: int, hi: int, total: int) -> list[tuple[int, ...]]:
    if length == 0:
        return [()] if total == 0 else []
    if total < length * lo or total > length * hi:
        return []
    out = []
    for value in range(lo, hi + 1):
        for rest in _gen_sequences_with_sum(length - 1, lo, hi, total - value):
            out.append((value,) + rest)
    return out


def _gen_pattern_options(length: int, run_color_count: int, left_active_width: int) -> list[tuple[str, ...]]:
    base = list(BASE_PATTERNS)
    if run_color_count >= 2 and left_active_width >= RUN_WIDTH * 2 + 1:
        base.append(DOUBLE_PATTERN)

    options: list[tuple[str, ...]] = [()]
    for _ in range(length):
        options = [prefix + (pattern,) for prefix in options for pattern in base]

    return [
        patterns
        for patterns in options
        if sum(2 if pattern == DOUBLE_PATTERN else 1 for pattern in patterns) >= run_color_count
        and any(pattern in ("LEFT", "FULL", DOUBLE_PATTERN) for pattern in patterns)
        and any(pattern in ("RIGHT", "FULL", DOUBLE_PATTERN) for pattern in patterns)
    ]


def _gen_take_color(rng: np.random.Generator, palette: np.ndarray, needed: list[int], exclude: int | None = None) -> int:
    for i, color in enumerate(needed):
        if color != exclude:
            return needed.pop(i)
    choices = [int(color) for color in palette if int(color) != exclude]
    return int(rng.choice(choices))


def _gen_row_layout(rng: np.random.Generator, group_count: int, total_h: int):
    row_total = total_h - NON_GROUP_ROWS
    height_lo, height_hi = GROUP_HEIGHT_BOUNDS
    gap_lo, gap_hi = GAP_BOUNDS
    gap_count = group_count - 1
    height_options = []
    for height_sum in range(group_count * height_lo, group_count * height_hi + 1):
        gap_total = row_total - height_sum
        if gap_total < gap_count * gap_lo or gap_total > gap_count * gap_hi:
            continue
        height_seqs = _gen_sequences_with_sum(group_count, height_lo, height_hi, height_sum)
        gap_seqs = _gen_sequences_with_sum(gap_count, gap_lo, gap_hi, gap_total)
        height_options.extend((heights, gaps) for heights in height_seqs for gaps in gap_seqs)
    if not height_options:
        return None
    return height_options[int(rng.integers(len(height_options)))]


def _gen_patterns_and_colors(rng, group_count: int, left_w: int, left_bg: int, right_bg: int):
    max_segments = group_count * (2 if left_w >= RUN_WIDTH * 2 + 1 else 1)
    run_color_count = int(rng.integers(1, min(MAX_RUN_COLORS, max_segments) + 1))
    pattern_options = _gen_pattern_options(group_count, run_color_count, left_w)
    if not pattern_options:
        return None
    patterns = pattern_options[int(rng.integers(len(pattern_options)))]

    run_choices = [c for c in COLOR_IDS if c not in (left_bg, right_bg)]
    run_colors = rng.choice(run_choices, size=run_color_count, replace=False)
    needed_colors = [int(color) for color in run_colors]
    rng.shuffle(needed_colors)

    group_colors: list[tuple[int, ...]] = []
    for pattern in patterns:
        if pattern == DOUBLE_PATTERN:
            left_color = _gen_take_color(rng, run_colors, needed_colors)
            right_color = _gen_take_color(rng, run_colors, needed_colors, exclude=left_color)
            group_colors.append((left_color, right_color))
        else:
            group_colors.append((_gen_take_color(rng, run_colors, needed_colors),))
    return patterns, group_colors


def _render_grid(total_h: int, total_w: int, divider: int, left_w: int, colors, row_layout, pattern_data):
    frame_color, left_bg, right_bg = colors
    group_heights, group_gaps = row_layout
    patterns, group_colors = pattern_data
    out = np.full((total_h, total_w), right_bg, dtype=int)
    out[:, 0] = frame_color
    out[:, -1] = frame_color
    out[0, :] = frame_color
    out[-1, :] = frame_color
    out[:, divider] = frame_color
    out[1 : total_h - 1, 1:divider] = left_bg
    out[1 : total_h - 1, divider + 1 : total_w - 1] = right_bg
    left_start = LEFT_ACTIVE_START
    left_end = left_start + left_w - 1
    left_slot = slice(left_start, left_start + RUN_WIDTH)
    right_slot = slice(left_end - RUN_WIDTH + 1, left_end + 1)
    target_start = divider + RIGHT_TARGET_OFFSET
    target_end = total_w - 3

    row = FIRST_ACTIVE_ROW
    for idx, (pattern, colors) in enumerate(zip(patterns, group_colors)):
        for r in range(row, row + group_heights[idx]):
            if pattern == "LEFT":
                out[r, left_slot] = colors[0]
                out[r, target_start : target_start + RUN_WIDTH] = colors[0]
            elif pattern == "RIGHT":
                out[r, right_slot] = colors[0]
                out[r, target_end - RUN_WIDTH + 1 : target_end + 1] = colors[0]
            elif pattern == "FULL":
                out[r, left_start : left_end + 1] = colors[0]
                out[r, target_start : target_end + 1] = colors[0]
            else:
                out[r, left_slot] = colors[0]
                out[r, right_slot] = colors[1]
                out[r, target_start : target_start + RUN_WIDTH] = colors[0]
                out[r, target_end - RUN_WIDTH + 1 : target_end + 1] = colors[1]
        row += group_heights[idx]
        if idx < len(group_gaps):
            row += group_gaps[idx]
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    frame_color = int(rng.choice(FRAME_COLORS))
    right_bg = int(rng.choice([c for c in COLOR_IDS if c not in (LEFT_BG, frame_color)]))
    colors = (frame_color, LEFT_BG, right_bg)
    width_pairs = [
        (wl, wr)
        for wl in LEFT_ACTIVE_WIDTHS
        for wr in RIGHT_ACTIVE_WIDTHS
        if wr >= RIGHT_TO_LEFT_MIN_RATIO * wl
    ]
    left_w, right_w = width_pairs[int(rng.integers(len(width_pairs)))]
    total_w = left_w + right_w + FIXED_WIDTH_CELLS
    divider = left_w + 3
    group_count = int(rng.integers(*GROUP_COUNTS))
    total_h = int(rng.integers(*TOTAL_HEIGHTS))

    row_layout = _gen_row_layout(rng, group_count, total_h)
    pattern_data = _gen_patterns_and_colors(rng, group_count, left_w, LEFT_BG, right_bg)
    if row_layout is None or pattern_data is None:
        return None

    out = _render_grid(total_h, total_w, divider, left_w, colors, row_layout, pattern_data)
    inp = out.copy()
    inp[1 : total_h - 1, divider + 1 : total_w - 1] = right_bg
    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 9841fdad input")
