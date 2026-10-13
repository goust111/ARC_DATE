from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
FRAME = 5
WINDOW_COLORS = (1, 2, 3, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
SIDE_ORDER = {side: i for i, side in enumerate(SIDES)}

INNER_H_RANGE = (5, 10)
INNER_W_RANGE = (5, 10)
MARGIN_RANGE = (1, 4)
GEN_RETRIES = 400
EXTRA_WINDOW_TRIES = 80


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty rectangular frame"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _color_runs(values: np.ndarray, frame_color: int) -> list[dict[str, int | str]]:
    runs: list[dict[str, int | str]] = []
    i = 0
    while i < len(values):
        color = int(values[i])
        if color == frame_color:
            i += 1
            continue
        assert color != BLACK, "frame edges must stay non-black"
        j = i + 1
        while j < len(values) and int(values[j]) == color:
            j += 1
        runs.append({"start": i, "length": j - i, "color": color})
        i = j
    return runs


def _extract_windows(
    grid: np.ndarray,
) -> tuple[tuple[int, int, int, int], int, int, int, list[dict[str, int | str]]]:
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1 = _bbox(g != BLACK)
    frame_color = int(g[r0, c0])
    assert frame_color != BLACK, "frame corner must be colored"
    assert int(g[r0, c1]) == frame_color
    assert int(g[r1, c0]) == frame_color
    assert int(g[r1, c1]) == frame_color

    inner_h = int(r1 - r0 - 1)
    inner_w = int(c1 - c0 - 1)
    assert inner_h >= 1 and inner_w >= 1, "frame must enclose a non-empty interior"

    top = g[r0, c0 : c1 + 1]
    bottom = g[r1, c0 : c1 + 1]
    left = g[r0 : r1 + 1, c0]
    right = g[r0 : r1 + 1, c1]
    assert np.all(top != BLACK) and np.all(bottom != BLACK)
    assert np.all(left != BLACK) and np.all(right != BLACK)
    assert np.all(g[r0 + 1 : r1, c0 + 1 : c1] == BLACK), "interior must start black"

    windows: list[dict[str, int | str]] = []
    for run in _color_runs(g[r0, c0 + 1 : c1], frame_color):
        windows.append({"side": "top", **run})
    for run in _color_runs(g[r1, c0 + 1 : c1], frame_color):
        windows.append({"side": "bottom", **run})
    for run in _color_runs(g[r0 + 1 : r1, c0], frame_color):
        windows.append({"side": "left", **run})
    for run in _color_runs(g[r0 + 1 : r1, c1], frame_color):
        windows.append({"side": "right", **run})

    assert windows, "expected at least one colored border window"
    for window in windows:
        length = int(window["length"])
        side = str(window["side"])
        if side in ("top", "bottom"):
            assert length <= inner_h, "top/bottom window depth must fit the interior height"
        else:
            assert length <= inner_w, "left/right window depth must fit the interior width"

    return (r0, c0, r1, c1), frame_color, inner_h, inner_w, windows


def _paint_block(
    first: np.ndarray,
    conflict: np.ndarray,
    row_slice: slice,
    col_slice: slice,
    color: int,
) -> None:
    rr = range(*row_slice.indices(first.shape[0]))
    cc = range(*col_slice.indices(first.shape[1]))
    for r in rr:
        for c in cc:
            prev = int(first[r, c])
            if prev == -1:
                first[r, c] = int(color)
            elif prev != int(color):
                conflict[r, c] = True


def _apply_rays(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    (r0, c0, r1, c1), _frame_color, inner_h, inner_w, windows = _extract_windows(g)

    first = np.full((inner_h, inner_w), -1, dtype=int)
    conflict = np.zeros((inner_h, inner_w), dtype=bool)

    for window in windows:
        side = str(window["side"])
        start = int(window["start"])
        length = int(window["length"])
        color = int(window["color"])
        if side in ("top", "bottom"):
            assert start + length <= inner_w
        else:
            assert start + length <= inner_h

        if side == "top":
            row_slice = slice(0, length)
            col_slice = slice(start, start + length)
        elif side == "bottom":
            row_slice = slice(inner_h - length, inner_h)
            col_slice = slice(start, start + length)
        elif side == "left":
            row_slice = slice(start, start + length)
            col_slice = slice(0, length)
        else:
            assert side == "right"
            row_slice = slice(start, start + length)
            col_slice = slice(inner_w - length, inner_w)

        _paint_block(first, conflict, row_slice, col_slice, color)

    out = g.copy()
    interior = out[r0 + 1 : r1, c0 + 1 : c1]
    painted = first != -1
    interior[painted] = first[painted]
    interior[conflict] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire inward color rays from border windows, using window length as both width and depth.

    Each non-grey run on one frame side paints an inward square block whose side
    length equals the run length. Overlaps between different colors turn yellow.
    """
    return _apply_rays(grid)


def _span_length(side: str, inner_h: int, inner_w: int) -> int:
    return inner_w if side in ("top", "bottom") else inner_h


def _depth_limit(side: str, inner_h: int, inner_w: int) -> int:
    return inner_h if side in ("top", "bottom") else inner_w


def _interval_ok(intervals: list[tuple[int, int]], start: int, length: int) -> bool:
    end = start + length - 1
    for other_start, other_end in intervals:
        if not (end + 1 < other_start or other_end + 1 < start):
            return False
    return True


def _register_window(
    windows: list[dict[str, int | str]],
    used_by_side: dict[str, list[tuple[int, int]]],
    side: str,
    start: int,
    length: int,
    color: int,
) -> bool:
    if start < 0 or length <= 0 or not _interval_ok(used_by_side[side], start, length):
        return False
    used_by_side[side].append((start, start + length - 1))
    used_by_side[side].sort()
    windows.append({"side": side, "start": start, "length": length, "color": color})
    return True


def _sample_color(used_colors: list[int], avoid: set[int] | None = None) -> int:
    blocked = set() if avoid is None else set(avoid)
    if used_colors and random.random() < 0.35:
        reusable = [color for color in used_colors if color not in blocked]
        if reusable:
            return int(random.choice(reusable))
    fresh = [color for color in WINDOW_COLORS if color not in blocked and color not in used_colors]
    if fresh:
        return int(random.choice(fresh))
    return int(random.choice([color for color in WINDOW_COLORS if color not in blocked]))


def _corner_cross_windows(inner_h: int, inner_w: int) -> list[dict[str, int | str]] | None:
    common_max = min(inner_h, inner_w)
    if common_max < 2:
        return None

    windows: list[dict[str, int | str]] = []
    used_by_side = {side: [] for side in SIDES}
    corner = random.choice(("tl", "tr", "bl", "br"))
    len_a = random.randint(2, common_max)
    len_b = random.randint(2, common_max)
    color_a = _sample_color([])
    color_b = _sample_color([color_a], avoid={color_a})

    if corner == "tl":
        start_h = random.randint(0, min(inner_w - len_a, len_b - 1))
        start_v = random.randint(0, min(inner_h - len_b, len_a - 1))
        pairs = [("top", start_h, len_a, color_a), ("left", start_v, len_b, color_b)]
    elif corner == "tr":
        start_h = random.randint(max(0, inner_w - len_a - len_b + 1), inner_w - len_a)
        start_v = random.randint(0, min(inner_h - len_b, len_a - 1))
        pairs = [("top", start_h, len_a, color_a), ("right", start_v, len_b, color_b)]
    elif corner == "bl":
        start_h = random.randint(0, min(inner_w - len_a, len_b - 1))
        start_v = random.randint(max(0, inner_h - len_a - len_b + 1), inner_h - len_b)
        pairs = [("bottom", start_h, len_a, color_a), ("left", start_v, len_b, color_b)]
    else:
        start_h = random.randint(max(0, inner_w - len_a - len_b + 1), inner_w - len_a)
        start_v = random.randint(max(0, inner_h - len_a - len_b + 1), inner_h - len_b)
        pairs = [("bottom", start_h, len_a, color_a), ("right", start_v, len_b, color_b)]

    for side, start, length, color in pairs:
        if not _register_window(windows, used_by_side, side, start, length, color):
            return None

    used_colors = [color_a, color_b]
    for _ in range(random.randint(0, 2)):
        for _attempt in range(EXTRA_WINDOW_TRIES):
            side = random.choice(SIDES)
            max_len = min(_span_length(side, inner_h, inner_w), _depth_limit(side, inner_h, inner_w), 4)
            length = random.randint(1, max_len)
            start = random.randint(0, _span_length(side, inner_h, inner_w) - length)
            color = _sample_color(used_colors)
            if _register_window(windows, used_by_side, side, start, length, color):
                used_colors.append(color)
                break

    return windows


def _opposite_cross_windows(inner_h: int, inner_w: int) -> list[dict[str, int | str]] | None:
    windows: list[dict[str, int | str]] = []
    used_by_side = {side: [] for side in SIDES}
    orientation = random.choice(("vertical", "horizontal"))
    color_a = _sample_color([])
    color_b = _sample_color([color_a], avoid={color_a})

    if orientation == "vertical":
        common_max = min(inner_h, inner_w)
        if common_max < 2:
            return None
        for _ in range(80):
            len_top = random.randint(2, common_max)
            len_bottom = random.randint(2, common_max)
            if len_top + len_bottom > inner_h:
                break
        else:
            return None
        start_top = random.randint(0, inner_w - len_top)
        low = max(0, start_top - len_bottom + 1)
        high = min(inner_w - len_bottom, start_top + len_top - 1)
        start_bottom = random.randint(low, high)
        pairs = [
            ("top", start_top, len_top, color_a),
            ("bottom", start_bottom, len_bottom, color_b),
        ]
    else:
        common_max = min(inner_h, inner_w)
        if common_max < 2:
            return None
        for _ in range(80):
            len_left = random.randint(2, common_max)
            len_right = random.randint(2, common_max)
            if len_left + len_right > inner_w:
                break
        else:
            return None
        start_left = random.randint(0, inner_h - len_left)
        low = max(0, start_left - len_right + 1)
        high = min(inner_h - len_right, start_left + len_left - 1)
        start_right = random.randint(low, high)
        pairs = [
            ("left", start_left, len_left, color_a),
            ("right", start_right, len_right, color_b),
        ]

    for side, start, length, color in pairs:
        if not _register_window(windows, used_by_side, side, start, length, color):
            return None

    used_colors = [color_a, color_b]
    for _ in range(random.randint(0, 2)):
        for _attempt in range(EXTRA_WINDOW_TRIES):
            side = random.choice(SIDES)
            max_len = min(_span_length(side, inner_h, inner_w), _depth_limit(side, inner_h, inner_w), 4)
            length = random.randint(1, max_len)
            start = random.randint(0, _span_length(side, inner_h, inner_w) - length)
            color = _sample_color(used_colors)
            if _register_window(windows, used_by_side, side, start, length, color):
                used_colors.append(color)
                break

    return windows


def _disjoint_windows(inner_h: int, inner_w: int) -> list[dict[str, int | str]] | None:
    orientation = random.choice(("horizontal", "vertical"))
    if orientation == "horizontal":
        sides = ("top", "bottom")
        span = inner_w
        depth = inner_h
    else:
        sides = ("left", "right")
        span = inner_h
        depth = inner_w

    n_windows = random.randint(2, 4)
    max_len = min(span, depth)
    for _ in range(120):
        lengths = [random.randint(1, max_len) for _ in range(n_windows)]
        if sum(lengths) + (n_windows - 1) <= span:
            break
    else:
        return None

    base_gaps = [0] + [1] * (n_windows - 1) + [0]
    extra = span - (sum(lengths) + sum(base_gaps))
    gap_add = [0] * (n_windows + 1)
    for _ in range(extra):
        gap_add[random.randint(0, n_windows)] += 1
    gaps = [base_gaps[i] + gap_add[i] for i in range(n_windows + 1)]

    starts: list[int] = []
    pos = gaps[0]
    for i, length in enumerate(lengths):
        starts.append(pos)
        pos += length + gaps[i + 1]

    side_assignments = [sides[0], sides[1]]
    for _ in range(n_windows - 2):
        side_assignments.append(random.choice(sides))
    random.shuffle(side_assignments)

    windows: list[dict[str, int | str]] = []
    used_colors: list[int] = []
    for side, start, length in zip(side_assignments, starts, lengths):
        color = _sample_color(used_colors)
        used_colors.append(color)
        windows.append({"side": side, "start": start, "length": length, "color": color})
    return windows


def _balanced_margins() -> tuple[int, int]:
    a = random.randint(*MARGIN_RANGE)
    b = a + random.choice((-1, 0, 1))
    b = max(MARGIN_RANGE[0], min(MARGIN_RANGE[1], b))
    return int(a), int(b)


def _render_grid(inner_h: int, inner_w: int, windows: list[dict[str, int | str]]) -> np.ndarray:
    top_margin, bottom_margin = _balanced_margins()
    left_margin, right_margin = _balanced_margins()

    frame_h = inner_h + 2
    frame_w = inner_w + 2
    grid_h = top_margin + frame_h + bottom_margin
    grid_w = left_margin + frame_w + right_margin

    grid = np.zeros((grid_h, grid_w), dtype=int)
    r0 = top_margin
    c0 = left_margin
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1

    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME

    for window in windows:
        side = str(window["side"])
        start = int(window["start"])
        length = int(window["length"])
        color = int(window["color"])
        if side == "top":
            grid[r0, c0 + 1 + start : c0 + 1 + start + length] = color
        elif side == "bottom":
            grid[r1, c0 + 1 + start : c0 + 1 + start + length] = color
        elif side == "left":
            grid[r0 + 1 + start : r0 + 1 + start + length, c0] = color
        else:
            grid[r0 + 1 + start : r0 + 1 + start + length, c1] = color

    return grid


def _normalize_windows(windows: list[dict[str, int | str]]) -> list[tuple[int, int, int, int]]:
    return sorted(
        (
            SIDE_ORDER[str(window["side"])],
            int(window["start"]),
            int(window["length"]),
            int(window["color"]),
        )
        for window in windows
    )


def _gen_attempt() -> np.ndarray | None:
    inner_h = random.randint(*INNER_H_RANGE)
    inner_w = random.randint(*INNER_W_RANGE)
    mode = random.choice(("corner_cross", "opposite_cross", "disjoint"))

    if mode == "corner_cross":
        windows = _corner_cross_windows(inner_h, inner_w)
    elif mode == "opposite_cross":
        windows = _opposite_cross_windows(inner_h, inner_w)
    else:
        windows = _disjoint_windows(inner_h, inner_w)

    if windows is None:
        return None

    grid = _render_grid(inner_h, inner_w, windows)

    try:
        _bbox_data, frame_color, parsed_h, parsed_w, parsed_windows = _extract_windows(grid)
        out = main(grid)
    except Exception:
        return None

    if frame_color != FRAME or parsed_h != inner_h or parsed_w != inner_w:
        return None
    if _normalize_windows(parsed_windows) != _normalize_windows(windows):
        return None

    changed = int(np.count_nonzero(out != grid))
    has_yellow = bool(np.any(out == YELLOW))
    if changed == 0:
        return None
    if mode == "disjoint" and has_yellow:
        return None
    if mode != "disjoint" and not has_yellow:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-window puzzle")
