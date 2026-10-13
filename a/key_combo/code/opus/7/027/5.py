import numpy as np


BG = 0
MARKER = 5
BAR_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MIN_BARS = 6
MAX_BARS = 8
MIN_HEIGHT = 3
MAX_HEIGHT = 10
MAX_ADJ_DIFF = 4
MAX_GENERATION_ATTEMPTS = 500

_RNG = np.random.default_rng()


def _rotate_right(values: list[int], steps: int) -> list[int]:
    if not values:
        return []
    k = int(steps) % len(values)
    if k == 0:
        return list(values)
    return list(values[-k:] + values[:-k])


def _route_indices(count: int, start: int, end: int, step: int) -> list[int]:
    route = [int(start)]
    cur = int(start)
    while cur != int(end):
        cur = (cur + int(step)) % int(count)
        route.append(cur)
    return route


def _route_cost(heights: list[int], route: list[int]) -> int:
    return int(
        sum(abs(int(heights[a]) - int(heights[b])) for a, b in zip(route, route[1:]))
    )


def _sorted_path(route: list[int]) -> list[int]:
    return sorted(set(map(int, route)))


def _is_wrap_subset(indices: list[int]) -> bool:
    ordered = sorted(map(int, indices))
    return ordered != list(range(ordered[0], ordered[-1] + 1))


def _extract_bars(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    marker_coords = np.argwhere(g == MARKER)
    assert marker_coords.shape[0] == 2, "expected exactly two gray markers"

    bar_cols = [c for c in range(w) if np.any((g[:, c] != BG) & (g[:, c] != MARKER))]
    assert MIN_BARS <= len(bar_cols) <= MAX_BARS, "unexpected bar count"

    bars: list[dict] = []
    marked_cols = {int(c) for _r, c in marker_coords}
    for col in bar_cols:
        cells = np.where((g[:, col] != BG) & (g[:, col] != MARKER))[0]
        assert cells.size > 0
        top = int(cells.min())
        bottom = int(cells.max())
        assert bottom == h - 1, "bars must be bottom-aligned"
        assert np.array_equal(cells, np.arange(top, h)), "bar must be contiguous"
        colors = set(map(int, g[cells, col]))
        assert len(colors) == 1, "bar must be monochrome"
        color = int(next(iter(colors)))
        assert color in BAR_COLORS, "unexpected bar color"
        marker_row = top - 1
        is_marked = False
        if col in marked_cols:
            assert marker_row >= 0 and int(g[marker_row, col]) == MARKER, "marker must sit above bar top"
            is_marked = True
        bars.append(
            {
                "col": int(col),
                "height": int(h - top),
                "color": color,
                "top": top,
                "marked": is_marked,
            }
        )

    cols = [bar["col"] for bar in bars]
    for a, b in zip(cols, cols[1:]):
        assert 1 <= b - a <= 2, "bars must be separated by gap 0 or 1"

    assert len({bar["color"] for bar in bars}) == len(bars), "bar colors must be distinct"
    heights = [bar["height"] for bar in bars]
    for a, b in zip(heights, heights[1:]):
        assert abs(a - b) <= MAX_ADJ_DIFF, "adjacent bar heights differ too much"

    marked = [i for i, bar in enumerate(bars) if bar["marked"]]
    assert len(marked) == 2
    return bars


def _analyze_routes(bars: list[dict]) -> dict:
    heights = [int(bar["height"]) for bar in bars]
    marked = [i for i, bar in enumerate(bars) if bar["marked"]]
    start, end = marked
    n = len(bars)

    forward = _route_indices(n, start, end, 1)
    backward = _route_indices(n, start, end, -1)
    forward_cost = _route_cost(heights, forward)
    backward_cost = _route_cost(heights, backward)
    assert forward_cost != backward_cost, "expected a unique cheaper cyclic route"

    chosen = forward if forward_cost < backward_cost else backward
    rejected = backward if chosen is forward else forward
    ordered = _sorted_path(chosen)
    steps = len(chosen) - 1
    assert 1 <= steps < len(chosen), "use move count as non-trivial shift"

    return {
        "start": start,
        "end": end,
        "forward": forward,
        "backward": backward,
        "forward_cost": forward_cost,
        "backward_cost": backward_cost,
        "chosen": chosen,
        "rejected": rejected,
        "ordered": ordered,
        "steps": steps,
        "wraps": _is_wrap_subset(ordered),
    }


def _recolor_bars(bars: list[dict], info: dict) -> list[int]:
    colors = [int(bar["color"]) for bar in bars]
    ordered = info["ordered"]
    path_colors = [colors[i] for i in ordered]
    rotated = _rotate_right(path_colors, info["steps"])
    out_colors = list(colors)
    for idx, color in zip(ordered, rotated):
        out_colors[int(idx)] = int(color)
    return out_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the bars on the cheaper cyclic ridge route between the two marked bar tops.

    Input structure:
    - The grid has black background, 6-8 bottom-aligned one-cell-wide colored bars, and exactly two gray markers.
    - Each gray marker sits exactly one cell above a bar top, and the ordered bar columns define a cyclic left-right ridge.

    Rule:
    - Treat each bar top as a node, with movement allowed only between neighboring bar slots in cyclic order.
    - The cost of moving between neighboring bars is the absolute difference of their heights; choose the unique cheaper route between the two marked bars.
    - Let the chosen route use K moves. Sort the visited bars from left to right, rotate their color list cyclically to the right by K, repaint only those bars, and remove the gray markers.

    Key insight:
    - Heights determine which cyclic subset of bars belongs to the cheapest route, and that same route length supplies the color-rotation amount.
    """
    g = np.asarray(grid, dtype=int)
    bars = _extract_bars(g)
    info = _analyze_routes(bars)
    out_colors = _recolor_bars(bars, info)

    out = np.zeros_like(g)
    h = g.shape[0]
    for bar, color in zip(bars, out_colors):
        out[h - bar["height"] :, bar["col"]] = int(color)
    return out


def _sample_heights(count: int) -> list[int]:
    heights = [int(_RNG.integers(MIN_HEIGHT, MAX_HEIGHT + 1))]
    for _ in range(count - 1):
        prev = heights[-1]
        low = max(MIN_HEIGHT, prev - MAX_ADJ_DIFF)
        high = min(MAX_HEIGHT, prev + MAX_ADJ_DIFF)
        heights.append(int(_RNG.integers(low, high + 1)))
    return heights


def _sample_columns(count: int) -> list[int]:
    left_margin = int(_RNG.integers(0, 3))
    cols = [left_margin]
    for _ in range(count - 1):
        cols.append(cols[-1] + 1 + int(_RNG.integers(0, 2)))
    return cols


def _pick_marked_pair(count: int) -> tuple[int, int] | None:
    pairs = []
    for a in range(count):
        for b in range(a + 1, count):
            cw = (b - a) % count
            ccw = (a - b) % count
            if 2 <= cw <= count - 3 and 2 <= ccw <= count - 3:
                pairs.append((a, b))
    if not pairs:
        return None
    return tuple(map(int, pairs[int(_RNG.integers(0, len(pairs)))]))


def _build_grid(cols: list[int], heights: list[int], colors: list[int], marks: tuple[int, int]) -> np.ndarray:
    max_height = max(heights)
    top_pad = int(_RNG.integers(1, 4))
    h = max_height + top_pad + 1
    right_margin = int(_RNG.integers(1, 4))
    w = cols[-1] + right_margin + 1
    grid = np.zeros((h, w), dtype=int)
    for i, (col, height, color) in enumerate(zip(cols, heights, colors)):
        grid[h - height :, col] = int(color)
        if i in marks:
            grid[h - height - 1, col] = MARKER
    return grid


def _check_semantics(inp: np.ndarray, out: np.ndarray, desired_wrap: bool) -> bool:
    in_bars = _extract_bars(inp)
    info = _analyze_routes(in_bars)
    if info["wraps"] != desired_wrap:
        return False
    if abs(info["forward_cost"] - info["backward_cost"]) < 2:
        return False
    if not (2 <= info["steps"] <= len(in_bars) - 3):
        return False
    if len(info["ordered"]) != info["steps"] + 1:
        return False

    expected = main(inp)
    if not np.array_equal(expected, out):
        return False

    out_bars = _extract_bars(np.where(out == MARKER, BG, out + 0)) if np.any(out == MARKER) else None
    if out_bars is not None:
        return False

    out_cols = [c for c in range(out.shape[1]) if np.any(out[:, c] != BG)]
    if out_cols != [bar["col"] for bar in in_bars]:
        return False

    ordered = info["ordered"]
    before_colors = [bar["color"] for bar in in_bars]
    after_colors = _recolor_bars(in_bars, info)
    for i, bar in enumerate(in_bars):
        col = bar["col"]
        top = out.shape[0] - bar["height"]
        segment = out[top:, col]
        if not np.all(segment == after_colors[i]):
            return False
        if i not in ordered and after_colors[i] != before_colors[i]:
            return False
    if np.any(out == MARKER):
        return False
    changed = [i for i, (a, b) in enumerate(zip(before_colors, after_colors)) if a != b]
    return len(changed) == len(ordered)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        desired_wrap = bool(_RNG.integers(0, 2))
        count = int(_RNG.integers(MIN_BARS, MAX_BARS + 1))
        marks = _pick_marked_pair(count)
        if marks is None:
            continue

        heights = _sample_heights(count)
        if len(set(heights)) < 3:
            continue
        cols = _sample_columns(count)
        colors = list(_RNG.permutation(BAR_COLORS)[:count])
        grid = _build_grid(cols, heights, colors, marks)

        try:
            bars = _extract_bars(grid)
            info = _analyze_routes(bars)
        except AssertionError:
            continue

        if info["wraps"] != desired_wrap:
            continue
        if len(info["ordered"]) >= count:
            continue

        out = main(grid)
        if np.array_equal(grid, out):
            continue
        if not _check_semantics(grid, out, desired_wrap):
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
