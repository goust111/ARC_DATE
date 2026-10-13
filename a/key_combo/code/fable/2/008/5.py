import random
from collections import Counter

import numpy as np


BG = 0
RED = 2
BAR_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]
PROFILE_WEIGHTS = [
    "no_chain",
    "partial",
    "partial",
    "full",
    "multi_hit",
    "multi_hit",
]
MAX_ATTEMPTS = 400


def _mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    assert counts, "expected a non-empty array"
    return int(counts.most_common(1)[0][0])


def _parse_bars(grid: np.ndarray) -> tuple[int, int, list[dict], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 4 and g.shape[1] >= 6, "expected a small 2D ARC grid"
    h, w = g.shape

    floor_color = int(g[-1, 0])
    assert np.all(g[-1, :] == floor_color), "bottom row must be a solid floor line"

    upper = g[:-1, :]
    bg = _mode(upper)
    bars: list[dict] = []
    trigger_index = None

    for c in range(w):
        rows = np.flatnonzero(upper[:, c] != bg)
        if rows.size == 0:
            continue
        assert int(rows[-1]) == h - 2, "every bar must stand directly on the floor"
        expected_rows = np.arange(int(rows[0]), h - 1)
        assert np.array_equal(rows, expected_rows), "each bar must be one contiguous vertical run"

        col = upper[rows, c]
        if int(col[0]) == RED:
            assert rows.size >= 2, "the triggered bar needs visible body color"
            body_colors = set(map(int, col[1:]))
            assert len(body_colors) == 1, "the triggered bar body must be monochrome"
            color = body_colors.pop()
            assert trigger_index is None, "expected exactly one red trigger marker"
            trigger_index = len(bars)
            trigger = True
        else:
            colors = set(map(int, col))
            assert len(colors) == 1, "each standing bar must be monochrome"
            color = colors.pop()
            trigger = False

        bars.append(
            {
                "c": int(c),
                "r0": int(rows[0]),
                "h": int(rows.size),
                "color": int(color),
                "trigger": trigger,
            }
        )

    assert bars and trigger_index is not None, "expected bars plus one triggered bar"
    cols = [bar["c"] for bar in bars]
    assert all(b - a > 1 for a, b in zip(cols, cols[1:])), "bars must be separated by at least one empty column"
    return bg, floor_color, bars, trigger_index


def _fallen_order(bars: list[dict], trigger_index: int) -> list[int]:
    order = [int(trigger_index)]
    frontier = bars[trigger_index]["c"] + bars[trigger_index]["h"]
    j = trigger_index + 1
    while j < len(bars) and bars[j]["c"] <= frontier:
        order.append(j)
        frontier = max(frontier, bars[j]["c"] + bars[j]["h"])
        j += 1
    return order


def _paint_fallen(input_grid: np.ndarray, bg: int, bars: list[dict], order: list[int]) -> np.ndarray:
    out = np.asarray(input_grid, dtype=int).copy()
    h, w = out.shape
    row = h - 2

    for idx in order:
        bar = bars[idx]
        out[bar["r0"] : h - 1, bar["c"]] = bg

    for idx in order:
        bar = bars[idx]
        end = bar["c"] + bar["h"]
        assert end < w, "generator and solver assume every fallen bar fits inside the grid"
        segment = out[row, bar["c"] + 1 : end + 1]
        segment[segment == bg] = int(bar["color"])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Topple the red-marked vertical bar chain to the right and draw the fallen bars on the floor.

    Input structure:
    - The bottom row is one solid floor line.
    - Several single-column vertical bars stand on that floor, separated by empty columns.
    - Exactly one bar has its top cell recolored red; that bar is the first one pushed.

    Rule:
    - A bar at column x with height h falls to the right: erase its standing cells, then draw a horizontal segment on the row just above the floor from columns x+1 through x+h in that bar's own color.
    - Any still-standing bar whose column lies inside that reach also falls rightward, so the effect propagates as a chain.
    - Resolve the chain from left to right; after all fallen bars are erased, paint their horizontal segments in fall order, and later bars may color only cells that are still background. Bars never reached by the chain stay standing, and the floor line stays unchanged.

    Key insight:
    - The bar height is exactly its rightward reach, so the puzzle is an interval-reachability chain on ordered columns rather than a simple “all bars on the right fall” rule.
    """
    bg, _floor_color, bars, trigger_index = _parse_bars(np.asarray(grid, dtype=int))
    order = _fallen_order(bars, trigger_index)
    return _paint_fallen(grid, bg, bars, order)


# --------------- generator ---------------


def _sample_columns(width: int, count: int) -> list[int] | None:
    candidates = list(range(1, width - 2))
    if len(candidates) < count:
        return None
    for _ in range(80):
        cols = sorted(random.sample(candidates, count))
        if all(b - a > 1 for a, b in zip(cols, cols[1:])):
            return cols
    return None


def _random_heights(max_heights: list[int]) -> list[int]:
    return [random.randint(2, hi) for hi in max_heights]


def _feasible_no_chain(cols: list[int], max_heights: list[int]) -> list[int]:
    feasible = []
    for i in range(len(cols) - 1):
        if min(max_heights[i], cols[i + 1] - cols[i] - 1) >= 2:
            feasible.append(i)
    return feasible


def _feasible_full(cols: list[int], max_heights: list[int]) -> list[int]:
    feasible = []
    for start in range(len(cols) - 1):
        if all(cols[i + 1] - cols[i] <= max_heights[i] for i in range(start, len(cols) - 1)):
            feasible.append(start)
    return feasible


def _feasible_partial(cols: list[int], max_heights: list[int]) -> list[tuple[int, int]]:
    feasible: list[tuple[int, int]] = []
    n = len(cols)
    for start in range(n - 2):
        for fallen_count in range(2, n - start):
            stop = start + fallen_count
            ok = True
            for i in range(start, stop - 1):
                lower = cols[i + 1] - cols[i]
                upper = min(max_heights[i], cols[stop] - cols[i] - 1)
                if lower > upper:
                    ok = False
                    break
            upper_last = min(max_heights[stop - 1], cols[stop] - cols[stop - 1] - 1)
            if ok and upper_last >= 2:
                feasible.append((start, fallen_count))
    return feasible


def _feasible_multi_hit(cols: list[int], max_heights: list[int]) -> list[tuple[int, int]]:
    feasible: list[tuple[int, int]] = []
    n = len(cols)
    for start in range(n - 2):
        for fallen_count in range(3, n - start + 1):
            stop = start + fallen_count
            trigger_upper = max_heights[start] if stop == n else min(max_heights[start], cols[stop] - cols[start] - 1)
            trigger_lower = cols[start + 2] - cols[start]
            if trigger_lower > trigger_upper:
                continue

            ok = True
            for i in range(start + 1, stop - 1):
                lower = cols[i + 1] - cols[i]
                upper = max_heights[i] if stop == n else min(max_heights[i], cols[stop] - cols[i] - 1)
                if lower > upper:
                    ok = False
                    break
            if not ok:
                continue

            if stop < n:
                upper_last = min(max_heights[stop - 1], cols[stop] - cols[stop - 1] - 1)
                if upper_last < 2:
                    continue

            feasible.append((start, fallen_count))
    return feasible


def _classify_chain(cols: list[int], heights: list[int], trigger_index: int) -> dict:
    n = len(cols)
    order = [trigger_index]
    frontier = cols[trigger_index] + heights[trigger_index]
    j = trigger_index + 1
    while j < n and cols[j] <= frontier:
        order.append(j)
        frontier = max(frontier, cols[j] + heights[j])
        j += 1

    has_multi_hit = any(
        sum(1 for j in range(i + 1, n) if cols[j] <= cols[i] + heights[i]) >= 2
        for i in order
    )
    has_overlap = any(cols[i] + heights[i] >= cols[j] + 1 for i in order for j in order if j > i)

    return {
        "order": order,
        "fallen_count": len(order),
        "right_total": n - trigger_index,
        "full_chain": len(order) == n - trigger_index,
        "no_chain": len(order) == 1,
        "partial_chain": 1 < len(order) < n - trigger_index,
        "has_multi_hit": has_multi_hit,
        "has_overlap": has_overlap,
    }


def _assign_profile_heights(profile: str, cols: list[int], max_heights: list[int]) -> tuple[list[int], int] | None:
    n = len(cols)
    heights = _random_heights(max_heights)

    if profile == "no_chain":
        options = _feasible_no_chain(cols, max_heights)
        if not options:
            return None
        trigger = random.choice(options)
        heights[trigger] = random.randint(2, min(max_heights[trigger], cols[trigger + 1] - cols[trigger] - 1))
        return heights, trigger

    if profile == "full":
        options = _feasible_full(cols, max_heights)
        if not options:
            return None
        trigger = random.choice(options)
        for i in range(trigger, n - 1):
            lower = cols[i + 1] - cols[i]
            heights[i] = random.randint(lower, max_heights[i])
        return heights, trigger

    if profile == "partial":
        options = _feasible_partial(cols, max_heights)
        if not options:
            return None
        trigger, fallen_count = random.choice(options)
        stop = trigger + fallen_count
        for i in range(trigger, stop - 1):
            lower = cols[i + 1] - cols[i]
            upper = min(max_heights[i], cols[stop] - cols[i] - 1)
            heights[i] = random.randint(lower, upper)
        upper_last = min(max_heights[stop - 1], cols[stop] - cols[stop - 1] - 1)
        heights[stop - 1] = random.randint(2, upper_last)
        return heights, trigger

    if profile == "multi_hit":
        options = _feasible_multi_hit(cols, max_heights)
        if not options:
            return None
        trigger, fallen_count = random.choice(options)
        stop = trigger + fallen_count
        upper = max_heights[trigger] if stop == n else min(max_heights[trigger], cols[stop] - cols[trigger] - 1)
        heights[trigger] = random.randint(cols[trigger + 2] - cols[trigger], upper)
        for i in range(trigger + 1, stop - 1):
            lower = cols[i + 1] - cols[i]
            upper_i = max_heights[i] if stop == n else min(max_heights[i], cols[stop] - cols[i] - 1)
            heights[i] = random.randint(lower, upper_i)
        if stop < n:
            upper_last = min(max_heights[stop - 1], cols[stop] - cols[stop - 1] - 1)
            heights[stop - 1] = random.randint(2, upper_last)
        return heights, trigger

    raise ValueError(f"unknown profile: {profile}")


def _build_bars(height: int, cols: list[int], heights: list[int], colors: list[int], trigger_index: int) -> list[dict]:
    bars = []
    for i, (c, h, color) in enumerate(zip(cols, heights, colors)):
        bars.append(
            {
                "c": int(c),
                "r0": int(height - 1 - h),
                "h": int(h),
                "color": int(color),
                "trigger": i == trigger_index,
            }
        )
    return bars


def _render_input(height: int, width: int, floor_color: int, bars: list[dict], trigger_index: int) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    grid[-1, :] = int(floor_color)
    for bar in bars:
        grid[bar["r0"] : height - 1, bar["c"]] = int(bar["color"])
    trigger = bars[trigger_index]
    grid[trigger["r0"], trigger["c"]] = RED
    return grid


def _expected_from_spec(height: int, width: int, floor_color: int, bars: list[dict], trigger_index: int) -> np.ndarray:
    inp = _render_input(height, width, floor_color, bars, trigger_index)
    order = _fallen_order(bars, trigger_index)
    return _paint_fallen(inp, BG, bars, order)


def _attempt() -> np.ndarray | None:
    profile = random.choice(PROFILE_WEIGHTS)

    height = random.randint(7, 15)
    width = random.randint(14, 30)
    bar_count = random.randint(3, 6)
    cols = _sample_columns(width, bar_count)
    if cols is None:
        return None

    max_heights = [min(height - 1, width - 1 - c) for c in cols]
    if min(max_heights) < 2:
        return None

    assigned = _assign_profile_heights(profile, cols, max_heights)
    if assigned is None:
        return None
    heights, trigger_index = assigned

    if sum(heights) * 2 >= (height - 1) * width:
        return None

    analysis = _classify_chain(cols, heights, trigger_index)
    if profile == "no_chain" and not analysis["no_chain"]:
        return None
    if profile == "partial" and not analysis["partial_chain"]:
        return None
    if profile == "full" and not analysis["full_chain"]:
        return None
    if profile == "multi_hit" and not (analysis["has_multi_hit"] and analysis["fallen_count"] >= 3):
        return None

    floor_color = random.choice(BAR_COLORS)
    color_pool = [color for color in BAR_COLORS if color != floor_color]
    if len(color_pool) < bar_count:
        return None
    colors = random.sample(color_pool, bar_count)
    bars = _build_bars(height, cols, heights, colors, trigger_index)

    inp = _render_input(height, width, floor_color, bars, trigger_index)
    expected = _expected_from_spec(height, width, floor_color, bars, trigger_index)

    try:
        parsed_bg, parsed_floor, parsed_bars, parsed_trigger = _parse_bars(inp)
        actual = main(inp)
    except Exception:
        return None

    if parsed_bg != BG or parsed_floor != floor_color:
        return None
    if parsed_trigger != trigger_index or len(parsed_bars) != len(bars):
        return None
    if not np.array_equal(actual, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chain-toppling ARC input")
