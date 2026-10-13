import random
from collections import Counter

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _full_width_rows(grid: np.ndarray) -> list[int]:
    return [r for r in range(grid.shape[0]) if np.all(grid[r] == grid[r, 0])]


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 4, "expected a 2D ARC grid"

    bg = get_bg(g)
    baseline_candidates = [r for r in _full_width_rows(g) if int(g[r, 0]) != bg]
    assert baseline_candidates, "expected a non-background full-width baseline"
    base_r = max(baseline_candidates)
    baseline_color = int(g[base_r, 0])
    assert base_r >= 2, "need space above the baseline"

    mask = (g[:base_r, :] != bg) & (g[:base_r, :] != baseline_color)
    items = []
    for c in range(g.shape[1]):
        rows = np.flatnonzero(mask[:, c])
        if rows.size == 0:
            continue
        r0 = int(rows[0])
        r1 = int(rows[-1])
        assert r1 == base_r - 1, "each object must stand immediately above the baseline"
        assert rows.size == (r1 - r0 + 1), "objects must be solid vertical runs"
        colors = np.unique(g[r0 : r1 + 1, c])
        assert len(colors) == 1, "each run must be monochrome"
        items.append({"c": int(c), "r0": r0, "h": int(rows.size), "color": int(colors[0])})

    assert len(items) >= 4, "expected trigger plus at least three dominoes"
    items.sort(key=lambda item: item["c"])

    left_trigger = items[0]["h"] == 1 and items[1]["c"] - items[0]["c"] == 1
    right_trigger = items[-1]["h"] == 1 and items[-1]["c"] - items[-2]["c"] == 1
    assert left_trigger ^ right_trigger, "trigger must be a unique outer singleton"

    if left_trigger:
        trigger = items[0]
        bars = items[1:]
        direction = 1
        trigger_side = "left"
    else:
        trigger = items[-1]
        bars = items[:-1]
        direction = -1
        trigger_side = "right"

    assert all(bar["h"] >= 2 for bar in bars), "domino bars must be at least height 2"
    assert len({bar["color"] for bar in bars}) == len(bars), "domino colors must be distinct"
    assert trigger["color"] not in {bar["color"] for bar in bars}, "trigger color must be unique"

    for prev, cur in zip(bars, bars[1:]):
        assert cur["c"] - prev["c"] >= 2, "dominoes must have visible gaps"

    expected_trigger_col = bars[0]["c"] - 1 if direction == 1 else bars[-1]["c"] + 1
    assert trigger["c"] == expected_trigger_col, "trigger must touch the outermost domino"
    assert trigger["r0"] == base_r - 1, "trigger must sit one row above the baseline"

    return {
        "grid": g,
        "bg": bg,
        "base_r": base_r,
        "baseline_color": baseline_color,
        "trigger": trigger,
        "trigger_side": trigger_side,
        "direction": direction,
        "bars": bars,
    }


def _simulate_chain(bars: list[dict], direction: int) -> tuple[list[dict], list[tuple[int, int, int]]]:
    order = bars if direction == 1 else list(reversed(bars))
    toppled: list[dict] = []
    segments: list[tuple[int, int, int]] = []
    for i, bar in enumerate(order):
        toppled.append(bar)
        next_bar = order[i + 1] if i + 1 < len(order) else None
        if next_bar is not None:
            gap = direction * (next_bar["c"] - bar["c"])
            assert gap > 0
            if gap <= bar["h"]:
                segments.append((int(bar["c"]), int(next_bar["c"] - direction), int(bar["color"])))
                continue
        segments.append((int(bar["c"]), int(bar["c"] + direction * (bar["h"] - 1)), int(bar["color"])))
        break
    return toppled, segments


def _cols_between(c0: int, c1: int) -> range:
    step = 1 if c1 >= c0 else -1
    return range(int(c0), int(c1) + step, step)


def _render_output(scene: dict, toppled: list[dict], segments: list[tuple[int, int, int]]) -> np.ndarray:
    out = scene["grid"].copy()
    bg = int(scene["bg"])
    base_r = int(scene["base_r"])

    for bar in toppled:
        out[base_r - int(bar["h"]) : base_r, int(bar["c"])] = bg
    for c0, c1, color in segments:
        out[base_r - 1, list(_cols_between(c0, c1))] = int(color)
    return out


def _render_expected(scene: dict) -> np.ndarray:
    toppled, segments = _simulate_chain(scene["bars"], scene["direction"])
    return _render_output(scene, toppled, segments)


def _render_first_only(scene: dict) -> np.ndarray:
    order = scene["bars"] if scene["direction"] == 1 else list(reversed(scene["bars"]))
    first = order[0]
    segment = (int(first["c"]), int(first["c"] + scene["direction"] * (first["h"] - 1)), int(first["color"]))
    return _render_output(scene, [first], [segment])


def _render_all_topple(scene: dict) -> np.ndarray:
    order = scene["bars"] if scene["direction"] == 1 else list(reversed(scene["bars"]))
    toppled = order[:]
    segments = []
    for i, bar in enumerate(order):
        if i + 1 < len(order):
            c1 = int(order[i + 1]["c"] - scene["direction"])
        else:
            c1 = int(bar["c"] + scene["direction"] * (bar["h"] - 1))
        segments.append((int(bar["c"]), c1, int(bar["color"])))
    return _render_output(scene, toppled, segments)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Topple domino bars away from a trigger and propagate the chain until a gap exceeds the current height.

    Input structure:
    - A single-color horizontal baseline spans the full width near the bottom of the grid.
    - Above the baseline stand several one-cell-wide vertical bars of distinct colors and varying heights.
    - A unique trigger pixel sits one row above the baseline, immediately outside the leftmost or rightmost bar.

    Rule:
    - The trigger-adjacent outer bar falls first, away from the trigger.
    - A bar of height h is removed and replaced by a same-color horizontal segment on the row above the baseline, starting at its foot column and extending in the fall direction.
    - If the next standing bar's foot lies within that bar's height-based reach, the segment stops one cell before that foot and the next bar falls; otherwise the segment uses its full reach and the chain stops.

    Key insight:
    - The output is determined by a directional prefix test: compare each fallen bar's height against the next foot-to-foot gap, and stop permanently at the first gap that is too large.
    """
    scene = _extract_scene(np.asarray(grid, dtype=int))
    return _render_expected(scene)


def _choose_mode() -> str:
    return random.choices(["partial", "all", "first_only"], weights=[4, 2, 2], k=1)[0]


def _sample_reaching_gap(height: int) -> int:
    return random.randint(2, height)


def _sample_failing_gap(height: int) -> int:
    return random.randint(height + 1, min(8, height + 3))


def _sample_layout(mode: str) -> tuple[list[int], list[int]]:
    if mode == "partial":
        n_bars = random.randint(4, 6)
    else:
        n_bars = random.randint(3, 6)

    heights = [random.randint(2, 7) for _ in range(n_bars)]
    gaps = []

    if mode == "all":
        for i in range(n_bars - 1):
            gaps.append(_sample_reaching_gap(heights[i]))
    elif mode == "first_only":
        gaps.append(_sample_failing_gap(heights[0]))
        for _ in range(n_bars - 2):
            gaps.append(random.randint(2, 8))
    else:
        break_idx = random.randint(1, n_bars - 2)
        for i in range(break_idx):
            gaps.append(_sample_reaching_gap(heights[i]))
        gaps.append(_sample_failing_gap(heights[break_idx]))
        for _ in range(break_idx + 1, n_bars - 1):
            gaps.append(random.randint(2, 8))

    return heights, gaps


def _build_scene(mode: str) -> dict | None:
    heights, gaps = _sample_layout(mode)
    n_bars = len(heights)
    trigger_side = random.choice(["left", "right"])

    near_pad = random.randint(0, 2)
    far_pad = heights[-1] - 1 + random.randint(0, 2)
    span = 1 + sum(gaps)
    width = near_pad + 1 + span + far_pad + 1
    if width > 30:
        return None

    top_pad = random.randint(1, 4)
    height = max(heights) + top_pad + 1
    if height > 30:
        return None

    base_r = height - 1
    first_col = near_pad + 1
    cols = [first_col]
    for gap in gaps:
        cols.append(cols[-1] + gap)

    if trigger_side == "right":
        cols = [width - 1 - c for c in reversed(cols)]

    palette = random.sample(range(1, 10), n_bars + 2)
    baseline_color = int(palette[0])
    trigger_color = int(palette[1])
    bar_colors = [int(color) for color in palette[2:]]

    inp = np.zeros((height, width), dtype=int)
    inp[base_r, :] = baseline_color

    bars = []
    for c, h, color in zip(sorted(cols), heights if trigger_side == "left" else list(reversed(heights)), bar_colors):
        inp[base_r - h : base_r, c] = color
        bars.append({"c": int(c), "h": int(h), "color": int(color)})

    if trigger_side == "left":
        trigger_c = bars[0]["c"] - 1
    else:
        trigger_c = bars[-1]["c"] + 1
    inp[base_r - 1, trigger_c] = trigger_color

    scene = _extract_scene(inp)
    expected = _render_expected(scene)

    toppled, _segments = _simulate_chain(scene["bars"], scene["direction"])
    toppled_count = len(toppled)
    if mode == "all" and toppled_count != len(scene["bars"]):
        return None
    if mode == "first_only" and toppled_count != 1:
        return None
    if mode == "partial" and not (1 < toppled_count < len(scene["bars"])):
        return None

    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None

    first_only = _render_first_only(scene)
    all_topple = _render_all_topple(scene)
    if mode == "all" and np.array_equal(expected, first_only):
        return None
    if mode in {"partial", "first_only"} and np.array_equal(expected, all_topple):
        return None
    if mode == "partial" and np.array_equal(expected, first_only):
        return None

    parsed_bars = [(bar["c"], bar["h"], bar["color"]) for bar in scene["bars"]]
    if len(set(parsed_bars)) != len(parsed_bars):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_scene(_choose_mode())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid domino-chain input")
