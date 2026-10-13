from __future__ import annotations

import itertools
import random

import numpy as np


STRIP_BG = 0
KEEP_COLOR = 1
SCENE_BG = 5
INPUT_COLORS = (2, 3, 4, 6, 7, 8, 9)
MAX_ATTEMPTS = 400


def _prefix_bar(row: np.ndarray) -> tuple[int, int] | None:
    nz = np.flatnonzero(row != STRIP_BG)
    if nz.size == 0:
        return None
    length = int(nz[-1]) + 1
    if not np.array_equal(nz, np.arange(length)):
        return None
    color = int(row[0])
    if color == STRIP_BG or color == KEEP_COLOR or color == SCENE_BG:
        return None
    if not np.all(row[:length] == color):
        return None
    if not np.all(row[length:] == STRIP_BG):
        return None
    return color, length


def _indicator_bars(grid: np.ndarray) -> list[tuple[int, int]]:
    bars: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        parsed = _prefix_bar(grid[r])
        if parsed is None:
            break
        bars.append(parsed)
    assert 2 <= len(bars) <= 3, "expected a 2-3 row indicator strip"
    colors = [color for color, _k in bars]
    assert len(colors) == len(set(colors)), "indicator colors must be distinct"
    return bars


def _components_of_color(scene: np.ndarray, color: int) -> list[dict]:
    mask = scene == int(color)
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []
    rows, cols = scene.shape

    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            col_set = {cc for _rr, cc in coords}
            row_vals = sorted(rr for rr, _cc in coords)
            assert len(col_set) == 1, "selected objects must be one-cell-wide vertical bars"
            assert row_vals == list(range(row_vals[0], row_vals[-1] + 1)), "bars must be vertically solid"
            comps.append(
                {
                    "coords": coords,
                    "col": next(iter(col_set)),
                    "r0": row_vals[0],
                    "r1": row_vals[-1],
                    "height": len(coords),
                }
            )
    return comps


def _ranked_bars(scene: np.ndarray, color: int) -> list[dict]:
    bars = _components_of_color(scene, color)
    heights = [bar["height"] for bar in bars]
    assert len(bars) >= 2, "each indicated color must have multiple bars"
    assert len(heights) == len(set(heights)), "same-color bar heights must be unique"
    return sorted(bars, key=lambda bar: (-bar["height"], bar["col"]))


def main(grid: np.ndarray) -> np.ndarray:
    """Keep only the top-k tallest bars of each indicated color, recoloring survivors blue.

    Input structure
    - The grid begins with a black instruction strip made of 2-3 horizontal prefix bars.
    - Each instruction bar starts at column 0; its color selects a scene color and its length encodes k.
    - Below the strip is a grey scene containing separated one-cell-wide vertical bars, including 2-4 bars for each indicated color.

    Rule
    - Read every instruction bar to obtain a pair (color C, keep-count k).
    - In the scene, collect all vertical bars of color C and rank them by height from tallest to shortest.
    - Recolor the tallest k bars of that color blue and erase the remaining bars of that color to grey.
    - Leave the instruction strip unchanged, and leave scene objects whose colors never appear in the strip unchanged.

    Key insight
    - The top bars are a two-part code: color chooses which bar family is ranked, while bar length decides how many of that family's highest-ranked members survive.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    indicators = _indicator_bars(g)
    strip_h = len(indicators)
    assert strip_h < g.shape[0], "expected a scene below the instruction strip"
    scene = g[strip_h:]
    assert np.any(scene == SCENE_BG), "expected a grey scene background"

    out = g.copy()
    for color, keep_count in indicators:
        ranked = _ranked_bars(scene, color)
        assert 1 <= keep_count < len(ranked), "indicator length must be a strict top-k cutoff"
        for idx, bar in enumerate(ranked):
            fill = KEEP_COLOR if idx < keep_count else SCENE_BG
            for r, c in bar["coords"]:
                out[strip_h + r, c] = int(fill)

    return out


def _count_options(group_count: int) -> list[tuple[int, ...]]:
    return [
        counts
        for counts in itertools.product(range(2, 5), repeat=group_count)
        if sum(counts) <= 8
    ]


COUNT_OPTIONS = {
    2: _count_options(2),
    3: _count_options(3),
}


def _sample_extra_counts(capacity_left: int) -> dict[int, int]:
    if capacity_left <= 0 or random.random() < 0.4:
        return {}
    total_extra = random.randint(1, min(2, capacity_left))
    extra_colors = random.sample([c for c in INPUT_COLORS], 2)
    if total_extra == 1:
        return {extra_colors[0]: 1}
    if random.random() < 0.5:
        return {extra_colors[0]: 2}
    return {extra_colors[0]: 1, extra_colors[1]: 1}


def _distinct_sample(pool: list[int], k: int) -> list[int]:
    assert len(pool) >= k
    vals = random.sample(pool, k)
    vals.sort(reverse=True)
    return vals


def _sample_heights(scene_h: int, counts: list[int]) -> list[list[int]]:
    high_idx, low_idx = random.sample(range(len(counts)), 2)
    high_pool = list(range(max(7, scene_h - 5), scene_h))
    low_pool = list(range(2, 7))
    mid_pool = list(range(2, scene_h))

    groups: list[list[int]] = []
    for idx, count in enumerate(counts):
        if idx == high_idx:
            groups.append(_distinct_sample(high_pool, count))
        elif idx == low_idx:
            groups.append(_distinct_sample(low_pool, count))
        else:
            groups.append(_distinct_sample(mid_pool, count))
    return groups


def _bar_interval(scene_h: int, height: int, floating: bool) -> tuple[int, int]:
    if floating:
        bottom = random.randint(height - 1, scene_h - 1)
    else:
        bottom = scene_h - 1
    return bottom - height + 1, bottom


def _place_bars(
    scene_h: int,
    scene_w: int,
    selected_colors: list[int],
    selected_heights: list[list[int]],
    extra_counts: dict[int, int],
) -> dict[int, list[dict]]:
    total_bars = sum(len(heights) for heights in selected_heights) + sum(extra_counts.values())
    start_choices = [start for start in (0, 1) if start + 2 * (total_bars - 1) < scene_w]
    start_col = random.choice(start_choices)
    columns = [start_col + 2 * idx for idx in range(total_bars)]
    random.shuffle(columns)

    layout: dict[int, list[dict]] = {}
    floating = random.random() < 0.5

    for color, heights in zip(selected_colors, selected_heights):
        specs: list[dict] = []
        for height in heights:
            col = columns.pop()
            r0, r1 = _bar_interval(scene_h, height, floating)
            specs.append({"col": col, "r0": r0, "r1": r1, "height": height})
        layout[int(color)] = specs

    for color, count in extra_counts.items():
        pool = list(range(2, scene_h))
        heights = _distinct_sample(pool, count)
        specs = []
        for height in heights:
            col = columns.pop()
            r0, r1 = _bar_interval(scene_h, height, floating)
            specs.append({"col": col, "r0": r0, "r1": r1, "height": height})
        layout[int(color)] = specs

    return layout


def _render_input(
    strip_h: int,
    scene_h: int,
    scene_w: int,
    indicators: list[tuple[int, int]],
    layout: dict[int, list[dict]],
) -> np.ndarray:
    grid = np.full((strip_h + scene_h, scene_w), SCENE_BG, dtype=int)
    grid[:strip_h] = STRIP_BG
    for row, (color, keep_count) in enumerate(indicators):
        grid[row, :keep_count] = int(color)
    for color, bars in layout.items():
        for bar in bars:
            grid[strip_h + bar["r0"] : strip_h + bar["r1"] + 1, bar["col"]] = int(color)
    return grid


def _expected_output(
    inp: np.ndarray,
    indicators: list[tuple[int, int]],
    layout: dict[int, list[dict]],
) -> np.ndarray:
    strip_h = len(indicators)
    out = inp.copy()
    for color, keep_count in indicators:
        ranked = sorted(layout[color], key=lambda bar: (-bar["height"], bar["col"]))
        for idx, bar in enumerate(ranked):
            fill = KEEP_COLOR if idx < keep_count else SCENE_BG
            out[strip_h + bar["r0"] : strip_h + bar["r1"] + 1, bar["col"]] = int(fill)
    return out


def _semantic_checks(
    inp: np.ndarray,
    out: np.ndarray,
    indicators: list[tuple[int, int]],
    layout: dict[int, list[dict]],
) -> bool:
    strip_h = len(indicators)
    if not np.array_equal(inp[:strip_h], out[:strip_h]):
        return False

    selected_colors = {color for color, _k in indicators}
    retained_heights: list[int] = []
    eliminated_heights: list[int] = []
    changed_to_blue = 0
    changed_to_grey = 0

    for color, keep_count in indicators:
        bars = sorted(layout[color], key=lambda bar: (-bar["height"], bar["col"]))
        if not (1 <= keep_count < len(bars)):
            return False
        heights = [bar["height"] for bar in bars]
        if len(heights) != len(set(heights)):
            return False
        retained_heights.extend(bar["height"] for bar in bars[:keep_count])
        eliminated_heights.extend(bar["height"] for bar in bars[keep_count:])
        for idx, bar in enumerate(bars):
            col = bar["col"]
            sl = slice(strip_h + bar["r0"], strip_h + bar["r1"] + 1)
            segment = out[sl, col]
            if idx < keep_count:
                if not np.all(segment == KEEP_COLOR):
                    return False
                changed_to_blue += int(segment.size)
            else:
                if not np.all(segment == SCENE_BG):
                    return False
                changed_to_grey += int(segment.size)

    if changed_to_blue == 0 or changed_to_grey == 0:
        return False
    if not eliminated_heights or not retained_heights:
        return False
    if max(eliminated_heights) <= min(retained_heights):
        return False

    extra_colors = set(layout) - selected_colors
    for color in extra_colors:
        before = inp[strip_h:] == int(color)
        after = out[strip_h:] == int(color)
        if not np.array_equal(before, after):
            return False

    if np.any(np.isin(out[strip_h:], list(selected_colors))):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    group_count = random.choice((2, 3))
    strip_h = group_count
    selected_counts = list(random.choice(COUNT_OPTIONS[group_count]))
    selected_colors = random.sample(list(INPUT_COLORS), group_count)
    keep_lengths = [random.randint(1, min(3, count - 1)) for count in selected_counts]
    indicators = list(zip(selected_colors, keep_lengths))

    capacity_left = 9 - sum(selected_counts)
    extra_counts = _sample_extra_counts(capacity_left)
    extra_counts = {
        color: count for color, count in extra_counts.items() if color not in selected_colors
    }

    scene_h = random.randint(12, 18)
    total_bars = sum(selected_counts) + sum(extra_counts.values())
    scene_w = random.randint(max(12, 2 * total_bars), 18)

    selected_heights = _sample_heights(scene_h, selected_counts)
    layout = _place_bars(scene_h, scene_w, selected_colors, selected_heights, extra_counts)
    inp = _render_input(strip_h, scene_h, scene_w, indicators, layout)
    expected = _expected_output(inp, indicators, layout)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _semantic_checks(inp, expected, indicators, layout):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid ranked-cutoff ARC sample")
