from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
BAR_COLORS = (3, 4, 5, 6, 7, 8, 9)
MAX_ATTEMPTS = 500


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bar_specs(grid: np.ndarray, bg: int) -> tuple[list[dict[str, int]], np.ndarray]:
    h, w = grid.shape
    bars: list[dict[str, int]] = []
    bar_mask = np.zeros((h, w), dtype=bool)
    last_col = -10

    for c in range(w):
        color = int(grid[h - 1, c])
        if color == bg:
            continue

        assert c - last_col >= 2, "bottom bars must be separated by a background column"
        last_col = c

        r = h - 1
        while r >= 0 and int(grid[r, c]) == color:
            bar_mask[r, c] = True
            r -= 1
        height = h - 1 - r
        assert height >= 1, "each bar must have positive height"
        bars.append({"col": c, "color": color, "old_height": height})

    assert len(bars) >= 1, "expected at least one bottom bar"
    colors = [bar["color"] for bar in bars]
    assert len(colors) == len(set(colors)), "bar colors must be distinct"
    return bars, bar_mask


def _assert_isolated_pixel(grid: np.ndarray, bg: int, r: int, c: int) -> None:
    h, w = grid.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                assert int(grid[rr, cc]) == bg, "scatter pixels must be isolated"


def _scatter_counts(grid: np.ndarray, bg: int, bars: list[dict[str, int]], bar_mask: np.ndarray) -> dict[int, int]:
    bar_colors = {bar["color"] for bar in bars}
    counts = {color: 0 for color in bar_colors}
    scatter = np.argwhere((grid != bg) & ~bar_mask)

    for r, c in scatter:
        rr, cc = int(r), int(c)
        color = int(grid[rr, cc])
        assert color in bar_colors, "scatter colors must be drawn from the bar colors"
        _assert_isolated_pixel(grid, bg, rr, cc)
        counts[color] += 1
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rewrite each bottom bar to the count of same-colored scatter pixels and color it by height change.

    Input structure:
    - The background is empty space.
    - Several one-cell-wide vertical bars of distinct colors stand on the bottom edge, separated from each other.
    - Above the bars, isolated single pixels are scattered; every scatter color matches one of the bar colors.

    Rule:
    - For each bar, count the isolated scatter pixels with the same color, clear all original bars and scatter pixels, and redraw that bar in its original column from the bottom up to the counted height.
    - If the count is zero, delete that bar completely.
    - For every surviving rewritten bar, recolor the whole bar blue when the counted height is greater than the old height, red when it is smaller, and leave it in its original color when the heights are equal.

    Key insight:
    - The same count is used twice: it becomes the new geometry, then it is compared against the erased old height to encode growth or shrinkage in the output color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, _w = g.shape
    bg = _dominant_color(g)

    bars, bar_mask = _bar_specs(g, bg)
    counts = _scatter_counts(g, bg, bars, bar_mask)

    out = np.full_like(g, bg)
    for bar in bars:
        col = int(bar["col"])
        color = int(bar["color"])
        old_height = int(bar["old_height"])
        new_height = int(counts[color])
        assert 0 <= new_height <= h, "scatter count must fit inside the grid height"

        if new_height == 0:
            continue
        if new_height > old_height:
            out_color = BLUE
        elif new_height < old_height:
            out_color = RED
        else:
            out_color = color
        out[h - new_height : h, col] = out_color

    return out


# ---------------- generator ----------------


def _choose_spaced_columns(width: int, count: int) -> list[int] | None:
    usable = list(range(1, width - 1))
    for _ in range(200):
        cols = sorted(random.sample(usable, count))
        if all(b - a >= 2 for a, b in zip(cols, cols[1:])):
            return cols
    return None


def _make_branch_plan(n_bars: int, max_old: int, max_count: int) -> tuple[int, list[dict[str, int | str]]] | None:
    if max_old < 3 or max_count < 2:
        return None
    shared = random.randint(2, min(5, max_count, max_old - 1))

    plans: list[dict[str, int | str]] = [
        {"branch": "grow", "old": shared - 1, "new": shared},
        {"branch": "shrink", "old": shared + 1, "new": shared},
        {"branch": "same", "old": shared, "new": shared},
        {"branch": "delete", "old": random.randint(1, max_old), "new": 0},
    ]

    for _ in range(n_bars - len(plans)):
        branch = random.choice(("grow", "shrink", "same", "delete"))
        if branch == "grow":
            old = random.randint(1, min(max_old - 1, max_count - 1))
            new = random.randint(old + 1, max_count)
        elif branch == "shrink":
            old = random.randint(2, max_old)
            new = random.randint(1, min(old - 1, max_count))
        elif branch == "same":
            old = random.randint(1, max_count)
            new = old
        else:
            old = random.randint(1, max_old)
            new = 0
        plans.append({"branch": branch, "old": old, "new": new})

    random.shuffle(plans)
    return shared, plans


def _can_place_dot(grid: np.ndarray, r: int, c: int) -> bool:
    if int(grid[r, c]) != BG:
        return False
    h, w = grid.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) != BG:
                return False
    return True


def _place_scatter(grid: np.ndarray, top_rows: int, color_counts: dict[int, int]) -> bool:
    dots = [color for color, count in color_counts.items() for _ in range(int(count))]
    random.shuffle(dots)
    h, w = grid.shape
    del h

    for color in dots:
        candidates = [(r, c) for r in range(top_rows) for c in range(w) if _can_place_dot(grid, r, c)]
        if not candidates:
            return False
        r, c = random.choice(candidates)
        grid[r, c] = int(color)
    return True


def _expected_from_specs(shape: tuple[int, int], bg: int, specs: list[dict[str, int | str]]) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), bg, dtype=int)
    for spec in specs:
        new_height = int(spec["new"])
        if new_height == 0:
            continue
        old_height = int(spec["old"])
        color = int(spec["color"])
        col = int(spec["col"])
        if new_height > old_height:
            out_color = BLUE
        elif new_height < old_height:
            out_color = RED
        else:
            out_color = color
        out[h - new_height : h, col] = out_color
    return out


def _semantic_checks(grid: np.ndarray, expected: np.ndarray, shared: int, specs: list[dict[str, int | str]]) -> bool:
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False

    branches = {str(spec["branch"]) for spec in specs}
    if not {"grow", "shrink", "same", "delete"}.issubset(branches):
        return False

    shared_specs = [spec for spec in specs if int(spec["new"]) == shared and str(spec["branch"]) in {"grow", "shrink", "same"}]
    if {str(spec["branch"]) for spec in shared_specs} != {"grow", "shrink", "same"}:
        return False

    h, _w = grid.shape
    for spec in specs:
        col = int(spec["col"])
        branch = str(spec["branch"])
        new_height = int(spec["new"])
        old_height = int(spec["old"])
        color = int(spec["color"])
        if int(np.count_nonzero(grid[:, col] == color)) < old_height:
            return False
        if branch == "delete":
            if np.any(expected[:, col] != BG):
                return False
        elif branch == "grow":
            if not np.all(expected[h - new_height : h, col] == BLUE):
                return False
        elif branch == "shrink":
            if not np.all(expected[h - new_height : h, col] == RED):
                return False
        elif branch == "same":
            if not np.all(expected[h - new_height : h, col] == color):
                return False
        if new_height > 0 and new_height == old_height and color in (BLUE, RED):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_bars = random.randint(4, len(BAR_COLORS))
    min_width = max(14, 2 * n_bars + 5)
    width = random.randint(min_width, 30)
    height = random.randint(18, 28)
    top_rows = random.randint(7, min(12, height - 8))
    max_old = height - top_rows - 2
    max_count = min(6, max_old)

    plan = _make_branch_plan(n_bars, max_old, max_count)
    if plan is None:
        return None
    shared, plans = plan

    cols = _choose_spaced_columns(width, n_bars)
    if cols is None:
        return None
    colors = random.sample(BAR_COLORS, n_bars)

    grid = np.full((height, width), BG, dtype=int)
    specs: list[dict[str, int | str]] = []
    color_counts: dict[int, int] = {}

    for col, color, plan_item in zip(cols, colors, plans):
        old_height = int(plan_item["old"])
        new_height = int(plan_item["new"])
        grid[height - old_height : height, col] = int(color)
        spec = {
            "col": int(col),
            "color": int(color),
            "old": old_height,
            "new": new_height,
            "branch": str(plan_item["branch"]),
        }
        specs.append(spec)
        color_counts[int(color)] = new_height

    if not _place_scatter(grid, top_rows, color_counts):
        return None

    expected = _expected_from_specs(grid.shape, BG, specs)
    if not _semantic_checks(grid, expected, shared, specs):
        return None
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
