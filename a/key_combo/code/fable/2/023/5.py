from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, colors_present, get_bg


MAX_GENERATION_ATTEMPTS = 400


def _analyze_grid(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]], list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    colors = colors_present(g, exclude={bg})
    assert len(colors) >= 3, "expected at least one source color and multiple bar colors"

    source_candidates: list[tuple[int, list[tuple[int, int]]]] = []
    bars: list[dict[str, int]] = []

    for color in colors:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == int(color))]
        rows = sorted({r for r, _c in coords})
        if len(rows) == 1:
            row = rows[0]
            row_vals = g[row]
            assert set(map(int, np.unique(row_vals))) <= {bg, int(color)}, "bar row must contain only itself and background"
            bars.append({"row": int(row), "color": int(color)})
        else:
            source_candidates.append((int(color), coords))

    assert len(source_candidates) == 1, "expected exactly one multi-row source color"
    assert len(bars) >= 2, "expected multiple horizontal bars"

    bars.sort(key=lambda item: int(item["row"]))
    bar_rows = [int(item["row"]) for item in bars]
    assert len(set(bar_rows)) == len(bar_rows), "bar rows must be distinct"

    source_color, source_coords = source_candidates[0]
    assert source_coords, "expected at least one source pixel"
    assert max(r for r, _c in source_coords) < bar_rows[0], "all sources must sit above the top bar"

    return bg, int(source_color), source_coords, bars


def _gap_width(row: np.ndarray, col: int, bg: int) -> int:
    if int(row[col]) != int(bg):
        return 0
    left = int(col)
    right = int(col)
    while left - 1 >= 0 and int(row[left - 1]) == int(bg):
        left -= 1
    while right + 1 < row.shape[0] and int(row[right + 1]) == int(bg):
        right += 1
    return int(right - left + 1)


def _drop_one(out: np.ndarray, bars: list[dict[str, int]], col: int, bg: int) -> None:
    for bar in bars:
        row = int(bar["row"])
        color = int(bar["color"])

        if int(out[row, col]) != int(bg):
            land = row - 1
            while land >= 0 and int(out[land, col]) != int(bg):
                land -= 1
            assert land >= 0, "no vertical room left above the stopping bar"
            out[land, col] = color
            return

        if _gap_width(out[row, :], col, bg) == 1:
            out[row, col] = color
            return


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop scattered source pixels through stacked broken horizontal bars, where single-cell holes plug and wider holes pass.

    Input structure:
    - The most common color is the background.
    - Several one-cell-thick horizontal bars of distinct colors are stacked from top to bottom; each bar row contains only its own color and background gaps.
    - Above the top bar, scattered source pixels all share one common color, and some columns may contain multiple source pixels.

    Rule:
    - Remove the source pixels from their original positions, then process them from nearest-to-the-bars to farthest, i.e. by descending source row.
    - For each source column, inspect the bars from top to bottom: a solid cell stops the falling pixel above that bar and colors it with the bar's color; a gap wider than one cell lets the pixel continue; a one-cell gap is plugged by the pixel itself, which takes the bar color and becomes part of that bar.
    - Plugged holes and already-settled pixels become obstacles for later pixels in the same column, so later pixels stack upward above the same stopping bar.
    - A pixel that passes through every bar disappears outside the grid.

    Key insight:
    - The bars are not static: early pixels can repair single-cell holes, so the correct output depends jointly on gap width and on the per-column falling order.
    """
    g = np.asarray(grid, dtype=int)
    bg, source_color, source_coords, bars = _analyze_grid(g)

    out = g.copy()
    out[out == int(source_color)] = int(bg)

    for _row, col in sorted(source_coords, key=lambda rc: (-int(rc[0]), int(rc[1]))):
        _drop_one(out, bars, int(col), int(bg))

    return out


def _sample_spaced_values(lo: int, hi: int, n: int, min_gap: int) -> list[int]:
    values = list(range(int(lo), int(hi) + 1))
    for _ in range(300):
        picks: list[int] = []
        shuffled = values[:]
        random.shuffle(shuffled)
        for value in shuffled:
            if all(abs(int(value) - int(prev)) >= int(min_gap) for prev in picks):
                picks.append(int(value))
                if len(picks) == int(n):
                    return sorted(picks)
    raise RuntimeError("failed to sample sufficiently spaced values")


def _mark_single_gap(mask: np.ndarray, col: int) -> bool:
    lo = max(0, int(col) - 1)
    hi = min(mask.shape[0], int(col) + 2)
    if bool(mask[lo:hi].any()):
        return False
    mask[int(col)] = True
    return True


def _mark_wide_gap(mask: np.ndarray, col: int) -> bool:
    starts = [int(col) - 1, int(col)]
    random.shuffle(starts)
    for start in starts:
        if start < 0 or start + 1 >= mask.shape[0]:
            continue
        lo = max(0, start - 1)
        hi = min(mask.shape[0], start + 3)
        if bool(mask[lo:hi].any()):
            continue
        mask[start : start + 2] = True
        return True
    return False


def _build_route_specs(n_bars: int, top_bar_row: int) -> list[dict]:
    max_top_stack = int(top_bar_row - 1)
    top_solid_count = random.randint(1, min(3, max_top_stack))
    top_single_count = random.randint(2, min(4, max_top_stack))
    mid_single_count = random.randint(2, min(3, max_top_stack))

    routes = [
        {
            "name": "top_solid",
            "statuses": ["solid"] * n_bars,
            "source_count": top_solid_count,
        },
        {
            "name": "top_single",
            "statuses": ["single"] + ["solid"] * (n_bars - 1),
            "source_count": top_single_count,
        },
        {
            "name": "mid_single",
            "statuses": ["wide", "single"] + ["solid"] * (n_bars - 2),
            "source_count": mid_single_count,
        },
    ]

    if n_bars == 3:
        routes.extend(
            [
                {
                    "name": "bottom_solid",
                    "statuses": ["wide", "wide", "solid"],
                    "source_count": random.randint(1, 2),
                },
                {
                    "name": "vanish",
                    "statuses": ["wide", "wide", "wide"],
                    "source_count": random.randint(1, 2),
                },
            ]
        )
    else:
        bottom_type = random.choice(["single", "solid"])
        bottom_count = random.randint(2, 3) if bottom_type == "single" else random.randint(1, 2)
        routes.extend(
            [
                {
                    "name": "lower_solid",
                    "statuses": ["wide", "wide", "solid", "solid"],
                    "source_count": random.randint(1, 2),
                },
                {
                    "name": "bottom_target",
                    "statuses": ["wide", "wide", "wide", bottom_type],
                    "source_count": bottom_count,
                },
                {
                    "name": "vanish",
                    "statuses": ["wide", "wide", "wide", "wide"],
                    "source_count": random.randint(1, 2),
                },
            ]
        )

    return routes


def _add_random_gaps(gap_masks: list[np.ndarray], occupancy_floor: int) -> None:
    for mask in gap_masks:
        n_extra = random.randint(0, 2)
        for _ in range(n_extra):
            if int(mask.shape[0] - mask.sum()) <= int(occupancy_floor):
                break
            for _ in range(40):
                if random.random() < 0.5:
                    col = random.randint(1, mask.shape[0] - 2)
                    if _mark_single_gap(mask, col):
                        break
                else:
                    col = random.randint(1, mask.shape[0] - 2)
                    if _mark_wide_gap(mask, col):
                        break


def _expected_from_plan(
    grid: np.ndarray,
    bg: int,
    source_color: int,
    bar_rows: list[int],
    bar_colors: list[int],
    routes: list[dict],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[out == int(source_color)] = int(bg)

    for route in routes:
        col = int(route["col"])
        count = int(route["source_count"])
        statuses = list(route["statuses"])

        stop_idx = None
        stop_kind = None
        for idx, status in enumerate(statuses):
            if status != "wide":
                stop_idx = int(idx)
                stop_kind = status
                break
        if stop_idx is None:
            continue

        color = int(bar_colors[stop_idx])
        row = int(bar_rows[stop_idx])
        if stop_kind == "single":
            assert count <= row + 1, "single-gap stack exceeds available height"
            out[row, col] = color
            for depth in range(1, count):
                out[row - depth, col] = color
        else:
            assert count <= row, "solid-stop stack exceeds available height"
            for depth in range(count):
                out[row - 1 - depth, col] = color

    return out


def _gen_attempt() -> np.ndarray | None:
    n_bars = random.choice([3, 4])
    top_bar_row = random.randint(6, 8)
    routes = _build_route_specs(n_bars, top_bar_row)

    width_min = 4 * len(routes) + 1
    width = random.randint(width_min, 30)
    route_cols = _sample_spaced_values(1, width - 2, len(routes), 4)
    random.shuffle(route_cols)
    for route, col in zip(routes, route_cols):
        route["col"] = int(col)

    bar_rows = [top_bar_row]
    for _ in range(n_bars - 1):
        bar_rows.append(int(bar_rows[-1] + random.randint(3, 5)))
    height = int(bar_rows[-1] + random.randint(3, 6))
    if height > 30:
        return None

    palette = random.sample(Color.ALL_COLORS, n_bars + 2)
    bg = int(palette[0])
    source_color = int(palette[1])
    bar_colors = [int(color) for color in palette[2:]]

    gap_masks = [np.zeros(width, dtype=bool) for _ in range(n_bars)]
    for route in routes:
        col = int(route["col"])
        for bar_idx, status in enumerate(route["statuses"]):
            ok = True
            if status == "single":
                ok = _mark_single_gap(gap_masks[bar_idx], col)
            elif status == "wide":
                ok = _mark_wide_gap(gap_masks[bar_idx], col)
            if not ok:
                return None

    _add_random_gaps(gap_masks, occupancy_floor=max(width // 2 + 1, width - 12))
    if any(int(width - mask.sum()) <= width // 2 for mask in gap_masks):
        return None

    grid = np.full((height, width), bg, dtype=int)
    for row, color, gap_mask in zip(bar_rows, bar_colors, gap_masks):
        grid[int(row), :] = int(color)
        grid[int(row), gap_mask] = int(bg)

    source_rows_used: set[int] = set()
    for route in routes:
        count = int(route["source_count"])
        rows = sorted(random.sample(range(top_bar_row - 1), count))
        route["source_rows"] = rows
        for row in rows:
            source_rows_used.add(int(row))
            grid[int(row), int(route["col"])] = int(source_color)

    if len(source_rows_used) < 2:
        return None
    if int(get_bg(grid)) != int(bg):
        return None

    expected = _expected_from_plan(grid, bg, source_color, bar_rows, bar_colors, routes)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    for route in routes:
        col = int(route["col"])
        rows = list(route["source_rows"])
        if any(int(actual[row, col]) != int(bg) for row in rows):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cascade-sieve input")
