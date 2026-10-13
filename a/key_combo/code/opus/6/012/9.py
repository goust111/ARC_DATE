from __future__ import annotations

import random

import numpy as np


BG = 0
BAR = 1
MARKER = 2

MIN_BARS = 6
MAX_BARS = 8
EVEN_HEIGHTS = tuple(range(2, 20, 2))
MIN_COL_STEP = 3
MAX_ATTEMPTS = 600


def _segment_bounds(top: int, bottom: int, half_name: str) -> tuple[int, int]:
    height = bottom - top + 1
    half = height // 2
    if half_name == "top":
        return top, top + half - 1
    if half_name == "bottom":
        return bottom - half + 1, bottom
    raise ValueError(f"unknown half {half_name}")


def _trace_ray(
    out: np.ndarray,
    start_r: int,
    start_c: int,
    dr: int,
    dc: int,
) -> tuple[int, bool]:
    painted = 0
    r = start_r + dr
    c = start_c + dc
    while 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
        if int(out[r, c]) != BG:
            return painted, True
        out[r, c] = MARKER
        painted += 1
        r += dr
        c += dc
    return painted, False


def _extract_bars(grid: np.ndarray) -> tuple[list[dict], list[dict], dict, str]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BG, BAR, MARKER}, "unexpected colors in input"

    bars: list[dict] = []
    total_non_bg = int(np.count_nonzero(g != BG))
    covered = 0
    for c in range(g.shape[1]):
        rows = np.flatnonzero(g[:, c] != BG)
        if rows.size == 0:
            continue
        top = int(rows[0])
        bottom = int(rows[-1])
        segment = g[top : bottom + 1, c]
        assert rows.size == bottom - top + 1, "bars must be solid"
        assert np.all((segment == BAR) | (segment == MARKER)), "bars must use only blue/red"
        height = bottom - top + 1
        assert height % 2 == 0, "bar heights must be even"
        marker_rows = np.flatnonzero(segment == MARKER) + top
        bars.append(
            {
                "col": int(c),
                "top": top,
                "bottom": bottom,
                "height": height,
                "marker_rows": marker_rows.astype(int),
            }
        )
        covered += height

    assert MIN_BARS <= len(bars) <= MAX_BARS, "expected 6-8 vertical bars"
    assert covered == total_non_bg, "input must contain only the bars"

    heights = [bar["height"] for bar in bars]
    assert len(set(heights)) == len(heights), "bar heights must be unique"

    source_bars = [bar for bar in bars if bar["marker_rows"].size > 0]
    assert len(source_bars) == 1, "expected exactly one marked bar"
    source_bar = source_bars[0]

    marker_rows = source_bar["marker_rows"]
    top, bottom = source_bar["top"], source_bar["bottom"]
    half = source_bar["height"] // 2
    top_rows = np.arange(top, top + half)
    bottom_rows = np.arange(bottom - half + 1, bottom + 1)
    if np.array_equal(marker_rows, top_rows):
        half_name = "top"
    elif np.array_equal(marker_rows, bottom_rows):
        half_name = "bottom"
    else:
        raise AssertionError("marker must occupy exactly one half of its bar")

    bars_by_rank = sorted(bars, key=lambda bar: bar["height"])
    assert [bar["height"] for bar in bars_by_rank] == sorted(heights)
    for rank, bar in enumerate(bars_by_rank, start=1):
        bar["rank"] = rank

    source_rank = next(bar["rank"] for bar in bars_by_rank if bar is source_bar)
    assert 2 * source_rank <= len(bars_by_rank), "marked bar rank must have a doubled target"
    return bars, bars_by_rank, source_bar, half_name


def _render_from_target(grid: np.ndarray, bars_by_rank: list[dict], target_rank: int, half_name: str) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[out == MARKER] = BAR

    target = bars_by_rank[target_rank - 1]
    seg_top, seg_bottom = _segment_bounds(target["top"], target["bottom"], half_name)
    out[seg_top : seg_bottom + 1, target["col"]] = MARKER

    _trace_ray(out, seg_top, target["col"], -1, -1)
    _trace_ray(out, seg_top, target["col"], -1, 1)
    _trace_ray(out, seg_bottom, target["col"], 1, -1)
    _trace_ray(out, seg_bottom, target["col"], 1, 1)
    return out


def _scene_stats(out: np.ndarray, clean: np.ndarray, target: dict, half_name: str) -> dict:
    seg_top, seg_bottom = _segment_bounds(target["top"], target["bottom"], half_name)
    work = clean.copy()
    work[seg_top : seg_bottom + 1, target["col"]] = MARKER

    rays = []
    painted_total = 0
    blocked_any = False
    for start_r, start_c, dr, dc in (
        (seg_top, target["col"], -1, -1),
        (seg_top, target["col"], -1, 1),
        (seg_bottom, target["col"], 1, -1),
        (seg_bottom, target["col"], 1, 1),
    ):
        painted, blocked = _trace_ray(work, start_r, start_c, dr, dc)
        rays.append(((start_r, start_c), (dr, dc), painted, blocked))
        painted_total += painted
        blocked_any = blocked_any or blocked

    return {
        "ray_pixels": painted_total,
        "blocked_any": blocked_any,
        "segment_rows": (seg_top, seg_bottom),
        "ray_details": rays,
        "expected": work,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the half-bar marker to the bar at doubled height rank, then emit red diagonal rays from the new segment endpoints.

    Input structure:
    - The grid is black background with 6-8 solid blue one-cell-wide vertical bars.
    - All bars have distinct even heights and share one common bottom row.
    - Exactly one bar has its top half or bottom half recolored red as the marker segment.

    Rule:
    - Sort the bars by height from shortest to tallest and find the marked bar's rank R.
    - Restore the current red half-segment to blue, then recolor the same half of the bar at rank 2R to red.
    - From the new red segment's top endpoint, extend red diagonals up-left and up-right; from its bottom endpoint, extend red diagonals down-left and down-right.
    - Each diagonal paints only background cells and stops before the first non-background cell or at the grid edge.

    Key insight:
    - The destination is determined by doubling the source bar's height rank rather than taking the next taller bar, while the preserved half decides the vertical ray origins.
    """
    _bars, bars_by_rank, source_bar, half_name = _extract_bars(grid)
    source_rank = int(source_bar["rank"])
    return _render_from_target(grid, bars_by_rank, 2 * source_rank, half_name)


def _sample_columns(width: int, count: int, side_margin: int) -> list[int] | None:
    candidates = list(range(side_margin, width - side_margin))
    random.shuffle(candidates)
    cols: list[int] = []
    for col in candidates:
        if all(abs(col - prev) >= MIN_COL_STEP for prev in cols):
            cols.append(int(col))
            if len(cols) == count:
                return cols
    return None


def _render_clean_scene(
    height: int,
    width: int,
    bottom_row: int,
    heights_by_rank: list[int],
    cols_by_rank: list[int],
) -> tuple[np.ndarray, list[dict]]:
    grid = np.zeros((height, width), dtype=int)
    bars_by_rank: list[dict] = []
    for rank, (bar_height, col) in enumerate(zip(heights_by_rank, cols_by_rank), start=1):
        top = bottom_row - bar_height + 1
        grid[top : bottom_row + 1, col] = BAR
        bars_by_rank.append(
            {
                "rank": rank,
                "height": int(bar_height),
                "col": int(col),
                "top": int(top),
                "bottom": int(bottom_row),
            }
        )
    return grid, bars_by_rank


def _mark_input(clean: np.ndarray, source: dict, half_name: str) -> np.ndarray:
    inp = clean.copy()
    seg_top, seg_bottom = _segment_bounds(source["top"], source["bottom"], half_name)
    inp[seg_top : seg_bottom + 1, source["col"]] = MARKER
    return inp


def _height_choices(count: int, source_rank: int) -> list[int] | None:
    target_rank = 2 * source_rank
    for _ in range(150):
        heights = sorted(random.sample(EVEN_HEIGHTS, count))
        if heights[target_rank - 1] == 2 * heights[source_rank - 1]:
            continue
        return heights
    return None


def _pick_source_rank(count: int) -> int:
    upper = count // 2
    nontrivial = list(range(2, upper + 1))
    if nontrivial and random.random() < 0.8:
        return random.choice(nontrivial)
    return random.randint(1, upper)


def _scene_is_nonmonotone(cols_by_rank: list[int]) -> bool:
    increasing = all(a < b for a, b in zip(cols_by_rank, cols_by_rank[1:]))
    decreasing = all(a > b for a, b in zip(cols_by_rank, cols_by_rank[1:]))
    return not (increasing or decreasing)


def _build_alternative(
    inp: np.ndarray,
    bars_by_rank: list[dict],
    target_rank: int,
    half_name: str,
    *,
    with_rays: bool,
) -> np.ndarray:
    out = np.asarray(inp, dtype=int).copy()
    out[out == MARKER] = BAR
    target = bars_by_rank[target_rank - 1]
    seg_top, seg_bottom = _segment_bounds(target["top"], target["bottom"], half_name)
    out[seg_top : seg_bottom + 1, target["col"]] = MARKER
    if with_rays:
        _trace_ray(out, seg_top, target["col"], -1, -1)
        _trace_ray(out, seg_top, target["col"], -1, 1)
        _trace_ray(out, seg_bottom, target["col"], 1, -1)
        _trace_ray(out, seg_bottom, target["col"], 1, 1)
    return out


def _validate_scene(
    inp: np.ndarray,
    expected: np.ndarray,
    source_rank: int,
    half_name: str,
    heights_by_rank: list[int],
    cols_by_rank: list[int],
    blocked_mode: str,
) -> bool:
    bars, bars_by_rank, source_bar, parsed_half = _extract_bars(inp)
    if parsed_half != half_name or int(source_bar["rank"]) != source_rank:
        return False
    if [bar["height"] for bar in bars_by_rank] != heights_by_rank:
        return False
    if [bar["col"] for bar in bars_by_rank] != cols_by_rank:
        return False

    bottoms = {bar["bottom"] for bar in bars}
    if len(bottoms) != 1:
        return False

    target_rank = 2 * source_rank
    if target_rank <= source_rank or target_rank > len(bars_by_rank):
        return False

    clean = inp.copy()
    clean[clean == MARKER] = BAR
    target = bars_by_rank[target_rank - 1]
    stats = _scene_stats(expected, clean, target, half_name)
    if stats["ray_pixels"] == 0 or stats["ray_pixels"] > (inp.shape[0] * inp.shape[1]) // 3:
        return False
    if blocked_mode == "blocked" and not stats["blocked_any"]:
        return False
    if blocked_mode == "open" and stats["blocked_any"]:
        return False

    if not np.array_equal(stats["expected"], expected):
        return False
    if not np.array_equal(main(inp), expected):
        return False

    alt_next = _build_alternative(inp, bars_by_rank, source_rank + 1, half_name, with_rays=True)
    if np.array_equal(alt_next, expected):
        return False

    alt_no_rays = _build_alternative(inp, bars_by_rank, target_rank, half_name, with_rays=False)
    if np.array_equal(alt_no_rays, expected):
        return False

    alt_half = _build_alternative(
        inp,
        bars_by_rank,
        target_rank,
        "bottom" if half_name == "top" else "top",
        with_rays=True,
    )
    if np.array_equal(alt_half, expected):
        return False

    double_height_rank = None
    desired_height = 2 * heights_by_rank[source_rank - 1]
    for idx, bar in enumerate(bars_by_rank, start=1):
        if bar["height"] == desired_height:
            double_height_rank = idx
            break
    if double_height_rank is not None:
        alt_double = _build_alternative(inp, bars_by_rank, double_height_rank, half_name, with_rays=True)
        if np.array_equal(alt_double, expected):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(MIN_BARS, MAX_BARS)
    source_rank = _pick_source_rank(count)
    heights_by_rank = _height_choices(count, source_rank)
    if heights_by_rank is None:
        return None

    max_height = max(heights_by_rank)
    top_margin = random.randint(2, 4)
    bottom_margin = random.randint(2, 4)
    height = max_height + top_margin + bottom_margin
    if height > 30:
        return None
    bottom_row = height - bottom_margin - 1

    side_margin = random.randint(1, 3)
    min_width = 2 * side_margin + 1 + (count - 1) * MIN_COL_STEP
    if min_width > 30:
        return None
    width = random.randint(max(18, min_width), min(30, min_width + 6))
    cols = _sample_columns(width, count, side_margin)
    if cols is None:
        return None
    random.shuffle(cols)
    cols_by_rank = cols
    if not _scene_is_nonmonotone(cols_by_rank):
        return None

    clean, bars_by_rank = _render_clean_scene(height, width, bottom_row, heights_by_rank, cols_by_rank)
    half_name = random.choice(("top", "bottom"))
    target_rank = 2 * source_rank
    source = bars_by_rank[source_rank - 1]

    inp = _mark_input(clean, source, half_name)
    expected = _render_from_target(inp, bars_by_rank, target_rank, half_name)
    blocked_mode = random.choices(("blocked", "open", "either"), weights=(2, 1, 1), k=1)[0]
    if not _validate_scene(inp, expected, source_rank, half_name, heights_by_rank, cols_by_rank, blocked_mode):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
