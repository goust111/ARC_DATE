from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

# Load shared ARC helpers via explicit path so SourceFileLoader import works from this file location.
_UTIL = None
for _root in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
    _candidate = _root / "new" / "solvers" / "arc_utils.py"
    if _candidate.exists():
        _UTIL = SourceFileLoader("arc_utils", str(_candidate)).load_module()
        break
if _UTIL is None:
    raise ImportError("failed to locate arc_utils.py")

get_bg = _UTIL.get_bg
find_components = _UTIL.find_components


def _extract_bars(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    """Find all valid bars and return bar metadata used by main()."""

    bars: list[dict[str, object]] = []
    used_body_colors: set[int] = set()
    used_cap_colors: set[int] = set()

    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        if int(comp["area"]) != 1:
            continue

        r0, c0 = map(int, comp["coords"][0])
        cap_color = int(comp["color"])
        if cap_color in used_cap_colors:
            continue
        if r0 + 1 >= grid.shape[0]:
            continue

        # A cap is a single pixel with background directly above (or top edge)
        # and a non-background body color directly below.
        if r0 > 0 and int(grid[r0 - 1, c0]) != bg:
            continue
        body_color = int(grid[r0 + 1, c0])
        if body_color in {bg, cap_color} or body_color in used_body_colors:
            continue

        body_rows = np.where(grid[:, c0] == body_color)[0]
        body_rows = body_rows[body_rows > r0]
        if body_rows.size == 0 or int(body_rows[0]) != r0 + 1:
            continue

        body_top = int(body_rows.min())
        body_bottom = int(body_rows.max())
        span = grid[body_top : body_bottom + 1, c0]

        # Bar body must stay in one column and only use body color or background in its span.
        if not np.all((span == body_color) | (span == bg)):
            continue

        gap_rows = [rr for rr in range(body_top, body_bottom + 1) if int(grid[rr, c0]) == bg]
        body_len = int(np.sum(span == body_color))
        if body_len <= 0:
            continue

        bars.append(
            {
                "cap_color": cap_color,
                "body_len": body_len,
                "gap_cells": [(rr, c0) for rr in gap_rows],
                "body_color": body_color,
            }
        )
        used_body_colors.add(body_color)
        used_cap_colors.add(cap_color)

    return bars


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each vertical bar's internal gaps with the cap color of the next bar in length order.

    Input structure:
    - Dominant background surrounds several disjoint vertical bars.
    - Each bar has a one-pixel cap at top and a different body color below it.
    - Each bar body is mainly one-column vertical, with background gaps inside.

    Rule:
    - Detect all bars, compute each bar body length excluding its cap.
    - Sort bars from longest body length to shortest (cyclic order).
    - Fill each bar's gap cells with the cap color of the next bar in that order.
    - Preserve all non-gap body pixels, caps, and background cells.

    Key insight:
    - Ranking is by body length (not position), and the borrowed color source is the next bar's cap, not a neighboring body.
    """

    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    bars = _extract_bars(g, bg)

    if len(bars) <= 1:
        return g.copy()

    lengths = [int(item["body_len"]) for item in bars]
    assert len(set(lengths)) == len(lengths), "bar body lengths must be unique"

    out = g.copy()
    ranked = sorted(bars, key=lambda item: int(item["body_len"]), reverse=True)
    for i, bar in enumerate(ranked):
        donor_color = int(ranked[(i + 1) % len(ranked)]["cap_color"])
        for rr, cc in bar["gap_cells"]:  # type: ignore[assignment]
            out[int(rr), int(cc)] = donor_color

    return out


# ------------------------------
# Generator
# ------------------------------

_ALL_COLORS = list(range(10))
_GRID_H_RANGE = (10, 28)
_GRID_W_RANGE = (12, 29)
_BAR_COUNT_RANGE = (2, 4)
_MAX_GAPS = 3
_MAX_ATTEMPTS = 300


def _paint_bar(
    grid: np.ndarray,
    cap_row: int,
    col: int,
    bg: int,
    body_color: int,
    cap_color: int,
    body_len: int,
    gap_count: int,
) -> list[tuple[int, int]]:
    """Paint one bar and return gap coordinates."""

    span_len = body_len + gap_count
    body_start = cap_row + 1
    body_end = body_start + span_len - 1

    assert 0 <= cap_row < body_start <= body_end < grid.shape[0]

    body_rows = list(range(body_start, body_end + 1))
    gap_candidates = body_rows[1:-1] if len(body_rows) >= 3 else []
    if gap_count > 0:
        assert gap_count <= len(gap_candidates)
        gap_rows = set(random.sample(gap_candidates, gap_count))
    else:
        gap_rows = set()

    for rr in body_rows:
        grid[rr, col] = bg if rr in gap_rows else body_color

    grid[cap_row, col] = cap_color
    return [(rr, col) for rr in sorted(gap_rows)]


def generate_input() -> np.ndarray:
    """Randomly generate valid test inputs for this puzzle."""

    for _ in range(_MAX_ATTEMPTS):
        H = random.randint(*_GRID_H_RANGE)
        W = random.randint(*_GRID_W_RANGE)
        if W - 2 < max(_BAR_COUNT_RANGE):
            continue

        bg = random.choice(_ALL_COLORS)
        available = [c for c in _ALL_COLORS if c != bg]

        bar_count = random.randint(*_BAR_COUNT_RANGE)
        if len(available) < 2 * bar_count:
            continue

        body_colors = random.sample(available, bar_count)
        remain = [c for c in available if c not in body_colors]
        cap_colors = random.sample(remain, bar_count)

        max_body_len = min(12, H - 4)
        if max_body_len < 3:
            continue
        body_lengths = random.sample(range(3, max_body_len + 1), bar_count)

        cols = random.sample(range(1, W - 1), bar_count)

        # 75% with at least one gap, 25% intentionally gap-free for parser coverage.
        require_gap = random.random() < 0.75
        forced_gap_idx = random.randrange(bar_count) if require_gap else None

        grid = np.full((H, W), bg, dtype=int)
        bars: list[dict[str, object]] = []

        for idx in range(bar_count):
            body_len = body_lengths[idx]
            max_gap = min(_MAX_GAPS, body_len - 1)

            if idx == forced_gap_idx:
                if max_gap <= 0:
                    bars = []
                    break
                gap_count = random.randint(1, max_gap)
            else:
                gap_count = random.randint(0, max_gap)

            span_len = body_len + gap_count
            if span_len + 1 >= H:
                bars = []
                break

            cap_row = random.randint(0, H - span_len - 1)
            gap_cells = _paint_bar(
                grid,
                cap_row=cap_row,
                col=cols[idx],
                bg=bg,
                body_color=body_colors[idx],
                cap_color=cap_colors[idx],
                body_len=body_len,
                gap_count=gap_count,
            )
            bars.append(
                {
                    "cap_color": cap_colors[idx],
                    "body_len": body_len,
                    "gap_cells": gap_cells,
                }
            )

        if len(bars) != bar_count:
            continue

        if np.count_nonzero(grid == bg) <= (H * W) // 2:
            continue

        try:
            out = main(grid)
        except Exception:
            continue

        if np.array_equal(out, grid) and random.random() < 0.75:
            continue

        if out.shape != grid.shape:
            continue

        return grid

    raise RuntimeError("failed to generate a valid input")
