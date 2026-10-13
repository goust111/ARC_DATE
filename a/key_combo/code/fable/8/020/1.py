from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

LEFT = -1
RIGHT = 1

CHAIN_PATTERNS: tuple[tuple[int, ...], ...] = (
    (RIGHT, RIGHT),
    (RIGHT, LEFT),
    (RIGHT, RIGHT, LEFT),
    (RIGHT, LEFT, RIGHT),
    (RIGHT, LEFT, LEFT),
    (RIGHT, RIGHT, LEFT, RIGHT),
    (RIGHT, LEFT, RIGHT, LEFT),
)
CHAIN_WEIGHTS = (1, 2, 2, 3, 1, 2, 2)
MAX_ATTEMPTS = 400


def _direction(marker_color: int) -> int:
    return RIGHT if marker_color in (RED, GREEN) else LEFT


def _bar_interval(col: int, height: int, marker_color: int) -> tuple[int, int, int]:
    direction = _direction(marker_color)
    if direction == RIGHT:
        lo = int(col)
        hi = int(col + height - 1)
        endpoint = hi
    else:
        lo = int(col - height + 1)
        hi = int(col)
        endpoint = lo
    return lo, hi, endpoint


def _extract_bars(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bars: list[dict[str, int]] = []
    covered = np.zeros_like(g, dtype=bool)

    for c in range(w):
        if int(g[h - 1, c]) == BG:
            continue
        top = h - 1
        while top > 0 and int(g[top - 1, c]) != BG:
            top -= 1

        segment = g[top:, c]
        marker = int(segment[0])
        assert marker in (BLUE, RED, GREEN), "top marker must be blue, red, or green"
        if segment.shape[0] > 1:
            assert np.all(segment[1:] == GREY), "bar body must be gray"

        bars.append({"col": int(c), "height": int(h - top), "marker": marker})
        covered[top:, c] = True

    assert bars, "expected at least one vertical bar"
    assert int(np.sum(g != BG)) == int(covered.sum()), "foreground must be only bottom-anchored bars"
    green_bars = [bar for bar in bars if bar["marker"] == GREEN]
    assert len(green_bars) == 1, "expected exactly one green starter bar"
    return bars


def _fallen_order(bars: list[dict[str, int]]) -> list[int]:
    by_col = {bar["col"]: bar for bar in bars}
    start_col = next(bar["col"] for bar in bars if bar["marker"] == GREEN)

    queue = [start_col]
    fallen: set[int] = set()
    order: list[int] = []

    while queue:
        col = queue.pop(0)
        if col in fallen:
            continue
        bar = by_col[col]
        fallen.add(col)
        order.append(col)

        lo, hi, _endpoint = _bar_interval(bar["col"], bar["height"], bar["marker"])
        targets = [
            other_col
            for other_col in by_col
            if other_col not in fallen and lo <= other_col <= hi
        ]
        if _direction(bar["marker"]) == RIGHT:
            targets.sort()
        else:
            targets.sort(reverse=True)
        queue.extend(targets)

    return order


def main(grid: np.ndarray) -> np.ndarray:
    """Topple the green starter bar so each struck bar falls in its own marker direction.

    Input structure:
    - The grid is black with several single-column bars whose bottoms sit on the last row.
    - Each bar is gray except for its top marker cell: red means that bar falls right, blue means it falls left.
    - Exactly one top marker is green; it is the starter bar and it always falls right.

    Rule:
    - Simulate the chain reaction starting from the green bar.
    - A falling bar of height h becomes a horizontal gray segment of length h on the last row, anchored at its base cell; its original top-marker color is placed at the far endpoint.
    - Whenever that horizontal segment covers the base column of a still-standing bar, that struck bar falls next according to its own top-marker color; bars never struck stay upright.

    Key insight:
    - A bar's vertical height is exactly its horizontal reach, but the next fall direction comes from the struck bar's own marker rather than from the incoming wave.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    h, w = g.shape
    bars = _extract_bars(g)
    fallen_cols = _fallen_order(bars)
    fallen_set = set(fallen_cols)

    out = np.full_like(g, BG)

    for bar in bars:
        if bar["col"] in fallen_set:
            continue
        top = h - bar["height"]
        out[top:, bar["col"]] = GREY
        out[top, bar["col"]] = bar["marker"]

    endpoint_colors: dict[int, int] = {}
    for col in fallen_cols:
        bar = next(bar for bar in bars if bar["col"] == col)
        lo, hi, endpoint = _bar_interval(bar["col"], bar["height"], bar["marker"])
        assert 0 <= lo <= hi < w, "fallen bar must stay inside the canvas"
        out[h - 1, lo : hi + 1] = GREY
        endpoint_colors[endpoint] = bar["marker"]

    for endpoint, marker in endpoint_colors.items():
        out[h - 1, endpoint] = marker

    return out


def _choose_chain() -> tuple[list[int], list[int], int, int]:
    for _ in range(MAX_ATTEMPTS):
        directions = list(random.choices(CHAIN_PATTERNS, weights=CHAIN_WEIGHTS, k=1)[0])
        cols = [0]
        lo = 0
        hi = 0

        for direction in directions[:-1]:
            gap = random.randint(2, 6)
            next_col = hi + gap if direction == RIGHT else lo - gap
            cols.append(next_col)
            lo = min(lo, next_col)
            hi = max(hi, next_col)

        last_col = cols[-1]
        last_dir = directions[-1]
        if last_dir == RIGHT:
            extra = random.randint(1, (hi - last_col) + 4)
            last_endpoint = last_col + extra
        else:
            extra = random.randint(1, (last_col - lo) + 4)
            last_endpoint = last_col - extra

        final_lo = min(lo, last_endpoint)
        final_hi = max(hi, last_endpoint)
        heights = [abs(cols[i + 1] - cols[i]) + 1 for i in range(len(cols) - 1)]
        heights.append(abs(last_endpoint - last_col) + 1)

        span_width = final_hi - final_lo + 1
        if span_width <= 24 and max(heights) <= 18:
            return directions, cols, final_lo, final_hi

    raise RuntimeError("failed to sample a valid chain layout")


def _sample_extra_columns(candidates: list[int], count: int) -> list[int]:
    chosen: list[int] = []
    random.shuffle(candidates)
    for col in candidates:
        if all(abs(col - old) >= 2 for old in chosen):
            chosen.append(col)
            if len(chosen) == count:
                break
    return chosen


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        directions, rel_cols, final_lo, final_hi = _choose_chain()
        chain_count = len(directions)

        extra_total = random.randint(4, min(10, 30 - (final_hi - final_lo + 1)))
        left_margin = random.randint(2, extra_total - 2)
        right_margin = extra_total - left_margin

        offset = left_margin - final_lo
        cols = [col + offset for col in rel_cols]
        union_lo = final_lo + offset
        union_hi = final_hi + offset
        width = union_hi + right_margin + 1

        last_col = cols[-1]
        last_dir = directions[-1]
        base_heights = [abs(cols[i + 1] - cols[i]) + 1 for i in range(chain_count - 1)]
        if last_dir == RIGHT:
            last_reach = random.randint(1, (union_hi - last_col) + 1)
        else:
            last_reach = random.randint(1, (last_col - union_lo) + 1)
        heights = base_heights + [last_reach + 1]

        max_height = max(heights)
        height = max_height + random.randint(1, min(6, 30 - max_height))

        bars: list[dict[str, int]] = []
        for idx, (col, bar_height, direction) in enumerate(zip(cols, heights, directions)):
            if idx == 0:
                marker = GREEN
            elif direction == RIGHT:
                marker = RED
            else:
                marker = BLUE
            bars.append({"col": int(col), "height": int(bar_height), "marker": int(marker)})

        left_candidates = list(range(0, max(0, union_lo - 1)))
        right_candidates = list(range(min(width, union_hi + 2), width))
        extra_candidates = left_candidates + right_candidates
        random.shuffle(extra_candidates)
        extra_count = min(random.randint(0, 3), len(extra_candidates))
        extra_cols = _sample_extra_columns(extra_candidates, extra_count)

        for col in extra_cols:
            bars.append(
                {
                    "col": int(col),
                    "height": int(random.randint(2, min(8, height - 1))),
                    "marker": int(random.choice((RED, BLUE))),
                }
            )

        bars.sort(key=lambda bar: bar["col"])
        grid = np.full((height, width), BG, dtype=int)
        for bar in bars:
            top = height - bar["height"]
            grid[top:, bar["col"]] = GREY
            grid[top, bar["col"]] = bar["marker"]

        try:
            out = main(grid)
        except AssertionError:
            continue

        if out.shape == grid.shape and np.any(out != grid):
            return grid

    raise RuntimeError("failed to generate a valid input grid")
