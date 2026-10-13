from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _UTILS_DIR.is_dir() and str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

try:
    from arc_utils import Color, get_bg
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREY = 5

    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, np.asarray(grid).reshape(-1))).most_common(1)[0][0])


BG = int(Color.BLACK)
PUSH = int(Color.RED)
DOMINO = int(Color.GREY)
POST = int(Color.BLUE)

MIN_BARS = 4
MAX_BARS = 6
MIN_HEIGHT = 2
MAX_EXTRA_HEIGHT = 7
MAX_ATTEMPTS = 800


@dataclass(frozen=True)
class Bar:
    col: int
    height: int
    color: int


def _marker_and_direction(grid: np.ndarray) -> tuple[tuple[int, int], int]:
    markers = np.argwhere(grid == PUSH)
    if markers.shape != (1, 2):
        raise ValueError("expected exactly one red push marker")
    r, c = map(int, markers[0])
    if r != grid.shape[0] - 1 or c not in (0, grid.shape[1] - 1):
        raise ValueError("expected the marker on the left or right end of the bottom row")
    return (r, c), (1 if c == 0 else -1)


def _parse_bars(grid: np.ndarray) -> list[Bar]:
    h, w = grid.shape
    bars: list[Bar] = []
    for c in range(w):
        color = int(grid[h - 1, c])
        if color not in (DOMINO, POST):
            continue
        r = h - 1
        while r >= 0 and int(grid[r, c]) == color:
            r -= 1
        bars.append(Bar(col=c, height=h - 1 - r, color=color))
    return bars


def _ordered_bars(bars: list[Bar], direction: int) -> list[Bar]:
    return sorted(bars, key=lambda bar: bar.col, reverse=(direction < 0))


def _toppled_indices(ordered: list[Bar]) -> list[int]:
    if not ordered or ordered[0].color != DOMINO:
        raise ValueError("the nearest object to the push marker must be a grey domino")
    toppled = [0]
    idx = 0
    while idx + 1 < len(ordered):
        cur = ordered[idx]
        nxt = ordered[idx + 1]
        distance = abs(nxt.col - cur.col)
        if cur.height > distance and nxt.color == DOMINO:
            idx += 1
            toppled.append(idx)
            continue
        break
    return toppled


def _render_output(
    shape: tuple[int, int],
    bars: list[Bar],
    marker: tuple[int, int],
    direction: int,
    toppled_indices: list[int],
    bg: int,
) -> np.ndarray:
    h, w = map(int, shape)
    out = np.full((h, w), int(bg), dtype=int)
    ordered = _ordered_bars(bars, direction)
    toppled_cols = {ordered[i].col for i in toppled_indices}

    for bar in bars:
        if bar.color == POST or bar.col not in toppled_cols:
            out[h - bar.height : h, bar.col] = int(bar.color)

    for idx in toppled_indices:
        bar = ordered[idx]
        for offset in range(bar.height):
            cc = bar.col + direction * offset
            if 0 <= cc < w:
                out[h - 1, cc] = DOMINO

    for bar in bars:
        if bar.color == POST:
            out[h - bar.height : h, bar.col] = POST
    out[marker] = PUSH
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Topple a bottom-row chain of grey vertical bars from the red border marker until a gap or blue post stops it.

    Input structure:
    - The background is uniform.
    - A single red marker sits on the left or right end of the bottom row and indicates push direction.
    - Several width-1 vertical bars stand on the bottom row in distinct columns; grey bars are movable dominoes and blue bars are fixed posts.

    Rule:
    - Order the bars from the marker inward.
    - The nearest grey bar topples first: it disappears vertically and becomes a grey bottom-row segment starting at its foot column and extending in the push direction for as many cells as its height.
    - If that horizontal segment reaches the next bar's foot column, the next grey bar topples in the same way; if the reached next bar is blue, the post stays upright and the chain ends there.
    - If a toppled bar does not reach the next bar, the chain stops; untouched bars remain vertical and the red marker stays in place.

    Key insight:
    - Which bars fall is determined by a step-by-step causal chain: each bar only tests whether its own height can reach the next foot column, and a fixed post terminates the propagation even when it is reachable.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    marker, direction = _marker_and_direction(g)
    bars = _parse_bars(g)
    ordered = _ordered_bars(bars, direction)
    toppled = _toppled_indices(ordered)
    return _render_output(g.shape, bars, marker, direction, toppled, bg)


def _sample_scenario() -> tuple[list[Bar], tuple[int, int], int, tuple[int, int]]:
    direction = random.choice((1, -1))
    n_bars = random.randint(MIN_BARS, MAX_BARS)
    chain_len_cap = n_bars - 2
    chain_choices = list(range(1, chain_len_cap + 1))
    chain_weights = [1] + [3] * (len(chain_choices) - 1)
    chain_len = random.choices(chain_choices, weights=chain_weights, k=1)[0]
    stop_idx = chain_len
    stop_mode = "post" if random.random() < 0.55 else "gap"

    start_gap = random.randint(2, 4)
    distances = [random.randint(2, 4) for _ in range(n_bars - 1)]
    logical_cols = [start_gap]
    for distance in distances:
        logical_cols.append(logical_cols[-1] + distance)
    width = logical_cols[-1] + random.randint(2, 4) + 1

    types = [DOMINO] * n_bars
    if stop_mode == "post":
        types[stop_idx] = POST
    else:
        types[stop_idx] = POST if random.random() < 0.35 else DOMINO
    for i in range(stop_idx + 1, n_bars):
        if random.random() < 0.25:
            types[i] = POST
    if POST not in types:
        types[random.randint(stop_idx + 1, n_bars - 1)] = POST

    heights = [0] * n_bars
    for i in range(chain_len - 1):
        heights[i] = distances[i] + 1 + random.choice((0, 1))
    if stop_mode == "post":
        heights[chain_len - 1] = distances[chain_len - 1] + 1
    else:
        heights[chain_len - 1] = max(MIN_HEIGHT, distances[chain_len - 1] - random.choice((0, 1)))
    for i in range(chain_len, n_bars):
        heights[i] = random.randint(MIN_HEIGHT, MAX_EXTRA_HEIGHT)

    height = max(max(heights) + random.randint(1, 3), 8)
    if direction == 1:
        marker = (height - 1, 0)
        cols = logical_cols
    else:
        marker = (height - 1, width - 1)
        cols = [width - 1 - col for col in logical_cols]
    bars = [Bar(col=int(col), height=int(hh), color=int(cc)) for col, hh, cc in zip(cols, heights, types)]
    return bars, marker, direction, (height, width)


def _validate_scenario(
    bars: list[Bar],
    marker: tuple[int, int],
    direction: int,
    shape: tuple[int, int],
) -> tuple[bool, np.ndarray | None]:
    h, w = map(int, shape)
    if max(shape) > 30 or min(shape) <= 0:
        return False, None
    if marker != (h - 1, 0) and marker != (h - 1, w - 1):
        return False, None
    if len(bars) < MIN_BARS:
        return False, None
    if any(not (0 <= bar.col < w and MIN_HEIGHT <= bar.height < h) for bar in bars):
        return False, None

    ordered = _ordered_bars(bars, direction)
    if len({bar.col for bar in ordered}) != len(ordered):
        return False, None
    if ordered[0].color != DOMINO:
        return False, None
    if not any(bar.color == POST for bar in ordered):
        return False, None

    toppled = _toppled_indices(ordered)
    if len(toppled) >= len(ordered) - 1:
        return False, None
    if len(toppled) == 0:
        return False, None

    stop_at = len(toppled)
    if stop_at >= len(ordered):
        return False, None
    next_bar = ordered[stop_at]
    last_toppled = ordered[toppled[-1]]
    stop_distance = abs(next_bar.col - last_toppled.col)

    if next_bar.color == POST:
        if last_toppled.height <= stop_distance:
            stop_reason = "gap"
        elif last_toppled.height == stop_distance + 1:
            stop_reason = "post"
        else:
            return False, None
    else:
        if last_toppled.height > stop_distance:
            return False, None
        stop_reason = "gap"

    for i in toppled[:-1]:
        cur = ordered[i]
        nxt = ordered[i + 1]
        distance = abs(nxt.col - cur.col)
        if cur.height <= distance:
            return False, None
        if i + 2 < len(ordered):
            after_next = ordered[i + 2]
            if cur.height > abs(after_next.col - cur.col):
                return False, None

    if stop_reason == "post" and next_bar.color != POST:
        return False, None
    if stop_reason == "gap" and last_toppled.height > stop_distance:
        return False, None

    grid = np.full((h, w), BG, dtype=int)
    grid[marker] = PUSH
    for bar in bars:
        grid[h - bar.height : h, bar.col] = int(bar.color)

    expected = _render_output((h, w), bars, marker, direction, toppled, BG)
    actual = main(grid)
    if np.array_equal(grid, expected):
        return False, None
    if not np.array_equal(actual, expected):
        return False, None
    if not any(
        np.all(grid[h - bar.height : h - 1, bar.col] == DOMINO) and np.all(expected[h - bar.height : h - 1, bar.col] == BG)
        for bar in (ordered[i] for i in toppled)
        if bar.height >= 2
    ):
        return False, None
    return True, grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        bars, marker, direction, shape = _sample_scenario()
        ok, grid = _validate_scenario(bars, marker, direction, shape)
        if ok and grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid domino-chain input")
