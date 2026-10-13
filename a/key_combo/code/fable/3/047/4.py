from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
BAR_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]
GEN_RETRIES = 800

_GEN_CALL_COUNT = 0


def _extract_bars(grid: np.ndarray) -> tuple[tuple[int, int], list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    H, W = g.shape
    assert H >= 2 and W >= 3, "grid too small for bars and trigger"

    trigger_locs = np.argwhere(g == RED)
    assert len(trigger_locs) == 1, "expected exactly one red trigger pixel"
    tr, tc = map(int, trigger_locs[0])
    assert tr == H - 1, "trigger must be on the bottom row"

    covered = np.zeros(g.shape, dtype=bool)
    covered[tr, tc] = True

    bars: list[dict[str, int]] = []
    for c in range(W):
        color = int(g[H - 1, c])
        if color in (BLACK, RED):
            continue

        r = H - 1
        while r >= 0 and int(g[r, c]) == color:
            r -= 1
        top = r + 1
        height = H - top
        assert height >= 1, "bar must include its bottom cell"
        assert np.all(g[:top, c] == BLACK), "bar column must be empty above the bar"

        covered[top:H, c] = True
        bars.append({"c": int(c), "h": int(height), "color": int(color), "top": int(top)})

    assert bars, "expected at least one vertical bar"
    assert len({b["h"] for b in bars}) == len(bars), "bar heights must be distinct"
    assert tc == bars[0]["c"] - 1, "red trigger must be immediately left of the leftmost bar"
    assert np.array_equal(g != BLACK, covered), "input may contain only bars and the trigger"
    return (tr, tc), bars


def _chain_length(bars: list[dict[str, int]]) -> int:
    count = 0
    for i, bar in enumerate(bars):
        count = i + 1
        if i == len(bars) - 1:
            break
        next_col = bars[i + 1]["c"]
        if bar["c"] + bar["h"] >= next_col:
            continue
        break
    return count


def _render(grid: np.ndarray, trigger: tuple[int, int], bars: list[dict[str, int]], fall_count: int) -> np.ndarray:
    H, W = grid.shape
    out = np.full_like(grid, BLACK)
    out[trigger] = RED

    for bar in bars[fall_count:]:
        c, h, color = bar["c"], bar["h"], bar["color"]
        out[H - h : H, c] = color

    for bar in bars[:fall_count]:
        c, h, color = bar["c"], bar["h"], bar["color"]
        assert c + h <= W, "fallen bar must fit inside the grid"
        out[H - 1, c : c + h] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tip a chain of bottom-rooted vertical bars to the right until the first unreachable gap.

    Input structure:
    - A black-background grid has several one-cell-wide colored vertical bars standing on the bottom row.
    - Bar heights are all distinct, and the bars are ordered left to right with uneven horizontal gaps.
    - A single red trigger pixel sits on the bottom row immediately left of the leftmost bar.

    Rule:
    - The trigger knocks down the leftmost bar; each fallen bar rotates around its bottom cell and becomes a bottom-row segment of its own color and length equal to its original height.
    - A fallen bar knocks down only the next bar to its right when its horizontal reach touches or overlaps that next bar's column; otherwise the chain stops.
    - Bars to the right of the first unreachable gap remain vertical, the trigger remains red, and later fallen bars overwrite earlier fallen bottom-row pixels.

    Key insight:
    - Each bar's height is exactly its rightward reach, so the output is determined by the first adjacent pair whose base-column gap is larger than the left bar's height.
    """
    g = np.asarray(grid, dtype=int)
    trigger, bars = _extract_bars(g)
    fall_count = _chain_length(bars)
    return _render(g, trigger, bars, fall_count)


def _choose_chain_length(n: int, branch: int) -> int:
    if branch == 1:
        return n
    return random.randint(3, n - 1)


def _choose_gaps(heights: list[int], chain_len: int) -> list[int] | None:
    n = len(heights)
    reach_links = chain_len - 1
    exact_link = random.randrange(reach_links)

    overlap_candidates = [i for i in range(reach_links) if i != exact_link and heights[i] >= 3]
    if not overlap_candidates:
        return None
    overlap_link = random.choice(overlap_candidates)

    gaps: list[int] = []
    for i in range(n - 1):
        h = heights[i]
        if i < reach_links:
            if i == exact_link:
                d = h
            elif i == overlap_link:
                d = random.randint(2, h - 1)
            else:
                d = random.randint(2, h)
        elif i == reach_links:
            d = h + random.randint(1, 4)
        else:
            d = random.randint(2, 7)
        gaps.append(int(d))

    if len(set(gaps)) < min(3, len(gaps)):
        return None
    return gaps


def _make_grid(H: int, W: int, cols: list[int], heights: list[int], colors: list[int]) -> np.ndarray:
    grid = np.full((H, W), BLACK, dtype=int)
    grid[H - 1, cols[0] - 1] = RED
    for c, h, color in zip(cols, heights, colors):
        grid[H - h : H, c] = color
    return grid


def _semantic_checks(grid: np.ndarray, expected_chain_len: int, branch: int) -> bool:
    try:
        trigger, bars = _extract_bars(grid)
        out = main(grid)
    except Exception:
        return False

    n = len(bars)
    if _chain_length(bars) != expected_chain_len:
        return False
    if np.array_equal(out, grid):
        return False

    gaps = [bars[i + 1]["c"] - bars[i]["c"] for i in range(n - 1)]
    reach_links = expected_chain_len - 1
    if not any(gaps[i] == bars[i]["h"] for i in range(reach_links)):
        return False
    if not any(gaps[i] < bars[i]["h"] for i in range(reach_links)):
        return False

    # No fallen bar should bypass the next-bar relation and touch a later bar directly.
    for i in range(expected_chain_len):
        reach_edge = bars[i]["c"] + bars[i]["h"]
        for j in range(i + 2, n):
            if reach_edge >= bars[j]["c"]:
                return False

    if branch == 1:
        if expected_chain_len != n:
            return False
    else:
        if expected_chain_len >= n:
            return False
        stop_gap = gaps[expected_chain_len - 1]
        if stop_gap <= bars[expected_chain_len - 1]["h"]:
            return False
        all_fall = _render(grid, trigger, bars, n)
        if np.array_equal(out, all_fall):
            return False
        for bar in bars[expected_chain_len:]:
            c, h = bar["c"], bar["h"]
            if not np.array_equal(out[grid.shape[0] - h : grid.shape[0], c], grid[grid.shape[0] - h : grid.shape[0], c]):
                return False

    only_first = _render(grid, trigger, bars, 1)
    if np.array_equal(out, only_first):
        return False
    return True


def _gen_attempt(branch: int) -> np.ndarray | None:
    H = random.randint(7, 13)
    n = random.randint(4, min(6, H - 2, len(BAR_COLORS)))
    heights = random.sample(range(2, H), n)
    colors = random.sample(BAR_COLORS, n)
    chain_len = _choose_chain_length(n, branch)

    gaps = _choose_gaps(heights, chain_len)
    if gaps is None:
        return None

    start_col = random.randint(1, 3)
    cols = [start_col]
    for gap in gaps:
        cols.append(cols[-1] + gap)

    # Keep the causal chain local: a bar can touch its next neighbor, not skip over it.
    for i in range(chain_len):
        reach_edge = cols[i] + heights[i]
        for j in range(i + 2, n):
            if reach_edge >= cols[j]:
                return None

    min_width = max(cols[-1] + 1, max(cols[i] + heights[i] for i in range(n)))
    if min_width > 30:
        return None
    W = random.randint(min_width, 30)

    grid = _make_grid(H, W, cols, heights, colors)
    if not _semantic_checks(grid, chain_len, branch):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    branch = _GEN_CALL_COUNT % 3
    _GEN_CALL_COUNT += 1
    for _ in range(GEN_RETRIES):
        result = _gen_attempt(branch)
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid chain-toppling input")
