from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7

BAR_COLORS = (BLUE, RED, GREEN, YELLOW, MAGENTA)
PAIRING_PATTERNS = (((0, 1), (2, 3)), ((0, 3), (1, 2)))
GEN_ATTEMPTS = 400


def _step_range(a: int, b: int) -> range:
    step = 2 if b >= a else -2
    return range(a, b + step, step)


def _find_baseline(grid: np.ndarray) -> int:
    rows = [int(r) for r in np.where(np.all(grid == ORANGE, axis=1))[0]]
    assert len(rows) == 1, "expected exactly one full-width orange baseline"
    return rows[0]


def _extract_bars(grid: np.ndarray, base_r: int) -> list[dict[str, int]]:
    bars: list[dict[str, int]] = []
    for c in range(grid.shape[1]):
        if base_r == 0 or int(grid[base_r - 1, c]) in (BG, ORANGE):
            continue
        color = int(grid[base_r - 1, c])
        top = base_r - 1
        while top >= 0 and int(grid[top, c]) == color:
            top -= 1
        height = base_r - (top + 1)
        assert height >= 2, "bars must be at least height 2"
        above = grid[:base_r, c]
        assert np.all((above == BG) | (above == color)), "unexpected foreground above baseline"
        bars.append({"col": int(c), "color": color, "height": int(height)})
    assert len(bars) >= 2 and len(bars) % 2 == 0, "expected an even number of bars"
    return sorted(bars, key=lambda bar: bar["col"])


def _guide_nodes(grid: np.ndarray, base_r: int) -> set[tuple[int, int]]:
    rows, cols = np.where(grid[base_r + 1 :, :] == GREY)
    return {(int(base_r + 1 + r), int(c)) for r, c in zip(rows, cols)}


def _path_pairs(grid: np.ndarray, base_r: int, bars: list[dict[str, int]]) -> dict[int, int]:
    guide = _guide_nodes(grid, base_r)
    starts = {(base_r + 1, int(bar["col"])) for bar in bars}
    assert starts <= guide, "every bar must have a guide pixel directly below it"

    unseen = set(guide)
    slot_cols = {int(bar["col"]) for bar in bars}
    pair_map: dict[int, int] = {}

    for start in sorted(starts):
        if start not in unseen:
            continue
        queue = deque([start])
        component = {start}
        unseen.remove(start)
        while queue:
            r, c = queue.popleft()
            for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
                nxt = (r + dr, c + dc)
                if nxt in unseen:
                    unseen.remove(nxt)
                    component.add(nxt)
                    queue.append(nxt)

        endpoints = sorted(
            int(c)
            for r, c in component
            if r == base_r + 1 and c in slot_cols
        )
        assert len(endpoints) == 2, "each guide component must connect exactly two slots"
        a, b = endpoints
        pair_map[a] = b
        pair_map[b] = a

    assert len(pair_map) == len(bars), "all bars must be paired"
    return pair_map


def _render_with_heights(grid: np.ndarray, base_r: int, bars: list[dict[str, int]], new_heights: dict[int, int]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    out[:base_r, :] = BG
    for bar in bars:
        col = int(bar["col"])
        color = int(bar["color"])
        height = int(new_heights[col])
        out[base_r - height : base_r, col] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap bar heights between slots connected by guide paths while keeping each slot's own color.

    Input structure:
    - A black background contains one full-width orange baseline.
    - Above the baseline sit several single-column vertical bars of distinct colors.
    - Below the baseline, grey guide pixels lie on a two-cell orthogonal lattice, with each guide component touching exactly two bar columns.

    Rule:
    - Identify every bar slot by its column, color, and height above the baseline.
    - Starting from the grey pixel directly below each bar column, follow the connected grey path using only two-cell horizontal or vertical jumps.
    - Each path component pairs two slots; swap the bars' heights across that pair.
    - Repaint the swapped heights in the original slot colors, leaving the orange baseline and all grey guide pixels unchanged.

    Key insight:
    - The grey paths do not mark output locations directly; they define the slot-pair topology that determines which bar heights exchange places.
    """
    g = np.asarray(grid, dtype=int)
    base_r = _find_baseline(g)
    bars = _extract_bars(g, base_r)
    pair_map = _path_pairs(g, base_r, bars)
    new_heights = {int(bar["col"]): int(next(b["height"] for b in bars if int(b["col"]) == pair_map[int(bar["col"])])) for bar in bars}
    return _render_with_heights(g, base_r, bars, new_heights)


def _pairing_to_map(cols: list[int], pattern: tuple[tuple[int, int], ...]) -> dict[int, int]:
    pair_map: dict[int, int] = {}
    for a_idx, b_idx in pattern:
        a = int(cols[a_idx])
        b = int(cols[b_idx])
        pair_map[a] = b
        pair_map[b] = a
    return pair_map


def _alt_pattern(current: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return PAIRING_PATTERNS[1] if current == PAIRING_PATTERNS[0] else PAIRING_PATTERNS[0]


def _choose_columns(width: int, parity: int, count: int, rng: random.Random) -> list[int] | None:
    candidates = [c for c in range(2, width - 2) if c % 2 == parity]
    if len(candidates) < count:
        return None
    for _ in range(100):
        cols = sorted(rng.sample(candidates, count))
        if all(cols[i + 1] - cols[i] >= 4 for i in range(count - 1)):
            return cols
    return None


def _vertical_segment(r0: int, r1: int, c: int) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r in _step_range(r0, r1)}


def _horizontal_segment(r: int, c0: int, c1: int) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for c in _step_range(c0, c1)}


def _build_path(
    base_r: int,
    a_col: int,
    b_col: int,
    lane_r: int,
    style: str,
) -> set[tuple[int, int]]:
    start_r = base_r + 1
    if style == "straight":
        assert lane_r == start_r
        return _horizontal_segment(start_r, a_col, b_col)

    points = set()
    points |= _vertical_segment(start_r, lane_r, a_col)
    points |= _horizontal_segment(lane_r, a_col, b_col)
    points |= _vertical_segment(lane_r, start_r, b_col)
    return points


def _pair_style_choices(pattern: tuple[tuple[int, int], ...], rng: random.Random) -> list[str]:
    if pattern == PAIRING_PATTERNS[0]:
        styles = ["u", "u"]
        if rng.random() < 0.5:
            styles[rng.randrange(2)] = "straight"
        return styles
    return ["u", "u"]


def _make_input_from_spec(
    height: int,
    width: int,
    base_r: int,
    cols: list[int],
    colors: list[int],
    heights: list[int],
    pattern: tuple[tuple[int, int], ...],
    lane_rows: list[int],
    styles: list[str],
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    grid[base_r, :] = ORANGE

    for col, color, bar_h in zip(cols, colors, heights):
        grid[base_r - bar_h : base_r, col] = int(color)

    for pair_idx, (a_idx, b_idx) in enumerate(pattern):
        a_col = int(cols[a_idx])
        b_col = int(cols[b_idx])
        lane_r = base_r + 1 if styles[pair_idx] == "straight" else int(lane_rows[pair_idx])
        nodes = _build_path(base_r, a_col, b_col, lane_r, styles[pair_idx])
        for r, c in nodes:
            grid[r, c] = GREY

    return grid


def _expected_from_pattern(inp: np.ndarray, pattern: tuple[tuple[int, int], ...]) -> np.ndarray:
    base_r = _find_baseline(inp)
    bars = _extract_bars(inp, base_r)
    cols = [int(bar["col"]) for bar in bars]
    pair_map = _pairing_to_map(cols, pattern)
    heights = {int(bar["col"]): int(bar["height"]) for bar in bars}
    new_heights = {col: heights[pair_map[col]] for col in cols}
    return _render_with_heights(inp, base_r, bars, new_heights)


def _validate_structure(inp: np.ndarray, pattern: tuple[tuple[int, int], ...]) -> bool:
    try:
        base_r = _find_baseline(inp)
        bars = _extract_bars(inp, base_r)
        if len(bars) != 4:
            return False
        heights = [int(bar["height"]) for bar in bars]
        if len(set(heights)) != len(heights):
            return False
        pair_map = _path_pairs(inp, base_r, bars)
        cols = [int(bar["col"]) for bar in bars]
        if pair_map != _pairing_to_map(cols, pattern):
            return False
        if np.array_equal(main(inp), inp):
            return False

        expected = _expected_from_pattern(inp, pattern)
        alternate = _expected_from_pattern(inp, _alt_pattern(pattern))
        if np.array_equal(expected, alternate):
            return False
        if not np.array_equal(main(inp), expected):
            return False

        identity = _render_with_heights(inp, base_r, bars, {int(bar["col"]): int(bar["height"]) for bar in bars})
        if np.array_equal(expected, identity):
            return False

        sorted_heights = sorted(heights)
        slot_cols = [int(bar["col"]) for bar in bars]
        sorted_like = _render_with_heights(
            inp,
            base_r,
            bars,
            {slot_cols[i]: sorted_heights[i] for i in range(len(slot_cols))},
        )
        if np.array_equal(expected, sorted_like):
            return False

        guide = _guide_nodes(inp, base_r)
        starts = {(base_r + 1, int(bar["col"])) for bar in bars}
        if starts != {node for node in guide if node[0] == base_r + 1 and node[1] in slot_cols}:
            return False
        return True
    except AssertionError:
        return False


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    width = rng.randint(17, 25)
    height = rng.randint(16, 24)
    base_r = rng.randint(7, min(11, height - 7))
    max_bar_h = min(8, base_r)
    if max_bar_h < 5:
        return None

    parity = rng.randrange(2)
    cols = _choose_columns(width, parity, 4, rng)
    if cols is None:
        return None

    lane_candidates = [r for r in range(base_r + 3, height - 1, 2)]
    if len(lane_candidates) < 2:
        return None

    pattern = rng.choice(PAIRING_PATTERNS)
    colors = rng.sample(list(BAR_COLORS), 4)
    heights = rng.sample(list(range(2, max_bar_h + 1)), 4)

    if pattern == PAIRING_PATTERNS[0]:
        lane_rows = sorted(rng.sample(lane_candidates, 2))
    else:
        lane_rows = sorted(rng.sample(lane_candidates, 2))
        lane_rows = [lane_rows[0], lane_rows[1]]
    styles = _pair_style_choices(pattern, rng)

    if pattern == PAIRING_PATTERNS[1]:
        lane_rows = [max(lane_rows), min(lane_rows)]

    inp = _make_input_from_spec(height, width, base_r, cols, colors, heights, pattern, lane_rows, styles)
    if not _validate_structure(inp, pattern):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        inp = _generate_attempt(rng)
        if inp is not None:
            return inp
    raise RuntimeError("generate_input failed after repeated attempts")
