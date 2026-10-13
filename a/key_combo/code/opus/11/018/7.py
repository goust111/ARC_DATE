from __future__ import annotations

import random

import numpy as np


BG = 0
BAR = 3
MARK = 2
LINK = 7

MIN_BARS = 5
MAX_BARS = 6
HEIGHT_OPTIONS = (4, 6, 8, 10, 12)
MIN_GAP = 3
MAX_TRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _runs(rows: np.ndarray) -> list[tuple[int, int]]:
    if rows.size == 0:
        return []
    out: list[tuple[int, int]] = []
    start = int(rows[0])
    prev = int(rows[0])
    for value in rows[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        out.append((start, prev))
        start = value
        prev = value
    out.append((start, prev))
    return out


def _find_bars(grid: np.ndarray) -> list[dict]:
    bar_mask = np.isin(grid, (BAR, MARK))
    bars: list[dict] = []
    H, W = grid.shape
    for c in range(W):
        rows = np.where(bar_mask[:, c])[0]
        for r0, r1 in _runs(rows):
            height = r1 - r0 + 1
            if height < 2:
                continue
            colors = set(map(int, grid[r0 : r1 + 1, c]))
            if not colors.issubset({BAR, MARK}):
                continue
            bars.append(
                {
                    "r0": int(r0),
                    "r1": int(r1),
                    "c": int(c),
                    "h": int(height),
                }
            )
    bars.sort(key=lambda item: (item["c"], item["r0"], item["r1"]))
    return bars


def _marker_half(grid: np.ndarray, bar: dict) -> str | None:
    segment = grid[bar["r0"] : bar["r1"] + 1, bar["c"]]
    red_rows = np.where(segment == MARK)[0]
    if red_rows.size == 0:
        return None
    half = bar["h"] // 2
    if red_rows.size != half:
        raise AssertionError("marker must occupy exactly half of a bar")
    if not np.all(np.diff(red_rows) == 1):
        raise AssertionError("marker cells must be contiguous")
    if np.array_equal(red_rows, np.arange(0, half)):
        return "top"
    if np.array_equal(red_rows, np.arange(half, bar["h"])):
        return "bottom"
    raise AssertionError("marker must be the top or bottom half")


def _end_has_link(grid: np.ndarray, row: int, col: int) -> bool:
    H, W = grid.shape
    found = False
    for dc in (-1, 1):
        cc = col + dc
        if 0 <= cc < W and grid[row, cc] == LINK:
            found = True
    return found


def _parse_chain(grid: np.ndarray) -> tuple[list[dict], int, str]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    assert bg == BG, "expected black background"

    bars = _find_bars(g)
    assert MIN_BARS <= len(bars) <= MAX_BARS, f"expected 5-6 bars, got {len(bars)}"

    marked_indices: list[int] = []
    marked_half = ""
    for idx, bar in enumerate(bars):
        assert bar["h"] % 2 == 0, "bar height must be even"
        assert 4 <= bar["h"] <= 12, "bar height out of expected range"
        half = _marker_half(g, bar)
        bar["top_link"] = _end_has_link(g, bar["r0"], bar["c"])
        bar["bottom_link"] = _end_has_link(g, bar["r1"], bar["c"])
        if half is not None:
            marked_indices.append(idx)
            marked_half = half

    assert len(marked_indices) == 1, "expected exactly one marked bar"
    source_idx = int(marked_indices[0])

    predecessors = {i: [] for i in range(len(bars))}
    successors = {i: [] for i in range(len(bars))}
    for i, src in enumerate(bars):
        if not src["bottom_link"]:
            continue
        for j, dst in enumerate(bars):
            if i == j or not dst["top_link"]:
                continue
            if src["r1"] == dst["r0"]:
                successors[i].append(j)
                predecessors[j].append(i)

    for idx in range(len(bars)):
        assert len(successors[idx]) <= 1, "chain branches at a source bar"
        assert len(predecessors[idx]) <= 1, "chain branches at a target bar"

    heads = [idx for idx in range(len(bars)) if not predecessors[idx]]
    assert len(heads) == 1, "expected one chain head"
    head = heads[0]

    order: list[int] = []
    seen: set[int] = set()
    cur = head
    while cur not in seen:
        order.append(cur)
        seen.add(cur)
        nxts = successors[cur]
        if not nxts:
            break
        cur = nxts[0]

    assert len(order) == len(bars), "chain must cover every bar"
    assert sum(len(v) for v in successors.values()) == len(bars) - 1, "expected a single linear chain"
    for pos, idx in enumerate(order):
        bars[idx]["chain_pos"] = pos
    return bars, source_idx, marked_half


def _paint_half(grid: np.ndarray, bar: dict, half: str, color: int) -> None:
    mid = bar["r0"] + bar["h"] // 2
    if half == "top":
        grid[bar["r0"] : mid, bar["c"]] = color
    else:
        grid[mid : bar["r1"] + 1, bar["c"]] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Advance the half-bar marker to the next vertical bar in the orange-link chain.

    Input structure:
    - A black background contains 5-6 one-cell-wide vertical bars made of green, with one bar partially recolored red.
    - The red segment occupies exactly the top half or bottom half of its bar.
    - Orange pixels may sit horizontally adjacent to bar tops and bottoms; aligned bottom-to-top link rows define a unique chain order.

    Rule:
    - Detect every vertical bar and read which bar currently contains the red half-marker.
    - Infer the directed bar chain from orange-linked endpoints: bar A points to bar B when A has an orange pixel beside its bottom cell, B has one beside its top cell, and those two endpoint rows are equal.
    - Restore the current red half to green, then repaint the same half of the next bar in the chain red; if the source bar is the tail, wrap to the chain head.

    Key insight:
    - The destination is not chosen by size or proximity: only the orange endpoint-alignment chain determines the next bar, while the red segment keeps its top-half or bottom-half identity.
    """
    g = np.asarray(grid, dtype=int)
    bars, source_idx, marked_half = _parse_chain(g)
    order = sorted(range(len(bars)), key=lambda idx: bars[idx]["chain_pos"])
    source_pos = bars[source_idx]["chain_pos"]
    target_idx = order[(source_pos + 1) % len(order)]

    out = g.copy()
    _paint_half(out, bars[source_idx], marked_half, BAR)
    _paint_half(out, bars[target_idx], marked_half, MARK)
    return out


def _sample_heights(count: int) -> list[int] | None:
    for _ in range(100):
        heights = [random.choice(HEIGHT_OPTIONS) for _ in range(count)]
        span = sum(heights) - (count - 1)
        if span <= 26:
            return heights
    return None


def _sample_columns(count: int, width: int) -> list[int] | None:
    candidates = list(range(1, width - 1))
    random.shuffle(candidates)
    chosen: list[int] = []
    for col in candidates:
        if all(abs(col - prev) >= MIN_GAP for prev in chosen):
            chosen.append(col)
            if len(chosen) == count:
                random.shuffle(chosen)
                return chosen
    return None


def _side_choices(required: bool) -> list[int]:
    if required:
        options = [[-1], [1], [-1, 1]]
        return random.choice(options)
    roll = random.random()
    if roll < 0.34:
        return []
    if roll < 0.67:
        return [random.choice((-1, 1))]
    return [-1, 1]


def _build_grid(
    heights: list[int],
    columns: list[int],
    top_margin: int,
    bottom_margin: int,
    target_idx: int,
    marked_half: str,
) -> np.ndarray:
    count = len(heights)
    top_rows = [0] * count
    top_rows[0] = top_margin
    for idx in range(1, count):
        top_rows[idx] = top_rows[idx - 1] + heights[idx - 1] - 1
    bottom_rows = [top_rows[idx] + heights[idx] - 1 for idx in range(count)]
    H = bottom_rows[-1] + 1 + bottom_margin
    W = max(columns) + 2
    grid = np.full((H, W), BG, dtype=int)

    for idx, (r0, h, c) in enumerate(zip(top_rows, heights, columns)):
        grid[r0 : r0 + h, c] = BAR
        if idx == target_idx:
            half_h = h // 2
            if marked_half == "top":
                grid[r0 : r0 + half_h, c] = MARK
            else:
                grid[r0 + half_h : r0 + h, c] = MARK

    for idx, (r0, r1, c) in enumerate(zip(top_rows, bottom_rows, columns)):
        for dc in _side_choices(required=(idx > 0)):
            grid[r0, c + dc] = LINK
        for dc in _side_choices(required=(idx < count - 1)):
            grid[r1, c + dc] = LINK
    return grid


def _derive_input_from_output(output_grid: np.ndarray, target_idx: int, marked_half: str) -> np.ndarray:
    inp = output_grid.copy()
    bars, _, _ = _parse_chain(output_grid)
    order = sorted(range(len(bars)), key=lambda idx: bars[idx]["chain_pos"])
    target_bar_idx = order[target_idx]
    source_idx = order[(target_idx - 1) % len(order)]
    _paint_half(inp, bars[target_bar_idx], marked_half, BAR)
    _paint_half(inp, bars[source_idx], marked_half, MARK)
    return inp


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(MIN_BARS, MAX_BARS)
    heights = _sample_heights(count)
    if heights is None:
        return None

    span = sum(heights) - (count - 1)
    top_margin = random.randint(1, 2)
    max_bottom_margin = 30 - (span + top_margin)
    if max_bottom_margin < 1:
        return None
    bottom_margin = random.randint(1, min(3, max_bottom_margin))
    H = span + top_margin + bottom_margin

    min_width = max(18, count * MIN_GAP + 3)
    max_width = min(30, min_width + 6)
    if min_width > max_width:
        return None
    W = random.randint(min_width, max_width)
    columns = _sample_columns(count, W)
    if columns is None:
        return None

    target_idx = random.randrange(count)
    marked_half = random.choice(("top", "bottom"))
    out = _build_grid(heights, columns, top_margin, H - (span + top_margin), target_idx, marked_half)
    if out.shape[0] > 30 or out.shape[1] > 30:
        return None

    try:
        inp = _derive_input_from_output(out, target_idx, marked_half)
    except AssertionError:
        return None

    try:
        if not np.array_equal(main(inp), out):
            return None
    except AssertionError:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
