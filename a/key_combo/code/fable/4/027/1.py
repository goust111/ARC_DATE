from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
WORK_CANVAS = 30
MIN_BAR_COUNT = 4
MAX_BAR_COUNT = 6
POSITION_MIN = 4
POSITION_MAX = 25
MIN_POSITION_GAP = 5
MIN_TAIL = 2
MAX_TAIL_ASSIGN_ATTEMPTS = 200
MAX_GENERATION_ATTEMPTS = 400
CROP_MARGIN_MIN = 1
CROP_MARGIN_MAX = 3

_RNG = random.Random()


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _fg_colors(grid: np.ndarray, bg: int) -> list[int]:
    return sorted(int(color) for color in np.unique(grid) if int(color) != int(bg))


def _contiguous_runs(values: list[int]) -> list[tuple[int, int]]:
    if not values:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(values[0])
    for value in values[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev))
        start = prev = value
    runs.append((start, prev))
    return runs


def _covering_run(line_positions: np.ndarray, required_positions: list[int]) -> tuple[int, int]:
    values = [int(v) for v in line_positions.tolist()]
    runs = _contiguous_runs(values)
    req_lo = int(min(required_positions))
    req_hi = int(max(required_positions))
    candidates = [(start, end) for start, end in runs if start <= req_lo <= end and start <= req_hi <= end]
    assert len(candidates) == 1, "bar span must be the unique non-background run containing its visible cells"
    return candidates[0]


def _extract_bars(grid: np.ndarray, bg: int) -> dict[int, dict]:
    bars: dict[int, dict] = {}
    for color in _fg_colors(grid, bg):
        coords = np.argwhere(grid == int(color))
        rows = sorted({int(r) for r, _c in coords})
        cols = sorted({int(c) for _r, c in coords})

        if len(rows) == 1 and len(cols) >= 2:
            row = rows[0]
            span = _covering_run(np.flatnonzero(grid[row] != bg), cols)
            bars[int(color)] = {
                "color": int(color),
                "orient": "H",
                "line": int(row),
                "start": int(span[0]),
                "end": int(span[1]),
                "length": int(span[1] - span[0] + 1),
            }
            continue

        if len(cols) == 1 and len(rows) >= 2:
            col = cols[0]
            span = _covering_run(np.flatnonzero(grid[:, col] != bg), rows)
            bars[int(color)] = {
                "color": int(color),
                "orient": "V",
                "line": int(col),
                "start": int(span[0]),
                "end": int(span[1]),
                "length": int(span[1] - span[0] + 1),
            }
            continue

        raise AssertionError("each color must correspond to one straight 1-cell-wide bar")

    assert bars, "expected at least one foreground bar"
    return bars


def _infer_depth_order(grid: np.ndarray, bars: dict[int, dict]) -> list[int]:
    colors = list(bars.keys())
    horizontals = [bar for bar in bars.values() if bar["orient"] == "H"]
    verticals = [bar for bar in bars.values() if bar["orient"] == "V"]
    assert horizontals and verticals, "expected both horizontal and vertical bars"

    edges = {color: set() for color in colors}
    indeg = {color: 0 for color in colors}

    for hbar in horizontals:
        row = int(hbar["line"])
        hcolor = int(hbar["color"])
        for vbar in verticals:
            col = int(vbar["line"])
            vcolor = int(vbar["color"])
            if not (int(hbar["start"]) <= col <= int(hbar["end"])):
                continue
            if not (int(vbar["start"]) <= row <= int(vbar["end"])):
                continue

            seen = int(grid[row, col])
            assert seen in (hcolor, vcolor), "a crossing cell must show one of the two crossing bars"
            top, bottom = (hcolor, vcolor) if seen == hcolor else (vcolor, hcolor)
            if bottom not in edges[top]:
                edges[top].add(bottom)
                indeg[bottom] += 1

    order: list[int] = []
    frontier = [color for color in colors if indeg[color] == 0]
    while frontier:
        assert len(frontier) == 1, "crossing relations must imply a unique total depth order"
        current = frontier.pop()
        order.append(int(current))
        for nxt in sorted(edges[current]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                frontier.append(int(nxt))

    assert len(order) == len(colors), "depth relation graph must be acyclic and cover all bars"
    return order


def _render_histogram(order: list[int], bars: dict[int, dict]) -> np.ndarray:
    max_len = max(int(bars[color]["length"]) for color in order)
    out = np.zeros((len(order), max_len), dtype=int)
    for row, color in enumerate(order):
        out[row, : int(bars[color]["length"])] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the bars' front-to-back occlusion order and output a left-aligned depth histogram.

    Input structure:
    - Black background with several 1-cell-wide monochrome bars; each bar color is unique.
    - Bars are strictly horizontal or vertical, and perpendicular bars may cross.
    - At each crossing only the upper bar's color is visible, while the lower bar remains hidden there.

    Rule:
    - Recover every bar's orientation and full span from its visible cells and the contiguous occupied run on that row or column.
    - For every horizontal/vertical crossing, read the visible color to determine which bar is above the other.
    - Take the transitive closure of those pairwise relations to obtain the unique order from top layer to bottom layer, then output one row per bar with that bar's color repeated for its length from the left edge.

    Key insight:
    - A single crossing only reveals one local above/below fact; the answer depends on combining all such facts into one global depth order, while the output ignores the original positions and keeps only each ordered bar's length and color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    bars = _extract_bars(g, bg)
    order = _infer_depth_order(g, bars)
    return _render_histogram(order, bars)


def _sample_spaced_positions(rng: random.Random, count: int) -> list[int] | None:
    candidates = list(range(POSITION_MIN, POSITION_MAX + 1))
    for _ in range(200):
        picked = sorted(rng.sample(candidates, count))
        if all(b - a >= MIN_POSITION_GAP for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _bar_sequence(bar_count: int, start_orient: str, rows: list[int], cols: list[int]) -> list[dict]:
    bars: list[dict] = []
    h_idx = 0
    v_idx = 0
    for idx in range(bar_count):
        orient = start_orient if idx % 2 == 0 else ("V" if start_orient == "H" else "H")
        if orient == "H":
            bars.append({"orient": "H", "line": int(rows[h_idx])})
            h_idx += 1
        else:
            bars.append({"orient": "V", "line": int(cols[v_idx])})
            v_idx += 1

    for idx, bar in enumerate(bars):
        essential = []
        if idx > 0:
            essential.append(int(bars[idx - 1]["line"]))
        if idx + 1 < len(bars):
            essential.append(int(bars[idx + 1]["line"]))
        bar["essential"] = sorted(essential)
    return bars


def _opposite_bounds(bar: dict, rows: list[int], cols: list[int]) -> tuple[int, int, int, int] | None:
    core_lo = int(min(bar["essential"]))
    core_hi = int(max(bar["essential"]))
    opposite_positions = cols if bar["orient"] == "H" else rows
    before = [pos for pos in opposite_positions if int(pos) < core_lo]
    after = [pos for pos in opposite_positions if int(pos) > core_hi]
    lower_bound = int(before[-1] + 2) if before else 1
    upper_bound = int(after[0] - 2) if after else WORK_CANVAS - 2
    if core_lo - lower_bound < MIN_TAIL or upper_bound - core_hi < MIN_TAIL:
        return None
    return core_lo, core_hi, lower_bound, upper_bound


def _assign_spans(rng: random.Random, bars: list[dict], rows: list[int], cols: list[int]) -> bool:
    bounds = []
    for bar in bars:
        current = _opposite_bounds(bar, rows, cols)
        if current is None:
            return False
        bounds.append(current)

    for _ in range(MAX_TAIL_ASSIGN_ATTEMPTS):
        lengths: set[int] = set()
        spans: list[tuple[int, int, int]] = []
        ok = True
        for core_lo, core_hi, lower_bound, upper_bound in bounds:
            left_tail = rng.randint(MIN_TAIL, core_lo - lower_bound)
            right_tail = rng.randint(MIN_TAIL, upper_bound - core_hi)
            start = int(core_lo - left_tail)
            end = int(core_hi + right_tail)
            length = int(end - start + 1)
            if length in lengths:
                ok = False
                break
            lengths.add(length)
            spans.append((start, end, length))
        if not ok:
            continue

        for bar, (start, end, length) in zip(bars, spans):
            bar["start"] = int(start)
            bar["end"] = int(end)
            bar["length"] = int(length)
        return True
    return False


def _paint_bar(canvas: np.ndarray, bar: dict) -> None:
    color = int(bar["color"])
    line = int(bar["line"])
    start = int(bar["start"])
    end = int(bar["end"])
    if bar["orient"] == "H":
        canvas[line, start : end + 1] = color
    else:
        canvas[start : end + 1, line] = color


def _crop_scene(rng: random.Random, canvas: np.ndarray) -> np.ndarray:
    coords = np.argwhere(canvas != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)

    top_margin = rng.randint(CROP_MARGIN_MIN, min(CROP_MARGIN_MAX, int(r0)))
    left_margin = rng.randint(CROP_MARGIN_MIN, min(CROP_MARGIN_MAX, int(c0)))
    bottom_margin = rng.randint(CROP_MARGIN_MIN, min(CROP_MARGIN_MAX, int(canvas.shape[0] - 1 - r1)))
    right_margin = rng.randint(CROP_MARGIN_MIN, min(CROP_MARGIN_MAX, int(canvas.shape[1] - 1 - c1)))

    rr0 = int(r0 - top_margin)
    cc0 = int(c0 - left_margin)
    rr1 = int(r1 + bottom_margin)
    cc1 = int(c1 + right_margin)
    return canvas[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()


def _expected_output(bars: list[dict]) -> np.ndarray:
    max_len = max(int(bar["length"]) for bar in bars)
    out = np.zeros((len(bars), max_len), dtype=int)
    for row, bar in enumerate(bars):
        out[row, : int(bar["length"])] = int(bar["color"])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bar_count = rng.randint(MIN_BAR_COUNT, MAX_BAR_COUNT)
    start_orient = rng.choice(["H", "V"])
    h_count = (bar_count + (1 if start_orient == "H" else 0)) // 2
    v_count = bar_count - h_count

    rows = _sample_spaced_positions(rng, h_count)
    cols = _sample_spaced_positions(rng, v_count)
    if rows is None or cols is None:
        return None

    bars = _bar_sequence(bar_count, start_orient, rows, cols)
    if not _assign_spans(rng, bars, rows, cols):
        return None

    for bar, color in zip(bars, rng.sample(PALETTE, bar_count)):
        bar["color"] = int(color)

    canvas = np.zeros((WORK_CANVAS, WORK_CANVAS), dtype=int)
    for bar in reversed(bars):
        _paint_bar(canvas, bar)

    grid = _crop_scene(rng, canvas)
    if max(grid.shape) > 30:
        return None

    for bar in bars:
        if int(np.count_nonzero(grid == int(bar["color"]))) < 2:
            return None

    expected = _expected_output(bars)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to produce a valid occlusion-order histogram task")
