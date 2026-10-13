from __future__ import annotations

import random
from typing import Any

import numpy as np


BAR_COUNT_RANGE = (2, 4)
SOURCE_COUNT_RANGE = (3, 5)
WIDTH_RANGE = (8, 16)
FIRST_BAR_ROW_RANGE = (2, 5)
BAR_GAP_RANGE = (2, 5)
BOTTOM_MARGIN_RANGE = (1, 4)
MAX_GENERATION_ATTEMPTS = 400
PALETTE = tuple(range(1, 10))


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _parse_input(grid: np.ndarray) -> tuple[int, list[dict[str, int]], list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 4 and w >= 4, "grid is too small for sources and bars"

    bg = _background_color(g)

    source_cols = np.argwhere(g[0] != bg).flatten().tolist()
    assert source_cols, "expected at least one top-row source"
    assert len(source_cols) == len(set(source_cols)), "source columns must be distinct"

    sources = []
    for c in sorted(int(c) for c in source_cols):
        color = int(g[0, c])
        assert color != bg
        sources.append({"col": c, "color": color})

    bars = []
    bar_rows = []
    for r in range(1, h):
        row = g[r]
        if np.all(row == bg):
            continue
        assert np.all(row != bg), "non-source rows must be full-width bars"
        gate = int(row[0])
        fill = int(row[1])
        assert gate != fill, "bar gate and fill colors must differ"
        assert int(row[-1]) == gate, "bar ends must share the gate color"
        assert np.all(row[1:-1] == fill), "bar interior must be a single fill color"
        bars.append({"row": r, "gate": gate, "fill": fill})
        bar_rows.append(r)

    assert bars, "expected at least one gate bar"
    assert all(bar_rows[i + 1] - bar_rows[i] >= 2 for i in range(len(bar_rows) - 1)), "bars must be separated"

    allowed = {(0, src["col"]) for src in sources}
    for r, c in np.argwhere(g != bg):
        rc = (int(r), int(c))
        if r == 0:
            assert rc in allowed, "top row may contain only isolated source pixels"

    return bg, sources, bars


def _trace_source(source_color: int, bars: list[dict[str, int]]) -> dict[str, Any]:
    current = int(source_color)
    passes = 0
    history = []
    stop_index = None

    for idx, bar in enumerate(bars):
        passed = current == int(bar["gate"])
        history.append(
            {
                "index": idx,
                "entry_color": current,
                "gate": int(bar["gate"]),
                "fill": int(bar["fill"]),
                "passed": passed,
            }
        )
        if not passed:
            stop_index = idx
            break
        current = int(bar["fill"])
        passes += 1

    return {
        "source_color": int(source_color),
        "passes": passes,
        "history": history,
        "stopped": stop_index is not None,
        "stop_index": stop_index,
        "final_color": current,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each top source downward, letting full-width two-tone bars gate and recolor the ray in sequence.

    Input structure:
    - The background is black.
    - The top row contains several isolated colored source pixels in distinct columns.
    - Lower down, some non-adjacent full-width horizontal bars appear; each bar has one fill color across the interior and a different gate color on its two end cells.

    Rule:
    - Copy the input grid and process each source column independently.
    - A source emits a downward ray starting just below the source, with current color equal to the source color.
    - Whenever the ray reaches a bar row, compare its current color with that bar's gate color: if they match, the ray passes through and its current color becomes the bar's fill color; otherwise it stops on the cell just above the bar.
    - Paint every non-bar cell visited by the ray with its current color, stop at the first failed gate, and leave all bars unchanged.

    Key insight:
    - Later bars test the ray's updated current color, so earlier successful passages can unlock or block later bars through chained recoloring.
    """
    g = np.asarray(grid, dtype=int)
    bg, sources, bars = _parse_input(g)
    out = g.copy()
    h, _w = g.shape

    for source in sources:
        current = int(source["color"])
        c = int(source["col"])
        paint_start = 1
        for bar in bars:
            row = int(bar["row"])
            if paint_start < row:
                out[paint_start:row, c] = current
            if current != int(bar["gate"]):
                break
            current = int(bar["fill"])
            paint_start = row + 1
        else:
            out[paint_start:h, c] = current
            continue

    return out


def _solve_using_source_color_only(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    _bg, sources, bars = _parse_input(g)
    out = g.copy()
    h, _w = g.shape

    for source in sources:
        source_color = int(source["color"])
        current = source_color
        c = int(source["col"])
        paint_start = 1
        for bar in bars:
            row = int(bar["row"])
            if paint_start < row:
                out[paint_start:row, c] = current
            if source_color != int(bar["gate"]):
                break
            current = int(bar["fill"])
            paint_start = row + 1
        else:
            out[paint_start:h, c] = current
            continue

    return out


def _solve_without_recolor(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    _bg, sources, bars = _parse_input(g)
    out = g.copy()
    h, _w = g.shape

    for source in sources:
        source_color = int(source["color"])
        c = int(source["col"])
        paint_start = 1
        for bar in bars:
            row = int(bar["row"])
            if paint_start < row:
                out[paint_start:row, c] = source_color
            if source_color != int(bar["gate"]):
                break
            paint_start = row + 1
        else:
            out[paint_start:h, c] = source_color
            continue

    return out


def _bar_rows(n_bars: int) -> list[int]:
    rows = [random.randint(*FIRST_BAR_ROW_RANGE)]
    for _ in range(n_bars - 1):
        rows.append(rows[-1] + random.randint(*BAR_GAP_RANGE))
    return rows


def _build_bars(n_bars: int) -> tuple[list[tuple[int, int]], list[int], bool]:
    if n_bars == 2:
        chain_reaches_bottom = True
    else:
        chain_reaches_bottom = random.random() < 0.55

    if chain_reaches_bottom:
        path = random.sample(PALETTE, n_bars + 1)
        bars = [(path[i], path[i + 1]) for i in range(n_bars)]
        lower_gates = [path[i] for i in range(1, n_bars)]
        return bars, lower_gates, True

    colors = random.sample(PALETTE, n_bars + 2)
    path = colors[:n_bars]
    final_gate = colors[n_bars]
    final_fill = colors[n_bars + 1]
    bars = [(path[i], path[i + 1]) for i in range(n_bars - 1)]
    bars.append((final_gate, final_fill))
    lower_gates = [path[i] for i in range(1, n_bars)] + [final_gate]
    return bars, lower_gates, False


def _choose_sources(bars: list[tuple[int, int]], lower_gates: list[int], width: int) -> list[tuple[int, int]]:
    source_count = random.randint(SOURCE_COUNT_RANGE[0], min(SOURCE_COUNT_RANGE[1], width - 2))
    chain_source = int(bars[0][0])

    chosen_colors = [chain_source]
    shuffled_lower = lower_gates[:]
    random.shuffle(shuffled_lower)
    for color in shuffled_lower:
        if color not in chosen_colors and len(chosen_colors) < source_count:
            chosen_colors.append(int(color))

    extras = [color for color in PALETTE if color not in chosen_colors]
    random.shuffle(extras)
    while len(chosen_colors) < source_count:
        chosen_colors.append(int(extras.pop()))

    cols = sorted(random.sample(range(1, width - 1), len(chosen_colors)))
    random.shuffle(chosen_colors)
    return list(zip(cols, chosen_colors))


def _build_input_grid() -> np.ndarray:
    n_bars = random.randint(*BAR_COUNT_RANGE)
    rows = _bar_rows(n_bars)
    bottom_margin = random.randint(*BOTTOM_MARGIN_RANGE)
    height = rows[-1] + 1 + bottom_margin
    width = random.randint(*WIDTH_RANGE)
    bars, lower_gates, _chain_reaches_bottom = _build_bars(n_bars)
    sources = _choose_sources(bars, lower_gates, width)

    grid = np.zeros((height, width), dtype=int)
    for c, color in sources:
        grid[0, c] = int(color)
    for row, (gate, fill) in zip(rows, bars):
        grid[row, 0] = int(gate)
        grid[row, width - 1] = int(gate)
        grid[row, 1 : width - 1] = int(fill)
    return grid


def _validate_example(grid: np.ndarray) -> bool:
    try:
        bg, sources, bars = _parse_input(grid)
        out = main(grid)
    except Exception:
        return False

    if bg != 0:
        return False
    if len(bars) < 2 or len(sources) < 3:
        return False
    if grid.shape[0] > 30 or grid.shape[1] > 30:
        return False
    if not np.any(out != grid):
        return False

    traces = [_trace_source(int(source["color"]), bars) for source in sources]
    pass_counts = [trace["passes"] for trace in traces]

    if max(pass_counts) < 2:
        return False
    if not any(trace["passes"] == 0 for trace in traces):
        return False
    if not any(trace["passes"] >= 1 and trace["final_color"] != trace["source_color"] for trace in traces):
        return False

    lower_gate_colors = {int(bar["gate"]) for bar in bars[1:]}
    if not any(
        trace["source_color"] in lower_gate_colors and trace["passes"] == 0
        for trace in traces
    ):
        return False

    if np.array_equal(out, _solve_using_source_color_only(grid)):
        return False
    if np.array_equal(out, _solve_without_recolor(grid)):
        return False

    changed_columns = [
        int(source["col"])
        for source in sources
        if np.any(out[1:, int(source["col"])] != grid[1:, int(source["col"])])
    ]
    if len(changed_columns) < 2:
        return False

    chain_source_color = int(bars[0]["gate"])
    chain_traces = [trace for trace in traces if trace["source_color"] == chain_source_color]
    if len(chain_traces) != 1:
        return False
    if chain_traces[0]["passes"] != max(pass_counts):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_input_grid()
        if _validate_example(grid):
            return grid
    raise RuntimeError("failed to generate a valid gate-ray example")
