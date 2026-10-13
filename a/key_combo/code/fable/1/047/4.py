from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import colors_present, paste


SIDE_TOP = "top"
SIDE_BOTTOM = "bottom"
MOVE_CROSS = "cross"
MOVE_SAME = "same"


def _runs_1d(values: np.ndarray) -> list[tuple[int, int, int]]:
    vals = [int(v) for v in values]
    runs: list[tuple[int, int, int]] = []
    start = 0
    for i in range(1, len(vals) + 1):
        if i == len(vals) or vals[i] != vals[start]:
            runs.append((vals[start], start, i - 1))
            start = i
    return runs


def _detect_panels(grid: np.ndarray) -> list[dict[str, int]]:
    h, w = map(int, grid.shape)

    top_runs = _runs_1d(grid[0, :])
    if len(top_runs) == 2 and top_runs[0][0] != top_runs[1][0]:
        return [
            {"r0": 0, "c0": top_runs[0][1], "r1": h - 1, "c1": top_runs[0][2], "color": top_runs[0][0]},
            {"r0": 0, "c0": top_runs[1][1], "r1": h - 1, "c1": top_runs[1][2], "color": top_runs[1][0]},
        ]

    left_runs = _runs_1d(grid[:, 0])
    if len(left_runs) == 2 and left_runs[0][0] != left_runs[1][0]:
        return [
            {"r0": left_runs[0][1], "c0": 0, "r1": left_runs[0][2], "c1": w - 1, "color": left_runs[0][0]},
            {"r0": left_runs[1][1], "c0": 0, "r1": left_runs[1][2], "c1": w - 1, "color": left_runs[1][0]},
        ]

    raise AssertionError("expected exactly two rectangular panels on the outer border")


def _other_panel_color(panel_color: int, panel_colors: tuple[int, int]) -> int:
    a, b = map(int, panel_colors)
    if int(panel_color) == a:
        return b
    if int(panel_color) == b:
        return a
    raise AssertionError("panel color must be one of the two panel colors")


def _extract_bars(
    grid: np.ndarray,
    panel: dict[str, int],
    panel_idx: int,
    shared_color: int,
    panel_colors: tuple[int, int],
) -> list[dict[str, object]]:
    r0 = int(panel["r0"])
    c0 = int(panel["c0"])
    r1 = int(panel["r1"])
    c1 = int(panel["c1"])
    panel_color = int(panel["color"])
    allowed = {int(shared_color), _other_panel_color(panel_color, panel_colors)}

    bars: list[dict[str, object]] = []
    for c in range(c0, c1 + 1):
        if int(grid[r1, c]) == panel_color:
            continue

        rr = r1
        while rr >= r0 and int(grid[rr, c]) != panel_color:
            rr -= 1
        bar_r0 = rr + 1
        run = np.asarray(grid[bar_r0 : r1 + 1, c], dtype=int)
        assert run.size >= 2 and run.size % 2 == 0
        assert set(map(int, run)).issubset(allowed)
        bars.append(
            {
                "panel_idx": int(panel_idx),
                "panel_color": int(panel_color),
                "col": int(c),
                "r0": int(bar_r0),
                "r1": int(r1),
                "height": int(run.size),
                "run": run,
            }
        )

    return bars


def _bar_mark_side(bar: dict[str, object], shared_color: int, panel_colors: tuple[int, int]) -> str | None:
    run = np.asarray(bar["run"], dtype=int)
    if np.all(run == int(shared_color)):
        return None

    marker_color = _other_panel_color(int(bar["panel_color"]), panel_colors)
    half = int(bar["height"]) // 2
    top = run[:half]
    bottom = run[half:]
    if np.all(top == marker_color) and np.all(bottom == int(shared_color)):
        return SIDE_TOP
    if np.all(top == int(shared_color)) and np.all(bottom == marker_color):
        return SIDE_BOTTOM
    raise AssertionError("expected one marked half at exactly one end of the source bar")


def _analyze_grid(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    panels = _detect_panels(g)
    panel_colors = tuple(int(panel["color"]) for panel in panels)
    non_panel_colors = [color for color in colors_present(g) if color not in panel_colors]
    assert len(non_panel_colors) == 1
    shared_color = int(non_panel_colors[0])

    bars: list[dict[str, object]] = []
    for panel_idx, panel in enumerate(panels):
        bars.extend(_extract_bars(g, panel, panel_idx, shared_color, panel_colors))

    assert len(bars) >= 2
    heights = [int(bar["height"]) for bar in bars]
    assert len(set(heights)) == len(heights)

    marked = [(idx, side) for idx, bar in enumerate(bars) if (side := _bar_mark_side(bar, shared_color, panel_colors)) is not None]
    assert len(marked) == 1
    source_idx, side = marked[0]

    order = sorted(range(len(bars)), key=lambda idx: int(bars[idx]["height"]))
    source_rank = order.index(source_idx)
    assert source_rank < len(order) - 1
    target_idx = order[source_rank + 1]
    move_kind = MOVE_CROSS if int(bars[source_idx]["panel_idx"]) != int(bars[target_idx]["panel_idx"]) else MOVE_SAME

    return {
        "grid": g,
        "panels": panels,
        "panel_colors": panel_colors,
        "shared_color": shared_color,
        "bars": bars,
        "source_idx": int(source_idx),
        "target_idx": int(target_idx),
        "side": side,
        "move_kind": move_kind,
    }


def _paint_half(grid: np.ndarray, bar: dict[str, object], side: str, color: int) -> None:
    height = int(bar["height"])
    half = height // 2
    r0 = int(bar["r0"]) if "r0" in bar else int(grid.shape[0]) - height
    c = int(bar["col"])

    if side == SIDE_TOP:
        grid[r0 : r0 + half, c] = int(color)
    else:
        grid[r0 + half : r0 + height, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """Advance the recolored half-marker to the globally next taller vertical bar.

    Input structure:
    - The grid contains exactly two large rectangular panels with different panel colors.
    - Inside the panels are several one-cell-wide vertical bars of one shared color; all bars are bottom-aligned within their panel and have distinct heights.
    - Exactly one bar has either its top half or its bottom half recolored to the opposite panel's base color.

    Rule:
    - Restore the currently recolored half of the source bar back to the shared bar color.
    - Sort all bars from both panels together by height and pick the unique bar whose height is the smallest one strictly greater than the source bar.
    - Recolor the same half (top or bottom) of that destination bar to the opposite panel color of the panel that contains the destination bar.

    Key insight:
    - The moving marker is the half-bar segment, not the whole bar, and its color is panel-relative: when the marker lands in a different panel, the marker color changes to that panel's opposite side color.
    """
    data = _analyze_grid(np.asarray(grid, dtype=int))
    out = np.array(data["grid"], copy=True)

    bars = data["bars"]
    source_bar = bars[int(data["source_idx"])]
    target_bar = bars[int(data["target_idx"])]
    side = str(data["side"])
    shared_color = int(data["shared_color"])
    panel_colors = tuple(map(int, data["panel_colors"]))

    _paint_half(out, source_bar, side, shared_color)
    target_color = _other_panel_color(int(target_bar["panel_color"]), panel_colors)
    _paint_half(out, target_bar, side, target_color)
    return out


# --------------- generator ---------------

COLOR_POOL = tuple(range(1, 10))
HEIGHT_RANGE = (16, 22)
PANEL_WIDTH_RANGE = (7, 13)
BAR_COUNT_RANGE = (2, 3)
HEIGHT_MIN = 4
GEN_RETRY_LIMIT = 400
GEN_COMBOS = (
    (MOVE_CROSS, SIDE_TOP),
    (MOVE_SAME, SIDE_BOTTOM),
    (MOVE_CROSS, SIDE_BOTTOM),
    (MOVE_SAME, SIDE_TOP),
)

_GEN_COUNTER = 0


def _sample_panel_columns(c0: int, c1: int, count: int) -> list[int]:
    lo = int(c0) + 1
    hi = int(c1) - 1
    parities = [0, 1]
    random.shuffle(parities)

    for parity in parities:
        start = lo + parity
        if start > hi:
            continue
        cols = list(range(start, hi + 1, 2))
        if len(cols) >= int(count):
            return sorted(random.sample(cols, int(count)))
    raise ValueError("panel is too narrow for the requested number of spaced bars")


def _render_base(spec: dict[str, object]) -> np.ndarray:
    height, width = map(int, spec["shape"])
    split = int(spec["split"])
    panel_colors = tuple(map(int, spec["panel_colors"]))
    shared_color = int(spec["shared_color"])

    grid = np.full((height, width), panel_colors[0], dtype=int)
    grid[:, split:] = panel_colors[1]
    for bar in spec["bars"]:
        patch = np.full((int(bar["height"]), 1), shared_color, dtype=int)
        paste(grid, patch, height - int(bar["height"]), int(bar["col"]))
    return grid


def _render_input(spec: dict[str, object]) -> np.ndarray:
    grid = _render_base(spec)
    source_bar = spec["bars"][int(spec["source_idx"])]
    marker_color = _other_panel_color(int(source_bar["panel_color"]), tuple(map(int, spec["panel_colors"])))
    _paint_half(grid, source_bar, str(spec["side"]), marker_color)
    return grid


def _render_output(spec: dict[str, object]) -> np.ndarray:
    grid = _render_base(spec)
    target_bar = spec["bars"][int(spec["target_idx"])]
    marker_color = _other_panel_color(int(target_bar["panel_color"]), tuple(map(int, spec["panel_colors"])))
    _paint_half(grid, target_bar, str(spec["side"]), marker_color)
    return grid


def _gen_attempt(desired_move_kind: str, desired_side: str) -> tuple[np.ndarray, np.ndarray, dict[str, object]] | None:
    left_width = random.randint(*PANEL_WIDTH_RANGE)
    right_width = random.randint(*PANEL_WIDTH_RANGE)
    width = left_width + right_width
    height = random.randint(*HEIGHT_RANGE)

    left_count = random.randint(*BAR_COUNT_RANGE)
    right_count = random.randint(*BAR_COUNT_RANGE)
    total_bars = left_count + right_count
    available_heights = list(range(HEIGHT_MIN, height - 1, 2))
    if len(available_heights) < total_bars:
        return None

    colors = random.sample(COLOR_POOL, 3)
    panel_colors = (int(colors[0]), int(colors[1]))
    shared_color = int(colors[2])

    try:
        left_cols = _sample_panel_columns(0, left_width - 1, left_count)
        right_cols = _sample_panel_columns(left_width, width - 1, right_count)
    except ValueError:
        return None

    heights = random.sample(available_heights, total_bars)
    bars: list[dict[str, int]] = []
    height_idx = 0
    for col in left_cols:
        bars.append({"panel_idx": 0, "panel_color": panel_colors[0], "col": int(col), "height": int(heights[height_idx])})
        height_idx += 1
    for col in right_cols:
        bars.append({"panel_idx": 1, "panel_color": panel_colors[1], "col": int(col), "height": int(heights[height_idx])})
        height_idx += 1

    order = sorted(range(len(bars)), key=lambda idx: int(bars[idx]["height"]))
    want_cross = desired_move_kind == MOVE_CROSS
    candidates = [
        rank
        for rank in range(len(order) - 1)
        if (int(bars[order[rank]]["panel_idx"]) != int(bars[order[rank + 1]]["panel_idx"])) == want_cross
    ]
    if not candidates:
        return None

    source_rank = random.choice(candidates)
    source_idx = order[source_rank]
    target_idx = order[source_rank + 1]

    spec: dict[str, object] = {
        "shape": (int(height), int(width)),
        "split": int(left_width),
        "panel_colors": panel_colors,
        "shared_color": int(shared_color),
        "bars": bars,
        "source_idx": int(source_idx),
        "target_idx": int(target_idx),
        "side": desired_side,
    }

    inp = _render_input(spec)
    expected = _render_output(spec)
    if np.array_equal(inp, expected):
        return None

    parsed = _analyze_grid(inp)
    if str(parsed["move_kind"]) != str(desired_move_kind) or str(parsed["side"]) != str(desired_side):
        return None
    if int(parsed["source_idx"]) != int(source_idx) or int(parsed["target_idx"]) != int(target_idx):
        return None
    if not np.array_equal(main(inp), expected):
        return None

    meta = {"move_kind": desired_move_kind, "side": desired_side, "bar_count": total_bars}
    return inp, expected, meta


def _generate_case(
    desired_move_kind: str | None = None,
    desired_side: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if desired_move_kind is None or desired_side is None:
        desired_move_kind, desired_side = GEN_COMBOS[_GEN_COUNTER % len(GEN_COMBOS)]
    for _ in range(GEN_RETRY_LIMIT):
        case = _gen_attempt(desired_move_kind, desired_side)
        if case is not None:
            return case
    raise RuntimeError("failed to generate a valid marker-advance case")


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    desired_move_kind, desired_side = GEN_COMBOS[_GEN_COUNTER % len(GEN_COMBOS)]
    inp, _expected, _meta = _generate_case(desired_move_kind, desired_side)
    _GEN_COUNTER += 1
    return inp
