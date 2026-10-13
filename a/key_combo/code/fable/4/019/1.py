from collections import defaultdict
from pathlib import Path
import random
from typing import Iterable

import sys
import numpy as np


# Make arc_utils importable when this file is loaded by SourceFileLoader.
_current = Path(__file__).resolve()
for parent in (_current.parent, *_current.parents):
    candidate = parent / "new" / "solvers" / "arc_utils.py"
    if candidate.exists():
        arc_utils_dir = str(candidate.parent)
        if arc_utils_dir not in sys.path:
            sys.path.append(arc_utils_dir)
        break
else:
    raise ImportError("arc_utils.py not found in project hierarchy")

from arc_utils import Color, colors_present, get_bg  # noqa: E402


Grid = np.ndarray


def _to_coords(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return row/col arrays for all True cells in mask."""
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected at least one cell"
    return coords[:, 0], coords[:, 1]


def _collect_bars(grid: Grid, bg: int) -> dict[int, tuple[str, int]]:
    """Reconstruct each bar from visible pixels: (orientation, line index)."""
    bars: dict[int, tuple[str, int]] = {}
    horizontal_rows = set()
    vertical_cols = set()

    for color in colors_present(grid, exclude={bg}):
        rows, cols = _to_coords(grid == color)
        uniq_rows = np.unique(rows)
        uniq_cols = np.unique(cols)
        assert len(uniq_rows) > 0 and len(uniq_cols) > 0

        # A bar is either a full-width horizontal stripe (fixed row)
        # or a full-height vertical stripe (fixed col).
        if len(uniq_rows) == 1 and len(uniq_cols) >= 2:
            orientation = "H"
            line = int(uniq_rows[0])
            assert line not in horizontal_rows, "two bars cannot share one full row"
            horizontal_rows.add(line)
        elif len(uniq_cols) == 1 and len(uniq_rows) >= 2:
            orientation = "V"
            line = int(uniq_cols[0])
            assert line not in vertical_cols, "two bars cannot share one full column"
            vertical_cols.add(line)
        else:
            # Degenerate visibility is not compatible with the bar model in this task.
            raise AssertionError(f"invalid bar geometry for color {color}")

        bars[int(color)] = (orientation, line)

    assert bars, "expected at least one bar"
    return bars


def _infer_order_from_crossings(grid: Grid, bg: int, bars: dict[int, tuple[str, int]]) -> list[int]:
    """Infer global depth order from every bar crossing pixel."""
    horizontal = {c: line for c, (ori, line) in bars.items() if ori == "H"}
    vertical = {c: line for c, (ori, line) in bars.items() if ori == "V"}
    assert horizontal and vertical, "need both horizontal and vertical bars"

    graph: dict[int, set[int]] = defaultdict(set)
    indegree: dict[int, int] = {color: 0 for color in bars}

    for h_color, h_row in horizontal.items():
        for v_color, v_col in vertical.items():
            if h_color == v_color:
                raise AssertionError("same color used for both orientations")
            visible = int(grid[h_row, v_col])
            if visible == h_color:
                # Horizontal bar is on top of vertical.
                lower, upper = v_color, h_color
            elif visible == v_color:
                # Vertical bar is on top of horizontal.
                lower, upper = h_color, v_color
            elif visible == bg:
                raise AssertionError("crossing cell cannot be background")
            else:
                raise AssertionError("crossing cell color does not match crossing bars")

            if upper not in graph[lower]:
                graph[lower].add(upper)

    for u, nbrs in graph.items():
        for v in nbrs:
            indegree[v] += 1

    order: list[int] = []
    zero = [c for c, d in indegree.items() if d == 0]

    # Unique reconstruction is required: each depth step must be decided.
    while zero:
        if len(zero) != 1:
            raise AssertionError("layer order is not unique")
        cur = zero.pop(0)
        order.append(cur)
        for nxt in sorted(graph[cur]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                zero.append(nxt)

    if len(order) != len(bars):
        raise AssertionError("cycle detected in layer constraints")

    return order


def _shifted_position(ori: str, line: int, depth: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Return a line segment from (r0,c0) inclusive-exclusive for a shifted bar."""
    h, w = map(int, shape)
    shift = int(depth) - 1

    if ori == "H":
        r0 = line + shift
        assert 0 <= r0 < h, "horizontal bar shifted out of bounds"
        return r0, 0, r0 + 1, w
    assert 0 <= line + shift < w, "vertical bar shifted out of bounds"
    c0 = line + shift
    return 0, c0, h, c0 + 1


def main(grid: np.ndarray) -> np.ndarray:
    """Reconstruct bar order and redraw full-span bars by depth-specified shifts.

    Input structure: A dominant background color and several width-1 bars, each of a
    unique foreground color, where each bar is a full row or full column possibly
    interrupted only at orthogonal crossings.

    Rule: infer every bar's full row/column from its visible coordinates, deduce the
    full depth order from crossing-visible-color constraints, shift each bar down/right by
    (depth-1) cells (bottom depth=1), then redraw bars from bottom to top on a copy.

    Key insight: crossing cells encode all hidden ordering, and the same depth index
    controls both the shift and final overlay priority.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)
    assert h <= 30 and w <= 30

    bg = int(get_bg(g))
    bars = _collect_bars(g, bg)
    order = _infer_order_from_crossings(g, bg, bars)

    depth_by_color = {color: idx + 1 for idx, color in enumerate(order)}
    out = np.full((h, w), bg, dtype=int)

    for color in order:
        ori, line = bars[color]
        depth = depth_by_color[color]
        r0, c0, r1, c1 = _shifted_position(ori, line, depth, g.shape)
        out[r0:r1, c0:c1] = int(color)

    return out


def _random_grid_size() -> tuple[int, int]:
    """Choose a random valid grid size for robust yet bounded generation."""
    h = random.randint(12, 26)
    w = random.randint(12, 26)
    return h, w


def _random_palette(background: int, total: int) -> list[int]:
    candidates = [c for c in Color.ALL_COLORS if c != int(background) and c != Color.WHITE and c != Color.PAD]
    return random.sample(candidates, total)


def _compose_visible_input(
    shape: tuple[int, int],
    bg: int,
    bar_specs: dict[int, tuple[str, int]],
    order: Iterable[int],
) -> Grid:
    """Compose visible input from bar specs and full depth order."""
    h, w = map(int, shape)
    grid = np.full((h, w), int(bg), dtype=int)

    # Draw from bottom to top so overlap is visible as final drawing.
    for color in order:
        ori, line = bar_specs[color]
        if ori == "H":
            grid[int(line), :] = int(color)
        else:
            grid[:, int(line)] = int(color)

    return grid


def generate_input() -> np.ndarray:
    """Generate random valid inputs satisfying the full-span-bar hidden-order assumptions."""
    max_attempts = 500
    for _ in range(max_attempts):
        h, w = _random_grid_size()

        max_depth = max(1, min(7, min(h, w) - 1))
        total_bars = random.randint(3, max_depth)
        n_h = random.randint(1, total_bars - 1)
        n_v = total_bars - n_h

        row_capacity = h - total_bars + 1
        col_capacity = w - total_bars + 1
        if n_h > row_capacity or n_v > col_capacity:
            continue

        bg = random.randint(0, 9)
        colors = _random_palette(bg, total_bars)

        random.shuffle(colors)
        h_colors = colors[:n_h]
        v_colors = colors[n_h:]

        h_lines = random.sample(range(row_capacity), n_h)
        v_lines = random.sample(range(col_capacity), n_v)
        # Randomize assignment to reduce color-position correlations.
        random.shuffle(h_lines)
        random.shuffle(v_lines)

        bar_specs: dict[int, tuple[str, int]] = {}
        for c, line in zip(h_colors, h_lines):
            bar_specs[c] = ("H", int(line))
        for c, line in zip(v_colors, v_lines):
            bar_specs[c] = ("V", int(line))

        # Branching for order patterns while preserving variety.
        if random.random() < 0.35:
            order = h_colors + v_colors
            random.shuffle(order)
        elif random.random() < 0.7:
            order = v_colors + h_colors
            random.shuffle(order)
        else:
            order = colors.copy()
            random.shuffle(order)

        inp = _compose_visible_input((h, w), bg, bar_specs, order)

        try:
            out = main(inp)
        except Exception:
            continue

        assert out.shape == inp.shape
        if np.array_equal(main(inp), out):
            return inp

    raise RuntimeError("failed to generate valid 019 input")
