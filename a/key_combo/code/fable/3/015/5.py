from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

try:
    from arc_utils import Color, find_components, get_bg, randomly_spaced_indices
except ModuleNotFoundError:
    _file = Path(__file__).resolve()
    _arc_utils_path = None
    for parent in _file.parents:
        candidate = parent / "new" / "solvers" / "arc_utils.py"
        if candidate.is_file():
            _arc_utils_path = candidate.parent
            break
    if _arc_utils_path is None:
        raise
    if str(_arc_utils_path) not in sys.path:
        sys.path.insert(0, str(_arc_utils_path))
    from arc_utils import Color, find_components, get_bg, randomly_spaced_indices


def _extract_bars(g: np.ndarray, *, bg: int) -> list[dict[str, int]]:
    components = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )

    bars: list[dict[str, int]] = []
    bar_color = None

    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        if c0 != c1:
            continue
        if r1 == r0:
            continue
        color = int(comp["color"])

        if bar_color is None:
            bar_color = color
        elif color != bar_color:
            raise AssertionError("all vertical bars must share one marker color")

        top = int(r0)
        marker_row = top - 1
        if marker_row <= 0:
            raise AssertionError("top row must stay empty and marker must be above each bar")

        marker_color = int(g[marker_row, c0])
        if marker_color == bg or marker_color == color:
            raise AssertionError("marker pixel above each bar must be non-background and non-bar color")

        bars.append(
            {
                "col": int(c0),
                "top": top,
                "height": int(r1 - r0 + 1),
                "marker": marker_color,
            }
        )

    assert bars, "expected at least one vertical bar"
    return bars


def _make_expected_top_row(g: np.ndarray, bars: list[dict[str, int]], *, bg: int) -> np.ndarray:
    heights = [b["height"] for b in bars]
    assert len(set(heights)) == len(bars), "bar heights must be unique"
    assert all(1 <= b["col"] < g.shape[1] - 1 for b in bars)

    bars_sorted = sorted(bars, key=lambda item: item["height"])
    seq = [b["marker"] for b in bars_sorted]
    period = len(seq)
    assert period > 0

    anchor_bar = bars_sorted[-1]
    anchor_col = anchor_bar["col"]
    anchor_idx = period - 1

    out = np.full(g.shape[1], bg, dtype=int)
    for c in range(g.shape[1]):
        out[c] = int(seq[(anchor_idx + (c - anchor_col)) % period])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort vertical bars by height and tile their top marker colors across the top row by anchored periodicity.

    Input structure
    - The dominant background is black.
    - Several disjoint one-pixel-wide vertical gray bars stand on the background.
    - Bar heights are all distinct, and each bar has one color marker pixel directly above its top.
    - The top row is empty.

    Rule
    - Read every bar's marker color, sort bars by height from shortest to tallest, and build the marker color sequence.
    - In the top row, write the sequence element that corresponds to the tallest bar at that bar's column.
    - Tile this sequence with period equal to the sequence length to the left and right to fill the entire top row.
    - Keep all bars and marker pixels unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"
    assert np.all(g[0] == bg), "top row must be empty in input"

    bars = _extract_bars(g, bg=bg)
    assert len(bars) >= 2, "expected at least two bars"

    expected_top_row = _make_expected_top_row(g, bars, bg=bg)

    out = g.copy()
    out[0, :] = expected_top_row
    return out


_MIN_SIDE = 12
_MAX_SIDE = 30
_MIN_BARS = 2
_MAX_BARS = 6
_ATTEMPTS = 500


def _candidate_markers(rng: random.Random) -> list[int]:
    palette = [c for c in Color.NOT_BLACK if c != Color.GREY]
    return list(palette)


def _generate_once() -> np.ndarray | None:
    rng = random.Random()

    height = int(rng.randint(_MIN_SIDE, _MAX_SIDE))
    width = int(rng.randint(_MIN_SIDE, _MAX_SIDE))

    max_by_width = max(_MIN_BARS, (width - 2) // 2)
    if max_by_width < _MIN_BARS:
        return None
    n_bars = int(rng.randint(_MIN_BARS, min(_MAX_BARS, max_by_width)))

    max_h = height - 2
    if max_h < 2 or max_h < n_bars:
        return None
    heights = rng.sample(range(2, max_h + 1), n_bars)

    try:
        cols = randomly_spaced_indices(width, n_bars, border_size=1, padding=1).astype(int)
        cols = cols.tolist()
    except Exception:
        return None
    rng.shuffle(cols)

    bg = Color.BLACK
    bar_color = Color.GREY
    markers = _candidate_markers(rng)
    if not markers:
        return None

    grid = np.full((height, width), bg, dtype=int)
    for height_val, col in sorted(zip(heights, cols), reverse=True):
        top = int(rng.randint(2, height - height_val))
        grid[top : top + height_val, col] = bar_color
        marker_color = int(rng.choice(markers))
        grid[top - 1, col] = marker_color

    try:
        bars = _extract_bars(grid, bg=bg)
        expected_top = _make_expected_top_row(grid, bars, bg=bg)
        out = main(grid)
        if not np.array_equal(out[1:, :], grid[1:, :]):
            return None
        if not np.array_equal(out[0], expected_top):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5.py input")
