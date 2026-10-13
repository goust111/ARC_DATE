from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import get_bg
except Exception:
    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


ARC_COLORS = list(range(10))
MAX_GENERATION_ATTEMPTS = 400


def _primitive_period(colors: list[int]) -> bool:
    n = len(colors)
    for p in range(1, n):
        if n % p == 0 and all(colors[i] == colors[i % p] for i in range(n)):
            return False
    return True


def _recover_palette(top_row: np.ndarray, bg: int, gap_col: int) -> list[int]:
    row = [int(v) for v in top_row]
    width = len(row)
    visible = [(c, row[c]) for c in range(width) if c != gap_col]
    assert visible and row[gap_col] == bg

    for period in range(1, width):
        residue_to_color: dict[int, int] = {}
        ok = True
        for c, color in visible:
            residue = c % period
            prev = residue_to_color.get(residue)
            if prev is None:
                residue_to_color[residue] = color
            elif prev != color:
                ok = False
                break
        if ok and len(residue_to_color) == period:
            return [residue_to_color[i] for i in range(period)]
    raise AssertionError("top row must encode a recoverable periodic palette")


def _fold_column(col: int, width: int) -> int:
    if width == 1:
        return 0
    period = 2 * (width - 1)
    folded = int(col) % period
    if folded >= width:
        folded = period - folded
    return folded


def _parse_input(grid: np.ndarray) -> dict[str, int | list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape
    assert height >= 4 and width >= 7

    bg = get_bg(g)
    gap_cols = np.flatnonzero(g[0] == bg)
    assert len(gap_cols) == 1, "top row must have exactly one background gap"
    gap_col = int(gap_cols[0])
    assert np.count_nonzero(g[0] != bg) == width - 1

    below = np.argwhere(g[1:] != bg)
    assert below.size > 0, "expected one marker bar below the palette row"
    below[:, 0] += 1
    marker_cols = {int(c) for _r, c in below}
    assert marker_cols == {gap_col}, "marker bar must sit in the gap column"

    marker_colors = {int(g[r, c]) for r, c in below}
    assert len(marker_colors) == 1, "marker bar must be monochrome"
    marker_color = next(iter(marker_colors))

    bar_rows = sorted(int(r) for r, _c in below)
    assert bar_rows == list(range(bar_rows[0], bar_rows[-1] + 1)), "marker bar must be contiguous"
    bar_top = bar_rows[0]

    palette = _recover_palette(g[0], bg, gap_col)
    assert all(color != bg for color in palette)

    return {
        "bg": int(bg),
        "gap_col": gap_col,
        "marker_color": int(marker_color),
        "bar_top": int(bar_top),
        "height": int(height),
        "width": int(width),
        "period": len(palette),
        "palette": palette,
    }


def _paint_wedge(grid: np.ndarray, spec: dict[str, int | list[int]], *, reflect: bool) -> np.ndarray:
    out = np.full_like(grid, int(spec["bg"]))
    gap_col = int(spec["gap_col"])
    marker_color = int(spec["marker_color"])
    bar_top = int(spec["bar_top"])
    height = int(spec["height"])
    width = int(spec["width"])
    palette = list(spec["palette"])
    period = int(spec["period"])

    out[:, gap_col] = marker_color
    for r in range(bar_top, height):
        steps = r - bar_top + 1
        for i in range(1, steps + 1):
            left_nominal = gap_col - i
            right_nominal = gap_col + i
            left_color = palette[left_nominal % period]
            right_color = palette[right_nominal % period]

            if reflect:
                out[r, _fold_column(left_nominal, width)] = left_color
                out[r, _fold_column(right_nominal, width)] = right_color
            else:
                if 0 <= left_nominal < width:
                    out[r, left_nominal] = left_color
                if 0 <= right_nominal < width:
                    out[r, right_nominal] = right_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a gap-centered wedge whose periodic side colors keep counting while bounced positions reflect off the wall.

    Input structure:
    - The top row is a periodic palette row with exactly one background gap.
    - The only other foreground is one monochrome vertical marker bar in the gap column.
    - The gap is noticeably closer to one side, so that side of the wedge hits a boundary early.

    Rule:
    - Clear the grid to background, then paint the whole gap column in the marker color.
    - Starting at the top row of the marker bar, row k paints k cells on each side of the gap column.
    - On each side, the i-th outward color follows the top-row palette cyclically, while the landing column reflects off boundaries; later reflected hits overwrite earlier paint on that row.

    Key insight:
    - The palette phase is determined by the periodic top row, but positions and colors evolve independently: colors keep advancing by distance, while columns fold back from the wall.
    """
    g = np.asarray(grid, dtype=int)
    spec = _parse_input(g)
    return _paint_wedge(g, spec, reflect=True)


def _build_input(
    *,
    height: int,
    width: int,
    bg: int,
    marker_color: int,
    gap_col: int,
    bar_top: int,
    palette: list[int],
) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    for c in range(width):
        grid[0, c] = palette[c % len(palette)]
    grid[0, gap_col] = bg
    grid[bar_top:, gap_col] = marker_color
    return grid


def _sample_palette(bg: int, marker_color: int) -> list[int]:
    usable = [color for color in ARC_COLORS if color not in {bg, marker_color}]
    period = random.randint(2, min(5, len(usable)))
    palette = random.sample(usable, period)
    assert _primitive_period(palette)
    return palette


def _sample_candidate() -> np.ndarray:
    near_gap = random.randint(3, 5)
    depth = random.randint(near_gap + 2, 2 * near_gap - 1)
    bar_top = random.randint(2, 8)
    height = bar_top + depth

    bg, marker_color = random.sample(ARC_COLORS, 2)
    palette = _sample_palette(bg, marker_color)

    min_width = max(3 * near_gap + 1, 2 * len(palette) + 3, depth + near_gap + 1, 11)
    width = random.randint(min_width, min(30, min_width + 12))

    left_biased = random.choice([True, False])
    gap_col = near_gap if left_biased else width - 1 - near_gap
    grid = _build_input(
        height=height,
        width=width,
        bg=bg,
        marker_color=marker_color,
        gap_col=gap_col,
        bar_top=bar_top,
        palette=palette,
    )
    return grid


def _validate_generated_grid(grid: np.ndarray) -> tuple[bool, str]:
    try:
        spec = _parse_input(grid)
        oracle = _paint_wedge(grid, spec, reflect=True)
        clipped = _paint_wedge(grid, spec, reflect=False)
    except AssertionError as exc:
        return False, str(exc)

    if not isinstance(oracle, np.ndarray) or oracle.ndim != 2:
        return False, "main output must be a 2D ndarray"
    if max(grid.shape) > 30 or max(oracle.shape) > 30:
        return False, "grid exceeds ARC size limit"
    if not np.array_equal(main(grid), oracle):
        return False, "main disagrees with oracle"

    gap_col = int(spec["gap_col"])
    width = int(spec["width"])
    depth = int(spec["height"]) - int(spec["bar_top"])
    left_touch = depth > gap_col
    right_touch = depth > (width - 1 - gap_col)
    if left_touch == right_touch:
        return False, "exactly one side must touch a wall"
    if np.array_equal(oracle, clipped):
        return False, "reflection must matter beyond simple clipping"

    palette = list(spec["palette"])
    if len(set(palette)) != len(palette):
        return False, "palette colors must stay visually distinct"
    if len(palette) < 2:
        return False, "palette period is too small"
    if np.unique(oracle).size < 3:
        return False, "output is too degenerate"
    return True, "ok"


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _sample_candidate()
        ok, _reason = _validate_generated_grid(grid)
        if ok:
            return grid
    raise RuntimeError("failed to generate a valid non-degenerate input")
