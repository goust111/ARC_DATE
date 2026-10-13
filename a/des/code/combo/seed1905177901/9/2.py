from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass

import numpy as np


GREY = 5
FOREGROUND_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 500


@dataclass(frozen=True)
class LineSpec:
    color: int
    orientation: str
    fixed: int
    start: int
    period: int
    count: int
    visible_indices: tuple[int, ...]

    def cells(self, visible_only: bool = False) -> list[tuple[int, int]]:
        indices = self.visible_indices if visible_only else range(self.count)
        if self.orientation == "H":
            return [(self.fixed, self.start + self.period * idx) for idx in indices]
        return [(self.start + self.period * idx, self.fixed) for idx in indices]


def get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.ravel(grid)))
    return max(counts, key=counts.get)


def _infer_line(coords: np.ndarray) -> tuple[str, int, list[int]]:
    rows = {int(r) for r in coords[:, 0]}
    cols = {int(c) for c in coords[:, 1]}
    if len(rows) == 1 and len(cols) >= 2:
        return "H", next(iter(rows)), sorted(int(c) for c in cols)
    if len(cols) == 1 and len(rows) >= 2:
        return "V", next(iter(cols)), sorted(int(r) for r in rows)
    raise AssertionError("each foreground color must lie on a single row or column")


def _infer_period(values: list[int]) -> int:
    diffs = [b - a for a, b in zip(values, values[1:])]
    assert diffs, "each line needs at least two visible pixels"
    period = diffs[0]
    for diff in diffs[1:]:
        period = math.gcd(period, diff)
    assert period >= 1
    return period


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete sparse horizontal or vertical dashed lines by inferring each color's period.

    Input structure
    - The dominant background color fills the canvas.
    - Each foreground color appears on one horizontal or vertical line.
    - Visible pixels of a color follow a periodic spacing, possibly with some
      interior markers omitted.

    Rule
    - For each foreground color, detect whether its pixels lie on one row or one column.
    - Infer the base period as the gcd of consecutive coordinate gaps.
    - Fill every missing marker position on that arithmetic progression between
      the two furthest known pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(get_bg(g))
    out = g.copy()

    for raw_color in np.unique(g):
        color = int(raw_color)
        if color == bg:
            continue

        coords = np.argwhere(g == color)
        orientation, fixed, values = _infer_line(coords)
        period = _infer_period(values)
        start = values[0]
        stop = values[-1]
        assert (stop - start) % period == 0

        for value in range(start, stop + 1, period):
            if orientation == "H":
                r, c = fixed, value
            else:
                r, c = value, fixed
            assert out[r, c] in (bg, color)
            out[r, c] = color

    return out


def _balanced_orientations(n_lines: int) -> list[str]:
    if n_lines == 1:
        return [random.choice(("H", "V"))]
    orientations = ["H", "V"] * (n_lines // 2)
    if n_lines % 2:
        orientations.append(random.choice(("H", "V")))
    random.shuffle(orientations)
    return orientations


def _shape_choices(axis_len: int) -> list[tuple[int, int]]:
    choices: list[tuple[int, int]] = []
    for period in range(2, min(5, axis_len - 1) + 1):
        max_count = min(7, (axis_len - 1) // period + 1)
        for count in range(4, max_count + 1):
            choices.append((period, count))
    return choices


def _choose_visible_indices(count: int) -> tuple[int, ...]:
    pair_start = random.choice((0, count - 2))
    visible = {0, count - 1, pair_start, pair_start + 1}

    remaining = [idx for idx in range(1, count - 1) if idx not in visible]
    if len(remaining) >= 2:
        extra_count = random.randint(0, len(remaining) - 1)
        visible.update(random.sample(remaining, extra_count))

    return tuple(sorted(visible))


def _blocked_neighborhood(cells: list[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    blocked.add((nr, nc))
    return blocked


def _place_line(
    height: int,
    width: int,
    orientation: str,
    color: int,
    blocked: set[tuple[int, int]],
    used_rows: set[int],
    used_cols: set[int],
) -> LineSpec | None:
    axis_len = width if orientation == "H" else height
    shapes = _shape_choices(axis_len)
    if not shapes:
        return None

    fixed_candidates = list(range(height if orientation == "H" else width))
    if orientation == "H":
        fixed_candidates = [r for r in fixed_candidates if r not in used_rows]
    else:
        fixed_candidates = [c for c in fixed_candidates if c not in used_cols]
    random.shuffle(fixed_candidates)
    random.shuffle(shapes)

    for fixed in fixed_candidates:
        for period, count in shapes:
            span = (count - 1) * period + 1
            starts = list(range(axis_len - span + 1))
            random.shuffle(starts)
            for start in starts:
                spec = LineSpec(
                    color=color,
                    orientation=orientation,
                    fixed=fixed,
                    start=start,
                    period=period,
                    count=count,
                    visible_indices=_choose_visible_indices(count),
                )
                cells = spec.cells(visible_only=False)
                if any(cell in blocked for cell in cells):
                    continue
                return spec
    return None


def _render_grid(height: int, width: int, specs: list[LineSpec], visible_only: bool) -> np.ndarray:
    grid = np.full((height, width), GREY, dtype=int)
    for spec in specs:
        for r, c in spec.cells(visible_only=visible_only):
            grid[r, c] = spec.color
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(10, 18)
    width = random.randint(10, 18)
    max_lines = min(4, len(FOREGROUND_POOL), max(1, (height + width) // 8))
    n_lines = random.randint(1, max_lines)

    colors = random.sample(FOREGROUND_POOL, n_lines)
    orientations = _balanced_orientations(n_lines)

    blocked: set[tuple[int, int]] = set()
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    specs: list[LineSpec] = []

    for color, orientation in zip(colors, orientations):
        spec = _place_line(height, width, orientation, color, blocked, used_rows, used_cols)
        if spec is None:
            return None
        specs.append(spec)
        blocked.update(_blocked_neighborhood(spec.cells(visible_only=False), height, width))
        if orientation == "H":
            used_rows.add(spec.fixed)
        else:
            used_cols.add(spec.fixed)

    inp = _render_grid(height, width, specs, visible_only=True)
    expected = _render_grid(height, width, specs, visible_only=False)
    if np.array_equal(inp, expected):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dashed-line completion input")
