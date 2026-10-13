from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
CYAN = 8

MAX_GEN_ATTEMPTS = 400


@dataclass(frozen=True)
class Bar:
    top: int
    col: int
    bottom: int

    @property
    def height(self) -> int:
        return self.bottom - self.top + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    mask = grid == int(color)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_start(grid: np.ndarray) -> tuple[int, int]:
    starts = np.argwhere(grid == GREEN)
    assert starts.shape[0] == 1, "expected exactly one green start pixel"
    r, c = map(int, starts[0])
    assert c == 0, "green start must be on the left edge"
    return r, c


def _blue_columns(grid: np.ndarray) -> list[int]:
    cols = sorted({int(c) for _r, c in np.argwhere(grid == BLUE)})
    return cols


def _find_bars(grid: np.ndarray) -> list[Bar]:
    bars: list[Bar] = []
    for comp in _components_of_color(grid, RED):
        r0, c0, r1, c1 = _bbox_coords(comp)
        assert c0 == c1, "red objects must be 1-cell-wide vertical bars"
        bars.append(Bar(r0, c0, r1))
    bars.sort(key=lambda bar: (bar.col, bar.top))
    return bars


def _ray_rows_by_column(grid: np.ndarray) -> dict[int, int]:
    h, w = grid.shape
    start_row, start_col = _find_start(grid)
    blue_cols = set(_blue_columns(grid))

    ray_rows: dict[int, int] = {}
    row = start_row
    for col in range(start_col + 1, w):
        if row >= h:
            break
        ray_rows[col] = row
        if col in blue_cols:
            row += 1
    return ray_rows


def _apply_rows(grid: np.ndarray, hit_rows: dict[int, int]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    out[(out == BLUE) | (out == GREEN)] = BG

    for bar in _find_bars(np.asarray(grid, dtype=int)):
        hit_row = hit_rows.get(bar.col)
        if hit_row is None or not (bar.top <= hit_row <= bar.bottom):
            continue
        out[bar.top : hit_row + 1, bar.col] = RED
        if hit_row < bar.bottom:
            out[hit_row + 1 : bar.bottom + 1, bar.col] = CYAN
    return out


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    return _apply_rows(g, _ray_rows_by_column(g))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each red vertical bar at the hit point of a left-to-right ray that steps downward at blue deflector columns.

    Input structure:
    - A black background contains 3-5 red 1-cell-wide vertical bars in distinct columns.
    - A single green pixel on the left edge marks the ray start row.
    - Several blue pixels occupy columns that do not contain red bars.

    Rule:
    - Fire a ray to the right from the green pixel's row.
    - At each visited column, first test that column's current row against any red bar there; if the row lies inside the bar, that row is the split point.
    - If the current column contains any blue pixel, the ray moves down by one row starting from the next column.
    - For every hit red bar, keep the hit row and everything above it red, recolor the cells below it cyan, and erase all blue and green pixels to black.

    Key insight:
    - The split row of each bar is not intrinsic to the bar itself; it is determined by the cumulative number of earlier blue deflector columns encountered by a single global ray.
    """
    return _solve(grid)


def _bar_modes(count: int, rng: np.random.Generator) -> list[str]:
    base = ["top", "middle", "bottom"]
    modes = [base[i % len(base)] for i in range(count)]
    if count > 3:
        extra = rng.choice(np.array(["top", "middle", "bottom"], dtype=object), size=count - 3, replace=True)
        for i, mode in enumerate(extra, start=3):
            modes[i] = str(mode)
    return modes


def _choose_bar_span(
    hit_row: int, height_limit: int, preferred_mode: str, rng: np.random.Generator
) -> tuple[int, int]:
    max_height = min(10, height_limit)
    mode_order = [preferred_mode] + [m for m in ("middle", "top", "bottom") if m != preferred_mode]

    for mode in mode_order:
        if mode == "top":
            heights = [h for h in range(3, max_height + 1) if hit_row + h - 1 < height_limit]
            if heights:
                height = int(rng.choice(heights))
                return hit_row, hit_row + height - 1
        elif mode == "bottom":
            heights = [h for h in range(3, max_height + 1) if hit_row - h + 1 >= 0]
            if heights:
                height = int(rng.choice(heights))
                return hit_row - height + 1, hit_row
        else:
            choices: list[tuple[int, int]] = []
            for height in range(3, max_height + 1):
                top_lo = max(0, hit_row - height + 2)
                top_hi = min(hit_row - 1, height_limit - height)
                for top in range(top_lo, top_hi + 1):
                    bottom = top + height - 1
                    if top < hit_row < bottom:
                        choices.append((top, bottom))
            if choices:
                return choices[int(rng.integers(len(choices)))]

    raise RuntimeError("failed to place a bar around the hit row")


def _build_candidate(rng: np.random.Generator) -> np.ndarray:
    bar_count = int(rng.integers(3, 6))
    deflector_count = int(rng.integers(2, min(4, bar_count - 1) + 1))

    gap_choices = rng.integers(4, 7, size=bar_count - 1)
    first_bar_col = int(rng.integers(3, 6))
    bar_cols = [first_bar_col]
    for gap in gap_choices:
        bar_cols.append(bar_cols[-1] + int(gap))

    max_width = min(30, bar_cols[-1] + 5)
    min_width = max(14, bar_cols[-1] + 2)
    if min_width > max_width:
        raise RuntimeError("bar layout exceeded width budget")
    width = int(rng.integers(min_width, max_width + 1))

    min_height = max(14, deflector_count + 8)
    height = int(rng.integers(min_height, 25))
    start_row_max = height - deflector_count - 4
    if start_row_max < 2:
        raise RuntimeError("not enough vertical room for the ray")
    start_row = int(rng.integers(2, start_row_max + 1))

    chosen_gaps = sorted(
        int(i) for i in rng.choice(np.arange(bar_count - 1), size=deflector_count, replace=False)
    )
    deflector_cols = [
        bar_cols[i] + (bar_cols[i + 1] - bar_cols[i]) // 2
        for i in chosen_gaps
    ]

    hit_rows = {
        bar_cols[i]: start_row + sum(gap_idx < i for gap_idx in chosen_gaps)
        for i in range(bar_count)
    }

    grid = np.full((height, width), BG, dtype=int)
    grid[start_row, 0] = GREEN

    used_blue_rows: set[tuple[int, int]] = set()
    for col in deflector_cols:
        row = int(rng.integers(0, height))
        while (row, col) in used_blue_rows:
            row = int(rng.integers(0, height))
        used_blue_rows.add((row, col))
        grid[row, col] = BLUE

    modes = _bar_modes(bar_count, rng)
    for idx, col in enumerate(bar_cols):
        top, bottom = _choose_bar_span(hit_rows[col], height, modes[idx], rng)
        grid[top : bottom + 1, col] = RED

    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False

    h, _w = g.shape
    start_row, start_col = _find_start(g)
    if start_col != 0:
        return False

    bars = _find_bars(g)
    blue_cols = _blue_columns(g)
    if not (3 <= len(bars) <= 5 and 2 <= len(blue_cols) <= 4):
        return False
    if len(blue_cols) != len(set(blue_cols)):
        return False
    if any(bar.height < 3 or bar.height > 10 for bar in bars):
        return False
    if any(b2.col - b1.col < 3 for b1, b2 in zip(bars, bars[1:])):
        return False
    if any(col == 0 or col in {bar.col for bar in bars} for col in blue_cols):
        return False
    if start_row + len(blue_cols) > h - 1:
        return False

    ray_rows = _ray_rows_by_column(g)
    hit_rows = []
    changed_count = 0
    for bar in bars:
        hit_row = ray_rows.get(bar.col)
        if hit_row is None or not (bar.top <= hit_row <= bar.bottom):
            return False
        hit_rows.append(hit_row)
        if hit_row < bar.bottom:
            changed_count += 1

    if changed_count == 0 or len(set(hit_rows)) < 2:
        return False

    true_out = _solve(g)
    no_deflect_out = _apply_rows(g, {bar.col: start_row for bar in bars})
    total_deflect_out = _apply_rows(g, {bar.col: start_row + len(blue_cols) for bar in bars})
    midpoint_out = _apply_rows(g, {bar.col: (bar.top + bar.bottom) // 2 for bar in bars})

    if np.array_equal(true_out, no_deflect_out):
        return False
    if np.array_equal(true_out, total_deflect_out):
        return False
    if np.array_equal(true_out, midpoint_out):
        return False

    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            grid = _build_candidate(rng)
        except RuntimeError:
            continue
        if _validate_generated(grid):
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_ATTEMPTS} attempts")
