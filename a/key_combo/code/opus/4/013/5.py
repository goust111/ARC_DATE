from __future__ import annotations

import math
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6

PATTERN_COLORS = (BLUE, RED, YELLOW, PINK)
GEN_RETRIES = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _local_orbit(
    r: int,
    c: int,
    height: int,
    width: int,
    axis_r: int,
    axis_c: int,
) -> tuple[tuple[int, int], ...]:
    rows = {int(r)}
    cols = {int(c)}

    rr = 2 * int(axis_r) - int(r)
    cc = 2 * int(axis_c) - int(c)
    if 0 <= rr < height:
        rows.add(int(rr))
    if 0 <= cc < width:
        cols.add(int(cc))

    orbit = {(rr0, cc0) for rr0 in rows for cc0 in cols}
    return tuple(sorted(orbit))


def _iter_local_orbits(
    height: int,
    width: int,
    axis_r: int,
    axis_c: int,
) -> Iterable[tuple[tuple[int, int], ...]]:
    seen = np.zeros((height, width), dtype=bool)
    for r in range(height):
        for c in range(width):
            if seen[r, c]:
                continue
            orbit = _local_orbit(r, c, height, width, axis_r, axis_c)
            for rr, cc in orbit:
                seen[rr, cc] = True
            yield orbit


def _solve_pattern(pattern: np.ndarray, axis_r: int, axis_c: int) -> np.ndarray:
    out = np.array(pattern, dtype=int, copy=True)
    height, width = map(int, out.shape)
    for orbit in _iter_local_orbits(height, width, axis_r, axis_c):
        values = {int(out[r, c]) for r, c in orbit}
        if GREEN not in values:
            continue
        visible = values - {GREEN, BLACK}
        if len(visible) != 1:
            raise ValueError("placeholder orbit is ambiguous")
        fill = next(iter(visible))
        for r, c in orbit:
            if int(out[r, c]) == GREEN:
                out[r, c] = fill
    return out


def _extract_pattern_and_rulers(
    grid: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int], int, int]:
    g = np.asarray(grid, dtype=int)
    pattern_mask = (g != BLACK) & (g != GREY)
    box = _bbox(pattern_mask)
    if box is None:
        raise ValueError("missing pattern")
    r0, c0, r1, c1 = box

    rows = np.arange(g.shape[0])[:, None]
    cols = np.arange(g.shape[1])[None, :]
    grey = g == GREY

    horizontal_mask = grey & (rows >= r0) & (rows <= r1) & (cols < c0)
    vertical_mask = grey & (rows < r0) & (cols >= c0) & (cols <= c1)

    h = int(horizontal_mask.sum())
    v = int(vertical_mask.sum())
    if h <= 0 or v <= 0:
        raise ValueError("missing ruler")
    if int(grey.sum()) != h + v:
        raise ValueError("unexpected grey pixels")

    pattern = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    return pattern, box, h, v


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the green placeholders by reflecting across ruler-defined symmetry axes.

    Input structure:
    - A black background contains one rectangular multicolor pattern built from blue,
      red, yellow, and pink cells, with some cells replaced by green placeholders.
    - A horizontal grey ruler extends leftward from the pattern's left side, and a
      vertical grey ruler extends upward from the pattern's top side.
    - The pattern is the only non-grey foreground region; the rulers stay outside the
      rectangle and do not enter its interior.

    Rule:
    - Measure the horizontal ruler length `h` and vertical ruler length `v`.
    - Place the vertical symmetry axis `h` columns to the right of the pattern's left
      edge and the horizontal symmetry axis `v` rows below the pattern's top edge.
    - For each orbit induced by horizontal and vertical reflection within the pattern
      rectangle, if the orbit contains green placeholders, find the unique visible
      non-green foreground color in that orbit and repaint all green cells with it.
    - Remove the grey rulers by turning them black and leave every other cell unchanged.

    Key insight:
    - The grey rulers do not indicate repetition counts; their lengths encode the
      off-center reflection axes that determine which cells must agree by symmetry.
    """
    g = np.asarray(grid, dtype=int)
    pattern, box, h, v = _extract_pattern_and_rulers(g)
    repaired = _solve_pattern(pattern, axis_r=v, axis_c=h)

    r0, c0, r1, c1 = box
    out = g.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = repaired
    out[out == GREY] = BLACK
    return out


def _make_complete_pattern(
    height: int,
    width: int,
    axis_r: int,
    axis_c: int,
    rng: np.random.Generator,
) -> np.ndarray:
    palette_size = int(rng.integers(3, len(PATTERN_COLORS) + 1))
    palette = [int(x) for x in rng.choice(PATTERN_COLORS, size=palette_size, replace=False)]

    orbits = list(_iter_local_orbits(height, width, axis_r, axis_c))
    order = list(rng.permutation(len(orbits)))
    pattern = np.zeros((height, width), dtype=int)

    forced_colors = palette.copy()
    for idx, orbit_idx in enumerate(order):
        color = forced_colors[idx] if idx < len(forced_colors) else int(rng.choice(palette))
        for r, c in orbits[orbit_idx]:
            pattern[r, c] = color
    return pattern


def _choose_placeholder_pattern(
    complete: np.ndarray,
    axis_r: int,
    axis_c: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    height, width = map(int, complete.shape)
    area = height * width
    min_green = math.ceil(area * 0.20)
    max_green = math.floor(area * 0.35)

    orbits = list(_iter_local_orbits(height, width, axis_r, axis_c))
    size2 = [orbit for orbit in orbits if len(orbit) == 2]
    size4 = [orbit for orbit in orbits if len(orbit) == 4]
    fillable = [orbit for orbit in orbits if len(orbit) >= 2]
    if not size2 or not size4:
        return None

    for _ in range(200):
        used: set[tuple[tuple[int, int], ...]] = set()
        plan: list[tuple[tuple[tuple[int, int], ...], int]] = []
        total = 0

        mandatory = [
            size2[int(rng.integers(len(size2)))],
            size4[int(rng.integers(len(size4)))],
        ]
        for orbit in mandatory:
            orbit_key = tuple(orbit)
            used.add(orbit_key)
            count = 1 if len(orbit) == 2 else int(rng.integers(1, len(orbit)))
            plan.append((orbit, count))
            total += count

        extras = [orbit for orbit in fillable if tuple(orbit) not in used]
        rng.shuffle(extras)
        while total < min_green and extras:
            orbit = extras.pop()
            max_add = min(len(orbit) - 1, max_green - total)
            if max_add <= 0:
                continue
            count = int(rng.integers(1, max_add + 1))
            if total + count < min_green and not extras:
                count = min(max_add, min_green - total)
            plan.append((orbit, count))
            total += count

        if not (min_green <= total <= max_green):
            continue

        pattern = complete.copy()
        for orbit, count in plan:
            chosen = rng.choice(len(orbit), size=count, replace=False)
            for idx in np.atleast_1d(chosen):
                r, c = orbit[int(idx)]
                pattern[r, c] = GREEN

        try:
            repaired = _solve_pattern(pattern, axis_r, axis_c)
        except ValueError:
            continue
        if not np.array_equal(repaired, complete):
            continue
        return pattern
    return None


def _compose_scene(
    pattern: np.ndarray,
    complete: np.ndarray,
    axis_r: int,
    axis_c: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = map(int, pattern.shape)
    top_margin = int(rng.integers(1, 4))
    left_margin = int(rng.integers(1, 4))
    bottom_margin = int(rng.integers(1, 4))
    right_margin = int(rng.integers(1, 4))

    r0 = top_margin + axis_r
    c0 = left_margin + axis_c
    grid_h = r0 + height + bottom_margin
    grid_w = c0 + width + right_margin

    grid = np.zeros((grid_h, grid_w), dtype=int)
    expected = np.zeros((grid_h, grid_w), dtype=int)

    grid[r0 : r0 + height, c0 : c0 + width] = pattern
    expected[r0 : r0 + height, c0 : c0 + width] = complete

    mid_r = r0 + height // 2
    mid_c = c0 + width // 2
    grid[mid_r, c0 - axis_c : c0] = GREY
    grid[r0 - axis_r : r0, mid_c] = GREY
    return grid, expected


def _independent_geometry_ok(
    grid: np.ndarray,
    expected: np.ndarray,
    axis_r: int,
    axis_c: int,
) -> bool:
    try:
        pattern, box, h, v = _extract_pattern_and_rulers(grid)
    except ValueError:
        return False
    if h != axis_c or v != axis_r:
        return False
    r0, c0, r1, c1 = box
    if (r1 - r0 + 1, c1 - c0 + 1) != pattern.shape:
        return False

    center_r = pattern.shape[0] // 2
    center_c = pattern.shape[1] // 2
    if axis_r == center_r or axis_c == center_c:
        return False

    try:
        correct = _solve_pattern(pattern, axis_r, axis_c)
    except ValueError:
        return False
    if not np.array_equal(correct, expected[r0 : r1 + 1, c0 : c1 + 1]):
        return False

    for wrong_axis_r, wrong_axis_c in ((axis_r, center_c), (center_r, axis_c)):
        try:
            wrong = _solve_pattern(pattern, wrong_axis_r, wrong_axis_c)
        except ValueError:
            continue
        if np.array_equal(wrong, correct):
            return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.choice(np.array([5, 7, 9], dtype=int)))
    width = int(rng.choice(np.array([5, 7, 9], dtype=int)))
    center_r = height // 2
    center_c = width // 2

    axis_r_choices = [r for r in range(1, height - 1) if r != center_r]
    axis_c_choices = [c for c in range(1, width - 1) if c != center_c]
    axis_r = int(rng.choice(np.array(axis_r_choices, dtype=int)))
    axis_c = int(rng.choice(np.array(axis_c_choices, dtype=int)))

    complete = _make_complete_pattern(height, width, axis_r, axis_c, rng)
    pattern = _choose_placeholder_pattern(complete, axis_r, axis_c, rng)
    if pattern is None:
        return None

    if len(set(int(x) for x in complete.flatten())) < 3:
        return None

    grid, expected = _compose_scene(pattern, complete, axis_r, axis_c, rng)
    if max(grid.shape) > 30:
        return None
    if not _independent_geometry_ok(grid, expected, axis_r, axis_c):
        return None

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if np.any(actual == GREY) or np.any(actual == GREEN):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ruler-axis symmetry input")
