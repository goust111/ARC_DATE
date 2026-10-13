import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

GEN_ATTEMPTS = 300
SIZE_RANGE = (8, 25)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _mode_color(values: np.ndarray) -> int:
    counts = np.bincount(np.asarray(values, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _overlap_color(bg: int, diag: int, upper: int | None, lower: int | None) -> int:
    used = {bg, diag}
    if upper is not None:
        used.add(upper)
    if lower is not None:
        used.add(lower)
    if YELLOW not in used:
        return YELLOW
    for color in range(10):
        if color not in used:
            return color
    return YELLOW


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the lower-triangle pattern across the main diagonal and merge it into the upper triangle.

    Input structure:
    - The grid is square and uses one background color over most cells.
    - The main diagonal from top-left to bottom-right is a one-cell-wide line of a single guide color.
    - Foreground pixels appear only above the diagonal and below the diagonal; the upper half supplies one pattern and the lower half supplies another.

    Rule:
    - Keep the grid size unchanged, but clear the entire lower triangle to the background color.
    - Reflect every lower-triangle foreground pixel across the main diagonal, so cell (r, c) contributes to (c, r).
    - In the upper triangle, keep upper-only pixels in their original color, keep reflected lower-only pixels in their original lower-half color, and recolor overlaps with a third marker color.
    - Preserve the diagonal guide line unchanged.

    Key insight:
    - The diagonal is a fold axis, so the lower pattern must be transposed onto the upper half; overlapping contributions are encoded by a distinct third color rather than by keeping either source color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    n, m = g.shape
    assert n == m, "expected a square grid"

    diag_idx = np.arange(n)
    diag_vals = g[diag_idx, diag_idx]
    diag_color = _mode_color(diag_vals)
    assert bool(np.all(diag_vals == diag_color)), "main diagonal must be uniform"

    off_diag_mask = ~np.eye(n, dtype=bool)
    bg = _mode_color(g[off_diag_mask])
    out = np.full_like(g, bg)
    out[diag_idx, diag_idx] = diag_color

    upper_color = None
    lower_color = None
    for r in range(n):
        for c in range(r + 1, n):
            upper_val = int(g[r, c])
            lower_val = int(g[c, r])
            if upper_val not in (bg, diag_color):
                upper_color = upper_val
            if lower_val not in (bg, diag_color):
                lower_color = lower_val
            if upper_color is not None and lower_color is not None:
                break
        if upper_color is not None and lower_color is not None:
            break

    overlap = _overlap_color(bg, diag_color, upper_color, lower_color)

    for r in range(n):
        for c in range(r + 1, n):
            upper_val = int(g[r, c])
            lower_val = int(g[c, r])
            has_upper = upper_val not in (bg, diag_color)
            has_lower = lower_val not in (bg, diag_color)
            if has_upper and has_lower:
                out[r, c] = overlap
            elif has_upper:
                out[r, c] = upper_val
            elif has_lower:
                out[r, c] = lower_val

    return out


def _upper_cells(n: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(n) for c in range(r + 1, n)]


def _neighbors_upper(cell: tuple[int, int], n: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in NEIGHBORS4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < cc < n:
            out.append((rr, cc))
    return out


def _dilate(cells: set[tuple[int, int]], n: int) -> set[tuple[int, int]]:
    expanded = set(cells)
    for cell in cells:
        expanded.update(_neighbors_upper(cell, n))
    return expanded


def _partition(total: int, parts: int, min_each: int) -> list[int]:
    assert total >= parts * min_each
    bins = [min_each] * parts
    for _ in range(total - parts * min_each):
        bins[random.randrange(parts)] += 1
    random.shuffle(bins)
    return bins


def _grow_component(
    n: int,
    size: int,
    allowed: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    if len(allowed) < size:
        return None

    seeds = list(allowed)
    random.shuffle(seeds)
    for seed in seeds[: min(len(seeds), 60)]:
        region = {seed}
        frontier = set(_neighbors_upper(seed, n)) & allowed
        while len(region) < size:
            if not frontier:
                break
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            region.add(cell)
            frontier |= (set(_neighbors_upper(cell, n)) & allowed) - region
        if len(region) == size:
            return region
    return None


def _sample_region(
    n: int,
    total_size: int,
    forbidden: set[tuple[int, int]],
    component_count: int,
) -> set[tuple[int, int]] | None:
    region: set[tuple[int, int]] = set()
    blocked = set(forbidden)
    all_cells = set(_upper_cells(n))

    for size in _partition(total_size, component_count, 1):
        allowed = all_cells - blocked
        component = _grow_component(n, size, allowed)
        if component is None:
            return None
        region |= component
        blocked |= _dilate(component, n)

    return region


def _render_input(
    n: int,
    shared: set[tuple[int, int]],
    red_only: set[tuple[int, int]],
    blue_only: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((n, n), BG, dtype=int)
    idx = np.arange(n)
    grid[idx, idx] = GREY

    for r, c in shared | red_only:
        grid[r, c] = RED
    for r, c in shared | blue_only:
        grid[c, r] = BLUE

    return grid


def _expected_output(
    n: int,
    shared: set[tuple[int, int]],
    red_only: set[tuple[int, int]],
    blue_only: set[tuple[int, int]],
) -> np.ndarray:
    out = np.full((n, n), BG, dtype=int)
    idx = np.arange(n)
    out[idx, idx] = GREY
    for r, c in red_only:
        out[r, c] = RED
    for r, c in blue_only:
        out[r, c] = BLUE
    for r, c in shared:
        out[r, c] = YELLOW
    return out


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*SIZE_RANGE)
    upper_area = n * (n - 1) // 2
    max_total = min(upper_area // 3, 3 * n)
    if max_total < 6:
        return None

    total_active = random.randint(6, max_total)
    shared_size, red_size, blue_size = _partition(total_active, 3, 2)

    shared_components = random.randint(1, min(2, shared_size))
    red_components = random.randint(1, min(3, red_size))
    blue_components = random.randint(1, min(3, blue_size))

    shared = _sample_region(n, shared_size, forbidden=set(), component_count=shared_components)
    if shared is None:
        return None

    red_only = _sample_region(n, red_size, forbidden=shared, component_count=red_components)
    if red_only is None:
        return None

    forbidden = shared | red_only
    blue_only = _sample_region(n, blue_size, forbidden=forbidden, component_count=blue_components)
    if blue_only is None:
        return None

    inp = _render_input(n, shared, red_only, blue_only)
    expected = _expected_output(n, shared, red_only, blue_only)

    try:
        out = main(inp)
    except Exception:
        return None
    if not np.array_equal(out, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
