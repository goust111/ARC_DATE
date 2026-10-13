from __future__ import annotations

import numpy as np

from arc_utils import get_bg, in_bounds
import random


PIVOT_COLOR = 5


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect every foreground pixel 180 degrees through the single grey pivot pixel.

    Input structure
    - A black background contains a small colored foreground mask.
    - Exactly one grey pixel sits near the mask and acts as the pivot.

    Rule
    - For each non-background pixel at (r, c), place the same color at
      (2 * pivot_r - r, 2 * pivot_c - c).
    - The grey pivot maps to itself, and all original foreground cells are otherwise cleared.

    Key insight
    - The grey pixel is the center of point symmetry; the output is the same colored mask after a half-turn about that pixel.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert bg == 0, "expected black background"

    pivots = np.argwhere(g == PIVOT_COLOR)
    assert len(pivots) == 1, "expected exactly one grey pivot pixel"
    pivot_r, pivot_c = map(int, pivots[0])

    out = np.full_like(g, bg)
    foreground = np.argwhere(g != bg)
    assert len(foreground) > 1, "expected foreground around the pivot"

    for r_raw, c_raw in foreground:
        r = int(r_raw)
        c = int(c_raw)
        rr = 2 * pivot_r - r
        cc = 2 * pivot_c - c
        assert in_bounds(g.shape, rr, cc), "reflected foreground must stay inside the grid"
        out[rr, cc] = int(g[r, c])

    assert np.count_nonzero(out != bg) == len(foreground), "reflection must preserve foreground cells"
    return out

# --------------- generator (o5x) ---------------

# Canvas bounds.
MIN_GRID_SIZE = 3
MAX_GRID_SIZE = 9

# Non-pivot mask composition.
MIN_MASK_CELLS = 4
MAX_MASK_CELLS = 8
MIN_DISTINCT_COLORS = 3
MAX_DISTINCT_COLORS = 4
FOREGROUND_PALETTE = [1, 2, 3, 4, 6, 8]
# Layout and retry controls.
ORIENTATION_SIGNS = [-1, 1]
MAX_GENERATION_ATTEMPTS = 300
BACKGROUND_COLOR = 0


def _gen_half_offsets(v_radius: int, h_radius: int, h_sign: int, v_sign: int) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    for dr in range(-v_radius, v_radius + 1):
        for dc_abs in range(1, h_radius + 1):
            offsets.append((dr, h_sign * dc_abs))
    for dr_abs in range(1, v_radius + 1):
        offsets.append((v_sign * dr_abs, 0))
    return offsets


def _gen_grow_offsets(
    allowed: list[tuple[int, int]], cell_count: int
) -> list[tuple[int, int]] | None:
    allowed_set = set(allowed)
    chosen: set[tuple[int, int]] = set()
    frontier = {
        (dr, dc)
        for dr, dc in allowed_set
        if max(abs(dr), abs(dc)) == 1
    }
    while len(chosen) < cell_count:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        chosen.add(cell)
        frontier.discard(cell)
        for cand in allowed_set - chosen:
            if max(abs(cand[0]), abs(cand[1])) == 1:
                frontier.add(cand)
                continue
            if any(max(abs(cand[0] - r), abs(cand[1] - c)) == 1 for r, c in chosen):
                frontier.add(cand)
    return list(chosen)


def _gen_color_sequence(cell_count: int, distinct_count: int) -> list[int]:
    colors = random.sample(FOREGROUND_PALETTE, distinct_count)
    counts = [1] * distinct_count
    for i in range(cell_count - distinct_count):
        counts[i % distinct_count] += 1
    seq: list[int] = []
    for color, count in zip(colors, counts):
        seq.extend([color] * count)
    random.shuffle(seq)
    return seq


def _gen_feasible_pivots(height: int, width: int, cell_count: int) -> list[tuple[int, int, int, int]]:
    pivots: list[tuple[int, int, int, int]] = []
    for pr in range(height):
        v_radius = min(pr, height - 1 - pr)
        for pc in range(width):
            h_radius = min(pc, width - 1 - pc)
            capacity = (2 * v_radius + 1) * h_radius + v_radius
            if h_radius >= 1 and capacity >= cell_count:
                pivots.append((pr, pc, v_radius, h_radius))
    return pivots


def _gen_feasible_sizes(cell_count: int, max_color_count: int) -> list[tuple[int, int]]:
    feasible_sizes: list[tuple[int, int]] = []
    for height in range(MIN_GRID_SIZE, MAX_GRID_SIZE + 1):
        for width in range(MIN_GRID_SIZE, MAX_GRID_SIZE + 1):
            black_count = height * width - cell_count - 1
            if black_count <= max_color_count:
                continue
            if _gen_feasible_pivots(height, width, cell_count):
                feasible_sizes.append((height, width))
    return feasible_sizes


def _gen_render_candidate(height, width, pr, pc, output_offsets, colors) -> np.ndarray | None:
    """Render the reflected pair and reject collisions or solver mismatches."""
    expected = np.full((height, width), BACKGROUND_COLOR, dtype=int)
    inp = np.full((height, width), BACKGROUND_COLOR, dtype=int)
    expected[pr, pc] = PIVOT_COLOR
    inp[pr, pc] = PIVOT_COLOR

    input_cells: set[tuple[int, int]] = {(pr, pc)}
    output_cells: set[tuple[int, int]] = {(pr, pc)}
    for (dr, dc), color in zip(output_offsets, colors):
        out_r, out_c = pr + dr, pc + dc
        in_r, in_c = pr - dr, pc - dc
        if (out_r, out_c) in output_cells or (in_r, in_c) in input_cells:
            return None
        if (out_r, out_c) in input_cells or (in_r, in_c) in output_cells:
            return None
        expected[out_r, out_c] = color
        inp[in_r, in_c] = color
        output_cells.add((out_r, out_c))
        input_cells.add((in_r, in_c))

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def _gen_attempt() -> np.ndarray | None:
    cell_count = random.randint(MIN_MASK_CELLS, MAX_MASK_CELLS)
    distinct_count = random.randint(MIN_DISTINCT_COLORS, min(MAX_DISTINCT_COLORS, cell_count))
    max_color_count = (cell_count + distinct_count - 1) // distinct_count

    feasible_sizes = _gen_feasible_sizes(cell_count, max_color_count)
    if not feasible_sizes:
        return None
    height, width = random.choice(feasible_sizes)
    pivots = _gen_feasible_pivots(height, width, cell_count)
    pr, pc, v_radius, h_radius = random.choice(pivots)

    h_sign = random.choice(ORIENTATION_SIGNS)
    v_sign = random.choice(ORIENTATION_SIGNS)
    allowed = _gen_half_offsets(v_radius, h_radius, h_sign, v_sign)
    if len(allowed) < cell_count:
        return None

    output_offsets = _gen_grow_offsets(allowed, cell_count)
    if output_offsets is None:
        return None
    colors = _gen_color_sequence(cell_count, distinct_count)

    return _gen_render_candidate(height, width, pr, pc, output_offsets, colors)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_ATTEMPTS} attempts")
