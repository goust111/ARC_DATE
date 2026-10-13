from __future__ import annotations

import numpy as np


BG = 0
MAX_GENERATION_ATTEMPTS = 300
HEIGHT_RANGE = (10, 18)
WIDTH_RANGE = (11, 19)
BLOCKED_RAY_PROB = 0.7


def _find_bar(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Return (row, c0, c1, color) for the unique bottom horizontal trigger bar."""
    height, width = map(int, grid.shape)
    for r in range(height - 1, -1, -1):
        non_bg = np.flatnonzero(grid[r] != bg)
        if non_bg.size < 2:
            continue
        c0 = int(non_bg[0])
        c1 = int(non_bg[-1])
        color = int(grid[r, c0])
        if non_bg.size != c1 - c0 + 1:
            continue
        if c1 - c0 + 1 >= width:
            continue
        if not np.all(grid[r, c0 : c1 + 1] == color):
            continue
        return int(r), c0, c1, color
    raise AssertionError("expected a single partial-width horizontal bar")


def _trace_ray(
    obstacles: np.ndarray,
    start_r: int,
    start_c: int,
    dc: int,
) -> tuple[list[tuple[int, int]], bool]:
    """Trace one up-left or up-right ray against fixed obstacles."""
    height, width = map(int, obstacles.shape)
    cells: list[tuple[int, int]] = []
    r = int(start_r) - 1
    c = int(start_c) + int(dc)
    while 0 <= r < height and 0 <= c < width:
        if bool(obstacles[r, c]):
            return cells, True
        cells.append((int(r), int(c)))
        r -= 1
        c += int(dc)
    return cells, False


def _settle_grid(
    shape: tuple[int, int],
    bar_row: int,
    bar_c0: int,
    bar_c1: int,
    bar_color: int,
    particle_color: int,
    counts: np.ndarray,
    bg: int,
) -> np.ndarray:
    """Render the bar and all gravity-settled stacks before splash rays."""
    height, width = map(int, shape)
    out = np.full((height, width), int(bg), dtype=int)
    out[bar_row, bar_c0 : bar_c1 + 1] = int(bar_color)

    for c in range(width):
        count = int(counts[c])
        if count <= 0:
            continue
        if bar_c0 <= c <= bar_c1:
            top = int(bar_row) - count
            assert top >= 0, "stack above the bar must fit inside the grid"
            out[top:bar_row, c] = int(particle_color)
        else:
            top = height - count
            assert top >= 0, "floor stack must fit inside the grid"
            out[top:, c] = int(particle_color)
    return out


def _apply_physics(
    shape: tuple[int, int],
    bar_row: int,
    bar_c0: int,
    bar_c1: int,
    bar_color: int,
    particle_color: int,
    counts: np.ndarray,
    bg: int = BG,
) -> np.ndarray:
    """Apply gravity stacking, then the non-mutually-blocking splash rays."""
    settled = _settle_grid(
        shape=shape,
        bar_row=bar_row,
        bar_c0=bar_c0,
        bar_c1=bar_c1,
        bar_color=bar_color,
        particle_color=particle_color,
        counts=counts,
        bg=bg,
    )
    out = settled.copy()
    obstacles = settled != int(bg)
    for c in range(int(bar_c0), int(bar_c1) + 1):
        if int(counts[c]) <= 0:
            continue
        for dc in (-1, 1):
            cells, _blocked = _trace_ray(obstacles, int(bar_row) - 1, int(c), dc)
            for r, cc in cells:
                out[r, cc] = int(particle_color)
    return out


def _ray_diagnostics(
    shape: tuple[int, int],
    bar_row: int,
    bar_c0: int,
    bar_c1: int,
    bar_color: int,
    particle_color: int,
    counts: np.ndarray,
    bg: int = BG,
) -> tuple[int, int]:
    """Return (unique_ray_cells, rays_blocked_by_stacks_or_bar)."""
    settled = _settle_grid(
        shape=shape,
        bar_row=bar_row,
        bar_c0=bar_c0,
        bar_c1=bar_c1,
        bar_color=bar_color,
        particle_color=particle_color,
        counts=counts,
        bg=bg,
    )
    obstacles = settled != int(bg)
    ray_cells: set[tuple[int, int]] = set()
    blocked_count = 0
    for c in range(int(bar_c0), int(bar_c1) + 1):
        if int(counts[c]) <= 0:
            continue
        for dc in (-1, 1):
            cells, blocked = _trace_ray(obstacles, int(bar_row) - 1, int(c), dc)
            ray_cells.update(cells)
            blocked_count += int(blocked)
    return len(ray_cells), int(blocked_count)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let the upper pixels fall into vertical stacks, then splash diagonal rays from the stacks resting on the trigger bar.

    Input structure
    - The background is black.
    - One monochrome horizontal bar occupies only part of a single row.
    - All other foreground cells lie above the bar, and the falling pixels share one color.
    - Multiple falling pixels may appear in the same column.

    Rule
    - Count how many falling pixels occur in each column.
    - In every column, let those pixels fall straight down and stack on the bar, on the grid floor, or on earlier settled pixels from the same column.
    - Only the nonempty stacks whose bottom cell rests directly on the bar emit splash rays.
    - From each such bottom cell, draw one diagonal ray up-left and one up-right.
    - Each ray stops before the first already settled stack pixel, the bar, or the grid boundary, and splash rays do not block one another.
    - Keep the bar fixed.

    Key insight
    - The important state is the per-column particle count after gravity: bar columns become emitters, while off-bar columns only settle to the floor and can act as blockers for the emitted diagonals.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = BG
    assert int(np.max(g.shape)) <= 30

    bar_row, bar_c0, bar_c1, bar_color = _find_bar(g, bg)
    assert bar_row >= 2, "need room for at least one splash step"
    assert np.all(g[bar_row + 1 :, :] == bg), "all movable pixels must start above the bar"
    assert np.all(g[bar_row, :bar_c0] == bg) and np.all(g[bar_row, bar_c1 + 1 :] == bg)

    particle_mask = g[:bar_row, :] != bg
    particle_colors = np.unique(g[:bar_row, :][particle_mask])
    assert particle_colors.size == 1, "expected exactly one falling-pixel color"
    particle_color = int(particle_colors[0])

    counts = particle_mask.sum(axis=0).astype(int)
    assert int(counts.sum()) > 0, "expected at least one falling pixel"
    return _apply_physics(
        shape=g.shape,
        bar_row=bar_row,
        bar_c0=bar_c0,
        bar_c1=bar_c1,
        bar_color=bar_color,
        particle_color=particle_color,
        counts=counts,
        bg=bg,
    )


def _sample_counts(
    rng: np.random.Generator,
    width: int,
    bar_c0: int,
    bar_c1: int,
    bar_row: int,
) -> tuple[np.ndarray, bool]:
    inside = list(range(int(bar_c0), int(bar_c1) + 1))
    outside = [c for c in range(width) if c < bar_c0 or c > bar_c1]
    stack_cap = min(4, (int(bar_row) + 1) // 2)
    counts = np.zeros(width, dtype=int)
    want_block = bool(rng.random() < BLOCKED_RAY_PROB and len(inside) >= 2)

    if want_block:
        pair_left = int(rng.integers(bar_c0, bar_c1))
        if bool(rng.integers(0, 2)):
            blocker, emitter = pair_left, pair_left + 1
        else:
            emitter, blocker = pair_left, pair_left + 1
        counts[emitter] = int(rng.integers(1, stack_cap + 1))
        counts[blocker] = int(rng.integers(2, stack_cap + 1))
    else:
        emitter = int(rng.choice(np.array(inside, dtype=int)))
        counts[emitter] = int(rng.integers(1, stack_cap + 1))

    extra_inside = [c for c in inside if counts[c] == 0]
    if extra_inside:
        extra_n = int(rng.integers(1, min(3, len(extra_inside)) + 1))
        for c in rng.choice(np.array(extra_inside, dtype=int), size=extra_n, replace=False):
            counts[int(c)] = int(rng.integers(1, stack_cap + 1))

    if outside and bool(rng.random() < 0.85):
        outside_n = int(rng.integers(1, min(4, len(outside)) + 1))
        for c in rng.choice(np.array(outside, dtype=int), size=outside_n, replace=False):
            counts[int(c)] = int(rng.integers(1, stack_cap + 1))

    if int(counts.sum()) > min(12, bar_row + 4):
        return counts, want_block
    return counts, want_block


def _render_input(
    rng: np.random.Generator,
    shape: tuple[int, int],
    bar_row: int,
    bar_c0: int,
    bar_c1: int,
    bar_color: int,
    particle_color: int,
    counts: np.ndarray,
) -> np.ndarray | None:
    height, width = map(int, shape)
    grid = np.full((height, width), BG, dtype=int)
    grid[bar_row, bar_c0 : bar_c1 + 1] = int(bar_color)
    parity_offset = int(rng.integers(0, 2))

    for c in range(width):
        count = int(counts[c])
        if count <= 0:
            continue
        allowed_rows = [r for r in range(bar_row) if (r + c + parity_offset) % 2 == 0]
        if count > len(allowed_rows):
            return None
        chosen = rng.choice(np.array(allowed_rows, dtype=int), size=count, replace=False)
        grid[np.array(chosen, dtype=int), c] = int(particle_color)

    if np.any((grid[:bar_row, 1:] != BG) & (grid[:bar_row, :-1] != BG)):
        return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    height = int(rng.integers(HEIGHT_RANGE[0], HEIGHT_RANGE[1] + 1))
    width = int(rng.integers(WIDTH_RANGE[0], WIDTH_RANGE[1] + 1))
    bar_row = int(rng.integers(max(6, height // 2), height - 2))
    bar_len = int(rng.integers(4, width - 2))
    if bar_len >= width:
        return None
    bar_c0 = int(rng.integers(1, width - bar_len))
    bar_c1 = bar_c0 + bar_len - 1

    bar_color = int(rng.integers(1, 10))
    particle_choices = [c for c in range(1, 10) if c != bar_color]
    particle_color = int(rng.choice(np.array(particle_choices, dtype=int)))

    counts, want_block = _sample_counts(rng, width, bar_c0, bar_c1, bar_row)
    if int(np.count_nonzero(counts)) < 3:
        return None

    grid = _render_input(
        rng=rng,
        shape=(height, width),
        bar_row=bar_row,
        bar_c0=bar_c0,
        bar_c1=bar_c1,
        bar_color=bar_color,
        particle_color=particle_color,
        counts=counts,
    )
    if grid is None:
        return None

    expected = _apply_physics(
        shape=(height, width),
        bar_row=bar_row,
        bar_c0=bar_c0,
        bar_c1=bar_c1,
        bar_color=bar_color,
        particle_color=particle_color,
        counts=counts,
        bg=BG,
    )
    ray_cells, blocked_rays = _ray_diagnostics(
        shape=(height, width),
        bar_row=bar_row,
        bar_c0=bar_c0,
        bar_c1=bar_c1,
        bar_color=bar_color,
        particle_color=particle_color,
        counts=counts,
        bg=BG,
    )

    if ray_cells < 2:
        return None
    if want_block and blocked_rays == 0:
        return None
    if np.array_equal(grid, expected):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-stacking splash input")
