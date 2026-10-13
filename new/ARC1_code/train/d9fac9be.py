import numpy as np

from arc_utils import get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """Extract the color enclosed by the unique 3x3 monochrome frame and return it as a 1x1 grid.

    Input structure
    - A background canvas contains scattered foreground pixels in two visible colors.
    - Exactly one 3x3 window has all eight border pixels in one foreground color and its center in a different foreground color.

    Rule
    - Find that window and output a 1x1 grid containing the center pixel color.

    Key insight
    - The complete 3x3 frame marks the meaningful sample; the scattered pixels of the same colors are distractors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 3 and w >= 3

    bg = get_bg(g)
    candidates = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            center = int(g[r, c])
            if center == bg:
                continue
            patch = g[r - 1 : r + 2, c - 1 : c + 2]
            ring = np.array(
                [
                    patch[0, 0],
                    patch[0, 1],
                    patch[0, 2],
                    patch[1, 0],
                    patch[1, 2],
                    patch[2, 0],
                    patch[2, 1],
                    patch[2, 2],
                ],
                dtype=int,
            )
            ring_colors = set(map(int, ring))
            if len(ring_colors) != 1:
                continue
            ring_color = next(iter(ring_colors))
            if ring_color != bg and ring_color != center:
                candidates.append(center)

    assert len(candidates) == 1
    return np.array([[candidates[0]]], dtype=int)

# --------------- generator (o5x) ---------------

# Canvas and palette bounds.
GEN_GRID_ROWS, GEN_GRID_COLS = (5, 13), (9, 13)
GEN_BACKGROUND = 0
GEN_COLOR_VALUES = np.arange(1, 10)

# Frame geometry and distractor counts.
GEN_FRAME_CELLS = 9
GEN_CENTER_EXTRA = (7, 13)
GEN_RING_EXTRA_MAX = 12

# Density/background guardrails.
GEN_DENSITY_CAP, GEN_BG_RING_MARGIN = 0.36, 18

# Retry budget.
GEN_MAX_ATTEMPTS = 300


def _gen_frame_candidates(grid):
    """Return 3x3 frame candidates as (row, col, ring_color, center_color)."""
    h, w = grid.shape
    frames = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            center = int(grid[r, c])
            if center == GEN_BACKGROUND:
                continue
            patch = grid[r - 1 : r + 2, c - 1 : c + 2]
            ring_color = int(patch[0, 0])
            if (
                ring_color != GEN_BACKGROUND
                and ring_color != center
                and patch[0, 1] == patch[0, 2] == patch[1, 0] == patch[1, 2] == ring_color
                and patch[2, 0] == patch[2, 1] == patch[2, 2] == ring_color
            ):
                frames.append((r, c, ring_color, center))
    return frames


def _gen_black_is_bg(grid):
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(counts[GEN_BACKGROUND]) > int(counts[1:].max())


def _gen_place_frame(grid, center_r, center_c, ring_color, center_color):
    grid[center_r - 1 : center_r + 2, center_c - 1 : center_c + 2] = ring_color
    grid[center_r, center_c] = center_color


def _gen_neighbor_load(grid, r, c):
    r0 = max(0, r - 1)
    r1 = min(grid.shape[0], r + 2)
    c0 = max(0, c - 1)
    c1 = min(grid.shape[1], c + 2)
    return int(np.count_nonzero(grid[r0:r1, c0:c1]))


def _gen_sample_distractor_counts(h, w):
    """Sample center then ring distractor counts, preserving rejection rules."""
    density_cap = int(np.floor(GEN_DENSITY_CAP * h * w))
    valid_extra_center = [n for n in range(*GEN_CENTER_EXTRA) if GEN_FRAME_CELLS + n <= density_cap]
    if not valid_extra_center:
        return None
    extra_center = int(valid_extra_center[np.random.randint(len(valid_extra_center))])
    max_extra_ring = min(
        GEN_RING_EXTRA_MAX,
        density_cap - GEN_FRAME_CELLS - extra_center,
        (h * w - GEN_BG_RING_MARGIN - extra_center) // 2,
    )
    if max_extra_ring < 0:
        return None
    extra_ring = int(np.random.randint(0, max_extra_ring + 1))
    return extra_ring, extra_center


def _gen_place_distractors(grid, distractors):
    """Place distractors in safe cells with the lowest local load."""
    for color in distractors:
        safe = []
        for r, c in np.argwhere(grid == GEN_BACKGROUND):
            trial = grid.copy()
            trial[int(r), int(c)] = int(color)
            if len(_gen_frame_candidates(trial)) == 1:
                safe.append((int(r), int(c)))
        if not safe:
            return False
        loads = [_gen_neighbor_load(grid, r, c) for r, c in safe]
        min_load = min(loads)
        best = [cell for cell, load in zip(safe, loads) if load == min_load]
        r, c = best[int(np.random.randint(len(best)))]
        grid[r, c] = int(color)
    return True


def _gen_is_valid(grid, center_color):
    """Run the final structural and solver-oracle checks."""
    frames = _gen_frame_candidates(grid)
    return (
        len(frames) == 1 and frames[0][3] == center_color and _gen_black_is_bg(grid)
        and np.array_equal(main(grid), np.array([[center_color]], dtype=int))
    )


def _gen_attempt():
    h = int(np.random.randint(*GEN_GRID_ROWS))
    w = int(np.random.randint(*GEN_GRID_COLS))
    colors = np.random.choice(GEN_COLOR_VALUES, size=2, replace=False)
    ring_color, center_color = map(int, colors)

    counts = _gen_sample_distractor_counts(h, w)
    if counts is None:
        return None
    extra_ring, extra_center = counts

    grid = np.zeros((h, w), dtype=int)
    center_r = int(np.random.randint(1, h - 1))
    center_c = int(np.random.randint(1, w - 1))
    _gen_place_frame(grid, center_r, center_c, ring_color, center_color)

    distractors = np.array([ring_color] * extra_ring + [center_color] * extra_center, dtype=int)
    if distractors.size:
        distractors = np.random.permutation(distractors)

    if not _gen_place_distractors(grid, distractors):
        return None
    if not _gen_is_valid(grid, center_color):
        return None
    return grid


def generate_input():
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid d9fac9be input")
