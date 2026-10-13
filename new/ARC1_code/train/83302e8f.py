import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each background region of the broken lattice green if it is one intact rectangular cell, otherwise yellow.

    Input structure
    - The grid has a single background color and one non-background color forming a broken rectangular lattice.
    - Background pixels form 4-connected regions: some are single intact cell interiors, others pass through gaps in the lattice.
    Rule
    - Preserve all non-background lattice pixels.
    - For each 4-connected background region, compare its area with its inclusive bounding-box area.
    - Recolor a region green when it exactly fills that rectangle; recolor it yellow when gaps connect multiple cells and leave lattice pixels inside its bounding box.
    Key insight
    - A single isolated cell is exactly a solid rectangle, while any open multi-cell area is a non-rectangular background component.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    foreground_colors = set(map(int, np.unique(g))) - {bg}
    assert len(foreground_colors) == 1, "expected one lattice color"

    out = g.copy()
    components = components_mask(g == bg, connectivity=4)
    assert components, "expected background regions"
    for coords in components:
        r0, c0, r1, c1 = bbox_coords(coords)
        bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        fill_color = 3 if int(coords.shape[0]) == bbox_area else 4
        out[coords[:, 0], coords[:, 1]] = fill_color

    return out

# --------------- generator (o5x) ---------------

# Colors: background plus allowed lattice colors from the task examples.
_GEN_BG = 0
_GEN_LATTICE_COLORS = (1, 2, 5, 6, 7, 8, 9)
# Canvas structure: buckets preserve the observed size mix, including ambiguous side lengths.
_GEN_SIZE_BUCKETS = (
    ((4, 4), (5, 3)),
    ((4, 5), (6, 3)),
    ((5, 4),),
    ((7, 3),),
    ((5, 5), (6, 4)),
)

# Gap sampling: one protected cell is kept sealed so at least one rectangular region exists.
_GEN_HORIZONTAL, _GEN_VERTICAL = 0, 1
_GEN_MIN_BREACH_DIVISOR, _GEN_MAX_BREACH_DIVISOR = 4, 3
_GEN_GAP_LEN_RANGE = (1, 2)
# Retry policy.
_GEN_MAX_ATTEMPTS = 300


def _candidate_edges(n_cells, protected_cell):
    edges = []
    for r in range(n_cells - 1):
        for c in range(n_cells):
            if protected_cell not in ((r, c), (r + 1, c)):
                edges.append((_GEN_HORIZONTAL, r, c))
    for r in range(n_cells):
        for c in range(n_cells - 1):
            if protected_cell not in ((r, c), (r, c + 1)):
                edges.append((_GEN_VERTICAL, r, c))
    return edges


def _punch_gap(grid, orientation, r, c, cell_size, rng):
    gap_min, gap_max = _GEN_GAP_LEN_RANGE
    gap_len = int(rng.integers(gap_min, min(gap_max, cell_size) + 1))
    offset = int(rng.integers(0, cell_size - gap_len + 1))
    if orientation == _GEN_HORIZONTAL:
        line_r = cell_size + r * (cell_size + 1)
        c0 = c * (cell_size + 1) + offset
        grid[line_r, c0 : c0 + gap_len] = _GEN_BG
    else:
        line_c = cell_size + c * (cell_size + 1)
        r0 = r * (cell_size + 1) + offset
        grid[r0 : r0 + gap_len, line_c] = _GEN_BG


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    size_bucket = _GEN_SIZE_BUCKETS[int(rng.integers(len(_GEN_SIZE_BUCKETS)))]
    n_cells, cell_size = size_bucket[int(rng.integers(len(size_bucket)))]
    lattice_color = _GEN_LATTICE_COLORS[int(rng.integers(len(_GEN_LATTICE_COLORS)))]
    grid_size = n_cells * (cell_size + 1) - 1
    grid = np.full((grid_size, grid_size), _GEN_BG, dtype=int)

    line_positions = [cell_size + k * (cell_size + 1) for k in range(n_cells - 1)]
    for pos in line_positions:
        grid[pos, :] = lattice_color
        grid[:, pos] = lattice_color

    protected_cell = (int(rng.integers(n_cells)), int(rng.integers(n_cells)))
    edges = _candidate_edges(n_cells, protected_cell)

    total_edges = 2 * n_cells * (n_cells - 1)
    min_breaches = (total_edges + _GEN_MIN_BREACH_DIVISOR - 1) // _GEN_MIN_BREACH_DIVISOR
    max_breaches = total_edges // _GEN_MAX_BREACH_DIVISOR
    breach_count = int(rng.integers(min_breaches, max_breaches + 1))
    chosen = rng.choice(len(edges), size=breach_count, replace=False)

    for edge_i in chosen:
        _punch_gap(grid, *edges[int(edge_i)], cell_size, rng)

    out = main(grid)
    if np.any(out == 3) and np.any(out == 4) and not np.any(out == _GEN_BG):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 83302e8f input")
