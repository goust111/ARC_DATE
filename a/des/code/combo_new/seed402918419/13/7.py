import random

import numpy as np


BACKGROUND = 0
ORTHOGONAL_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
# The second wave is described as diagonal growth; using the full Chebyshev
# frontier matches the intended square wavefront without parity holes.
SQUARE_WAVE_DIRS = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)

GRID_MIN, GRID_MAX = 10, 22
INTERIOR_MARGIN = 2
SEED_SEP = 2
ORTH_SEED_RANGE = (1, 3)
DIAG_SEED_RANGE = (1, 3)
GEN_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _foreground_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == BACKGROUND:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ORTHOGONAL_DIRS:
                    nr, nc = cr + dr, cc + dc
                    if not _in_bounds(grid.shape, nr, nc) or visited[nr, nc]:
                        continue
                    if int(grid[nr, nc]) == BACKGROUND:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))
            comps.append(comp)

    return comps


def _find_legend(grid: np.ndarray) -> tuple[int, int, int, set[tuple[int, int]]]:
    h, w = grid.shape
    candidates = []
    for comp in _foreground_components(grid):
        if len(comp) != 2:
            continue
        rows = {r for r, _ in comp}
        cols = {c for _, c in comp}
        on_top_or_bottom = len(rows) == 1 and next(iter(rows)) in {0, h - 1}
        on_left_or_right = len(cols) == 1 and next(iter(cols)) in {0, w - 1}
        if on_top_or_bottom or on_left_or_right:
            candidates.append(set(comp))

    assert len(candidates) == 1, "expected exactly one 2-pixel legend on the grid edge"
    legend_cells = candidates[0]

    legend_colors = [int(grid[r, c]) for r, c in legend_cells]
    assert len(set(legend_colors)) == 2, "legend must contain two different colors"

    boundary_candidates = []
    for color in set(legend_colors):
        outside_legend = int(np.count_nonzero(grid == color)) - legend_colors.count(color)
        if outside_legend == 0:
            boundary_candidates.append(color)
    assert len(boundary_candidates) == 1, "legend must contain exactly one boundary-only color"

    boundary_color = int(boundary_candidates[0])
    orth_color = next(color for color in legend_colors if color != boundary_color)

    other_colors = [int(color) for color in np.unique(grid) if int(color) not in {BACKGROUND, orth_color, boundary_color}]
    assert len(other_colors) == 1, "expected exactly one second seed color besides the legend colors"
    diag_color = int(other_colors[0])

    return orth_color, diag_color, boundary_color, legend_cells


def _collect_claims(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    directions: tuple[tuple[int, int], ...],
    own_color: int,
    boundary_color: int,
    frozen: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    claims: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if not _in_bounds(grid.shape, nr, nc) or (nr, nc) in frozen:
                continue
            value = int(grid[nr, nc])
            if value in {own_color, boundary_color}:
                continue
            claims.add((nr, nc))
    return claims


def _simulate_growth(
    grid: np.ndarray,
    orth_color: int,
    diag_color: int,
    boundary_color: int,
    frozen: set[tuple[int, int]],
) -> tuple[np.ndarray, dict[str, int]]:
    out = np.array(grid, dtype=int, copy=True)
    stats = {"same_step": 0, "occupied": 0}

    while True:
        orth_cells = [tuple(map(int, cell)) for cell in np.argwhere(out == orth_color) if tuple(map(int, cell)) not in frozen]
        diag_cells = [tuple(map(int, cell)) for cell in np.argwhere(out == diag_color)]

        orth_claims = _collect_claims(out, orth_cells, ORTHOGONAL_DIRS, orth_color, boundary_color, frozen)
        diag_claims = _collect_claims(out, diag_cells, SQUARE_WAVE_DIRS, diag_color, boundary_color, frozen)

        if not orth_claims and not diag_claims:
            break

        updates: dict[tuple[int, int], int] = {}
        both = orth_claims & diag_claims

        for cell in both:
            if int(out[cell]) == BACKGROUND:
                stats["same_step"] += 1
            else:
                stats["occupied"] += 1
            updates[cell] = boundary_color

        for cell in orth_claims - both:
            value = int(out[cell])
            if value == BACKGROUND:
                updates[cell] = orth_color
            elif value == diag_color:
                updates[cell] = boundary_color
                stats["occupied"] += 1

        for cell in diag_claims - both:
            value = int(out[cell])
            if value == BACKGROUND:
                updates[cell] = diag_color
            elif value == orth_color:
                updates[cell] = boundary_color
                stats["occupied"] += 1

        if not updates:
            break

        for cell, color in updates.items():
            out[cell] = color

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow two seed colors with different wavefront topologies and mark their contact boundary with a legend color.

    Input structure:
    - The background is black.
    - The main field contains isolated single-pixel seeds in exactly two foreground colors.
    - One grid edge contains a unique 2-pixel legend whose two colors are the orthogonal-growth color and the boundary color.

    Rule:
    - Keep the 2-pixel edge legend unchanged and do not treat it as a seed.
    - The legend's non-boundary color grows by 4-neighbor expansion, producing a diamond-like wavefront.
    - The other seed color grows by a diagonal-oriented square wavefront, expanding one Chebyshev step per round.
    - Grow both colors synchronously: black cells claimed by one wave take that color, while cells claimed by both waves in the same round or entered after the other color already occupied them become the boundary color.
    - Continue until the black background has been fully replaced by one of the two growth colors or by the boundary color.

    Key insight:
    - The edge legend identifies both the collision color and which seed color uses the orthogonal wavefront; the remaining seed color must therefore use the square diagonal-style growth.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    orth_color, diag_color, boundary_color, legend_cells = _find_legend(g)
    out, _ = _simulate_growth(g, orth_color, diag_color, boundary_color, legend_cells)

    assert not np.any(out == BACKGROUND), "growth should cover the whole grid"
    return out


def _place_legend(
    grid: np.ndarray,
    rng: random.Random,
    orth_color: int,
    boundary_color: int,
) -> set[tuple[int, int]]:
    h, w = grid.shape
    edge = rng.choice(("top", "bottom", "left", "right"))
    colors = [orth_color, boundary_color]
    if rng.random() < 0.5:
        colors.reverse()

    if edge == "top":
        c = rng.randrange(0, w - 1)
        cells = [(0, c), (0, c + 1)]
    elif edge == "bottom":
        c = rng.randrange(0, w - 1)
        cells = [(h - 1, c), (h - 1, c + 1)]
    elif edge == "left":
        r = rng.randrange(0, h - 1)
        cells = [(r, 0), (r + 1, 0)]
    else:
        r = rng.randrange(0, h - 1)
        cells = [(r, w - 1), (r + 1, w - 1)]

    for (r, c), color in zip(cells, colors):
        grid[r, c] = color
    return set(cells)


def _is_far_enough(cell: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    r, c = cell
    return all(max(abs(r - rr), abs(c - cc)) >= SEED_SEP for rr, cc in occupied)


def _candidate_positions(shape: tuple[int, int], legend_cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = shape
    cells = []
    for r in range(INTERIOR_MARGIN, h - INTERIOR_MARGIN):
        for c in range(INTERIOR_MARGIN, w - INTERIOR_MARGIN):
            if _is_far_enough((r, c), list(legend_cells)):
                cells.append((r, c))
    return cells


def _pick_seed_sets(
    shape: tuple[int, int],
    legend_cells: set[tuple[int, int]],
    orth_count: int,
    diag_count: int,
    rng: random.Random,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    candidates = _candidate_positions(shape, legend_cells)
    rng.shuffle(candidates)
    if len(candidates) < orth_count + diag_count:
        return None

    min_anchor_sep = max(4, min(shape) // 3)
    anchors: list[tuple[int, int]] = []
    for cell in candidates:
        if not anchors:
            anchors.append(cell)
            continue
        if max(abs(cell[0] - anchors[0][0]), abs(cell[1] - anchors[0][1])) >= min_anchor_sep:
            anchors.append(cell)
            break
    if len(anchors) < 2:
        return None

    occupied = [anchors[0], anchors[1]]
    orth_cells = [anchors[0]]
    diag_cells = [anchors[1]]

    for cell in candidates:
        if cell in occupied or not _is_far_enough(cell, occupied):
            continue
        if len(orth_cells) < orth_count:
            orth_cells.append(cell)
            occupied.append(cell)
        elif len(diag_cells) < diag_count:
            diag_cells.append(cell)
            occupied.append(cell)
        if len(orth_cells) == orth_count and len(diag_cells) == diag_count:
            return orth_cells, diag_cells

    return None


def _visible_color_count(grid: np.ndarray, color: int, exclude: set[tuple[int, int]]) -> int:
    count = int(np.count_nonzero(grid == color))
    for cell in exclude:
        if int(grid[cell]) == color:
            count -= 1
    return count


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    grid = np.zeros((h, w), dtype=int)

    orth_color, diag_color, boundary_color = rng.sample(list(range(1, 10)), 3)
    legend_cells = _place_legend(grid, rng, orth_color, boundary_color)

    orth_count = rng.randint(*ORTH_SEED_RANGE)
    diag_count = rng.randint(*DIAG_SEED_RANGE)
    seed_sets = _pick_seed_sets(grid.shape, legend_cells, orth_count, diag_count, rng)
    if seed_sets is None:
        return None
    orth_cells, diag_cells = seed_sets

    for r, c in orth_cells:
        grid[r, c] = orth_color
    for r, c in diag_cells:
        grid[r, c] = diag_color

    out, stats = _simulate_growth(grid, orth_color, diag_color, boundary_color, legend_cells)
    if np.any(out == BACKGROUND):
        return None

    collision_boundary = _visible_color_count(out, boundary_color, legend_cells)
    visible_orth = _visible_color_count(out, orth_color, legend_cells)
    visible_diag = _visible_color_count(out, diag_color, legend_cells)

    if collision_boundary < 4 or visible_orth < 4 or visible_diag < 4:
        return None
    if stats["same_step"] == 0 or stats["occupied"] == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
