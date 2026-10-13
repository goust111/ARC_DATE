from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
QUADRANT_COLORS = (1, 2, 3, 7)
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
PRIORITY_ORDER = (0, 1, 2, 3)  # TL, TR, BL, BR
GEN_Q_OPTIONS = (6, 7)
GEN_MAX_ATTEMPTS = 400


def _find_separator(grid: np.ndarray) -> tuple[int, int]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == SEPARATOR)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEPARATOR)]
    assert len(rows) == 1 and len(cols) == 1, "expected exactly one solid separator cross"

    split_r = rows[0]
    split_c = cols[0]
    assert split_r == split_c, "separator must split the grid into equal quadrants"
    assert grid.shape == (2 * split_r + 1, 2 * split_c + 1), "input must be centered around the separator"
    return int(split_r), int(split_c)


def _extract_quadrants(grid: np.ndarray) -> list[np.ndarray]:
    split_r, split_c = _find_separator(grid)
    quadrants = [
        grid[:split_r, :split_c],
        grid[:split_r, split_c + 1 :],
        grid[split_r + 1 :, :split_c],
        grid[split_r + 1 :, split_c + 1 :],
    ]
    shapes = {quad.shape for quad in quadrants}
    assert len(shapes) == 1, "quadrants must have the same shape"
    return quadrants


def _overlay_quadrants(quadrants: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    qh, qw = quadrants[0].shape
    out = np.full((qh, qw), BACKGROUND, dtype=int)
    occupancy = np.zeros((qh, qw), dtype=int)
    for quad in quadrants:
        occupancy += (quad != BACKGROUND).astype(int)
        mask = (out == BACKGROUND) & (quad != BACKGROUND)
        out[mask] = quad[mask]
    return out, occupancy >= 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay four quadrant masks, then emit diagonal rays from every multi-quadrant overlap.

    Input structure:
    - A square odd-sized grid is split by one solid grey row and one solid grey column.
    - The four equal quadrants contain black background plus sparse foreground pixels.
    - Each quadrant uses exactly one foreground color, and aligned quadrant cells may overlap in position.

    Rule:
    - Remove the grey separator cross and compare aligned cells from the quadrants in row-major quadrant order:
      top-left, top-right, bottom-left, bottom-right.
    - The first non-background pixel at each aligned position becomes the base output color there.
    - Any aligned position where at least two quadrants contributed a foreground pixel is a ray source.
    - Process ray sources in row-major order; from each source, extend four diagonal rays in the source color.
      A ray paints only background cells and stops before the grid edge or the first non-background cell.

    Key insight:
    - The separator only aligns four transparent layers; the true signal is which merged cells were contributed by
      multiple layers, because those merged overlap points become the diagonal ray emitters.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    quadrants = _extract_quadrants(g)
    out, overlap_mask = _overlay_quadrants(quadrants)
    h, w = out.shape

    for r, c in np.argwhere(overlap_mask):
        color = int(out[r, c])
        assert color != BACKGROUND, "overlap points must remain foreground after overlay"
        for dr, dc in DIAGONALS:
            nr = int(r) + dr
            nc = int(c) + dc
            while 0 <= nr < h and 0 <= nc < w:
                if int(out[nr, nc]) != BACKGROUND:
                    break
                out[nr, nc] = color
                nr += dr
                nc += dc

    return out


def _ray_cells_from_source(source: tuple[int, int], q: int) -> tuple[tuple[int, int], ...]:
    r, c = source
    cells: list[tuple[int, int]] = []
    for dr, dc in DIAGONALS:
        nr = r + dr
        nc = c + dc
        while 0 <= nr < q and 0 <= nc < q:
            cells.append((nr, nc))
            nr += dr
            nc += dc
    return tuple(cells)


def _first_two_ray_cells(source: tuple[int, int], q: int) -> tuple[tuple[int, int], ...]:
    r, c = source
    cells: list[tuple[int, int]] = []
    for dr, dc in DIAGONALS:
        nr = r + dr
        nc = c + dc
        for _ in range(2):
            if not (0 <= nr < q and 0 <= nc < q):
                return ()
            cells.append((nr, nc))
            nr += dr
            nc += dc
    return tuple(cells)


@lru_cache(maxsize=None)
def _valid_source_layouts(q: int, count: int) -> tuple[tuple[tuple[tuple[int, int], ...], ...], tuple[tuple[tuple[int, int], ...], ...]]:
    candidates = [(r, c) for r in range(2, q - 2) for c in range(2, q - 2)]
    non_colliding: list[tuple[tuple[int, int], ...]] = []
    colliding: list[tuple[tuple[int, int], ...]] = []

    for combo in combinations(candidates, count):
        full_paths = {src: set(_ray_cells_from_source(src, q)) for src in combo}
        first_two = {src: set(_first_two_ray_cells(src, q)) for src in combo}
        if any(not first_two[src] for src in combo):
            continue

        valid = True
        for a, b in combinations(combo, 2):
            if a in full_paths[b] or b in full_paths[a]:
                valid = False
                break
            if a in first_two[b] or b in first_two[a]:
                valid = False
                break
        if not valid:
            continue

        seen: dict[tuple[int, int], tuple[int, int]] = {}
        collision = False
        for src in combo:
            for cell in full_paths[src]:
                other = seen.get(cell)
                if other is not None and other != src:
                    collision = True
                else:
                    seen[cell] = src

        if collision:
            colliding.append(combo)
        else:
            non_colliding.append(combo)

    return tuple(non_colliding), tuple(colliding)


def _choose_source_layout() -> tuple[int, tuple[tuple[int, int], ...], bool]:
    want_collision = random.random() < 0.45
    if want_collision:
        options = []
        for count in (2, 3):
            _, colliding = _valid_source_layouts(7, count)
            if colliding:
                options.append((7, colliding))
        q, layouts = random.choice(options)
        return q, random.choice(layouts), True

    q = random.choice(GEN_Q_OPTIONS)
    candidate_specs = [(q, 2)]
    if q == 7:
        candidate_specs.append((7, 2))
    random.shuffle(candidate_specs)
    for qv, count in candidate_specs:
        non_colliding, _ = _valid_source_layouts(qv, count)
        if non_colliding:
            return qv, random.choice(non_colliding), False
    non_colliding, _ = _valid_source_layouts(6, 2)
    return 6, random.choice(non_colliding), False


def _pick_scattered_cells(
    pool: list[tuple[int, int]],
    need: int,
    anchors: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    if need == 0:
        return []
    remaining = list(pool)
    chosen: list[tuple[int, int]] = []
    random.shuffle(remaining)

    while len(chosen) < need and remaining:
        scored = []
        base = list(anchors) + chosen
        for cell in remaining:
            if base:
                score = min(max(abs(cell[0] - r), abs(cell[1] - c)) for r, c in base)
            else:
                score = 99
            scored.append((score, random.random(), cell))
        scored.sort(reverse=True)
        best_score, _, best_cell = scored[0]
        if best_score < 1:
            return None
        chosen.append(best_cell)
        remaining.remove(best_cell)

    if len(chosen) != need:
        return None
    return chosen


def _assemble_grid(quadrants: list[np.ndarray]) -> np.ndarray:
    q = quadrants[0].shape[0]
    grid = np.full((2 * q + 1, 2 * q + 1), BACKGROUND, dtype=int)
    grid[q, :] = SEPARATOR
    grid[:, q] = SEPARATOR
    grid[:q, :q] = quadrants[0]
    grid[:q, q + 1 :] = quadrants[1]
    grid[q + 1 :, :q] = quadrants[2]
    grid[q + 1 :, q + 1 :] = quadrants[3]
    return grid


def _generate_attempt() -> np.ndarray | None:
    q, sources, has_collision_layout = _choose_source_layout()
    quadrant_colors = random.sample(list(QUADRANT_COLORS), 4)

    source_owners: dict[tuple[int, int], tuple[int, ...]] = {}
    for source in sources:
        owner_count = random.choice((2, 2, 3, 4))
        owners = tuple(sorted(random.sample(PRIORITY_ORDER, owner_count)))
        source_owners[source] = owners

    quadrant_cells: list[set[tuple[int, int]]] = [set() for _ in range(4)]
    for source, owners in source_owners.items():
        for owner in owners:
            quadrant_cells[owner].add(source)

    ray_cells = set()
    for source in sources:
        ray_cells.update(_ray_cells_from_source(source, q))
    reserved = set(sources) | ray_cells
    available = [(r, c) for r in range(q) for c in range(q) if (r, c) not in reserved]

    targets = []
    for idx in PRIORITY_ORDER:
        current = len(quadrant_cells[idx])
        lower = max(3, current + 1)
        upper = min(8, lower + 3)
        targets.append(random.randint(lower, upper))

    if sum(target - len(quadrant_cells[idx]) for idx, target in enumerate(targets)) > len(available):
        return None

    random.shuffle(available)
    for idx in random.sample(list(PRIORITY_ORDER), len(PRIORITY_ORDER)):
        need = targets[idx] - len(quadrant_cells[idx])
        anchors = set(sources) | quadrant_cells[idx]
        picked = _pick_scattered_cells(available, need, anchors)
        if picked is None:
            return None
        quadrant_cells[idx].update(picked)
        used = set(picked)
        available = [cell for cell in available if cell not in used]

    quadrants = [np.full((q, q), BACKGROUND, dtype=int) for _ in range(4)]
    for idx, cells in enumerate(quadrant_cells):
        for r, c in cells:
            quadrants[idx][r, c] = quadrant_colors[idx]

    overlay, overlap_mask = _overlay_quadrants(quadrants)
    overlap_positions = {tuple(map(int, pos)) for pos in np.argwhere(overlap_mask)}
    if overlap_positions != set(sources):
        return None

    for source in sources:
        for cell in _first_two_ray_cells(source, q):
            if int(overlay[cell]) != BACKGROUND:
                return None

    for quad in quadrants:
        colors = {int(color) for color in np.unique(quad) if int(color) != BACKGROUND}
        if len(colors) != 1:
            return None
        area = int(np.count_nonzero(quad))
        if not (3 <= area <= 8):
            return None

    grid = _assemble_grid(quadrants)
    out = main(grid)
    if out.shape != (q, q):
        return None
    if np.array_equal(out, overlay):
        return None
    if np.count_nonzero(out != overlay) < 8:
        return None
    if has_collision_layout and np.count_nonzero(out != overlay) < 10:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
