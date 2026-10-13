from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox_coords, components_mask


def _plus_cells(shape: tuple[int, int], row: int, col: int) -> list[tuple[int, int]]:
    height, width = map(int, shape)
    cells: list[tuple[int, int]] = []
    for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
        r = int(row) + dr
        c = int(col) + dc
        if 0 <= r < height and 0 <= c < width:
            cells.append((r, c))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each yellow-marked plus motif by recoloring its grey cells red.

    Input structure
    - The grid contains black empty cells plus grey and yellow foreground cells.
    - Each 8-connected yellow object is a subset of one clipped five-cell plus
      made from yellow and grey cells.

    Rule
    - For every 8-connected yellow object, find the unique plus center whose
      in-bounds center/up/down/left/right cells contain the yellow object and no
      black cells.
    - Recolor the grey cells of that plus red; keep yellow and black cells fixed.

    Key insight
    - The yellow pixels are partial cross markers, and the surrounding grey cells
      in the same cross are the missing cells to highlight.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height > 0 and width > 0, "expected a nonempty grid"

    empty = Color.BLACK
    marker = Color.YELLOW
    grey = Color.GREY
    red = Color.RED
    assert set(map(int, np.unique(g))) <= {empty, marker, grey}, "expected only black, yellow, and grey input colors"

    out = g.copy()
    yellow_objects = components_mask(g == marker, connectivity=8)
    assert yellow_objects, "expected at least one yellow marker object"

    for coords in yellow_objects:
        component = {tuple(map(int, coord)) for coord in coords}
        r0, c0, r1, c1 = bbox_coords(coords)
        candidates: list[list[tuple[int, int]]] = []
        for row in range(max(0, r0 - 1), min(height, r1 + 2)):
            for col in range(max(0, c0 - 1), min(width, c1 + 2)):
                cells = _plus_cells(g.shape, row, col)
                cell_set = set(cells)
                if not component <= cell_set:
                    continue
                values = [int(g[r, c]) for r, c in cells]
                if all(value in {marker, grey} for value in values) and grey in values:
                    candidates.append(cells)

        assert len(candidates) == 1, "expected each yellow object to imply one plus motif"
        for r, c in candidates[0]:
            if g[r, c] == grey:
                out[r, c] = red

    return out

# --------------- generator (o5x) ---------------

# Canvas and retry bounds; ranges use numpy's upper-exclusive integers.
GRID_HEIGHT_RANGE = (14, 20)
GRID_WIDTH_RANGE = (15, 20)
MOTIF_COUNT_RANGE = (4, 8)
GEN_RETRY_LIMIT = 300

# Motif geometry and yellow-subset constraints.
MIN_CENTER_DISTANCE = 6
EDGE_CENTER_PROBABILITY = 0.25
YELLOW_COUNT_RANGE = (2, 4)
YELLOW_CONNECTIVITY = 8

# Background noise and post-render density guard.
GREY_PROBABILITY_RANGE = (0.45, 0.50)
GREY_DENSITY_RANGE = (0.40, 0.55)

# ARC colors used by the generator.
GEN_BLACK = Color.BLACK
GEN_RED = Color.RED
GEN_YELLOW = Color.YELLOW
GEN_GREY = Color.GREY

Shape = tuple[int, int]
Cell = tuple[int, int]
CellList = list[Cell]
Motif = tuple[CellList, CellList]
MotifChoice = tuple[CellList, CellList, CellList]
MotifPlan = tuple[list[Motif], CellList, list[CellList]]
Rng = np.random.Generator


def _gen_is_edge(row: int, col: int, height: int, width: int) -> bool:
    return row in {0, height - 1} or col in {0, width - 1}


def _gen_is_yellow_connected(shape: Shape, cells: CellList) -> bool:
    mask = np.zeros(shape, dtype=bool)
    rows, cols = zip(*cells)
    mask[rows, cols] = True
    return len(components_mask(mask, connectivity=YELLOW_CONNECTIVITY)) == 1


def _gen_pattern_blockers(shape: Shape, center: Cell, yellow_cells: CellList, rng: Rng) -> CellList | None:
    height, width = shape
    yellow_set = set(yellow_cells)
    true_set = set(_plus_cells(shape, center[0], center[1]))
    blockers: list[tuple[int, int]] = []

    for row in range(height):
        for col in range(width):
            if (row, col) == center:
                continue
            cells = _plus_cells(shape, row, col)
            if not yellow_set <= set(cells):
                continue
            options = [cell for cell in cells if cell not in true_set]
            if not options:
                return None
            blockers.append(options[int(rng.integers(len(options)))])
    return blockers


def _gen_choose_motif(shape: Shape, center: Cell, rng: Rng) -> MotifChoice | None:
    cells = _plus_cells(shape, center[0], center[1])
    choices: list[MotifChoice] = []
    min_yellow, max_yellow = YELLOW_COUNT_RANGE

    for bits in range(1, 1 << len(cells)):
        yellow_cells = [cells[i] for i in range(len(cells)) if bits & (1 << i)]
        yellow_count = len(yellow_cells)
        if not min_yellow <= yellow_count <= max_yellow or yellow_count == len(cells):
            continue
        if not _gen_is_yellow_connected(shape, yellow_cells):
            continue
        blockers = _gen_pattern_blockers(shape, center, yellow_cells, rng)
        if blockers is None:
            continue
        yellow_set = set(yellow_cells)
        grey_cells = [cell for cell in cells if cell not in yellow_set]
        choices.append((yellow_cells, grey_cells, blockers))

    if not choices:
        return None
    return choices[int(rng.integers(len(choices)))]


def _gen_extend_centers(candidates: CellList, chosen: CellList, count: int, start: int) -> CellList | None:
    if len(chosen) == count:
        return chosen

    remaining = count - len(chosen)
    for idx in range(start, len(candidates)):
        row, col = candidates[idx]
        if any(abs(row - r) + abs(col - c) < MIN_CENTER_DISTANCE for r, c in chosen):
            continue
        if len(candidates) - idx < remaining:
            return None
        result = _gen_extend_centers(candidates, chosen + [(row, col)], count, idx + 1)
        if result is not None:
            return result
    return None


def _gen_pick_centers(height: int, width: int, count: int, rng: Rng) -> CellList | None:
    use_edge = bool(rng.random() < EDGE_CENTER_PROBABILITY)
    interiors = [
        (row, col)
        for row in range(1, height - 1)
        for col in range(1, width - 1)
    ]
    edges = [
        (row, col)
        for row in range(height)
        for col in range(width)
        if _gen_is_edge(row, col, height, width)
    ]
    rng.shuffle(interiors)
    rng.shuffle(edges)

    if not use_edge:
        return _gen_extend_centers(interiors, [], count, 0)

    for edge in edges:
        remaining = [
            point
            for point in interiors
            if abs(point[0] - edge[0]) + abs(point[1] - edge[1]) >= MIN_CENTER_DISTANCE
        ]
        result = _gen_extend_centers(remaining, [edge], count, 0)
        if result is not None:
            return result
    return None


def _gen_build_motif_plan(shape: Shape, centers: CellList, rng: Rng) -> MotifPlan | None:
    """Select motif colors and blocker pixels for the already-chosen centers."""
    motifs: list[Motif] = []
    all_blockers: CellList = []
    grey_cells_by_motif: list[CellList] = []
    for center in centers:
        motif = _gen_choose_motif(shape, center, rng)
        if motif is None:
            return None
        yellow_cells, grey_cells, blockers = motif
        motifs.append((yellow_cells, grey_cells))
        all_blockers.extend(blockers)
        grey_cells_by_motif.append(grey_cells)
    return motifs, all_blockers, grey_cells_by_motif


def _gen_render_grid(shape: Shape, motifs: list[Motif], blockers: CellList, rng: Rng) -> np.ndarray:
    """Build background noise, clear blockers, then paint motif cells."""
    grey_probability = float(rng.uniform(*GREY_PROBABILITY_RANGE))
    grid = np.where(rng.random(shape) < grey_probability, GEN_GREY, GEN_BLACK).astype(int)

    if blockers:
        rows, cols = zip(*blockers)
        grid[rows, cols] = GEN_BLACK

    for yellow_cells, grey_cells in motifs:
        for cells, color in ((grey_cells, GEN_GREY), (yellow_cells, GEN_YELLOW)):
            rows, cols = zip(*cells)
            grid[rows, cols] = color
    return grid


def _gen_expected_output(grid: np.ndarray, grey_cells_by_motif: list[CellList]) -> np.ndarray:
    out = grid.copy()
    for grey_cells in grey_cells_by_motif:
        rows, cols = zip(*grey_cells)
        out[rows, cols] = GEN_RED
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    height = int(rng.integers(*GRID_HEIGHT_RANGE))
    width = int(rng.integers(*GRID_WIDTH_RANGE))
    count = int(rng.integers(*MOTIF_COUNT_RANGE))
    shape = (height, width)

    centers = _gen_pick_centers(height, width, count, rng)
    if centers is None:
        return None

    motif_plan = _gen_build_motif_plan(shape, centers, rng)
    if motif_plan is None:
        return None
    motifs, all_blockers, grey_cells_by_motif = motif_plan

    grid = _gen_render_grid(shape, motifs, all_blockers, rng)
    grey_density = float(np.mean(grid == GEN_GREY))
    if not GREY_DENSITY_RANGE[0] <= grey_density <= GREY_DENSITY_RANGE[1]:
        return None

    expected = _gen_expected_output(grid, grey_cells_by_motif)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 14754a24 input")
