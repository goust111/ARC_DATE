from __future__ import annotations

import numpy as np

from arc_utils import Color, find_components, get_bg, in_bounds
import random


def _coord_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in coords}


def _is_contiguous_diagonal(coords: set[tuple[int, int]]) -> bool:
    if len(coords) < 2:
        return False

    if len({r - c for r, c in coords}) == 1:
        rows = sorted(r for r, _c in coords)
        cols = sorted(c for _r, c in coords)
    elif len({r + c for r, c in coords}) == 1:
        rows = sorted(r for r, _c in coords)
        cols = sorted(c for _r, c in coords)
    else:
        return False

    return (
        rows == list(range(rows[0], rows[-1] + 1))
        and cols == list(range(cols[0], cols[-1] + 1))
        and len(coords) == rows[-1] - rows[0] + 1
    )


def _ray_seed(g: np.ndarray, bg: int) -> tuple[int, set[tuple[int, int]]]:
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True)
    candidates: list[tuple[int, set[tuple[int, int]]]] = []
    for comp in comps:
        coords = _coord_set(comp["coords"])
        if _is_contiguous_diagonal(coords):
            candidates.append((int(comp["color"]), coords))

    assert len(candidates) == 1, "expected one diagonal ray seed"
    return candidates[0]


def _initial_state(
    shape: tuple[int, int], coords: set[tuple[int, int]]
) -> tuple[int, int, int, int]:
    if len({r - c for r, c in coords}) == 1:
        first = min(coords)
        last = max(coords)
        endpoints = ((first, (-1, -1)), (last, (1, 1)))
    else:
        assert len({r + c for r, c in coords}) == 1, "ray seed must be diagonal"
        top = min(coords)
        bottom = max(coords)
        endpoints = ((top, (-1, 1)), (bottom, (1, -1)))

    candidates = []
    for (r, c), (dr, dc) in endpoints:
        if in_bounds(shape, r + dr, c + dc):
            candidates.append((r, c, dr, dc))

    assert len(candidates) == 1, "ray seed must have exactly one in-bounds extension"
    return candidates[0]


def _mirror_orientations(g: np.ndarray, bg: int) -> dict[tuple[int, int], str]:
    orientations: dict[tuple[int, int], str] = {}
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[Color.RED])
    for comp in comps:
        coords = _coord_set(comp["coords"])
        rows = sorted({r for r, _c in coords})
        cols = sorted({c for _r, c in coords})

        if len(rows) == 1:
            assert cols == list(range(cols[0], cols[-1] + 1)), "red mirror must be contiguous"
            orientation = "horizontal"
        elif len(cols) == 1:
            assert rows == list(range(rows[0], rows[-1] + 1)), "red mirror must be contiguous"
            orientation = "vertical"
        else:
            raise AssertionError("red mirror must be a horizontal or vertical segment")

        for rc in coords:
            orientations[rc] = orientation

    return orientations


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the diagonal blue ray through the grid, reflecting it off red line-segment mirrors.

    Input structure
    - Black-background grid with one diagonal 8-connected blue ray seed.
    - Red foreground objects, when present, are straight horizontal or vertical mirror segments.
    - Other foreground bars are transparent to the ray and may be overwritten where the ray crosses them.

    Rule
    - Start from the ray seed endpoint whose outward diagonal extension remains inside the grid.
    - Paint the ray continuously in that diagonal direction until it leaves the grid.
    - If the next cell is a red mirror, do not paint it; reflect by flipping the row direction on a horizontal mirror or the column direction on a vertical mirror.

    Key insight
    - The non-red bars are decoys: only red bars reflect the diagonal path, while all other crossed cells become blue.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    ray_color, seed = _ray_seed(g, bg)
    r, c, dr, dc = _initial_state(g.shape, seed)
    mirrors = _mirror_orientations(g, bg)

    out = g.copy()
    seen: set[tuple[int, int, int, int]] = set()
    while True:
        state = (r, c, dr, dc)
        assert state not in seen, "ray path must not loop"
        seen.add(state)

        nr, nc = r + dr, c + dc
        if not in_bounds(g.shape, nr, nc):
            break

        if int(g[nr, nc]) == Color.RED:
            orientation = mirrors.get((nr, nc))
            assert orientation is not None, "red cell must belong to a mirror segment"
            if orientation == "horizontal":
                dr = -dr
            else:
                assert orientation == "vertical"
                dc = -dc
            continue

        out[nr, nc] = ray_color
        r, c = nr, nc

    return out

# --------------- generator (att1) ---------------

# Canvas and color layout.
GRID_HEIGHT = 16
GRID_WIDTH = 16
BACKGROUND_COLOR = 0

# Ray and seed sampling ranges.
REFLECTION_COUNT_RANGE = (0, 4)
SEED_LENGTH_RANGE = (3, 6)
TOTAL_PATH_LENGTH_RANGE = (15, 16)
NEW_CELLS_RANGE = (9, 13)
DIAGONAL_CHOICES = (-1, 1)
START_EDGES = ("top", "bottom", "left", "right")

# Bar structure ranges and orientations.
BAR_LENGTH_RANGE = (4, 14)
MIRROR_ORIENTATIONS = ["horizontal", "vertical"]
DECOY_COUNT_RANGE = (0, 3)
DECOY_PLACEMENT_ATTEMPTS = 30

# Rejection sampling budget.
MAX_GENERATION_ATTEMPTS = 300


def _inside(r: int, c: int) -> bool:
    return 0 <= r < GRID_HEIGHT and 0 <= c < GRID_WIDTH


def _random_start_state() -> tuple[int, int, int, int]:
    """Choose a boundary cell with a diagonal direction whose backward extension is OOB."""
    edge = random.choice(START_EDGES)
    if edge == "top":
        return 0, random.randint(0, GRID_WIDTH - 1), 1, random.choice(DIAGONAL_CHOICES)
    if edge == "bottom":
        return (
            GRID_HEIGHT - 1,
            random.randint(0, GRID_WIDTH - 1),
            -1,
            random.choice(DIAGONAL_CHOICES),
        )
    if edge == "left":
        return random.randint(0, GRID_HEIGHT - 1), 0, random.choice(DIAGONAL_CHOICES), 1
    return (
        random.randint(0, GRID_HEIGHT - 1),
        GRID_WIDTH - 1,
        random.choice(DIAGONAL_CHOICES),
        -1,
    )


def _build_seed_cells(
    r0: int,
    c0: int,
    dr: int,
    dc: int,
    seed_len: int,
) -> list[tuple[int, int]] | None:
    cells = []
    for i in range(seed_len):
        sr, sc = r0 + i * dr, c0 + i * dc
        if not _inside(sr, sc):
            return None
        cells.append((sr, sc))
    return cells


def _split_path_segments(new_cells_needed: int, n_reflections: int) -> list[int] | None:
    """Partition painted cells across straight ray segments before each reflection."""
    if n_reflections == 0:
        return [new_cells_needed]
    if new_cells_needed < n_reflections + 1:
        return None

    cuts = sorted(random.sample(range(1, new_cells_needed), n_reflections))
    segments = []
    prev = 0
    for cut in cuts:
        segments.append(cut - prev)
        prev = cut
    segments.append(new_cells_needed - prev)
    return segments


def _walk_segment(
    r: int,
    c: int,
    dr: int,
    dc: int,
    seg_len: int,
    occupied: set[tuple[int, int]],
    path_cells: list[tuple[int, int]],
) -> tuple[int, int] | None:
    """Append seg_len new ray cells and return the segment endpoint."""
    for _ in range(seg_len):
        nr, nc = r + dr, c + dc
        if not _inside(nr, nc) or (nr, nc) in occupied:
            return None
        path_cells.append((nr, nc))
        occupied.add((nr, nc))
        r, c = nr, nc
    return r, c


def _reflected_direction(orientation: str, dr: int, dc: int) -> tuple[int, int]:
    if orientation == "horizontal":
        return -dr, dc
    return dr, -dc


def _segment_start_bounds(anchor: int, length: int, limit: int) -> tuple[int, int] | None:
    start_min = max(0, anchor - length + 1)
    start_max = min(anchor, limit - length)
    if start_min > start_max:
        return None

    interior_min = max(start_min, 1)
    interior_max = min(start_max, limit - length - 1) if length < limit else start_max
    if interior_min <= interior_max:
        return interior_min, interior_max
    return start_min, start_max


def _bar_cells_through(
    orientation: str,
    r: int,
    c: int,
    length: int,
) -> list[tuple[int, int]] | None:
    if orientation == "horizontal":
        bounds = _segment_start_bounds(c, length, GRID_WIDTH)
        if bounds is None:
            return None
        start = random.randint(*bounds)
        return [(r, start + j) for j in range(length)]

    bounds = _segment_start_bounds(r, length, GRID_HEIGHT)
    if bounds is None:
        return None
    start = random.randint(*bounds)
    return [(start + j, c) for j in range(length)]


def _try_place_mirror(
    r: int,
    c: int,
    dr: int,
    dc: int,
    occupied: set[tuple[int, int]],
) -> tuple[str, list[tuple[int, int]], int, int] | None:
    """Try both mirror orientations at the next ray cell and return the reflected direction."""
    mr, mc = r + dr, c + dc
    if not _inside(mr, mc) or (mr, mc) in occupied:
        return None

    orientations = MIRROR_ORIENTATIONS.copy()
    random.shuffle(orientations)
    for orientation in orientations:
        new_dr, new_dc = _reflected_direction(orientation, dr, dc)
        next_r, next_c = r + new_dr, c + new_dc
        if not _inside(next_r, next_c) or (next_r, next_c) in occupied:
            continue

        bar_len = random.randint(*BAR_LENGTH_RANGE)
        bar_cells = _bar_cells_through(orientation, mr, mc, bar_len)
        if bar_cells is None:
            continue

        bar_set = set(bar_cells)
        if bar_set & occupied:
            continue

        occupied |= bar_set
        return orientation, bar_cells, new_dr, new_dc
    return None


def _random_decoy_cells(orientation: str, length: int) -> list[tuple[int, int]] | None:
    if orientation == "horizontal":
        row = random.randint(0, GRID_HEIGHT - 1)
        start = random.randint(0, max(0, GRID_WIDTH - length))
        start = max(1, min(start, GRID_WIDTH - length - 1))
        if start < 0 or start + length > GRID_WIDTH:
            return None
        return [(row, start + j) for j in range(length)]

    col = random.randint(0, GRID_WIDTH - 1)
    start = random.randint(0, max(0, GRID_HEIGHT - length))
    start = max(1, min(start, GRID_HEIGHT - length - 1))
    if start < 0 or start + length > GRID_HEIGHT:
        return None
    return [(start + j, col) for j in range(length)]


def _place_decoys(
    occupied: set[tuple[int, int]],
    path_cells: list[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    decoy_bars = []
    path_set = set(path_cells)
    for _ in range(random.randint(*DECOY_COUNT_RANGE)):
        orientation = random.choice(MIRROR_ORIENTATIONS)
        length = random.randint(*BAR_LENGTH_RANGE)

        for _ in range(DECOY_PLACEMENT_ATTEMPTS):
            cells = _random_decoy_cells(orientation, length)
            if cells is None:
                continue

            cell_set = set(cells)
            if cell_set & occupied:
                continue
            if len(cell_set & path_set) > 2:
                continue

            decoy_bars.append(cells)
            occupied |= cell_set
            break
    return decoy_bars


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _render_input(
    seed_cells: list[tuple[int, int]],
    mirror_bars: list[tuple[str, list[tuple[int, int]]]],
    decoy_bars: list[list[tuple[int, int]]],
) -> np.ndarray:
    grid = np.full((GRID_HEIGHT, GRID_WIDTH), BACKGROUND_COLOR, dtype=int)
    _paint_cells(grid, seed_cells, Color.BLUE)
    for _orientation, bar_cells in mirror_bars:
        _paint_cells(grid, bar_cells, Color.RED)
    for cells in decoy_bars:
        _paint_cells(grid, cells, Color.GREEN)
    return grid


def _matches_solver_oracle(
    grid: np.ndarray,
    seed_cells: list[tuple[int, int]],
    path_cells: list[tuple[int, int]],
) -> bool:
    try:
        output = main(grid)
    except Exception:
        return False

    expected = grid.copy()
    _paint_cells(expected, path_cells, Color.BLUE)
    if not np.array_equal(output, expected):
        return False

    new_count = len(path_cells)
    total_path = len(seed_cells) + new_count
    return (
        NEW_CELLS_RANGE[0] <= new_count <= NEW_CELLS_RANGE[1]
        and TOTAL_PATH_LENGTH_RANGE[0] <= total_path <= TOTAL_PATH_LENGTH_RANGE[1]
    )


def _gen_attempt() -> np.ndarray | None:
    n_reflections = random.randint(*REFLECTION_COUNT_RANGE)
    r0, c0, dr, dc = _random_start_state()
    br, bc = r0 - dr, c0 - dc
    if _inside(br, bc):
        return None

    seed_len = random.randint(*SEED_LENGTH_RANGE)
    seed_cells = _build_seed_cells(r0, c0, dr, dc, seed_len)
    if seed_cells is None:
        return None

    total_target = random.randint(*TOTAL_PATH_LENGTH_RANGE)
    new_cells_needed = total_target - seed_len
    segments = _split_path_segments(new_cells_needed, n_reflections)
    if segments is None:
        return None

    r, c = seed_cells[-1]
    cur_dr, cur_dc = dr, dc
    occupied = set(seed_cells)
    path_cells: list[tuple[int, int]] = []
    mirror_bars: list[tuple[str, list[tuple[int, int]]]] = []

    for seg_idx, seg_len in enumerate(segments):
        endpoint = _walk_segment(r, c, cur_dr, cur_dc, seg_len, occupied, path_cells)
        if endpoint is None:
            return None
        r, c = endpoint

        if seg_idx < n_reflections:
            placed_mirror = _try_place_mirror(r, c, cur_dr, cur_dc, occupied)
            if placed_mirror is None:
                return None
            orientation, bar_cells, cur_dr, cur_dc = placed_mirror
            mirror_bars.append((orientation, bar_cells))

    final_nr, final_nc = r + cur_dr, c + cur_dc
    if _inside(final_nr, final_nc):
        return None

    decoy_bars = _place_decoys(occupied, path_cells)
    grid = _render_input(seed_cells, mirror_bars, decoy_bars)
    if not _matches_solver_oracle(grid, seed_cells, path_cells):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
