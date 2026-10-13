from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
WALLPAPER_COLORS = (RED, YELLOW)

SMALL_MAX_AREA = 4
LARGE_MIN_AREA = 6

GRID_MIN = 11
GRID_MAX = 16
PERIOD_MIN = 2
PERIOD_MAX = 4

SMALL_COUNT_RANGE = (2, 4)
LARGE_COUNT_RANGE = (1, 3)
SMALL_AREA_RANGE = (1, 4)
LARGE_AREA_RANGE = (6, 12)
RETRY_LIMIT = 500


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _normalize_cells(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _bbox_shape(cells: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    return max_r + 1, max_c + 1


def _random_polyomino(area: int, max_h: int, max_w: int) -> tuple[tuple[int, int], ...]:
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            random.shuffle(frontier)
            placed = False
            for candidate in frontier:
                trial = _normalize_cells(cells | {candidate})
                h, w = _bbox_shape(trial)
                if h <= max_h and w <= max_w:
                    cells = set(trial)
                    placed = True
                    break
            if not placed:
                break
        if len(cells) == area:
            return _normalize_cells(cells)
    raise RuntimeError(f"failed to build a polyomino of area {area}")


def _black_components(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != BLACK or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == BLACK and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _consistent_tile_for_period(
    grid: np.ndarray, row_period: int, col_period: int
) -> dict[tuple[int, int], int] | None:
    tile: dict[tuple[int, int], int] = {}
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLACK:
                continue
            key = (r % row_period, c % col_period)
            if key in tile and tile[key] != color:
                return None
            tile[key] = color
    if len(tile) != row_period * col_period:
        return None
    return tile


def _infer_period_and_tile(
    grid: np.ndarray,
) -> tuple[int, int, dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, int, dict[tuple[int, int], int]]] = []
    for row_period in range(1, h + 1):
        for col_period in range(1, w + 1):
            tile = _consistent_tile_for_period(g, row_period, col_period)
            if tile is not None:
                candidates.append((row_period, col_period, tile))

    assert candidates, "no consistent period candidate"
    min_area = min(row_period * col_period for row_period, col_period, _ in candidates)
    best = [candidate for candidate in candidates if candidate[0] * candidate[1] == min_area]
    assert len(best) == 1, "period is not uniquely determined"
    return best[0]


def _reconstruct_wallpaper(
    shape: tuple[int, int], row_period: int, col_period: int, tile: dict[tuple[int, int], int]
) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            out[r, c] = int(tile[(r % row_period, c % col_period)])
    return out


def _expected_output(grid: np.ndarray) -> np.ndarray:
    row_period, col_period, tile = _infer_period_and_tile(grid)
    wallpaper = _reconstruct_wallpaper(grid.shape, row_period, col_period, tile)
    out = np.asarray(grid, dtype=int).copy()
    for comp in _black_components(out):
        area = int(comp.shape[0])
        assert area != 5, "area 5 is reserved to avoid threshold ambiguity"
        if area <= SMALL_MAX_AREA:
            out[comp[:, 0], comp[:, 1]] = wallpaper[comp[:, 0], comp[:, 1]]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Repair only the small black stains on a periodic two-color wallpaper.

    Input structure:
    - A rectangular wallpaper made only of red and yellow repeats with an unknown
      2D period, and some cells are replaced by black stain pixels.
    - Black pixels form 4-connected components of varying areas.
    - The visible non-black cells are sufficient to determine every wallpaper
      residue class uniquely.

    Rule:
    - Ignore black cells and infer the unique smallest row and column periods
      whose residue classes give every visible cell a consistent color.
    - Find every 4-connected black component and measure its area.
    - If a black component has area at most four, refill each of its cells with
      the wallpaper color for that residue class; if its area is at least six,
      keep it black.

    Key insight:
    - The result is the merge of two independent analyses: periodic wallpaper
      recovery determines what color belongs at each location, and component area
      determines which black regions should actually be restored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.size > 0
    assert set(map(int, np.unique(g))) <= {BLACK, RED, YELLOW}
    assert RED in g and YELLOW in g

    out = _expected_output(g)
    return out


def _tile_has_exact_period(tile: np.ndarray, row_period: int, col_period: int) -> bool:
    inferred_row, inferred_col, _ = _infer_period_and_tile(tile)
    return inferred_row == row_period and inferred_col == col_period


def _random_tile(row_period: int, col_period: int) -> np.ndarray:
    for _ in range(400):
        tile = np.random.choice(WALLPAPER_COLORS, size=(row_period, col_period))
        if len(set(map(int, tile.ravel()))) != 2:
            continue
        if _tile_has_exact_period(tile, row_period, col_period):
            return tile.astype(int)
    raise RuntimeError("failed to build a tile with the requested minimal period")


def _wallpaper_from_tile(tile: np.ndarray, rows: int, cols: int) -> np.ndarray:
    row_period, col_period = tile.shape
    out = np.zeros((rows, cols), dtype=int)
    for r in range(rows):
        for c in range(cols):
            out[r, c] = int(tile[r % row_period, c % col_period])
    return out


def _can_place_component(
    occupied: np.ndarray, component: tuple[tuple[int, int], ...], r0: int, c0: int
) -> bool:
    h, w = occupied.shape
    placed = {(r0 + r, c0 + c) for r, c in component}
    for r, c in placed:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if occupied[r, c]:
            return False
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in placed:
                continue
            if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                return False
    return True


def _place_component(
    occupied: np.ndarray, component: tuple[tuple[int, int], ...]
) -> np.ndarray | None:
    h, w = occupied.shape
    comp_h, comp_w = _bbox_shape(component)
    options = []
    for r0 in range(h - comp_h + 1):
        for c0 in range(w - comp_w + 1):
            if _can_place_component(occupied, component, r0, c0):
                options.append((r0, c0))
    if not options:
        return None
    r0, c0 = random.choice(options)
    coords = np.array([(r0 + r, c0 + c) for r, c in component], dtype=int)
    occupied[coords[:, 0], coords[:, 1]] = True
    return coords


def _residue_classes_visible(
    grid: np.ndarray, row_period: int, col_period: int
) -> bool:
    for rr in range(row_period):
        for cc in range(col_period):
            visible = False
            for r in range(rr, grid.shape[0], row_period):
                for c in range(cc, grid.shape[1], col_period):
                    if int(grid[r, c]) != BLACK:
                        visible = True
                        break
                if visible:
                    break
            if not visible:
                return False
    return True


def _build_case_output(
    wallpaper: np.ndarray, large_components: list[np.ndarray]
) -> np.ndarray:
    out = wallpaper.copy()
    for comp in large_components:
        out[comp[:, 0], comp[:, 1]] = BLACK
    return out


def _build_case_input(
    output_grid: np.ndarray, small_components: list[np.ndarray]
) -> np.ndarray:
    inp = output_grid.copy()
    for comp in small_components:
        inp[comp[:, 0], comp[:, 1]] = BLACK
    return inp


def _boundary_aware_areas() -> tuple[list[int], list[int]]:
    small_count = random.randint(*SMALL_COUNT_RANGE)
    large_count = random.randint(*LARGE_COUNT_RANGE)

    small_areas = [random.randint(*SMALL_AREA_RANGE) for _ in range(small_count)]
    small_areas[0] = random.choice([3, 4])
    if small_count > 1:
        small_areas[1] = random.choice([1, 2, 4])

    large_areas = [random.randint(*LARGE_AREA_RANGE) for _ in range(large_count)]
    large_areas[0] = random.choice([6, 7])
    return small_areas, large_areas


def _component_shape(area: int, *, large: bool) -> tuple[tuple[int, int], ...]:
    if area == 1:
        return ((0, 0),)
    if area == 2:
        return random.choice([((0, 0), (0, 1)), ((0, 0), (1, 0))])
    max_h = 6 if large else 4
    max_w = 6 if large else 4
    return _random_polyomino(area, max_h=max_h, max_w=max_w)


def _generate_case() -> tuple[np.ndarray, np.ndarray, tuple[int, int], np.ndarray]:
    rows = random.randint(GRID_MIN, GRID_MAX)
    cols = random.randint(GRID_MIN, GRID_MAX)
    row_period = random.randint(PERIOD_MIN, PERIOD_MAX)
    col_period = random.randint(PERIOD_MIN, PERIOD_MAX)
    tile = _random_tile(row_period, col_period)
    wallpaper = _wallpaper_from_tile(tile, rows, cols)

    small_areas, large_areas = _boundary_aware_areas()
    occupied = np.zeros((rows, cols), dtype=bool)
    small_components: list[np.ndarray] = []
    large_components: list[np.ndarray] = []

    sizes = [(area, False) for area in small_areas] + [(area, True) for area in large_areas]
    random.shuffle(sizes)
    for area, is_large in sizes:
        coords = _place_component(occupied, _component_shape(area, large=is_large))
        if coords is None:
            raise RuntimeError("placement failed")
        if is_large:
            large_components.append(coords)
        else:
            small_components.append(coords)

    output_grid = _build_case_output(wallpaper, large_components)
    input_grid = _build_case_input(output_grid, small_components)
    return input_grid, output_grid, (row_period, col_period), wallpaper


def _semantic_case_is_valid(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    true_period: tuple[int, int],
    wallpaper: np.ndarray,
) -> bool:
    if input_grid.shape != output_grid.shape:
        return False
    if max(input_grid.shape) > 30:
        return False
    if not _residue_classes_visible(input_grid, *true_period):
        return False

    try:
        inferred_row, inferred_col, tile = _infer_period_and_tile(input_grid)
    except AssertionError:
        return False
    if (inferred_row, inferred_col) != true_period:
        return False

    reconstructed = _reconstruct_wallpaper(input_grid.shape, inferred_row, inferred_col, tile)
    if not np.array_equal(reconstructed, wallpaper):
        return False

    components = _black_components(input_grid)
    areas = sorted(int(comp.shape[0]) for comp in components)
    if sum(area <= SMALL_MAX_AREA for area in areas) < 2:
        return False
    if sum(area >= LARGE_MIN_AREA for area in areas) < 1:
        return False
    if any(area == 5 for area in areas):
        return False

    try:
        solved = main(input_grid)
    except AssertionError:
        return False
    if not np.array_equal(solved, output_grid):
        return False

    if np.array_equal(input_grid, output_grid):
        return False
    if np.array_equal(wallpaper, output_grid):
        return False

    fill_all = wallpaper.copy()
    keep_all = input_grid.copy()
    if np.array_equal(fill_all, output_grid):
        return False
    if np.array_equal(keep_all, output_grid):
        return False

    if Counter(map(int, input_grid.ravel()))[BLACK] == 0:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        try:
            input_grid, output_grid, period, wallpaper = _generate_case()
        except RuntimeError:
            continue
        if _semantic_case_is_valid(input_grid, output_grid, period, wallpaper):
            return input_grid
    raise RuntimeError("failed to generate a valid stained-wallpaper input")
