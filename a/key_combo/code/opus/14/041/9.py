from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]

GRID_SIDE_MIN = 15
GRID_SIDE_MAX = 20
DOMINO_COUNT_MIN = 3
DOMINO_COUNT_MAX = 5
SINGLE_COUNT_MIN = 3
SINGLE_COUNT_MAX = 5
GEN_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _domino_outline(coords: list[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = _bbox(coords)
    er0 = max(0, r0 - 1)
    ec0 = max(0, c0 - 1)
    er1 = min(h - 1, r1 + 1)
    ec1 = min(w - 1, c1 + 1)
    domino = set(coords)
    outline: set[tuple[int, int]] = set()
    for r in range(er0, er1 + 1):
        for c in range(ec0, ec1 + 1):
            if r in (er0, er1) or c in (ec0, ec1):
                if (r, c) not in domino:
                    outline.add((r, c))
    return outline


def _classify_red_objects(grid: np.ndarray) -> tuple[set[tuple[int, int]], list[list[tuple[int, int]]]]:
    singles: set[tuple[int, int]] = set()
    dominoes: list[list[tuple[int, int]]] = []
    for comp in _components_of_color(grid, RED):
        area = len(comp)
        if area == 1:
            singles.add(comp[0])
            continue
        assert area == 2, "red components must be single pixels or dominoes"
        (r0, c0), (r1, c1) = comp
        assert abs(r0 - r1) + abs(c0 - c1) == 1, "area-2 component must be a domino"
        dominoes.append(comp)
    return singles, dominoes


def _paint_domino_outlines(
    grid: np.ndarray, dominoes: list[list[tuple[int, int]]]
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    out = grid.copy()
    outline_cells: set[tuple[int, int]] = set()
    for domino in dominoes:
        for r, c in sorted(_domino_outline(domino, grid.shape)):
            if int(out[r, c]) == BLACK:
                out[r, c] = GREEN
                outline_cells.add((r, c))
    return out, outline_cells


def _spiral_path(seed: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r, c = seed
    max_delta = max(r, h - 1 - r, c, w - 1 - c)
    max_leg = 2 * (max_delta + 1)
    leg = 2
    direction_index = 0
    path: list[tuple[int, int]] = []
    while leg <= max_leg:
        dr, dc = DIRS4[direction_index]
        for _ in range(leg):
            r += dr
            c += dc
            if _in_bounds(shape, r, c):
                path.append((r, c))
        direction_index = (direction_index + 1) % 4
        if direction_index in (0, 2):
            leg += 2
    return path


def _simulate_with_sets(
    grid: np.ndarray,
    singles: set[tuple[int, int]],
    domino_cells: set[tuple[int, int]],
    outline_cells: set[tuple[int, int]],
) -> dict:
    seed_coords = np.argwhere(grid == BLUE)
    assert seed_coords.shape == (1, 2), "expected exactly one blue seed"
    seed = tuple(map(int, seed_coords[0]))
    out = grid.copy()
    r, c = seed
    h, w = grid.shape
    max_delta = max(r, h - 1 - r, c, w - 1 - c)
    max_leg = 2 * (max_delta + 1)
    leg = 2
    direction_index = 0
    painted: list[tuple[int, int]] = []
    stop_kind = None
    stop_coord = None
    step_index = None

    while leg <= max_leg:
        dr, dc = DIRS4[direction_index]
        for _ in range(leg):
            r += dr
            c += dc
            if not _in_bounds(grid.shape, r, c):
                continue
            coord = (r, c)
            if coord in outline_cells:
                stop_kind = "outline"
                stop_coord = coord
                step_index = len(painted)
                return {
                    "out": out,
                    "painted": painted,
                    "stop_kind": stop_kind,
                    "stop_coord": stop_coord,
                    "step_index": step_index,
                }
            if coord in domino_cells:
                stop_kind = "domino"
                stop_coord = coord
                step_index = len(painted)
                return {
                    "out": out,
                    "painted": painted,
                    "stop_kind": stop_kind,
                    "stop_coord": stop_coord,
                    "step_index": step_index,
                }
            assert int(out[r, c]) in (BLACK, BLUE, RED)
            assert coord in singles or int(out[r, c]) != RED
            out[r, c] = BLUE
            painted.append(coord)
        direction_index = (direction_index + 1) % 4
        if direction_index in (0, 2):
            leg += 2

    return {
        "out": out,
        "painted": painted,
        "stop_kind": None,
        "stop_coord": None,
        "step_index": None,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Outline only the red dominoes, then let a blue spiral stop at those generated walls while passing through lone red pixels.

    Input structure:
    - The grid is a black background containing one blue seed pixel and several red 4-connected objects.
    - Every red object is either a single isolated pixel or a two-cell horizontal/vertical domino.
    - The grid contains no pre-existing green pixels; any green in the output comes only from outlined dominoes.

    Rule:
    - Classify each red 4-connected object by area: size 1 stays unchanged, size 2 is a domino.
    - For every domino, expand its bounding box by one cell in each direction, clip to the grid, and paint the perimeter green wherever the current cell is black.
    - Starting from the blue seed, trace a spiral with directions up, right, down, left and leg lengths 2, 2, 4, 4, 6, 6, ...
    - Each in-bounds spiral step paints blue onto black cells and also onto red singleton cells, but the spiral stops before entering any green outline cell or any red domino cell.

    Key insight:
    - The red objects must first be split into singles versus dominoes, because only dominoes create the green obstacle walls that later determine where the spiral can no longer continue.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))).issubset({BLACK, BLUE, RED})
    singles, dominoes = _classify_red_objects(g)
    outlined, outline_cells = _paint_domino_outlines(g, dominoes)
    domino_cells = {cell for domino in dominoes for cell in domino}
    return _simulate_with_sets(outlined, singles, domino_cells, outline_cells)["out"]


def _chebyshev_far_enough(coords: list[tuple[int, int]], occupied_red: set[tuple[int, int]]) -> bool:
    for r, c in coords:
        for rr, cc in occupied_red:
            if max(abs(r - rr), abs(c - cc)) <= 1:
                return False
    return True


def _coords_far_from_seed(coords: list[tuple[int, int]], seed: tuple[int, int]) -> bool:
    sr, sc = seed
    return all(abs(r - sr) + abs(c - sc) >= 2 for r, c in coords)


def _domino_valid(
    coords: list[tuple[int, int]],
    shape: tuple[int, int],
    seed: tuple[int, int],
    occupied_red: set[tuple[int, int]],
    occupied_outline: set[tuple[int, int]],
    protected_blockers: set[tuple[int, int]],
) -> bool:
    if not _coords_far_from_seed(coords, seed):
        return False
    if not _chebyshev_far_enough(coords, occupied_red):
        return False
    if any(cell in occupied_outline for cell in coords):
        return False
    outline = _domino_outline(coords, shape)
    if any(cell == seed for cell in outline):
        return False
    if any(cell in occupied_red or cell in occupied_outline for cell in outline):
        return False
    if any(cell in protected_blockers for cell in outline | set(coords)):
        return False
    return True


def _single_valid(
    coord: tuple[int, int],
    seed: tuple[int, int],
    occupied_red: set[tuple[int, int]],
    occupied_outline: set[tuple[int, int]],
) -> bool:
    if coord == seed or coord in occupied_red or coord in occupied_outline:
        return False
    if not _coords_far_from_seed([coord], seed):
        return False
    return _chebyshev_far_enough([coord], occupied_red)


def _all_domino_candidates(
    shape: tuple[int, int],
    seed: tuple[int, int],
    occupied_red: set[tuple[int, int]],
    occupied_outline: set[tuple[int, int]],
    protected_blockers: set[tuple[int, int]],
    require_clipped: bool | None = None,
) -> list[tuple[list[tuple[int, int]], set[tuple[int, int]]]]:
    h, w = shape
    candidates: list[tuple[list[tuple[int, int]], set[tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            for coords in (((r, c), (r, c + 1)), ((r, c), (r + 1, c))):
                if not all(_in_bounds(shape, rr, cc) for rr, cc in coords):
                    continue
                coord_list = [(int(rr), int(cc)) for rr, cc in coords]
                if not _domino_valid(
                    coord_list,
                    shape,
                    seed,
                    occupied_red,
                    occupied_outline,
                    protected_blockers,
                ):
                    continue
                r0, c0, r1, c1 = _bbox(coord_list)
                clipped = r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1
                if require_clipped is not None and clipped != require_clipped:
                    continue
                candidates.append((coord_list, _domino_outline(coord_list, shape)))
    return candidates


def _all_single_candidates(
    shape: tuple[int, int],
    seed: tuple[int, int],
    occupied_red: set[tuple[int, int]],
    occupied_outline: set[tuple[int, int]],
    allowed: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    h, w = shape
    cells = allowed if allowed is not None else {(r, c) for r in range(h) for c in range(w)}
    return [
        (r, c)
        for r, c in cells
        if _single_valid((r, c), seed, occupied_red, occupied_outline)
    ]


def _choose_seed(rng: np.random.Generator, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    candidates = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            path = _spiral_path((r, c), shape)
            if len(path) >= 12:
                edge_dist = min(r, h - 1 - r, c, w - 1 - c)
                candidates.append(((r, c), edge_dist))
    near = [seed for seed, edge_dist in candidates if edge_dist <= 3]
    inner = [seed for seed, edge_dist in candidates if edge_dist >= 4]
    if inner and rng.random() < 0.5:
        return inner[int(rng.integers(0, len(inner)))]
    if near:
        return near[int(rng.integers(0, len(near)))]
    return candidates[int(rng.integers(0, len(candidates)))][0]


def _pick_blocking_domino(
    rng: np.random.Generator,
    shape: tuple[int, int],
    seed: tuple[int, int],
    path: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], set[tuple[int, int]], int] | None:
    occupied_red: set[tuple[int, int]] = set()
    occupied_outline: set[tuple[int, int]] = set()
    upper = min(len(path) - 2, 40)
    if upper < 7:
        return None
    stop_indices = list(range(6, upper + 1))
    rng.shuffle(stop_indices)
    for stop_index in stop_indices:
        stop = path[stop_index]
        protected = set(path[:stop_index])
        candidates = []
        for coords, outline in _all_domino_candidates(
            shape,
            seed,
            occupied_red,
            occupied_outline,
            protected,
            require_clipped=None,
        ):
            if stop not in outline:
                continue
            grid = np.zeros(shape, dtype=int)
            grid[seed] = BLUE
            for r, c in coords:
                grid[r, c] = RED
            no_outline = _simulate_with_sets(grid, set(), set(coords), set())
            if no_outline["stop_kind"] == "outline":
                continue
            if no_outline["step_index"] is not None and no_outline["step_index"] <= stop_index:
                continue
            candidates.append((coords, outline))
        if candidates:
            chosen = candidates[int(rng.integers(0, len(candidates)))]
            return chosen[0], chosen[1], stop_index
    return None


def _render_input(
    shape: tuple[int, int],
    seed: tuple[int, int],
    dominoes: list[list[tuple[int, int]]],
    singles: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    grid[seed] = BLUE
    for r, c in singles:
        grid[r, c] = RED
    for domino in dominoes:
        for r, c in domino:
            grid[r, c] = RED
    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if int(np.sum(grid == BLUE)) != 1:
        return False
    if set(map(int, np.unique(grid))) - {BLACK, BLUE, RED}:
        return False

    singles, dominoes = _classify_red_objects(grid)
    if not (DOMINO_COUNT_MIN <= len(dominoes) <= DOMINO_COUNT_MAX):
        return False
    if not (SINGLE_COUNT_MIN <= len(singles) <= SINGLE_COUNT_MAX):
        return False

    outlined, outline_cells = _paint_domino_outlines(grid, dominoes)
    domino_cells = {cell for domino in dominoes for cell in domino}
    with_outline = _simulate_with_sets(outlined, singles, domino_cells, outline_cells)
    no_outline = _simulate_with_sets(grid.copy(), singles, domino_cells, set())

    if with_outline["stop_kind"] != "outline":
        return False
    if with_outline["step_index"] is None:
        return False
    if no_outline["step_index"] is not None and no_outline["step_index"] <= with_outline["step_index"]:
        return False
    if np.array_equal(with_outline["out"], no_outline["out"]):
        return False

    seed = tuple(map(int, np.argwhere(grid == BLUE)[0]))
    for r, c in singles | domino_cells:
        if abs(r - seed[0]) + abs(c - seed[1]) < 2:
            return False

    for comp in _components_of_color(grid, RED):
        if len(comp) not in (1, 2):
            return False

    return np.array_equal(main(grid), with_outline["out"])


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    shape = (
        int(rng.integers(GRID_SIDE_MIN, GRID_SIDE_MAX + 1)),
        int(rng.integers(GRID_SIDE_MIN, GRID_SIDE_MAX + 1)),
    )
    seed = _choose_seed(rng, shape)
    path = _spiral_path(seed, shape)
    blocking = _pick_blocking_domino(rng, shape, seed, path)
    if blocking is None:
        return None
    blocker_domino, blocker_outline, stop_index = blocking

    domino_target = int(rng.integers(DOMINO_COUNT_MIN, DOMINO_COUNT_MAX + 1))
    single_target = int(rng.integers(SINGLE_COUNT_MIN, SINGLE_COUNT_MAX + 1))

    dominoes: list[list[tuple[int, int]]] = [blocker_domino]
    singles: set[tuple[int, int]] = set()
    occupied_red = set(blocker_domino)
    occupied_outline = set(blocker_outline)
    protected_blockers = set(path[: stop_index + 1])

    if rng.random() < 0.7:
        early_path = set(path[:stop_index])
        on_path_candidates = _all_single_candidates(shape, seed, occupied_red, occupied_outline, early_path)
        if on_path_candidates:
            count = 1 + int(rng.random() < 0.35 and len(on_path_candidates) >= 2)
            rng.shuffle(on_path_candidates)
            for cell in on_path_candidates:
                if len(singles) >= count:
                    break
                if _single_valid(cell, seed, occupied_red | singles, occupied_outline):
                    singles.add(cell)

    need_clipped_extra = rng.random() < 0.45
    while len(dominoes) < domino_target:
        require_clipped = True if need_clipped_extra else None
        candidates = _all_domino_candidates(
            shape,
            seed,
            occupied_red,
            occupied_outline,
            protected_blockers,
            require_clipped=require_clipped,
        )
        if not candidates and require_clipped is True:
            need_clipped_extra = False
            continue
        if not candidates:
            return None
        coords, outline = candidates[int(rng.integers(0, len(candidates)))]
        dominoes.append(coords)
        occupied_red.update(coords)
        occupied_outline.update(outline)
        need_clipped_extra = False

    while len(singles) < single_target:
        candidates = _all_single_candidates(shape, seed, occupied_red | singles, occupied_outline)
        if not candidates:
            return None
        cell = candidates[int(rng.integers(0, len(candidates)))]
        singles.add(cell)

    grid = _render_input(shape, seed, dominoes, singles)
    return grid if _validate_instance(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
