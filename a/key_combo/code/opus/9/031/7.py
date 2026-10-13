from __future__ import annotations

import itertools
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

ARM_OFFSETS = (
    (-3, 0),
    (-2, 0),
    (2, 0),
    (3, 0),
    (0, -3),
    (0, -2),
    (0, 2),
    (0, 3),
)
GAP_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAGONALS = ((-1, 1), (1, 1), (1, -1), (-1, -1))
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GRID_MIN = 14
GRID_MAX = 21
CROSS_COUNT_CHOICES = (2, 3)
GEN_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _is_cross_center(grid: np.ndarray, r: int, c: int) -> bool:
    if int(grid[r, c]) != BG:
        return False
    for dr, dc in ARM_OFFSETS:
        rr, cc = r + dr, c + dc
        if not _in_bounds(grid.shape, rr, cc) or int(grid[rr, cc]) != BLUE:
            return False
    for dr, dc in GAP_OFFSETS:
        rr, cc = r + dr, c + dc
        if not _in_bounds(grid.shape, rr, cc) or int(grid[rr, cc]) != BG:
            return False
    return True


def _scan_centers_and_directions(grid: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = grid.shape
    found: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(3, h - 3):
        for c in range(3, w - 3):
            if not _is_cross_center(grid, r, c):
                continue
            directions = [(dr, dc) for dr, dc in DIAGONALS if int(grid[r + dr, c + dc]) == RED]
            assert len(directions) == 1, "each hidden center must have exactly one diagonal red direction marker"
            found.append(((r, c), directions[0]))
    return found


def _render_from_centers(
    grid: np.ndarray,
    center_dirs: list[tuple[tuple[int, int], tuple[int, int]]],
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    occupancy = np.array(grid, dtype=int, copy=True)

    for (r, c), _direction in center_dirs:
        out[r, c] = YELLOW
        occupancy[r, c] = YELLOW

    for (r, c), (dr, dc) in center_dirs:
        rr, cc = r + dr, c + dc
        while _in_bounds(grid.shape, rr, cc):
            if int(occupancy[rr, cc]) == BG:
                out[rr, cc] = GREEN
                occupancy[rr, cc] = GREEN
            rr += dr
            cc += dc

    out[grid == RED] = BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect hidden four-bar centers, turn them into ray anchors, and extend diagonal rays to the grid edge.

    Input structure:
    - The grid is mostly black background.
    - Blue pixels form isolated two-cell horizontal or vertical bars; some quartets of bars imply a missing cross center with one background gap on each cardinal side.
    - Every true hidden center has exactly one red diagonal neighbor that indicates a diagonal ray direction.

    Rule:
    - Scan background pixels and detect every hidden center whose left/right arms are blue horizontal dominoes at offsets 2-3 and whose up/down arms are blue vertical dominoes at offsets 2-3, with the four cardinal gap cells still black.
    - Recolor each detected center yellow.
    - For each yellow center, read the unique red diagonal neighbor, then extend a ray from the center in that same diagonal direction until leaving the grid; recolor only black cells on that path to green and leave non-black cells unchanged so the ray visually passes through them.
    - Remove all red direction markers at the end; keep every blue bar unchanged.

    Key insight:
    - The blue bars determine where the anchor is, while the red pixel determines which diagonal to use; neither clue alone fixes the final ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"
    assert set(map(int, np.unique(g))).issubset({BG, BLUE, RED}), "input uses only black, blue, and red"

    center_dirs = _scan_centers_and_directions(g)
    assert center_dirs, "expected at least one valid hidden center"
    return _render_from_centers(g, center_dirs)


def _cross_bar_cells(center: tuple[int, int], arm: str) -> tuple[tuple[int, int], tuple[int, int]]:
    r, c = center
    if arm == "U":
        return ((r - 3, c), (r - 2, c))
    if arm == "D":
        return ((r + 2, c), (r + 3, c))
    if arm == "L":
        return ((r, c - 3), (r, c - 2))
    if arm == "R":
        return ((r, c + 2), (r, c + 3))
    raise ValueError("unknown arm")


def _ray_cells(shape: tuple[int, int], center: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r, c = center
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    rr, cc = r + dr, c + dc
    while 0 <= rr < h and 0 <= cc < w:
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in NEIGHBORS4]


def _blue_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(grid == BLUE):
        cell = (int(r), int(c))
        if cell in seen:
            continue
        seen.add(cell)
        queue = deque([cell])
        component = [cell]
        while queue:
            rr, cc = queue.popleft()
            for nr, nc in _neighbors4(rr, cc):
                if not _in_bounds((h, w), nr, nc):
                    continue
                nxt = (nr, nc)
                if nxt in seen or int(grid[nr, nc]) != BLUE:
                    continue
                seen.add(nxt)
                queue.append(nxt)
                component.append(nxt)
        components.append(component)
    return components


def _all_blue_components_are_dominoes(grid: np.ndarray) -> bool:
    for component in _blue_components(grid):
        if len(component) != 2:
            return False
        (r0, c0), (r1, c1) = sorted(component)
        if not ((r0 == r1 and abs(c0 - c1) == 1) or (c0 == c1 and abs(r0 - r1) == 1)):
            return False
    return True


def _bar_fits(grid: np.ndarray, cells: tuple[tuple[int, int], tuple[int, int]]) -> bool:
    h, w = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BG:
            return False
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cell_set:
                continue
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != BG:
                return False
    return True


def _place_bar(grid: np.ndarray, cells: tuple[tuple[int, int], tuple[int, int]]) -> None:
    for r, c in cells:
        grid[r, c] = BLUE


def _cross_bbox(center: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = center
    return r - 3, c - 3, r + 3, c + 3


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _choose_centers(rng: np.random.Generator, h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(3, h - 3) for c in range(3, w - 3)]
    rng.shuffle(candidates)
    centers: list[tuple[int, int]] = []
    for center in candidates:
        bbox = _cross_bbox(center)
        if any(_boxes_overlap(bbox, _cross_bbox(other), margin=1) for other in centers):
            continue
        centers.append(center)
        if len(centers) == count:
            return centers
    return None


def _choose_directions(
    shape: tuple[int, int], centers: list[tuple[int, int]], rng: np.random.Generator
) -> list[tuple[int, int]] | None:
    count = len(centers)
    direction_sets = list(itertools.permutations(DIAGONALS, count))
    rng.shuffle(direction_sets)
    for directions in direction_sets:
        ray_sets = [set(_ray_cells(shape, center, direction)) for center, direction in zip(centers, directions, strict=True)]
        if any(len(ray_set) < 3 for ray_set in ray_sets):
            continue
        bad = False
        for i, center in enumerate(centers):
            for j, other in enumerate(centers):
                if i == j:
                    continue
                if other in ray_sets[i]:
                    bad = True
                    break
                if (other[0] + directions[j][0], other[1] + directions[j][1]) in ray_sets[i]:
                    bad = True
                    break
            if bad:
                break
        if bad:
            continue
        if any(ray_sets[i] & ray_sets[j] for i in range(count) for j in range(i + 1, count)):
            continue
        return list(directions)
    return None


def _try_place_penetration_bar(
    grid: np.ndarray,
    center: tuple[int, int],
    direction: tuple[int, int],
    forbidden: set[tuple[int, int]],
    rng: np.random.Generator,
) -> bool:
    ray = _ray_cells(grid.shape, center, direction)[2:-1]
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for cell in ray:
        r, c = cell
        options = [
            ((r, c), (r, c + 1)),
            ((r, c - 1), (r, c)),
            ((r, c), (r + 1, c)),
            ((r - 1, c), (r, c)),
        ]
        rng.shuffle(options)
        for bar in options:
            if set(bar) & forbidden:
                continue
            if _bar_fits(grid, bar):
                candidates.append(bar)
    if not candidates:
        return False
    bar = candidates[int(rng.integers(0, len(candidates)))]
    _place_bar(grid, bar)
    return True


def _random_bar_candidates(grid: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = grid.shape
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(h):
        for c in range(w - 1):
            cells = ((r, c), (r, c + 1))
            if _bar_fits(grid, cells):
                candidates.append(cells)
    for r in range(h - 1):
        for c in range(w):
            cells = ((r, c), (r + 1, c))
            if _bar_fits(grid, cells):
                candidates.append(cells)
    return candidates


def _planned_output(
    grid: np.ndarray,
    centers: list[tuple[int, int]],
    directions: list[tuple[int, int]],
) -> np.ndarray:
    center_dirs = list(zip(centers, directions, strict=True))
    return _render_from_centers(grid, center_dirs)


def _validate_instance(
    grid: np.ndarray,
    centers: list[tuple[int, int]],
    directions: list[tuple[int, int]],
    expect_penetration: bool,
) -> bool:
    try:
        center_dirs = _scan_centers_and_directions(grid)
    except AssertionError:
        return False
    if sorted(center_dirs) != sorted(zip(centers, directions, strict=True)):
        return False
    if not _all_blue_components_are_dominoes(grid):
        return False

    expected = _planned_output(grid, centers, directions)
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    if np.count_nonzero(actual == YELLOW) != len(centers):
        return False
    if np.any(actual == RED):
        return False

    if expect_penetration:
        hit = False
        for center, direction in zip(centers, directions, strict=True):
            for r, c in _ray_cells(grid.shape, center, direction):
                if int(grid[r, c]) == BLUE:
                    hit = True
                    if int(actual[r, c]) != BLUE:
                        return False
        if not hit:
            return False
    return True


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_MIN, GRID_MAX))
    w = int(rng.integers(GRID_MIN, GRID_MAX))
    count = int(rng.choice(CROSS_COUNT_CHOICES))

    centers = _choose_centers(rng, h, w, count)
    if centers is None:
        return None
    directions = _choose_directions((h, w), centers, rng)
    if directions is None:
        return None

    grid = np.zeros((h, w), dtype=int)

    for center in centers:
        for arm in "UDLR":
            _place_bar(grid, _cross_bar_cells(center, arm))

    for center, direction in zip(centers, directions, strict=True):
        rr, cc = center[0] + direction[0], center[1] + direction[1]
        if int(grid[rr, cc]) != BG:
            return None
        grid[rr, cc] = RED

    forbidden = set(centers)
    for center, direction in zip(centers, directions, strict=True):
        forbidden.update(_ray_cells(grid.shape, center, direction)[:2])

    penetration_flags = [False] * count
    if count == 2:
        desired_true = {int(rng.integers(0, count))}
    else:
        desired_true = set(rng.choice(count, size=2, replace=False).tolist())
    if count >= 2 and len(desired_true) < count:
        clean_idx = next(i for i in range(count) if i not in desired_true)
    else:
        clean_idx = None

    for idx, (center, direction) in enumerate(zip(centers, directions, strict=True)):
        want_penetration = idx in desired_true
        if want_penetration:
            placed = _try_place_penetration_bar(grid, center, direction, forbidden, rng)
            if not placed:
                return None
            penetration_flags[idx] = True
        elif clean_idx is not None and idx == clean_idx:
            penetration_flags[idx] = False

    partial_count = int(rng.integers(1, 4))
    partial_centers = [(r, c) for r in range(3, h - 3) for c in range(3, w - 3)]
    rng.shuffle(partial_centers)
    placed_partials = 0
    for center in partial_centers:
        if center in centers:
            continue
        if any(_boxes_overlap(_cross_bbox(center), _cross_bbox(real_center), margin=0) for real_center in centers):
            continue
        arms = list("UDLR")
        rng.shuffle(arms)
        arm_count = int(rng.integers(2, 4))
        chosen = arms[:arm_count]
        bars = [_cross_bar_cells(center, arm) for arm in chosen]
        if not all(_bar_fits(grid, bar) for bar in bars):
            continue
        gap_ok = all(int(grid[center[0] + dr, center[1] + dc]) == BG for dr, dc in GAP_OFFSETS)
        diag_ok = all(int(grid[center[0] + dr, center[1] + dc]) == BG for dr, dc in DIAGONALS)
        if not (gap_ok and diag_ok):
            continue
        for bar in bars:
            _place_bar(grid, bar)
        placed_partials += 1
        if placed_partials == partial_count:
            break
    if placed_partials == 0:
        return None

    extra_bars_target = int(rng.integers(1, 6))
    for _ in range(extra_bars_target):
        candidates = _random_bar_candidates(grid)
        if not candidates:
            break
        bar = candidates[int(rng.integers(0, len(candidates)))]
        _place_bar(grid, bar)

    if not _validate_instance(grid, centers, directions, expect_penetration=any(penetration_flags)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-cross ray puzzle instance")
