import random
from collections import deque
from itertools import combinations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
MARKER_COLORS = (3, 4, 5, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _marker_coords(grid: np.ndarray) -> list[tuple[int, int]]:
    coords = [
        (int(r), int(c))
        for r, c in np.argwhere((grid != BLACK) & (grid != RED))
    ]
    coords.sort()
    return coords


def _trace_rays(
    grid: np.ndarray, markers: list[tuple[int, int]] | None = None
) -> np.ndarray:
    """Draw all marker rays using only red cells and other markers as obstacles."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    active_markers = _marker_coords(g) if markers is None else sorted(markers)
    marker_set = set(active_markers)

    out = g.copy()
    for r, c in active_markers:
        color = int(g[r, c])
        for dr, dc in DIRS4:
            rr = r + dr
            cc = c + dc
            if not _in_bounds((h, w), rr, cc):
                continue
            if int(g[rr, cc]) == RED or (rr, cc) in marker_set:
                continue

            while True:
                if int(out[rr, cc]) == BLACK:
                    out[rr, cc] = color

                nr = rr + dr
                nc = cc + dc
                if not _in_bounds((h, w), nr, nc):
                    break
                if int(g[nr, nc]) == RED or (nr, nc) in marker_set:
                    break
                rr, cc = nr, nc

    return out


def _exterior_black_mask(grid: np.ndarray) -> np.ndarray:
    """Return the black cells reachable from the boundary through 4-neighbor steps."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def push(r: int, c: int) -> None:
        if int(g[r, c]) != BLACK or exterior[r, c]:
            return
        exterior[r, c] = True
        q.append((r, c))

    for r in range(h):
        push(r, 0)
        push(r, w - 1)
    for c in range(w):
        push(0, c)
        push(h - 1, c)

    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            rr = r + dr
            cc = c + dc
            if _in_bounds((h, w), rr, cc):
                push(rr, cc)

    return exterior


def _enclosed_black_mask(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    return (g == BLACK) & ~_exterior_black_mask(g)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend marker rays, then fill every background region enclosed by those rays and the red walls.

    Input structure:
    - The grid has a black background with scattered red wall pixels and 2-3 colored single-pixel markers.
    - Marker colors are non-black and non-red, and each marker blocks other rays just like a wall endpoint.
    - The red walls alone do not seal any background region, but after all rays are drawn they can help form a closed barrier.

    Rule:
    - From each marker, cast rays up, down, left, and right starting at the adjacent cell.
    - A ray keeps painting cells in the marker's color until its next step would hit a red wall, another marker, or leave the grid; the obstacle cell itself is not painted.
    - Compute every ray against the original walls and marker positions, then treat all non-black cells (red walls, markers, and rays) as barriers.
    - Flood-fill from the boundary through black cells only, and recolor every remaining unreachable black cell blue.

    Key insight:
    - The target blue regions are not determined by the red walls alone; they emerge only after all marker rays are simulated simultaneously and merged into the wall network.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    with_rays = _trace_rays(g)
    enclosed = _enclosed_black_mask(with_rays)

    out = with_rays.copy()
    out[enclosed] = BLUE
    return out


def _rotate_grid(grid: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(grid, k=k).copy()


def _bbox_of_nonblack(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid != BLACK)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _outside_expanded_box(
    r: int, c: int, box: tuple[int, int, int, int], margin: int
) -> bool:
    r0, c0, r1, c1 = box
    return not (r0 - margin <= r <= r1 + margin and c0 - margin <= c <= c1 + margin)


def _isolated_from_nonblack(grid: np.ndarray, r: int, c: int) -> bool:
    if int(grid[r, c]) != BLACK:
        return False
    h, w = grid.shape
    for dr, dc in DIRS8:
        rr = r + dr
        cc = c + dc
        if _in_bounds((h, w), rr, cc) and int(grid[rr, cc]) != BLACK:
            return False
    return True


def _add_segment_if_clear(
    grid: np.ndarray, cells: list[tuple[int, int]], color: int
) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if not _in_bounds((h, w), r, c) or int(grid[r, c]) != BLACK:
            return False
        for dr, dc in DIRS8:
            rr = r + dr
            cc = c + dc
            if _in_bounds((h, w), rr, cc) and int(grid[rr, cc]) != BLACK:
                return False
    for r, c in cells:
        grid[r, c] = color
    return True


def _candidate_segments(
    grid: np.ndarray, box: tuple[int, int, int, int]
) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    candidates: list[list[tuple[int, int]]] = []
    for length in range(2, 5):
        for r in range(1, h - 1):
            for c in range(1, w - length):
                cells = [(r, c + k) for k in range(length)]
                if all(_outside_expanded_box(rr, cc, box, 2) for rr, cc in cells):
                    candidates.append(cells)
        for r in range(1, h - length):
            for c in range(1, w - 1):
                cells = [(r + k, c) for k in range(length)]
                if all(_outside_expanded_box(rr, cc, box, 2) for rr, cc in cells):
                    candidates.append(cells)
    random.shuffle(candidates)
    return candidates


def _add_distractor_segments(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    target_count = random.randint(0, 2)
    if target_count == 0:
        return

    added = 0
    for cells in _candidate_segments(grid, box):
        color = RED
        test = grid.copy()
        if not _add_segment_if_clear(test, cells, color):
            continue
        if np.any(_enclosed_black_mask(test)):
            continue
        grid[:] = test
        added += 1
        if added >= target_count:
            return


def _try_add_third_marker(
    grid: np.ndarray, box: tuple[int, int, int, int]
) -> np.ndarray | None:
    used_colors = {int(grid[r, c]) for r, c in _marker_coords(grid)}
    free_colors = [color for color in MARKER_COLORS if color not in used_colors]
    if not free_colors:
        return None

    h, w = grid.shape
    candidates = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if _outside_expanded_box(r, c, box, 2)
        and _isolated_from_nonblack(grid, r, c)
    ]
    random.shuffle(candidates)

    for r, c in candidates[:80]:
        test = grid.copy()
        test[r, c] = random.choice(free_colors)
        if _validate_generated_input(test):
            return test
    return None


def _build_base_grid() -> np.ndarray:
    h = random.randint(12, 18)
    w = random.randint(12, 18)
    grid = np.zeros((h, w), dtype=int)

    interior_h = random.randint(3, min(6, h - 6))
    interior_w = random.randint(3, min(7, w - 6))
    r_top = random.randint(2, h - interior_h - 4)
    c_left = random.randint(2, w - interior_w - 4)
    r_bottom = r_top + interior_h + 1
    c_right = c_left + interior_w + 1

    grid[r_bottom, c_left : c_right + 1] = RED
    color_left, color_right = random.sample(MARKER_COLORS, 2)
    grid[r_top, c_left] = color_left
    grid[r_top, c_right] = color_right

    box = (r_top, c_left, r_bottom, c_right)
    _add_distractor_segments(grid, box)

    if random.random() < 0.55:
        third = _try_add_third_marker(grid, box)
        if third is not None:
            grid = third

    return _rotate_grid(grid, random.randrange(4))


def _single_marker_grids(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    markers = _marker_coords(g)
    out: list[np.ndarray] = []
    for keep in markers:
        solo = g.copy()
        for marker in markers:
            if marker != keep:
                solo[marker] = BLACK
        out.append(solo)
    return out


def _pair_has_enclosure(grid: np.ndarray) -> bool:
    markers = _marker_coords(grid)
    for pair in combinations(markers, 2):
        if np.any(_enclosed_black_mask(_trace_rays(grid, list(pair)))):
            return True
    return False


def _markers_only_has_enclosure(grid: np.ndarray) -> bool:
    stripped = np.where(np.asarray(grid, dtype=int) == RED, BLACK, grid)
    return bool(np.any(_enclosed_black_mask(_trace_rays(stripped))))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if (
                        _in_bounds((h, w), nr, nc)
                        and mask[nr, nc]
                        and not seen[nr, nc]
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _validate_generated_input(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or min(g.shape) < 1 or max(g.shape) > 30:
        return False

    markers = _marker_coords(g)
    if len(markers) not in (2, 3):
        return False

    marker_colors = [int(g[r, c]) for r, c in markers]
    if len(set(marker_colors)) != len(marker_colors):
        return False
    if any(color in (BLACK, RED, BLUE) for color in marker_colors):
        return False

    red_cells = {(int(r), int(c)) for r, c in np.argwhere(g == RED)}
    if not red_cells:
        return False

    for i, (r, c) in enumerate(markers):
        for rr, cc in markers[i + 1 :]:
            if max(abs(r - rr), abs(c - cc)) <= 1:
                return False
        for dr, dc in DIRS8:
            nr = r + dr
            nc = c + dc
            if (nr, nc) in red_cells:
                return False

    if np.any(_enclosed_black_mask(g)):
        return False

    if _markers_only_has_enclosure(g):
        return False

    for solo in _single_marker_grids(g):
        if np.any(_enclosed_black_mask(_trace_rays(solo))):
            return False

    if not _pair_has_enclosure(g):
        return False

    with_rays = _trace_rays(g)
    enclosed = _enclosed_black_mask(with_rays)
    if not np.any(enclosed):
        return False

    enclosed_components = _components(enclosed)
    if max(len(comp) for comp in enclosed_components) < 4:
        return False

    if np.count_nonzero(with_rays != g) == 0:
        return False

    expected = with_rays.copy()
    expected[enclosed] = BLUE
    if not np.array_equal(main(g), expected):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_base_grid()
        if _validate_generated_input(grid):
            return grid
    raise RuntimeError("failed to generate a valid ray-enclosure ARC input")
