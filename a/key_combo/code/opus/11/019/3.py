from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

QUAD_ORDER = ("NW", "NE", "SE", "SW")
GRID_SIZES = (15, 17)
STATUS_OPTIONS = ((True, True), (True, False), (False, True))
_GEN_STATUS_COUNTER = 0


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    shifted = coords - np.array([r0, c0], dtype=int)
    mask[shifted[:, 0], shifted[:, 1]] = True
    return mask


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_coords(pts)
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _rotate_cw(mask: np.ndarray, turns: int) -> np.ndarray:
    return _normalize_mask(np.rot90(mask, -turns))


def _quadrant_bounds(shape: tuple[int, int], quad: str) -> tuple[int, int, int, int]:
    h, w = shape
    mr, mc = h // 2, w // 2
    if quad == "NW":
        return 0, mr - 1, 0, mc - 1
    if quad == "NE":
        return 0, mr - 1, mc + 1, w - 1
    if quad == "SW":
        return mr + 1, h - 1, 0, mc - 1
    if quad == "SE":
        return mr + 1, h - 1, mc + 1, w - 1
    raise ValueError(quad)


def _quadrant_of_coords(coords: np.ndarray, shape: tuple[int, int]) -> str:
    center = coords.mean(axis=0)
    mr, mc = shape[0] // 2, shape[1] // 2
    vertical = "N" if center[0] < mr else "S"
    horizontal = "W" if center[1] < mc else "E"
    return vertical + horizontal


def _rotation_turns(src_quad: str, dst_quad: str) -> int:
    return (QUAD_ORDER.index(dst_quad) - QUAD_ORDER.index(src_quad)) % 4


def _obstacle_forbidden(grid: np.ndarray) -> np.ndarray:
    forbidden = np.zeros_like(grid, dtype=bool)
    for r, c in np.argwhere(grid == ORANGE):
        for rr in range(int(r) - 1, int(r) + 2):
            for cc in range(int(c) - 1, int(c) + 2):
                if _in_bounds(grid.shape, rr, cc):
                    forbidden[rr, cc] = True
    return forbidden


def _placeholder_frontier(grid: np.ndarray, square_coords: np.ndarray, forbidden: np.ndarray) -> set[tuple[int, int]]:
    target: set[tuple[int, int]] = set()
    for r, c in square_coords:
        for nr, nc in _neighbors4(int(r), int(c)):
            if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] == BLACK and not forbidden[nr, nc]:
                target.add((nr, nc))
    return target


def _is_reachable(grid: np.ndarray, start: tuple[int, int], square_coords: np.ndarray) -> bool:
    forbidden = _obstacle_forbidden(grid)
    targets = _placeholder_frontier(grid, square_coords, forbidden)
    if not targets:
        return False

    q = deque()
    seen = set()
    for nr, nc in _neighbors4(*start):
        if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] == BLACK and not forbidden[nr, nc]:
            seen.add((nr, nc))
            q.append((nr, nc))
    while q:
        cell = q.popleft()
        if cell in targets:
            return True
        for nr, nc in _neighbors4(*cell):
            nxt = (nr, nc)
            if (
                _in_bounds(grid.shape, nr, nc)
                and nxt not in seen
                and grid[nr, nc] == BLACK
                and not forbidden[nr, nc]
            ):
                seen.add(nxt)
                q.append(nxt)
    return False


def _shortest_path(grid: np.ndarray, start: tuple[int, int], square_coords: np.ndarray) -> list[tuple[int, int]] | None:
    forbidden = _obstacle_forbidden(grid)
    targets = _placeholder_frontier(grid, square_coords, forbidden)
    if not targets:
        return None

    q = deque()
    parent: dict[tuple[int, int], tuple[int, int] | None] = {}
    for nr, nc in _neighbors4(*start):
        if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] == BLACK and not forbidden[nr, nc]:
            cell = (nr, nc)
            parent[cell] = None
            q.append(cell)

    end = None
    while q:
        cell = q.popleft()
        if cell in targets:
            end = cell
            break
        for nr, nc in _neighbors4(*cell):
            nxt = (nr, nc)
            if (
                _in_bounds(grid.shape, nr, nc)
                and nxt not in parent
                and grid[nr, nc] == BLACK
                and not forbidden[nr, nc]
            ):
                parent[nxt] = cell
                q.append(nxt)

    if end is None:
        return None

    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each reachable placeholder square with the source shape rotated by the quadrant relation, leaving unreachable squares unchanged.

    Input structure:
    - A black grid is conceptually split into four quadrants by its center row and center column.
    - One quadrant contains a small 4-connected green source shape and a nearby red endpoint pixel.
    - Two other quadrants contain solid yellow 5x5 placeholder squares, while several orange single-pixel obstacles sit on black background.

    Rule:
    - Extract the unique green source component, normalize it, and build its four right-angle rotations.
    - For each yellow 5x5 square, test whether a 4-neighbor path exists from the red endpoint to a black cell on that square's boundary ring; the path may use only black cells and must avoid every orange obstacle together with its full 8-neighborhood.
    - If a square is unreachable, keep it yellow. If it is reachable, determine that square's quadrant relative to the source quadrant, choose the matching rotation (0/90/180/270 clockwise), clear the 5x5 square to black, and paste the rotated green shape aligned to the square's top-left corner.

    Key insight:
    - The output is controlled by two independent inferences at once: reachability through obstacle-expanded forbidden zones decides which squares activate, and quadrant rotation decides which source variant each activated square receives.
    """
    g = np.asarray(grid, dtype=int)

    green_components = _components(g == GREEN)
    assert len(green_components) == 1
    source_coords = green_components[0]
    source_mask = _mask_from_coords(source_coords)

    yellow_components = []
    for coords in _components(g == YELLOW):
        r0, c0, r1, c1 = _bbox_coords(coords)
        if len(coords) == 25 and (r1 - r0 + 1, c1 - c0 + 1) == (5, 5):
            yellow_components.append(coords)
    assert len(yellow_components) == 2

    red_pixels = np.argwhere(g == RED)
    assert len(red_pixels) == 1
    start = tuple(map(int, red_pixels[0]))

    source_quad = _quadrant_of_coords(source_coords, g.shape)
    out = g.copy()
    for square_coords in yellow_components:
        if not _is_reachable(g, start, square_coords):
            continue
        square_quad = _quadrant_of_coords(square_coords, g.shape)
        turns = _rotation_turns(source_quad, square_quad)
        variant = _rotate_cw(source_mask, turns)
        assert variant.shape[0] <= 5 and variant.shape[1] <= 5
        r0, c0, r1, c1 = _bbox_coords(square_coords)
        out[r0 : r1 + 1, c0 : c1 + 1] = BLACK
        rr, cc = np.argwhere(variant).T
        out[r0 + rr, c0 + cc] = GREEN
    return out


def _random_pattern() -> np.ndarray:
    for _ in range(400):
        h = random.choice((3, 4))
        w = random.choice((3, 4))
        target = random.randint(4, 7)
        seed = (random.randrange(h), random.randrange(w))
        cells = {seed}
        frontier = {p for p in _neighbors4(*seed) if _in_bounds((h, w), *p)}
        while len(cells) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nxt in _neighbors4(*cell):
                if _in_bounds((h, w), *nxt) and nxt not in cells:
                    frontier.add(nxt)
        if len(cells) != target:
            continue
        coords = np.array(sorted(cells), dtype=int)
        mask = _mask_from_coords(coords)
        rotations = {_rotate_cw(mask, k).tobytes() for k in range(4)}
        if len(rotations) != 4:
            continue
        if mask.shape[0] > 4 or mask.shape[1] > 4:
            continue
        return mask
    raise RuntimeError("failed to generate an asymmetric source pattern")


def _top_left_from_anchor(shape: tuple[int, int], quad: str, item_shape: tuple[int, int], anchor: str) -> tuple[int, int]:
    qr0, qr1, qc0, qc1 = _quadrant_bounds(shape, quad)
    h = qr1 - qr0 + 1
    w = qc1 - qc0 + 1
    ih, iw = item_shape
    max_roff = h - ih
    max_coff = w - iw
    assert max_roff >= 0 and max_coff >= 0
    jitter_r = random.randint(0, min(2, max_roff))
    jitter_c = random.randint(0, min(2, max_coff))

    if anchor == "outer":
        if quad == "NW":
            return qr0 + jitter_r, qc0 + jitter_c
        if quad == "NE":
            return qr0 + jitter_r, qc1 - iw + 1 - jitter_c
        if quad == "SW":
            return qr1 - ih + 1 - jitter_r, qc0 + jitter_c
        return qr1 - ih + 1 - jitter_r, qc1 - iw + 1 - jitter_c

    jitter_r = 0
    jitter_c = 0
    if quad == "NW":
        return qr1 - ih + 1 - jitter_r, qc1 - iw + 1 - jitter_c
    if quad == "NE":
        return qr1 - ih + 1 - jitter_r, qc0 + jitter_c
    if quad == "SW":
        return qr0 + jitter_r, qc1 - iw + 1 - jitter_c
    return qr0 + jitter_r, qc0 + jitter_c


def _endpoint_for_quad(grid: np.ndarray, quad: str) -> tuple[int, int]:
    qr0, qr1, qc0, qc1 = _quadrant_bounds(grid.shape, quad)
    cr, cc = (qr0 + qr1) // 2, (qc0 + qc1) // 2
    candidates = [(r, c) for r in range(qr0, qr1 + 1) for c in range(qc0, qc1 + 1)]
    candidates.sort(key=lambda rc: (abs(rc[0] - cr) + abs(rc[1] - cc), rc))
    for r, c in candidates:
        if grid[r, c] != BLACK:
            continue
        exits = 0
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] == BLACK:
                exits += 1
        if exits >= 2:
            return r, c
    raise RuntimeError("no endpoint location")


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rr, cc = np.argwhere(mask).T
    grid[top + rr, left + cc] = color


def _square_coords(top: int, left: int, size: int = 5) -> np.ndarray:
    return np.array([(top + r, left + c) for r in range(size) for c in range(size)], dtype=int)


def _rotate_coord(coord: tuple[int, int], shape: tuple[int, int], turns: int) -> tuple[int, int]:
    h, w = shape
    r, c = coord
    for _ in range(turns % 4):
        r, c = w - 1 - c, r
        h, w = w, h
    return int(r), int(c)


def _rotate_coords(coords: np.ndarray, shape: tuple[int, int], turns: int) -> np.ndarray:
    return np.array([_rotate_coord((int(r), int(c)), shape, turns) for r, c in coords], dtype=int)


def _status_with_obstacles(
    base_grid: np.ndarray,
    start: tuple[int, int],
    placeholder_coords: list[np.ndarray],
    obstacles: list[tuple[int, int]],
) -> tuple[bool, bool]:
    g = base_grid.copy()
    for r, c in obstacles:
        g[r, c] = ORANGE
    return tuple(_is_reachable(g, start, coords) for coords in placeholder_coords)  # type: ignore[return-value]


def _protected_cells(base_grid: np.ndarray) -> np.ndarray:
    protected = np.zeros_like(base_grid, dtype=bool)
    for r, c in np.argwhere(base_grid != BLACK):
        for rr in range(int(r) - 1, int(r) + 2):
            for cc in range(int(c) - 1, int(c) + 2):
                if _in_bounds(base_grid.shape, rr, cc):
                    protected[rr, cc] = True
    return protected


def _candidate_obstacles(
    base_grid: np.ndarray,
    start: tuple[int, int],
    placeholder_coords: list[np.ndarray],
) -> tuple[list[tuple[int, int]], list[list[tuple[int, int]]]]:
    h, w = base_grid.shape
    mr, mc = h // 2, w // 2
    protected = _protected_cells(base_grid)
    path_cells: list[list[tuple[int, int]]] = []
    path_halo: set[tuple[int, int]] = set()

    for coords in placeholder_coords:
        path = _shortest_path(base_grid, start, coords)
        if path is None or len(path) < 2:
            raise RuntimeError("base layout has no path")
        path_cells.append(path)
        for r, c in path:
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    if _in_bounds(base_grid.shape, rr, cc) and base_grid[rr, cc] == BLACK and not protected[rr, cc]:
                        path_halo.add((rr, cc))

    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if base_grid[r, c] != BLACK or protected[r, c]:
                continue
            if abs(r - mr) <= 3 or abs(c - mc) <= 3 or (r, c) in path_halo:
                candidates.append((r, c))
    random.shuffle(candidates)
    return candidates, path_cells


def _find_obstacles(
    base_grid: np.ndarray,
    start: tuple[int, int],
    placeholder_coords: list[np.ndarray],
    desired: tuple[bool, bool],
) -> list[tuple[int, int]] | None:
    if desired[0] != desired[1]:
        block_index = 0 if not desired[0] else 1
        preset = _preset_blockers(base_grid, placeholder_coords[block_index])
        if preset and _status_with_obstacles(base_grid, start, placeholder_coords, preset) == desired:
            return preset

    candidates, paths = _candidate_obstacles(base_grid, start, placeholder_coords)
    path_pools: list[list[tuple[int, int]]] = []
    for path in paths:
        pool: set[tuple[int, int]] = set()
        for r, c in path:
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    if (rr, cc) in candidates:
                        pool.add((rr, cc))
        path_pools.append(list(pool))

    target_count = random.randint(3, 5) if desired == (True, True) else random.randint(4, 5)
    core_candidates = candidates
    if desired[0] != desired[1]:
        block_index = 0 if not desired[0] else 1
        r0, c0, r1, c1 = _bbox_coords(placeholder_coords[block_index])
        local = [
            cell
            for cell in candidates
            if r0 - 2 <= cell[0] <= r1 + 2 and c0 - 2 <= cell[1] <= c1 + 2
        ]
        core_candidates = local + [
            c for c in path_pools[block_index] if c not in set(local)
        ] + [
            c for c in candidates if c not in set(local) and c not in set(path_pools[block_index])
        ]

    min_core = 0 if desired == (True, True) else 1
    max_core = min(4, len(core_candidates))
    for core_size in range(min_core, max_core + 1):
        for _ in range(240):
            if core_size == 0:
                core: list[tuple[int, int]] = []
            else:
                if len(core_candidates) < core_size:
                    continue
                core = random.sample(core_candidates, core_size)
            if _status_with_obstacles(base_grid, start, placeholder_coords, core) != desired:
                continue

            obstacles = list(core)
            filler_pool = [c for c in candidates if c not in obstacles]
            random.shuffle(filler_pool)
            for cand in filler_pool:
                trial = obstacles + [cand]
                if _status_with_obstacles(base_grid, start, placeholder_coords, trial) == desired:
                    obstacles.append(cand)
                if len(obstacles) == target_count:
                    return obstacles
            if 3 <= len(obstacles) <= 5 and _status_with_obstacles(base_grid, start, placeholder_coords, obstacles) == desired:
                return obstacles
    return None


def _preset_blockers(base_grid: np.ndarray, square_coords: np.ndarray) -> list[tuple[int, int]] | None:
    quad = _quadrant_of_coords(square_coords, base_grid.shape)
    r0, c0, r1, c1 = _bbox_coords(square_coords)
    if quad == "NW":
        cells = [(r0 + 1, c1 + 2), (r0 + 3, c1 + 2), (r1 + 2, c0 + 1), (r1 + 2, c0 + 3), (r0 - 2, c0 - 2)]
    elif quad == "NE":
        cells = [(r0 + 1, c0 - 2), (r0 + 3, c0 - 2), (r1 + 2, c0 + 1), (r1 + 2, c0 + 3), (r0 - 2, c1 + 2)]
    elif quad == "SW":
        cells = [(r0 + 1, c1 + 2), (r0 + 3, c1 + 2), (r0 - 2, c0 + 1), (r0 - 2, c0 + 3), (r1 + 2, c0 - 2)]
    else:
        cells = [(r0 + 1, c0 - 2), (r0 + 3, c0 - 2), (r0 - 2, c0 + 1), (r0 - 2, c0 + 3), (r1 + 2, c1 + 2)]

    unique = []
    for cell in cells:
        if cell not in unique:
            unique.append(cell)
    if len(unique) < 4:
        return None
    if any(not _in_bounds(base_grid.shape, *cell) or base_grid[cell] != BLACK for cell in unique):
        return None
    return unique


def _template_obstacles(base_grid: np.ndarray, desired: tuple[bool, bool]) -> list[tuple[int, int]] | None:
    if base_grid.shape != (15, 15) or desired == (True, True):
        return None
    templates = {
        (True, False): [(9, 0), (6, 3), (6, 6), (8, 13), (9, 13)],
        (False, True): [(0, 9), (3, 6), (6, 6), (13, 8), (13, 9)],
    }
    obstacles = templates[desired]
    if any(base_grid[cell] != BLACK for cell in obstacles):
        return None
    return obstacles


def _build_base_layout(desired: tuple[bool, bool]) -> tuple[np.ndarray, tuple[int, int], list[np.ndarray], tuple[bool, bool]]:
    n = 15 if desired != (True, True) else random.choice(GRID_SIZES)
    grid = np.zeros((n, n), dtype=int)

    source_quad = "NW"
    empty_quad = "SE"
    placeholder_quads = ["NE", "SW"]

    source_mask = _random_pattern()
    src_top, src_left = _top_left_from_anchor(grid.shape, source_quad, source_mask.shape, "outer")
    _paint_mask(grid, source_mask, src_top, src_left, GREEN)

    start = _endpoint_for_quad(grid, source_quad)
    grid[start] = RED

    placeholder_coords: list[np.ndarray] = []
    for quad in placeholder_quads:
        top, left = _top_left_from_anchor(grid.shape, quad, (5, 5), "inner")
        grid[top : top + 5, left : left + 5] = YELLOW
        placeholder_coords.append(_square_coords(top, left))

    if _status_with_obstacles(grid, start, placeholder_coords, []) != (True, True):
        raise RuntimeError("base layout is not fully reachable")
    return grid, start, placeholder_coords, desired


def generate_input() -> np.ndarray:
    global _GEN_STATUS_COUNTER
    desired = STATUS_OPTIONS[_GEN_STATUS_COUNTER % len(STATUS_OPTIONS)]
    _GEN_STATUS_COUNTER += 1
    for _ in range(500):
        try:
            base_grid, start, placeholder_coords, desired = _build_base_layout(desired)
            obstacles = _template_obstacles(base_grid, desired)
            if obstacles is None:
                obstacles = _find_obstacles(base_grid, start, placeholder_coords, desired)
            if obstacles is None:
                continue
            grid = base_grid.copy()
            for r, c in obstacles:
                grid[r, c] = ORANGE

            if _status_with_obstacles(base_grid, start, placeholder_coords, obstacles) != desired:
                continue
            turns = random.randint(0, 3)
            if turns:
                grid = np.rot90(grid, -turns)
            out = main(grid)
            if np.array_equal(out, grid):
                continue
            return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
