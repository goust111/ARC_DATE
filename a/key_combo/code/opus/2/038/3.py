from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
TEAL = 8

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 500


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def find_blue_components(grid: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[tuple[tuple[int, int], ...]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != BLUE:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == BLUE:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords.sort()
            components.append(tuple(coords))

    return components


def path_cells(red: tuple[int, int], teal: tuple[int, int]) -> set[tuple[int, int]]:
    rr, rc = red
    tr, tc = teal
    cells: set[tuple[int, int]] = set()

    step_c = 1 if tc > rc else -1
    for c in range(rc, tc + step_c, step_c):
        cells.add((rr, c))

    step_r = 1 if tr > rr else -1
    for r in range(rr, tr + step_r, step_r):
        cells.add((r, tc))

    cells.discard(red)
    cells.discard(teal)
    return cells


def distance_map(shape: tuple[int, int], start: tuple[int, int], blocked: np.ndarray) -> np.ndarray:
    h, w = shape
    dist = np.full((h, w), -1, dtype=int)
    sr, sc = start
    if blocked[sr, sc]:
        return dist

    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if in_bounds(shape, nr, nc) and not blocked[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def red_side_mask(shape: tuple[int, int], red: tuple[int, int], blocked: np.ndarray) -> np.ndarray:
    dist = distance_map(shape, red, blocked)
    rows = np.arange(shape[0])[:, None]
    cols = np.arange(shape[1])[None, :]
    manhattan = np.abs(rows - red[0]) + np.abs(cols - red[1])
    return (~blocked) & (dist == manhattan)


def representative(coords: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return min(coords)


def grown_neighbors(coords: set[tuple[int, int]], grid: np.ndarray) -> set[tuple[int, int]]:
    add: set[tuple[int, int]] = set()
    for r, c in coords:
        for nr, nc in neighbors4(r, c):
            if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == BLACK:
                add.add((nr, nc))
    return add


def erode_object(coords: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    keep: set[tuple[int, int]] = set()
    for r, c in coords:
        if all(in_bounds(shape, nr, nc) and (nr, nc) in coords for nr, nc in neighbors4(r, c)):
            keep.add((r, c))
    return keep


def scene_data(grid: np.ndarray):
    reds = np.argwhere(grid == RED)
    teals = np.argwhere(grid == TEAL)
    assert len(reds) == 1 and len(teals) == 1, "expected exactly one red and one teal endpoint"

    red = tuple(int(v) for v in reds[0])
    teal = tuple(int(v) for v in teals[0])
    assert red[0] != teal[0] and red[1] != teal[1], "endpoints must differ in both row and column"

    yellow_cells = path_cells(red, teal)
    blocked = np.zeros_like(grid, dtype=bool)
    for r, c in yellow_cells:
        blocked[r, c] = True
    red_side = red_side_mask(grid.shape, red, blocked)
    return red, teal, yellow_cells, blocked, red_side


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw an L-shaped divider between the endpoints, then grow the blue objects on the red side and shrink the objects on the teal side.

    Input structure:
    - A black background contains exactly one red endpoint pixel and one teal endpoint pixel.
    - Several blue 4-connected foreground objects of size 4 to 7 are scattered away from the endpoints.
    - The endpoints are in different rows and columns, so the hidden connector is an L shape.

    Rule:
    - Draw a yellow path from the red pixel horizontally to the teal column, then vertically to the teal pixel; keep the red and teal endpoint colors unchanged.
    - For each blue object, use its lexicographically smallest pixel as the representative. If a shortest 4-neighbor path from that representative to the red endpoint can avoid the yellow divider, the object is on the red side; otherwise it is on the teal side.
    - Red-side objects grow by painting every black 4-neighbor of any object pixel blue, simultaneously across all red-side objects. Teal-side objects shrink by removing every blue pixel that lacks blue neighbors on at least one cardinal side.

    Key insight:
    The L path is both a visible connector and a spatial partition: the same blue shape either dilates or erodes depending on which side of the divider its representative pixel reaches the red endpoint from most directly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    red, teal, yellow_cells, _blocked, red_side = scene_data(g)
    components = find_blue_components(g)

    out = g.copy()
    for r, c in yellow_cells:
        out[r, c] = YELLOW

    growth_targets: set[tuple[int, int]] = set()
    shrink_remove: set[tuple[int, int]] = set()

    for comp in components:
        rep = representative(comp)
        comp_set = set(comp)
        if red_side[rep]:
            growth_targets |= grown_neighbors(comp_set, g)
        else:
            keep = erode_object(comp_set, g.shape)
            shrink_remove |= comp_set - keep

    for r, c in shrink_remove:
        out[r, c] = BLACK
    for r, c in growth_targets:
        if int(out[r, c]) == BLACK:
            out[r, c] = BLUE

    out[red] = RED
    out[teal] = TEAL
    return out


BASE_SHAPES = [
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (3, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (2, 3)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (3, 1), (1, 3)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (3, 1)),
]


def normalize_shape(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = bbox_coords(list(coords))
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def transform_shape(coords: tuple[tuple[int, int], ...], rotation: int, mirror: bool) -> tuple[tuple[int, int], ...]:
    transformed: list[tuple[int, int]] = []
    for r, c in coords:
        rr, cc = r, -c if mirror else c
        for _ in range(rotation % 4):
            rr, cc = cc, -rr
        transformed.append((rr, cc))
    return normalize_shape(tuple(transformed))


def unique_transforms(coords: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    seen = set()
    out = []
    for rotation in range(4):
        for mirror in (False, True):
            shape = transform_shape(coords, rotation, mirror)
            if shape not in seen:
                seen.add(shape)
                out.append(shape)
    return out


def shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    _r0, _c0, r1, c1 = bbox_coords(list(shape))
    return r1 + 1, c1 + 1


def translate_shape(shape: tuple[tuple[int, int], ...], top: int, left: int) -> set[tuple[int, int]]:
    return {(top + r, left + c) for r, c in shape}


def expand_cells(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int) -> set[tuple[int, int]]:
    frontier = set(cells)
    expanded = set(cells)
    for _ in range(radius):
        new_frontier: set[tuple[int, int]] = set()
        for r, c in frontier:
            for nr, nc in neighbors4(r, c):
                if in_bounds(shape, nr, nc) and (nr, nc) not in expanded:
                    new_frontier.add((nr, nc))
        expanded |= new_frontier
        frontier = new_frontier
    return expanded


ALL_SHAPES = tuple(
    dict.fromkeys(shape for base in BASE_SHAPES for shape in unique_transforms(base))
)
SHRINK_SHAPES = tuple(shape for shape in ALL_SHAPES if erode_object(set(shape), shape_dims(shape)))


def sample_axis_pair(size: int, rng: random.Random) -> tuple[int, int]:
    if rng.random() < 0.5:
        a = rng.randint(2, size - 7)
        b = rng.randint(a + 4, size - 3)
        return a, b
    b = rng.randint(2, size - 7)
    a = rng.randint(b + 4, size - 3)
    return a, b


def placement_candidates(
    region: np.ndarray,
    shapes: tuple[tuple[tuple[int, int], ...], ...],
    forbidden: set[tuple[int, int]],
    path_and_markers: set[tuple[int, int]],
    side: str,
) -> list[set[tuple[int, int]]]:
    h, w = region.shape
    candidates: list[set[tuple[int, int]]] = []

    for shape in shapes:
        sh, sw = shape_dims(shape)
        for top in range(h - sh + 1):
            for left in range(w - sw + 1):
                cells = translate_shape(shape, top, left)
                if any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells):
                    continue
                if cells & forbidden:
                    continue
                if any(not region[r, c] for r, c in cells):
                    continue

                if side == "red":
                    growth = expand_cells(cells, region.shape, 1) - cells
                    if len(growth) < 2 or growth & path_and_markers:
                        continue
                else:
                    keep = erode_object(cells, region.shape)
                    if not keep or len(cells - keep) < 2:
                        continue

                candidates.append(cells)

    return candidates


def validate_input(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    if np.count_nonzero(grid == RED) != 1 or np.count_nonzero(grid == TEAL) != 1:
        return False

    components = find_blue_components(grid)
    if len(components) < 4:
        return False
    if any(not (4 <= len(comp) <= 7) for comp in components):
        return False

    red, teal, yellow_cells, _blocked, red_side = scene_data(grid)
    red_count = 0
    teal_count = 0
    marker_neighbors = set(neighbors4(*red)) | set(neighbors4(*teal))
    for comp in components:
        comp_set = set(comp)
        if comp_set & marker_neighbors:
            return False
        rep = representative(comp)
        if red_side[rep]:
            red_count += 1
        else:
            teal_count += 1
            if not erode_object(comp_set, grid.shape):
                return False

    if red_count < 2 or teal_count < 2:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if any(int(out[r, c]) != YELLOW for r, c in yellow_cells):
        return False
    if int(out[red]) != RED or int(out[teal]) != TEAL:
        return False
    return True


def generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(14, 18)
    w = rng.randint(14, 18)
    rr, tr = sample_axis_pair(h, rng)
    rc, tc = sample_axis_pair(w, rng)

    grid = np.zeros((h, w), dtype=int)
    red = (rr, rc)
    teal = (tr, tc)
    grid[red] = RED
    grid[teal] = TEAL

    yellow = path_cells(red, teal)
    blocked = np.zeros_like(grid, dtype=bool)
    for r, c in yellow:
        blocked[r, c] = True
    red_region = red_side_mask(grid.shape, red, blocked)
    teal_region = (~blocked) & (~red_region)
    red_region[red] = False
    red_region[teal] = False
    teal_region[red] = False
    teal_region[teal] = False

    path_and_markers = set(yellow) | {red, teal}
    forbidden = expand_cells(set(yellow), grid.shape, 1) | expand_cells({red, teal}, grid.shape, 1)

    red_target = rng.randint(2, 3)
    teal_target = rng.randint(2, 3)

    for side, region, target, shapes in (
        ("teal", teal_region, teal_target, SHRINK_SHAPES),
        ("red", red_region, red_target, ALL_SHAPES),
    ):
        for _ in range(target):
            shape_order = list(shapes)
            rng.shuffle(shape_order)
            candidates = placement_candidates(region, tuple(shape_order), forbidden, path_and_markers, side)
            if not candidates:
                return None
            cells = rng.choice(candidates)
            for r, c in cells:
                grid[r, c] = BLUE
            forbidden |= expand_cells(cells, grid.shape, 2)

    return grid if validate_input(grid) else None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid samples after {GEN_ATTEMPTS} attempts")
