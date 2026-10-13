from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
YELLOW = 4
ORANGE = 7
INPUT_COLORS = (1, 2, 3, 5, 6, 8, 9)

ORTHO_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAG_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))

GENERATE_ATTEMPTS = 400
PLACE_ATTEMPTS = 120


def _normalize_coords(coords: set[tuple[int, int]] | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _c in coords)
    min_c = min(c for _r, c in coords)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in coords))


def _bbox(coords: set[tuple[int, int]] | list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int]:
    max_r = max(r for r, _c in coords)
    max_c = max(c for _r, c in coords)
    return int(max_r + 1), int(max_c + 1)


def _apply_random_symmetry(
    coords: tuple[tuple[int, int], ...],
    rng: np.random.Generator,
) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    if bool(rng.integers(0, 2)):
        pts = [(r, -c) for r, c in pts]
    for _ in range(int(rng.integers(0, 4))):
        pts = [(c, -r) for r, c in pts]
    return _normalize_coords(pts)


def _grow_blob(size: int, rng: np.random.Generator) -> tuple[tuple[int, int], ...] | None:
    cells = {(0, 0)}
    while len(cells) < size:
        frontier: list[tuple[tuple[int, int], int]] = []
        seen: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in ORTHO_DIRS:
                nr, nc = r + dr, c + dc
                if (nr, nc) in cells or (nr, nc) in seen:
                    continue
                seen.add((nr, nc))
                shared = sum((nr + rr, nc + cc) in cells for rr, cc in ORTHO_DIRS)
                frontier.append(((nr, nc), 1 + 3 * shared))
        if not frontier:
            return None
        weights = np.array([w for _cell, w in frontier], dtype=float)
        weights /= weights.sum()
        choice = int(rng.choice(len(frontier), p=weights))
        cells.add(frontier[choice][0])
    return _apply_random_symmetry(_normalize_coords(cells), rng)


def _grow_path(
    size: int,
    rng: np.random.Generator,
    *,
    max_turns: int,
    straight_bias: int,
) -> tuple[tuple[int, int], ...] | None:
    for _ in range(80):
        path = [(0, 0)]
        used = {(0, 0)}
        last_dir: tuple[int, int] | None = None
        turns = 0
        while len(path) < size:
            r, c = path[-1]
            weighted: list[tuple[tuple[int, int], tuple[int, int], int]] = []
            for dr, dc in ORTHO_DIRS:
                nr, nc = r + dr, c + dc
                if (nr, nc) in used:
                    continue
                next_turns = turns + int(last_dir is not None and last_dir != (dr, dc))
                if next_turns > max_turns:
                    continue
                weight = straight_bias if last_dir == (dr, dc) else 1
                for _ in range(weight):
                    weighted.append(((nr, nc), (dr, dc), next_turns))
            if not weighted:
                break
            nr_nc, last_dir, turns = weighted[int(rng.integers(0, len(weighted)))]
            path.append(nr_nc)
            used.add(nr_nc)
        if len(path) == size:
            return _apply_random_symmetry(_normalize_coords(used), rng)
    return None


def _make_l_shape(size: int, rng: np.random.Generator) -> tuple[tuple[int, int], ...]:
    splits = [(a, size - a + 1) for a in range(2, size) if size - a + 1 >= 2]
    arm_r, arm_c = splits[int(rng.integers(0, len(splits)))]
    cells = {(r, 0) for r in range(arm_r)}
    cells.update((0, c) for c in range(arm_c))
    return _apply_random_symmetry(_normalize_coords(cells), rng)


def _sample_shape(size: int, rng: np.random.Generator) -> tuple[tuple[int, int], ...]:
    for _ in range(120):
        mode = int(rng.integers(0, 4))
        shape: tuple[tuple[int, int], ...] | None
        if mode == 0:
            shape = _grow_path(size, rng, max_turns=0 if size <= 4 else 1, straight_bias=5)
        elif mode == 1:
            shape = _grow_path(size, rng, max_turns=2, straight_bias=3)
        elif mode == 2:
            shape = _make_l_shape(size, rng)
        else:
            shape = _grow_blob(size, rng)
        if shape is None:
            continue
        h, w = _bbox(shape)
        if h <= size and w <= size:
            return shape
    raise RuntimeError(f"failed to build shape of size {size}")


def _expand_unbounded(
    cells: set[tuple[int, int]],
    diagonal: bool,
    depth: int,
) -> set[tuple[int, int]]:
    moves = DIAG_DIRS if diagonal else ORTHO_DIRS
    frontier = set(cells)
    visited = set(cells)
    halo: set[tuple[int, int]] = set()
    for _ in range(depth):
        next_frontier: set[tuple[int, int]] = set()
        for r, c in frontier:
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in visited and (nr, nc) not in next_frontier:
                    next_frontier.add((nr, nc))
        if not next_frontier:
            break
        visited.update(next_frontier)
        halo.update(next_frontier)
        frontier = next_frontier
    return halo


def _expand_bounded(
    cells: set[tuple[int, int]],
    shape: tuple[int, int],
    diagonal: bool,
    depth: int,
) -> set[tuple[int, int]]:
    height, width = map(int, shape)
    moves = DIAG_DIRS if diagonal else ORTHO_DIRS
    frontier = set(cells)
    visited = set(cells)
    halo: set[tuple[int, int]] = set()
    for _ in range(depth):
        next_frontier: set[tuple[int, int]] = set()
        for r, c in frontier:
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                if (nr, nc) not in visited and (nr, nc) not in next_frontier:
                    next_frontier.add((nr, nc))
        if not next_frontier:
            break
        visited.update(next_frontier)
        halo.update(next_frontier)
        frontier = next_frontier
    return halo


def _distance_halo(
    cells: set[tuple[int, int]],
    shape: tuple[int, int],
    diagonal: bool,
    depth: int,
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int]]:
    height, width = map(int, shape)
    moves = DIAG_DIRS if diagonal else ORTHO_DIRS
    dist = {cell: 0 for cell in cells}
    queue = deque(cells)
    while queue:
        r, c = queue.popleft()
        if dist[(r, c)] >= depth:
            continue
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if (nr, nc) in dist:
                continue
            dist[(nr, nc)] = dist[(r, c)] + 1
            queue.append((nr, nc))
    halo = {cell for cell, value in dist.items() if 1 <= value <= depth}
    return halo, dist


def _nonzero_components(grid: np.ndarray) -> list[np.ndarray]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if grid[r, c] == BLACK or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in ORTHO_DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] != BLACK and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Add a size-driven halo whose geometry and depth are both encoded by each foreground component's pixel count.

    Input structure:
    - The grid is a black background with several isolated non-black 4-connected foreground components.
    - Each component is monochrome in valid inputs, has area 3-7, and keeps its original color.
    - Components are spaced so their derived halo regions normally do not overlap.

    Rule:
    - Find every 4-connected non-black component and count its pixels as N.
    - If N is even, grow N/2 diagonal-only layers from the component and mark those reachable cells yellow.
    - If N is odd, grow (N-1)/2 orthogonal-only layers from the component and mark those reachable cells orange.
    - Skip out-of-bounds cells, never overwrite original foreground pixels, and if halo regions overlap then the larger-N component takes priority.

    Key insight:
    - The same quantity N is used twice: parity chooses the neighborhood geometry, while the magnitude chooses how many expansion layers are drawn.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    components = _nonzero_components(g)
    halo_infos: list[tuple[int, int, int, set[tuple[int, int]]]] = []
    for index, coords in enumerate(components):
        cells = {tuple(map(int, cell)) for cell in coords}
        size = len(cells)
        diagonal = size % 2 == 0
        depth = size // 2 if diagonal else (size - 1) // 2
        mark_color = YELLOW if diagonal else ORANGE
        halo = _expand_bounded(cells, g.shape, diagonal, depth)
        halo_infos.append((size, index, mark_color, halo))

    best_size = np.full(g.shape, -1, dtype=int)
    for size, index, mark_color, halo in sorted(halo_infos, key=lambda item: (-item[0], item[1])):
        for r, c in halo:
            if g[r, c] != BLACK:
                continue
            if size > int(best_size[r, c]):
                best_size[r, c] = int(size)
                out[r, c] = int(mark_color)
    return out


def _component_spec(size: int, color: int, rng: np.random.Generator, force_edge: bool) -> dict:
    cells_local = _sample_shape(size, rng)
    cells_set = set(cells_local)
    diagonal = size % 2 == 0
    depth = size // 2 if diagonal else (size - 1) // 2
    halo_local = _expand_unbounded(cells_set, diagonal, depth)
    shape_h, shape_w = _bbox(cells_local)
    return {
        "size": int(size),
        "color": int(color),
        "cells_local": cells_local,
        "cells_local_set": cells_set,
        "halo_local": halo_local,
        "shape_h": int(shape_h),
        "shape_w": int(shape_w),
        "diagonal": bool(diagonal),
        "depth": int(depth),
        "mark_color": YELLOW if diagonal else ORANGE,
        "force_edge": bool(force_edge),
    }


def _touches_border(coords: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    height, width = map(int, shape)
    return any(r == 0 or c == 0 or r == height - 1 or c == width - 1 for r, c in coords)


def _place_components(
    specs: list[dict],
    shape: tuple[int, int],
    rng: np.random.Generator,
) -> list[dict] | None:
    height, width = map(int, shape)
    reserved: set[tuple[int, int]] = set()
    placed: list[dict] = []
    ordered = sorted(
        specs,
        key=lambda spec: (0 if spec["force_edge"] else 1, -(len(spec["cells_local_set"]) + len(spec["halo_local"]))),
    )

    for spec in ordered:
        candidates: list[dict] = []
        for r0 in range(height - spec["shape_h"] + 1):
            for c0 in range(width - spec["shape_w"] + 1):
                cells_actual = {(r0 + r, c0 + c) for r, c in spec["cells_local"]}
                halo_actual = {
                    (r0 + r, c0 + c)
                    for r, c in spec["halo_local"]
                    if 0 <= r0 + r < height and 0 <= c0 + c < width
                }
                footprint = cells_actual | halo_actual
                if footprint & reserved:
                    continue
                if spec["force_edge"] and not _touches_border(footprint, shape):
                    continue
                candidates.append(
                    {
                        **spec,
                        "cells_actual": cells_actual,
                        "halo_actual": halo_actual,
                        "footprint": footprint,
                    }
                )
        if not candidates:
            return None
        chosen = candidates[int(rng.integers(0, len(candidates)))]
        reserved.update(chosen["footprint"])
        placed.append(chosen)
    return placed


def _render_input(placed: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec in placed:
        for r, c in spec["cells_actual"]:
            grid[r, c] = int(spec["color"])
    return grid


def _oracle_from_specs(grid: np.ndarray, placed: list[dict]) -> np.ndarray:
    out = grid.copy()
    best_size = np.full(grid.shape, -1, dtype=int)
    for spec in sorted(placed, key=lambda item: (-item["size"], item["color"])):
        for r, c in spec["halo_actual"]:
            if grid[r, c] != BLACK:
                continue
            if spec["size"] > int(best_size[r, c]):
                best_size[r, c] = int(spec["size"])
                out[r, c] = int(spec["mark_color"])
    return out


def _oracle_from_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    best_size = np.full(g.shape, -1, dtype=int)
    infos: list[tuple[int, int, int, set[tuple[int, int]]]] = []
    for index, coords in enumerate(_nonzero_components(g)):
        cells = {tuple(map(int, cell)) for cell in coords}
        size = len(cells)
        diagonal = size % 2 == 0
        depth = size // 2 if diagonal else (size - 1) // 2
        halo, _dist = _distance_halo(cells, g.shape, diagonal, depth)
        infos.append((size, index, YELLOW if diagonal else ORANGE, halo))
    for size, index, mark_color, halo in sorted(infos, key=lambda item: (-item[0], item[1])):
        for r, c in halo:
            if g[r, c] != BLACK:
                continue
            if size > int(best_size[r, c]):
                best_size[r, c] = int(size)
                out[r, c] = int(mark_color)
    return out


def _validate_generated(grid: np.ndarray, placed: list[dict], expected: np.ndarray) -> bool:
    parsed = _nonzero_components(grid)
    if not (4 <= len(parsed) <= 6 and len(parsed) == len(placed)):
        return False

    by_color = {int(spec["color"]): spec for spec in placed}
    sizes: list[int] = []
    occupied: set[tuple[int, int]] = set()
    even_max_visible = False
    odd_max_visible = False

    for coords in parsed:
        cells = {tuple(map(int, cell)) for cell in coords}
        colors = {int(grid[r, c]) for r, c in cells}
        if len(colors) != 1:
            return False
        color = next(iter(colors))
        if color not in by_color:
            return False
        spec = by_color[color]
        if cells != spec["cells_actual"]:
            return False
        size = len(cells)
        if size != int(spec["size"]) or not (3 <= size <= 7):
            return False
        occupied |= cells
        sizes.append(size)

    if not {3, 4, 5, 6}.issubset(set(sizes)):
        return False
    if not any(size % 2 == 0 for size in sizes) or not any(size % 2 == 1 for size in sizes):
        return False

    footprints = [spec["footprint"] for spec in placed]
    for i in range(len(footprints)):
        for j in range(i + 1, len(footprints)):
            if footprints[i] & footprints[j]:
                return False

    for spec in placed:
        halo, dist = _distance_halo(spec["cells_actual"], grid.shape, spec["diagonal"], spec["depth"])
        if halo != spec["halo_actual"]:
            return False
        visible = halo - occupied
        if not visible:
            return False
        max_layer_visible = any(dist[cell] == spec["depth"] for cell in visible if cell in dist)
        if spec["diagonal"] and spec["depth"] >= 2 and max_layer_visible:
            even_max_visible = True
        if (not spec["diagonal"]) and spec["depth"] >= 2 and max_layer_visible:
            odd_max_visible = True
        expected_color = spec["mark_color"]
        if not np.any(expected == expected_color):
            return False

    if not even_max_visible or not odd_max_visible:
        return False
    if not np.any(expected == YELLOW) or not np.any(expected == ORANGE):
        return False
    if np.array_equal(expected, grid):
        return False
    if not np.array_equal(_oracle_from_grid(grid), expected):
        return False
    if not np.array_equal(main(grid), expected):
        return False
    return True


def _sample_sizes(count: int, rng: np.random.Generator) -> list[int]:
    sizes = [3, 4, 5, 6]
    if count >= 5:
        sizes.append(7)
    if count >= 6:
        sizes.append(int(rng.choice(np.array([3, 4, 5, 6], dtype=int))))
    rng.shuffle(sizes)
    return sizes


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    count = int(rng.integers(4, 7))
    sizes = _sample_sizes(count, rng)
    height = int(rng.integers(17 + 2 * (count - 4), 22 + 2 * (count - 4)))
    width = int(rng.integers(17 + 2 * (count - 4), 22 + 2 * (count - 4)))
    force_edge = bool(rng.integers(0, 2))
    edge_index = int(rng.integers(0, count)) if force_edge else -1
    colors = list(rng.choice(np.array(INPUT_COLORS, dtype=int), size=count, replace=False))

    for _ in range(PLACE_ATTEMPTS):
        specs = [
            _component_spec(size, colors[i], rng, force_edge and i == edge_index)
            for i, size in enumerate(sizes)
        ]
        placed = _place_components(specs, (height, width), rng)
        if placed is None:
            continue
        grid = _render_input(placed, (height, width))
        expected = _oracle_from_specs(grid, placed)
        if _validate_generated(grid, placed, expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 1")
