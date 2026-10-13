from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
GREEN = 3
COLOR_POOL = (1, 2, 4, 5, 6, 7, 8, 9)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
OPPOSITE = {
    (-1, 0): (1, 0),
    (1, 0): (-1, 0),
    (0, -1): (0, 1),
    (0, 1): (0, -1),
}


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _cross_cells(center: tuple[int, int], missing_dir: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    cells = {(r, c)}
    for direction in DIRS:
        if direction != missing_dir:
            dr, dc = direction
            cells.add((r + dr, c + dc))
    return cells


def _full_plus_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}


def _same_color_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLACK or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if not in_bounds(grid.shape, nr, nc):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            components.append({"color": color, "cells": cells})
    return components


def _component_to_cross(component: dict) -> dict:
    cells = {tuple(cell) for cell in component["cells"]}
    assert len(cells) == 4, "each foreground component must have four cells"

    centers = []
    for r, c in cells:
        present_dirs = []
        for dr, dc in DIRS:
            if (r + dr, c + dc) in cells:
                present_dirs.append((dr, dc))
        if len(present_dirs) == 3:
            centers.append((r, c, present_dirs))
        else:
            assert len(present_dirs) == 1, "non-center cross cells must be single-arm leaves"

    assert len(centers) == 1, "each cross must have exactly one center"
    r, c, present_dirs = centers[0]
    missing = [direction for direction in DIRS if direction not in present_dirs]
    assert len(missing) == 1
    missing_dir = missing[0]

    assert cells == _cross_cells((r, c), missing_dir), "component must be a one-arm-missing cross"
    return {
        "color": int(component["color"]),
        "center": (r, c),
        "missing_dir": missing_dir,
        "cells": cells,
    }


def _parse_crosses(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int], int]:
    components = _same_color_components(grid)
    crosses = []
    coord_to_cross: dict[tuple[int, int], int] = {}
    for component in components:
        cross = _component_to_cross(component)
        cross["index"] = len(crosses)
        crosses.append(cross)
        for cell in cross["cells"]:
            assert cell not in coord_to_cross
            coord_to_cross[cell] = cross["index"]

    start_candidates = [cross["index"] for cross in crosses if int(cross["color"]) == GREEN]
    assert len(start_candidates) == 1, "there must be exactly one green start cross"
    return crosses, coord_to_cross, start_candidates[0]


def _trace_from_cross(cross: dict, grid: np.ndarray, coord_to_cross: dict[tuple[int, int], int]) -> tuple[list[tuple[int, int]], int | None]:
    r, c = cross["center"]
    dr, dc = cross["missing_dir"]
    ray = []
    r += dr
    c += dc

    while in_bounds(grid.shape, r, c):
        if int(grid[r, c]) == BLACK:
            ray.append((r, c))
            r += dr
            c += dc
            continue

        hit = coord_to_cross.get((r, c))
        assert hit is not None and hit != cross["index"], "ray must hit another cross cell or leave the grid"
        return ray, hit

    return ray, None


def _follow_chain(grid: np.ndarray, crosses: list[dict], coord_to_cross: dict[tuple[int, int], int], start_index: int) -> tuple[list[int], list[list[tuple[int, int]]]]:
    order = []
    rays = []
    seen = set()
    current = start_index

    while True:
        assert current not in seen, "chain must not cycle"
        seen.add(current)
        order.append(current)

        ray, nxt = _trace_from_cross(crosses[current], grid, coord_to_cross)
        rays.append(ray)
        if nxt is None:
            return order, rays
        current = nxt


def _paint_chain(grid: np.ndarray, crosses: list[dict], order: list[int], rays: list[list[tuple[int, int]]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for cross_index, ray in zip(order, rays):
        color = int(crosses[cross_index]["color"])
        for r, c in ray:
            out[r, c] = color
    return out


def _paint_all_rays(grid: np.ndarray, crosses: list[dict], coord_to_cross: dict[tuple[int, int], int]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for cross in crosses:
        ray, _ = _trace_from_cross(cross, grid, coord_to_cross)
        for r, c in ray:
            out[r, c] = int(cross["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Draw only the causally activated rays emitted by one-arm-missing crosses, starting from the green cross.

    Input structure:
    - The grid has a black background and several monochromatic incomplete crosses.
    - Each foreground object is a 4-connected center-plus-three-arms cross: a 3x3 plus shape with exactly one missing arm.
    - Exactly one cross is green; it is the start marker. Other crosses may be struck later or remain untouched.

    Rule:
    - Read each cross's firing direction from the missing arm.
    - Starting from the green cross, color the background cells along that ray with the cross's own color until the ray first hits another cross or exits the grid.
    - A hit cross becomes active and fires along its own missing-arm direction; untouched crosses never fire.
    - Preserve every cross itself and add only the rays produced by this single activation chain.

    Key insight:
    - The direction cue is negative space: the absent arm encodes where the ray goes, and only the cross reached by the previous ray is allowed to continue the process.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.any(g == GREEN), "expected a green start cross"

    crosses, coord_to_cross, start_index = _parse_crosses(g)
    order, rays = _follow_chain(g, crosses, coord_to_cross, start_index)
    return _paint_chain(g, crosses, order, rays)


_RNG = np.random.default_rng()
GEN_RETRY_LIMIT = 400
CHAIN_LEN_RANGE = (3, 6)
DISTRACTOR_RANGE = (1, 4)
GRID_SIZE_RANGE = (15, 27)
SEGMENT_LENGTHS = (4, 5, 6, 7, 8)
DISTRACTOR_PLACEMENT_TRIES = 250


def _shuffled(seq, rng: np.random.Generator) -> list:
    items = list(seq)
    rng.shuffle(items)
    return items


def _random_center(shape: tuple[int, int], rng: np.random.Generator) -> tuple[int, int]:
    h, w = shape
    return int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2))


def _segment_ray_cells(src: tuple[int, int], dst: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    stop = (dst[0] - dr, dst[1] - dc)
    r, c = src
    cells = []
    while True:
        r += dr
        c += dc
        if (r, c) == stop:
            return cells
        cells.append((r, c))


def _edge_ray_cells(src: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    h, w = shape
    r, c = src
    cells = []
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return cells
        cells.append((r, c))


def _spec_grid(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec in specs:
        for r, c in _cross_cells(spec["center"], spec["missing_dir"]):
            assert in_bounds(shape, r, c)
            assert int(grid[r, c]) == BLACK
            grid[r, c] = int(spec["color"])
    return grid


def _build_chain_specs(shape: tuple[int, int], chain_len: int, rng: np.random.Generator) -> list[dict] | None:
    centers = [_random_center(shape, rng)]
    directions = []
    reserved_plus = set(_full_plus_cells(centers[0]))
    reserved_path: set[tuple[int, int]] = set()
    previous_direction = None
    turned = False

    for _ in range(chain_len - 1):
        placed = False
        options = _shuffled(DIRS, rng)
        if previous_direction is not None:
            forbidden = OPPOSITE[previous_direction]
            options = [direction for direction in options if direction != forbidden]

        for direction in options:
            for distance in _shuffled(SEGMENT_LENGTHS, rng):
                nr = centers[-1][0] + direction[0] * distance
                nc = centers[-1][1] + direction[1] * distance
                next_center = (nr, nc)
                next_plus = _full_plus_cells(next_center)
                if any(not in_bounds(shape, r, c) for r, c in next_plus):
                    continue
                if next_plus & reserved_plus:
                    continue

                corridor = _segment_ray_cells(centers[-1], next_center, direction)
                if len(corridor) < 2:
                    continue
                tail = set(corridor[1:])
                if tail & reserved_plus:
                    continue
                if set(corridor) & reserved_path:
                    continue

                directions.append(direction)
                centers.append(next_center)
                reserved_plus |= next_plus
                reserved_path |= set(corridor)
                if previous_direction is not None and direction != previous_direction:
                    turned = True
                previous_direction = direction
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    final_options = _shuffled(DIRS, rng)
    if previous_direction is not None:
        forbidden = OPPOSITE[previous_direction]
        final_options = [direction for direction in final_options if direction != forbidden]
    if not turned and previous_direction is not None:
        final_options = [direction for direction in final_options if direction != previous_direction] + [
            direction for direction in final_options if direction == previous_direction
        ]

    final_direction = None
    for direction in final_options:
        corridor = _edge_ray_cells(centers[-1], direction, shape)
        if len(corridor) < 2:
            continue
        if set(corridor[1:]) & reserved_plus:
            continue
        if set(corridor) & reserved_path:
            continue
        directions.append(direction)
        final_direction = direction
        if previous_direction is not None and direction != previous_direction:
            turned = True
        break

    if final_direction is None or not turned:
        return None

    extra_colors = rng.choice(np.array(COLOR_POOL, dtype=int), size=chain_len - 1, replace=False).tolist()
    colors = [GREEN] + [int(color) for color in extra_colors]
    return [
        {"center": center, "missing_dir": direction, "color": color}
        for center, direction, color in zip(centers, directions, colors)
    ]


def _chain_ray_cells(chain_specs: list[dict], shape: tuple[int, int]) -> tuple[set[tuple[int, int]], list[tuple[int, int]]]:
    grid = _spec_grid(shape, chain_specs)
    crosses, coord_to_cross, start_index = _parse_crosses(grid)
    center_to_index = {tuple(cross["center"]): int(cross["index"]) for cross in crosses}
    expected_order = [center_to_index[tuple(spec["center"])] for spec in chain_specs]
    order, rays = _follow_chain(grid, crosses, coord_to_cross, start_index)
    if order != expected_order:
        raise AssertionError("constructed chain does not activate in the intended order")
    if any(len(ray) < 2 for ray in rays):
        raise AssertionError("each active ray should be visibly long")
    ray_cells = set(cell for ray in rays for cell in ray)
    if len(ray_cells) != sum(len(ray) for ray in rays):
        raise AssertionError("active rays should stay visually separated")
    return ray_cells, expected_order


def _place_distractors(chain_specs: list[dict], shape: tuple[int, int], count: int, rng: np.random.Generator) -> list[dict] | None:
    chain_ray_cells, _ = _chain_ray_cells(chain_specs, shape)
    used_colors = {int(spec["color"]) for spec in chain_specs}
    available_colors = [color for color in COLOR_POOL if color not in used_colors]
    if count > len(available_colors):
        return None

    placed: list[dict] = []
    occupied_cross_cells = set()
    for spec in chain_specs:
        occupied_cross_cells |= _cross_cells(spec["center"], spec["missing_dir"])
    reserved_silent_rays: set[tuple[int, int]] = set()

    for color in _shuffled(available_colors, rng)[:count]:
        placed_this = False
        for _ in range(DISTRACTOR_PLACEMENT_TRIES):
            center = _random_center(shape, rng)
            direction = _shuffled(DIRS, rng)[0]
            cells = _cross_cells(center, direction)
            if any(not in_bounds(shape, r, c) for r, c in cells):
                continue
            if cells & occupied_cross_cells:
                continue
            if cells & chain_ray_cells:
                continue
            if cells & reserved_silent_rays:
                continue

            candidate = {"center": center, "missing_dir": direction, "color": int(color)}
            specs = chain_specs + placed + [candidate]
            grid = _spec_grid(shape, specs)
            crosses, coord_to_cross, _start_index = _parse_crosses(grid)
            center_to_index = {tuple(cross["center"]): int(cross["index"]) for cross in crosses}
            candidate_index = center_to_index[center]
            ray, _ = _trace_from_cross(crosses[candidate_index], grid, coord_to_cross)
            ray_cells = set(ray)

            if len(ray) < 2:
                continue
            if ray_cells & chain_ray_cells:
                continue
            if ray_cells & occupied_cross_cells:
                continue
            if ray_cells & reserved_silent_rays:
                continue

            placed.append(candidate)
            occupied_cross_cells |= cells
            reserved_silent_rays |= ray_cells
            placed_this = True
            break
        if not placed_this:
            return None

    return placed


def _validate_case(grid: np.ndarray, chain_specs: list[dict]) -> bool:
    try:
        crosses, coord_to_cross, start_index = _parse_crosses(grid)
        center_to_index = {tuple(cross["center"]): int(cross["index"]) for cross in crosses}
        expected_order = [center_to_index[tuple(spec["center"])] for spec in chain_specs]
        order, rays = _follow_chain(grid, crosses, coord_to_cross, start_index)
        if order != expected_order:
            return False
        if len(order) != len(chain_specs):
            return False

        true_output = _paint_chain(grid, crosses, order, rays)
        if not np.array_equal(true_output, main(grid)):
            return False
        if np.array_equal(true_output, grid):
            return False

        all_fire_output = _paint_all_rays(grid, crosses, coord_to_cross)
        if np.array_equal(all_fire_output, true_output):
            return False

        active_cells = set(cell for ray in rays for cell in ray)
        if any(_cross_cells(spec["center"], spec["missing_dir"]) & active_cells for spec in chain_specs):
            return False
        if len(active_cells) != sum(len(ray) for ray in rays):
            return False

        return True
    except AssertionError:
        return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*GRID_SIZE_RANGE))
    w = int(rng.integers(*GRID_SIZE_RANGE))
    shape = (h, w)

    chain_len = int(rng.integers(*CHAIN_LEN_RANGE))
    max_distractors = min(DISTRACTOR_RANGE[1] - 1, len(COLOR_POOL) + 1 - chain_len)
    if max_distractors < DISTRACTOR_RANGE[0]:
        return None
    distractor_count = int(rng.integers(DISTRACTOR_RANGE[0], max_distractors + 1))

    chain_specs = _build_chain_specs(shape, chain_len, rng)
    if chain_specs is None:
        return None

    try:
        _chain_ray_cells(chain_specs, shape)
    except AssertionError:
        return None

    distractors = _place_distractors(chain_specs, shape, distractor_count, rng)
    if distractors is None:
        return None

    specs = chain_specs + distractors
    grid = _spec_grid(shape, specs)
    if not _validate_case(grid, chain_specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chained missing-cross input")
