from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, colors_present, get_bg


DIRS: tuple[tuple[int, int], ...] = ((-1, -1), (-1, 1), (1, -1), (1, 1))
BG = int(Color.BLACK)
WALL = int(Color.GREY)
PAIR_COLORS = [int(c) for c in Color.NOT_BLACK if int(c) != WALL]

GEN_HEIGHT_RANGE = (12, 20)
GEN_WIDTH_RANGE = (12, 20)
GEN_PAIR_COUNT_RANGE = (2, 4)
GEN_EXTRA_SEGMENT_RANGE = (0, 2)
GEN_SEGMENT_MIN_LEN = 3
GEN_MIN_ROUTE_STEPS = 4
GEN_PAIR_SEARCH_ATTEMPTS = 240
GEN_MAX_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < int(shape[0]) and 0 <= c < int(shape[1])


def _reflect_step(
    grid: np.ndarray, r: int, c: int, dr: int, dc: int, wall: int = WALL
) -> tuple[int, int, int, int, bool, bool] | None:
    """Reflect row/col components against borders or orthogonal wall faces, then move one diagonal step."""
    height, width = map(int, grid.shape)

    row_out = not (0 <= r + dr < height)
    col_out = not (0 <= c + dc < width)
    row_wall = (not row_out) and int(grid[r + dr, c]) == wall
    col_wall = (not col_out) and int(grid[r, c + dc]) == wall

    next_dr = -dr if (row_out or row_wall) else dr
    next_dc = -dc if (col_out or col_wall) else dc
    nr, nc = r + next_dr, c + next_dc

    if not _in_bounds(grid.shape, nr, nc):
        return None
    if int(grid[nr, nc]) == wall:
        return None
    return nr, nc, next_dr, next_dc, bool(row_wall or col_wall), bool(row_out or col_out)


def _trace_to_target(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    target: tuple[int, int],
    *,
    bg: int | None = None,
) -> dict[str, object] | None:
    if bg is None:
        bg = int(get_bg(grid))

    r, c = map(int, start)
    dr, dc = map(int, direction)
    seen_states = {(r, c, dr, dc)}
    path: list[tuple[int, int]] = []
    wall_bounces = 0
    boundary_bounces = 0

    for _ in range(int(grid.size) * 8):
        step = _reflect_step(grid, r, c, dr, dc)
        if step is None:
            return None
        nr, nc, dr, dc, hit_wall, hit_boundary = step
        wall_bounces += int(hit_wall)
        boundary_bounces += int(hit_boundary)

        if (nr, nc) == target:
            return {
                "path": path,
                "wall_bounces": wall_bounces,
                "boundary_bounces": boundary_bounces,
                "direction": direction,
            }

        if int(grid[nr, nc]) != bg:
            return None

        path.append((nr, nc))
        state = (nr, nc, dr, dc)
        if state in seen_states:
            return None
        seen_states.add(state)
        r, c = nr, nc

    return None


def _trace_orbit(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    *,
    bg: int = BG,
) -> dict[str, object] | None:
    """Trace a simple prefix in a wall scene, stopping before any self-intersection or state loop."""
    r, c = map(int, start)
    dr, dc = map(int, direction)
    seen_states = {(r, c, dr, dc)}
    seen_cells = {start}
    cells: list[tuple[int, int]] = []
    wall_prefix: list[int] = []
    boundary_prefix: list[int] = []
    wall_bounces = 0
    boundary_bounces = 0

    for _ in range(int(grid.size) * 8):
        step = _reflect_step(grid, r, c, dr, dc)
        if step is None:
            return None
        nr, nc, dr, dc, hit_wall, hit_boundary = step
        if int(grid[nr, nc]) != bg:
            return None

        wall_bounces += int(hit_wall)
        boundary_bounces += int(hit_boundary)
        state = (nr, nc, dr, dc)
        if state in seen_states or (nr, nc) in seen_cells:
            break

        cells.append((nr, nc))
        wall_prefix.append(wall_bounces)
        boundary_prefix.append(boundary_bounces)
        seen_states.add(state)
        seen_cells.add((nr, nc))
        r, c = nr, nc

    if not cells:
        return None
    return {"cells": cells, "wall_prefix": wall_prefix, "boundary_prefix": boundary_prefix}


def _working_routes_between(
    grid: np.ndarray, start: tuple[int, int], target: tuple[int, int]
) -> list[dict[str, object]]:
    bg = int(get_bg(grid))
    routes: list[dict[str, object]] = []
    for direction in DIRS:
        route = _trace_to_target(grid, start, direction, target, bg=bg)
        if route is not None:
            routes.append(route)
    return routes


def _working_route_for_color(grid: np.ndarray, color: int) -> dict[str, object]:
    endpoints = np.argwhere(grid == int(color))
    assert endpoints.shape == (2, 2), f"color {color} must appear exactly twice"
    start = tuple(map(int, endpoints[0]))
    target = tuple(map(int, endpoints[1]))
    routes = _working_routes_between(grid, start, target)
    assert len(routes) == 1, f"color {color} must have exactly one valid launch direction"
    return routes[0]


def _is_direct_unbounced_diagonal(
    grid: np.ndarray, start: tuple[int, int], target: tuple[int, int]
) -> bool:
    dr = int(target[0] - start[0])
    dc = int(target[1] - start[1])
    if dr == 0 or abs(dr) != abs(dc):
        return False

    step_dir = ((dr > 0) - (dr < 0), (dc > 0) - (dc < 0))
    r, c = start
    for _ in range(abs(dr)):
        step = _reflect_step(grid, r, c, step_dir[0], step_dir[1])
        if step is None:
            return False
        nr, nc, next_dr, next_dc, hit_wall, hit_boundary = step
        if hit_wall or hit_boundary or (next_dr, next_dc) != step_dir:
            return False
        r, c = nr, nc
    return (r, c) == target


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-colored endpoint pair by tracing the unique 45-degree ray that reflects off borders and gray wall segments.

    Input structure:
    - A dominant background contains several singleton endpoint pairs, with each non-gray foreground color appearing exactly twice.
    - Additional gray cells form straight horizontal or vertical interior wall segments that stay fixed.
    - Each endpoint pair is intended to lie on one reflected diagonal route, not on multiple competing ones.

    Rule:
    - For each color pair, launch a 45-degree ray from one endpoint and test the four diagonal directions.
    - Whenever the next row step would cross the top or bottom border, or the orthogonal cell in that row direction is gray, flip the row direction; do the analogous flip for the column direction against left/right borders or side walls.
    - After applying those reflections, advance one diagonal cell; the unique direction that reaches the matching endpoint defines the path.
    - Paint every traversed background cell on that path with the pair color, while leaving endpoints and gray walls unchanged.

    Key insight:
    - The visible grid is a folded billiards board: the correct pair is straight only in the mentally unfolded copies, so the solver must simulate reflections and eliminate the three wrong launch directions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    bg = int(get_bg(g))
    wall = WALL
    assert bg != wall, "background cannot be the gray wall color"
    assert np.any(g == wall), "expected at least one gray wall segment"

    out = g.copy()
    colors = colors_present(g, exclude={bg, wall})
    assert colors, "expected at least one endpoint color"

    for color in colors:
        route = _working_route_for_color(g, int(color))
        for r, c in route["path"]:
            assert int(out[r, c]) in (bg, int(color)), "different color paths must not overlap"
            out[r, c] = int(color)
    return out


def _segment_cells(
    orientation: str, r0: int, c0: int, length: int
) -> list[tuple[int, int]]:
    if orientation == "h":
        return [(r0, c0 + offset) for offset in range(length)]
    return [(r0 + offset, c0) for offset in range(length)]


def _can_place_segment(mask: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    cell_set = set(cells)
    for r, c in cells:
        if not (1 <= r < mask.shape[0] - 1 and 1 <= c < mask.shape[1] - 1):
            return False
        if mask[r, c]:
            return False

    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if not _in_bounds(mask.shape, rr, cc):
                    continue
                if mask[rr, cc] and (rr, cc) not in cell_set:
                    return False
    return True


def _random_segment(mask: np.ndarray, orientation: str) -> list[tuple[int, int]] | None:
    height, width = map(int, mask.shape)
    max_len = min(8, (width - 2) if orientation == "h" else (height - 2))
    if max_len < GEN_SEGMENT_MIN_LEN:
        return None

    for _ in range(100):
        length = random.randint(GEN_SEGMENT_MIN_LEN, max_len)
        if orientation == "h":
            if width - length - 1 < 1:
                continue
            r0 = random.randint(1, height - 2)
            c0 = random.randint(1, width - length - 1)
        else:
            if height - length - 1 < 1:
                continue
            r0 = random.randint(1, height - length - 1)
            c0 = random.randint(1, width - 2)
        cells = _segment_cells(orientation, r0, c0, length)
        if _can_place_segment(mask, cells):
            return cells
    return None


def _build_wall_mask(height: int, width: int) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=bool)
    orientations = ["h", "v"]
    orientations.extend(random.choice(["h", "v"]) for _ in range(random.randint(*GEN_EXTRA_SEGMENT_RANGE)))
    random.shuffle(orientations)

    for orientation in orientations:
        cells = _random_segment(mask, orientation)
        if cells is None:
            return None
        for r, c in cells:
            mask[r, c] = True
    return mask


def _render_expected(
    inp: np.ndarray, pair_specs: list[dict[str, object]]
) -> np.ndarray:
    out = inp.copy()
    for spec in pair_specs:
        color = int(spec["color"])
        for r, c in spec["path"]:
            assert int(out[r, c]) == BG
            out[r, c] = color
    return out


def _find_pair_candidate(
    grid: np.ndarray,
    color: int,
    used_paths: set[tuple[int, int]],
    occupied_endpoints: set[tuple[int, int]],
    requirement: dict[str, bool],
) -> dict[str, object] | None:
    starts = [
        (int(r), int(c))
        for r, c in np.argwhere(grid == BG)
        if (int(r), int(c)) not in used_paths and (int(r), int(c)) not in occupied_endpoints
    ]
    if not starts:
        return None

    for _ in range(GEN_PAIR_SEARCH_ATTEMPTS):
        start = random.choice(starts)
        direction = random.choice(DIRS)
        orbit = _trace_orbit(grid, start, direction, bg=BG)
        if orbit is None or len(orbit["cells"]) < GEN_MIN_ROUTE_STEPS:
            continue

        order = list(range(len(orbit["cells"])))
        random.shuffle(order)
        for idx in order:
            target = orbit["cells"][idx]
            prefix_path = orbit["cells"][:idx]
            if idx + 1 < GEN_MIN_ROUTE_STEPS:
                continue
            if orbit["wall_prefix"][idx] + orbit["boundary_prefix"][idx] == 0:
                continue
            if requirement["need_wall"] and orbit["wall_prefix"][idx] == 0:
                continue
            if requirement["need_boundary"] and orbit["boundary_prefix"][idx] == 0:
                continue
            if target in used_paths or target in occupied_endpoints:
                continue
            if any(cell in used_paths or cell in occupied_endpoints for cell in prefix_path):
                continue
            if _is_direct_unbounced_diagonal(grid, start, target):
                continue

            trial = grid.copy()
            trial[start] = int(color)
            trial[target] = int(color)

            forward = _working_routes_between(trial, start, target)
            backward = _working_routes_between(trial, target, start)
            if len(forward) != 1 or len(backward) != 1:
                continue
            route = forward[0]
            if route["path"] != prefix_path:
                continue
            if route["wall_bounces"] + route["boundary_bounces"] == 0:
                continue
            if requirement["need_wall"] and int(route["wall_bounces"]) == 0:
                continue
            if requirement["need_boundary"] and int(route["boundary_bounces"]) == 0:
                continue

            return {
                "color": int(color),
                "start": start,
                "target": target,
                "path": route["path"],
                "wall_bounces": int(route["wall_bounces"]),
                "boundary_bounces": int(route["boundary_bounces"]),
            }
    return None


def _validate_scene(inp: np.ndarray, expected: np.ndarray, pair_specs: list[dict[str, object]]) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False

    seen_cells: set[tuple[int, int]] = set()
    for spec in pair_specs:
        endpoints = [tuple(spec["start"]), tuple(spec["target"])]
        if len(set(endpoints)) != 2:
            return False
        for endpoint in endpoints:
            if endpoint in seen_cells:
                return False
            seen_cells.add(endpoint)

        forward = _working_routes_between(inp, endpoints[0], endpoints[1])
        backward = _working_routes_between(inp, endpoints[1], endpoints[0])
        if len(forward) != 1 or len(backward) != 1:
            return False
        route = forward[0]
        if route["path"] != spec["path"]:
            return False
        if int(route["wall_bounces"]) != int(spec["wall_bounces"]):
            return False
        if int(route["boundary_bounces"]) != int(spec["boundary_bounces"]):
            return False
        for cell in route["path"]:
            if cell in seen_cells:
                return False
            seen_cells.add(cell)

    return np.array_equal(main(inp), expected)


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GEN_HEIGHT_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)
    wall_mask = _build_wall_mask(height, width)
    if wall_mask is None:
        return None

    inp = np.full((height, width), BG, dtype=int)
    inp[wall_mask] = WALL

    pair_count = random.randint(*GEN_PAIR_COUNT_RANGE)
    colors = random.sample(PAIR_COLORS, pair_count)
    requirements = [{"need_wall": True, "need_boundary": False}, {"need_wall": False, "need_boundary": True}]
    while len(requirements) < pair_count:
        requirements.append(
            {
                "need_wall": random.random() < 0.55,
                "need_boundary": random.random() < 0.55,
            }
        )
    random.shuffle(requirements)

    pair_specs: list[dict[str, object]] = []
    used_paths: set[tuple[int, int]] = set()
    occupied_endpoints: set[tuple[int, int]] = set()

    for color, requirement in zip(colors, requirements):
        candidate = _find_pair_candidate(inp, int(color), used_paths, occupied_endpoints, requirement)
        if candidate is None:
            return None

        start = tuple(candidate["start"])
        target = tuple(candidate["target"])
        inp[start] = int(color)
        inp[target] = int(color)
        occupied_endpoints.update((start, target))
        used_paths.update(candidate["path"])
        pair_specs.append(candidate)

    has_wall = any(int(spec["wall_bounces"]) > 0 for spec in pair_specs)
    has_boundary = any(int(spec["boundary_bounces"]) > 0 for spec in pair_specs)
    if not has_wall or not has_boundary:
        return None

    expected = _render_expected(inp, pair_specs)
    if not _validate_scene(inp, expected, pair_specs):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid reflected-diagonal ARC input")
