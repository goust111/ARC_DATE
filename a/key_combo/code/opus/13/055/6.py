import random
from collections import deque

import numpy as np


BG = 0
PATH = 2
DIVIDER = 5
WALL = 8
GEN_RETRIES = 400


def _region_slices(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    divider_cols = [c for c in range(w) if np.all(grid[:, c] == DIVIDER)]
    cuts = [-1] + divider_cols + [w]
    regions = []
    for i in range(len(cuts) - 1):
        c0 = cuts[i] + 1
        c1 = cuts[i + 1] - 1
        if c0 <= c1:
            regions.append((0, c0, h, c1 + 1))
    assert len(regions) == 4, "expected four vertical regions"
    widths = {c1 - c0 for _, c0, _, c1 in regions}
    assert len(widths) == 1, "expected equal-width regions"
    return regions


def _extract_template_params(region: np.ndarray) -> tuple[int, int]:
    wall_rows = [r for r in range(region.shape[0]) if np.any(region[r] == WALL)]
    assert len(wall_rows) == 1, "expected one horizontal template wall"
    wall_row = wall_rows[0]
    wall_cols = {int(c) for c in np.where(region[wall_row] == WALL)[0]}
    gap_cols = [c for c in range(region.shape[1]) if c not in wall_cols]
    assert len(gap_cols) == 3 and gap_cols[2] - gap_cols[0] == 2, "expected one 3-cell door gap"
    return wall_row, gap_cols[1]


def _build_local_wall(size: int, wall_row: int, door_center: int) -> np.ndarray:
    wall = np.full((size, size), BG, dtype=int)
    wall[wall_row, :] = WALL
    wall[wall_row, door_center - 1 : door_center + 2] = BG
    return wall


def _describe_wall(region: np.ndarray) -> dict:
    coords = np.argwhere(region == WALL)
    assert coords.size > 0, "expected wall pixels"
    rows = sorted({int(r) for r, _ in coords})
    cols = sorted({int(c) for _, c in coords})
    if len(rows) == 1:
        row = rows[0]
        wall_cols = {int(c) for c in cols}
        gap_cols = [c for c in range(region.shape[1]) if c not in wall_cols]
        assert len(gap_cols) == 3 and gap_cols[2] - gap_cols[0] == 2
        return {"orientation": "h", "wall_row": row, "door_center": gap_cols[1]}
    assert len(cols) == 1, "expected one straight wall"
    col = cols[0]
    wall_rows = {int(r) for r in rows}
    gap_rows = [r for r in range(region.shape[0]) if r not in wall_rows]
    assert len(gap_rows) == 3 and gap_rows[2] - gap_rows[0] == 2
    return {"orientation": "v", "wall_col": col, "door_center": gap_rows[1]}


def _paint_segment(out: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r0, c0 = start
    r1, c1 = end
    assert r0 == r1 or c0 == c1, "path must be orthogonal"
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + step, step):
            if out[r0, c] != WALL:
                out[r0, c] = PATH
        return
    step = 1 if r1 >= r0 else -1
    for r in range(r0, r1 + step, step):
        if out[r, c0] != WALL:
            out[r, c0] = PATH


def _build_filled_maze(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int, int]], list[dict]]:
    regions = _region_slices(grid)
    size = regions[0][3] - regions[0][1]
    template_region = grid[:, regions[0][1] : regions[0][3]]
    wall_row, door_center = _extract_template_params(template_region)
    template_wall = _build_local_wall(size, wall_row, door_center)

    maze = grid.copy()
    descriptors = []
    for idx, (r0, c0, r1, c1) in enumerate(regions):
        wall = np.rot90(template_wall, k=-idx)
        if idx > 0:
            mask = wall == WALL
            sub = maze[r0:r1, c0:c1]
            sub[mask] = WALL
        descriptors.append(_describe_wall(wall))
    return maze, regions, descriptors


def _marker_positions(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    markers = [tuple(map(int, rc)) for rc in np.argwhere(grid == PATH)]
    assert len(markers) == 2, "expected exactly two markers"
    markers.sort(key=lambda rc: (rc[1], rc[0]))
    return markers[0], markers[1]


def _draw_path(maze: np.ndarray, regions: list[tuple[int, int, int, int]], descriptors: list[dict]) -> np.ndarray:
    out = maze.copy()
    start, end = _marker_positions(maze)
    current = start

    for idx, ((r0, c0, _r1, c1), desc) in enumerate(zip(regions, descriptors)):
        if desc["orientation"] == "h":
            wall_r = r0 + desc["wall_row"]
            door_c = c0 + desc["door_center"]
            assert current[0] != wall_r, "current point must not be on the wall row"
            from_above = current[0] < wall_r
            approach_r = wall_r - 1 if from_above else wall_r + 1
            exit_r = wall_r + 1 if from_above else wall_r - 1
            _paint_segment(out, current, (approach_r, current[1]))
            _paint_segment(out, (approach_r, current[1]), (approach_r, door_c))
            _paint_segment(out, (approach_r, door_c), (exit_r, door_c))
            current = (exit_r, door_c)
        else:
            wall_c = c0 + desc["wall_col"]
            door_r = r0 + desc["door_center"]
            assert current[1] != wall_c, "current point must not be on the wall column"
            from_left = current[1] < wall_c
            approach_c = wall_c - 1 if from_left else wall_c + 1
            exit_c = wall_c + 1 if from_left else wall_c - 1
            _paint_segment(out, current, (current[0], approach_c))
            _paint_segment(out, (current[0], approach_c), (door_r, approach_c))
            _paint_segment(out, (door_r, approach_c), (door_r, exit_c))
            current = (door_r, exit_c)

        if idx < len(regions) - 1:
            next_c0 = regions[idx + 1][1]
            _paint_segment(out, current, (current[0], next_c0))
            current = (current[0], next_c0)

    _paint_segment(out, current, (current[0], end[1]))
    _paint_segment(out, (current[0], end[1]), end)
    return out


def _component_count(mask: np.ndarray) -> int:
    seen = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        maze, regions, descriptors = _build_filled_maze(grid)
        out = main(grid)
        if np.array_equal(out, grid) or np.array_equal(out, maze):
            return False
        start, end = _marker_positions(grid)
        if start[0] == end[0]:
            return False

        region_size = regions[0][3] - regions[0][1]
        local_centers = []
        for idx in range(4):
            wall = np.rot90(_build_local_wall(region_size, *_extract_template_params(grid[:, regions[0][1] : regions[0][3]])), k=-idx)
            coords = np.argwhere(wall == WALL)
            if len({int(r) for r, _ in coords}) == 1:
                rr = int(coords[0, 0])
                wall_cols = {int(c) for _, c in coords}
                gap_cols = [c for c in range(region_size) if c not in wall_cols]
                local_centers.append((rr, gap_cols[1]))
            else:
                cc = int(coords[0, 1])
                wall_rows = {int(r) for r, _ in coords}
                gap_rows = [r for r in range(region_size) if r not in wall_rows]
                local_centers.append((gap_rows[1], cc))
        if len(set(local_centers)) != 4:
            return False

        for (r0, c0, r1, c1), desc in zip(regions, descriptors):
            if desc["orientation"] == "h":
                center = (r0 + desc["wall_row"], c0 + desc["door_center"])
            else:
                center = (r0 + desc["door_center"], c0 + desc["wall_col"])
            if out[center] != PATH:
                return False
            region_added = (out[r0:r1, c0:c1] == PATH) & (grid[r0:r1, c0:c1] != PATH)
            if not region_added.any():
                return False

        if np.any((maze == WALL) & (out == PATH)):
            return False
        if _component_count(out == PATH) != 1:
            return False

        added = (out == PATH) & (grid != PATH)
        if int(added.sum()) < 8:
            return False

        naive_same = grid.copy()
        size = region_size
        template_wall = _build_local_wall(size, *_extract_template_params(grid[:, regions[0][1] : regions[0][3]]))
        for idx, (r0, c0, r1, c1) in enumerate(regions[1:], start=1):
            if idx == 2:
                naive_same[r0:r1, c0:c1][template_wall == WALL] = WALL
        if np.array_equal(naive_same, maze):
            return False
        return True
    except Exception:
        return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Generate rotated maze walls from the left template, then draw the red route through each door center.

    Input structure:
    - The grid is split by full gray divider columns into four equal square regions on a black background.
    - The leftmost region contains a one-cell-thick teal wall template: one full-span horizontal wall with a three-cell door gap.
    - The other three regions are empty black slots except for one red marker in the first region and one red marker in the last region.

    Rule:
    - Read the wall template inside the first region and recreate it in later regions with successive clockwise rotations: 90°, 180°, and 270°.
    - Each rotated copy keeps the same one-wall, one-door structure, so horizontal walls become vertical walls and the door position rotates with them.
    - After the four-region maze is completed, draw a red orthogonal path from the left marker to the right marker by entering each region in order, moving to that region's door center, crossing through the center, and continuing into the next region through the gray divider column.

    Key insight:
    - Most of the maze is absent from the input; only by inferring the successive rotations can you predict the later door locations, and those door locations uniquely determine the route.
    """
    g = np.asarray(grid, dtype=int)
    maze, regions, descriptors = _build_filled_maze(g)
    return _draw_path(maze, regions, descriptors)


def _sample_instance(rng: random.Random) -> np.ndarray:
    size = rng.choice((5, 6))
    width = 4 * size + 3
    grid = np.full((size, width), BG, dtype=int)
    for divider_col in (size, 2 * size + 1, 3 * size + 2):
        grid[:, divider_col] = DIVIDER

    region0_c0 = 0
    region3_c0 = 3 * (size + 1)

    while True:
        wall_row = rng.randint(1, size - 2)
        door_center = rng.randint(1, size - 2)
        local_centers = [
            (wall_row, door_center),
            (door_center, size - 1 - wall_row),
            (size - 1 - wall_row, size - 1 - door_center),
            (size - 1 - door_center, wall_row),
        ]
        if len(set(local_centers)) == 4:
            break

    wall = _build_local_wall(size, wall_row, door_center)
    grid[:, region0_c0 : region0_c0 + size] = wall

    start_above = rng.choice((True, False))
    if start_above:
        start_rows = list(range(0, wall_row))
    else:
        start_rows = list(range(wall_row + 1, size))
    start_cols = [c for c in range(size) if c != door_center]
    start = (rng.choice(start_rows), rng.choice(start_cols))
    grid[start] = PATH

    region4_wall_col = wall_row
    region4_door_row = size - 1 - door_center
    end_rows = [r for r in range(size) if r != region4_door_row]
    end_cols = list(range(region4_wall_col + 1, size))
    end = (rng.choice(end_rows), region3_c0 + rng.choice(end_cols))
    grid[end] = PATH
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _sample_instance(random.Random())
        if _validate_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid rotated-door maze input")
