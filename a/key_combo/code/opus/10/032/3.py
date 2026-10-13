from __future__ import annotations

import numpy as np


BG = 0
MARKER = 2
OBSTACLE = 3
FILL = 4

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _adjacent_to_obstacle(grid: np.ndarray, pos: tuple[int, int]) -> bool:
    r, c = pos
    return any(
        _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == OBSTACLE
        for rr, cc in _neighbors4(r, c)
    )


def _border_reachable_bg(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if int(grid[r, c]) == BG and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if int(grid[r, c]) == BG and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors4(r, c):
            if _in_bounds(grid.shape, rr, cc) and not seen[rr, cc] and int(grid[rr, cc]) == BG:
                seen[rr, cc] = True
                stack.append((rr, cc))
    return seen


def _find_marker(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == MARKER)
    assert coords.shape == (1, 2), "expected exactly one red marker pixel"
    return tuple(map(int, coords[0]))


def _find_sightline(grid: np.ndarray, marker: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    hits: list[tuple[tuple[int, int], tuple[int, int]]] = []
    mr, mc = marker

    for dr, dc in DIRS4:
        r, c = mr + dr, mc + dc
        last_bg: tuple[int, int] | None = None
        while _in_bounds(grid.shape, r, c):
            value = int(grid[r, c])
            if value == OBSTACLE:
                if last_bg is not None:
                    hits.append(((dr, dc), last_bg))
                break
            if value != BG:
                break
            last_bg = (r, c)
            r += dr
            c += dc

    assert len(hits) == 1, "expected one unobstructed cardinal sightline to the obstacle"
    direction, start = hits[0]
    assert _adjacent_to_obstacle(grid, start), "sightline endpoint must touch the obstacle"
    return direction, start


def _candidate_paths(grid: np.ndarray, start: tuple[int, int], direction: tuple[int, int]) -> list[list[tuple[int, int]]]:
    dr, dc = direction
    turns = [(dr, -1), (dr, 1)] if dr else [(-1, dc), (1, dc)]
    paths: list[list[tuple[int, int]]] = []

    for tr, tc in turns:
        path = [start]
        r, c = start
        while True:
            nr, nc = r + tr, c + tc
            if not _in_bounds(grid.shape, nr, nc):
                break
            if int(grid[nr, nc]) != BG:
                break
            if not _adjacent_to_obstacle(grid, (nr, nc)):
                break
            path.append((nr, nc))
            r, c = nr, nc
        paths.append(path)
    return paths


def _paint_trail_and_fill(grid: np.ndarray, marker: tuple[int, int], trail: list[tuple[int, int]]) -> np.ndarray:
    before = _border_reachable_bg(grid)

    out = grid.copy()
    out[marker] = BG
    for r, c in trail:
        out[r, c] = MARKER

    after = _border_reachable_bg(out)
    newly_enclosed = before & (out == BG) & ~after
    out[newly_enclosed] = FILL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Leave a red contour trail beside the obstacle, then flood-fill the background region that this new trail closes off.

    Input structure:
    - A black background contains one connected green obstacle with an open concavity and exactly one red marker pixel.
    - The marker lies outside the obstacle and has exactly one horizontal or vertical clear sightline to it through black cells.
    - The black cells remain globally open before the trail is drawn; any closure must be caused by the later red trail.

    Rule:
    - Move from the marker along its unique sightline to the last black cell before the green obstacle.
    - From that cell, try the two contour-following diagonal directions that keep each visited black cell 4-neighbor-adjacent to the obstacle, and choose the strictly longer one.
    - Recolor every cell on the chosen contour walk red, remove the original marker position, then find black cells that were border-reachable before the trail but are no longer border-reachable after it.
    - Flood-fill exactly those newly enclosed black cells yellow, while preserving the green obstacle and the red trail.

    Key insight:
    The red walk is not the final object; it acts as a dynamically added boundary segment whose only purpose is to change background connectivity so a formerly open region becomes enclosed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    marker = _find_marker(g)
    direction, start = _find_sightline(g, marker)
    paths = _candidate_paths(g, start, direction)
    lengths = [len(path) - 1 for path in paths]
    assert lengths[0] != lengths[1], "contour branch lengths must be distinct"
    trail = paths[0] if lengths[0] > lengths[1] else paths[1]
    return _paint_trail_and_fill(g, marker, trail)


def _transform(arr: np.ndarray, rotations: int, flip_ud: bool) -> np.ndarray:
    out = np.rot90(arr, rotations)
    if flip_ud:
        out = np.flipud(out)
    return out.copy()


def _build_canonical_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    long_len = int(rng.integers(4, 7))
    short_len = int(rng.integers(0, min(3, long_len)))
    chamber_h = int(rng.integers(2, 5))
    chamber_w = int(rng.integers(3, 6))
    chamber_shift = int(rng.integers(0, 2))
    top_pad = int(rng.integers(2, 5))
    bottom_pad = int(rng.integers(2, 5))
    left_pad = int(rng.integers(4, 7))
    right_pad = int(rng.integers(3, 5))

    start_r = top_pad + int(rng.integers(1, 3))
    start_c = left_pad + 1
    end_r = start_r + long_len
    end_c = start_c + long_len

    obs_r0 = start_r - top_pad
    obs_c0 = start_c - 1
    obs_r1 = max(end_r + bottom_pad, end_r - chamber_shift + chamber_h + 1)
    obs_c1 = end_c + int(rng.integers(2, 4))

    h = obs_r1 + int(rng.integers(3, 6))
    w = max(obs_c1 + right_pad, end_c + chamber_w + 2)
    if max(h, w) > 30:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[obs_r0:obs_r1, obs_c0:obs_c1] = OBSTACLE

    long_path = [(start_r + i, start_c + i) for i in range(long_len + 1)]
    short_path = [(start_r - i, start_c + i) for i in range(1, short_len + 1)]
    for r, c in long_path + short_path:
        if not _in_bounds(grid.shape, r, c):
            return None
        grid[r, c] = BG

    grid[start_r, : start_c + 1] = BG

    for r, c in long_path[1:]:
        if c + 1 < w:
            grid[r, c + 1] = BG

    chamber_r0 = end_r - chamber_shift
    chamber_c0 = end_c + 1
    chamber_r1 = chamber_r0 + chamber_h
    chamber_c1 = chamber_c0 + chamber_w
    if chamber_r1 > h or chamber_c1 > w:
        return None
    grid[chamber_r0:chamber_r1, chamber_c0:chamber_c1] = BG

    marker = (start_r, 0)
    if int(grid[marker]) != BG:
        return None
    inp = grid.copy()
    inp[marker] = MARKER

    if np.any((grid == BG) & ~_border_reachable_bg(grid)):
        return None

    expected = _paint_trail_and_fill(inp, marker, long_path)
    yellow_count = int(np.sum(expected == FILL))
    if yellow_count < 2:
        return None

    rotations = int(rng.integers(0, 4))
    flip_ud = bool(rng.integers(0, 2))
    inp = _transform(inp, rotations, flip_ud)
    expected = _transform(expected, rotations, flip_ud)

    marker2 = _find_marker(inp)
    direction, start = _find_sightline(inp, marker2)
    paths = _candidate_paths(inp, start, direction)
    lengths = [len(path) - 1 for path in paths]
    if lengths[0] == lengths[1] or max(lengths) < 3:
        return None
    if np.any((inp == BG) & ~_border_reachable_bg(inp)):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        case = _build_canonical_case(rng)
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid input for puzzle 3")
