from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_MAX_ATTEMPTS = 1200


def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue

            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(coords)

    return components


def _parse_instance(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "grid must fit ARC limits"
    assert set(map(int, np.unique(g))) <= {BG, BLUE, RED, GREEN}, "unexpected input colors"

    rect_mask = (g == BLUE) | (g == RED)
    rect_coords = np.argwhere(rect_mask)
    assert rect_coords.size > 0, "expected one blue rectangle with red markers"

    r0, c0 = map(int, rect_coords.min(axis=0))
    r1, c1 = map(int, rect_coords.max(axis=0))
    rect = g[r0 : r1 + 1, c0 : c1 + 1]
    rect_h, rect_w = rect.shape
    assert rect_h >= 5 and rect_w >= 5, "rectangle must leave room for an inner wall area"
    assert np.all((rect == BLUE) | (rect == RED)), "rectangle bbox must contain only blue base and red markers"
    assert int(np.count_nonzero(rect_mask)) == rect_h * rect_w, "blue/red area must be one full rectangle"

    markers = [tuple(map(int, rc)) for rc in np.argwhere(rect == RED)]
    assert 2 <= len(markers) <= 4, "expected 2 to 4 markers"
    marker_rows = [r for r, _c in markers]
    marker_cols = [c for _r, c in markers]
    assert len(set(marker_rows)) == len(marker_rows), "markers must occupy distinct rows"
    assert len(set(marker_cols)) == len(marker_cols), "markers must occupy distinct columns"
    assert all(0 < r < rect_h - 1 and 0 < c < rect_w - 1 for r, c in markers), "markers must stay off the border"

    frame_components = _components4(g == GREEN)
    assert len(frame_components) == 2, "expected exactly two doorway frames"

    doors: dict[str, dict[str, tuple[int, int]]] = {}
    for component in frame_components:
        coords = np.asarray(component, dtype=int)
        fr0, fc0 = map(int, coords.min(axis=0))
        fr1, fc1 = map(int, coords.max(axis=0))
        assert (fr1 - fr0 + 1, fc1 - fc0 + 1) == (3, 3), "each frame must occupy a 3x3 box"

        if fc1 < c0:
            side = "left"
            gap = (fr0 + 1, fc1)
            boundary = (gap[0], c0)
            expected = np.full((3, 3), GREEN, dtype=int)
            expected[1, 2] = BG
            assert np.array_equal(g[fr0 : fr1 + 1, fc0 : fc1 + 1], expected), "left frame doorway must face right"
            assert np.all(g[gap[0], gap[1] : boundary[1]] == BG), "left doorway corridor must be black"
        elif fc0 > c1:
            side = "right"
            gap = (fr0 + 1, fc0)
            boundary = (gap[0], c1)
            expected = np.full((3, 3), GREEN, dtype=int)
            expected[1, 0] = BG
            assert np.array_equal(g[fr0 : fr1 + 1, fc0 : fc1 + 1], expected), "right frame doorway must face left"
            assert np.all(g[gap[0], boundary[1] + 1 : gap[1] + 1] == BG), "right doorway corridor must be black"
        else:
            raise AssertionError("frames must sit strictly to the left and right of the rectangle")

        assert r0 <= boundary[0] <= r1, "doorway must line up with a rectangle row"
        assert g[boundary] == BLUE, "exterior doorway cell must be blue"
        assert side not in doors, "expected one frame per side"
        doors[side] = {"gap": gap, "boundary": boundary}

    assert set(doors) == {"left", "right"}, "missing a left or right frame"
    return {
        "grid": g,
        "rect_bounds": (r0, c0, r1, c1),
        "rect": rect,
        "markers": markers,
        "doors": doors,
    }


def _expand_walls(rect_shape: tuple[int, int], markers: list[tuple[int, int]]) -> np.ndarray:
    height, width = rect_shape
    maze = np.full((height, width), BLUE, dtype=int)
    for r, c in markers:
        maze[r, 1 : width - 1] = RED
        maze[1 : height - 1, c] = RED
    return maze


def _reachable(
    maze: np.ndarray,
    seed: tuple[int, int],
    moves: list[tuple[int, int]],
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    r0, c0, r1, c1 = bounds

    def valid(r: int, c: int) -> bool:
        return r0 <= r <= r1 and c0 <= c <= c1 and maze[r, c] == BLUE

    assert valid(*seed), "seed must be a passable cell"
    seen = np.zeros_like(maze, dtype=bool)
    seen[seed] = True
    queue = deque([seed])
    while queue:
        r, c = queue.popleft()
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if valid(nr, nc) and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return seen


def _path_union(maze: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> np.ndarray:
    row_step = _sign(end[0] - start[0])
    assert end[1] > start[1], "this puzzle always routes from the left doorway to the right doorway"

    moves = [(0, 1)]
    if row_step:
        moves.append((row_step, 0))

    bounds = (min(start[0], end[0]), start[1], max(start[0], end[0]), end[1])
    forward = _reachable(maze, start, moves, bounds)
    backward = _reachable(maze, end, [(-dr, -dc) for dr, dc in moves], bounds)
    path = forward & backward
    assert path[start] and path[end], "expected at least one valid monotone route"
    return path


def _analyze(grid: np.ndarray) -> dict[str, object]:
    info = _parse_instance(grid)
    rect = info["rect"]
    rect_h, rect_w = rect.shape
    markers = info["markers"]
    doors = info["doors"]
    r0, c0, _r1, _c1 = info["rect_bounds"]

    maze = _expand_walls((rect_h, rect_w), markers)
    start_global = doors["left"]["boundary"]
    end_global = doors["right"]["boundary"]
    start = (start_global[0] - r0, 0)
    end = (end_global[0] - r0, rect_w - 1)
    path = _path_union(maze, start, end)

    manhattan = abs(end[0] - start[0]) + (end[1] - start[1])
    plain_area = (abs(end[0] - start[0]) + 1) * rect_w
    return {
        **info,
        "maze": maze,
        "path": path,
        "start": start,
        "end": end,
        "path_len": manhattan,
        "union_size": int(np.count_nonzero(path)),
        "plain_area": int(plain_area),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand sparse red markers into interior row/column walls and paint the monotone doorway path union.

    Input structure:
    - A black canvas contains one solid blue rectangle with 2 to 4 red marker pixels strictly inside it.
    - A green 3x3 doorway frame sits to the left of the rectangle and another sits to the right; each frame has one black side gap facing the rectangle.
    - The blue rectangle and red markers together define the maze region, while the green frames only identify the two doorway rows.

    Rule:
    - Read the blue rectangle as a cropped maze canvas.
    - For every red marker, paint its full interior row and full interior column red, leaving the rectangle border blue as the perimeter corridor.
    - The blue boundary cell aligned with each frame gap is that side's doorway cell.
    - Inside the rectangle, treat red cells as walls and paint yellow every blue cell that lies on some shortest monotone left-to-right path from the left doorway cell to the right doorway cell.
    - Also paint the two black frame-gap doorway pixels yellow; keep every other pixel unchanged.

    Key insight:
    - Each marker simultaneously creates one horizontal and one vertical wall, so a few sparse markers carve a constrained monotone maze whose solution is the union of all equal-length valid routes.
    """
    g = np.asarray(grid, dtype=int)
    solved = _analyze(g)

    out = g.copy()
    r0, c0, _r1, _c1 = solved["rect_bounds"]
    path_coords = np.argwhere(solved["path"])
    for rr, cc in path_coords:
        out[r0 + int(rr), c0 + int(cc)] = YELLOW

    for side in ("left", "right"):
        gap_r, gap_c = solved["doors"][side]["gap"]
        out[gap_r, gap_c] = YELLOW

    return out


def _build_candidate(
    rect_h: int,
    rect_w: int,
    rect_r0: int,
    rect_c0: int,
    left_row: int,
    right_row: int,
    left_extra: int,
    right_extra: int,
    markers: list[tuple[int, int]],
    grid_shape: tuple[int, int],
) -> np.ndarray:
    height, width = grid_shape
    grid = np.full((height, width), BG, dtype=int)
    grid[rect_r0 : rect_r0 + rect_h, rect_c0 : rect_c0 + rect_w] = BLUE

    for r, c in markers:
        grid[rect_r0 + r, rect_c0 + c] = RED

    left_gap_r = rect_r0 + left_row
    left_gap_c = rect_c0 - left_extra - 1
    left_center_c = left_gap_c - 1
    grid[left_gap_r - 1 : left_gap_r + 2, left_center_c - 1 : left_center_c + 2] = GREEN
    grid[left_gap_r, left_gap_c] = BG

    right_gap_r = rect_r0 + right_row
    right_gap_c = rect_c0 + rect_w + right_extra
    right_center_c = right_gap_c + 1
    grid[right_gap_r - 1 : right_gap_r + 2, right_center_c - 1 : right_center_c + 2] = GREEN
    grid[right_gap_r, right_gap_c] = BG
    return grid


def _sample_marker_positions(rng: np.random.Generator, rect_h: int, rect_w: int) -> list[tuple[int, int]]:
    count = int(rng.integers(2, 5))
    rows = rng.choice(np.arange(1, rect_h - 1), size=count, replace=False)
    cols = rng.choice(np.arange(1, rect_w - 1), size=count, replace=False)
    cols = cols[rng.permutation(count)]
    return [(int(r), int(c)) for r, c in zip(rows, cols)]


def _sample_rows(rng: np.random.Generator, rect_h: int) -> tuple[int, int]:
    descending = bool(rng.integers(0, 2))
    if descending:
        left = int(rng.integers(1, rect_h))
        right = int(rng.integers(0, left))
    else:
        left = int(rng.integers(0, rect_h - 1))
        right = int(rng.integers(left + 1, rect_h))
    return left, right


def _is_good_instance(analysis: dict[str, object], wanted_edge: str) -> bool:
    path = analysis["path"]
    union_size = int(analysis["union_size"])
    plain_area = int(analysis["plain_area"])
    touches_top = bool(path[0].any())
    touches_bottom = bool(path[-1].any())

    if union_size >= plain_area:
        return False
    if not np.any(path[1:-1, 1:-1]):
        return False
    if wanted_edge == "top":
        return touches_top and not touches_bottom
    if wanted_edge == "bottom":
        return touches_bottom and not touches_top
    raise AssertionError(f"unexpected route mode: {wanted_edge}")


def _gen_attempt(rng: np.random.Generator, wanted_edge: str, strict: bool) -> np.ndarray | None:
    rect_h = int(rng.integers(8, 14))
    rect_w = int(rng.integers(9, 15))
    left_extra = int(rng.integers(0, 3))
    right_extra = int(rng.integers(0, 3))

    top_margin = int(rng.integers(2, 6))
    bottom_margin = int(rng.integers(2, 6))
    outer_left = int(rng.integers(1, 4))
    outer_right = int(rng.integers(1, 4))

    rect_r0 = top_margin
    rect_c0 = outer_left + left_extra + 3
    height = rect_h + top_margin + bottom_margin
    width = rect_c0 + rect_w + right_extra + 3 + outer_right
    if height > 30 or width > 30:
        return None

    left_row, right_row = _sample_rows(rng, rect_h)
    markers = _sample_marker_positions(rng, rect_h, rect_w)
    candidate = _build_candidate(
        rect_h,
        rect_w,
        rect_r0,
        rect_c0,
        left_row,
        right_row,
        left_extra,
        right_extra,
        markers,
        (height, width),
    )

    try:
        analysis = _analyze(candidate)
    except AssertionError:
        return None

    if strict and not _is_good_instance(analysis, wanted_edge):
        return None
    if not strict:
        if int(analysis["union_size"]) >= int(analysis["plain_area"]):
            return None
        if not np.any(analysis["path"][1:-1, 1:-1]):
            return None

    expected = main(candidate)
    if expected.shape != candidate.shape:
        return None
    return candidate


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for attempt in range(GEN_MAX_ATTEMPTS):
        wanted_edge = "top" if bool(rng.integers(0, 2)) else "bottom"
        strict = attempt < GEN_MAX_ATTEMPTS * 3 // 4
        grid = _gen_attempt(rng, wanted_edge=wanted_edge, strict=strict)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid maze-doorway input")
