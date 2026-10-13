import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
BROWN = 9

PATH_BODY_COLORS = (YELLOW, MAGENTA, AZURE, BROWN)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

FRAME_H_RANGE = (8, 16)
FRAME_W_RANGE = (8, 14)
GAP_RANGE = (1, 4)
PATH_RETRY_LIMIT = 200
GEN_RETRY_LIMIT = 400


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(pts)
    return out


def bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int]] = []
    for pts in components(grid == GREY):
        r0, c0, r1, c1 = bbox(pts)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        expected = set()
        for r in range(r0, r1 + 1):
            expected.add((r, c0))
            expected.add((r, c1))
        for c in range(c0, c1 + 1):
            expected.add((r0, c))
            expected.add((r1, c))
        if set(pts) == expected:
            candidates.append((r0, c0, r1, c1))
    if len(candidates) != 1:
        raise ValueError("expected exactly one grey rectangular frame")
    return candidates[0]


def _trace_path(grid: np.ndarray, frame: tuple[int, int, int, int]) -> tuple[list[tuple[int, int]], list[int]]:
    r0, c0, r1, c1 = frame
    path_cells = {
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if int(grid[r, c]) not in (BLACK, GREY)
    }
    if not path_cells:
        raise ValueError("frame interior must contain a path")

    starts = [pt for pt in path_cells if int(grid[pt]) == RED]
    ends = [pt for pt in path_cells if int(grid[pt]) == BLUE]
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("path must have unique red start and blue end")
    start = starts[0]
    end = ends[0]

    degree: dict[tuple[int, int], int] = {}
    for pt in path_cells:
        degree[pt] = sum((nb in path_cells) for nb in neighbors4(*pt))
    endpoints = sorted(pt for pt, deg in degree.items() if deg == 1)
    if endpoints != sorted([start, end]):
        raise ValueError("path must be a simple non-branching chain")
    for pt, deg in degree.items():
        if pt not in (start, end) and deg != 2:
            raise ValueError("path interior nodes must have degree 2")

    directions: list[tuple[int, int]] = []
    colors: list[int] = []
    visited = {start}
    prev: tuple[int, int] | None = None
    cur = start
    while cur != end:
        nxts = [nb for nb in neighbors4(*cur) if nb in path_cells and nb != prev]
        if len(nxts) != 1:
            raise ValueError("path walk became ambiguous")
        nxt = nxts[0]
        directions.append((nxt[0] - cur[0], nxt[1] - cur[1]))
        colors.append(int(grid[nxt]))
        prev, cur = cur, nxt
        if cur in visited and cur != end:
            raise ValueError("path should not loop")
        visited.add(cur)

    if visited != path_cells:
        raise ValueError("path cells must all belong to the red-blue chain")
    return directions, colors


def _region_bbox(grid: np.ndarray, frame: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, _c0, r1, c1 = frame
    frame_w = c1 - _c0 + 1
    region_c0 = grid.shape[1] - frame_w
    if region_c0 <= c1:
        raise ValueError("right execution area must lie to the right of the frame")
    return r0, region_c0, r1, grid.shape[1] - 1


def _execute(
    grid: np.ndarray,
    region: tuple[int, int, int, int],
    directions: list[tuple[int, int]],
    colors: list[int],
) -> np.ndarray:
    r0, c0, r1, c1 = region
    green_pts = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if int(grid[r, c]) == GREEN]
    if len(green_pts) != 1:
        raise ValueError("right area must contain exactly one green start")
    start = green_pts[0]

    out = grid.copy()
    pos = start
    for (dr, dc), color in zip(directions, colors):
        nr, nc = pos[0] + dr, pos[1] + dc
        blocked = nr < r0 or nr > r1 or nc < c0 or nc > c1 or int(grid[nr, nc]) == ORANGE
        if not blocked:
            pos = (nr, nc)
        out[pos] = color

    out[start] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a colored path inside a grey frame, then replay its directions in the right-hand area.

    The walked-to path colors are used as paint colors. Blocked moves still consume a color but keep
    the cursor in place. The green start marker is restored in the output.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    frame = _frame_bbox(g)
    directions, colors = _trace_path(g, frame)
    region = _region_bbox(g, frame)
    return _execute(g, region, directions, colors)


def _random_induced_path(inner_h: int, inner_w: int, rng: random.Random) -> list[tuple[int, int]]:
    min_len = max(4, min(inner_h, inner_w))
    max_len = max(min_len, min(inner_h * inner_w - 1, 2 * (inner_h + inner_w)))

    for _ in range(PATH_RETRY_LIMIT):
        start = (rng.randrange(inner_h), rng.randrange(inner_w))
        path = [start]
        used = {start}

        while len(path) < max_len:
            cur = path[-1]
            options: list[tuple[int, int]] = []
            for dr, dc in DIRS4:
                nr, nc = cur[0] + dr, cur[1] + dc
                if not (0 <= nr < inner_h and 0 <= nc < inner_w):
                    continue
                if (nr, nc) in used:
                    continue
                occupied_nbrs = [nb for nb in neighbors4(nr, nc) if nb in used]
                if len(occupied_nbrs) == 1 and occupied_nbrs[0] == cur:
                    options.append((nr, nc))
            if not options:
                break

            prev_dir = None
            if len(path) >= 2:
                prev = path[-2]
                prev_dir = (cur[0] - prev[0], cur[1] - prev[1])

            rng.shuffle(options)
            if prev_dir is not None and rng.random() < 0.55:
                straight = [pt for pt in options if (pt[0] - cur[0], pt[1] - cur[1]) == prev_dir]
                nxt = straight[0] if straight else options[0]
            else:
                nxt = options[0]

            path.append(nxt)
            used.add(nxt)

            if len(path) >= min_len and rng.random() < 0.18:
                break

        if len(path) >= min_len:
            keep = rng.randint(min_len, len(path))
            return path[:keep]

    raise RuntimeError("failed to generate a simple path")


def _path_colors(path_len: int, rng: random.Random) -> list[int]:
    colors = [RED]
    for _ in range(path_len - 2):
        colors.append(rng.choice(PATH_BODY_COLORS))
    colors.append(BLUE)
    return colors


def _cumulative_offsets(directions: list[tuple[int, int]]) -> list[tuple[int, int]]:
    offsets = [(0, 0)]
    cr = cc = 0
    for dr, dc in directions:
        cr += dr
        cc += dc
        offsets.append((cr, cc))
    return offsets


def _simulate_local(
    height: int,
    width: int,
    start: tuple[int, int],
    directions: list[tuple[int, int]],
    walls: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], int, int]:
    pos = start
    positions = [start]
    blocked = 0
    success = 0
    for dr, dc in directions:
        nr, nc = pos[0] + dr, pos[1] + dc
        if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in walls:
            pos = (nr, nc)
            success += 1
        else:
            blocked += 1
        positions.append(pos)
    return positions, blocked, success


def _start_without_boundary(height: int, width: int, directions: list[tuple[int, int]], rng: random.Random) -> tuple[int, int]:
    offsets = _cumulative_offsets(directions)
    min_r = min(r for r, _ in offsets)
    max_r = max(r for r, _ in offsets)
    min_c = min(c for _, c in offsets)
    max_c = max(c for _, c in offsets)

    row_lo = -min_r
    row_hi = height - 1 - max_r
    col_lo = -min_c
    col_hi = width - 1 - max_c
    if row_lo > row_hi or col_lo > col_hi:
        raise RuntimeError("path displacements should fit the execution area")
    return rng.randint(row_lo, row_hi), rng.randint(col_lo, col_hi)


def _boundary_block_start(
    height: int,
    width: int,
    directions: list[tuple[int, int]],
    rng: random.Random,
) -> tuple[int, int] | None:
    starts = [(r, c) for r in range(height) for c in range(width)]
    rng.shuffle(starts)
    for start in starts:
        _positions, blocked, success = _simulate_local(height, width, start, directions, set())
        if blocked > 0 and success > 0:
            return start
    return None


def _wall_block_setup(
    height: int,
    width: int,
    directions: list[tuple[int, int]],
    rng: random.Random,
) -> tuple[tuple[int, int], set[tuple[int, int]], list[tuple[int, int]]] | None:
    start = _start_without_boundary(height, width, directions, rng)
    no_wall_positions, _blocked, success = _simulate_local(height, width, start, directions, set())
    if success == 0:
        return None

    candidate_steps = list(range(1, len(no_wall_positions)))
    rng.shuffle(candidate_steps)
    for step_idx in candidate_steps:
        wall = no_wall_positions[step_idx]
        if wall == start:
            continue
        walls = {wall}
        positions, blocked, success = _simulate_local(height, width, start, directions, walls)
        if blocked > 0 and success > 0:
            return start, walls, positions
    return None


def _extra_walls(
    height: int,
    width: int,
    forbidden: set[tuple[int, int]],
    rng: random.Random,
    min_count: int,
    max_count: int,
) -> set[tuple[int, int]]:
    available = [(r, c) for r in range(height) for c in range(width) if (r, c) not in forbidden]
    if not available:
        return set()
    rng.shuffle(available)
    count = min(len(available), rng.randint(min_count, max_count))
    return set(available[:count])


def _assemble_grid(
    frame_h: int,
    frame_w: int,
    gap: int,
    local_path: list[tuple[int, int]],
    path_colors: list[int],
    start: tuple[int, int],
    walls: set[tuple[int, int]],
) -> np.ndarray:
    total_w = frame_w + gap + frame_w
    grid = np.full((frame_h, total_w), BLACK, dtype=int)

    for r in range(frame_h):
        grid[r, 0] = GREY
        grid[r, frame_w - 1] = GREY
    for c in range(frame_w):
        grid[0, c] = GREY
        grid[frame_h - 1, c] = GREY

    for idx, (lr, lc) in enumerate(local_path):
        grid[lr + 1, lc + 1] = path_colors[idx]

    region_c0 = total_w - frame_w
    sr, sc = start
    grid[sr, region_c0 + sc] = GREEN
    for wr, wc in walls:
        grid[wr, region_c0 + wc] = ORANGE
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    frame_h = rng.randint(*FRAME_H_RANGE)
    frame_w = rng.randint(*FRAME_W_RANGE)
    if frame_h < 6 or frame_w < 6:
        return None

    local_path = _random_induced_path(frame_h - 2, frame_w - 2, rng)
    colors = _path_colors(len(local_path), rng)
    directions = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(local_path, local_path[1:])]

    roll = rng.random()
    if roll < 0.5:
        branch_mode = "clear"
    elif roll < 0.75:
        branch_mode = "boundary"
    else:
        branch_mode = "wall"

    if branch_mode == "clear":
        start = _start_without_boundary(frame_h, frame_w, directions, rng)
        positions, blocked, success = _simulate_local(frame_h, frame_w, start, directions, set())
        if blocked != 0 or success == 0:
            return None
        occupied = set(positions)
        walls = _extra_walls(frame_h, frame_w, occupied | {start}, rng, 1, max(1, min(6, frame_h * frame_w // 10)))
        if not walls:
            return None
    elif branch_mode == "boundary":
        start = _boundary_block_start(frame_h, frame_w, directions, rng)
        if start is None:
            return None
        positions, blocked, success = _simulate_local(frame_h, frame_w, start, directions, set())
        if blocked == 0 or success == 0:
            return None
        occupied = set(positions)
        walls = _extra_walls(frame_h, frame_w, occupied | {start}, rng, 1, max(1, min(6, frame_h * frame_w // 10)))
        if not walls:
            return None
    else:
        setup = _wall_block_setup(frame_h, frame_w, directions, rng)
        if setup is None:
            return None
        start, walls, positions = setup
        _positions, blocked, success = _simulate_local(frame_h, frame_w, start, directions, walls)
        if blocked == 0 or success == 0:
            return None
        occupied = set(positions)
        walls |= _extra_walls(frame_h, frame_w, occupied | {start} | walls, rng, 0, max(0, min(4, frame_h * frame_w // 12)))

    if start in walls:
        return None

    gap = rng.randint(*GAP_RANGE)
    grid = _assemble_grid(frame_h, frame_w, gap, local_path, colors, start, walls)

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
