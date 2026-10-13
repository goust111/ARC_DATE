import numpy as np

from arc_utils import neighbors4
import random
from collections import deque


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the closest reachable maroon pixels into the 3×3 block around the unique red center.

    Input structure:
    - bg = orange floor, magenta walls, scattered maroon (9) pixels.
    - A unique red (2) pixel marks the center of a 3×3 destination block.

    Rule:
    - Maroon pixels connected to the destination through floor/maroon cells (not blocked by walls) are candidates.
    - Rank candidates by distance to the destination; the closest ones fill the block in row-major order.
    - Moved maroons are erased (replaced with floor color).

    Key insight: Wall barriers restrict connectivity — only maroon pixels reachable via flood-fill through the floor qualify, so proximity alone isn't enough.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    centers = np.argwhere(grid == 2)
    assert len(centers) == 1, "grid must have exactly one red (2) center pixel"
    cr, cc = map(int, centers[0])

    # Destination = 3x3 block around the center.
    dest_cells: list[tuple[int, int]] = []
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if 0 <= r < h and 0 <= c < w:
                dest_cells.append((r, c))
    dest_cells.sort()
    dest_set = set(dest_cells)

    # Infer floor color from the ring around the destination.
    ring_colors: list[int] = []
    for r, c in dest_cells:
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and (rr, cc2) not in dest_set:
                ring_colors.append(int(grid[rr, cc2]))

    cand_ring = [x for x in ring_colors if x not in (2, 3, 9)]
    assert cand_ring, "ring around destination must contain passable (non-maroon, non-red, non-wall) cells to infer floor color"
    _vals, _cnts = np.unique(cand_ring, return_counts=True)
    ring_blocker = int(_vals[_cnts.argmax()])
    other = [int(x) for x in np.unique(grid) if int(x) not in (2, 3, 9, ring_blocker)]
    floor = other[0] if len(other) == 1 else (max(other, key=lambda c: int((grid == c).sum())) if other else None)
    if floor is None:
        floor = ring_blocker

    passable = {floor, 9}

    # BFS start = passable cells adjacent to the destination.
    starts: list[tuple[int, int]] = []
    seen = set()
    for r, c in dest_cells:
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and int(grid[rr, cc2]) in passable:
                if (rr, cc2) not in seen:
                    seen.add((rr, cc2))
                    starts.append((rr, cc2))
    assert starts, "there must be at least one passable cell adjacent to the destination block to begin BFS"

    INF = 10**9
    dist = np.full((h, w), INF, dtype=int)
    q = list(starts)
    for r, c in starts:
        dist[r, c] = 0

    qi = 0
    while qi < len(q):
        r, c = q[qi]
        qi += 1
        d = dist[r, c]
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and int(grid[rr, cc2]) in passable and dist[rr, cc2] == INF:
                dist[rr, cc2] = d + 1
                q.append((rr, cc2))

    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == 9)]
    maroons = [t for t in maroons if t[0] < INF]
    maroons.sort()  # by distance then row then col

    k = min(len(dest_cells), len(maroons))
    chosen = {(r, c) for _, r, c in maroons[:k]}

    for r, c in chosen:
        out[r, c] = floor

    for r, c in dest_cells[:k]:
        out[r, c] = 9

    return out

# --------------- generator (c) ---------------

# ---- CFG: tunable generation parameters ----
GRID_H, GRID_W = 11, 11          # canvas size (fixed per spec)
CENTER_R, CENTER_C = 5, 5        # dest block center
FLOOR_COLOR, WALL_COLOR = 7, 6   # orange floor, magenta wall
DEST_COLOR, RED_CENTER = 3, 2    # green ring, red center
MAROON = 9

WALL_PROB = 0.75                  # probability of placing walls
RING_GAP_RANGE = (1, 4)          # number of corridor openings in wall ring
WALL_STYLES = ['arms', 'frame', 'mixed']

# Wall extension parameters per style
ARM_EXTEND_PROB = 0.5             # per-cell chance of extending an arm (arms style)
ARM_LENGTH_RANGE = (1, 5)         # arm extension length (arms style)
FRAME_DIST_RANGE = (3, 4)        # outer frame distance from center
FRAME_CONNECT_PROB = 0.4          # chance of connecting inner ring to outer frame
FRAME_GAP_REMOVE_PROB = 0.3       # chance of removing an outer frame cell
MIXED_EXTEND_PROB = 0.6           # per-cell chance of extending (mixed style)

FILL_K_RANGE = (2, 9)            # how many maroon fill the dest block
DISTRACTOR_MAX = 10               # max unreachable distractor maroons
TOTAL_MAROON_RANGE = (4, 16)     # total maroon count bounds
MIN_REACHABLE = 2                 # minimum reachable floor cells to accept layout
GEN_RETRIES = 300                 # max generation attempts


def _bfs_reachable(grid, dest_set):
    """BFS from floor cells adjacent to dest through floor cells. Returns set of reachable (r,c)."""
    visited = set()
    queue = deque()
    for dr, dc in dest_set:
        for nr, nc in neighbors4(dr, dc):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                if grid[nr, nc] == FLOOR_COLOR and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in visited:
                if grid[nr, nc] == FLOOR_COLOR and (nr, nc) not in dest_set:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return visited


def _outward_dirs(r, c):
    """Return directions pointing away from center for a ring cell."""
    dirs = []
    if r < CENTER_R: dirs.append((-1, 0))
    if r > CENTER_R: dirs.append((1, 0))
    if c < CENTER_C: dirs.append((0, -1))
    if c > CENTER_C: dirs.append((0, 1))
    return dirs


def _extend_arm(wall_cells, dest_set, r, c, max_len):
    """Extend a wall arm outward from (r,c) in a random outward direction, up to max_len cells."""
    dirs = _outward_dirs(r, c)
    if not dirs:
        return
    dr, dc = random.choice(dirs)
    length = random.randint(1, max_len)
    nr, nc = r, c
    for _ in range(length):
        nr, nc = nr + dr, nc + dc
        if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
            wall_cells.add((nr, nc))
        else:
            break


def _build_walls(grid, dest_set):
    """Build wall barrier around dest with corridor openings, extending outward."""
    # Ring = cells 4-adjacent to dest but not in dest
    ring = set()
    for dr, dc in dest_set:
        for nr, nc in neighbors4(dr, dc):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                ring.add((nr, nc))
    ring = sorted(ring)

    # Choose gaps as corridor openings
    num_gaps = random.randint(*RING_GAP_RANGE)
    gap_cells = set(random.sample(ring, num_gaps))
    wall_ring = [cell for cell in ring if cell not in gap_cells]
    wall_cells = set(wall_ring)

    style = random.choice(WALL_STYLES)

    if style == 'arms':
        for r, c in wall_ring:
            if random.random() < ARM_EXTEND_PROB:
                _extend_arm(wall_cells, dest_set, r, c, ARM_LENGTH_RANGE[1])

    elif style == 'frame':
        # Outer frame at distance 3-4 from center
        dist2 = random.randint(*FRAME_DIST_RANGE)
        for r in range(CENTER_R - dist2, CENTER_R + dist2 + 1):
            for c in range(CENTER_C - dist2, CENTER_C + dist2 + 1):
                if 0 <= r < GRID_H and 0 <= c < GRID_W and (r, c) not in dest_set:
                    if r in (CENTER_R - dist2, CENTER_R + dist2) or c in (CENTER_C - dist2, CENTER_C + dist2):
                        wall_cells.add((r, c))
        # Connect inner ring to outer frame
        for r, c in wall_ring:
            dirs = _outward_dirs(r, c)
            if dirs and random.random() < FRAME_CONNECT_PROB:
                dr, dc = random.choice(dirs)
                nr, nc = r + dr, c + dc
                while 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                    wall_cells.add((nr, nc))
                    nr, nc = nr + dr, nc + dc
        # Remove some outer frame cells for gaps
        outer_frame = [cell for cell in wall_cells if cell not in set(wall_ring)]
        if outer_frame:
            n_remove = random.randint(1, max(1, len(outer_frame) // 4))
            for rc in random.sample(outer_frame, min(n_remove, len(outer_frame))):
                if random.random() < FRAME_GAP_REMOVE_PROB:
                    wall_cells.discard(rc)

    else:  # mixed
        for r, c in wall_ring:
            if random.random() < MIXED_EXTEND_PROB:
                max_len = max(1, GRID_H - abs(r - CENTER_R) - 2)
                _extend_arm(wall_cells, dest_set, r, c, max_len)

    for r, c in wall_cells:
        grid[r, c] = WALL_COLOR


def _gen_attempt():
    """Build a candidate input grid; returns the grid or None on failure."""
    dest_cells = [(r, c) for r in range(CENTER_R - 1, CENTER_R + 2)
                  for c in range(CENTER_C - 1, CENTER_C + 2)]
    dest_set = set(dest_cells)

    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in dest_cells:
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER

    if random.random() < WALL_PROB:
        _build_walls(grid, dest_set)

    reachable_floor = _bfs_reachable(grid, dest_set)
    if len(reachable_floor) < MIN_REACHABLE:
        return None

    all_floor = {(r, c) for r in range(GRID_H) for c in range(GRID_W)
                 if grid[r, c] == FLOOR_COLOR and (r, c) not in dest_set}
    unreachable_floor = all_floor - reachable_floor

    # Choose fill count k
    max_k = min(FILL_K_RANGE[1], len(reachable_floor))
    if max_k < FILL_K_RANGE[0]:
        return None
    k = random.randint(FILL_K_RANGE[0], max_k)

    # Choose distractor count
    max_dist = min(DISTRACTOR_MAX, len(unreachable_floor))
    num_distractors = random.randint(0, max_dist) if max_dist > 0 else 0

    # Enforce total maroon count within bounds
    total = k + num_distractors
    if total < TOTAL_MAROON_RANGE[0]:
        needed = TOTAL_MAROON_RANGE[0] - total
        if len(unreachable_floor) >= num_distractors + needed:
            num_distractors += needed
        elif max_k >= k + needed:
            k += needed
        else:
            return None
    if k + num_distractors > TOTAL_MAROON_RANGE[1]:
        num_distractors = max(0, TOTAL_MAROON_RANGE[1] - k)

    source_cells = random.sample(list(reachable_floor), k)
    distractor_cells = random.sample(list(unreachable_floor), num_distractors)

    # Build INPUT: dest is green+red, maroon on source and distractor positions
    inp = grid.copy()
    for r, c in source_cells + distractor_cells:
        inp[r, c] = MAROON

    # Build expected OUTPUT: first k dest cells (row-major) become maroon, sources erased
    out = grid.copy()
    for r, c in dest_cells[:k]:
        out[r, c] = MAROON
    for r, c in distractor_cells:
        out[r, c] = MAROON

    # Validate with solver
    try:
        result = main(inp)
        if np.array_equal(result, out):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
