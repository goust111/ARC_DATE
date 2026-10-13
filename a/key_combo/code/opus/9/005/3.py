import numpy as np
from collections import deque


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
TEAL = 8

NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 300


def _bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _diamond_outline(center, radius):
    cr, cc = center
    pts = []
    for dr in range(-radius, radius + 1):
        dc = radius - abs(dr)
        pts.append((cr + dr, cc + dc))
        if dc != 0:
            pts.append((cr + dr, cc - dc))
    return pts


def _find_seed(grid: np.ndarray):
    coords = np.argwhere(grid == RED)
    assert coords.shape[0] == 1, "expected exactly one red seed"
    return int(coords[0, 0]), int(coords[0, 1])


def _find_diamond(grid: np.ndarray):
    mask = grid == GRAY
    b = _bbox(mask)
    assert b is not None, "expected one gray diamond outline"
    r0, c0, r1, c1 = b
    assert r1 - r0 == c1 - c0, "diamond bbox must be square"
    assert (r1 - r0) % 2 == 0, "diamond bbox side must be odd"
    center = ((r0 + r1) // 2, (c0 + c1) // 2)
    radius = (r1 - r0) // 2

    expected = np.zeros_like(mask, dtype=bool)
    for r, c in _diamond_outline(center, radius):
        expected[r, c] = True
    assert np.array_equal(mask, expected), "gray pixels must form one clean diamond outline"
    return center, radius


def _dilate_diamond(grid: np.ndarray):
    out = np.asarray(grid, dtype=int).copy()
    center, radius = _find_diamond(out)
    for r, c in _diamond_outline(center, radius + 1):
        out[r, c] = GRAY
    return out


def _bfs_distances(grid: np.ndarray, seed):
    h, w = grid.shape
    dist = np.full((h, w), -1, dtype=int)
    q = deque([seed])
    dist[seed] = 0

    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if dist[nr, nc] != -1:
                continue
            if int(grid[nr, nc]) != BLACK:
                continue
            dist[nr, nc] = dist[r, c] + 1
            q.append((nr, nc))

    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the gray diamond so it seals corridor doors, then flood-fill the remaining reachable maze by path-parity.

    Input structure:
    - A black background grid contains teal maze walls that leave one-cell-wide black corridors.
    - One gray Manhattan-diamond outline lies on corridor cells, and one red seed pixel lies in a corridor outside the diamond.
    - The single gray outline is the only gray object and determines the diamond center and radius.

    Rule:
    - Keep the original gray outline and add a second gray outline with the same center and radius + 1, overwriting walls or corridors if needed.
    - Starting from the red seed, run 4-connected BFS through black corridor cells in this modified maze.
    - Recolor reachable corridor cells by shortest-path parity: odd distance becomes green, even positive distance becomes yellow, while the seed stays red.
    - Leave unreachable black corridors black, and leave teal walls plus all gray diamond pixels unchanged.

    Key insight:
    - The gray object is not the final target; its one-step dilation changes maze connectivity first, and only then does path-distance parity determine the flood-fill colors.
    """
    g = np.asarray(grid, dtype=int)
    out = _dilate_diamond(g)
    seed = _find_seed(g)
    dist = _bfs_distances(out, seed)

    for r, c in np.argwhere(dist > 0):
        out[r, c] = GREEN if int(dist[r, c]) % 2 == 1 else YELLOW

    out[seed] = RED
    return out


def _carve_cells(grid: np.ndarray, cells):
    h, w = grid.shape
    for r, c in cells:
        if 0 <= r < h and 0 <= c < w:
            grid[r, c] = BLACK


def _line_cells(start, direction, length):
    r, c = start
    dr, dc = direction
    return [(r + k * dr, c + k * dc) for k in range(length)]


def _try_carve_new_cells(grid: np.ndarray, cells):
    h, w = grid.shape
    new_cells = []
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != TEAL:
            return False
        new_cells.append((r, c))
    _carve_cells(grid, new_cells)
    return True


def _reachable_black(grid: np.ndarray, seed):
    dist = _bfs_distances(grid, seed)
    reachable = {seed}
    for r, c in np.argwhere(dist > 0):
        reachable.add((int(r), int(c)))
    return reachable, dist


def _sample_branch_length(max_len):
    if max_len <= 0:
        return 0
    low = 1 if max_len == 1 else 2
    return int(np.random.randint(low, max_len + 1))


def _add_twigs(grid: np.ndarray, arm_cells, arm_name):
    if len(arm_cells) < 2:
        return

    if arm_name in ("W", "E"):
        branch_dirs = ((-1, 0), (1, 0))
    else:
        branch_dirs = ((0, -1), (0, 1))

    twig_count = int(np.random.randint(1, 4))
    anchor_pool = arm_cells[max(1, len(arm_cells) // 3):]
    if not anchor_pool:
        return

    for _ in range(twig_count):
        ar, ac = anchor_pool[np.random.randint(0, len(anchor_pool))]
        dr, dc = branch_dirs[np.random.randint(0, len(branch_dirs))]

        max_len = 0
        nr, nc = ar + dr, ac + dc
        while 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) == TEAL and max_len < 3:
            max_len += 1
            nr += dr
            nc += dc

        twig_len = _sample_branch_length(max_len)
        if twig_len == 0:
            continue
        _try_carve_new_cells(grid, _line_cells((ar + dr, ac + dc), (dr, dc), twig_len))


def _gen_attempt():
    radius = int(np.random.randint(2, 5))
    room_size = 2 * radius + 3

    min_h = max(19, room_size + 8)
    max_h = min(29, room_size + 13)
    min_w = max(19, room_size + 8)
    max_w = min(29, room_size + 13)
    h = int(np.random.randint(min_h, max_h + 1))
    w = int(np.random.randint(min_w, max_w + 1))

    room_r0 = int(np.random.randint(4, h - room_size - 3))
    room_c0 = int(np.random.randint(4, w - room_size - 3))
    room_r1 = room_r0 + room_size - 1
    room_c1 = room_c0 + room_size - 1
    cr = (room_r0 + room_r1) // 2
    cc = (room_c0 + room_c1) // 2

    grid = np.full((h, w), TEAL, dtype=int)
    grid[room_r0:room_r1 + 1, room_c0:room_c1 + 1] = BLACK

    for r, c in _diamond_outline((cr, cc), radius):
        grid[r, c] = GRAY

    ports = {
        "W": ((cr, room_c0), (0, -1), room_c0 - 1),
        "E": ((cr, room_c1), (0, 1), w - room_c1 - 2),
        "N": ((room_r0, cc), (-1, 0), room_r0 - 1),
        "S": ((room_r1, cc), (1, 0), h - room_r1 - 2),
    }
    valid_names = [name for name, (_, _, max_len) in ports.items() if max_len >= 4]
    if len(valid_names) < 3:
        return None

    active_count = 4 if len(valid_names) == 4 and np.random.rand() < 0.55 else 3
    active_names = list(np.random.choice(valid_names, size=active_count, replace=False))

    arm_cells = {}
    for name in active_names:
        (pr, pc), direction, max_len = ports[name]
        arm_len = int(np.random.randint(3, max_len + 1))
        cells = _line_cells((pr + direction[0], pc + direction[1]), direction, arm_len)
        if not _try_carve_new_cells(grid, cells):
            return None
        arm_cells[name] = cells
        _add_twigs(grid, cells, name)

    seed_side = active_names[int(np.random.randint(0, len(active_names)))]
    seed_candidates = arm_cells[seed_side]
    seed = seed_candidates[int(np.random.randint(0, len(seed_candidates)))]
    grid[seed] = RED

    return grid, (cr, cc), radius, active_names, arm_cells, seed


def _validate_sample(bundle):
    if bundle is None:
        return False

    grid, center, radius, active_names, arm_cells, seed = bundle
    assert int(grid[seed]) == RED
    assert seed not in set(_diamond_outline(center, radius))

    before_reach, before_dist = _reachable_black(grid, seed)
    after_grid = _dilate_diamond(grid)
    after_reach, after_dist = _reachable_black(after_grid, seed)
    if len(before_reach) <= len(after_reach):
        return False

    other_arm_cut = 0
    for name in active_names:
        cells = arm_cells[name]
        before_hits = sum(cell in before_reach for cell in cells)
        after_hits = sum(cell in after_reach for cell in cells)
        if before_hits > 0 and after_hits == 0:
            other_arm_cut += 1
    if other_arm_cut == 0:
        return False

    blocked_delta = before_reach - after_reach
    blocked_delta.discard(seed)
    if len(blocked_delta) < 6:
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if np.count_nonzero(out == RED) != 1:
        return False
    if not np.any(out == GREEN) or not np.any(out == YELLOW):
        return False

    for r, c in np.argwhere(after_dist > 0):
        expected = GREEN if int(after_dist[r, c]) % 2 == 1 else YELLOW
        if int(out[r, c]) != expected:
            return False

    for r, c in blocked_delta:
        if int(after_grid[r, c]) == BLACK and int(out[r, c]) != BLACK:
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        bundle = _gen_attempt()
        if _validate_sample(bundle):
            return bundle[0]
    raise RuntimeError("failed to generate a valid diamond-topology maze input")
