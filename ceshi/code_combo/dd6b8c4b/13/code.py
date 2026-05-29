# Combo variant of dd6b8c4b (index 13)
# combo_spec: D2/0+D16/1
# Name: 多色墙壁分层阻断+背景色限制穿越方向

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4


# Design analysis for the fused variant
# a) Concrete behavior:
#    The old task had one floor color and symmetric BFS. Here the floor is a
#    visible stripe code: even rows are orange and only expand horizontally,
#    odd rows are teal and only expand vertically. Example: from an orange cell
#    at (4,3), moving to (4,2) is legal but moving to (3,3) is not; if row 4
#    contains a blue gate at (4,1), the search can slide horizontally into that
#    gate and then continue vertically from there. Magenta cells still block
#    everything. So the reachable maroon set depends on both where the search is
#    and which color it is currently standing on.
# b) Complexity calibration:
#    The variant uses only four non-token roles: orange horizontal floor,
#    teal vertical floor, magenta solid wall, blue vertical gate-wall. That is
#    enough to make the behavior visibly different from the original task
#    without introducing too many color-to-rule mappings.
# c) Hidden assumptions broken from the original solver:
#    - There is no single floor color anymore, so erased maroons cannot be
#      replaced with one global value.
#    - Reachability is no longer undirected; the current cell color controls the
#      outgoing directions.
#    - One wall color is traversable in one axis, so "non-wall == passable" is
#      not equivalent to "wall == blocked".
#    The solver below removes these assumptions explicitly.
# d) Placement legality:
#    The generator paints the stripe background first, then reserves the 3x3
#    destination block, then places ring barriers, then extends them outward.
#    Maroons are only placed on stripe-floor cells, never on the destination or
#    on wall/gate cells, so erasing a moved token always restores a well-defined
#    visible background color.
# e/f/g) Visual observability and anti-degeneration:
#    The stripe pattern repeats across the whole board, so the two background
#    roles are learnable from only a few examples. Blue gates are sparse
#    vertical columns near the wall structure, so they read as a second wall
#    species rather than random noise. The generator rejects any sample whose
#    output would be unchanged under a relaxed undirected BFS, ensuring the
#    fusion behavior is actually exercised instead of silently collapsing back
#    to the original task.


GRID_H, GRID_W = 11, 11
CENTER_R, CENTER_C = 5, 5

H_FLOOR = 7          # orange: horizontal-only expansion
V_FLOOR = 8          # teal: vertical-only expansion
SOLID_WALL = 6       # magenta: blocks everything
VERT_GATE = 1        # blue: vertical-only gate-wall
DEST_COLOR = 3
RED_CENTER = 2
MAROON = 9

PASSABLE = {H_FLOOR, V_FLOOR, VERT_GATE, MAROON}

RING_OPEN_RANGE = (2, 4)
BLUE_RING_RANGE = (2, 5)
SOLID_ARM_PROB = 0.45
SOLID_ARM_RANGE = (1, 4)
BLUE_SPAN_RANGE = (1, 4)
EXTRA_BLUE_COLS_RANGE = (1, 2)
EXTRA_SOLID_BARS_RANGE = (0, 2)
EXTRA_SOLID_BAR_LEN = (2, 5)

FILL_K_RANGE = (2, 9)
TOTAL_MAROON_RANGE = (4, 16)
DISTRACTOR_MAX = 10
MIN_REACHABLE = 4
GEN_RETRIES = 500

INF = 10**9


def _base_floor_at(r: int, c: int) -> int:
    return H_FLOOR if (r % 2 == 0) else V_FLOOR


def _dest_cells_around(cr: int, cc: int, h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if 0 <= r < h and 0 <= c < w:
                cells.append((r, c))
    cells.sort()
    return cells


def _ring_cells(dest_set: set[tuple[int, int]]) -> list[tuple[int, int]]:
    ring = set()
    for r, c in dest_set:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                ring.add((nr, nc))
    return sorted(ring)


def _effective_color(grid: np.ndarray, r: int, c: int) -> int:
    color = int(grid[r, c])
    if color == MAROON:
        return _base_floor_at(r, c)
    return color


def _out_dirs(grid: np.ndarray, r: int, c: int) -> tuple[tuple[int, int], ...]:
    color = _effective_color(grid, r, c)
    if color == H_FLOOR:
        return ((0, -1), (0, 1))
    if color in (V_FLOOR, VERT_GATE):
        return ((-1, 0), (1, 0))
    return ()


def _directed_dist(grid: np.ndarray, dest_cells: list[tuple[int, int]]) -> np.ndarray:
    h, w = grid.shape
    dest_set = set(dest_cells)
    dist = np.full((h, w), INF, dtype=int)
    q = deque()
    seen = set()
    for r, c in dest_cells:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dest_set and int(grid[nr, nc]) in PASSABLE:
                if (nr, nc) not in seen:
                    seen.add((nr, nc))
                    dist[nr, nc] = 0
                    q.append((nr, nc))
    assert q, "there must be at least one non-solid neighbor around the destination"

    while q:
        r, c = q.popleft()
        step = dist[r, c] + 1
        for dr, dc in _out_dirs(grid, r, c):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dest_set and int(grid[nr, nc]) in PASSABLE:
                if dist[nr, nc] == INF:
                    dist[nr, nc] = step
                    q.append((nr, nc))
    return dist


def _relaxed_dist(grid: np.ndarray, dest_cells: list[tuple[int, int]]) -> np.ndarray:
    h, w = grid.shape
    dest_set = set(dest_cells)
    dist = np.full((h, w), INF, dtype=int)
    q = deque()
    seen = set()
    for r, c in dest_cells:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dest_set and int(grid[nr, nc]) in PASSABLE:
                if (nr, nc) not in seen:
                    seen.add((nr, nc))
                    dist[nr, nc] = 0
                    q.append((nr, nc))

    while q:
        r, c = q.popleft()
        step = dist[r, c] + 1
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dest_set and int(grid[nr, nc]) in PASSABLE:
                if dist[nr, nc] == INF:
                    dist[nr, nc] = step
                    q.append((nr, nc))
    return dist


def _apply_with_dist(
    grid: np.ndarray, dist: np.ndarray, dest_cells: list[tuple[int, int]]
) -> tuple[np.ndarray, set[tuple[int, int]], int]:
    out = grid.copy()
    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == MAROON)]
    maroons = [item for item in maroons if item[0] < INF]
    maroons.sort()
    k = min(len(dest_cells), len(maroons))
    chosen = {(r, c) for _, r, c in maroons[:k]}

    for r, c in chosen:
        out[r, c] = _base_floor_at(r, c)
    for r, c in dest_cells[:k]:
        out[r, c] = MAROON
    return out, chosen, k


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull the nearest direction-reachable maroon tokens into the 3x3 target around the red center.

    Input structure:
    - The destination is still the green 3x3 block with a unique red center.
    - Outside it, the background alternates between orange rows and teal rows.
      Orange cells only spread horizontally; teal cells only spread vertically.
    - Two wall colors appear: magenta is a solid wall, while blue acts like a
      vertical gate that blocks horizontal spread but still allows vertical travel.

    Rule:
    - Start from all non-solid cells adjacent to the destination block.
    - Run a direction-dependent flood-fill: the current cell color decides which
      outgoing directions are legal.
    - Reachable maroon cells are ranked by directed distance, then by row and
      column; the closest ones fill the destination in row-major order.
    - Each moved maroon is erased back to the stripe background color of its
      original row.

    Key insight:
    Reachability is no longer a plain connected-component test. A path can fail
    even when no solid wall blocks it, because orange floor forbids vertical
    expansion, teal/blue cells forbid horizontal expansion, and only the
    interplay between both wall types and both background types reveals which
    maroons can actually reach the target.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must contain exactly one red center"
    cr, cc = map(int, centers[0])
    dest_cells = _dest_cells_around(cr, cc, h, w)
    dist = _directed_dist(grid, dest_cells)
    out, _, _ = _apply_with_dist(grid, dist, dest_cells)
    return out


def _paint_background(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> None:
    for r in range(GRID_H):
        for c in range(GRID_W):
            if (r, c) not in dest_set:
                grid[r, c] = _base_floor_at(r, c)


def _outward_dirs(r: int, c: int) -> list[tuple[int, int]]:
    dirs = []
    if r < CENTER_R:
        dirs.append((-1, 0))
    if r > CENTER_R:
        dirs.append((1, 0))
    if c < CENTER_C:
        dirs.append((0, -1))
    if c > CENTER_C:
        dirs.append((0, 1))
    return dirs


def _extend_blue_column(grid: np.ndarray, dest_set: set[tuple[int, int]], r: int, c: int) -> None:
    up = random.randint(*BLUE_SPAN_RANGE)
    down = random.randint(*BLUE_SPAN_RANGE)
    for rr in range(r - up, r + down + 1):
        if 0 <= rr < GRID_H and (rr, c) not in dest_set and int(grid[rr, c]) != SOLID_WALL:
            grid[rr, c] = VERT_GATE


def _extend_solid_arm(grid: np.ndarray, dest_set: set[tuple[int, int]], r: int, c: int) -> None:
    dirs = _outward_dirs(r, c)
    if not dirs:
        return
    dr, dc = random.choice(dirs)
    length = random.randint(*SOLID_ARM_RANGE)
    nr, nc = r, c
    for _ in range(length):
        nr += dr
        nc += dc
        if not (0 <= nr < GRID_H and 0 <= nc < GRID_W) or (nr, nc) in dest_set:
            break
        if int(grid[nr, nc]) == VERT_GATE:
            break
        grid[nr, nc] = SOLID_WALL


def _add_extra_blue_columns(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> None:
    cols = [c for c in range(GRID_W) if c not in (CENTER_C - 1, CENTER_C, CENTER_C + 1)]
    random.shuffle(cols)
    for c in cols[: random.randint(*EXTRA_BLUE_COLS_RANGE)]:
        top = random.randint(0, GRID_H - 3)
        bottom = random.randint(top + 1, min(GRID_H - 1, top + 5))
        for r in range(top, bottom + 1):
            if (r, c) not in dest_set and int(grid[r, c]) != SOLID_WALL:
                grid[r, c] = VERT_GATE


def _add_extra_solid_bars(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> None:
    for _ in range(random.randint(*EXTRA_SOLID_BARS_RANGE)):
        horizontal = random.random() < 0.5
        length = random.randint(*EXTRA_SOLID_BAR_LEN)
        if horizontal:
            r = random.choice([rr for rr in range(GRID_H) if abs(rr - CENTER_R) >= 2])
            c0 = random.randint(0, GRID_W - length)
            for c in range(c0, c0 + length):
                if (r, c) not in dest_set and int(grid[r, c]) != VERT_GATE:
                    grid[r, c] = SOLID_WALL
        else:
            c = random.choice([cc for cc in range(GRID_W) if abs(cc - CENTER_C) >= 2])
            r0 = random.randint(0, GRID_H - length)
            for r in range(r0, r0 + length):
                if (r, c) not in dest_set and int(grid[r, c]) != VERT_GATE:
                    grid[r, c] = SOLID_WALL


def _build_barriers(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> None:
    ring = _ring_cells(dest_set)
    top_bottom = [(r, c) for r, c in ring if r in (CENTER_R - 2, CENTER_R + 2)]
    left_right = [(r, c) for r, c in ring if c in (CENTER_C - 2, CENTER_C + 2)]

    openings = {
        random.choice(top_bottom),
        random.choice(left_right),
    }
    ring_left = [cell for cell in ring if cell not in openings]
    extra_openings = random.randint(RING_OPEN_RANGE[0] - len(openings), RING_OPEN_RANGE[1] - len(openings))
    if extra_openings > 0:
        openings.update(random.sample(ring_left, extra_openings))
    ring_left = [cell for cell in ring if cell not in openings]

    blue_count = random.randint(BLUE_RING_RANGE[0], min(BLUE_RING_RANGE[1], len(ring_left) - 1))
    blue_ring = set(random.sample(ring_left, blue_count))
    solid_ring = set(ring_left) - blue_ring

    for r, c in solid_ring:
        grid[r, c] = SOLID_WALL
    for r, c in blue_ring:
        grid[r, c] = VERT_GATE

    for r, c in blue_ring:
        if random.random() < 0.8:
            _extend_blue_column(grid, dest_set, r, c)
    for r, c in solid_ring:
        if random.random() < SOLID_ARM_PROB:
            _extend_solid_arm(grid, dest_set, r, c)

    _add_extra_blue_columns(grid, dest_set)
    _add_extra_solid_bars(grid, dest_set)


def _sorted_cells(cells: set[tuple[int, int]], dist: np.ndarray) -> list[tuple[int, int]]:
    return sorted(cells, key=lambda rc: (int(dist[rc[0], rc[1]]), rc[0], rc[1]))


def _select_tokens(
    grid: np.ndarray,
    dest_cells: list[tuple[int, int]],
    directed: np.ndarray,
    relaxed: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], np.ndarray] | None:
    background_cells = {
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if int(grid[r, c]) in (H_FLOOR, V_FLOOR)
    }
    reachable_dir = {rc for rc in background_cells if int(directed[rc[0], rc[1]]) < INF}
    reachable_rel = {rc for rc in background_cells if int(relaxed[rc[0], rc[1]]) < INF}
    relaxed_only = reachable_rel - reachable_dir
    unreachable_all = background_cells - reachable_dir

    if len(reachable_dir) < MIN_REACHABLE or not relaxed_only:
        return None
    if not any(r % 2 == 0 for r, _ in reachable_dir):
        return None
    if not any(r % 2 == 1 for r, _ in reachable_dir):
        return None
    if not np.any(grid == VERT_GATE) or not np.any(grid == SOLID_WALL):
        return None
    if not np.any((grid == VERT_GATE) & (directed < INF)):
        return None

    dir_sorted = _sorted_cells(reachable_dir, directed)
    relaxed_only_sorted = _sorted_cells(relaxed_only, relaxed)
    unreachable_sorted = _sorted_cells(unreachable_all, relaxed)
    max_k = min(FILL_K_RANGE[1], len(dir_sorted))
    if max_k < FILL_K_RANGE[0]:
        return None

    for _ in range(80):
        k = random.randint(FILL_K_RANGE[0], max_k)
        far_start = min(max(1, len(dir_sorted) // 3), len(dir_sorted) - 1)
        source_pool = dir_sorted[far_start:] if len(dir_sorted[far_start:]) >= k else dir_sorted
        source_cells = random.sample(source_pool, k)

        trap_cap = min(3, len(relaxed_only_sorted), DISTRACTOR_MAX)
        trap_count = random.randint(1, trap_cap)
        trap_pool = relaxed_only_sorted[: max(trap_count + 1, min(len(relaxed_only_sorted), 6))]
        trap_cells = random.sample(trap_pool, trap_count)

        used = set(source_cells) | set(trap_cells)
        extra_pool = [rc for rc in unreachable_sorted if rc not in used]
        max_total = min(TOTAL_MAROON_RANGE[1], k + DISTRACTOR_MAX, k + trap_count + len(extra_pool))
        min_total = max(TOTAL_MAROON_RANGE[0], k + trap_count)
        if min_total > max_total:
            continue
        total = random.randint(min_total, max_total)
        extra_count = total - k - trap_count
        extra_cells = random.sample(extra_pool, extra_count) if extra_count > 0 else []

        inp = grid.copy()
        for r, c in source_cells + trap_cells + extra_cells:
            inp[r, c] = MAROON

        directed_out, _, _ = _apply_with_dist(inp, directed, dest_cells)
        relaxed_out, _, _ = _apply_with_dist(inp, relaxed, dest_cells)
        if not np.array_equal(directed_out, relaxed_out):
            return source_cells, trap_cells + extra_cells, directed_out
    return None


def _gen_attempt() -> np.ndarray | None:
    dest_cells = _dest_cells_around(CENTER_R, CENTER_C, GRID_H, GRID_W)
    dest_set = set(dest_cells)

    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    _paint_background(grid, dest_set)
    for r, c in dest_cells:
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER

    _build_barriers(grid, dest_set)

    directed = _directed_dist(grid, dest_cells)
    relaxed = _relaxed_dist(grid, dest_cells)
    selected = _select_tokens(grid, dest_cells, directed, relaxed)
    if selected is None:
        return None
    source_cells, distractor_cells, expected = selected

    inp = grid.copy()
    for r, c in source_cells + distractor_cells:
        inp[r, c] = MAROON

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
