import random
from collections import deque

import numpy as np


GEN_RETRIES = 400
ROOM_GRID_OPTIONS = ((2, 3), (2, 4), (3, 3), (3, 4), (4, 3))
ROOM_HEIGHT_RANGE = (5, 7)
ROOM_WIDTH_RANGE = (7, 9)
PATTERN_HEIGHT_MAX = 3
PATTERN_WIDTH_MAX = 5
BACKGROUND = 0
DEFAULT_WALL = 3
PATTERN_COLORS = tuple(c for c in range(10) if c not in (BACKGROUND, DEFAULT_WALL))


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _infer_wall_color(grid: np.ndarray, bg: int) -> int:
    h, w = map(int, grid.shape)
    row_thresh = max(2, (w + 1) // 2)
    col_thresh = max(2, (h + 1) // 2)
    best = None
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        row_hits = sum(int(np.count_nonzero(grid[r] == color) >= row_thresh) for r in range(h))
        col_hits = sum(int(np.count_nonzero(grid[:, c] == color) >= col_thresh) for c in range(w))
        score = row_hits + col_hits
        cand = (score, row_hits, col_hits, color)
        if best is None or cand > best:
            best = cand
    assert best is not None and best[0] > 0, "failed to infer wall color"
    return int(best[3])


def _wall_lines(grid: np.ndarray, wall: int) -> tuple[list[int], list[int]]:
    h, w = map(int, grid.shape)
    row_thresh = max(2, (w + 1) // 2)
    col_thresh = max(2, (h + 1) // 2)
    rows = [r for r in range(h) if np.count_nonzero(grid[r] == wall) >= row_thresh]
    cols = [c for c in range(w) if np.count_nonzero(grid[:, c] == wall) >= col_thresh]
    rows = [r for r in rows if all(grid[r, c] == wall for c in cols)]
    cols = [c for c in cols if all(grid[r, c] == wall for r in rows)]
    assert len(rows) >= 2 and len(cols) >= 2, "expected a rectangular wall lattice"
    return rows, cols


def _extract_rooms(
    grid: np.ndarray, wall_rows: list[int], wall_cols: list[int], bg: int
) -> tuple[list[dict], dict[int, list[int]]]:
    room_rows = len(wall_rows) - 1
    room_cols = len(wall_cols) - 1
    rooms: list[dict] = []
    room_id: dict[tuple[int, int], int] = {}
    for rr in range(room_rows):
        for cc in range(room_cols):
            r0 = int(wall_rows[rr] + 1)
            r1 = int(wall_rows[rr + 1] - 1)
            c0 = int(wall_cols[cc] + 1)
            c1 = int(wall_cols[cc + 1] - 1)
            assert r0 <= r1 and c0 <= c1, "room interior must be non-empty"
            idx = len(rooms)
            room_id[(rr, cc)] = idx
            rooms.append(
                {
                    "grid_pos": (rr, cc),
                    "bbox": (r0, c0, r1, c1),
                    "outer_touches": int(rr == 0)
                    + int(rr == room_rows - 1)
                    + int(cc == 0)
                    + int(cc == room_cols - 1),
                }
            )

    adj = {idx: [] for idx in range(len(rooms))}
    for rr in range(room_rows):
        for cc in range(room_cols - 1):
            wall_c = int(wall_cols[cc + 1])
            r0 = int(wall_rows[rr] + 1)
            r1 = int(wall_rows[rr + 1] - 1)
            if np.any(grid[r0 : r1 + 1, wall_c] == bg):
                a = room_id[(rr, cc)]
                b = room_id[(rr, cc + 1)]
                adj[a].append(b)
                adj[b].append(a)
    for rr in range(room_rows - 1):
        for cc in range(room_cols):
            wall_r = int(wall_rows[rr + 1])
            c0 = int(wall_cols[cc] + 1)
            c1 = int(wall_cols[cc + 1] - 1)
            if np.any(grid[wall_r, c0 : c1 + 1] == bg):
                a = room_id[(rr, cc)]
                b = room_id[(rr + 1, cc)]
                adj[a].append(b)
                adj[b].append(a)
    return rooms, adj


def _choose_start(rooms: list[dict], adj: dict[int, list[int]], active: set[int]) -> int:
    degrees = {idx: sum(int(nb in active) for nb in adj[idx]) for idx in active}
    isolated = [idx for idx, deg in degrees.items() if deg == 0]
    if len(isolated) == 1:
        return int(isolated[0])
    ends = [idx for idx, deg in degrees.items() if deg <= 1]
    assert ends, "room graph must contain an endpoint"
    return max(
        ends,
        key=lambda idx: (
            int(rooms[idx]["outer_touches"]),
            -int(rooms[idx]["grid_pos"][0]),
            -int(rooms[idx]["grid_pos"][1]),
        ),
    )


def _room_order(rooms: list[dict], adj: dict[int, list[int]]) -> list[int]:
    remaining = set(range(len(rooms)))
    order: list[int] = []
    while remaining:
        cur = _choose_start(rooms, adj, remaining)
        while True:
            order.append(int(cur))
            remaining.remove(int(cur))
            unseen = [nb for nb in adj[cur] if nb in remaining]
            if not unseen:
                break
            assert len(unseen) == 1, "room graph must be a linear chain"
            cur = int(unseen[0])
    return order


def _crop_pattern(grid: np.ndarray, room: dict, bg: int, wall: int) -> np.ndarray:
    r0, c0, r1, c1 = map(int, room["bbox"])
    window = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = (window != bg) & (window != wall)
    assert mask.any(), "each room must contain a pattern"
    pr0, pc0, pr1, pc1 = _bbox(mask)
    return window[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()


def _assemble_output(patterns: list[np.ndarray], bg: int) -> np.ndarray:
    max_h = max(int(p.shape[0]) for p in patterns)
    total_w = sum(int(p.shape[1]) for p in patterns) + max(0, len(patterns) - 1)
    out = np.full((max_h, total_w), int(bg), dtype=int)
    cur_c = 0
    for pattern in patterns:
        ph, pw = map(int, pattern.shape)
        out[:ph, cur_c : cur_c + pw] = pattern
        cur_c += pw + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the room order from door gaps in the wall lattice and concatenate the
    cropped room patterns into a single row.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    wall = _infer_wall_color(g, bg)
    wall_rows, wall_cols = _wall_lines(g, wall)
    rooms, adj = _extract_rooms(g, wall_rows, wall_cols, bg)
    order = _room_order(rooms, adj)
    patterns = [_crop_pattern(g, rooms[idx], bg, wall) for idx in order]
    return _assemble_output(patterns, bg)


def _neighbors(cell: tuple[int, int], room_rows: int, room_cols: int):
    r, c = cell
    if r > 0:
        yield (r - 1, c)
    if r + 1 < room_rows:
        yield (r + 1, c)
    if c > 0:
        yield (r, c - 1)
    if c + 1 < room_cols:
        yield (r, c + 1)


def _remaining_connected(
    room_rows: int, room_cols: int, visited: set[tuple[int, int]]
) -> bool:
    all_cells = {
        (r, c) for r in range(room_rows) for c in range(room_cols) if (r, c) not in visited
    }
    if not all_cells:
        return True
    start = next(iter(all_cells))
    seen = {start}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nb in _neighbors(cur, room_rows, room_cols):
            if nb in all_cells and nb not in seen:
                seen.add(nb)
                q.append(nb)
    return len(seen) == len(all_cells)


def _touch_count(cell: tuple[int, int], room_rows: int, room_cols: int) -> int:
    r, c = cell
    return int(r == 0) + int(r == room_rows - 1) + int(c == 0) + int(c == room_cols - 1)


def _hamiltonian_path(
    room_rows: int, room_cols: int, start: tuple[int, int]
) -> list[tuple[int, int]] | None:
    total = room_rows * room_cols
    path = [start]
    visited = {start}

    def search(cur: tuple[int, int]) -> bool:
        if len(path) == total:
            return _touch_count(cur, room_rows, room_cols) < 2
        cands = [nb for nb in _neighbors(cur, room_rows, room_cols) if nb not in visited]
        random.shuffle(cands)
        cands.sort(
            key=lambda cell: sum(
                int(nb not in visited) for nb in _neighbors(cell, room_rows, room_cols)
            )
        )
        for nxt in cands:
            visited.add(nxt)
            path.append(nxt)
            if _remaining_connected(room_rows, room_cols, visited) and search(nxt):
                return True
            path.pop()
            visited.remove(nxt)
        return False

    if search(start):
        return path
    return None


def _pattern(room_h: int, room_w: int) -> np.ndarray:
    ph = random.randint(1, min(PATTERN_HEIGHT_MAX, room_h - 2))
    pw = random.randint(1, min(PATTERN_WIDTH_MAX, room_w - 2))
    n_colors = random.randint(1, min(3, len(PATTERN_COLORS), ph * pw))
    palette = random.sample(list(PATTERN_COLORS), n_colors)
    mode = random.choice(("solid", "rows", "cols", "checker", "noise"))
    arr = np.empty((ph, pw), dtype=int)
    if mode == "solid":
        arr.fill(int(palette[0]))
    elif mode == "rows":
        for r in range(ph):
            arr[r, :] = int(palette[r % n_colors])
    elif mode == "cols":
        for c in range(pw):
            arr[:, c] = int(palette[c % n_colors])
    elif mode == "checker":
        for r in range(ph):
            for c in range(pw):
                arr[r, c] = int(palette[(r + c) % n_colors])
    else:
        arr[:, :] = np.random.choice(np.asarray(palette, dtype=int), size=(ph, pw))
    if n_colors > 1:
        flat = arr.ravel()
        for idx, color in enumerate(palette):
            flat[idx % flat.size] = int(color)
        arr = flat.reshape(arr.shape)
    return arr


def _make_grid(path: list[tuple[int, int]]) -> tuple[np.ndarray, list[np.ndarray]]:
    room_rows = max(r for r, _ in path) + 1
    room_cols = max(c for _, c in path) + 1
    heights = [random.randint(*ROOM_HEIGHT_RANGE) for _ in range(room_rows)]
    widths = [random.randint(*ROOM_WIDTH_RANGE) for _ in range(room_cols)]

    wall_rows = [0]
    for h in heights:
        wall_rows.append(wall_rows[-1] + h + 1)
    wall_cols = [0]
    for w in widths:
        wall_cols.append(wall_cols[-1] + w + 1)

    grid = np.full((wall_rows[-1] + 1, wall_cols[-1] + 1), BACKGROUND, dtype=int)
    grid[wall_rows, :] = DEFAULT_WALL
    grid[:, wall_cols] = DEFAULT_WALL

    for a, b in zip(path, path[1:]):
        if a[0] == b[0]:
            rr = a[0]
            cc = min(a[1], b[1]) + 1
            wall_c = int(wall_cols[cc])
            door_r = random.randint(int(wall_rows[rr] + 1), int(wall_rows[rr + 1] - 1))
            grid[door_r, wall_c] = BACKGROUND
        else:
            rr = min(a[0], b[0]) + 1
            cc = a[1]
            wall_r = int(wall_rows[rr])
            door_c = random.randint(int(wall_cols[cc] + 1), int(wall_cols[cc + 1] - 1))
            grid[wall_r, door_c] = BACKGROUND

    patterns: list[np.ndarray] = []
    for rr, cc in path:
        room_h = heights[rr]
        room_w = widths[cc]
        pat = _pattern(room_h, room_w)
        patterns.append(pat)
        r0 = int(wall_rows[rr] + 1)
        c0 = int(wall_cols[cc] + 1)
        top = random.randint(r0 + 1, r0 + room_h - int(pat.shape[0]) - 1)
        left = random.randint(c0 + 1, c0 + room_w - int(pat.shape[1]) - 1)
        grid[top : top + pat.shape[0], left : left + pat.shape[1]] = pat
    return grid, patterns


def _gen_attempt() -> np.ndarray | None:
    room_rows, room_cols = random.choice(ROOM_GRID_OPTIONS)
    corners = [(0, 0), (0, room_cols - 1), (room_rows - 1, 0), (room_rows - 1, room_cols - 1)]
    random.shuffle(corners)
    path = None
    for start in corners:
        path = _hamiltonian_path(room_rows, room_cols, start)
        if path is not None:
            break
    if path is None or _touch_count(path[-1], room_rows, room_cols) >= 2:
        return None

    grid, patterns = _make_grid(path)
    expected = _assemble_output(patterns, BACKGROUND)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
