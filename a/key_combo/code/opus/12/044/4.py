from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
WALL_COLORS = (4, 6, 7, 8, 9)
MARKER_COLORS = {BLUE, RED}
SPECIAL_INPUT_COLORS = {BLACK, BLUE, RED}
DIRS8 = [
    (-1, 0),   # up
    (-1, 1),   # up-right
    (0, 1),    # right
    (1, 1),    # down-right
    (1, 0),    # down
    (1, -1),   # down-left
    (0, -1),   # left
    (-1, -1),  # up-left
]

GEN_HEIGHT_RANGE = (14, 19)
GEN_WIDTH_RANGE = (14, 19)
GEN_PERIOD_RANGE = (3, 6)
GEN_TILE_ATTEMPTS = 150
GEN_ERASE_ATTEMPTS = 120
GEN_PAIR_SAMPLES = 260
GEN_ATTEMPTS = 320


def _single_coord(grid: np.ndarray, color: int) -> tuple[int, int]:
    pts = np.argwhere(grid == color)
    assert len(pts) == 1, f"expected exactly one {color}-cell"
    return int(pts[0, 0]), int(pts[0, 1])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _visible_wallpaper_tile(
    grid: np.ndarray, row_period: int, col_period: int
) -> dict[tuple[int, int], int] | None:
    tile: dict[tuple[int, int], int] = {}
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            color = int(grid[r, c])
            if color in SPECIAL_INPUT_COLORS:
                continue
            key = (r % row_period, c % col_period)
            prev = tile.get(key)
            if prev is None:
                tile[key] = color
            elif prev != color:
                return None
    return tile if tile else None


def _restore_wallpaper(grid: np.ndarray) -> tuple[int, int, dict[tuple[int, int], int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, int, dict[tuple[int, int], int]]] = []
    for row_period in range(1, h + 1):
        for col_period in range(1, w + 1):
            tile = _visible_wallpaper_tile(g, row_period, col_period)
            if tile is not None:
                candidates.append((row_period, col_period, tile))
    assert candidates, "no consistent wallpaper period"
    min_area = min(rp * cp for rp, cp, _ in candidates)
    best = [cand for cand in candidates if cand[0] * cand[1] == min_area]
    assert len(best) == 1, "smallest wallpaper period must be unique"
    row_period, col_period, tile = best[0]

    restored = np.zeros_like(g)
    for (rr, cc), color in tile.items():
        restored[rr::row_period, cc::col_period] = color
    return row_period, col_period, tile, restored


def _lex_shortest_path(
    walkable: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]] | None:
    h, w = walkable.shape
    prev_r = np.full((h, w), -1, dtype=int)
    prev_c = np.full((h, w), -1, dtype=int)
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque([start])
    seen[start] = True

    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            break
        for dr, dc in DIRS8:
            nr, nc = r + dr, c + dc
            if not _in_bounds((h, w), nr, nc) or seen[nr, nc] or not walkable[nr, nc]:
                continue
            seen[nr, nc] = True
            prev_r[nr, nc] = r
            prev_c[nr, nc] = c
            q.append((nr, nc))

    if not seen[goal]:
        return None

    path: list[tuple[int, int]] = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = (int(prev_r[cur]), int(prev_c[cur]))
    path.append(start)
    path.reverse()
    return path


def _dist_and_order(walkable: np.ndarray, start: tuple[int, int]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    h, w = walkable.shape
    dist = np.full((h, w), -1, dtype=int)
    dist[start] = 0
    q: deque[tuple[int, int]] = deque([start])
    order: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        order.append((r, c))
        for dr, dc in DIRS8:
            nr, nc = r + dr, c + dc
            if not _in_bounds((h, w), nr, nc) or not walkable[nr, nc] or dist[nr, nc] != -1:
                continue
            dist[nr, nc] = dist[r, c] + 1
            q.append((nr, nc))
    return dist, order


def _count_shortest_paths(
    walkable: np.ndarray, start: tuple[int, int], goal: tuple[int, int], cap: int = 2
) -> int:
    dist, order = _dist_and_order(walkable, start)
    if dist[goal] < 0:
        return 0
    ways = np.zeros(walkable.shape, dtype=int)
    ways[start] = 1
    for r, c in order:
        if ways[r, c] == 0:
            continue
        for dr, dc in DIRS8:
            nr, nc = r + dr, c + dc
            if not _in_bounds(walkable.shape, nr, nc):
                continue
            if dist[nr, nc] == dist[r, c] + 1:
                ways[nr, nc] = min(cap, ways[nr, nc] + ways[r, c])
    return int(ways[goal])


def main(grid: np.ndarray) -> np.ndarray:
    """Restore an erased periodic wallpaper, then connect the blue marker to the red marker through its background gaps.

    Input structure:
    - A black-background grid contains a repeating multi-color wallpaper whose
      non-black cells may be partially erased back to black.
    - Exactly one blue pixel and one red pixel sit on positions that belong to
      the wallpaper's true black background.

    Rule:
    - Ignore the blue and red markers, infer the unique smallest row/column
      wallpaper period consistent with the visible non-black cells, and restore
      the missing wallpaper cells.
    - Treat every restored non-black wallpaper cell as an obstacle, while the
      blue and red marker cells remain walkable.
    - Find a shortest 8-neighbor path from blue to red that stays on black
      background cells; if several shortest paths exist, choose the one whose
      direction-code sequence is lexicographically smallest under the order
      up, up-right, right, down-right, down, down-left, left, up-left.
    - Paint every interior path cell green, keeping the blue and red endpoints.

    Key insight:
    - The erased wallpaper must be completed first because those restored cells
      decide which apparent shortcuts are actually blocked, and the final path
      is the lexicographically smallest shortest route through the remaining
      periodic background corridor network.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _, _, _, wallpaper = _restore_wallpaper(g)
    start = _single_coord(g, BLUE)
    goal = _single_coord(g, RED)
    assert int(wallpaper[start]) == BLACK and int(wallpaper[goal]) == BLACK

    walkable = wallpaper == BLACK
    walkable[start] = True
    walkable[goal] = True
    path = _lex_shortest_path(walkable, start, goal)
    assert path is not None, "markers must be connected through background"

    out = wallpaper.copy()
    for r, c in path[1:-1]:
        out[r, c] = GREEN
    out[start] = BLUE
    out[goal] = RED
    return out


def _has_smaller_period(tile: np.ndarray) -> bool:
    rp, cp = tile.shape
    for p in range(1, rp):
        if np.array_equal(tile, tile[np.arange(rp)[:, None] % p, np.arange(cp)[None, :]]):
            return True
    for q in range(1, cp):
        if np.array_equal(tile, tile[np.arange(rp)[:, None], np.arange(cp)[None, :] % q]):
            return True
    return False


def _wallpaper_from_tile(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    rows = np.arange(h)[:, None] % tile.shape[0]
    cols = np.arange(w)[None, :] % tile.shape[1]
    return tile[rows, cols]


def _connected_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                for dr, dc in DIRS8:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _sample_tile(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray | None:
    h, w = shape
    for _ in range(GEN_TILE_ATTEMPTS):
        rp = int(rng.integers(*GEN_PERIOD_RANGE))
        cp = int(rng.integers(*GEN_PERIOD_RANGE))
        palette_size = int(rng.integers(2, 5))
        palette = rng.choice(WALL_COLORS, size=palette_size, replace=False)
        density = float(rng.uniform(0.33, 0.58))

        tile = np.zeros((rp, cp), dtype=int)
        fg_mask = rng.random((rp, cp)) < density
        if fg_mask.sum() < max(4, (rp * cp) // 4):
            continue
        tile[fg_mask] = rng.choice(palette, size=int(fg_mask.sum()), replace=True)
        if len(set(map(int, tile[fg_mask].tolist()))) < 2:
            continue
        if _has_smaller_period(tile):
            continue

        wallpaper = _wallpaper_from_tile(tile, (h, w))
        bg_ratio = float(np.mean(wallpaper == BLACK))
        if not 0.35 <= bg_ratio <= 0.70:
            continue
        if _connected_count(wallpaper == BLACK) != 1:
            continue
        return tile
    return None


def _sample_erased_cells(
    wallpaper: np.ndarray, row_period: int, col_period: int, rng: np.random.Generator
) -> set[tuple[int, int]] | None:
    fg_coords = [tuple(map(int, rc)) for rc in np.argwhere(wallpaper != BLACK)]
    if len(fg_coords) < 10:
        return None

    residue_to_coords: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in fg_coords:
        residue_to_coords.setdefault((r % row_period, c % col_period), []).append((r, c))

    min_erase = max(4, int(np.ceil(0.08 * len(fg_coords))))
    max_erase = max(min_erase, int(np.floor(0.15 * len(fg_coords))))

    for _ in range(GEN_ERASE_ATTEMPTS):
        target = int(rng.integers(min_erase, max_erase + 1))
        erased: set[tuple[int, int]] = set()
        frontier: list[tuple[int, int]] = []

        while len(erased) < target:
            if not frontier or rng.random() < 0.35:
                cand = fg_coords[int(rng.integers(len(fg_coords)))]
            else:
                br, bc = frontier[int(rng.integers(len(frontier)))]
                neighs = [
                    (nr, nc)
                    for nr, nc in ((br - 1, bc), (br + 1, bc), (br, bc - 1), (br, bc + 1))
                    if _in_bounds(wallpaper.shape, nr, nc) and int(wallpaper[nr, nc]) != BLACK
                ]
                cand = neighs[int(rng.integers(len(neighs)))] if neighs else fg_coords[int(rng.integers(len(fg_coords)))]
            erased.add(cand)
            frontier.append(cand)
            if len(frontier) > 18:
                frontier.pop(0)

        ok = True
        for residue, coords in residue_to_coords.items():
            if all(coord in erased for coord in coords):
                ok = False
                break
        if ok:
            return erased
    return None


def _pair_indices(n: int, rng: np.random.Generator, limit: int) -> list[tuple[int, int]]:
    all_pairs = n * (n - 1) // 2
    if all_pairs <= limit:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(pairs)
        return pairs
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []
    while len(pairs) < limit:
        i, j = rng.choice(n, size=2, replace=False)
        a, b = (int(i), int(j))
        if a > b:
            a, b = b, a
        if (a, b) not in seen:
            seen.add((a, b))
            pairs.append((a, b))
    return pairs


def _find_marker_pair(
    wallpaper: np.ndarray, erased_wallpaper: np.ndarray, rng: np.random.Generator
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]] | None:
    walk_clean = wallpaper == BLACK
    walk_erased = erased_wallpaper == BLACK
    restored_as_obstacle = (wallpaper != BLACK) & (erased_wallpaper == BLACK)
    bg_cells = [tuple(map(int, rc)) for rc in np.argwhere(walk_clean)]
    if len(bg_cells) < 2:
        return None

    best = None
    for i, j in _pair_indices(len(bg_cells), rng, GEN_PAIR_SAMPLES):
        start = bg_cells[i]
        goal = bg_cells[j]
        if abs(start[0] - goal[0]) < 3 or abs(start[1] - goal[1]) < 3:
            continue

        clean_path = _lex_shortest_path(walk_clean, start, goal)
        if clean_path is None or len(clean_path) - 1 < 6:
            continue
        if _count_shortest_paths(walk_clean, start, goal, cap=2) < 2:
            continue

        erased_path = _lex_shortest_path(walk_erased, start, goal)
        if erased_path is None:
            continue
        restored_used = sum(int(restored_as_obstacle[r, c]) for r, c in erased_path[1:-1])
        if restored_used == 0:
            continue
        if len(erased_path) == len(clean_path) and erased_path == clean_path:
            continue

        score = (
            len(clean_path) - len(erased_path),
            restored_used,
            len(clean_path),
            abs(start[0] - goal[0]) + abs(start[1] - goal[1]),
        )
        if best is None or score > best[0]:
            best = (score, start, goal, clean_path, erased_path)

    if best is None:
        return None
    return best[1], best[2], best[3], best[4]


def _oracle_output(
    wallpaper: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> np.ndarray:
    walkable = wallpaper == BLACK
    path = _lex_shortest_path(walkable, start, goal)
    assert path is not None
    out = wallpaper.copy()
    for r, c in path[1:-1]:
        out[r, c] = GREEN
    out[start] = BLUE
    out[goal] = RED
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*GEN_HEIGHT_RANGE))
    w = int(rng.integers(*GEN_WIDTH_RANGE))

    tile = _sample_tile(rng, (h, w))
    if tile is None:
        return None
    wallpaper = _wallpaper_from_tile(tile, (h, w))
    row_period, col_period = tile.shape

    erased_cells = _sample_erased_cells(wallpaper, row_period, col_period, rng)
    if erased_cells is None:
        return None
    erased_wallpaper = wallpaper.copy()
    for r, c in erased_cells:
        erased_wallpaper[r, c] = BLACK

    pair_info = _find_marker_pair(wallpaper, erased_wallpaper, rng)
    if pair_info is None:
        return None
    start, goal, clean_path, erased_path = pair_info

    inp = erased_wallpaper.copy()
    inp[start] = BLUE
    inp[goal] = RED

    try:
        _, _, _, restored = _restore_wallpaper(inp)
    except AssertionError:
        return None
    if not np.array_equal(restored, wallpaper):
        return None

    expected = _oracle_output(wallpaper, start, goal)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    restored_as_obstacle = (wallpaper != BLACK) & (erased_wallpaper == BLACK)
    if not any(restored_as_obstacle[r, c] for r, c in erased_path[1:-1]):
        return None
    if _count_shortest_paths(wallpaper == BLACK, start, goal, cap=2) < 2:
        return None

    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-wallpaper path instance")
