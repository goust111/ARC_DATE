from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
CHANNEL = 8
MARKER = 4
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
MAX_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _feasible_top_lefts(channel_mask: np.ndarray, n: int) -> np.ndarray:
    h, w = channel_mask.shape
    if n < 1 or n > h or n > w:
        return np.zeros((0, 0), dtype=bool)

    sat = np.pad(channel_mask.astype(int), ((1, 0), (1, 0)))
    sat = sat.cumsum(axis=0).cumsum(axis=1)
    window = sat[n:, n:] - sat[:-n, n:] - sat[n:, :-n] + sat[:-n, :-n]
    return window == n * n


def _boundary_starts(feasible: np.ndarray) -> list[tuple[int, int]]:
    if feasible.size == 0:
        return []

    h2, w2 = feasible.shape
    starts = []
    for r, c in np.argwhere(feasible):
        r = int(r)
        c = int(c)
        if r == 0 or c == 0 or r == h2 - 1 or c == w2 - 1:
            starts.append((r, c))
    return starts


def _placement_dist(feasible: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray]:
    starts = _boundary_starts(feasible)
    dist = -np.ones(feasible.shape, dtype=int)
    q = deque()
    for r, c in starts:
        dist[r, c] = 0
        q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if (
                0 <= nr < feasible.shape[0]
                and 0 <= nc < feasible.shape[1]
                and feasible[nr, nc]
                and dist[nr, nc] == -1
            ):
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return starts, dist


def _deepest_reachable_top_left(channel_mask: np.ndarray, n: int) -> tuple[int, int, np.ndarray, np.ndarray]:
    feasible = _feasible_top_lefts(channel_mask, n)
    starts, dist = _placement_dist(feasible)
    assert starts, "expected at least one boundary entry placement"

    best = None
    best_key = None
    for r, c in np.argwhere(dist >= 0):
        r = int(r)
        c = int(c)
        key = (int(dist[r, c]), r, c)
        if best_key is None or key > best_key:
            best_key = key
            best = (r, c)

    assert best is not None, "expected at least one reachable placement"
    return best[0], best[1], feasible, dist


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if (
                _in_bounds(mask.shape, nr, nc)
                and mask[nr, nc]
                and (nr, nc) not in seen
            ):
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _boundary_opening_ok(channel_mask: np.ndarray) -> bool:
    sides = []
    if np.any(channel_mask[0]):
        sides.append(channel_mask[0])
    if np.any(channel_mask[-1]):
        sides.append(channel_mask[-1])
    if np.any(channel_mask[:, 0]):
        sides.append(channel_mask[:, 0])
    if np.any(channel_mask[:, -1]):
        sides.append(channel_mask[:, -1])
    if len(sides) != 1:
        return False

    line = np.asarray(sides[0], dtype=bool)
    runs = 0
    prev = False
    for val in line:
        cur = bool(val)
        if cur and not prev:
            runs += 1
        prev = cur
    return runs == 1


def _count_turns(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(1, len(path) - 1):
        d1 = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        d2 = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if d1 != d2:
            turns += 1
    return turns


def _reachable_capacity(
    rows: int,
    cols: int,
    current: tuple[int, int],
    visited: set[tuple[int, int]],
) -> int:
    seen = {current}
    q = deque([current])
    count = 0
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if nr == 0:
                continue
            if (nr, nc) in seen or (nr, nc) in visited:
                continue
            seen.add((nr, nc))
            count += 1
            q.append((nr, nc))
    return count


def _build_coarse_path(rows: int, cols: int, target_len: int) -> list[tuple[int, int]] | None:
    start_col = random.randrange(cols)
    path = [(0, start_col), (1, start_col)]
    visited = set(path)

    def dfs() -> bool:
        if len(path) == target_len:
            return True

        current = path[-1]
        prev = path[-2]
        prev_dir = (current[0] - prev[0], current[1] - prev[1])
        remaining = target_len - len(path)

        candidates = []
        for dr, dc in random.sample(list(DIR4), len(DIR4)):
            nr = current[0] + dr
            nc = current[1] + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if nr == 0 or (nr, nc) in visited:
                continue

            new_dir = (dr, dc)
            score = random.random()
            if new_dir != prev_dir:
                score += 0.35
            if 0 < nr < rows - 1 and 0 < nc < cols - 1:
                score += 0.15
            candidates.append((score, (nr, nc)))

        candidates.sort(reverse=True)
        for _score, nxt in candidates:
            visited.add(nxt)
            path.append(nxt)
            if _reachable_capacity(rows, cols, nxt, visited) >= remaining - 1 and dfs():
                return True
            path.pop()
            visited.remove(nxt)
        return False

    if not dfs():
        return None

    if _count_turns(path) < 2:
        return None
    if len({r for r, _ in path}) < 3:
        return None
    if len({c for _, c in path}) < 2:
        return None
    return path


def _sample_offsets(n: int, count: int, *, touch_boundary: bool) -> tuple[list[int], int]:
    gaps = [random.randint(n + 1, n + 4) for _ in range(count - 1)]
    start = 0 if touch_boundary else random.randint(1, 3)
    offsets = [start]
    for gap in gaps:
        offsets.append(offsets[-1] + gap)
    margin_far = random.randint(1, 3)
    size = offsets[-1] + n + margin_far
    return offsets, size


def _render_channel(grid: np.ndarray, pixel_path: list[tuple[int, int]], n: int) -> None:
    for pr, pc in pixel_path:
        grid[pr : pr + n, pc : pc + n] = CHANNEL
    for (pr1, pc1), (pr2, pc2) in zip(pixel_path, pixel_path[1:]):
        if pr1 == pr2:
            c0 = min(pc1, pc2)
            c1 = max(pc1, pc2) + n
            grid[pr1 : pr1 + n, c0:c1] = CHANNEL
        else:
            r0 = min(pr1, pr2)
            r1 = max(pr1, pr2) + n
            grid[r0:r1, pc1 : pc1 + n] = CHANNEL


def _add_chokepoint(
    grid: np.ndarray,
    a: tuple[int, int],
    b: tuple[int, int],
    n: int,
) -> None:
    h, w = grid.shape
    pr1, pc1 = a
    pr2, pc2 = b
    narrow = random.randint(1, n - 1)
    remove = n - narrow
    side = random.randint(0, 1)

    if pr1 == pr2:
        c_lo = min(pc1, pc2) + n
        c_hi = max(pc1, pc2)
        hold_len = random.randint(1, min(5, max(1, c_hi - c_lo - 2 * remove)))
        total_len = 2 * remove + hold_len
        c_mid = (c_lo + c_hi) // 2
        c_start = max(c_lo, c_mid - total_len // 2)
        for step in range(total_len):
            c = c_start + step
            if c < 0 or c >= w:
                continue
            if step < remove:
                depth = step + 1
            elif step >= total_len - remove:
                depth = total_len - step
            else:
                depth = remove
            for k in range(depth):
                rr = pr1 + k if side == 0 else pr1 + n - 1 - k
                if 0 <= rr < h:
                    grid[rr, c] = BG
    else:
        r_lo = min(pr1, pr2) + n
        r_hi = max(pr1, pr2)
        hold_len = random.randint(1, min(5, max(1, r_hi - r_lo - 2 * remove)))
        total_len = 2 * remove + hold_len
        r_mid = (r_lo + r_hi) // 2
        r_start = max(r_lo, r_mid - total_len // 2)
        for step in range(total_len):
            r = r_start + step
            if r < 0 or r >= h:
                continue
            if step < remove:
                depth = step + 1
            elif step >= total_len - remove:
                depth = total_len - step
            else:
                depth = remove
            for k in range(depth):
                cc = pc1 + k if side == 0 else pc1 + n - 1 - k
                if 0 <= cc < w:
                    grid[r, cc] = BG


def _place_markers(grid: np.ndarray, n: int) -> bool:
    h, w = grid.shape
    candidates = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if grid[r, c] != BG:
                continue
            if any(grid[r + dr, c + dc] != BG for dr, dc in DIR8):
                continue
            candidates.append((r, c))

    random.shuffle(candidates)
    chosen = []
    for cell in candidates:
        if any(max(abs(cell[0] - rr), abs(cell[1] - cc)) <= 1 for rr, cc in chosen):
            continue
        chosen.append(cell)
        if len(chosen) == n:
            break

    if len(chosen) != n:
        return False

    for r, c in chosen:
        grid[r, c] = MARKER
    return True


def _branch_matches(feasible: np.ndarray, dist: np.ndarray) -> bool:
    return bool(np.any(feasible & (dist < 0)))


def _gen_attempt() -> np.ndarray | None:
    blocked = random.random() < 0.5
    n = random.randint(2, 5) if blocked else random.randint(1, 5)

    coarse_rows = random.randint(3, 5)
    coarse_cols = random.randint(3, 5)
    available = 1 + (coarse_rows - 1) * coarse_cols
    min_len = max(6, coarse_rows + coarse_cols - 1)
    if min_len >= available:
        return None
    target_len = random.randint(min_len, available - 1)

    coarse_path = _build_coarse_path(coarse_rows, coarse_cols, target_len)
    if coarse_path is None:
        return None

    offsets_r, h = _sample_offsets(n, coarse_rows, touch_boundary=True)
    offsets_c, w = _sample_offsets(n, coarse_cols, touch_boundary=False)
    if h > 30 or w > 30:
        return None

    pixel_path = [(offsets_r[r], offsets_c[c]) for r, c in coarse_path]
    grid = np.full((h, w), BG, dtype=int)
    _render_channel(grid, pixel_path, n)

    if blocked:
        start_idx = max(1, len(pixel_path) // 3)
        candidates = list(range(start_idx, len(pixel_path) - 2))
        random.shuffle(candidates)
        carved = False
        for idx in candidates:
            before = grid.copy()
            _add_chokepoint(grid, pixel_path[idx], pixel_path[idx + 1], n)
            channel_mask = grid == CHANNEL
            if _mask_connected(channel_mask):
                carved = True
                break
            grid = before
        if not carved:
            return None

    channel_mask = grid == CHANNEL
    if not _mask_connected(channel_mask):
        return None
    if not _boundary_opening_ok(channel_mask):
        return None

    if not _place_markers(grid, n):
        return None

    k = random.randrange(4)
    if k:
        grid = np.rot90(grid, k)

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None

    _, _, feasible, dist = _deepest_reachable_top_left(grid == CHANNEL, n)
    if _branch_matches(feasible, dist) != blocked:
        return None

    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the yellow markers to size a square, then slide that unseen square into the teal corridor as far as it can go.

    Input structure:
    - A black background contains one teal winding corridor with exactly one opening on the outer border.
    - Several isolated yellow marker pixels sit outside the corridor; no actual square is drawn in the input.

    Rule:
    - Let N be the number of yellow markers.
    - Imagine an N×N solid square entering through the boundary opening and moving one cell at a time only where its full N×N footprint stays teal.
    - Remove the scattered markers, then color the deepest reachable N×N placement inside the corridor yellow while keeping every other cell unchanged.

    Key insight:
    - The marker pixels encode the missing square's size rather than its location, and the first corridor pinch narrower than N determines the square's stopping point.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    n = int(np.sum(g == MARKER))
    assert n >= 1, "expected at least one yellow marker"

    r, c, _feasible, _dist = _deepest_reachable_top_left(g == CHANNEL, n)

    out = g.copy()
    out[out == MARKER] = bg
    out[r : r + n, c : c + n] = MARKER
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for fable/3/054/1")
