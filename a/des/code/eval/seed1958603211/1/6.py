import heapq
import random
from collections import Counter

import numpy as np


BG = 0
PIPE = 5
MARK = 9
ENTRY_COLORS = [1, 2, 3, 4, 6, 7, 8]
MAX_GENERATION_ATTEMPTS = 400

DIR_E = 0
DIR_N = 1
DIR_S = 2
STEPS = {
    DIR_E: (0, 1),
    DIR_N: (-1, 0),
    DIR_S: (1, 0),
}


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _parse_structure(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    right_colors = {int(g[r, w - 1]) for r in range(h) if int(g[r, w - 1]) != bg}
    assert len(right_colors) == 1, "expected one pipe/exit color on the right edge"
    pipe = int(next(iter(right_colors)))

    entries = [(r, int(g[r, 0])) for r in range(h) if int(g[r, 0]) not in {bg, pipe}]
    entries.sort()
    assert len(entries) >= 1, "expected at least one left-edge entry"
    assert len({color for _, color in entries}) == len(entries), "entry colors must be distinct"

    entry_mask = np.zeros_like(g, dtype=bool)
    for r, _color in entries:
        entry_mask[r, 0] = True

    return bg, pipe, entries, entry_mask


def _priority_at(g: np.ndarray, pipe_like: np.ndarray, pipe: int, r: int, c: int) -> str | None:
    h, w = g.shape
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return None

    left = bool(pipe_like[r, c - 1])
    right = int(g[r, c + 1]) == pipe
    up = int(g[r - 1, c]) == pipe
    down = int(g[r + 1, c]) == pipe
    if not (left and right and up and down):
        return None

    ul = c > 1 and int(g[r - 1, c - 1]) not in {BG, pipe}
    ur = c < w - 2 and int(g[r - 1, c + 1]) not in {BG, pipe}
    if ul == ur:
        return None
    return "V" if ul else "H"


def _next_dirs(g: np.ndarray, pipe_like: np.ndarray, pipe: int, r: int, c: int, direction: int) -> list[int]:
    h, w = g.shape
    right = c + 1 < w and int(g[r, c + 1]) == pipe
    up = r > 0 and int(g[r - 1, c]) == pipe
    down = r + 1 < h and int(g[r + 1, c]) == pipe
    priority = _priority_at(g, pipe_like, pipe, r, c)

    if direction == DIR_E:
        if priority == "H":
            return [DIR_E] if right else []
        if priority == "V":
            return ([DIR_N] if up else []) + ([DIR_S] if down else [])
        out = []
        if right:
            out.append(DIR_E)
        if up:
            out.append(DIR_N)
        if down:
            out.append(DIR_S)
        return out

    if direction == DIR_N:
        if priority == "H":
            return [DIR_E] if right else []
        if priority == "V":
            return [DIR_N] if up else []
        out = []
        if up:
            out.append(DIR_N)
        if right:
            out.append(DIR_E)
        return out

    if priority == "H":
        return [DIR_E] if right else []
    if priority == "V":
        return [DIR_S] if down else []
    out = []
    if down:
        out.append(DIR_S)
    if right:
        out.append(DIR_E)
    return out


def _solve_with_trace(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg, pipe, entries, entry_mask = _parse_structure(g)

    pipe_like = (g == pipe) | entry_mask
    dist = np.full((h, w), 10 ** 9, dtype=int)
    owner = np.full((h, w), -1, dtype=int)
    seen_dirs = [[set() for _ in range(w)] for _ in range(h)]

    pq: list[tuple[int, int, int, int, int]] = []
    for sid, (r, _color) in enumerate(entries):
        heapq.heappush(pq, (0, sid, r, 0, DIR_E))

    while pq:
        steps, sid, r, c, direction = heapq.heappop(pq)
        if steps > dist[r, c]:
            continue
        if steps == dist[r, c] and owner[r, c] != -1 and sid > owner[r, c]:
            continue
        if steps < dist[r, c] or sid < owner[r, c]:
            dist[r, c] = steps
            owner[r, c] = sid
            seen_dirs[r][c].clear()
        if sid != owner[r, c] or direction in seen_dirs[r][c]:
            continue
        seen_dirs[r][c].add(direction)

        for next_dir in _next_dirs(g, pipe_like, pipe, r, c, direction):
            dr, dc = STEPS[next_dir]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if int(g[nr, nc]) != pipe:
                continue
            heapq.heappush(pq, (steps + 1, sid, nr, nc, next_dir))

    out = g.copy()
    for sid, (_r, color) in enumerate(entries):
        pass
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) == pipe and owner[r, c] != -1:
                out[r, c] = entries[int(owner[r, c])][1]
    return out, owner, entries


def main(grid: np.ndarray) -> np.ndarray:
    """Propagate left-edge colors through an orthogonal pipe network with marked cross priorities."""
    out, _owner, _entries = _solve_with_trace(grid)
    return out


def _full_crosses(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    crosses = []
    for r in range(2, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != PIPE:
                continue
            if not (
                int(grid[r, c - 1]) == PIPE
                and int(grid[r, c + 1]) == PIPE
                and int(grid[r - 1, c]) == PIPE
                and int(grid[r + 1, c]) == PIPE
            ):
                continue
            if c <= 1 or c >= w - 2:
                continue
            if int(grid[r - 1, c - 1]) != BG or int(grid[r - 1, c + 1]) != BG:
                continue
            crosses.append((r, c))
    return crosses


def _pick_rows(rng: random.Random, h: int, count: int) -> list[int]:
    lanes = list(range(2, h - 1, 2))
    return sorted(rng.sample(lanes, count))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(13, 21)
    w = rng.randint(22, 30)
    lane_rows = list(range(2, h - 1, 2))
    if len(lane_rows) < 3:
        return None

    max_entries = min(4, len(lane_rows) - 1)
    if max_entries < 2:
        return None
    n_entries = rng.randint(2, max_entries)

    min_rows = max(3, n_entries)
    max_rows = min(len(lane_rows), n_entries + 2)
    if min_rows > max_rows:
        return None
    n_active_rows = rng.randint(min_rows, max_rows)
    active_rows = _pick_rows(rng, h, n_active_rows)
    entry_rows = sorted(rng.sample(active_rows, n_entries))
    inactive_rows = [r for r in active_rows if r not in entry_rows]

    col_candidates = list(range(3, w - 5, 3))
    need_cols = len(inactive_rows) + 1
    if len(col_candidates) < need_cols:
        return None
    picked_cols = sorted(rng.sample(col_candidates, need_cols))
    activation_cols = picked_cols[: len(inactive_rows)]
    long_col = picked_cols[-1]
    if activation_cols and long_col <= max(activation_cols):
        return None

    start_col = {row: 1 for row in entry_rows}
    connectors: list[tuple[int, int, int]] = []
    active_now = set(entry_rows)
    pending_rows = inactive_rows[:]
    rng.shuffle(pending_rows)
    for col, target_row in zip(activation_cols, pending_rows):
        sources = sorted(active_now)
        source_row = rng.choice(sources)
        connectors.append((col, source_row, target_row))
        start_col[target_row] = col
        active_now.add(target_row)

    connectors.append((long_col, active_rows[0], active_rows[-1]))

    exit_rows = set(entry_rows)
    non_entry_rows = [row for row in active_rows if row not in entry_rows]
    max_swaps = min(len(non_entry_rows), max(1, n_entries // 2))
    if max_swaps and rng.random() < 0.7:
        swaps = rng.randint(1, max_swaps)
        out_rows = rng.sample(entry_rows, swaps)
        in_rows = rng.sample(non_entry_rows, swaps)
        exit_rows.difference_update(out_rows)
        exit_rows.update(in_rows)
    exit_rows = sorted(exit_rows)

    touches: dict[int, set[int]] = {row: {start_col[row]} for row in active_rows}
    for col, r0, r1 in connectors:
        lo, hi = sorted((r0, r1))
        for row in active_rows:
            if lo <= row <= hi and start_col[row] <= col:
                touches[row].add(col)

    end_col: dict[int, int] = {}
    for row in active_rows:
        min_end = max(max(touches[row]) + 1, start_col[row] + 2)
        if row in exit_rows:
            end_col[row] = w - 1
        else:
            if min_end > w - 3:
                return None
            end_col[row] = rng.randint(min_end, w - 3)

    grid = np.full((h, w), BG, dtype=int)
    for row in active_rows:
        grid[row, start_col[row] : end_col[row] + 1] = PIPE
    for col, r0, r1 in connectors:
        lo, hi = sorted((r0, r1))
        grid[lo : hi + 1, col] = PIPE

    for row, color in zip(entry_rows, rng.sample(ENTRY_COLORS, n_entries)):
        grid[row, 0] = color

    crosses = _full_crosses(grid)
    if not crosses:
        return None

    mark_kinds: dict[tuple[int, int], str] = {}
    for cross in crosses:
        mark_kinds[cross] = rng.choice(["none", "H", "V"])

    if len(crosses) >= 2:
        kinds = set(mark_kinds.values())
        if "none" not in kinds:
            mark_kinds[rng.choice(crosses)] = "none"
        if "H" not in kinds:
            mark_kinds[rng.choice(crosses)] = "H"
        if "V" not in kinds:
            mark_kinds[rng.choice(crosses)] = "V"

    for (r, c), kind in mark_kinds.items():
        if kind == "H":
            grid[r - 1, c + 1] = MARK
        elif kind == "V":
            grid[r - 1, c - 1] = MARK

    try:
        out, _owner, entries = _solve_with_trace(grid)
    except AssertionError:
        return None

    pipe_pixels = grid == PIPE
    recolored = set(map(int, out[pipe_pixels])) - {PIPE}
    if len(recolored) < min(2, n_entries):
        return None

    if not any(int(out[row, w - 1]) != PIPE for row in exit_rows):
        return None

    if not np.any((grid == PIPE) & (out != PIPE)):
        return None

    if len(entries) != len(exit_rows):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-network input")
