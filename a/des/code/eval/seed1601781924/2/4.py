from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
WALL = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _degrees(pset: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    deg: dict[tuple[int, int], int] = {}
    for cell in pset:
        deg[cell] = sum((nbr in pset) for nbr in _neighbors4(*cell))
    return deg


def _walk_path(pset: set[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]] | None:
    path = [start]
    seen = {start}
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxts = [nbr for nbr in _neighbors4(*cur) if nbr in pset and nbr != prev]
        if not nxts:
            break
        if len(nxts) != 1:
            return None
        nxt = nxts[0]
        if nxt in seen:
            return None
        path.append(nxt)
        seen.add(nxt)
        prev, cur = cur, nxt
    if len(seen) != len(pset):
        return None
    return path


def _is_key_endpoint(g: np.ndarray, pset: set[tuple[int, int]], cell: tuple[int, int]) -> bool:
    color = int(g[cell])
    if color in (BLACK, WALL):
        return False
    for nbr in _neighbors4(*cell):
        if nbr not in pset:
            continue
        nbr_color = int(g[nbr])
        if nbr_color not in (BLACK, WALL) and nbr_color != color:
            return False
    return True


def _analyze_channel(g: np.ndarray, comp: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    if len(comp) < 2:
        return None
    pset = set(comp)
    deg = _degrees(pset)
    if any(v > 2 for v in deg.values()):
        return None
    endpoints = [cell for cell, v in deg.items() if v == 1]
    if len(endpoints) != 2:
        return None
    keys = [cell for cell in endpoints if _is_key_endpoint(g, pset, cell)]
    if len(keys) != 1:
        return None
    path = _walk_path(pset, keys[0])
    if path is None or len(path) != len(comp):
        return None
    return path


def _paint_channel(inp: np.ndarray, out: np.ndarray, path: list[tuple[int, int]]) -> None:
    current = int(inp[path[0]])
    i = 0
    while i < len(path):
        if i + 1 < len(path):
            c0 = int(inp[path[i]])
            c1 = int(inp[path[i + 1]])
            if c0 not in (BLACK, WALL) and c1 not in (BLACK, WALL) and c0 != c1 and c0 == current:
                r0, c0p = path[i]
                r1, c1p = path[i + 1]
                out[r0, c0p] = current
                current = int(inp[path[i + 1]])
                out[r1, c1p] = current
                i += 2
                continue
        r, c = path[i]
        out[r, c] = current
        i += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find keyed channel entrances, fill with the key color until the matching side of a two-color
    lock, then continue with the lock's far-side color. Non-channel distractor pixels are cleared.
    """
    g = np.asarray(grid, dtype=int)
    out = np.where(g == WALL, WALL, BLACK)
    for comp in _components(g != WALL):
        path = _analyze_channel(g, comp)
        if path is not None:
            _paint_channel(g, out, path)
    return out


def _can_start(cell: tuple[int, int], occupied: set[tuple[int, int]], h: int, w: int) -> bool:
    r, c = cell
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    if cell in occupied:
        return False
    return all(nbr not in occupied for nbr in _neighbors4(r, c))


def _can_extend(
    cell: tuple[int, int],
    prev: tuple[int, int],
    blocked: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    r, c = cell
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    if cell in blocked:
        return False
    adj = [nbr for nbr in _neighbors4(r, c) if nbr in blocked]
    return set(adj) <= {prev}


def _future_options(
    cell: tuple[int, int],
    prev: tuple[int, int],
    blocked: set[tuple[int, int]],
    h: int,
    w: int,
) -> int:
    temp = set(blocked)
    temp.add(cell)
    count = 0
    for nbr in _neighbors4(*cell):
        if _can_extend(nbr, cell, temp, h, w):
            count += 1
    return count


def _grow_path(
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    target_len: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    starts = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if _can_start((r, c), occupied, h, w)
    ]
    if not starts:
        return None

    for _ in range(90):
        start = rng.choice(starts)
        path = [start]
        blocked = set(occupied)
        blocked.add(start)
        prev_dir: tuple[int, int] | None = None

        while len(path) < target_len:
            cur = path[-1]
            options: list[tuple[int, tuple[int, int]]] = []
            for dr, dc in DIR4:
                nxt = (cur[0] + dr, cur[1] + dc)
                if not _can_extend(nxt, cur, blocked, h, w):
                    continue
                score = _future_options(nxt, cur, blocked, h, w)
                if len(path) + 1 < target_len and score == 0:
                    continue
                bonus = 2 if prev_dir is not None and (dr, dc) != prev_dir else 1
                options.append((score + bonus, nxt))
            if not options:
                break

            weights = [max(1, score) for score, _ in options]
            nxt = rng.choices([cell for _, cell in options], weights=weights, k=1)[0]
            prev_dir = (nxt[0] - cur[0], nxt[1] - cur[1])
            path.append(nxt)
            blocked.add(nxt)

        if len(path) == target_len:
            return path

    return None


def _sample_lock_starts(last_max: int, count: int, rng: random.Random) -> list[int] | None:
    if last_max < 1:
        return None
    positions: list[int] = []
    for remaining in range(count, 0, -1):
        min_pos = positions[-1] + 3 if positions else 1
        max_pos = last_max - 3 * (remaining - 1)
        if min_pos > max_pos:
            return None
        positions.append(rng.randint(min_pos, max_pos))
    return positions


def _lock_positions(length: int, count: int, end_lock: bool, rng: random.Random) -> list[int] | None:
    if end_lock:
        if count == 1:
            return [length - 2] if length >= 3 else None
        earlier = _sample_lock_starts(length - 5, count - 1, rng)
        if earlier is None:
            return None
        return earlier + [length - 2]
    return _sample_lock_starts(length - 3, count, rng)


def _feasible_lock_specs(length: int) -> list[tuple[int, bool]]:
    specs: list[tuple[int, bool]] = []
    for count in (1, 2, 3):
        for end_lock in (False, True):
            if _lock_positions(length, count, end_lock, random.Random(0)) is not None:
                specs.append((count, end_lock))
    return specs


def _place_channel_markers(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    ordered = list(path)
    if rng.random() < 0.5:
        ordered.reverse()

    specs = _feasible_lock_specs(len(ordered))
    assert specs, "path must be long enough for at least one lock"
    lock_count, end_lock = rng.choice(specs)
    positions = _lock_positions(len(ordered), lock_count, end_lock, rng)
    assert positions is not None

    colors = rng.sample(PALETTE, lock_count + 1)
    key_color = colors[0]
    grid[ordered[0]] = key_color
    for idx, pos in enumerate(positions):
        grid[ordered[pos]] = colors[idx]
        grid[ordered[pos + 1]] = colors[idx + 1]

    return positions, colors


def _place_distractors(
    grid: np.ndarray,
    passable: set[tuple[int, int]],
    count: int,
    rng: random.Random,
) -> int:
    h, w = grid.shape
    placed = 0
    occupied = set(passable)
    for _ in range(count * 20):
        if placed >= count:
            break
        r = rng.randint(1, h - 2)
        c = rng.randint(1, w - 2)
        cell = (r, c)
        if cell in occupied:
            continue
        if any(nbr in occupied for nbr in _neighbors4(r, c)):
            continue
        grid[cell] = rng.choice(PALETTE)
        occupied.add(cell)
        placed += 1
    return placed


def _validate_generated(grid: np.ndarray, expected_channels: int) -> bool:
    comps = _components(grid != WALL)
    keyed = 0
    for comp in comps:
        if len(comp) == 1:
            r, c = comp[0]
            if int(grid[r, c]) in (BLACK, WALL):
                return False
            continue
        path = _analyze_channel(grid, comp)
        if path is None:
            return False
        keyed += 1

    if keyed != expected_channels:
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False

    for comp in comps:
        if len(comp) == 1:
            r, c = comp[0]
            if int(out[r, c]) != BLACK:
                return False
        else:
            for r, c in comp:
                if int(out[r, c]) in (BLACK, WALL):
                    return False
    return True


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(12, 24)
    w = rng.randint(12, 24)
    grid = np.full((h, w), WALL, dtype=int)

    max_channels = 4 if h * w >= 250 else 3
    channel_count = rng.randint(1, max_channels)

    occupied: set[tuple[int, int]] = set()
    paths: list[list[tuple[int, int]]] = []

    for i in range(channel_count):
        remaining = channel_count - i - 1
        max_len = min(18, (h - 2) * (w - 2) - len(occupied) - 3 * remaining)
        min_len = 4 if remaining == 0 else 5
        if max_len < min_len:
            return None
        target_len = rng.randint(min_len, max_len)
        path = _grow_path(h, w, occupied, target_len, rng)
        if path is None:
            return None
        paths.append(path)
        occupied.update(path)
        for cell in path:
            grid[cell] = BLACK

    for path in paths:
        _place_channel_markers(grid, path, rng)

    distractor_target = rng.randint(0, min(6, (h * w) // 40 + 2))
    _place_distractors(grid, occupied, distractor_target, rng)

    if not _validate_generated(grid, len(paths)):
        return None
    return grid


def generate_input() -> np.ndarray:
    """
    Random channel puzzle generator.

    Variations:
    - grid size
    - number and length of channels
    - channel geometry and orientation
    - lock count and whether the last lock sits at the end
    - key / lock colors
    - number and placement of isolated distractors
    """
    for _ in range(400):
        rng = random.Random(random.randrange(1 << 30))
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid channel-lock puzzle")
