import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 300


def _candidate_switches(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    switches = []
    for r in range(h - 1):
        for c in range(w - 1):
            if not (r == 0 or c == 0 or r == h - 2 or c == w - 2):
                continue
            block = grid[r : r + 2, c : c + 2]
            colors, counts = np.unique(block, return_counts=True)
            colors = [int(x) for x in colors]
            counts = [int(x) for x in counts]
            if len(colors) != 2 or sorted(counts) != [2, 2]:
                continue
            if BLACK in colors:
                continue
            switches.append({"r": r, "c": c, "colors": tuple(colors)})
    return switches


def _resolve_switches(grid: np.ndarray) -> list[dict]:
    switches = _candidate_switches(grid)
    if not switches:
        return []

    switch_mask = np.zeros(grid.shape, dtype=bool)
    for sw in switches:
        switch_mask[sw["r"] : sw["r"] + 2, sw["c"] : sw["c"] + 2] = True

    window_freq = Counter()
    for sw in switches:
        for color in sw["colors"]:
            window_freq[int(color)] += 1

    outside_count = {
        int(color): int(np.sum((grid == int(color)) & ~switch_mask))
        for color in window_freq
    }

    resolved = []
    for sw in switches:
        a, b = sw["colors"]
        if outside_count[a] > 0 and outside_count[b] == 0:
            signal = a
        elif outside_count[b] > 0 and outside_count[a] == 0:
            signal = b
        elif window_freq[a] != window_freq[b]:
            signal = a if window_freq[a] < window_freq[b] else b
        else:
            signal = a if outside_count[a] >= outside_count[b] else b

        block = grid[sw["r"] : sw["r"] + 2, sw["c"] : sw["c"] + 2]
        pos = np.argwhere(block == signal)
        rows = {int(r) for r in pos[:, 0]}
        cols = {int(c) for c in pos[:, 1]}
        if len(rows) == 2 and len(cols) == 2:
            keep = True
        elif len(rows) == 1:
            keep = False
        else:
            continue

        resolved.append(
            {
                "r": int(sw["r"]),
                "c": int(sw["c"]),
                "color": int(signal),
                "keep": bool(keep),
            }
        )
    return resolved


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    switches = _resolve_switches(g)
    if not switches:
        return out

    switch_mask = np.zeros(g.shape, dtype=bool)
    closed_colors = set()
    for sw in switches:
        switch_mask[sw["r"] : sw["r"] + 2, sw["c"] : sw["c"] + 2] = True
        if not sw["keep"]:
            closed_colors.add(int(sw["color"]))

    for color in closed_colors:
        out[(g == int(color)) & ~switch_mask] = BLACK
    return out


def _valid_next(
    nr: int,
    nc: int,
    occupied: np.ndarray,
    visited: set[tuple[int, int]],
    tail: tuple[int, int],
) -> bool:
    h, w = occupied.shape
    if not (2 <= nr < h - 2 and 2 <= nc < w - 2):
        return False
    if occupied[nr, nc] or (nr, nc) in visited:
        return False

    seen_neighbors = 0
    for dr, dc in DIR4:
        rr, cc = nr + dr, nc + dc
        if (rr, cc) in visited:
            seen_neighbors += 1
            if (rr, cc) != tail:
                return False
    return seen_neighbors == 1


def _random_path(
    h: int,
    w: int,
    occupied: np.ndarray,
    rng: random.Random,
    min_len: int,
    max_len: int,
) -> list[tuple[int, int]] | None:
    target_len = rng.randint(min_len, max_len)
    starts = [(r, c) for r in range(2, h - 2) for c in range(2, w - 2) if not occupied[r, c]]
    rng.shuffle(starts)

    for start in starts[:80]:
        path = [start]
        visited = {start}
        prev_dir = None

        while len(path) < target_len:
            r, c = path[-1]
            dirs = list(DIR4)
            rng.shuffle(dirs)
            if prev_dir is not None:
                dirs.sort(key=lambda d: 0 if d == prev_dir else 1)

            moves = []
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if _valid_next(nr, nc, occupied, visited, path[-1]):
                    weight = 3 if prev_dir is not None and (dr, dc) == prev_dir else 1
                    moves.extend([(nr, nc, (dr, dc))] * weight)

            if not moves:
                break

            nr, nc, step_dir = rng.choice(moves)
            path.append((nr, nc))
            visited.add((nr, nc))
            prev_dir = step_dir

        if len(path) >= min_len:
            return path
    return None


def _switch_positions(h: int, w: int, count: int, rng: random.Random) -> list[tuple[int, int]]:
    side_to_pos = {
        "top": lambda: (0, rng.randint(2, w - 4)),
        "bottom": lambda: (h - 2, rng.randint(2, w - 4)),
        "left": lambda: (rng.randint(2, h - 4), 0),
        "right": lambda: (rng.randint(2, h - 4), w - 2),
    }
    sides = ["top", "bottom", "left", "right"]
    chosen_sides = sides if count == 4 else rng.sample(sides, count)
    return [side_to_pos[side]() for side in chosen_sides]


def _paint_switch(grid: np.ndarray, r: int, c: int, color: int, keep: bool, rng: random.Random) -> None:
    block = np.full((2, 2), GREY, dtype=int)
    if keep:
        coords = [(0, 0), (1, 1)] if rng.random() < 0.5 else [(0, 1), (1, 0)]
    else:
        coords = [(0, 0), (0, 1)] if rng.random() < 0.5 else [(1, 0), (1, 1)]
    for rr, cc in coords:
        block[rr, cc] = int(color)
    grid[r : r + 2, c : c + 2] = block


def _expected_output(grid: np.ndarray, switches: list[dict]) -> np.ndarray:
    out = grid.copy()
    switch_mask = np.zeros(grid.shape, dtype=bool)
    for sw in switches:
        switch_mask[sw["r"] : sw["r"] + 2, sw["c"] : sw["c"] + 2] = True
    for sw in switches:
        if not sw["keep"]:
            out[(grid == int(sw["color"])) & ~switch_mask] = BLACK
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(12, 18)
    w = rng.randint(12, 18)
    switched_count = rng.choice((2, 4))
    unswitched_count = rng.choice((0, 1))
    total_paths = switched_count + unswitched_count
    if total_paths > len(COLOR_POOL):
        return None

    colors = rng.sample(COLOR_POOL, total_paths)
    switched_colors = colors[:switched_count]
    unswitched_colors = colors[switched_count:]

    grid = np.full((h, w), BLACK, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    min_len = 5
    max_len = max(min_len + 1, min(14, (h - 4) * (w - 4) // max(total_paths, 1)))
    for color in colors:
        path = _random_path(h, w, occupied, rng, min_len, max_len)
        if path is None:
            return None
        for r, c in path:
            grid[r, c] = int(color)
            occupied[r, c] = True

    positions = _switch_positions(h, w, switched_count, rng)
    states = [True] * (switched_count // 2) + [False] * (switched_count - switched_count // 2)
    rng.shuffle(states)

    switches = []
    for (r, c), color, keep in zip(positions, switched_colors, states):
        _paint_switch(grid, r, c, color, keep, rng)
        switches.append({"r": int(r), "c": int(c), "color": int(color), "keep": bool(keep)})

    # Keep unswitched paths genuinely unaffected by any switch.
    for color in unswitched_colors:
        if any(sw["color"] == color for sw in switches):
            return None

    expected = _expected_output(grid, switches)
    if not np.array_equal(main(grid), expected):
        return None
    if np.array_equal(grid, expected):
        return None
    if len(_resolve_switches(grid)) != switched_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
