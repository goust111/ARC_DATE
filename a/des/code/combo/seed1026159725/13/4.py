import random
from collections import deque

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
GRAY = 5
ORANGE = 7

PIPE_COLORS = {RED, GREEN, GRAY, ORANGE}
BACKGROUND_CHOICES = [0, 4, 6, 8, 9]
GEN_ATTEMPTS = 400
TRANSFORMS = (
    lambda x, y: (x, y),
    lambda x, y: (x, -y),
    lambda x, y: (-x, y),
    lambda x, y: (-x, -y),
    lambda x, y: (y, x),
    lambda x, y: (y, -x),
    lambda x, y: (-y, x),
    lambda x, y: (-y, -x),
)
DIRS = {
    "E": (1, 0),
    "W": (-1, 0),
    "N": (0, -1),
    "S": (0, 1),
}


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _polyline(start: tuple[int, int], segments: list[tuple[str, int]]) -> list[tuple[int, int]]:
    path = [start]
    x, y = start
    for direction, length in segments:
        dx, dy = DIRS[direction]
        for _ in range(length):
            x += dx
            y += dy
            path.append((x, y))
    return path


def _add_path(occupied: set[tuple[int, int]], path: list[tuple[int, int]]) -> bool:
    if not path:
        return False
    if path[0] not in occupied:
        occupied.add(path[0])
    for i in range(1, len(path)):
        x, y = path[i]
        prev = path[i - 1]
        if (x, y) in occupied:
            return False
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if (nx, ny) in occupied and (nx, ny) != prev:
                return False
        occupied.add((x, y))
    return True


def _build_branch(start: tuple[int, int], primary_dir: str) -> list[tuple[int, int]]:
    first_len = random.randint(2, 5)
    segments = [(primary_dir, first_len)]
    if random.random() < 0.7:
        if primary_dir in ("N", "S"):
            turn_dir = random.choice(["E", "W"])
        else:
            turn_dir = random.choice(["N", "S"])
        segments.append((turn_dir, random.randint(1, 4)))
    return _polyline(start, segments)


def _transform_paths(paths: dict[str, list[tuple[int, int]]]) -> dict[str, list[tuple[int, int]]]:
    transform = random.choice(TRANSFORMS)
    transformed: dict[str, list[tuple[int, int]]] = {}
    all_rc: list[tuple[int, int]] = []
    for name, path in paths.items():
        tpath = []
        for x, y in path:
            tx, ty = transform(x, y)
            rc = (ty, tx)
            tpath.append(rc)
            all_rc.append(rc)
        transformed[name] = tpath

    min_r = min(r for r, _c in all_rc)
    min_c = min(c for _r, c in all_rc)
    for name, path in transformed.items():
        transformed[name] = [(r - min_r, c - min_c) for r, c in path]
    return transformed


def _simulate_reachable(grid: np.ndarray) -> set[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    source = np.argwhere(g == RED)
    assert len(source) == 1, "expected exactly one red source"

    sr, sc = map(int, source[0])
    seen = {(sr, sc)}
    q: deque[tuple[int, int]] = deque([(sr, sc)])

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]):
                continue
            value = int(g[nr, nc])
            if value not in PIPE_COLORS:
                continue
            if value == ORANGE:
                continue
            if (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return seen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color every pipe pixel that receives water from the red source.

    The pipe network is made of gray pixels. Green valves pass water, orange
    valves stop water, and the red source marks the entry point. Reachable gray
    pipe pixels turn blue while valves and the source keep their original color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    reachable = _simulate_reachable(g)
    out = g.copy()
    for r, c in reachable:
        if int(g[r, c]) == GRAY:
            out[r, c] = BLUE
    return out


def _gen_attempt() -> np.ndarray | None:
    source = (0, 0)
    trunk_a = random.randint(2, 4)
    trunk_b = random.randint(2, 4)
    trunk_c = random.randint(1, 4)
    trunk = _polyline(source, [("E", trunk_a + trunk_b + trunk_c)])
    junction1 = trunk[trunk_a]
    junction2 = trunk[trunk_a + trunk_b]

    open_branch = _build_branch(junction1, "N")
    closed_branch = _build_branch(junction1, "S")

    paths: dict[str, list[tuple[int, int]]] = {
        "trunk": trunk,
        "open": open_branch,
        "closed": closed_branch,
    }

    if random.random() < 0.6:
        extra_side = random.choice(["N", "S"])
        paths["extra"] = _build_branch(junction2, extra_side)

    occupied: set[tuple[int, int]] = set()
    for name in ("trunk", "open", "closed", "extra"):
        path = paths.get(name)
        if path is not None and not _add_path(occupied, path):
            return None

    open_idx = random.randint(1, len(open_branch) - 2)
    closed_idx = random.randint(1, len(closed_branch) - 2)

    transformed = _transform_paths(paths)
    open_actual = transformed["open"]
    closed_actual = transformed["closed"]
    all_coords = [rc for path in transformed.values() for rc in path]
    height = max(r for r, _c in all_coords) + 1
    width = max(c for _r, c in all_coords) + 1

    H = height + random.randint(2, 6)
    W = width + random.randint(2, 6)
    row_shift = random.randint(1, H - height - 1)
    col_shift = random.randint(1, W - width - 1)

    shifted: dict[str, list[tuple[int, int]]] = {}
    for name, path in transformed.items():
        shifted[name] = [(r + row_shift, c + col_shift) for r, c in path]

    bg = random.choice(BACKGROUND_CHOICES)
    grid = np.full((H, W), bg, dtype=int)

    pipe_cells = {rc for path in shifted.values() for rc in path}
    for r, c in pipe_cells:
        grid[r, c] = GRAY

    source_rc = shifted["trunk"][0]
    grid[source_rc] = RED
    grid[shifted["open"][open_idx]] = GREEN
    grid[shifted["closed"][closed_idx]] = ORANGE

    if random.random() < 0.4 and "extra" in shifted and len(shifted["extra"]) >= 4:
        extra_idx = random.randint(1, len(shifted["extra"]) - 2)
        grid[shifted["extra"][extra_idx]] = random.choice([GREEN, ORANGE])

    out = main(grid)

    open_tail = shifted["open"][open_idx + 1 :]
    closed_tail = shifted["closed"][closed_idx + 1 :]
    if not open_tail or not closed_tail:
        return None
    if not any(int(out[r, c]) == BLUE for r, c in open_tail):
        return None
    if not all(int(out[r, c]) == GRAY for r, c in closed_tail):
        return None
    if int(out[source_rc]) != RED:
        return None
    if int(out[shifted["open"][open_idx]]) != GREEN:
        return None
    if int(out[shifted["closed"][closed_idx]]) != ORANGE:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-and-valve input")
