import random
from collections import deque

import numpy as np


BG = 0
WALL = 5


OPEN_PATTERNS = {
    (-1, 0): {
        "walls": {(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2)},
        "inside": [(0, 1), (1, 1)],
    },
    (1, 0): {
        "walls": {(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (1, 2), (2, 2)},
        "inside": [(1, 1), (2, 1)],
    },
    (0, -1): {
        "walls": {(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (2, 2), (1, 2)},
        "inside": [(1, 0), (1, 1)],
    },
    (0, 1): {
        "walls": {(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (2, 2), (1, 0)},
        "inside": [(1, 1), (1, 2)],
    },
}


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for sr, sc in zip(*np.where(mask)):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _find_cups(g: np.ndarray):
    cups = []
    for comp in _components(g == WALL):
        r0, c0, r1, c1 = _bbox(comp)
        if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
            continue
        rel_walls = {(r - r0, c - c0) for r, c in comp}
        for open_out, spec in OPEN_PATTERNS.items():
            if rel_walls == spec["walls"]:
                inside = [(r0 + r, c0 + c) for r, c in spec["inside"]]
                locks = [int(g[r, c]) for r, c in inside if int(g[r, c]) not in (BG, WALL)]
                assert len(locks) == 1
                if open_out == (-1, 0):
                    entry = min(inside)
                elif open_out == (1, 0):
                    entry = max(inside)
                elif open_out == (0, -1):
                    entry = min(inside, key=lambda p: p[1])
                else:
                    entry = max(inside, key=lambda p: p[1])
                dr, dc = -open_out[0], -open_out[1]
                center = (r0 + 1, c0 + 1)
                exit_start = (center[0] + 2 * dr, center[1] + 2 * dc)
                cups.append(
                    {
                        "entry": entry,
                        "dir": (dr, dc),
                        "inside": inside,
                        "lock": locks[0],
                        "exit": exit_start,
                    }
                )
                break
    return cups


def _emitters(g: np.ndarray, cup_inside: set[tuple[int, int]]):
    h, w = g.shape
    result = []
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            color = int(g[r, c])
            if color in (BG, WALL) or (r, c) in cup_inside:
                continue
            dirs = []
            if r == 0:
                dirs.append((1, 0))
            if r == h - 1:
                dirs.append((-1, 0))
            if c == 0:
                dirs.append((0, 1))
            if c == w - 1:
                dirs.append((0, -1))
            if len(dirs) == 1:
                result.append((r + dirs[0][0], c + dirs[0][1], dirs[0], color))
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace edge beams through grey cup relays. A beam entering a cup mouth fills
    the cup interior with its incoming color and spawns a new beam from the cup
    bottom using the cup's original lock color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    cups = _find_cups(g)
    entry_map = {(cup["entry"], cup["dir"]): cup for cup in cups}
    cup_inside = {cell for cup in cups for cell in cup["inside"]}
    queue = deque(_emitters(g, cup_inside))

    while queue:
        r, c, (dr, dc), color = queue.popleft()
        while 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
            cup = entry_map.get(((r, c), (dr, dc)))
            if cup is not None:
                for rr, cc in cup["inside"]:
                    out[rr, cc] = color
                nr, nc = cup["exit"]
                queue.append((nr, nc, (dr, dc), cup["lock"]))
                break
            if out[r, c] == BG:
                out[r, c] = color
                r += dr
                c += dc
                continue
            break
    return out


def _place_cup(grid, center, open_out, lock):
    r0, c0 = center[0] - 1, center[1] - 1
    if not (0 <= r0 and 0 <= c0 and r0 + 3 <= grid.shape[0] and c0 + 3 <= grid.shape[1]):
        return False
    cells = [(r0 + r, c0 + c) for r, c in OPEN_PATTERNS[open_out]["walls"]]
    cells.append(center)
    for r, c in cells:
        if grid[r, c] != BG:
            return False
    for r, c in OPEN_PATTERNS[open_out]["walls"]:
        grid[r0 + r, c0 + c] = WALL
    grid[center] = lock
    return True


def _gen_attempt():
    h = random.randint(21, 29)
    w = random.randint(21, 29)
    grid = np.zeros((h, w), dtype=int)
    palette = [1, 2, 3, 4, 6, 7, 8, 9]
    chain_count = random.randint(2, 3)
    cup_counts = [random.randint(1, 2) for _ in range(chain_count)]
    if sum(cup_counts) < 3:
        cup_counts[0] += 1
    locks = iter(random.sample(palette, sum(cup_counts)))

    family = random.choice(["vertical", "horizontal"])
    if family == "vertical":
        candidates = list(range(3, w - 3))
        random.shuffle(candidates)
        lanes = []
        for col in candidates:
            if all(abs(col - old) >= 4 for old in lanes):
                lanes.append(col)
                if len(lanes) == chain_count:
                    break
        if len(lanes) < chain_count:
            return None
        directions = [(1, 0), (-1, 0)]
        while len(directions) < chain_count:
            directions.append(random.choice([(1, 0), (-1, 0)]))
        random.shuffle(directions)
        for col, direction, count in zip(lanes, directions, cup_counts):
            coords = list(range(4, h - 4, 5))
            if direction == (-1, 0):
                coords = list(reversed(coords))
            if len(coords) < count:
                return None
            edge = (0, col) if direction == (1, 0) else (h - 1, col)
            grid[edge] = random.choice(palette)
            open_out = (-direction[0], -direction[1])
            for row in coords[:count]:
                if not _place_cup(grid, (row, col), open_out, next(locks)):
                    return None
    else:
        candidates = list(range(3, h - 3))
        random.shuffle(candidates)
        lanes = []
        for row in candidates:
            if all(abs(row - old) >= 4 for old in lanes):
                lanes.append(row)
                if len(lanes) == chain_count:
                    break
        if len(lanes) < chain_count:
            return None
        directions = [(0, 1), (0, -1)]
        while len(directions) < chain_count:
            directions.append(random.choice([(0, 1), (0, -1)]))
        random.shuffle(directions)
        for row, direction, count in zip(lanes, directions, cup_counts):
            coords = list(range(4, w - 4, 5))
            if direction == (0, -1):
                coords = list(reversed(coords))
            if len(coords) < count:
                return None
            edge = (row, 0) if direction == (0, 1) else (row, w - 1)
            grid[edge] = random.choice(palette)
            open_out = (-direction[0], -direction[1])
            for col in coords[:count]:
                if not _place_cup(grid, (row, col), open_out, next(locks)):
                    return None
    try:
        cups = _find_cups(grid)
        emitters = _emitters(grid, {cell for cup in cups for cell in cup["inside"]})
        assert len(cups) >= 3 and len(emitters) >= 2
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
