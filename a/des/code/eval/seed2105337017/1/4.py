import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
SIGNAL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_ORDER = ["up", "down", "left", "right"]


def _parse_towers(g: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int], set[tuple[int, int]]]:
    h, w = g.shape
    towers = []
    arm_to_tower = {}
    arm_cells = set()
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            color = int(g[r, c])
            if color in (BG, GREY):
                continue
            arms = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            if all(int(g[rr, cc]) == GREY for rr, cc in arms):
                idx = len(towers)
                towers.append({"center": (r, c), "color": color})
                for cell in arms:
                    arm_to_tower[cell] = idx
                    arm_cells.add(cell)
    towers.sort(key=lambda t: t["center"])
    arm_to_tower.clear()
    arm_cells.clear()
    for idx, tower in enumerate(towers):
        r, c = tower["center"]
        for cell in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            arm_to_tower[cell] = idx
            arm_cells.add(cell)
    return towers, arm_to_tower, arm_cells


def _receivers(g: np.ndarray, arm_cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {tuple(map(int, cell)) for cell in np.argwhere(g == GREY) if tuple(map(int, cell)) not in arm_cells}


def _perpendicular(direction: str) -> tuple[str, str]:
    if direction in ("up", "down"):
        return "left", "right"
    return "up", "down"


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    towers, arm_to_tower, arm_cells = _parse_towers(g)
    receiver_cells = _receivers(g, arm_cells)

    for start_idx, tower in enumerate(towers):
        q = deque((start_idx, direction, tower["color"]) for direction in DIR_ORDER)
        seen_relays = set()
        while q:
            tower_idx, direction, color = q.popleft()
            cr, cc = towers[tower_idx]["center"]
            dr, dc = DIRS[direction]
            r, c = cr + 2 * dr, cc + 2 * dc
            while 0 <= r < h and 0 <= c < w:
                cell = (r, c)
                if cell in receiver_cells:
                    out[r, c] = int(color)
                    break
                if cell in arm_to_tower:
                    hit_idx = arm_to_tower[cell]
                    key = (start_idx, hit_idx, direction)
                    if key not in seen_relays:
                        seen_relays.add(key)
                        hit_color = towers[hit_idx]["color"]
                        for nd in _perpendicular(direction):
                            q.append((hit_idx, nd, hit_color))
                    break
                if int(g[r, c]) == BG:
                    out[r, c] = int(color)
                else:
                    break
                r += dr
                c += dc
    return out


def _clear_with_margin(occ: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = occ.shape
    for r, c in cells:
        if not (1 <= r < h - 1 and 1 <= c < w - 1):
            return False
        if occ[r - 1 : r + 2, c - 1 : c + 2].any():
            return False
    for r, c in cells:
        occ[r, c] = True
    return True


def _tower_cells(r: int, c: int) -> list[tuple[int, int]]:
    return [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _draw_tower(grid: np.ndarray, occ: np.ndarray, r: int, c: int, color: int) -> bool:
    cells = _tower_cells(r, c)
    if not _clear_with_margin(occ, cells):
        return False
    grid[r, c] = color
    for rr, cc in cells[1:]:
        grid[rr, cc] = GREY
    return True


def _draw_receiver(grid: np.ndarray, occ: np.ndarray, r: int, c: int) -> bool:
    if not _clear_with_margin(occ, [(r, c)]):
        return False
    grid[r, c] = GREY
    return True


def _try_add_direct_receiver(grid: np.ndarray, occ: np.ndarray, tower: tuple[int, int], direction: str) -> bool:
    h, w = grid.shape
    r, c = tower
    dr, dc = DIRS[direction]
    max_steps = []
    rr, cc = r + 4 * dr, c + 4 * dc
    while 2 <= rr < h - 2 and 2 <= cc < w - 2:
        if not occ[rr - 1 : rr + 2, cc - 1 : cc + 2].any():
            max_steps.append((rr, cc))
        rr += dr
        cc += dc
    if not max_steps:
        return False
    rr, cc = random.choice(max_steps[: min(8, len(max_steps))])
    return _draw_receiver(grid, occ, rr, cc)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(20, 30)
    w = random.randint(20, 30)
    grid = np.zeros((h, w), dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    colors = SIGNAL_COLORS[:]
    random.shuffle(colors)

    horizontal = random.choice([True, False])
    if horizontal:
        row = random.randint(5, h - 6)
        distance = random.randint(6, min(11, w - 8))
        if random.choice([True, False]):
            c1 = random.randint(4, w - distance - 4)
            centers = [(row, c1), (row, c1 + distance)]
            relay_dir = random.choice(["up", "down"])
        else:
            c2 = random.randint(distance + 4, w - 5)
            centers = [(row, c2), (row, c2 - distance)]
            relay_dir = random.choice(["up", "down"])
    else:
        col = random.randint(5, w - 6)
        distance = random.randint(6, min(11, h - 8))
        if random.choice([True, False]):
            r1 = random.randint(4, h - distance - 4)
            centers = [(r1, col), (r1 + distance, col)]
            relay_dir = random.choice(["left", "right"])
        else:
            r2 = random.randint(distance + 4, h - 5)
            centers = [(r2, col), (r2 - distance, col)]
            relay_dir = random.choice(["left", "right"])

    for i, (r, c) in enumerate(centers):
        if not _draw_tower(grid, occ, r, c, colors[i]):
            return None

    relay_center = centers[1]
    dr, dc = DIRS[relay_dir]
    for dist in random.sample(range(4, 10), 6):
        rr = relay_center[0] + dr * dist
        cc = relay_center[1] + dc * dist
        if 2 <= rr < h - 2 and 2 <= cc < w - 2 and _draw_receiver(grid, occ, rr, cc):
            break
    else:
        return None

    for idx in range(random.randint(0, 2)):
        for _ in range(80):
            r = random.randint(4, h - 5)
            c = random.randint(4, w - 5)
            if _draw_tower(grid, occ, r, c, colors[2 + idx]):
                centers.append((r, c))
                break

    receiver_count = random.randint(2, 5)
    for _ in range(receiver_count):
        tower = random.choice(centers)
        direction = random.choice(DIR_ORDER)
        _try_add_direct_receiver(grid, occ, tower, direction)

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid relay signal input")
