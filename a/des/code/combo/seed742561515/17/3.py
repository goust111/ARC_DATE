from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
COLOR_POOL = tuple(range(1, 10))
MAX_ATTEMPTS = 500


def _inside(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            comp.append((cr, cc))
            for dr, dc in DIRS.values():
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _border(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells.update((r1, c) for c in range(c0, c1 + 1))
    cells.update((r, c0) for r in range(r0, r1 + 1))
    cells.update((r, c1) for r in range(r0, r1 + 1))
    return cells


def _gap_direction(box: tuple[int, int, int, int], missing: tuple[int, int]) -> tuple[int, int] | None:
    r0, c0, r1, c1 = box
    r, c = missing
    if r == r0 and c0 < c < c1:
        return DIRS["up"]
    if r == r1 and c0 < c < c1:
        return DIRS["down"]
    if c == c0 and r0 < r < r1:
        return DIRS["left"]
    if c == c1 and r0 < r < r1:
        return DIRS["right"]
    return None


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BG):
        for comp in _components(grid == color):
            if len(comp) < 8:
                continue
            box = _bbox(comp)
            r0, c0, r1, c1 = box
            if r1 - r0 < 3 or c1 - c0 < 3:
                continue
            border = _border(box)
            comp_set = set(comp)
            missing = list(border - comp_set)
            extra = comp_set - border
            if extra or len(missing) != 1:
                continue
            direction = _gap_direction(box, missing[0])
            if direction is None:
                continue
            frames.append({"color": color, "box": box, "gap": missing[0], "direction": direction})
    return frames


def _sprite_for_frame(grid: np.ndarray, frame: dict) -> tuple[int, list[tuple[int, int]]] | None:
    r0, c0, r1, c1 = frame["box"]
    sub = grid[r0 + 1 : r1, c0 + 1 : c1]
    mask = (sub != BG) & (sub != int(frame["color"]))
    comps = _components(mask)
    if len(comps) != 1:
        return None
    cells = [(r0 + 1 + r, c0 + 1 + c) for r, c in comps[0]]
    color = int(grid[cells[0]])
    if any(int(grid[r, c]) != color for r, c in cells):
        return None
    return color, cells


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    moves: list[tuple[int, tuple[int, int], list[tuple[int, int]]]] = []

    for frame in _find_frames(g):
        sprite = _sprite_for_frame(g, frame)
        if sprite is None:
            continue
        color, cells = sprite
        moves.append((color, frame["direction"], cells))

    for color, direction, cells in moves:
        dr, dc = direction
        steps = len(cells)
        for r, c in cells:
            out[r, c] = BG
        for r, c in cells:
            nr, nc = r + steps * dr, c + steps * dc
            if _inside(out.shape, nr, nc):
                out[nr, nc] = color
    return out


def _expanded(cells: set[tuple[int, int]], shape: tuple[int, int], pad: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - pad, r + pad + 1):
            for cc in range(c - pad, c + pad + 1):
                if _inside(shape, rr, cc):
                    out.add((rr, cc))
    return out


def _try_group(grid: np.ndarray, occupied: set[tuple[int, int]], colors: list[int]) -> bool:
    frame_color = colors.pop()
    sprite_color = colors.pop()
    direction = random.choice(tuple(DIRS.values()))
    dr, dc = direction
    steps = random.randint(2, 5)

    if direction in (DIRS["up"], DIRS["down"]):
        fh = steps + 2
        fw = random.choice((5, 7, 9))
    else:
        fh = random.choice((5, 7, 9))
        fw = steps + 2

    h, w = grid.shape
    for _ in range(150):
        r0 = random.randint(steps + 2, h - fh - steps - 3)
        c0 = random.randint(steps + 2, w - fw - steps - 3)
        r1, c1 = r0 + fh - 1, c0 + fw - 1
        if direction == DIRS["up"]:
            gap = (r0, (c0 + c1) // 2)
            sprite = {(r, gap[1]) for r in range(r0 + 1, r1)}
        elif direction == DIRS["down"]:
            gap = (r1, (c0 + c1) // 2)
            sprite = {(r, gap[1]) for r in range(r0 + 1, r1)}
        elif direction == DIRS["left"]:
            gap = ((r0 + r1) // 2, c0)
            sprite = {(gap[0], c) for c in range(c0 + 1, c1)}
        else:
            gap = ((r0 + r1) // 2, c1)
            sprite = {(gap[0], c) for c in range(c0 + 1, c1)}

        frame = _border((r0, c0, r1, c1)) - {gap}
        final = {(r + steps * dr, c + steps * dc) for r, c in sprite}
        reserve = frame | sprite | final
        if any(not _inside(grid.shape, r, c) for r, c in reserve):
            continue
        if final & frame:
            continue
        if _expanded(reserve, grid.shape, 1) & occupied:
            continue

        for r, c in frame:
            grid[r, c] = frame_color
        for r, c in sprite:
            grid[r, c] = sprite_color
        occupied.update(_expanded(reserve, grid.shape, 1))
        return True

    colors.extend([frame_color, sprite_color])
    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(24, 34)
    w = random.randint(24, 34)
    grid = np.full((h, w), BG, dtype=int)
    occupied: set[tuple[int, int]] = set()
    colors = list(COLOR_POOL)
    random.shuffle(colors)
    count = random.randint(2, 4)
    if len(colors) < 2 * count:
        return None
    for _ in range(count):
        if not _try_group(grid, occupied, colors):
            return None
    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid notched-frame input")
