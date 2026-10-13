import random
from collections import deque

import numpy as np


BG = 0
WALL = 5
RIBBON_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTIONS = ("left", "right", "top", "bottom")

GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
CONTAINER_COUNT_RANGE = (1, 4)
INTERIOR_H_RANGE = (3, 8)
INTERIOR_W_RANGE = (3, 8)
PLACEMENT_RETRIES = 120
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) != color:
                continue
            visited[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _boundary_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _detect_containers(grid: np.ndarray) -> list[dict]:
    bg = _get_bg(grid)
    containers: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for comp in _color_components(grid, color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            comp_set = set(comp)
            boundary = _boundary_cells(r0, c0, r1, c1)
            if comp_set - boundary:
                continue
            missing = list(boundary - comp_set)
            if len(missing) != 1:
                continue
            gap_r, gap_c = missing[0]
            if (gap_r, gap_c) in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
                continue
            side = None
            if gap_c == c0:
                side = "left"
            elif gap_c == c1:
                side = "right"
            elif gap_r == r0:
                side = "top"
            elif gap_r == r1:
                side = "bottom"
            if side is None:
                continue
            containers.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "gap": (gap_r, gap_c),
                    "side": side,
                }
            )
    containers.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["side"]))
    return containers


def _read_ribbon(grid: np.ndarray, container: dict) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    bg = _get_bg(grid)
    r0, c0, r1, c1 = container["bbox"]
    gap_r, gap_c = container["gap"]
    side = container["side"]
    wall_color = int(container["color"])
    if side == "left":
        start_r, start_c = gap_r, c0 - 1
        dr, dc = 0, -1
    elif side == "right":
        start_r, start_c = gap_r, c1 + 1
        dr, dc = 0, 1
    elif side == "top":
        start_r, start_c = r0 - 1, gap_c
        dr, dc = -1, 0
    else:
        start_r, start_c = r1 + 1, gap_c
        dr, dc = 1, 0

    ribbon: list[tuple[int, int, int]] = []
    r, c = start_r, start_c
    while 0 <= r < h and 0 <= c < w:
        value = int(grid[r, c])
        if value == bg or value == wall_color:
            break
        ribbon.append((r, c, value))
        r += dr
        c += dc
    return ribbon


def _serpentine_cells(container: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = container["bbox"]
    gap_r, gap_c = container["gap"]
    side = container["side"]
    cells: list[tuple[int, int]] = []
    if side in {"left", "right"}:
        base = list(range(c0 + 1, c1))
        if side == "right":
            base = base[::-1]
        for offset, r in enumerate(range(gap_r, r1)):
            cols = base if offset % 2 == 0 else base[::-1]
            for c in cols:
                cells.append((r, c))
    else:
        base = list(range(r0 + 1, r1))
        if side == "bottom":
            base = base[::-1]
        for offset, c in enumerate(range(gap_c, c1)):
            rows = base if offset % 2 == 0 else base[::-1]
            for r in rows:
                cells.append((r, c))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each exterior ribbon into its matching open rectangular container and
    lay it inside with a side-dependent serpentine fold.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    for container in _detect_containers(g):
        ribbon = _read_ribbon(g, container)
        for r, c, _ in ribbon:
            out[r, c] = bg
        for (r, c), (_, _, color) in zip(_serpentine_cells(container), ribbon):
            out[r, c] = color

    return out


def _ribbon_cells(r0: int, c0: int, r1: int, c1: int, direction: str, length: int) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    if direction == "left":
        gap = (r0 + 1, c0)
        cells = [(gap[0], c0 - step) for step in range(1, length + 1)]
    elif direction == "right":
        gap = (r0 + 1, c1)
        cells = [(gap[0], c1 + step) for step in range(1, length + 1)]
    elif direction == "top":
        gap = (r0, c0 + 1)
        cells = [(r0 - step, gap[1]) for step in range(1, length + 1)]
    else:
        gap = (r1, c0 + 1)
        cells = [(r1 + step, gap[1]) for step in range(1, length + 1)]
    return gap, cells


def _max_outward_space(h: int, w: int, r0: int, c0: int, r1: int, c1: int, direction: str) -> int:
    if direction == "left":
        return c0
    if direction == "right":
        return w - 1 - c1
    if direction == "top":
        return r0
    return h - 1 - r1


def _blocked_box(h: int, w: int, r0: int, c0: int, r1: int, c1: int, direction: str, length: int) -> tuple[int, int, int, int]:
    if direction == "left":
        br0, bc0, br1, bc1 = r0, c0 - length, r1, c1
    elif direction == "right":
        br0, bc0, br1, bc1 = r0, c0, r1, c1 + length
    elif direction == "top":
        br0, bc0, br1, bc1 = r0 - length, c0, r1, c1
    else:
        br0, bc0, br1, bc1 = r0, c0, r1 + length, c1
    return max(0, br0 - 1), max(0, bc0 - 1), min(h - 1, br1 + 1), min(w - 1, bc1 + 1)


def _choose_ribbon_length(rng: random.Random, capacity: int, max_outward: int) -> int | None:
    if max_outward <= 0:
        return None
    mode = rng.choice(("short", "partial", "overflow"))
    if mode == "short":
        hi = min(max_outward, max(1, capacity - 1))
        return rng.randint(1, hi)
    if mode == "partial":
        lo = max(1, capacity // 3)
        hi = min(max_outward, capacity)
        if lo > hi:
            lo = 1
        return rng.randint(lo, hi)
    overflow_hi = min(max_outward, capacity + max(2, min(5, capacity // 2)))
    if overflow_hi >= capacity + 1:
        return rng.randint(capacity + 1, overflow_hi)
    hi = min(max_outward, capacity)
    lo = max(1, min(hi, capacity // 2))
    return rng.randint(lo, hi)


def _make_ribbon_colors(rng: random.Random, length: int) -> list[int]:
    motif_len = rng.randint(2, min(4, len(RIBBON_COLORS)))
    motif = rng.sample(RIBBON_COLORS, k=motif_len)
    offset = rng.randrange(motif_len)
    return [motif[(offset + i) % motif_len] for i in range(length)]


def _draw_container(
    grid: np.ndarray,
    occupied: np.ndarray,
    rng: random.Random,
    direction: str,
) -> bool:
    h, w = grid.shape
    for _ in range(PLACEMENT_RETRIES):
        ih = rng.randint(*INTERIOR_H_RANGE)
        iw = rng.randint(*INTERIOR_W_RANGE)
        fh = ih + 2
        fw = iw + 2
        if h - fh < 2 or w - fw < 2:
            continue
        r0 = rng.randint(1, h - fh - 1)
        c0 = rng.randint(1, w - fw - 1)
        r1 = r0 + fh - 1
        c1 = c0 + fw - 1
        max_outward = _max_outward_space(h, w, r0, c0, r1, c1, direction)
        length = _choose_ribbon_length(rng, ih * iw, max_outward)
        if length is None:
            continue
        br0, bc0, br1, bc1 = _blocked_box(h, w, r0, c0, r1, c1, direction, length)
        if np.any(occupied[br0 : br1 + 1, bc0 : bc1 + 1]):
            continue

        gap, ribbon_cells = _ribbon_cells(r0, c0, r1, c1, direction, length)
        colors = _make_ribbon_colors(rng, length)

        grid[r0, c0 : c1 + 1] = WALL
        grid[r1, c0 : c1 + 1] = WALL
        grid[r0 : r1 + 1, c0] = WALL
        grid[r0 : r1 + 1, c1] = WALL
        grid[gap] = BG
        for (rr, cc), color in zip(ribbon_cells, colors):
            grid[rr, cc] = color

        occupied[br0 : br1 + 1, bc0 : bc1 + 1] = True
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    count = rng.randint(*CONTAINER_COUNT_RANGE)
    directions = rng.sample(DIRECTIONS, k=min(count, len(DIRECTIONS)))

    for direction in directions:
        if not _draw_container(grid, occupied, rng, direction):
            return None

    if not directions:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
