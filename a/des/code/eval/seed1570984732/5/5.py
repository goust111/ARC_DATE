import random
from collections import Counter, deque

import numpy as np


BG = 0
GREY = 5
PALETTE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = {
    "right": (0, 1),
    "left": (0, -1),
    "down": (1, 0),
    "up": (-1, 0),
}


def _bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _palette(grid: np.ndarray, bg: int) -> tuple[list[int], set[tuple[int, int]]]:
    cells = []
    colors = []
    for c, val in enumerate(grid[0]):
        if int(val) in (bg, GREY):
            break
        colors.append(int(val))
        cells.append((0, c))
    return colors, set(cells)


def _bbox(comp: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [p[0] for p in comp]
    cs = [p[1] for p in comp]
    return min(rs), min(cs), max(rs), max(cs)


def _is_rect_boundary(comp: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(comp)
    if r1 - r0 < 3 or c1 - c0 < 3:
        return False
    s = set(comp)
    perimeter = set()
    for c in range(c0, c1 + 1):
        perimeter.add((r0, c))
        perimeter.add((r1, c))
    for r in range(r0, r1 + 1):
        perimeter.add((r, c0))
        perimeter.add((r, c1))
    return s == perimeter


def _regions(grid: np.ndarray) -> list[dict]:
    regs = []
    for comp in _components(grid == GREY):
        if _is_rect_boundary(comp):
            regs.append({"bbox": _bbox(comp), "boundary": set(comp)})
    return regs


def _arrow_direction(comp: list[tuple[int, int]]) -> str | None:
    r0, c0, _, _ = _bbox(comp)
    norm = {(r - r0, c - c0) for r, c in comp}
    shapes = {
        "right": {(1, 0), (1, 1), (1, 2), (0, 2), (2, 2)},
        "left": {(1, 2), (1, 1), (1, 0), (0, 0), (2, 0)},
        "down": {(0, 1), (1, 1), (2, 1), (2, 0), (2, 2)},
        "up": {(2, 1), (1, 1), (0, 1), (0, 0), (0, 2)},
    }
    for direction, shape in shapes.items():
        if norm == shape:
            return direction
    return None


def _arrow_tip(comp: list[tuple[int, int]], direction: str) -> tuple[int, int]:
    if direction == "right":
        c = max(p[1] for p in comp)
        rs = sorted(p[0] for p in comp if p[1] == c)
        return rs[len(rs) // 2], c
    if direction == "left":
        c = min(p[1] for p in comp)
        rs = sorted(p[0] for p in comp if p[1] == c)
        return rs[len(rs) // 2], c
    if direction == "down":
        r = max(p[0] for p in comp)
        cs = sorted(p[1] for p in comp if p[0] == r)
        return r, cs[len(cs) // 2]
    r = min(p[0] for p in comp)
    cs = sorted(p[1] for p in comp if p[0] == r)
    return r, cs[len(cs) // 2]


def _arrows(grid: np.ndarray, bg: int, palette_cells: set[tuple[int, int]]) -> list[dict]:
    arrows = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (bg, GREY)):
        for comp in _components(grid == color):
            if len(comp) != 5 or any(p in palette_cells for p in comp):
                continue
            direction = _arrow_direction(comp)
            if direction is None:
                continue
            arrows.append(
                {
                    "color": color,
                    "cells": set(comp),
                    "direction": direction,
                    "tip": _arrow_tip(comp, direction),
                }
            )
    return arrows


def _pair_region(arrow: dict, regs: list[dict], shape: tuple[int, int]) -> dict | None:
    h, w = shape
    dr, dc = DIRS[arrow["direction"]]
    r, c = arrow["tip"]
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w:
        if any((r, c) in reg["boundary"] for reg in regs):
            for reg in regs:
                if (r, c) in reg["boundary"]:
                    return reg
        r += dr
        c += dc
    return None


def _island_count(reg: dict, grey_components: list[list[tuple[int, int]]]) -> int:
    r0, c0, r1, c1 = reg["bbox"]
    count = 0
    for comp in grey_components:
        s = set(comp)
        if s == reg["boundary"]:
            continue
        if all(r0 < r < r1 and c0 < c < c1 for r, c in comp):
            count += 1
    return count


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = _bg(g)
    palette, palette_cells = _palette(g, bg)
    assert palette, "missing palette row"
    regs = _regions(g)
    grey_components = _components(g == GREY)

    for arrow in _arrows(g, bg, palette_cells):
        reg = _pair_region(arrow, regs, g.shape)
        if reg is None:
            continue
        n_islands = _island_count(reg, grey_components)
        fill = palette[min(n_islands, len(palette) - 1)]
        r0, c0, r1, c1 = reg["bbox"]
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if out[r, c] == bg:
                    out[r, c] = fill
        if arrow["color"] == fill:
            for r, c in reg["boundary"]:
                out[r, c] = fill
        for r, c in arrow["cells"]:
            out[r, c] = bg
    return out


def _draw_arrow(grid: np.ndarray, tip: tuple[int, int], direction: str, color: int) -> set[tuple[int, int]]:
    r, c = tip
    if direction == "right":
        cells = {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c - 2)}
    elif direction == "left":
        cells = {(r, c), (r - 1, c), (r + 1, c), (r, c + 1), (r, c + 2)}
    elif direction == "down":
        cells = {(r, c), (r, c - 1), (r, c + 1), (r - 1, c), (r - 2, c)}
    else:
        cells = {(r, c), (r, c - 1), (r, c + 1), (r + 1, c), (r + 2, c)}
    for rr, cc in cells:
        grid[rr, cc] = color
    return cells


def _free(cells: set[tuple[int, int]], occupied: set[tuple[int, int]], h: int, w: int) -> bool:
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w) or (r, c) in occupied or r == 0:
            return False
    return True


def _rect_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r in range(r0, r0 + h)
        for c in range(c0, c0 + w)
        if r in (r0, r0 + h - 1) or c in (c0, c0 + w - 1)
    }


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(24, 30)
    w = random.randint(24, 30)
    grid = np.full((h, w), BG, dtype=int)
    palette = random.sample(PALETTE_COLORS, random.randint(4, 5))
    for c, color in enumerate(palette):
        grid[0, c] = color
    occupied = {(0, c) for c in range(len(palette))}

    n_regions = random.randint(2, 4)
    made = 0
    for _ in range(200):
        if made >= n_regions:
            break
        rh = random.randint(5, 8)
        rw = random.randint(5, 8)
        side = random.choice(["left", "right", "up", "down"])
        r0 = random.randint(5, h - rh - 3)
        c0 = random.randint(4, w - rw - 3)
        boundary = _rect_cells(r0, c0, rh, rw)
        if not _free(boundary, occupied, h, w):
            continue
        if side in ("left", "right"):
            ar = random.randint(r0 + 1, r0 + rh - 2)
            tip = (ar, c0 - 1) if side == "left" else (ar, c0 + rw)
            direction = "right" if side == "left" else "left"
        else:
            ac = random.randint(c0 + 1, c0 + rw - 2)
            tip = (r0 - 1, ac) if side == "up" else (r0 + rh, ac)
            direction = "down" if side == "up" else "up"
        fill_index = random.randint(0, min(3, len(palette) - 1))
        fill = palette[fill_index]
        arrow_color = fill if random.random() < 0.5 else random.choice([p for p in palette if p != fill])
        arrow_cells = _draw_arrow(np.full((h, w), BG, dtype=int), tip, direction, arrow_color)
        if not _free(arrow_cells, occupied | boundary, h, w):
            continue

        for r, c in boundary:
            grid[r, c] = GREY
        occupied |= boundary
        island_spots = [
            (r, c)
            for r in range(r0 + 2, r0 + rh - 2)
            for c in range(c0 + 2, c0 + rw - 2)
        ]
        random.shuffle(island_spots)
        for r, c in island_spots[:fill_index]:
            grid[r, c] = GREY
            occupied.add((r, c))
        _draw_arrow(grid, tip, direction, arrow_color)
        occupied |= arrow_cells
        made += 1

    if made < 2:
        return None
    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid island-palette input")
