import random
from collections import deque

import numpy as np


BG = 0
DIRS4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _components_by_color(grid: np.ndarray) -> list[tuple[int, set[tuple[int, int]]]]:
    h, w = grid.shape
    seen: set[tuple[int, int]] = set()
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or (r, c) in seen:
                continue
            q = deque([(r, c)])
            seen.add((r, c))
            comp: set[tuple[int, int]] = set()
            while q:
                cr, cc = q.popleft()
                comp.add((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and int(grid[nr, nc]) == color:
                        seen.add((nr, nc))
                        q.append((nr, nc))
            comps.append((color, comp))
    return comps


def _parse_arrow(cells: set[tuple[int, int]]) -> dict | None:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    for r in range(min(rows) + 1, max(rows)):
        tip = max(cols)
        expected = {(r - 1, tip), (r, tip), (r + 1, tip)}
        rod = {(r, c) for c in range(min(cols), tip)}
        if len(rod) >= 2 and cells == expected | rod:
            return {"dir": (0, 1), "tail": (r, min(cols)), "tip": (r, tip)}
        tip = min(cols)
        expected = {(r - 1, tip), (r, tip), (r + 1, tip)}
        rod = {(r, c) for c in range(tip + 1, max(cols) + 1)}
        if len(rod) >= 2 and cells == expected | rod:
            return {"dir": (0, -1), "tail": (r, max(cols)), "tip": (r, tip)}
    for c in range(min(cols) + 1, max(cols)):
        tip = max(rows)
        expected = {(tip, c - 1), (tip, c), (tip, c + 1)}
        rod = {(r, c) for r in range(min(rows), tip)}
        if len(rod) >= 2 and cells == expected | rod:
            return {"dir": (1, 0), "tail": (min(rows), c), "tip": (tip, c)}
        tip = min(rows)
        expected = {(tip, c - 1), (tip, c), (tip, c + 1)}
        rod = {(r, c) for r in range(tip + 1, max(rows) + 1)}
        if len(rod) >= 2 and cells == expected | rod:
            return {"dir": (-1, 0), "tail": (max(rows), c), "tip": (tip, c)}
    return None


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    h, w = grid.shape
    arrows = []
    targets = []
    for color, comp in _components_by_color(grid):
        parsed = _parse_arrow(comp)
        if parsed is None:
            targets.append({"color": color, "cells": comp})
        else:
            parsed["color"] = color
            arrows.append(parsed)

    for arrow in arrows:
        dr, dc = arrow["dir"]
        tr, tc = arrow["tip"]
        hit = None
        r, c = tr + dr, tc + dc
        while 0 <= r < h and 0 <= c < w:
            for target in targets:
                if (r, c) in target["cells"]:
                    hit = target
                    break
            if hit is not None:
                break
            r += dr
            c += dc
        if hit is None:
            continue
        for rr, cc in hit["cells"]:
            out[rr, cc] = BG
        r0, c0, r1, c1 = _bbox(hit["cells"])
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        tail_r, tail_c = arrow["tail"]
        if (dr, dc) == (0, 1):
            nr0 = tail_r - height // 2
            nc0 = tail_c - width
        elif (dr, dc) == (0, -1):
            nr0 = tail_r - height // 2
            nc0 = tail_c + 1
        elif (dr, dc) == (1, 0):
            nr0 = tail_r - height
            nc0 = tail_c - width // 2
        else:
            nr0 = tail_r + 1
            nc0 = tail_c - width // 2
        for rr, cc in hit["cells"]:
            nr = nr0 + (rr - r0)
            nc = nc0 + (cc - c0)
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = arrow["color"]
    return out


def _arrow_cells(direction: tuple[int, int], tail: tuple[int, int], length: int) -> set[tuple[int, int]]:
    r, c = tail
    if direction == (0, 1):
        tip_c = c + length
        return {(r, cc) for cc in range(c, tip_c)} | {(r - 1, tip_c), (r, tip_c), (r + 1, tip_c)}
    if direction == (0, -1):
        tip_c = c - length
        return {(r, cc) for cc in range(tip_c + 1, c + 1)} | {(r - 1, tip_c), (r, tip_c), (r + 1, tip_c)}
    if direction == (1, 0):
        tip_r = r + length
        return {(rr, c) for rr in range(r, tip_r)} | {(tip_r, c - 1), (tip_r, c), (tip_r, c + 1)}
    tip_r = r - length
    return {(rr, c) for rr in range(tip_r + 1, r + 1)} | {(tip_r, c - 1), (tip_r, c), (tip_r, c + 1)}


def _target_shape(rng: random.Random) -> set[tuple[int, int]]:
    return rng.choice(
        [
            {(0, 0), (0, 1)},
            {(0, 0), (1, 0)},
            {(0, 0), (0, 1), (1, 0), (1, 1)},
            {(0, 0), (1, 0), (1, 1)},
            {(0, 0), (0, 1), (0, 2)},
        ]
    )


def _stamp(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if r < 0 or r >= h or c < 0 or c >= w or int(grid[r, c]) != BG:
            return False
    for r, c in cells:
        grid[r, c] = color
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        h = rng.randint(16, 28)
        w = rng.randint(16, 28)
        direction = rng.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        length = rng.randint(2, 5)
        arrow_color, target_color = rng.sample(COLORS, 2)
        shape = _target_shape(rng)
        sh_h = max(r for r, _ in shape) + 1
        sh_w = max(c for _, c in shape) + 1
        grid = np.zeros((h, w), dtype=int)
        if direction == (0, 1):
            row = rng.randint(max(2, sh_h), h - max(3, sh_h + 1))
            tail_c = rng.randint(sh_w + 1, w - length - sh_w - 4)
            target_top = row - sh_h // 2
            target_left = tail_c + length + rng.randint(2, 4)
            tail = (row, tail_c)
        elif direction == (0, -1):
            row = rng.randint(max(2, sh_h), h - max(3, sh_h + 1))
            tail_c = rng.randint(length + sh_w + 4, w - sh_w - 2)
            target_top = row - sh_h // 2
            target_left = tail_c - length - rng.randint(2, 4) - sh_w + 1
            tail = (row, tail_c)
        elif direction == (1, 0):
            col = rng.randint(max(2, sh_w), w - max(3, sh_w + 1))
            tail_r = rng.randint(sh_h + 1, h - length - sh_h - 4)
            target_top = tail_r + length + rng.randint(2, 4)
            target_left = col - sh_w // 2
            tail = (tail_r, col)
        else:
            col = rng.randint(max(2, sh_w), w - max(3, sh_w + 1))
            tail_r = rng.randint(length + sh_h + 4, h - sh_h - 2)
            target_top = tail_r - length - rng.randint(2, 4) - sh_h + 1
            target_left = col - sh_w // 2
            tail = (tail_r, col)
        arrow = _arrow_cells(direction, tail, length)
        if not _stamp(grid, arrow, arrow_color):
            continue
        target = {(target_top + r, target_left + c) for r, c in shape}
        if not _stamp(grid, target, target_color):
            continue
        if not np.array_equal(main(grid), grid):
            return grid
    raise RuntimeError("generate_input failed")
