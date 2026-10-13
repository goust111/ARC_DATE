import random
from collections import deque
from typing import List, Optional, Sequence, Tuple

import numpy as np


def _components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: List[List[Tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: List[Tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_rectangle_and_indicator(
    grid: np.ndarray, comp: Sequence[Tuple[int, int]]
) -> Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int]]]:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    candidates = [
        (min_r, max_r, min_c, max_c),
        (min_r + 1, max_r, min_c, max_c),
        (min_r, max_r - 1, min_c, max_c),
        (min_r, max_r, min_c + 1, max_c),
        (min_r, max_r, min_c, max_c - 1),
    ]

    comp_set = set(comp)
    for top, bottom, left, right in candidates:
        if bottom - top + 1 < 4 or right - left + 1 < 4:
            continue
        rect_cells = {(r, c) for r in range(top, bottom + 1) for c in range(left, right + 1)}
        if not rect_cells.issubset(comp_set):
            continue
        outside = list(comp_set - rect_cells)
        if len(outside) != 1:
            continue
        ind_r, ind_c = outside[0]
        if not (top <= ind_r <= bottom and left - 1 <= ind_c <= right + 1) and not (
            left <= ind_c <= right and top - 1 <= ind_r <= bottom + 1
        ):
            continue
        if min(abs(ind_r - top), abs(ind_r - bottom), abs(ind_c - left), abs(ind_c - right)) > 1:
            continue
        colors = {int(grid[r, c]) for r, c in rect_cells}
        if len(colors) != 2 or 0 in colors or int(grid[ind_r, ind_c]) not in colors:
            continue
        return (top, bottom, left, right), (ind_r, ind_c)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Parse:
        Group each patterned non-black rectangle together with its unique
        adjacent indicator pixel.
    Analyze:
        Recover the true rectangle bounds, read the indicator color as the
        keep-color, and use the indicator row as the output sort key.
    Transform:
        Erase every non-keep color inside the rectangle to black while keeping
        the rectangle crop size unchanged.
    Return:
        Stack the cropped erase-results from smallest indicator row to largest
        in a left-aligned output grid padded with black.
    """
    grid = np.asarray(grid, dtype=int)
    pieces = []
    for comp in _components(grid != 0):
        found = _find_rectangle_and_indicator(grid, comp)
        if found is None:
            continue
        (top, bottom, left, right), (ind_r, ind_c) = found
        keep_color = int(grid[ind_r, ind_c])
        block = grid[top : bottom + 1, left : right + 1].copy()
        block[block != keep_color] = 0
        pieces.append({"row": ind_r, "block": block})

    pieces.sort(key=lambda item: item["row"])
    width = max(piece["block"].shape[1] for piece in pieces)
    height = sum(piece["block"].shape[0] for piece in pieces)
    out = np.zeros((height, width), dtype=int)
    r0 = 0
    for piece in pieces:
        block = piece["block"]
        out[r0 : r0 + block.shape[0], : block.shape[1]] = block
        r0 += block.shape[0]
    return out


def _pattern(height: int, width: int, colors: Tuple[int, int]) -> np.ndarray:
    a, b = colors
    kind = random.choice(["h", "v", "checker"])
    grid = np.zeros((height, width), dtype=int)
    for r in range(height):
        for c in range(width):
            if kind == "h":
                grid[r, c] = a if r % 2 == 0 else b
            elif kind == "v":
                grid[r, c] = a if c % 2 == 0 else b
            else:
                grid[r, c] = a if (r + c) % 2 == 0 else b
    return grid


def generate_input() -> np.ndarray:
    h = 16
    w = 16
    positions = [(1, 1), (6, 9), (10, 2)]
    pairs = random.sample(
        [(1, 2), (1, 3), (1, 4), (1, 7), (2, 3), (2, 4), (2, 7), (3, 4), (3, 7), (4, 7)],
        3,
    )

    for _ in range(5000):
        grid = np.zeros((h, w), dtype=int)
        rows = []
        ok = True
        for (top, left), colors in zip(positions, pairs):
            rect_h = random.randint(4, 5)
            rect_w = random.randint(4, 5)
            block = _pattern(rect_h, rect_w, colors)
            grid[top : top + rect_h, left : left + rect_w] = block

            sides = []
            if top > 0:
                sides.append("up")
            if top + rect_h < h:
                sides.append("down")
            if left > 0:
                sides.append("left")
            if left + rect_w < w:
                sides.append("right")
            random.shuffle(sides)
            placed = False
            for side in sides:
                if side == "up":
                    ind_r = top - 1
                    ind_c = random.randint(left, left + rect_w - 1)
                elif side == "down":
                    ind_r = top + rect_h
                    ind_c = random.randint(left, left + rect_w - 1)
                elif side == "left":
                    ind_r = random.randint(top, top + rect_h - 1)
                    ind_c = left - 1
                else:
                    ind_r = random.randint(top, top + rect_h - 1)
                    ind_c = left + rect_w
                if grid[ind_r, ind_c] != 0 or ind_r in rows:
                    continue
                grid[ind_r, ind_c] = random.choice(colors)
                rows.append(ind_r)
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return grid
    raise RuntimeError("failed to generate a valid task-8 input")
