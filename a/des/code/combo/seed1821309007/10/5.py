import random
from collections import deque

import numpy as np


BG = 7       # orange
RED = 2
GREEN = 3
BLUE = 1
OBJECT_COLORS = [0, 4, 5, 6, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400


def _components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            comps.append(
                {
                    "color": color,
                    "cells": cells,
                    "bbox": (min(rs), min(cs), max(rs), max(cs)),
                }
            )
    comps.sort(key=lambda comp: comp["bbox"])
    return comps


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int, h: int, w: int) -> list[tuple[int, int]]:
    cells = []
    for c in range(c0, c1 + 1):
        if 0 <= r0 < h and 0 <= c < w:
            cells.append((r0, c))
        if 0 <= r1 < h and 0 <= c < w and r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        if 0 <= r < h and 0 <= c0 < w:
            cells.append((r, c0))
        if 0 <= r < h and 0 <= c1 < w and c1 != c0:
            cells.append((r, c1))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    for comp in _components(g, bg=BG):
        r0, c0, r1, c1 = comp["bbox"]
        color = comp["color"]
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        mask = patch == color
        bh, bw = mask.shape

        if bh == bw:
            transformed = np.rot90(mask, -1)
            frame_color = RED
        elif bw > bh:
            transformed = np.fliplr(mask)
            frame_color = GREEN
        else:
            transformed = np.flipud(mask)
            frame_color = BLUE

        out[r0 : r1 + 1, c0 : c1 + 1] = BG
        out[r0 : r1 + 1, c0 : c1 + 1][transformed] = color

        for rr, cc in _perimeter_cells(r0 - 1, c0 - 1, r1 + 1, c1 + 1, h, w):
            out[rr, cc] = frame_color

    return out


def _mask_connected(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if len(points) == 0:
        return False
    start = tuple(points[0])
    q = deque([start])
    seen = {start}
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(points)


def _make_connected_mask(height: int, width: int, rng: random.Random) -> np.ndarray:
    area = height * width
    for _ in range(80):
        mask = np.ones((height, width), dtype=bool)
        max_remove = min(area // 3, area - max(height, width) - 1)
        min_remove = 1 if area >= 6 else 0
        remove_goal = rng.randint(min_remove, max_remove) if max_remove >= min_remove else 0

        cells = [(r, c) for r in range(height) for c in range(width)]
        rng.shuffle(cells)
        removed = 0
        for r, c in cells:
            if removed >= remove_goal:
                break
            if not mask[r, c]:
                continue
            candidate = mask.copy()
            candidate[r, c] = False
            if not candidate[0, :].any():
                continue
            if not candidate[-1, :].any():
                continue
            if not candidate[:, 0].any():
                continue
            if not candidate[:, -1].any():
                continue
            if not _mask_connected(candidate):
                continue
            mask = candidate
            removed += 1

        if _mask_connected(mask):
            return mask
    return np.ones((height, width), dtype=bool)


def _category_dims(category: str, rng: random.Random) -> tuple[int, int]:
    if category == "square":
        side = rng.randint(2, 5)
        return side, side
    if category == "wide":
        height = rng.randint(2, 4)
        width = rng.randint(height + 1, min(7, height + 3))
        return height, width
    width = rng.randint(2, 4)
    height = rng.randint(width + 1, min(7, width + 3))
    return height, width


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _try_place_objects(height: int, width: int, categories: list[str], rng: random.Random) -> np.ndarray | None:
    grid = np.full((height, width), BG, dtype=int)
    reserved = []
    color_pool = OBJECT_COLORS.copy()
    rng.shuffle(color_pool)

    for i, category in enumerate(categories):
        bh, bw = _category_dims(category, rng)
        mask = _make_connected_mask(bh, bw, rng)
        color = color_pool[i % len(color_pool)]

        placed = False
        for _ in range(120):
            r0 = rng.randint(2, height - bh - 3)
            c0 = rng.randint(2, width - bw - 3)
            bbox = (r0, c0, r0 + bh - 1, c0 + bw - 1)
            clearance = (r0 - 2, c0 - 2, r0 + bh + 1, c0 + bw + 1)
            if any(_rects_overlap(clearance, other) for other in reserved):
                continue
            patch = grid[r0 : r0 + bh, c0 : c0 + bw]
            if np.any(patch != BG):
                continue
            patch[mask] = color
            reserved.append(clearance)
            placed = True
            break
        if not placed:
            return None

    return grid


def _branch_list(rng: random.Random) -> list[str]:
    count = rng.randint(3, 6)
    categories = ["square", "wide", "tall"]
    extras = count - 3
    if extras > 0:
        extra_pool = ["square", "wide", "tall"] * 2
        rng.shuffle(extra_pool)
        categories.extend(extra_pool[:extras])
    rng.shuffle(categories)
    return categories


def _has_all_branches(grid: np.ndarray) -> bool:
    found = set()
    for comp in _components(grid, bg=BG):
        r0, c0, r1, c1 = comp["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h == w:
            found.add("square")
        elif w > h:
            found.add("wide")
        else:
            found.add("tall")
    return found == {"square", "wide", "tall"}


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        height = rng.randint(15, 24)
        width = rng.randint(15, 24)
        categories = _branch_list(rng)
        grid = _try_place_objects(height, width, categories, rng)
        if grid is None:
            continue
        if len(_components(grid, bg=BG)) != len(categories):
            continue
        if not _has_all_branches(grid):
            continue
        return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
