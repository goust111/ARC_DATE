import random
from collections import deque

import numpy as np


BG = 3
YELLOW = 4
PALETTE = (1, 2, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 300


def _components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [p[0] for p in cells]
            cols = [p[1] for p in cells]
            comps.append(
                {
                    "color": color,
                    "cells": cells,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _has_concavity(grid: np.ndarray, comp: dict) -> bool:
    color = int(comp["color"])
    r0, c0, r1, c1 = comp["bbox"]
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if int(grid[r, c]) != BG:
                continue
            neighbors = 0
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if r0 <= nr <= r1 and c0 <= nc <= c1 and int(grid[nr, nc]) == color:
                    neighbors += 1
            if neighbors >= 3:
                return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for comp in _components(g):
        if not _has_concavity(g, comp):
            continue
        r0, c0, r1, c1 = comp["bbox"]
        color = int(comp["color"])
        out[r0 : r1 + 1, c0 : c1 + 1] = color
        marker_col = c1 + 1
        marker_row = (r0 + r1) // 2
        if marker_col < g.shape[1]:
            out[marker_row, marker_col] = YELLOW
    return out


def _rect_mask(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _concave_mask(rng: random.Random, h: int, w: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    mode = rng.choice(("edge_notch", "edge_notch", "inner_hole"))
    if mode == "inner_hole" and h >= 3 and w >= 3:
        rr = rng.randrange(1, h - 1)
        cc = rng.randrange(1, w - 1)
        mask[rr, cc] = False
        return mask

    sides = []
    if w >= 3:
        sides.extend(["top", "bottom"])
    if h >= 3:
        sides.extend(["left", "right"])
    side = rng.choice(sides)
    if side == "top":
        mask[0, rng.randrange(1, w - 1)] = False
    elif side == "bottom":
        mask[h - 1, rng.randrange(1, w - 1)] = False
    elif side == "left":
        mask[rng.randrange(1, h - 1), 0] = False
    else:
        mask[rng.randrange(1, h - 1), w - 1] = False
    return mask


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    h_grid, w_grid = occupied.shape
    r_start = max(0, r0 - 1)
    c_start = max(0, c0 - 1)
    r_end = min(h_grid, r0 + h + 1)
    c_end = min(w_grid, c0 + w + 2)
    return not bool(occupied[r_start:r_end, c_start:c_end].any())


def _paint_shape(grid: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> None:
    rows, cols = np.nonzero(mask)
    grid[r0 + rows, c0 + cols] = color


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(12, 22)
    w = rng.randint(14, 26)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    n_objects = rng.randint(3, 6)
    kinds = ["concave", "convex"] * 3
    rng.shuffle(kinds)
    kinds = kinds[:n_objects]
    if "concave" not in kinds:
        kinds[0] = "concave"
    if "convex" not in kinds:
        kinds[-1] = "convex"

    for kind in kinds:
        placed = False
        for _ in range(80):
            if kind == "convex":
                obj_h = rng.randint(2, 5)
                obj_w = rng.randint(2, 6)
                mask = _rect_mask(obj_h, obj_w)
            else:
                obj_h = rng.randint(3, 6)
                obj_w = rng.randint(3, 6)
                mask = _concave_mask(rng, obj_h, obj_w)

            if obj_w + 2 > w or obj_h + 2 > h:
                continue

            r0 = rng.randint(1, h - obj_h - 1)
            c0 = rng.randint(1, w - obj_w - 2)
            if not _can_place(occupied, r0, c0, obj_h, obj_w):
                continue

            color = rng.choice(PALETTE)
            _paint_shape(grid, mask, color, r0, c0)

            r_start = max(0, r0 - 1)
            c_start = max(0, c0 - 1)
            r_end = min(h, r0 + obj_h + 1)
            c_end = min(w, c0 + obj_w + 2)
            occupied[r_start:r_end, c_start:c_end] = True
            placed = True
            break
        if not placed:
            return None

    comps = _components(grid)
    if len(comps) != n_objects:
        return None

    concave_count = sum(1 for comp in comps if _has_concavity(grid, comp))
    if concave_count == 0 or concave_count == len(comps):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after repeated attempts")
