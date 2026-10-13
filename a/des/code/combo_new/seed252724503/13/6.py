import random

import numpy as np

BLACK = 0
BLUE = 1
YELLOW = 4
GREY = 5
AZURE = 8
MAROON = 9


def _object_stats(mask: np.ndarray):
    seen = np.zeros_like(mask, dtype=bool)
    objects = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            objects.append(
                {
                    "cells": cells,
                    "r0": min(rows),
                    "r1": max(rows),
                    "c0": min(cols),
                    "c1": max(cols),
                }
            )
    return objects


def _direction_from_light(pos, shape):
    r, c = pos
    h, w = shape
    if r == 0:
        return (1, 0)
    if r == h - 1:
        return (-1, 0)
    if c == 0:
        return (0, 1)
    return (0, -1)


def _shadow_color(blockers: int) -> int:
    return AZURE if blockers == 0 else BLUE if blockers == 1 else MAROON


def _shadow_shift(obj, direction):
    dr, dc = direction
    if dr != 0:
        return dr * (obj["r1"] - obj["r0"] + 1), 0
    return 0, dc * (obj["c1"] - obj["c0"] + 1)


def _blocker_count(obj, others, direction):
    dr, dc = direction
    count = 0
    for other in others:
        if other is obj:
            continue
        if dr != 0:
            overlap = not (other["c1"] < obj["c0"] or other["c0"] > obj["c1"])
            in_front = other["r1"] < obj["r0"] if dr > 0 else other["r0"] > obj["r1"]
        else:
            overlap = not (other["r1"] < obj["r0"] or other["r0"] > obj["r1"])
            in_front = other["c1"] < obj["c0"] if dc > 0 else other["c0"] > obj["c1"]
        if overlap and in_front:
            count += 1
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each grey object to its back side and color the copy by how many objects block the light.

    Input structure:
    - Black background with one yellow edge pixel acting as a light source.
    - Several disconnected grey foreground objects lie inside the grid.
    - The light source implies a cardinal illumination direction from the touched edge into the grid.

    Rule:
    - For each grey object, count how many other grey objects lie closer to the light along overlapping beam columns or rows.
    - Place one translated copy of the object immediately behind it on the dark side.
    - Color that copy azure for 0 blockers, blue for 1 blocker, and maroon for 2 or more blockers.

    Key insight:
    - Shadow placement depends on the light direction, but shadow color depends on the number of foreground layers already encountered along the same beam path.
    """
    out = grid.copy()
    light = tuple(zip(*np.where(grid == YELLOW)))[0]
    direction = _direction_from_light(light, grid.shape)
    objects = _object_stats(grid == GREY)
    for obj in objects:
        color = _shadow_color(_blocker_count(obj, objects, direction))
        sr, sc = _shadow_shift(obj, direction)
        for r, c in obj["cells"]:
            nr, nc = r + sr, c + sc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and out[nr, nc] == BLACK:
                out[nr, nc] = color
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()

    def build_top_case():
        h = rng.randint(16, 24)
        w = rng.randint(16, 24)
        grid = np.zeros((h, w), dtype=int)
        grid[0, w // 2] = YELLOW
        objects = []
        occ = set()
        for _ in range(rng.randint(2, 4)):
            for _ in range(120):
                oh = rng.randint(2, 3)
                ow = rng.randint(2, 4)
                r0 = rng.randint(2, h - 2 * oh - 2)
                c0 = rng.randint(1, w - ow - 2)
                cells = {(r, c) for r in range(r0, r0 + oh) for c in range(c0, c0 + ow)}
                shadow = {(r + oh, c) for r, c in cells}
                if any(p in occ or p == (0, w // 2) for p in cells | shadow):
                    continue
                if any(r == 0 for r, _ in shadow) or any(r >= h or c >= w for r, c in shadow):
                    continue
                for r, c in cells:
                    grid[r, c] = GREY
                occ |= cells | shadow
                objects.append((r0, c0, oh, ow))
                break
            else:
                return None
        return grid

    for _ in range(400):
        grid = build_top_case()
        if grid is None:
            continue
        k = rng.randrange(4)
        rotated = np.rot90(grid, k)
        return rotated.copy()
    raise RuntimeError("failed to generate task 13/6")
