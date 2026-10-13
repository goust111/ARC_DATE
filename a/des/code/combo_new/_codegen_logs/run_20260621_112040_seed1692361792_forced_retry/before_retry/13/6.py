import random
from collections import deque

import numpy as np


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _checker(h, w):
    grid = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            grid[r, c] = 0 if (r + c) % 2 == 0 else 8
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each hollow foreground object toward its off-center hole by a distance equal to that hole's area.

    Input structure:
    - The background is a two-color checkerboard.
    - Several monochrome foreground objects sit on the checkerboard; each object is roughly rectangular and contains one rectangular checkerboard hole fully enclosed inside it.
    - The hole is offset toward one side of the object rather than centered.

    Rule:
    - For each object, locate its enclosed checkerboard hole and measure the hole's area in pixels.
    - Determine whether the hole is offset upward, downward, leftward, or rightward relative to the object's center.
    - Translate the entire object, including its hole, in that direction by exactly the hole area.
    - Restore the vacated cells to the original checkerboard pattern.

    Key insight:
    - The hole encodes two attributes at once: its offset gives the movement direction, and its pixel count gives the movement distance.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg_colors = set(int(g[r, c]) for r in range(h) for c in range(w) if (r + c) % 2 == 0) | set(int(g[r, c]) for r in range(h) for c in range(w) if (r + c) % 2 == 1)
    out = _checker(h, w)
    object_colors = sorted(set(map(int, g.ravel())) - bg_colors)
    for color in object_colors:
        for comp in _components(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            bbox_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for r, c in comp:
                bbox_mask[r - r0, c - c0] = True
            holes = []
            for bg_comp in _components(~bbox_mask):
                if any(r in (0, bbox_mask.shape[0] - 1) or c in (0, bbox_mask.shape[1] - 1) for r, c in bg_comp):
                    continue
                holes.append(bg_comp)
            assert len(holes) == 1
            hole = holes[0]
            area = len(hole)
            obj_center = ((r0 + r1) / 2.0, (c0 + c1) / 2.0)
            hr0, hc0, hr1, hc1 = _bbox(hole)
            hole_center = (r0 + (hr0 + hr1) / 2.0, c0 + (hc0 + hc1) / 2.0)
            dr = hole_center[0] - obj_center[0]
            dc = hole_center[1] - obj_center[1]
            if abs(dr) >= abs(dc):
                step = (area if dr > 0 else -area, 0)
            else:
                step = (0, area if dc > 0 else -area)
            for r, c in comp:
                out[r + step[0], c + step[1]] = color
    return out


def generate_input() -> np.ndarray:
    for _ in range(500):
        h = random.randint(14, 24)
        w = random.randint(14, 24)
        grid = _checker(h, w)
        n = random.randint(2, 4)
        colors = random.sample([1, 2, 3, 4, 5, 6, 7, 9], n)
        occupied = np.zeros((h, w), dtype=bool)
        ok = True
        for color in colors:
            placed = False
            for _attempt in range(120):
                oh = random.randint(5, 8)
                ow = random.randint(5, 8)
                hh = random.randint(1, 2)
                hw = random.randint(2, 3)
                if oh - hh - 1 < 1 or ow - hw - 1 < 1:
                    continue
                area = hh * hw
                direction = random.choice(["up", "down", "left", "right"])
                r_min = 1 + area
                r_max = h - oh - 2 - area
                c_min = 1 + area
                c_max = w - ow - 2 - area
                if r_min > r_max or c_min > c_max:
                    continue
                r0 = random.randint(r_min, r_max)
                c0 = random.randint(c_min, c_max)
                hole_r = r0 + (1 if direction == "up" else oh - hh - 1 if direction == "down" else random.randint(1, oh - hh - 1))
                hole_c = c0 + (1 if direction == "left" else ow - hw - 1 if direction == "right" else random.randint(1, ow - hw - 1))
                shift = (-area, 0) if direction == "up" else (area, 0) if direction == "down" else (0, -area) if direction == "left" else (0, area)
                new_box = (r0 + shift[0], c0 + shift[1], r0 + shift[0] + oh - 1, c0 + shift[1] + ow - 1)
                if new_box[0] < 0 or new_box[1] < 0 or new_box[2] >= h or new_box[3] >= w:
                    continue
                cells = {(r, c) for r in range(r0, r0 + oh) for c in range(c0, c0 + ow)}
                move_cells = {(r + shift[0], c + shift[1]) for r, c in cells}
                if any(occupied[r, c] for r, c in cells | move_cells):
                    continue
                for r in range(r0, r0 + oh):
                    for c in range(c0, c0 + ow):
                        if hole_r <= r < hole_r + hh and hole_c <= c < hole_c + hw:
                            continue
                        grid[r, c] = color
                        occupied[r, c] = True
                for r in range(r0, r0 + oh):
                    for c in range(c0, c0 + ow):
                        occupied[r, c] = True
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return grid
    raise RuntimeError("failed to generate valid input")
