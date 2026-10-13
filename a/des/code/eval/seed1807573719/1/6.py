from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
ORANGE = 7
SIGNAL_COLORS = (2, 3, 4, 6, 8, 9)
DIRS = {"top": (-1, 0), "bottom": (1, 0), "left": (0, -1), "right": (0, 1)}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _mapping(grid: np.ndarray) -> dict[int, int]:
    w = grid.shape[1]
    mapping = {}
    for r in range(grid.shape[0]):
        a = int(grid[r, w - 2])
        b = int(grid[r, w - 1])
        if a == BG and b == BG:
            if mapping:
                break
            continue
        if a in (BG, BLUE, ORANGE) or b in (BG, BLUE):
            continue
        mapping[a] = b
    assert mapping, "expected right-edge two-column color mapping"
    return mapping


def _clock_specs(grid: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int], int]]:
    h, w = grid.shape
    specs = []
    side_mid = {"top": (0, 1), "bottom": (2, 1), "left": (1, 0), "right": (1, 2)}
    for r in range(h - 2):
        for c in range(w - 2):
            sub = grid[r : r + 3, c : c + 3]
            center = int(sub[1, 1])
            if center in (BG, BLUE, ORANGE):
                continue
            ok = []
            for side, (rr, cc) in side_mid.items():
                ring = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
                if int(sub[rr, cc]) != BG:
                    continue
                if all(int(sub[ar, ac]) == BLUE for ar, ac in ring if (ar, ac) != (rr, cc)):
                    ok.append(side)
            if len(ok) == 1:
                specs.append(((r + 1, c + 1), DIRS[ok[0]], center))
    return specs


def _mirror_orientations(grid: np.ndarray) -> dict[tuple[int, int], str]:
    out = {}
    for comp in _components(grid == ORANGE):
        r0, c0 = comp.min(axis=0)
        r1, c1 = comp.max(axis=0)
        if len(comp) != 3:
            continue
        if r0 == r1 and c1 - c0 == 2:
            orient = "horizontal"
        elif c0 == c1 and r1 - r0 == 2:
            orient = "vertical"
        else:
            continue
        for r, c in comp:
            out[(int(r), int(c))] = orient
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mapping = _mapping(g)
    mirrors = _mirror_orientations(g)
    out = g.copy()
    painted: set[tuple[int, int]] = set()

    for center, direction, color in _clock_specs(g):
        r, c = center
        dr, dc = direction
        current = int(color)
        for _ in range(h * w * 2):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                break
            cell = (nr, nc)
            value = int(g[nr, nc])
            if value == BLUE or cell in painted:
                break
            if cell in mirrors:
                orient = mirrors[cell]
                if orient == "horizontal" and dc == 0:
                    dr = -dr
                    current = mapping.get(current, current)
                elif orient == "vertical" and dr == 0:
                    dc = -dc
                    current = mapping.get(current, current)
                r, c = nr, nc
                continue
            if value != BG:
                break
            out[nr, nc] = current
            painted.add(cell)
            r, c = nr, nc
    return out


def _place_clock(grid: np.ndarray, top: int, left: int, side: str, color: int) -> bool:
    patch = np.full((3, 3), BLUE, dtype=int)
    patch[1, 1] = color
    rr, cc = {"top": (0, 1), "bottom": (2, 1), "left": (1, 0), "right": (1, 2)}[side]
    patch[rr, cc] = BG
    region = grid[top : top + 3, left : left + 3]
    if region.shape != (3, 3) or np.any(region != BG):
        return False
    region[:, :] = patch
    return True


def _place_mirror(grid: np.ndarray, r: int, c: int, orient: str) -> bool:
    cells = [(r, c + i) for i in range(3)] if orient == "horizontal" else [(r + i, c) for i in range(3)]
    if any(not (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]) or grid[rr, cc] != BG for rr, cc in cells):
        return False
    for rr, cc in cells:
        grid[rr, cc] = ORANGE
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 25)
    w = random.randint(18, 28)
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample(list(SIGNAL_COLORS), 4)
    for i in range(3):
        grid[i, w - 2] = colors[i]
        grid[i, w - 1] = colors[i + 1]

    count = random.randint(1, 3)
    used = np.zeros((h, w), dtype=bool)
    used[:4, w - 3 : w] = True
    for i in range(count):
        for _ in range(100):
            side = random.choice(list(DIRS))
            top = random.randint(2, h - 5)
            left = random.randint(2, w - 7)
            if used[max(0, top - 2) : min(h, top + 5), max(0, left - 2) : min(w, left + 5)].any():
                continue
            color = colors[i]
            if not _place_clock(grid, top, left, side, color):
                continue
            used[max(0, top - 2) : min(h, top + 5), max(0, left - 2) : min(w, left + 5)] = True
            cr, cc = top + 1, left + 1
            dr, dc = DIRS[side]
            if random.random() < 0.8:
                dist = random.randint(3, 6)
                mr, mc = cr + dr * dist, cc + dc * dist
                if 2 <= mr < h - 3 and 2 <= mc < w - 4:
                    orient = "horizontal" if dc == 0 else "vertical"
                    _place_mirror(grid, mr, mc - 1 if orient == "horizontal" else mc, orient)
            break
        else:
            return None
    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate clock-mirror input")
