import random
from collections import deque
import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                x, y = q.popleft()
                cells.append((x, y))
                for dx, dy in DIR4:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and int(grid[nx, ny]) == color:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(cells)
    return comps


def _wall_direction(g):
    h, w = g.shape
    if np.all(g[0, :] == 2):
        return (-1, 0), lambda cells: min(r for r, _ in cells)
    if np.all(g[h - 1, :] == 2):
        return (1, 0), lambda cells: -max(r for r, _ in cells)
    if np.all(g[:, 0] == 2):
        return (0, -1), lambda cells: min(c for _, c in cells)
    if np.all(g[:, w - 1] == 2):
        return (0, 1), lambda cells: -max(c for _, c in cells)
    return (1, 0), lambda cells: -max(r for r, _ in cells)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    forbidden = set()
    for gray in _components(g, 5):
        blue = set()
        for r, c in gray:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == 1:
                    blue.add((nr, nc))
        radius = len(blue)
        for r, c in gray:
            for dr in range(-radius, radius + 1):
                rem = radius - abs(dr)
                for dc in range(-rem, rem + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        forbidden.add((nr, nc))

    dominoes = []
    for color in sorted(set(map(int, np.unique(g))) - {0, 1, 2, 5}):
        for cells in _components(g, color):
            if len(cells) == 2:
                dominoes.append({"color": color, "cells": cells})

    out = g.copy()
    for dom in dominoes:
        for r, c in dom["cells"]:
            out[r, c] = 0
    direction, key = _wall_direction(g)
    occupied = {(r, c) for r in range(h) for c in range(w) if int(out[r, c]) == 2}
    occupied |= forbidden
    for dom in sorted(dominoes, key=lambda d: key(d["cells"])):
        cells = list(dom["cells"])
        while True:
            nxt = [(r + direction[0], c + direction[1]) for r, c in cells]
            if any(r < 0 or r >= h or c < 0 or c >= w or (r, c) in occupied for r, c in nxt):
                break
            cells = nxt
        for r, c in cells:
            out[r, c] = dom["color"]
            occupied.add((r, c))
    out[out == 1] = 5
    return out


def _can_place(g, cells):
    h, w = g.shape
    if any(r < 1 or r >= h - 1 or c < 1 or c >= w - 1 or g[r, c] != 0 for r, c in cells):
        return False
    for r, c in cells:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and g[nr, nc] not in (0,):
                return False
    return True


def generate_input() -> np.ndarray:
    h = random.randint(14, 18)
    w = random.randint(14, 18)
    g = np.zeros((h, w), dtype=int)
    g[h - 1, :] = 2
    obstacle_specs = [
        ([(6, 3), (6, 4)], [(5, 3)]),
        ([(8, w - 5), (9, w - 5), (9, w - 4)], [(8, w - 6), (10, w - 5)]),
        ([(4, w // 2), (4, w // 2 + 1)], [(3, w // 2), (5, w // 2), (4, w // 2 - 1)]),
    ]
    for gray, blue in random.sample(obstacle_specs, random.randint(2, 3)):
        for r, c in gray:
            g[r, c] = 5
        for r, c in blue:
            if 0 <= r < h and 0 <= c < w and g[r, c] == 0:
                g[r, c] = 1
    colors = random.sample([3, 4, 6, 7], 4)
    cols = random.sample(range(1, w - 2), 4)
    for color, col in zip(colors, cols):
        for _ in range(100):
            row = random.randint(0, 4)
            if random.choice([True, False]):
                cells = [(row, col), (row + 1, col)]
            else:
                cells = [(row, col), (row, col + 1)]
            if _can_place(g, cells):
                for r, c in cells:
                    g[r, c] = color
                break
    return g
