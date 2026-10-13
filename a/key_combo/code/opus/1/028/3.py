import random
from collections import deque
import numpy as np


DIR8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)


def _components(g):
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == 0:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                x, y = q.popleft()
                cells.append((x, y))
                for dx, dy in DIR8:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and int(g[nx, ny]) == color:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            comps.append({"color": color, "cells": cells, "bbox": (min(rs), min(cs), max(rs), max(cs))})
    return comps


def _strict_contains(a, b):
    return a[0] < b[0] and a[1] < b[1] and a[2] > b[2] and a[3] > b[3]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    comps = _components(g)
    parent = {}
    for i, child in enumerate(comps):
        candidates = [
            (j, par) for j, par in enumerate(comps)
            if i != j and par["color"] != child["color"] and _strict_contains(par["bbox"], child["bbox"])
        ]
        if candidates:
            parent[i] = min(candidates, key=lambda x: (x[1]["bbox"][2] - x[1]["bbox"][0] + 1) * (x[1]["bbox"][3] - x[1]["bbox"][1] + 1))[0]
    children = {}
    for child, par in parent.items():
        children.setdefault(par, []).append(child)
    order = sorted(children, key=lambda i: (comps[i]["bbox"][2] - comps[i]["bbox"][0] + 1) * (comps[i]["bbox"][3] - comps[i]["bbox"][1] + 1))
    out = g.copy()
    for par in order:
        kids = children[par]
        if len(kids) == 1:
            for r, c in comps[kids[0]]["cells"]:
                out[r, c] = 0
        elif len(kids) == 2:
            x, y = kids
            bx, by = comps[x]["bbox"], comps[y]["bbox"]
            x_offsets = [(r - bx[0], c - bx[1]) for r, c in comps[x]["cells"]]
            y_offsets = [(r - by[0], c - by[1]) for r, c in comps[y]["cells"]]
            for k in (x, y):
                for r, c in comps[k]["cells"]:
                    out[r, c] = 0
            color = comps[par]["color"]
            for dr, dc in x_offsets:
                out[by[0] + dr, by[1] + dc] = color
            for dr, dc in y_offsets:
                out[bx[0] + dr, bx[1] + dc] = color
    return out


def _draw_frame(g, top, left, hh, ww, color):
    g[top, left:left + ww] = color
    g[top + hh - 1, left:left + ww] = color
    g[top:top + hh, left] = color
    g[top:top + hh, left + ww - 1] = color


def generate_input() -> np.ndarray:
    g = np.zeros((18, 18), dtype=int)
    spots = [(1, 1), (1, 10), (10, 1), (10, 10)]
    random.shuffle(spots)
    container_colors = random.sample([1, 2, 4, 7], 4)
    inner_colors = [3, 5, 6, 8, 9]
    counts = [1, 2, random.choice([1, 2]), random.choice([1, 2])]
    if 1 not in counts:
        counts[0] = 1
    if 2 not in counts:
        counts[1] = 2
    for idx, ((top, left), color, count) in enumerate(zip(spots, container_colors, counts)):
        _draw_frame(g, top, left, 7, 7, color)
        colors = random.sample(inner_colors, count)
        if count == 1:
            cells = [(top + 3, left + 3), (top + 3, left + 4), (top + 4, left + 3)]
            for r, c in cells:
                g[r, c] = colors[0]
        else:
            for r, c in [(top + 2, left + 2), (top + 2, left + 3), (top + 3, left + 2)]:
                g[r, c] = colors[0]
            for r, c in [(top + 4, left + 4), (top + 4, left + 5), (top + 5, left + 4)]:
                g[r, c] = colors[1]
    return g
