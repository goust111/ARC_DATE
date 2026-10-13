import random
from collections import deque
import numpy as np


def _comps_color(g, color):
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or g[r, c] != color:
                continue
            q = deque([(r, c)]); seen[r, c] = True; cells = []
            while q:
                x, y = q.popleft(); cells.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and g[nx, ny] == color:
                        seen[nx, ny] = True; q.append((nx, ny))
            rs = [x for x, _ in cells]; cs = [y for _, y in cells]
            comps.append({"bbox": (min(rs), max(rs), min(cs), max(cs)), "cells": cells})
    return comps


def _edges(rects):
    es = []
    for i, a in enumerate(rects):
        ar0, ar1, ac0, ac1 = a["bbox"]
        for j in range(i + 1, len(rects)):
            br0, br1, bc0, bc1 = rects[j]["bbox"]
            ov_r = (max(ar0 + 1, br0 + 1), min(ar1 - 1, br1 - 1))
            ov_c = (max(ac0 + 1, bc0 + 1), min(ac1 - 1, bc1 - 1))
            if ac1 < bc0 and ov_r[0] <= ov_r[1]:
                es.append((bc0 - ac1 - 1, i, j, "h", ov_r))
            if bc1 < ac0 and ov_r[0] <= ov_r[1]:
                es.append((ac0 - bc1 - 1, i, j, "h", ov_r))
            if ar1 < br0 and ov_c[0] <= ov_c[1]:
                es.append((br0 - ar1 - 1, i, j, "v", ov_c))
            if br1 < ar0 and ov_c[0] <= ov_c[1]:
                es.append((ar0 - br1 - 1, i, j, "v", ov_c))
    return sorted(es)


def _mst(rects):
    parent = list(range(len(rects)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    tree = []
    for e in _edges(rects):
        _, i, j, _, _ = e
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            tree.append(e)
    return tree


def _marker(g, rect):
    r0, r1, c0, c1 = rect["bbox"]
    vals = [(r, c, int(g[r, c])) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if g[r, c] not in (0, 3)]
    return vals[0][2] if vals else 2


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    rects = _comps_color(g, 3)
    tree = _mst(rects)
    out = g.copy()
    adj = {i: [] for i in range(len(rects))}
    for e in tree:
        dist, i, j, orient, span = e
        adj[i].append((dist, j, e)); adj[j].append((dist, i, e))
        a, b = rects[i]["bbox"], rects[j]["bbox"]
        if orient == "h":
            rr = (span[0] + span[1]) // 2
            c0, c1 = sorted([a[3], b[2]]) if a[3] < b[2] else sorted([b[3], a[2]])
            out[rr, c0:c1 + 1] = 0
            out[rr - 1, c0 + 1:c1] = 3
            out[rr + 1, c0 + 1:c1] = 3
        else:
            cc = (span[0] + span[1]) // 2
            r0, r1 = sorted([a[1], b[0]]) if a[1] < b[0] else sorted([b[1], a[0]])
            out[r0:r1 + 1, cc] = 0
            out[r0 + 1:r1, cc - 1] = 3
            out[r0 + 1:r1, cc + 1] = 3
    root = max(range(len(rects)), key=lambda i: (rects[i]["bbox"][1] - rects[i]["bbox"][0] - 1) * (rects[i]["bbox"][3] - rects[i]["bbox"][2] - 1))
    order = {}
    q = deque([root]); order[root] = 0
    while q:
        x = q.popleft()
        for _, y, _ in sorted(adj[x]):
            if y not in order:
                order[y] = len(order); q.append(y)
    for i, rect in enumerate(rects):
        col = _marker(g, rect)
        r0, r1, c0, c1 = rect["bbox"]
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                out[r, c] = col if (r + c) % 4 == order.get(i, 0) % 4 else 0
    return out


def _rect(g, top, left, ht, wd):
    g[top, left:left + wd] = 3; g[top + ht - 1, left:left + wd] = 3
    g[top:top + ht, left] = 3; g[top:top + ht, left + wd - 1] = 3


def generate_input() -> np.ndarray:
    g = np.zeros((22, 22), dtype=int)
    rects = [(1, 1, 7, 7), (1, 13, 5, 6), (11, 2, 6, 6), (11, 13, 7, 7)]
    if random.choice([True, False]):
        rects.append((7, 8, 4, 5))
    cols = random.sample([1, 2, 4, 6, 7, 8, 9], len(rects))
    random.shuffle(rects)
    for (top, left, ht, wd), col in zip(rects, cols):
        _rect(g, top, left, ht, wd)
        g[random.randint(top + 1, top + ht - 2), random.randint(left + 1, left + wd - 2)] = col
    return g
