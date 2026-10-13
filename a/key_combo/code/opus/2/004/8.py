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
    p = list(range(len(rects)))
    def f(x):
        while p[x] != x:
            p[x] = p[p[x]]; x = p[x]
        return x
    tree = []
    for e in _edges(rects):
        _, i, j, _, _ = e
        a, b = f(i), f(j)
        if a != b:
            p[a] = b; tree.append(e)
    return tree


def _source_and_motif(g, rects):
    for i, rect in enumerate(rects):
        r0, r1, c0, c1 = rect["bbox"]
        gray_cols = [c for c in range(c0 + 1, c1) if sum(g[r, c] == 5 for r in range(r0 + 1, r1)) >= 2]
        if gray_cols:
            gc = gray_cols[0]
            rows = [r for r in range(r0 + 1, r1) if g[r, gc] == 5]
            motif = [[int(g[r, c]) for c in range(gc + 1, c1)] for r in rows]
            return i, np.array(motif, dtype=int)
    return 0, np.zeros((1, 1), dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    rects = _comps_color(g, 1)
    root, motif = _source_and_motif(g, rects)
    hmot, wmot = motif.shape
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
            out[rr, c0:c1 + 1] = 0; out[rr - 1, c0 + 1:c1] = 1; out[rr + 1, c0 + 1:c1] = 1
        else:
            cc = (span[0] + span[1]) // 2
            r0, r1 = sorted([a[1], b[0]]) if a[1] < b[0] else sorted([b[1], a[0]])
            out[r0:r1 + 1, cc] = 0; out[r0 + 1:r1, cc - 1] = 1; out[r0 + 1:r1, cc + 1] = 1
    order = {root: 0}; q = deque([root])
    while q:
        x = q.popleft()
        for _, y, _ in sorted(adj[x]):
            if y not in order:
                order[y] = len(order); q.append(y)
    for i, rect in enumerate(rects):
        r0, r1, c0, c1 = rect["bbox"]
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                out[r, c] = 0
        if i == root:
            continue
        k = order.get(i, 0)
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                out[r, c] = motif[(r - r0 - 1 + k) % hmot, (c - c0 - 1) % wmot]
    return out


def _rect(g, top, left, ht, wd):
    g[top, left:left + wd] = 1; g[top + ht - 1, left:left + wd] = 1
    g[top:top + ht, left] = 1; g[top:top + ht, left + wd - 1] = 1


def generate_input() -> np.ndarray:
    g = np.zeros((22, 22), dtype=int)
    rects = [(1, 1, 8, 8), (1, 13, 5, 6), (12, 2, 6, 6), (12, 13, 6, 7)]
    if random.choice([True, False]):
        rects.append((7, 9, 5, 5))
    for top, left, ht, wd in rects:
        _rect(g, top, left, ht, wd)
    src = rects[0]
    top, left, ht, wd = src
    hmot = random.randint(2, 3)
    gray_c = left + 1
    for r in range(top + 1, top + 1 + hmot):
        g[r, gray_c] = 5
    colors = random.sample([2, 3, 4, 6, 7, 8, 9], 3)
    for rr in range(hmot):
        for cc in range(1, min(4, wd - 2)):
            if random.choice([True, False]) or cc == 1:
                g[top + 1 + rr, gray_c + cc] = random.choice(colors)
    return g
