import random
from collections import deque

import numpy as np


def _components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == 0 or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and int(grid[ny, nx]) == color:
                        seen[ny, nx] = True
                        q.append((ny, nx))
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


def main(grid: np.ndarray) -> np.ndarray:
    frames = _components(grid)
    for f in frames:
        r1, c1, r2, c2 = f["bbox"]
        area = 2 * ((r2 - r1 + 1) + (c2 - c1 + 1)) - 4
        f["valid_frame"] = len(f["cells"]) == area

    frames = [f for f in frames if f["valid_frame"]]
    for child in frames:
        containers = []
        cr1, cc1, cr2, cc2 = child["bbox"]
        for parent in frames:
            if parent is child:
                continue
            pr1, pc1, pr2, pc2 = parent["bbox"]
            if pr1 < cr1 and pc1 < cc1 and cr2 < pr2 and cc2 < pc2:
                containers.append(parent)
        child["parent"] = min(
            containers,
            key=lambda f: (f["bbox"][2] - f["bbox"][0] + 1) * (f["bbox"][3] - f["bbox"][1] + 1),
            default=None,
        )

    def depth(frame):
        if frame.get("depth") is None:
            frame["depth"] = 1 if frame["parent"] is None else depth(frame["parent"]) + 1
        return frame["depth"]

    for f in frames:
        depth(f)
    ordered = sorted(frames, key=lambda f: (f["depth"], f["bbox"][1]))
    max_depth = max((f["depth"] for f in ordered), default=0)
    out = np.zeros((max_depth, len(ordered)), dtype=int)
    for c, f in enumerate(ordered):
        out[max_depth - f["depth"] :, c] = f["color"]
    return out


def _draw_frame(grid, box, color):
    r1, c1, r2, c2 = box
    grid[r1, c1 : c2 + 1] = color
    grid[r2, c1 : c2 + 1] = color
    grid[r1 : r2 + 1, c1] = color
    grid[r1 : r2 + 1, c2] = color


def _boxes_ok(boxes):
    for i, a in enumerate(boxes):
        ar1, ac1, ar2, ac2 = a
        for b in boxes[i + 1 :]:
            br1, bc1, br2, bc2 = b
            contains = (ar1 < br1 and ac1 < bc1 and br2 < ar2 and bc2 < ac2) or (
                br1 < ar1 and bc1 < ac1 and ar2 < br2 and ac2 < bc2
            )
            disjoint = ar2 < br1 - 1 or br2 < ar1 - 1 or ac2 < bc1 - 1 or bc2 < ac1 - 1
            if not contains and not disjoint:
                return False
    return True


def generate_input() -> np.ndarray:
    colors = [1, 2, 3, 4, 7]
    for _ in range(500):
        h = random.randint(14, 19)
        w = random.randint(14, 19)
        n = random.randint(3, 5)
        boxes = []
        outer_h = random.randint(9, h - 2)
        outer_w = random.randint(9, w - 2)
        r1 = random.randint(0, h - outer_h)
        c1 = random.randint(0, w - outer_w)
        chain_len = random.randint(2, min(3, n))
        box = (r1, c1, r1 + outer_h - 1, c1 + outer_w - 1)
        boxes.append(box)
        for _j in range(1, chain_len):
            pr1, pc1, pr2, pc2 = boxes[-1]
            margin = 2
            if pr2 - pr1 + 1 < 7 or pc2 - pc1 + 1 < 7:
                break
            box = (
                pr1 + margin,
                pc1 + margin + random.randint(0, 1),
                pr2 - margin,
                pc2 - margin,
            )
            if box[2] - box[0] + 1 >= 3 and box[3] - box[1] + 1 >= 3:
                boxes.append(box)
        tries = 0
        while len(boxes) < n and tries < 200:
            tries += 1
            bh = random.randint(3, 6)
            bw = random.randint(3, 6)
            nr = random.randint(0, h - bh)
            nc = random.randint(0, w - bw)
            cand = (nr, nc, nr + bh - 1, nc + bw - 1)
            if _boxes_ok(boxes + [cand]):
                boxes.append(cand)
        if len(boxes) != n:
            continue
        grid = np.zeros((h, w), dtype=int)
        random.shuffle(colors)
        for box, color in zip(boxes, colors):
            _draw_frame(grid, box, color)
        if len(_components(grid)) == n:
            return grid
    return generate_input()
