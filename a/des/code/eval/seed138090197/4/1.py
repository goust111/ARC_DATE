import random
from collections import deque

import numpy as np


BG = 0
COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _norm(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return frozenset((r - min(rs), c - min(cs)) for r, c in cells)


def _transforms(cells):
    pts = list(cells)
    outs = []
    for swap in [False, True]:
        for sr in [1, -1]:
            for sc in [1, -1]:
                cur = []
                for r, c in pts:
                    x, y = (c, r) if swap else (r, c)
                    cur.append((sr * x, sc * y))
                outs.append(_norm(cur))
    return set(outs)


def _components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            col = int(grid[r, c])
            if col == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == col:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": col, "cells": cells})
    return comps


def _classify(grid):
    frames = []
    pieces = []
    for comp in _components(grid):
        cells = set(comp["cells"])
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        perimeter = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        if h >= 4 and w >= 4 and cells <= perimeter and len(cells) >= len(perimeter) - max(h, w):
            gap = sorted(p for p in perimeter if p not in cells and int(grid[p]) == BG)
            if gap:
                sides = []
                if all(r == r0 for r, _ in gap):
                    sides.append("top")
                if all(r == r1 for r, _ in gap):
                    sides.append("bottom")
                if all(c == c0 for _, c in gap):
                    sides.append("left")
                if all(c == c1 for _, c in gap):
                    sides.append("right")
                if sides:
                    frames.append({"color": comp["color"], "bbox": (r0, c0, r1, c1), "gap": gap, "side": sides[0]})
                    continue
        pieces.append({"color": comp["color"], "cells": cells, "shape": _norm(cells), "all_shapes": _transforms(cells)})
    return frames, pieces


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    frames, pieces = _classify(grid)
    used = set()
    for frame in frames:
        gap_shape = _norm(frame["gap"])
        match_idx = None
        for i, piece in enumerate(pieces):
            if i in used:
                continue
            if gap_shape in piece["all_shapes"]:
                match_idx = i
                break
        if match_idx is None:
            continue
        piece = pieces[match_idx]
        used.add(match_idx)
        for r, c in piece["cells"]:
            out[r, c] = BG
        for r, c in frame["gap"]:
            out[r, c] = piece["color"]
        r0, c0, r1, c1 = frame["bbox"]
        if frame["side"] == "top":
            cells = [(r0, c) for c in range(c0, c1 + 1)]
        elif frame["side"] == "bottom":
            cells = [(r1, c) for c in range(c0, c1 + 1)]
        elif frame["side"] == "left":
            cells = [(r, c0) for r in range(r0, r1 + 1)]
        else:
            cells = [(r, c1) for r in range(r0, r1 + 1)]
        for r, c in cells:
            out[r, c] = piece["color"]
    return out


def _free(grid, cells, pad=0):
    h, w = grid.shape
    for r, c in cells:
        for rr in range(r - pad, r + pad + 1):
            for cc in range(c - pad, c + pad + 1):
                if not (1 <= rr < h - 1 and 1 <= cc < w - 1):
                    return False
                if int(grid[rr, cc]) != BG:
                    return False
    return True


def _draw_frame(grid, frame_color, piece_color):
    h, w = grid.shape
    for _ in range(200):
        fh = random.randint(5, 9)
        fw = random.randint(5, 10)
        r0 = random.randint(2, h - fh - 3)
        c0 = random.randint(2, w - fw - 3)
        r1, c1 = r0 + fh - 1, c0 + fw - 1
        perim = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if r in (r0, r1) or c in (c0, c1)]
        side = random.choice(["top", "bottom", "left", "right"])
        glen = random.randint(1, 3)
        if side in ("top", "bottom"):
            gr = r0 if side == "top" else r1
            gc = random.randint(c0 + 1, c1 - glen)
            gap = [(gr, gc + i) for i in range(glen)]
        else:
            gc = c0 if side == "left" else c1
            gr = random.randint(r0 + 1, r1 - glen)
            gap = [(gr + i, gc) for i in range(glen)]
        draw_cells = [p for p in perim if p not in gap]
        if not _free(grid, draw_cells, pad=1):
            continue
        for r, c in draw_cells:
            grid[r, c] = frame_color
        for _j in range(100):
            pr = random.randint(2, h - 3)
            pc = random.randint(2, w - 3)
            if side in ("top", "bottom"):
                piece = [(pr, pc + i) for i in range(glen)]
            else:
                piece = [(pr + i, pc) for i in range(glen)]
            if _free(grid, piece, pad=1):
                for r, c in piece:
                    grid[r, c] = piece_color
                return True
    return False


def generate_input() -> np.ndarray:
    for _ in range(300):
        h = random.randint(22, 32)
        w = random.randint(22, 32)
        grid = np.zeros((h, w), dtype=int)
        colors = random.sample(COLORS, 6)
        n = random.randint(1, 3)
        ok = True
        for i in range(n):
            if not _draw_frame(grid, colors[i], colors[i + 3]):
                ok = False
                break
        if ok and not np.array_equal(main(grid), grid):
            return grid
    raise RuntimeError("failed to generate seed138090197/4/1 input")
