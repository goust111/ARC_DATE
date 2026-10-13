import random
from collections import Counter, deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9
COLORS = [BLUE, RED, GREEN, YELLOW, GREY, MAGENTA, ORANGE, AZURE, MAROON]



def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def _norm(cells):
    if not cells:
        return frozenset()
    r0, _, c0, _ = _bbox(cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _paint(grid, cells, color):
    h, w = grid.shape
    for r, c in cells:
        if 0 <= r < h and 0 <= c < w:
            grid[r, c] = color


def _rect(grid, r0, r1, c0, c1, color):
    h, w = grid.shape
    for r in range(max(0, r0), min(h, r1 + 1)):
        for c in range(max(0, c0), min(w, c1 + 1)):
            grid[r, c] = color


def _outline(grid, r0, r1, c0, c1, color):
    for c in range(c0, c1 + 1):
        _paint(grid, [(r0, c), (r1, c)], color)
    for r in range(r0, r1 + 1):
        _paint(grid, [(r, c0), (r, c1)], color)


def _perimeter(r0, r1, c0, c1):
    out = set()
    for c in range(c0, c1 + 1):
        out.add((r0, c))
        out.add((r1, c))
    for r in range(r0, r1 + 1):
        out.add((r, c0))
        out.add((r, c1))
    return out


def _holes(grid, r0, r1, c0, c1):
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    out = []
    for comp in _components(sub == BG):
        if any(r == 0 or c == 0 or r == sub.shape[0] - 1 or c == sub.shape[1] - 1 for r, c in comp):
            continue
        out.append([(r0 + r, c0 + c) for r, c in comp])
    return out


def _shape_pool():
    return [
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 1)],
        [(0, 1), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ]


def main_1(grid):
    out = np.array(grid, copy=True)
    counts = Counter(int(x) for x in grid.ravel() if x != BG)
    if not counts:
        return out
    frame_color = counts.most_common(1)[0][0]
    frame_comps = [comp for comp in _components(grid == frame_color) if len(comp) >= 8]
    fragments = []
    for comp in _components((grid != BG) & (grid != frame_color)):
        color = int(grid[comp[0][0], comp[0][1]])
        fragments.append({"shape": _norm(comp), "color": color, "cells": comp})
    used = set()
    for frame in frame_comps:
        r0, r1, c0, c1 = _bbox(frame)
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        for hole in _holes(grid, r0, r1, c0, c1):
            shape = _norm(hole)
            for idx, frag in enumerate(fragments):
                if idx in used or frag["shape"] != shape:
                    continue
                _paint(out, hole, frag["color"])
                _paint(out, frag["cells"], frame_color)
                used.add(idx)
                break
    return out


def gen_1():
    grid = np.zeros((30, 30), dtype=int)
    frame_color = BLUE
    shapes = random.sample(_shape_pool(), random.randint(2, 4))
    frag_colors = random.sample([RED, GREEN, YELLOW, MAGENTA, ORANGE, AZURE, MAROON], len(shapes))
    frame_positions = [(2, 2), (2, 15), (14, 2), (14, 15)]
    random.shuffle(frame_positions)
    for idx, shape in enumerate(shapes):
        r0, c0 = frame_positions[idx]
        rh, rw = 7, 7
        grid[r0 : r0 + rh, c0 : c0 + rw] = frame_color
        hr, hc = r0 + 2, c0 + 2
        for dr, dc in shape:
            grid[hr + dr, hc + dc] = BG
        for _attempt in range(200):
            fr, fc = random.randint(20, 27), random.randint(1, 26)
            cells = [(fr + dr, fc + dc) for dr, dc in shape]
            if all(0 <= r < 30 and 0 <= c < 30 and grid[r, c] == BG for r, c in cells):
                _paint(grid, cells, frag_colors[idx])
                break
    return grid


def _arrow_direction(grid, cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    cellset = set(cells)
    if c1 - c0 >= r1 - r0:
        row = Counter(r for r, _ in cells).most_common(1)[0][0]
        if (row - 1, c1) in cellset and (row + 1, c1) in cellset:
            return "right"
        return "left"
    col = Counter(c for _, c in cells).most_common(1)[0][0]
    if (r1, col - 1) in cellset and (r1, col + 1) in cellset:
        return "down"
    return "up"


def main_2(grid):
    out = np.array(grid, copy=True)
    for comp in _components(grid != BG):
        vals = [int(grid[r, c]) for r, c in comp]
        if MAROON not in vals:
            continue
        body_vals = [v for v in vals if v != MAROON]
        if not body_vals:
            continue
        color = Counter(body_vals).most_common(1)[0][0]
        r0, r1, c0, c1 = _bbox(comp)
        rh, rw = r1 - r0 + 1, c1 - c0 + 1
        arrow = [(r, c) for r, c in comp if grid[r, c] == MAROON]
        direction = _arrow_direction(grid, arrow)
        n = len(body_vals)
        if direction == "right":
            nr0, nc0 = r0, c1 + n
        elif direction == "left":
            nr0, nc0 = r0, c0 - n - rw + 1
        elif direction == "down":
            nr0, nc0 = r1 + n, c0
        else:
            nr0, nc0 = r0 - n - rh + 1, c0
        _outline(out, nr0, nr0 + rh - 1, nc0, nc0 + rw - 1, color)
    return out


def gen_2():
    grid = np.zeros((30, 30), dtype=int)
    colors = random.sample([BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, AZURE], random.randint(1, 3))
    for color in colors:
        for _attempt in range(200):
            rh, rw = random.randint(4, 5), random.randint(4, 5)
            direction = random.choice(["right", "left", "down", "up"])
            margin = rh * rw
            r0 = random.randint(8 if direction == "up" else 2, 30 - rh - (8 if direction == "down" else 2))
            c0 = random.randint(8 if direction == "left" else 2, 30 - rw - (8 if direction == "right" else 2))
            pad = [(r, c) for r in range(r0 - 1, r0 + rh + 1) for c in range(c0 - 1, c0 + rw + 1)]
            if not all(0 <= r < 30 and 0 <= c < 30 and grid[r, c] == BG for r, c in pad):
                continue
            grid[r0 : r0 + rh, c0 : c0 + rw] = color
            mr, mc = r0 + rh // 2, c0 + rw // 2
            if direction == "right":
                cells = [(mr, mc - 1), (mr, mc), (mr, mc + 1), (mr - 1, mc + 1), (mr + 1, mc + 1)]
            elif direction == "left":
                cells = [(mr, mc + 1), (mr, mc), (mr, mc - 1), (mr - 1, mc - 1), (mr + 1, mc - 1)]
            elif direction == "down":
                cells = [(mr - 1, mc), (mr, mc), (mr + 1, mc), (mr + 1, mc - 1), (mr + 1, mc + 1)]
            else:
                cells = [(mr + 1, mc), (mr, mc), (mr - 1, mc), (mr - 1, mc - 1), (mr - 1, mc + 1)]
            _paint(grid, cells, MAROON)
            break
    return grid


def main_3(grid):
    order = [int(x) for x in grid[0] if x != BG]
    body = grid[1:]
    label_rows = [(r, int(body[r, 0])) for r in range(body.shape[0]) if body[r, 0] not in (BG, YELLOW)]
    if not label_rows:
        return body.copy()
    if len(label_rows) > 1:
        band_h = label_rows[1][0] - label_rows[0][0]
    else:
        band_h = body.shape[0]
    bands = {}
    for start, color in label_rows:
        bands[color] = body[start : min(body.shape[0], start + band_h), :].copy()
    out_bands = []
    for color in order:
        if color not in bands:
            continue
        band = bands[color].copy()
        flip = np.any(band[:, -1] == YELLOW)
        inner = band[:, 1:-1]
        if flip:
            inner = inner[:, ::-1]
        new_band = np.zeros_like(band)
        new_band[:, 1:-1] = inner
        out_bands.append(new_band)
    return np.vstack(out_bands) if out_bands else body.copy()


def gen_3():
    band_count = random.randint(3, 5)
    band_h = random.choice([3, 4])
    w = random.randint(9, 16)
    tags = random.sample([BLUE, RED, GREEN, MAGENTA, ORANGE, AZURE, MAROON], band_count)
    order = tags[:]
    random.shuffle(order)
    grid = np.zeros((1 + band_count * band_h, w), dtype=int)
    grid[0, :band_count] = order
    for i, tag in enumerate(tags):
        r0 = 1 + i * band_h
        grid[r0, 0] = tag
        if random.random() < 0.5:
            grid[r0 + band_h // 2, w - 1] = YELLOW
        pat_color = random.choice([c for c in COLORS if c not in (tag, YELLOW)])
        for _ in range(random.randint(3, min(8, (w - 2) * band_h))):
            grid[r0 + random.randrange(band_h), random.randint(1, w - 2)] = pat_color
    return grid


def main_4(grid):
    out = np.array(grid, copy=True)
    branches = []
    pipe_cells = set()
    for color in set(int(x) for x in grid.ravel() if x != BG):
        # Horizontal maximal runs.
        for r in range(grid.shape[0]):
            c = 0
            while c < grid.shape[1]:
                if grid[r, c] != color:
                    c += 1
                    continue
                start = c
                while c + 1 < grid.shape[1] and grid[r, c + 1] == color:
                    c += 1
                run = [(r, cc) for cc in range(start, c + 1)]
                if len(run) >= 3:
                    for pr, pc in run:
                        for sr, sc in ((pr - 1, pc), (pr + 1, pc)):
                            if 0 <= sr < grid.shape[0] and grid[sr, sc] == color:
                                direction = (sr - pr, 0)
                                branches.append((color, (sr, sc), direction, len(run)))
                    pipe_cells.update(run)
                c += 1
        for c in range(grid.shape[1]):
            r = 0
            while r < grid.shape[0]:
                if grid[r, c] != color:
                    r += 1
                    continue
                start = r
                while r + 1 < grid.shape[0] and grid[r + 1, c] == color:
                    r += 1
                run = [(rr, c) for rr in range(start, r + 1)]
                if len(run) >= 3:
                    for pr, pc in run:
                        for sr, sc in ((pr, pc - 1), (pr, pc + 1)):
                            if 0 <= sc < grid.shape[1] and grid[sr, sc] == color:
                                direction = (0, sc - pc)
                                branches.append((color, (sr, sc), direction, len(run)))
                    pipe_cells.update(run)
                r += 1
    for r, c in pipe_cells:
        out[r, c] = BG
    for color, (sr, sc), (dr, dc), length in branches:
        for k in range(length):
            rr, cc = sr + dr * k, sc + dc * k
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                out[rr, cc] = color
    return out


def gen_4():
    grid = np.zeros((30, 30), dtype=int)
    for color in random.sample([BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, AZURE, MAROON], random.randint(2, 4)):
        for _attempt in range(200):
            length = random.randint(3, 7)
            horizontal = random.random() < 0.5
            if horizontal:
                r, c = random.randint(4, 25), random.randint(3, 30 - length - 3)
                run = [(r, c + i) for i in range(length)]
                side = random.choice([-1, 1])
                seeds = [(r + side, c + random.randint(0, length - 1))]
            else:
                r, c = random.randint(3, 30 - length - 3), random.randint(4, 25)
                run = [(r + i, c) for i in range(length)]
                side = random.choice([-1, 1])
                seeds = [(r + random.randint(0, length - 1), c + side)]
            cells = run + seeds
            if all(grid[rr, cc] == BG for rr, cc in cells):
                _paint(grid, cells, color)
                break
    return grid


def main_5(grid):
    out = np.zeros_like(grid)
    boxes = []
    for color in set(int(x) for x in grid.ravel() if x != BG):
        cells = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1]) if grid[r, c] == color]
        if cells:
            r0, r1, c0, c1 = _bbox(cells)
            boxes.append({"color": color, "box": (r0, r1, c0, c1), "count": len(cells), "area": (r1 - r0 + 1) * (c1 - c0 + 1)})
    for box in sorted(boxes, key=lambda b: -b["area"]):
        r0, r1, c0, c1 = box["box"]
        _rect(out, r0, r1, c0, c1, box["color"])
    for outer in boxes:
        r0, r1, c0, c1 = outer["box"]
        inners = []
        for inner in boxes:
            if inner is outer:
                continue
            ir0, ir1, ic0, ic1 = inner["box"]
            if r0 < ir0 and ir1 < r1 and c0 < ic0 and ic1 < c1:
                inners.append(inner)
        if not inners:
            continue
        border_color = max(inners, key=lambda b: b["count"])["color"]
        _outline(out, r0, r1, c0, c1, border_color)
    return out


def gen_5():
    grid = np.zeros((30, 30), dtype=int)
    colors = random.sample([BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, AZURE, MAROON], random.randint(3, 5))
    boxes = []
    outer = (4, 20, 4, 22)
    boxes.append((colors[0], outer))
    if random.random() < 0.7:
        boxes.append((colors[1], (9, 13, 8, 13)))
        rest = colors[2:]
    else:
        rest = colors[1:]
    for color in rest:
        for _attempt in range(100):
            rh, rw = random.randint(3, 6), random.randint(3, 6)
            r0, c0 = random.randint(1, 24), random.randint(1, 24)
            box = (r0, r0 + rh - 1, c0, c0 + rw - 1)
            if all(not (max(box[0], b[1][0]) <= min(box[1], b[1][1]) and max(box[2], b[1][2]) <= min(box[3], b[1][3])) for b in boxes):
                boxes.append((color, box))
                break
    for color, (r0, r1, c0, c1) in boxes:
        coords = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        for r, c in random.sample(coords, random.randint(2, min(8, len(coords)))):
            grid[r, c] = color
        grid[r0, c0] = color
        grid[r1, c1] = color
    return grid


def main_6(grid):
    out = np.array(grid, copy=True)
    comps = _components(grid != BG)
    frames = []
    segments = []
    for comp in comps:
        r0, r1, c0, c1 = _bbox(comp)
        vals = [int(grid[r, c]) for r, c in comp]
        if r1 - r0 >= 3 and c1 - c0 >= 3:
            frames.append(comp)
        elif r0 == r1 or c0 == c1:
            segments.append({"color": vals[0], "len": len(comp), "cells": comp})
    for frame in frames:
        r0, r1, c0, c1 = _bbox(frame)
        vals = [int(grid[r, c]) for r, c in frame]
        main_color = Counter(vals).most_common(1)[0][0]
        sides = {
            "top": [(r0, c) for c in range(c0, c1 + 1)],
            "bottom": [(r1, c) for c in range(c0, c1 + 1)],
            "left": [(r, c0) for r in range(r0, r1 + 1)],
            "right": [(r, c1) for r in range(r0, r1 + 1)],
        }
        gap_side = None
        gap_color = None
        for side, cells in sides.items():
            non_bg = [int(grid[p]) for p in cells if grid[p] != BG]
            if 0 < len(non_bg) <= 2:
                gap_side = side
                gap_color = Counter(non_bg).most_common(1)[0][0]
                break
        if gap_side is None:
            continue
        seg = next((s for s in segments if s["color"] == gap_color), None)
        if seg is None:
            continue
        _paint(out, seg["cells"], BG)
        _paint(out, sides[gap_side], gap_color)
        L = seg["len"]
        if gap_side == "top":
            _rect(out, r0 + 1, min(r1 - 1, r0 + L), c0 + 1, c1 - 1, main_color)
        elif gap_side == "bottom":
            _rect(out, max(r0 + 1, r1 - L), r1 - 1, c0 + 1, c1 - 1, main_color)
        elif gap_side == "left":
            _rect(out, r0 + 1, r1 - 1, c0 + 1, min(c1 - 1, c0 + L), main_color)
        else:
            _rect(out, r0 + 1, r1 - 1, max(c0 + 1, c1 - L), c1 - 1, main_color)
    return out


def gen_6():
    grid = np.zeros((30, 30), dtype=int)
    for main_color, gap_color in zip(random.sample([BLUE, RED, GREEN, YELLOW], 3), random.sample([MAGENTA, ORANGE, AZURE, MAROON, GREY], 3)):
        for _attempt in range(200):
            rh, rw = random.randint(5, 8), random.randint(5, 8)
            r0, c0 = random.randint(2, 22), random.randint(2, 22)
            r1, c1 = r0 + rh - 1, c0 + rw - 1
            side = random.choice(["top", "bottom", "left", "right"])
            frame = _perimeter(r0, r1, c0, c1)
            if any(grid[r, c] != BG for r, c in frame):
                continue
            _paint(grid, frame, main_color)
            if side == "top":
                for c in range(c0, c1 + 1):
                    grid[r0, c] = BG
                grid[r0, c0] = grid[r0, c1] = gap_color
            elif side == "bottom":
                for c in range(c0, c1 + 1):
                    grid[r1, c] = BG
                grid[r1, c0] = grid[r1, c1] = gap_color
            elif side == "left":
                for r in range(r0, r1 + 1):
                    grid[r, c0] = BG
                grid[r0, c0] = grid[r1, c0] = gap_color
            else:
                for r in range(r0, r1 + 1):
                    grid[r, c1] = BG
                grid[r0, c1] = grid[r1, c1] = gap_color
            length = random.randint(1, 4)
            for _seg in range(100):
                horizontal = random.random() < 0.5
                sr, sc = random.randint(1, 28), random.randint(1, 28)
                cells = [(sr, sc + i) for i in range(length)] if horizontal else [(sr + i, sc) for i in range(length)]
                if all(0 <= r < 30 and 0 <= c < 30 and grid[r, c] == BG for r, c in cells):
                    _paint(grid, cells, gap_color)
                    break
            break
    return grid


def main_7(grid):
    out = np.array(grid, copy=True)
    palette = [int(x) for x in grid[-1] if x != BG]
    steps = {color: i + 1 for i, color in enumerate(palette)}
    top = grid[:-1]
    h, w = top.shape
    h_lines = [r for r in range(h) if np.all(top[r, :] == MAROON)]
    v_lines = [c for c in range(w) if np.all(top[:, c] == MAROON)]
    r_edges = [-1] + h_lines + [h]
    c_edges = [-1] + v_lines + [w]
    for i in range(len(r_edges) - 1):
        for j in range(len(c_edges) - 1):
            r0, r1 = r_edges[i] + 1, r_edges[i + 1] - 1
            c0, c1 = c_edges[j] + 1, c_edges[j + 1] - 1
            if r0 > r1 or c0 > c1:
                continue
            anchors = [(r, c, int(grid[r, c])) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if grid[r, c] in steps]
            path = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if grid[r, c] == GREY}
            if len(anchors) != 1 or not path:
                continue
            ar, ac, color = anchors[0]
            starts = [(ar + dr, ac + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)) if (ar + dr, ac + dc) in path]
            if not starts:
                continue
            cur = starts[0]
            prev = (ar, ac)
            for _ in range(1, steps[color]):
                nxts = [p for p in ((cur[0] + 1, cur[1]), (cur[0] - 1, cur[1]), (cur[0], cur[1] + 1), (cur[0], cur[1] - 1)) if p in path and p != prev]
                if not nxts:
                    break
                prev, cur = cur, nxts[0]
            out[cur] = color
    return out


def gen_7():
    rows, cols = random.choice([(2, 2), (2, 3), (3, 2)])
    cell_h, cell_w = 5, 6
    top_h = rows * cell_h + rows - 1
    w = cols * cell_w + cols - 1
    grid = np.zeros((top_h + 1, w), dtype=int)
    for r in range(cell_h, top_h, cell_h + 1):
        grid[r, :] = MAROON
    for c in range(cell_w, w, cell_w + 1):
        grid[:top_h, c] = MAROON
    palette = random.sample([BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, AZURE], random.randint(3, 5))
    grid[-1, : len(palette)] = palette
    for i in range(rows):
        for j in range(cols):
            r0, c0 = i * (cell_h + 1), j * (cell_w + 1)
            color = random.choice(palette)
            direction = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            if direction == (1, 0):
                ar, ac = r0 + 1, c0 + random.randint(1, cell_w - 2)
            elif direction == (-1, 0):
                ar, ac = r0 + cell_h - 2, c0 + random.randint(1, cell_w - 2)
            elif direction == (0, 1):
                ar, ac = r0 + random.randint(1, cell_h - 2), c0 + 1
            else:
                ar, ac = r0 + random.randint(1, cell_h - 2), c0 + cell_w - 2
            grid[ar, ac] = color
            length = random.randint(2, 5)
            for k in range(1, length + 1):
                rr, cc = ar + direction[0] * k, ac + direction[1] * k
                if r0 <= rr < r0 + cell_h and c0 <= cc < c0 + cell_w:
                    grid[rr, cc] = GREY
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    'Grow same-colored branches from side seeds and erase their pipes.\n\nInput structure:\nBlack background contains horizontal or vertical one-pixel colored pipes of\nlength at least three. Same-colored singleton seeds touch pipe sides but are\nnot on the pipe axis.\nRule:\nFor each side seed, grow a straight branch away from the adjacent pipe. The\nbranch length equals the pipe length and includes the seed cell. After all\nbranches are drawn, erase the original pipe cells.\nKey insight:\nSide adjacency assigns each seed to a pipe; relative side gives direction,\nand pipe length gives branch length.'
    return main_4(grid)


def generate_input() -> np.ndarray:
    return gen_4()
