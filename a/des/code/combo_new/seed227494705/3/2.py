import random
from collections import Counter, defaultdict, deque

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

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bg(grid):
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                x, y = q.popleft()
                cells.append((x, y))
                for dx, dy in DIR4:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            out.append(cells)
    return out


def _bbox(cells):
    rs = [p[0] for p in cells]
    cs = [p[1] for p in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _mask(cells):
    r0, c0, r1, c1 = _bbox(cells)
    m = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        m[r - r0, c - c0] = True
    return m


def _norm(cells):
    r0, c0, _, _ = _bbox(cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _components_by_color(grid, exclude=()):
    ans = []
    for col in sorted(set(int(x) for x in np.unique(grid)) - set(exclude)):
        for cells in _components(grid == col):
            ans.append({"color": col, "cells": cells, "bbox": _bbox(cells)})
    return ans


def _draw_rect_frame(g, r0, c0, h, w, color):
    g[r0, c0:c0 + w] = color
    g[r0 + h - 1, c0:c0 + w] = color
    g[r0:r0 + h, c0] = color
    g[r0:r0 + h, c0 + w - 1] = color


def _is_rect_frame(cells):
    r0, c0, r1, c1 = _bbox(cells)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    border = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    s = set(cells)
    return len(s - border) == 0 and len(border - s) <= 2


def _place(g, mask, r0, c0, color):
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if mask[r, c]:
                g[r0 + r, c0 + c] = color


def _random_free_rect(g, h, w, pad=1, tries=300):
    rows, cols = g.shape
    for _ in range(tries):
        r = random.randint(pad, rows - h - pad)
        c = random.randint(pad, cols - w - pad)
        if np.all(g[r - pad:r + h + pad, c - pad:c + w + pad] == BG):
            return r, c
    return None


def main_1(grid):
    g = np.array(grid, copy=True)
    bg = _bg(g)
    comps = _components_by_color(g, exclude={bg})
    legends = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        if r1 - r0 == 1 and c1 - c0 == 1 and (r0 == 0 or c0 == 0):
            legends.append(comp)
    if not legends:
        return g
    if sum(1 for x in legends if x["bbox"][1] == 0) >= len(legends) / 2:
        legends.sort(key=lambda x: (x["bbox"][0], x["bbox"][1]))
    else:
        legends.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
    palette = [x["color"] for x in legends]
    legend_cells = {p for comp in legends for p in comp["cells"]}
    for comp in comps:
        if any(p in legend_cells for p in comp["cells"]):
            continue
        if not _is_rect_frame(comp["cells"]):
            continue
        r0, c0, r1, c1 = comp["bbox"]
        interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        n = sum(1 for r, c in interior if g[r, c] == bg)
        if n > 0 and n <= len(palette):
            for r, c in interior:
                if g[r, c] == bg:
                    g[r, c] = palette[n - 1]
    return g


def generate_1():
    h, w = random.randint(14, 22), random.randint(16, 26)
    g = np.zeros((h, w), dtype=int)
    pal = random.sample([2, 3, 4, 6, 7, 8, 9], 5)
    vertical = random.choice([True, False])
    if vertical:
        for i, col in enumerate(pal):
            g[1 + i * 2:3 + i * 2, 0:2] = col
        start_c = 5
    else:
        for i, col in enumerate(pal):
            g[0:2, 1 + i * 2:3 + i * 2] = col
        start_c = 2
    outline = BLUE
    used = []
    for n in random.sample(range(1, 6), random.randint(2, 4)):
        fh, fw = 3, n + 2
        pos = _random_free_rect(g, fh, fw, 1)
        if pos is None:
            continue
        r, c = pos
        if c < start_c:
            c = start_c
        if c + fw >= w:
            c = w - fw - 1
        _draw_rect_frame(g, r, c, fh, fw, outline)
        used.append((r, c, fh, fw))
    return g


def main_2(grid):
    g = np.array(grid, copy=True)
    bg = _bg(g)
    frame_color = GREY
    # Fill every background cavity in the central frame by the color of the
    # smallest matching edge fragment. This handles the generated rectangular
    # notch cases and the described shape-to-gap pairing in a direct way.
    comps = _components_by_color(g, exclude={bg, frame_color})
    pieces = [c for c in comps if not (c["bbox"][0] > 2 and c["bbox"][2] < g.shape[0] - 3 and c["bbox"][1] > 2 and c["bbox"][3] < g.shape[1] - 3)]
    pieces.sort(key=lambda x: len(x["cells"]))
    for p in pieces:
        for r, c in p["cells"]:
            g[r, c] = bg
    barrier = g == frame_color
    labels = -np.ones(g.shape, dtype=int)
    regs = []
    for idx, cells in enumerate(_components((g == bg) & (~barrier))):
        for r, c in cells:
            labels[r, c] = idx
        regs.append(cells)
    center = np.array(g.shape) / 2.0
    for p in pieces:
        mask = _mask(p["cells"])
        ph, pw = mask.shape
        best = None
        for cells in regs:
            if any(r in (0, g.shape[0] - 1) or c in (0, g.shape[1] - 1) for r, c in cells):
                continue
            r0, c0, r1, c1 = _bbox(cells)
            if r1 - r0 + 1 >= ph and c1 - c0 + 1 >= pw:
                best = cells
                break
        if best is None:
            continue
        r0, c0, r1, c1 = _bbox(best)
        _place(g, mask, r0, c0, p["color"])
        cr = (r0 + r1) / 2.0
        cc = (c0 + c1) / 2.0
        qmask = np.zeros(g.shape, dtype=bool)
        if cr < center[0]:
            rr = range(0, int(center[0]))
        else:
            rr = range(int(center[0]) + 1, g.shape[0])
        if cc < center[1]:
            cc_range = range(0, int(center[1]))
        else:
            cc_range = range(int(center[1]) + 1, g.shape[1])
        for r in rr:
            for c in cc_range:
                if g[r, c] == bg:
                    qmask[r, c] = True
        g[qmask] = p["color"]
    return g


def generate_2():
    g = np.zeros((21, 21), dtype=int)
    g[10, 4:17] = GREY
    g[4:17, 10] = GREY
    pieces = []
    gaps = [((10, 7), (1, 2)), ((7, 10), (2, 1)), ((10, 13), (1, 2)), ((13, 10), (2, 1))]
    colors = random.sample([2, 3, 4, 6, 7, 8, 9], random.randint(2, 4))
    random.shuffle(gaps)
    for col, (pos, size) in zip(colors, gaps):
        r, c = pos
        h, w = size
        g[r:r + h, c:c + w] = BG
        pr = random.choice([1, 18])
        pc = random.randint(1, 16)
        m = np.ones((h, w), dtype=bool)
        _place(g, m, pr, pc, col)
    return g


def _find_two_by_two_blocks(g, color):
    out = []
    h, w = g.shape
    for r in range(h - 1):
        for c in range(w - 1):
            if np.all(g[r:r + 2, c:c + 2] == color):
                out.append((r, c))
    return out


def main_3(grid):
    g = np.array(grid, copy=True)
    bg = _bg(g)
    colors = [c for c in np.unique(g) if c not in (bg, GREY)]
    paths = {}
    targets = {}
    for col in colors:
        blocks = _find_two_by_two_blocks(g, int(col))
        if not blocks:
            continue
        src = blocks[0]
        frames = []
        for r in range(g.shape[0] - 3):
            for c in range(g.shape[1] - 3):
                if (
                    np.all(g[r, c:c + 4] == GREY)
                    and np.all(g[r + 3, c:c + 4] == GREY)
                    and np.all(g[r:r + 4, c] == GREY)
                    and np.all(g[r:r + 4, c + 3] == GREY)
                    and np.all(g[r + 1:r + 3, c + 1:c + 3] == bg)
                ):
                    frames.append((r, c))
        if not frames:
            continue
        sr, sc = src
        spt = (sr, sc)
        best = min(frames, key=lambda x: abs(x[0] - sr) + abs(x[1] - sc))
        tr, tc = best
        tpt = (tr + 1, tc + 1)
        cells = []
        if abs(sr - tr) < abs(sc - tc):
            rr = sr
            for c in range(min(sc, tc), max(sc, tc) + 1):
                cells.append((rr, c))
        else:
            cc = sc
            for r in range(min(sr, tr), max(sr, tr) + 1):
                cells.append((r, cc))
        paths[int(col)] = cells
        targets[int(col)] = best
    for col, cells in paths.items():
        fill = col
        for r, c in cells:
            hits = [other for other, ocells in paths.items() if other != col and (r, c) in ocells]
            if hits:
                fill = hits[-1]
        if col in targets:
            r, c = targets[col]
            g[r + 1:r + 3, c + 1:c + 3] = fill
    return g


def generate_3():
    g = np.zeros((20, 20), dtype=int)
    colors = random.sample([2, 3, 4, 6], random.randint(2, 4))
    rows = random.sample([4, 8, 12, 16], len(colors))
    cols = random.sample([4, 8, 12, 16], len(colors))
    for i, col in enumerate(colors):
        if i % 2 == 0:
            r = rows[i]
            g[r:r + 2, 1:3] = col
            _draw_rect_frame(g, r - 1, 16, 4, 4, GREY)
            g[r, 3:16] = col
        else:
            c = cols[i]
            g[1:3, c:c + 2] = col
            _draw_rect_frame(g, 16, c - 1, 4, 4, GREY)
            g[3:16, c] = col
    # Randomize visible crossing colors.
    for r in range(20):
        for c in range(20):
            present = [col for col in colors if g[r, c] == col]
            if present:
                g[r, c] = random.choice(present)
    return g


def _rot_order(mask):
    same90 = mask.shape == np.rot90(mask).shape and np.array_equal(mask, np.rot90(mask))
    same180 = mask.shape == np.rot90(mask, 2).shape and np.array_equal(mask, np.rot90(mask, 2))
    same270 = mask.shape == np.rot90(mask, 3).shape and np.array_equal(mask, np.rot90(mask, 3))
    if same90 and same180 and same270:
        return 4
    if same180:
        return 2
    return 1


def main_4(grid):
    g = np.array(grid, copy=True)
    bg = _bg(g)
    groups = {4: [], 2: [], 1: []}
    for comp in _components_by_color(g, exclude={bg}):
        m = _mask(comp["cells"])
        groups[_rot_order(m)].append((comp["color"], m))
    rows = []
    for order in (4, 2, 1):
        if not groups[order]:
            continue
        parts = groups[order]
        height = max(m.shape[0] for _, m in parts)
        width = sum(m.shape[1] for _, m in parts) + len(parts) - 1
        row = np.full((height, width), bg, dtype=int)
        x = 0
        for color, m in parts:
            y = (height - m.shape[0]) // 2
            _place(row, m, y, x, color)
            x += m.shape[1] + 1
        rows.append(row)
    if not rows:
        return g
    width = max(r.shape[1] for r in rows)
    padded = []
    for i, r in enumerate(rows):
        p = np.full((r.shape[0], width), bg, dtype=int)
        p[:r.shape[0], :r.shape[1]] = r
        padded.append(p)
        if i != len(rows) - 1:
            padded.append(np.full((1, width), GREY, dtype=int))
    return np.vstack(padded)


def generate_4():
    g = np.zeros((20, 20), dtype=int)
    shapes = [
        np.ones((2, 2), dtype=bool),
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
        np.ones((1, 4), dtype=bool),
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    ]
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(4, 7))
    for col in colors:
        m = random.choice(shapes)
        if random.random() < 0.5:
            m = np.rot90(m, random.randint(0, 3))
        pos = _random_free_rect(g, m.shape[0], m.shape[1], 1)
        if pos:
            _place(g, m, pos[0], pos[1], col)
    return g


def _find_separator_indices(g, axis):
    if axis == 0:
        return [i for i in range(g.shape[0]) if np.all(g[i, 1:-1] == GREY)]
    return [i for i in range(g.shape[1]) if np.all(g[1:-1, i] == GREY)]


def _arrow_ops(g, row_starts, col_starts, bh, bw):
    ops = []
    step = {RED: 1, BLUE: 2, GREEN: 3}
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            col = int(g[r, c])
            if col not in step:
                continue
            # Tip on the inner side with two wings on the border.
            if r == 1 and 0 < c < w - 1 and g[0, c - 1] == col and g[0, c + 1] == col:
                bi = min(range(len(col_starts)), key=lambda k: abs((col_starts[k] + bw // 2) - c))
                ops.append((r, c, "col", bi, "down", step[col]))
            elif r == h - 2 and 0 < c < w - 1 and g[h - 1, c - 1] == col and g[h - 1, c + 1] == col:
                bi = min(range(len(col_starts)), key=lambda k: abs((col_starts[k] + bw // 2) - c))
                ops.append((r, c, "col", bi, "up", step[col]))
            elif c == 1 and 0 < r < h - 1 and g[r - 1, 0] == col and g[r + 1, 0] == col:
                bi = min(range(len(row_starts)), key=lambda k: abs((row_starts[k] + bh // 2) - r))
                ops.append((r, c, "row", bi, "right", step[col]))
            elif c == w - 2 and 0 < r < h - 1 and g[r - 1, w - 1] == col and g[r + 1, w - 1] == col:
                bi = min(range(len(row_starts)), key=lambda k: abs((row_starts[k] + bh // 2) - r))
                ops.append((r, c, "row", bi, "left", step[col]))
    return sorted(ops)


def main_5(grid):
    g = np.array(grid, copy=True)
    row_seps = _find_separator_indices(g, 0)
    col_seps = _find_separator_indices(g, 1)
    if len(row_seps) < 2 or len(col_seps) < 2:
        return g
    row_starts = [row_seps[i] + 1 for i in range(len(row_seps) - 1)]
    col_starts = [col_seps[i] + 1 for i in range(len(col_seps) - 1)]
    bh = row_seps[1] - row_seps[0] - 1
    bw = col_seps[1] - col_seps[0] - 1
    blocks = [[g[r:r + bh, c:c + bw].copy() for c in col_starts] for r in row_starts]
    for _, _, kind, idx, direction, k in _arrow_ops(g, row_starts, col_starts, bh, bw):
        if kind == "row":
            n = len(blocks[idx])
            shift = k % n
            if direction == "right":
                blocks[idx] = blocks[idx][-shift:] + blocks[idx][:-shift]
            else:
                blocks[idx] = blocks[idx][shift:] + blocks[idx][:shift]
        else:
            n = len(blocks)
            col = [blocks[r][idx] for r in range(n)]
            shift = k % n
            if direction == "down":
                col = col[-shift:] + col[:-shift]
            else:
                col = col[shift:] + col[:shift]
            for r in range(n):
                blocks[r][idx] = col[r]
    out = g.copy()
    for i, r in enumerate(row_starts):
        for j, c in enumerate(col_starts):
            out[r:r + bh, c:c + bw] = blocks[i][j]
    return out


def generate_5():
    bh = random.choice([2, 3])
    bw = random.choice([2, 3])
    nr = random.choice([2, 3])
    nc = random.choice([2, 3])
    h = 2 + nr * bh + (nr + 1)
    w = 2 + nc * bw + (nc + 1)
    g = np.zeros((h, w), dtype=int)
    row_seps = [1 + i * (bh + 1) for i in range(nr + 1)]
    col_seps = [1 + i * (bw + 1) for i in range(nc + 1)]
    for r in row_seps:
        g[r, 1:w - 1] = GREY
    for c in col_seps:
        g[1:h - 1, c] = GREY
    colors = [1, 2, 3, 4, 6, 7, 8, 9]
    for i in range(nr):
        for j in range(nc):
            r = row_seps[i] + 1
            c = col_seps[j] + 1
            g[r:r + bh, c:c + bw] = random.choice(colors)
            g[r, c] = random.choice(colors)
    # Add one or two inward arrows.
    for _ in range(random.randint(1, 3)):
        col = random.choice([RED, BLUE, GREEN])
        side = random.choice(["top", "bottom", "left", "right"])
        if side in ("top", "bottom"):
            j = random.randrange(nc)
            c = col_seps[j] + 1 + bw // 2
            if side == "top":
                g[0, c - 1:c + 2:2] = col
                g[1, c] = col
            else:
                g[h - 1, c - 1:c + 2:2] = col
                g[h - 2, c] = col
        else:
            i = random.randrange(nr)
            r = row_seps[i] + 1 + bh // 2
            if side == "left":
                g[r - 1:r + 2:2, 0] = col
                g[r, 1] = col
            else:
                g[r - 1:r + 2:2, w - 1] = col
                g[r, w - 2] = col
    return g


def main_6(grid):
    g = np.array(grid, copy=True)
    bg = _bg(g)
    for comp in _components_by_color(g, exclude={bg}):
        if comp["color"] == GREY and len(comp["cells"]) >= 8:
            r0, c0, r1, c1 = comp["bbox"]
            inside = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
            seed = [(r, c) for r, c in inside if g[r, c] not in (bg, GREY)]
            if not seed:
                continue
            sr, sc = seed[0]
            scol = int(g[sr, sc])
            cells = set(comp["cells"])
            rect = _is_rect_frame(comp["cells"])
            if rect:
                for c in range(c0 + 1, c1):
                    if g[sr, c] == bg:
                        g[sr, c] = scol
                for r in range(r0 + 1, r1):
                    if g[r, sc] == bg:
                        g[r, sc] = scol
            else:
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    r, c = sr + dr, sc + dc
                    while 0 <= r < g.shape[0] and 0 <= c < g.shape[1] and g[r, c] != GREY:
                        if g[r, c] == bg:
                            g[r, c] = scol
                        r += dr
                        c += dc
    return g


def _draw_diamond(g, cr, cc, rad, color):
    for d in range(-rad, rad + 1):
        g[cr + d, cc + (rad - abs(d))] = color
        g[cr + d, cc - (rad - abs(d))] = color


def generate_6():
    g = np.zeros((24, 24), dtype=int)
    colors = random.sample([2, 3, 4, 6, 7, 8, 9], random.randint(2, 4))
    slots = [(2, 2), (2, 14), (13, 2), (13, 14)]
    random.shuffle(slots)
    for col, (r, c) in zip(colors, slots):
        if random.choice([True, False]):
            h = w = random.choice([5, 7])
            _draw_rect_frame(g, r, c, h, w, GREY)
            sr = random.randint(r + 1, r + h - 2)
            sc = random.randint(c + 1, c + w - 2)
            g[sr, sc] = col
        else:
            rad = random.choice([3, 4])
            cr, cc = r + rad, c + rad
            _draw_diamond(g, cr, cc, rad, GREY)
            g[cr + random.randint(-1, 1), cc + random.randint(-1, 1)] = col
    return g


def main_7(grid):
    g = np.array(grid, copy=True)
    bg = _bg(g)
    h, w = g.shape
    center = ((h - 1) / 2.0, (w - 1) / 2.0)
    for col in sorted(set(int(x) for x in np.unique(g)) - {bg}):
        pts = list(zip(*np.where(g == col)))
        if len(pts) != 2:
            continue
        (r1, c1), (r2, c2) = pts
        if r1 == r2:
            for c in range(min(c1, c2), max(c1, c2) + 1):
                g[r1, c] = col
        elif c1 == c2:
            for r in range(min(r1, r2), max(r1, r2) + 1):
                g[r, c1] = col
        else:
            corner_a = (r1, c2)
            corner_b = (r2, c1)
            da = (corner_a[0] - center[0]) ** 2 + (corner_a[1] - center[1]) ** 2
            db = (corner_b[0] - center[0]) ** 2 + (corner_b[1] - center[1]) ** 2
            if da <= db:
                cr, cc = corner_a
                for c in range(min(c1, cc), max(c1, cc) + 1):
                    g[r1, c] = col
                for r in range(min(r1, r2), max(r1, r2) + 1):
                    g[r, cc] = col
            else:
                cr, cc = corner_b
                for r in range(min(r1, cr), max(r1, cr) + 1):
                    g[r, c1] = col
                for c in range(min(c1, c2), max(c1, c2) + 1):
                    g[cr, c] = col
    return g


def generate_7():
    h, w = random.randint(12, 24), random.randint(12, 24)
    g = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 5))
    used = set()
    for col in colors:
        for _ in range(100):
            typ = random.choice(["row", "col", "bend"])
            if typ == "row":
                r = random.randint(1, h - 2)
                c1, c2 = sorted(random.sample(range(1, w - 1), 2))
                pts = [(r, c1), (r, c2)]
            elif typ == "col":
                c = random.randint(1, w - 2)
                r1, r2 = sorted(random.sample(range(1, h - 1), 2))
                pts = [(r1, c), (r2, c)]
            else:
                pts = [(random.randint(1, h - 2), random.randint(1, w - 2)),
                       (random.randint(1, h - 2), random.randint(1, w - 2))]
                if pts[0][0] == pts[1][0] or pts[0][1] == pts[1][1]:
                    continue
            if pts[0] not in used and pts[1] not in used:
                for p in pts:
                    g[p] = col
                    used.add(p)
                break
    return g

def main(grid):
    "Match edge fragments to frame gaps and tint the pointed quadrant.\n\nInput structure: A background canvas with a grey cross or L-like frame, rectangular gaps in that frame, and colored fragments near the edge.\nRule: Clear each fragment, place it into a compatible gap, and fill the background cells of the quadrant indicated by the gap with the fragment color.\nKey insight: Fragment shape chooses the missing notch, while the notch's position chooses the region that receives the fragment color."
    return main_2(grid)


def generate_input():
    return generate_2()
