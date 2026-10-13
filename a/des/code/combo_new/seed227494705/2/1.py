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
    comps = []
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
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
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
    out = []
    for col in sorted(set(int(x) for x in np.unique(grid)) - set(exclude)):
        for cells in _components(grid == col):
            out.append({"color": col, "cells": cells, "bbox": _bbox(cells)})
    return out


def _draw_frame(g, r0, c0, h, w, color):
    g[r0, c0:c0 + w] = color
    g[r0 + h - 1, c0:c0 + w] = color
    g[r0:r0 + h, c0] = color
    g[r0:r0 + h, c0 + w - 1] = color


def _is_frame(cells):
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


def _put(g, mask, r0, c0, color):
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if mask[r, c]:
                g[r0 + r, c0 + c] = color


def _free(g, r, c, h, w, pad=1):
    r0 = max(0, r - pad)
    c0 = max(0, c - pad)
    r1 = min(g.shape[0], r + h + pad)
    c1 = min(g.shape[1], c + w + pad)
    return np.all(g[r0:r1, c0:c1] == BG)


def _place_free(g, h, w, pad=1, tries=300):
    for _ in range(tries):
        r = random.randint(pad, g.shape[0] - h - pad)
        c = random.randint(pad, g.shape[1] - w - pad)
        if _free(g, r, c, h, w, pad):
            return r, c
    return None


def _palette(g):
    h, w = g.shape
    cols = [int(g[r, w - 1]) for r in range(h) if g[r, w - 1] not in (BG, GREY)]
    if len(cols) >= 2:
        return cols
    rows = [int(g[h - 1, c]) for c in range(w) if g[h - 1, c] not in (BG, GREY)]
    return rows


def main_1(grid):
    g = np.array(grid, copy=True)
    pal = _palette(g)
    grey_comps = [{"cells": c, "bbox": _bbox(c)} for c in _components(g == GREY)]
    markers = [x for x in grey_comps if len(x["cells"]) <= 4]
    frames = [x for x in grey_comps if _is_frame(x["cells"])]
    for fr in frames:
        r0, c0, r1, c1 = fr["bbox"]
        n = 0
        for m in markers:
            mr0, mc0, mr1, mc1 = m["bbox"]
            if r0 < mr0 and mr1 < r1 and c0 < mc0 and mc1 < c1:
                n += 1
        if n < len(pal):
            for r, c in fr["cells"]:
                g[r, c] = pal[n]
    return g


def generate_1():
    g = np.zeros((24, 24), dtype=int)
    pal = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 5)
    if random.choice([True, False]):
        for i, col in enumerate(pal):
            g[2 + i, 23] = col
    else:
        for i, col in enumerate(pal):
            g[23, 2 + i] = col
    for n in random.sample(range(5), random.randint(2, 4)):
        h = random.randint(5, 7)
        w = random.randint(5, 8)
        pos = _place_free(g, h, w, 1)
        if not pos:
            continue
        r, c = pos
        _draw_frame(g, r, c, h, w, GREY)
        spots = [(rr, cc) for rr in range(r + 2, r + h - 2) for cc in range(c + 2, c + w - 2)]
        random.shuffle(spots)
        for rr, cc in spots[:n]:
            g[rr, cc] = GREY
    return g


def _find_color_frames(g, bg):
    frames = []
    for comp in _components_by_color(g, exclude={bg, GREY}):
        if _is_frame(comp["cells"]):
            frames.append(comp)
    return frames


def main_2(grid):
    g = np.array(grid, copy=True)
    bg = _bg(g)
    frames = _find_color_frames(g, bg)
    frame_cells = {p for fr in frames for p in fr["cells"]}
    comps = _components_by_color(g, exclude={bg, GREY})
    for fr in frames:
        color = fr["color"]
        r0, c0, r1, c1 = fr["bbox"]
        pieces = [c for c in comps if c["color"] == color and not any(p in frame_cells for p in c["cells"])]
        pieces.sort(key=lambda x: (-len(x["cells"]), x["bbox"]))
        for p in pieces:
            for r, c in p["cells"]:
                g[r, c] = bg
        for p in pieces:
            m = _mask(p["cells"])
            ph, pw = m.shape
            placed = False
            for rr in range(r0 + 1, r1 - ph + 1):
                if placed:
                    break
                for cc in range(c0 + 1, c1 - pw + 1):
                    ok = True
                    for a in range(ph):
                        for b in range(pw):
                            if m[a, b] and g[rr + a, cc + b] != bg:
                                ok = False
                    if ok:
                        _put(g, m, rr, cc, color)
                        placed = True
                        break
    return g


def generate_2():
    g = np.zeros((25, 25), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 3))
    slots = [(2, 2), (2, 14), (14, 3)]
    for col, (r, c) in zip(colors, slots):
        h, w = random.choice([(7, 8), (8, 7), (7, 7)])
        _draw_frame(g, r, c, h, w, col)
        g[r + 1:r + h - 1, c + 1:c + w - 1] = GREY
        pieces = [
            np.ones((1, 2), dtype=bool),
            np.ones((2, 1), dtype=bool),
            np.array([[1, 1], [1, 0]], dtype=bool),
        ]
        random.shuffle(pieces)
        for m in pieces[:random.randint(1, 3)]:
            # carve a compatible hole inside the grey fill
            for _ in range(50):
                rr = random.randint(r + 1, r + h - m.shape[0] - 1)
                cc = random.randint(c + 1, c + w - m.shape[1] - 1)
                if np.all(g[rr:rr + m.shape[0], cc:cc + m.shape[1]][m] == GREY):
                    g[rr:rr + m.shape[0], cc:cc + m.shape[1]][m] = BG
                    break
            pos = _place_free(g, m.shape[0], m.shape[1], 1)
            if pos:
                _put(g, m, pos[0], pos[1], col)
    return g


def _arrow_cells(center, direction):
    r, c = center
    d = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}[direction]
    tip = (r + 2 * d[0], c + 2 * d[1])
    shaft = [(r, c), (r + d[0], c + d[1]), tip]
    if direction in ("up", "down"):
        wings = [(tip[0], tip[1] - 1), (tip[0], tip[1] + 1)]
    else:
        wings = [(tip[0] - 1, tip[1]), (tip[0] + 1, tip[1])]
    return shaft + wings


def main_3(grid):
    g = np.array(grid, copy=True)
    bg = _bg(g)
    corners = {(0, 0), (0, g.shape[1] - 1), (g.shape[0] - 1, 0), (g.shape[0] - 1, g.shape[1] - 1)}
    marked = {p for p in corners if g[p] == MAGENTA}
    mode = "none"
    if (0, 0) in marked and (g.shape[0] - 1, g.shape[1] - 1) in marked:
        mode = "main"
    elif (0, g.shape[1] - 1) in marked and (g.shape[0] - 1, 0) in marked:
        mode = "anti"
    main_map = {"up": "left", "down": "right", "left": "up", "right": "down"}
    anti_map = {"up": "right", "down": "left", "left": "down", "right": "up"}
    for comp in _components_by_color(g, exclude={bg, MAGENTA, GREY}):
        r0, c0, r1, c1 = comp["bbox"]
        if (r1 - r0 + 1) < 5 or (c1 - c0 + 1) < 5:
            continue
        if len(comp["cells"]) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        dirs = []
        if r0 == 0:
            dirs.append("up")
        if r1 == g.shape[0] - 1:
            dirs.append("down")
        if c0 == 0:
            dirs.append("left")
        if c1 == g.shape[1] - 1:
            dirs.append("right")
        if len(dirs) != 1:
            continue
        direction = dirs[0]
        if mode == "main":
            direction = main_map[direction]
        elif mode == "anti":
            direction = anti_map[direction]
        center = ((r0 + r1) // 2, (c0 + c1) // 2)
        for r, c in _arrow_cells(center, direction):
            if 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
                g[r, c] = GREY
    return g


def generate_3():
    h, w = random.randint(18, 26), random.randint(18, 26)
    g = np.zeros((h, w), dtype=int)
    mode = random.choice(["none", "main", "anti"])
    if mode == "main":
        g[0, 0] = MAGENTA
        g[h - 1, w - 1] = MAGENTA
    elif mode == "anti":
        g[0, w - 1] = MAGENTA
        g[h - 1, 0] = MAGENTA
    colors = random.sample([1, 2, 3, 4, 7, 8, 9], 4)
    specs = [
        (0, random.randint(2, w - 8), random.randint(5, 7), random.randint(5, 7)),
        (h - 6, random.randint(2, w - 8), random.randint(5, 6), random.randint(5, 7)),
        (random.randint(2, h - 8), 0, random.randint(5, 7), random.randint(5, 6)),
        (random.randint(2, h - 8), w - 6, random.randint(5, 7), random.randint(5, 6)),
    ]
    for col, (r, c, hh, ww) in zip(colors, specs):
        if r + hh > h:
            r = h - hh
        if c + ww > w:
            c = w - ww
        g[r:r + hh, c:c + ww] = col
    return g


def _trace_path_cells(g):
    bg = _bg(g)
    mask = (g != bg)
    cells = [(r, c) for r, c in zip(*np.where(mask))]
    deg = {}
    s = set(cells)
    for p in cells:
        deg[p] = sum((p[0] + dr, p[1] + dc) in s for dr, dc in DIR4)
    ends = sorted([p for p in cells if deg[p] == 1])
    if not ends:
        return []
    cur = ends[0]
    prev = None
    path = [cur]
    while True:
        nxts = []
        for dr, dc in DIR4:
            q = (cur[0] + dr, cur[1] + dc)
            if q in s and q != prev:
                nxts.append(q)
        if not nxts:
            break
        prev, cur = cur, nxts[0]
        path.append(cur)
        if len(path) > len(cells) + 2:
            break
    return path


def main_4(grid):
    g = np.array(grid, copy=True)
    path = _trace_path_cells(g)
    if len(path) < 3:
        return g
    turns = []
    seg_starts = [0]
    for i in range(1, len(path) - 1):
        d1 = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        d2 = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if d1 != d2:
            turns.append((i, d1, d2))
            seg_starts.append(i)
    seg_starts.append(len(path) - 1)
    path_set = set(path)
    for t_index, (i, din, dout) in enumerate(turns):
        prev_start = seg_starts[t_index]
        next_end = seg_starts[t_index + 2] if t_index + 2 < len(seg_starts) else len(path) - 1
        len_prev = i - prev_start + 1
        len_next = next_end - i + 1
        color = int(grid[path[i - 1]])
        corner = path[i]
        for a in range(1, len_prev + 1):
            for b in range(1, len_next + 1):
                r = corner[0] - din[0] * a + dout[0] * b
                c = corner[1] - din[1] * a + dout[1] * b
                if 0 <= r < g.shape[0] and 0 <= c < g.shape[1] and (r, c) not in path_set:
                    g[r, c] = color
    for r, c in path:
        g[r, c] = grid[r, c]
    return g


def generate_4():
    g = np.zeros((24, 24), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 4)
    pts = [(3, 3), (3, 12), (12, 12), (12, 18), (18, 18)]
    if random.random() < 0.5:
        pts = [(r, 22 - c) for r, c in pts]
    for idx in range(len(pts) - 1):
        (r1, c1), (r2, c2) = pts[idx], pts[idx + 1]
        col = colors[idx % len(colors)]
        if r1 == r2:
            for c in range(min(c1, c2), max(c1, c2) + 1):
                g[r1, c] = col
        else:
            for r in range(min(r1, r2), max(r1, r2) + 1):
                g[r, c1] = col
    return g


def main_5(grid):
    g = np.array(grid, copy=True)
    bg = _bg(g)
    grey = GREY
    frames = [c for c in _components_by_color(g, exclude={bg}) if c["color"] == grey and _is_frame(c["cells"])]
    if len(frames) < 2:
        return g
    seed_frame, templ = frames[:2]
    def interior_colors(fr):
        r0, c0, r1, c1 = fr["bbox"]
        return [(r, c, int(g[r, c])) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if g[r, c] not in (bg, grey)]
    if all((r in (templ["bbox"][0] + 1, templ["bbox"][2] - 1) or c in (templ["bbox"][1] + 1, templ["bbox"][3] - 1)) for r, c, _ in interior_colors(seed_frame)):
        seed_frame, templ = templ, seed_frame
    sr0, sc0, sr1, sc1 = seed_frame["bbox"]
    seed = [(r, c, int(g[r, c])) for r in range(sr0 + 1, sr1) for c in range(sc0 + 1, sc1) if g[r, c] not in (bg, grey)]
    tr0, tc0, tr1, tc1 = templ["bbox"]
    markers = [(r, c, int(g[r, c])) for r in range(tr0 + 1, tr1) for c in range(tc0 + 1, tc1) if g[r, c] not in (bg, grey)]
    out = np.zeros_like(g)
    for r, c, col in seed:
        out[r, c] = col
    if not seed:
        return out
    rmin, cmin, rmax, cmax = _bbox([(r, c) for r, c, _ in seed])
    by_dir = defaultdict(list)
    for r, c, col in markers:
        if r == tr0 + 1:
            by_dir["up"].append(col)
        elif r == tr1 - 1:
            by_dir["down"].append(col)
        elif c == tc0 + 1:
            by_dir["left"].append(col)
        elif c == tc1 - 1:
            by_dir["right"].append(col)
    for d, cols in by_dir.items():
        col = Counter(cols).most_common(1)[0][0]
        n = len(cols)
        for layer in range(1, n + 1):
            for r, c, _ in seed:
                if d == "up" and r == rmin:
                    out[r - layer, c] = col
                elif d == "down" and r == rmax:
                    out[r + layer, c] = col
                elif d == "left" and c == cmin:
                    out[r, c - layer] = col
                elif d == "right" and c == cmax:
                    out[r, c + layer] = col
    return out


def generate_5():
    g = np.zeros((20, 24), dtype=int)
    size = random.choice([5, 6])
    _draw_frame(g, 3, 3, size, size, GREY)
    seed_col = random.choice([2, 3, 4, 6, 7, 8, 9])
    g[5:7, 5:7] = seed_col
    _draw_frame(g, 3, 14, size, size, GREY)
    dirs = ["up", "down", "left", "right"]
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 4)
    for d, col in zip(dirs, colors):
        n = random.randint(0, 3)
        if d == "up":
            cells = [(4, 15 + i) for i in range(n)]
        elif d == "down":
            cells = [(3 + size - 2, 15 + i) for i in range(n)]
        elif d == "left":
            cells = [(4 + i, 15) for i in range(n)]
        else:
            cells = [(4 + i, 14 + size - 2) for i in range(n)]
        for r, c in cells:
            g[r, c] = col
    return g


def _partitions(g):
    rows = [-1] + [r for r in range(g.shape[0]) if np.all(g[r, :] == GREY)] + [g.shape[0]]
    cols = [-1] + [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)] + [g.shape[1]]
    boxes = []
    for i in range(len(rows) - 1):
        for j in range(len(cols) - 1):
            r0, r1 = rows[i] + 1, rows[i + 1] - 1
            c0, c1 = cols[j] + 1, cols[j + 1] - 1
            if r0 <= r1 and c0 <= c1:
                boxes.append((r0, c0, r1, c1))
    return boxes


def main_6(grid):
    g = np.array(grid, copy=True)
    bg = _bg(g)
    cells = []
    for box in _partitions(g):
        r0, c0, r1, c1 = box
        sub = g[r0:r1 + 1, c0:c1 + 1]
        objs = _components_by_color(sub, exclude={bg, GREY, AZURE})
        if objs:
            obj = max(objs, key=lambda x: len(x["cells"]))
            obj["cells"] = [(r + r0, c + c0) for r, c in obj["cells"]]
            obj["bbox"] = _bbox(obj["cells"])
            obj["box"] = box
            cells.append(obj)
    by_color = defaultdict(list)
    for obj in cells:
        by_color[obj["color"]].append(obj)
    for color, pair in by_color.items():
        if len(pair) != 2:
            continue
        pair.sort(key=lambda x: len(x["cells"]))
        small, large = pair
        sr0, sc0, sr1, sc1 = small["box"]
        g[sr0:sr1 + 1, sc0:sc1 + 1] = bg
        lr0, lc0, lr1, lc1 = large["box"]
        azure = list(zip(*np.where(g[lr0:lr1 + 1, lc0:lc1 + 1] == AZURE)))
        if not azure:
            continue
        ar0, ac0, ar1, ac1 = _bbox([(r + lr0, c + lc0) for r, c in azure])
        m = _mask(small["cells"])
        _put(g, m, ar0, ac0, color)
    return g


def generate_6():
    g = np.zeros((21, 21), dtype=int)
    g[10, :] = GREY
    g[:, 10] = GREY
    colors = random.sample([2, 3, 4, 6, 7, 9], 2)
    boxes = [(0, 0, 9, 9), (0, 11, 9, 20), (11, 0, 20, 9), (11, 11, 20, 20)]
    random.shuffle(boxes)
    for col, small_box, large_box in [(colors[0], boxes[0], boxes[1]), (colors[1], boxes[2], boxes[3])]:
        sr0, sc0, _, _ = small_box
        lr0, lc0, _, _ = large_box
        m = random.choice([
            np.array([[1, 1], [1, 0]], dtype=bool),
            np.ones((2, 2), dtype=bool),
            np.ones((1, 3), dtype=bool),
        ])
        _put(g, m, sr0 + 2, sc0 + 2, col)
        big = np.ones((m.shape[0] + 2, m.shape[1] + 2), dtype=bool)
        _put(g, big, lr0 + 1, lc0 + 1, col)
        g[lr0 + 5:lr0 + 5 + m.shape[0], lc0 + 5:lc0 + 5 + m.shape[1]] = AZURE
    return g


def main_7(grid):
    g = np.array(grid, copy=True)
    bg = _bg(g)
    grey_frames = [c for c in _components_by_color(g, exclude={bg}) if c["color"] == GREY and _is_frame(c["cells"])]
    if not grey_frames:
        return g
    frame = max(grey_frames, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))
    r0, c0, r1, c1 = frame["bbox"]
    cols = []
    for comp in _components_by_color(g, exclude={bg, GREY}):
        br0, bc0, br1, bc1 = comp["bbox"]
        if r0 < br0 <= br1 < r1 and c0 < bc0 <= bc1 < c1:
            cols.append((len(comp["cells"]), comp["color"]))
    order = [c for _, c in sorted(cols, reverse=True)]
    for k, col in enumerate(order, start=1):
        rr0, cc0, rr1, cc1 = r0 - k, c0 - k, r1 + k, c1 + k
        if rr0 < 0 or cc0 < 0 or rr1 >= g.shape[0] or cc1 >= g.shape[1]:
            continue
        g[rr0, cc0:cc1 + 1] = col
        g[rr1, cc0:cc1 + 1] = col
        g[rr0:rr1 + 1, cc0] = col
        g[rr0:rr1 + 1, cc1] = col
    return g


def generate_7():
    g = np.zeros((26, 26), dtype=int)
    _draw_frame(g, 6, 6, 14, 14, GREY)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(3, 5))
    rects = [(8, 8, 6, 7), (10, 11, 6, 6), (7, 12, 5, 5), (12, 8, 5, 8), (9, 9, 4, 9)]
    for col, (r, c, h, w) in zip(colors, rects):
        g[r:r + h, c:c + w] = col
    return g

def main(grid):
    'Recolor each grey outline by counting its internal grey hole markers.\n\nInput structure: Black background, an ordered edge palette, and grey closed frames containing small isolated grey marker blocks.\nRule: Count the marker blocks inside each frame and repaint only that frame boundary with the palette color at the counted index.\nKey insight: The palette is an ordered lookup table and the number of enclosed markers is the lookup key.'
    return main_1(grid)


def generate_input():
    return generate_1()
