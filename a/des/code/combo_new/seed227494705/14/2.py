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


def _shape_dims(shape):
    if not shape:
        return 0, 0
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


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


def _connected(shape):
    if not shape:
        return False
    return len(_components(np.array([[((r, c) in shape) for c in range(max(x[1] for x in shape) + 1)] for r in range(max(x[0] for x in shape) + 1)], dtype=bool))) == 1


def _palette_row(grid):
    best = None
    for r in range(grid.shape[0]):
        c = 0
        while c < grid.shape[1]:
            if grid[r, c] == BG:
                c += 1
                continue
            start = c
            vals = []
            while c < grid.shape[1] and grid[r, c] != BG:
                vals.append(int(grid[r, c]))
                c += 1
            if len(vals) >= 2 and len(vals) == len(set(vals)):
                if best is None or len(vals) > len(best[2]):
                    best = (r, start, vals)
        c += 1
    return best


def main_1(grid):
    out = np.array(grid, copy=True)
    pal = _palette_row(grid)
    if pal is None:
        return out
    pr, pc, palette = pal
    palette_cells = {(pr, c) for c in range(pc, pc + len(palette))}
    for color in set(int(x) for x in grid.ravel() if x != BG):
        mask = grid == color
        for r, c in palette_cells:
            if grid[r, c] == color:
                mask[r, c] = False
        comps = _components(mask)
        frames = [comp for comp in comps if len(comp) >= 8 and (_bbox(comp)[1] - _bbox(comp)[0] >= 2) and (_bbox(comp)[3] - _bbox(comp)[2] >= 2)]
        anchors = [comp[0] for comp in comps if len(comp) == 1]
        for frame in frames:
            if not anchors:
                continue
            r0, r1, c0, c1 = _bbox(frame)
            side_data = []
            top_missing = [(r0, c) for c in range(c0 + 1, c1) if grid[r0, c] != color]
            bot_missing = [(r1, c) for c in range(c0 + 1, c1) if grid[r1, c] != color]
            left_missing = [(r, c0) for r in range(r0 + 1, r1) if grid[r, c0] != color]
            right_missing = [(r, c1) for r in range(r0 + 1, r1) if grid[r, c1] != color]
            for side, miss in (("top", top_missing), ("bottom", bot_missing), ("left", left_missing), ("right", right_missing)):
                if miss:
                    side_data.append((side, len(miss)))
            if not side_data:
                continue
            side, gap_len = max(side_data, key=lambda x: x[1])
            if gap_len - 1 >= len(palette):
                continue
            ar, ac = min(anchors, key=lambda p: abs(p[0] - (r0 + r1) // 2) + abs(p[1] - (c0 + c1) // 2))
            fill = palette[gap_len - 1]
            if side == "top":
                _rect(out, ar - gap_len, ar - 1, ac, ac + gap_len - 1, fill)
            elif side == "bottom":
                _rect(out, ar + 1, ar + gap_len, ac, ac + gap_len - 1, fill)
            elif side == "left":
                _rect(out, ar, ar + gap_len - 1, ac - gap_len, ac - 1, fill)
            else:
                _rect(out, ar, ar + gap_len - 1, ac + 1, ac + gap_len, fill)
            out[ar, ac] = color
    return out


def gen_1():
    h = w = 30
    grid = np.zeros((h, w), dtype=int)
    n = random.randint(2, 4)
    frame_colors = random.sample([BLUE, RED, GREEN, YELLOW], n)
    palette = random.sample([MAGENTA, ORANGE, AZURE, MAROON, GREY], n)
    grid[0, 1 : 1 + n] = palette
    used = grid != BG
    for i, color in enumerate(frame_colors):
        gap_len = i + 1
        for _attempt in range(200):
            rh, rw = random.randint(5, 8), random.randint(5, 8)
            r0, c0 = random.randint(5, h - rh - 5), random.randint(5, w - rw - 5)
            r1, c1 = r0 + rh - 1, c0 + rw - 1
            side = random.choice(["top", "bottom", "left", "right"])
            if side in ("top", "bottom") and gap_len > rw - 2:
                continue
            if side in ("left", "right") and gap_len > rh - 2:
                continue
            frame = _perimeter(r0, r1, c0, c1)
            if side == "top":
                start = random.randint(c0 + 1, c1 - gap_len)
                frame -= {(r0, c) for c in range(start, start + gap_len)}
                anchor = (max(1, r0 - 2), start)
            elif side == "bottom":
                start = random.randint(c0 + 1, c1 - gap_len)
                frame -= {(r1, c) for c in range(start, start + gap_len)}
                anchor = (min(h - 2, r1 + 2), start)
            elif side == "left":
                start = random.randint(r0 + 1, r1 - gap_len)
                frame -= {(r, c0) for r in range(start, start + gap_len)}
                anchor = (start, max(1, c0 - 2))
            else:
                start = random.randint(r0 + 1, r1 - gap_len)
                frame -= {(r, c1) for r in range(start, start + gap_len)}
                anchor = (start, min(w - 2, c1 + 2))
            pad = [(r, c) for r in range(r0 - 3, r1 + 4) for c in range(c0 - 3, c1 + 4) if 0 <= r < h and 0 <= c < w]
            if all(not used[r, c] for r, c in pad) and not used[anchor]:
                _paint(grid, frame, color)
                grid[anchor] = color
                for r, c in pad:
                    used[r, c] = True
                used[anchor] = True
                break
    return grid


def _full_rect_pair():
    for _ in range(500):
        h, w = 4, 4
        cells = {(r, c) for r in range(h) for c in range(w)}
        a = {cell for cell in cells if random.random() < 0.5}
        b = cells - a
        if len(a) < 3 or len(b) < 3:
            continue
        for shape in (a, b):
            rs = [r for r, _ in shape]
            cs = [c for _, c in shape]
            if min(rs) != 0 or max(rs) != h - 1 or min(cs) != 0 or max(cs) != w - 1:
                break
            if not _connected(shape):
                break
        else:
            return a, b
    h, w = 4, 4
    a = {(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 0)}
    return a, {(r, c) for r in range(h) for c in range(w)} - a


def main_2(grid):
    v_lines = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    edges = [-1] + v_lines + [grid.shape[1]]
    pieces = []
    for idx in range(len(edges) - 1):
        c0, c1 = edges[idx] + 1, edges[idx + 1] - 1
        if c0 > c1:
            continue
        panel = grid[:, c0 : c1 + 1]
        comps = _components((panel != BG) & (panel != GREY))
        if not comps:
            continue
        comp = max(comps, key=len)
        color = int(panel[comp[0][0], comp[0][1]])
        abs_comp = [(r, c0 + c) for r, c in comp]
        pieces.append({"idx": idx, "shape": _norm(abs_comp), "color": color})
    pairs = []
    used = set()
    for i in range(len(pieces)):
        if i in used:
            continue
        for j in range(i + 1, len(pieces)):
            if j in used:
                continue
            s1, s2 = pieces[i]["shape"], pieces[j]["shape"]
            union = s1 | s2
            if s1 & s2:
                continue
            rh, rw = _shape_dims(union)
            if len(union) == rh * rw:
                pairs.append((rh * rw, pieces[i], pieces[j], rh, rw))
                used.update({i, j})
                break
    pairs.sort(key=lambda x: x[0])
    if not pairs:
        return np.zeros((1, 1), dtype=int)
    out_h = max(p[3] for p in pairs)
    out_w = sum(p[4] for p in pairs) + len(pairs) - 1
    out = np.zeros((out_h, out_w), dtype=int)
    ccur = 0
    for _, a, b, rh, rw in pairs:
        left, right = (a, b) if a["idx"] < b["idx"] else (b, a)
        for r, c in left["shape"]:
            out[r, ccur + c] = left["color"]
        for r, c in right["shape"]:
            out[r, ccur + c] = right["color"]
        ccur += rw + 1
    return out


def gen_2():
    pair_count = 2
    pairs = [_full_rect_pair() for _ in range(pair_count)]
    max_h = max(max(r for r, _ in a | b) + 1 for a, b in pairs)
    max_w = max(max(c for _, c in a | b) + 1 for a, b in pairs)
    panel_h = max_h + 2
    panel_w = max_w + 2
    panel_count = pair_count * 2
    h = panel_h
    w = panel_count * panel_w + panel_count - 1
    grid = np.zeros((h, w), dtype=int)
    for c in range(panel_w, w, panel_w + 1):
        grid[:, c] = GREY
    colors = random.sample([BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, AZURE, MAROON], panel_count)
    panels = []
    for a, b in pairs:
        panels.extend([a, b])
    order = list(range(panel_count))
    random.shuffle(order)
    for panel_index, piece_index in enumerate(order):
        shape = panels[piece_index]
        c_base = panel_index * (panel_w + 1)
        color = colors[piece_index]
        for r, c in shape:
            grid[1 + r, c_base + 1 + c] = color
    return grid


def main_3(grid):
    out = np.array(grid, copy=True)
    for comp in _components(grid == GREEN):
        r0, r1, c0, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        seeds = [(r, c, int(grid[r, c])) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if grid[r, c] not in (BG, GREEN)]
        if len(seeds) != 1:
            continue
        sr, sc, color = seeds[0]
        side_counts = {
            "top": sum(grid[r0, c] != GREEN for c in range(c0 + 1, c1)),
            "bottom": sum(grid[r1, c] != GREEN for c in range(c0 + 1, c1)),
            "left": sum(grid[r, c0] != GREEN for r in range(r0 + 1, r1)),
            "right": sum(grid[r, c1] != GREEN for r in range(r0 + 1, r1)),
        }
        side = max(side_counts, key=side_counts.get)
        if side_counts[side] == 0:
            continue
        if side == "top":
            dist = sr - r0
            _rect(out, r0 - dist, r0 - 1, c0 + 1, c1 - 1, color)
        elif side == "bottom":
            dist = r1 - sr
            _rect(out, r1 + 1, r1 + dist, c0 + 1, c1 - 1, color)
        elif side == "left":
            dist = sc - c0
            _rect(out, r0 + 1, r1 - 1, c0 - dist, c0 - 1, color)
        else:
            dist = c1 - sc
            _rect(out, r0 + 1, r1 - 1, c1 + 1, c1 + dist, color)
    return out


def gen_3():
    h = w = 30
    grid = np.zeros((h, w), dtype=int)
    for color in random.sample([BLUE, RED, YELLOW, MAGENTA, ORANGE, AZURE, MAROON], random.randint(2, 4)):
        for _attempt in range(200):
            rh, rw = random.randint(5, 8), random.randint(5, 8)
            r0, c0 = random.randint(5, h - rh - 5), random.randint(5, w - rw - 5)
            r1, c1 = r0 + rh - 1, c0 + rw - 1
            side = random.choice(["top", "bottom", "left", "right"])
            frame = _perimeter(r0, r1, c0, c1)
            if side == "top":
                frame.remove((r0, random.randint(c0 + 1, c1 - 1)))
            elif side == "bottom":
                frame.remove((r1, random.randint(c0 + 1, c1 - 1)))
            elif side == "left":
                frame.remove((random.randint(r0 + 1, r1 - 1), c0))
            else:
                frame.remove((random.randint(r0 + 1, r1 - 1), c1))
            pad = [(r, c) for r in range(r0 - rh, r1 + rh + 1) for c in range(c0 - rw, c1 + rw + 1) if 0 <= r < h and 0 <= c < w]
            if all(grid[r, c] == BG for r, c in pad):
                _paint(grid, frame, GREEN)
                grid[random.randint(r0 + 1, r1 - 1), random.randint(c0 + 1, c1 - 1)] = color
                break
    return grid


def _trace_path(path_cells, source_cells):
    path = set(path_cells)
    neigh = {p: [q for q in ((p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1)) if q in path] for p in path}
    endpoints = [p for p, ns in neigh.items() if len(ns) <= 1]
    if not endpoints:
        endpoints = [next(iter(path))]
    source = set(source_cells)
    start = min(endpoints, key=lambda p: min(abs(p[0] - r) + abs(p[1] - c) for r, c in source))
    ordered = [start]
    prev = None
    cur = start
    while True:
        nxts = [p for p in neigh[cur] if p != prev]
        if not nxts:
            break
        prev, cur = cur, nxts[0]
        ordered.append(cur)
    return ordered


def main_4(grid):
    out = np.array(grid, copy=True)
    source_comps = _components((grid != BG) & (grid != GREY) & (grid != AZURE))
    receivers = [_bbox(comp) for comp in _components(grid == AZURE) if len(comp) >= 8]
    for path in _components(grid == GREY):
        src = None
        for comp in source_comps:
            if any(abs(r - pr) + abs(c - pc) == 1 for r, c in comp for pr, pc in path):
                src = comp
                break
        if src is None:
            continue
        recv = None
        for box in receivers:
            rr0, rr1, cc0, cc1 = box
            border = _perimeter(rr0, rr1, cc0, cc1)
            if any(abs(r - pr) + abs(c - pc) == 1 for r, c in border for pr, pc in path):
                recv = box
                break
        if recv is None:
            continue
        ordered = _trace_path(path, src)
        dirs = []
        for (r1, c1), (r2, c2) in zip(ordered, ordered[1:]):
            dirs.append((np.sign(r2 - r1), np.sign(c2 - c1)))
        turns = sum(1 for a, b in zip(dirs, dirs[1:]) if a != b)
        color = int(grid[src[0][0], src[0][1]])
        sr0, sr1, sc0, sc1 = _bbox(src)
        arr = np.zeros((sr1 - sr0 + 1, sc1 - sc0 + 1), dtype=int)
        for r, c in src:
            arr[r - sr0, c - sc0] = 1
        arr = np.rot90(arr, k=(-turns) % 4)
        for r, c in path + src:
            out[r, c] = BG
        rr0, rr1, cc0, cc1 = recv
        ih, iw = rr1 - rr0 - 1, cc1 - cc0 - 1
        top = rr0 + 1 + (ih - arr.shape[0]) // 2
        left = cc0 + 1 + (iw - arr.shape[1]) // 2
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                if arr[r, c]:
                    out[top + r, left + c] = color
    return out


def gen_4():
    grid = np.zeros((30, 30), dtype=int)
    color = random.choice([BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, MAROON])
    shape = random.choice([
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (0, 1), (1, 1)],
        [(0, 1), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
    ])
    sr, sc = random.randint(4, 10), random.randint(2, 6)
    for dr, dc in shape:
        grid[sr + dr, sc + dc] = color
    rr, rc = random.randint(8, 20), random.randint(20, 24)
    _paint(grid, _perimeter(rr, rr + 5, rc, rc + 5), AZURE)
    start = (sr + 1, sc + 3)
    end = (rr + 2, rc - 1)
    bends = random.randint(0, 3)
    points = [start]
    if bends == 0:
        points += [(start[0], end[1])]
    elif bends == 1:
        midc = random.randint(10, 16)
        points += [(start[0], midc), (end[0], midc)]
    else:
        midc = random.randint(10, 15)
        midr = random.randint(3, 24)
        points += [(start[0], midc), (midr, midc), (midr, end[1])]
    points.append(end)
    for (r1, c1), (r2, c2) in zip(points, points[1:]):
        if r1 == r2:
            for c in range(min(c1, c2), max(c1, c2) + 1):
                if grid[r1, c] == BG:
                    grid[r1, c] = GREY
        else:
            for r in range(min(r1, r2), max(r1, r2) + 1):
                if grid[r, c1] == BG:
                    grid[r, c1] = GREY
    return grid


def main_5(grid):
    red_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == RED)]
    if not red_rows:
        return np.array(grid, copy=True)
    sep = red_rows[0]
    out = np.array(grid, copy=True)
    canvas_rows = range(sep + 1, grid.shape[0])
    for r in range(sep):
        cols = [c for c in range(grid.shape[1]) if grid[r, c] not in (BG, RED)]
        if not cols:
            continue
        color = int(grid[r, cols[0]])
        if len(cols) >= 3:
            gaps = [b - a for a, b in zip(cols, cols[1:])]
            uniform = len(set(gaps)) == 1
        else:
            uniform = len(cols) == 2
        if uniform:
            step = cols[1] - cols[0] if len(cols) > 1 else grid.shape[1]
            targets = list(range(cols[0], grid.shape[1], step))
        else:
            targets = cols
        for c in targets:
            for rr in canvas_rows:
                out[rr, c] = color
    return out


def gen_5():
    w = random.randint(12, 30)
    rule_h = random.randint(3, 6)
    canvas_h = random.randint(5, 12)
    grid = np.zeros((rule_h + 1 + canvas_h, w), dtype=int)
    grid[rule_h, :] = RED
    for r, color in enumerate(random.sample([BLUE, GREEN, YELLOW, MAGENTA, ORANGE, AZURE, MAROON], rule_h)):
        if random.random() < 0.5:
            step = random.randint(2, 5)
            start = random.randint(0, step - 1)
            cols = list(range(start, w, step))[: random.randint(2, max(2, min(5, (w - start + step - 1) // step)))]
        else:
            cols = sorted(random.sample(range(w), random.randint(3, min(6, w))))
            if len(set(b - a for a, b in zip(cols, cols[1:]))) == 1:
                cols[-1] = (cols[-1] + 1) % w
                cols = sorted(set(cols))
        for c in cols:
            grid[r, c] = color
    return grid


def main_6(grid):
    out = np.array(grid, copy=True)
    grey = _components(grid == GREY)
    if not grey:
        return out
    key_box = _bbox(max(grey, key=len))
    kr0, kr1, kc0, kc1 = key_box
    key_shapes = {}
    key_mask = np.zeros_like(grid, dtype=bool)
    key_mask[kr0 : kr1 + 1, kc0 : kc1 + 1] = True
    for comp in _components((grid != BG) & (grid != GREY) & key_mask):
        color = int(grid[comp[0][0], comp[0][1]])
        key_shapes[_norm(comp)] = color
    tile_mask = (grid != BG) & (grid != GREY) & (~key_mask)
    for comp in _components(tile_mask):
        r0, r1, c0, c1 = _bbox(comp)
        holes = _holes(grid, r0, r1, c0, c1)
        if not holes:
            continue
        shape = _norm(max(holes, key=len))
        if shape not in key_shapes:
            continue
        for r, c in comp:
            out[r, c] = key_shapes[shape]
    return out


def gen_6():
    grid = np.zeros((30, 30), dtype=int)
    key_count = random.randint(2, 4)
    _paint(grid, _perimeter(1, 12, 1, 8), GREY)
    shapes = random.sample([
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 1)],
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ], key_count)
    key_colors = random.sample([BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, AZURE, MAROON], key_count)
    for i, (shape, color) in enumerate(zip(shapes, key_colors)):
        base_r = 2 + i * 3
        base_c = 3
        for dr, dc in shape:
            grid[base_r + dr, base_c + dc] = color
    tile_colors = random.sample([c for c in COLORS if c not in key_colors and c != GREY], key_count)
    for i, shape in enumerate(shapes):
        r0 = 2 + (i % 2) * 12 + random.randint(0, 2)
        c0 = 12 + (i // 2) * 8 + random.randint(0, 2)
        rh, rw = 6, 6
        grid[r0 : r0 + rh, c0 : c0 + rw] = tile_colors[i]
        hr, hc = r0 + 2, c0 + 2
        for dr, dc in shape:
            grid[hr + dr, hc + dc] = BG
    return grid


def _room_windows(grid, box, color):
    r0, r1, c0, c1 = box
    wins = []
    for side, cells in (
        ("top", [(r0, c) for c in range(c0 + 1, c1)]),
        ("bottom", [(r1, c) for c in range(c0 + 1, c1)]),
        ("left", [(r, c0) for r in range(r0 + 1, r1)]),
        ("right", [(r, c1) for r in range(r0 + 1, r1)]),
    ):
        missing = [p for p in cells if grid[p] != color]
        for p in missing:
            wins.append((side, p))
    return wins


def main_7(grid):
    out = np.array(grid, copy=True)
    rooms = []
    for color in set(int(x) for x in grid.ravel() if x != BG):
        for comp in _components(grid == color):
            r0, r1, c0, c1 = _bbox(comp)
            if r1 - r0 >= 4 and c1 - c0 >= 4 and all(r in (r0, r1) or c in (c0, c1) for r, c in comp):
                rooms.append({"color": color, "box": (r0, r1, c0, c1)})
    wall_colors = {room["color"] for room in rooms}
    source = None
    pattern = []
    for room in rooms:
        r0, r1, c0, c1 = room["box"]
        cells = [(r, c, int(grid[r, c])) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if grid[r, c] not in wall_colors and grid[r, c] != BG]
        if cells:
            source = room
            pattern = cells
            break
    if source is None:
        return out
    sr0, sr1, sc0, sc1 = source["box"]
    for side, (wr, wc) in _room_windows(grid, source["box"], source["color"]):
        candidates = []
        for room in rooms:
            if room is source:
                continue
            r0, r1, c0, c1 = room["box"]
            if side == "right" and c0 > sc1 and r0 <= wr <= r1:
                candidates.append((c0 - sc1, room))
            elif side == "left" and c1 < sc0 and r0 <= wr <= r1:
                candidates.append((sc0 - c1, room))
            elif side == "bottom" and r0 > sr1 and c0 <= wc <= c1:
                candidates.append((r0 - sr1, room))
            elif side == "top" and r1 < sr0 and c0 <= wc <= c1:
                candidates.append((sr0 - r1, room))
        if not candidates:
            continue
        target = min(candidates, key=lambda x: x[0])[1]
        tr0, tr1, tc0, tc1 = target["box"]
        if side == "right":
            sref, tref = (wr, sc1 - 1), (wr, tc0 + 1)
            flip = "h"
        elif side == "left":
            sref, tref = (wr, sc0 + 1), (wr, tc1 - 1)
            flip = "h"
        elif side == "bottom":
            sref, tref = (sr1 - 1, wc), (tr0 + 1, wc)
            flip = "v"
        else:
            sref, tref = (sr0 + 1, wc), (tr1 - 1, wc)
            flip = "v"
        for r, c, col in pattern:
            dr, dc = r - sref[0], c - sref[1]
            nr, nc = (tref[0] + dr, tref[1] - dc) if flip == "h" else (tref[0] - dr, tref[1] + dc)
            if tr0 < nr < tr1 and tc0 < nc < tc1:
                out[nr, nc] = col
    return out


def gen_7():
    grid = np.zeros((30, 30), dtype=int)
    source_color, target_color = random.sample([BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, AZURE, MAROON], 2)
    source_box = (9, 17, 3, 11)
    if random.random() < 0.5:
        target_box = (9, 17, 13, 21)
        side, twin = "right", "left"
        wr, wc = random.randint(11, 15), 11
    else:
        target_box = (19, 27, 3, 11)
        side, twin = "bottom", "top"
        wr, wc = 17, random.randint(5, 9)
    for box, color, open_side in ((source_box, source_color, side), (target_box, target_color, twin)):
        r0, r1, c0, c1 = box
        frame = _perimeter(r0, r1, c0, c1)
        if open_side in ("left", "right"):
            c = c0 if open_side == "left" else c1
            frame.discard((wr, c))
        else:
            r = r0 if open_side == "top" else r1
            frame.discard((r, wc))
        _paint(grid, frame, color)
    pattern_color = random.choice([c for c in COLORS if c not in (source_color, target_color)])
    for dr, dc in random.choice([
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
    ]):
        grid[12 + dr, 6 + dc] = pattern_color
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    "Pair complementary panel fragments and pack their completed rectangles by area.\n\nInput structure:\nGrey vertical dividers split the black grid into panels. Each panel contains\none monochrome irregular connected shape, and shapes come in complementary\npairs whose normalized union is a solid rectangle.\nRule:\nMatch complementary shapes geometrically, compute each completed rectangle's\narea, and output the pairs from left to right in increasing area with one\nblack column between them. The earlier input panel's color paints its cells,\nand the partner color paints the remaining cells.\nKey insight:\nPairing is based on geometric complementarity, then ordering is based on the\narea of the assembled rectangle."
    return main_2(grid)


def generate_input() -> np.ndarray:
    return gen_2()
