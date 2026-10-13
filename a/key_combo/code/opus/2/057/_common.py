import random
from collections import Counter, defaultdict, deque

import numpy as np


def _components(mask, diag=False):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diag:
        dirs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            q = [(r, c)]
            seen[r, c] = True
            pts = []
            while q:
                x, y = q.pop()
                pts.append((x, y))
                for dr, dc in dirs:
                    nx, ny = x + dr, y + dc
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(points):
    rs = points[:, 0]
    cs = points[:, 1]
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _main_1(grid):
    g = np.array(grid, dtype=int)
    out = g.copy()
    obstacles = (g == 1) | (g == 3)
    for comp in _components(~obstacles):
        touches = np.any((comp[:, 0] == 0) | (comp[:, 0] == g.shape[0] - 1) | (comp[:, 1] == 0) | (comp[:, 1] == g.shape[1] - 1))
        if touches:
            for r, c in comp:
                if g[r, c] == 0:
                    out[r, c] = 5
            continue
        marks = [(int(r), int(c)) for r, c in comp if g[r, c] == 2]
        if not marks:
            continue
        mr, mc = marks[0]
        for r, c in comp:
            if g[r, c] in (0, 2):
                out[r, c] = 4 if max(abs(int(r) - mr), abs(int(c) - mc)) % 2 == 0 else 7
    return out


def _generate_1():
    g = np.zeros((17, 17), dtype=int)
    boxes = [(2, 2, 6, 6), (9, 3, 14, 8), (4, 10, 12, 15)]
    colors = [1, 3, 1]
    random.shuffle(boxes)
    for (r0, c0, r1, c1), color in zip(boxes, colors):
        g[r0, c0 : c1 + 1] = color
        g[r1, c0 : c1 + 1] = color
        g[r0 : r1 + 1, c0] = color
        g[r0 : r1 + 1, c1] = color
        g[random.randint(r0 + 1, r1 - 1), random.randint(c0 + 1, c1 - 1)] = 2
    for _ in range(random.randint(4, 8)):
        g[random.randrange(17), random.randrange(17)] = random.choice([1, 3]) if random.random() < 0.35 else g[random.randrange(17), random.randrange(17)]
    return g


def _main_2(grid):
    g = np.array(grid, dtype=int)
    out = g.copy()
    h, w = g.shape
    stripe_cols = []
    for c in range(w):
        vals = np.unique(g[:, c])
        if len(vals) == 1 and vals[0] not in (0, 2, 5):
            stripe_cols.append(c)
    palette = [int(g[0, c]) for c in sorted(stripe_cols)]
    seeds = np.argwhere(g == 2)
    if not palette or seeds.size == 0:
        return out
    r, c = map(int, seeds[0])
    dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    length = 2
    edge = 0
    dir_idx = 0
    while length <= 40:
        color = palette[edge % len(palette)]
        dr, dc = dirs[dir_idx % 4]
        for _ in range(length):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if out[nr, nc] not in (0,):
                    return out
                out[nr, nc] = color
            r, c = nr, nc
        edge += 1
        dir_idx += 1
        if edge % 2 == 0:
            length += 2
    return out


def _generate_2():
    g = np.zeros((15, 15), dtype=int)
    cols = sorted(random.sample(range(1, 14), random.randint(3, 4)))
    colors = random.sample([1, 3, 4, 6], len(cols))
    for c, color in zip(cols, colors):
        g[:, c] = color
    empties = [c for c in range(3, 12) if c not in cols]
    c = random.choice(empties)
    g[7, c] = 2
    for _ in range(random.randint(0, 2)):
        rr, cc = random.randrange(15), random.randrange(15)
        if g[rr, cc] == 0:
            g[rr, cc] = 5
    return g


def _main_3(grid):
    g = np.array(grid, dtype=int)
    out = g.copy()
    anchors = []
    for r in range(g.shape[0] - 1):
        for c in range(g.shape[1] - 1):
            if np.all(g[r : r + 2, c : c + 2] == 1):
                anchors.append((r, c))
    if not anchors:
        return out
    ar, ac = anchors[0]
    green = _components(g == 3)
    if not green:
        return out
    sprite_comp = max(green, key=len)
    r0, c0, r1, c1 = _bbox(sprite_comp)
    sprite = (g[r0 : r1 + 1, c0 : c1 + 1] == 3).astype(int) * 3
    markers = [(r, c, int(g[r, c])) for r in range(g.shape[0]) for c in range(g.shape[1]) if g[r, c] not in (0, 1, 3)]
    for r, c, color in sorted(markers):
        row_aligned = r in (ar, ar + 1)
        col_aligned = c in (ac, ac + 1)
        if not (row_aligned or col_aligned):
            continue
        if row_aligned:
            step = 1 if c < ac else -1
            for cc in range(c + step, ac if step == 1 else ac + 1, step):
                if out[r, cc] == 0:
                    out[r, cc] = color
            patch = np.fliplr(sprite)
        else:
            step = 1 if r < ar else -1
            for rr in range(r + step, ar if step == 1 else ar + 1, step):
                if out[rr, c] == 0:
                    out[rr, c] = color
            patch = np.flipud(sprite)
        top = r - patch.shape[0] // 2
        left = c - patch.shape[1] // 2
        for rr in range(patch.shape[0]):
            for cc in range(patch.shape[1]):
                if patch[rr, cc] and 0 <= top + rr < g.shape[0] and 0 <= left + cc < g.shape[1] and out[top + rr, left + cc] == 0:
                    out[top + rr, left + cc] = 3
    return out


def _generate_3():
    g = np.zeros((16, 16), dtype=int)
    ar, ac = random.randint(6, 8), random.randint(6, 8)
    g[ar : ar + 2, ac : ac + 2] = 1
    sprite = np.array([[3, 0, 0], [3, 3, 0], [0, 3, 3]], dtype=int)
    g[2:5, 2:5] = sprite
    marks = [(ar, 1), (ar + 1, 14), (1, ac), (14, ac + 1)]
    colors = random.sample([2, 4, 6, 7], 4)
    for (r, c), color in zip(random.sample(marks, random.randint(3, 4)), colors):
        g[r, c] = color
    return g


def _hole_count(crop):
    return int(np.sum(crop == 0))


def _main_4(grid):
    g = np.array(grid, dtype=int)
    h, w = g.shape
    seed_rows = []
    for r in range(h):
        vals = np.unique(g[r, :])
        if len(vals) == 1 and vals[0] not in (0, 5):
            seed_rows.append((r, int(vals[0])))
    blocks = _components(g == 5)
    groups = defaultdict(list)
    recolored = g.copy()
    for comp in blocks:
        r0, c0, r1, c1 = _bbox(comp)
        if c1 - c0 + 1 < 3 or r1 - r0 + 1 < 3:
            continue
        center = (r0 + r1) // 2
        sr, scolor = min(seed_rows, key=lambda x: (abs(center - x[0]), x[0])) if seed_rows else (0, 5)
        for r, c in comp:
            recolored[r, c] = scolor
        groups[(sr, scolor)].append((r0, c0, r1, c1, _hole_count(g[r0 : r1 + 1, c0 : c1 + 1]), scolor))
    crops = []
    for key in sorted(groups):
        cnts = Counter(item[4] for item in groups[key])
        uniq = [item for item in groups[key] if cnts[item[4]] == 1]
        if not uniq:
            continue
        r0, c0, r1, c1, _, _ = uniq[0]
        crops.append(recolored[r0 : r1 + 1, c0 : c1 + 1])
    if not crops:
        return np.zeros((1, 1), dtype=int)
    out_h = sum(c.shape[0] for c in crops) + len(crops) - 1
    out_w = max(c.shape[1] for c in crops)
    out = np.zeros((out_h, out_w), dtype=int)
    r = 0
    for crop in crops:
        out[r : r + crop.shape[0], : crop.shape[1]] = crop
        r += crop.shape[0] + 1
    return out


def _block(g, r, c, hh, ww, holes):
    g[r : r + hh, c : c + ww] = 5
    cells = [(rr, cc) for rr in range(r + 1, r + hh - 1) for cc in range(c + 1, c + ww - 1)]
    random.shuffle(cells)
    for rr, cc in cells[:holes]:
        g[rr, cc] = 0


def _generate_4():
    g = np.zeros((20, 15), dtype=int)
    rows = [3, 9, 15]
    colors = random.sample([1, 2, 3, 4, 6], 3)
    for r, color in zip(rows, colors):
        g[r, :] = color
    specs = [(0, 1, 4, 4, 0), (0, 7, 4, 5, 1), (5, 1, 4, 4, 2), (5, 8, 4, 5, 2), (11, 1, 4, 4, 1), (11, 8, 4, 5, 3)]
    for spec in specs:
        _block(g, *spec)
    return g


def _find_period(g):
    h, w = g.shape
    best = None
    for ph in range(1, min(6, h) + 1):
        for pw in range(1, min(6, w) + 1):
            ok = True
            vals = {}
            for r in range(h):
                for c in range(w):
                    if g[r, c] == 0:
                        continue
                    key = (r % ph, c % pw)
                    if key in vals and vals[key] != int(g[r, c]):
                        ok = False
                        break
                    vals[key] = int(g[r, c])
                if not ok:
                    break
            if ok and len(vals) == ph * pw:
                area = ph * pw
                if best is None or area < best[0]:
                    best = (area, ph, pw, vals)
    return best


def _main_5(grid):
    g = np.array(grid, dtype=int)
    info = _find_period(g)
    if info is None:
        full = g.copy()
    else:
        _, ph, pw, vals = info
        full = g.copy()
        for r in range(g.shape[0]):
            for c in range(g.shape[1]):
                full[r, c] = vals[(r % ph, c % pw)]
    out = full.copy()
    h, w = out.shape
    starts = []
    for r in range(h):
        for c in range(w):
            if full[r, c] == 3 and (r == 0 or c == 0 or full[r - 1, c - 1] != 3):
                starts.append((r, c))
    for r, c in starts:
        seq = []
        rr, cc = r, c
        while rr < h and cc < w and full[rr, cc] == 3:
            seq.append((rr, cc))
            rr += 1
            cc += 1
        for i, (rr, cc) in enumerate(seq):
            if i % 2 == 1:
                out[rr, cc] = 4
    return out


def _generate_5():
    tile = np.array([[3, 1, 6], [2, 3, 1], [6, 2, 3]], dtype=int)
    g = np.tile(tile, (4, 4))
    while True:
        out = g.copy()
        holes = random.sample([(r, c) for r in range(12) for c in range(12)], random.randint(10, 18))
        for r, c in holes:
            out[r, c] = 0
        if _find_period(out) is not None:
            return out


def _main_6(grid):
    g = np.array(grid, dtype=int)
    out = g.copy()
    markers = [tuple(p) for p in np.argwhere(g == 4)]
    obstacles = [tuple(p) for p in np.argwhere(g == 5)]
    cover = np.zeros(g.shape, dtype=int)
    for mr, mc in sorted(markers):
        if obstacles:
            d = min(max(abs(mr - r), abs(mc - c)) for r, c in obstacles)
        else:
            d = 3
        cells = []
        if d >= 2:
            cells += [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1)]
        if d >= 3:
            cells += [(-1, -1, 2), (-1, 1, 2), (1, -1, 2), (1, 1, 2)]
        for dr, dc, color in cells:
            r, c = mr + dr, mc + dc
            if 0 <= r < g.shape[0] and 0 <= c < g.shape[1] and g[r, c] not in (4, 5):
                out[r, c] = color
                cover[r, c] += 1
    out[(cover >= 2) & (g != 4) & (g != 5)] = 3
    out[g == 4] = 4
    out[g == 5] = 5
    return out


def _generate_6():
    g = np.zeros((14, 14), dtype=int)
    cells = [(r, c) for r in range(1, 13) for c in range(1, 13)]
    random.shuffle(cells)
    markers = []
    while len(markers) < random.randint(4, 6) and cells:
        r, c = cells.pop()
        if all(max(abs(r - rr), abs(c - cc)) >= 3 for rr, cc in markers):
            markers.append((r, c))
            g[r, c] = 4
    for _ in range(random.randint(3, 5)):
        r, c = random.randrange(14), random.randrange(14)
        if g[r, c] == 0:
            g[r, c] = 5
    return g


def _main_7(grid):
    g = np.array(grid, dtype=int)
    out = g.copy()
    h, w = g.shape
    cr, cc = h // 2, w // 2
    blues = np.argwhere(g == 1)
    for r, c in blues:
        pts = [(int(r), int(c))]
        dr, dc = int(r) - cr, int(c) - cc
        pts += [(cr - dc, cc + dr), (cr - dr, cc - dc), (cr + dc, cc - dr)]
        for rr, cc2 in pts:
            if 0 <= rr < h and 0 <= cc2 < w and out[rr, cc2] == 0:
                out[rr, cc2] = 1
    seeds = [(r, c, int(out[r, c])) for r in range(h) for c in range(w) if out[r, c] not in (0, 1)]
    for sr, sc, color in seeds:
        q = deque([(sr, sc)])
        seen = {(sr, sc)}
        while q:
            r, c = q.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in seen:
                    continue
                if out[nr, nc] == 0:
                    out[nr, nc] = color
                    seen.add((nr, nc))
                    q.append((nr, nc))
    return out


def _generate_7():
    g = np.zeros((13, 13), dtype=int)
    # Two partial corner L shapes; rotation in main completes the rest.
    for k in range(1, random.randint(4, 6)):
        g[1, k] = 1
        g[k, 1] = 1
    if random.random() < 0.7:
        for k in range(7, 12):
            g[11, k] = 1
            g[k, 11] = 1
    for color in random.sample([2, 3, 4], random.randint(2, 3)):
        r, c = random.randint(3, 9), random.randint(3, 9)
        if g[r, c] == 0:
            g[r, c] = color
    return g


def main_for(task_id, grid):
    return {
        1: _main_1,
        2: _main_2,
        3: _main_3,
        4: _main_4,
        5: _main_5,
        6: _main_6,
        7: _main_7,
    }[task_id](grid)


def generate_for(task_id):
    return {
        1: _generate_1,
        2: _generate_2,
        3: _generate_3,
        4: _generate_4,
        5: _generate_5,
        6: _generate_6,
        7: _generate_7,
    }[task_id]()
