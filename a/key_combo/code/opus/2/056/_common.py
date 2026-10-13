import random
from collections import deque

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
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and mask[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(points):
    rs = points[:, 0]
    cs = points[:, 1]
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _place_nonoverlap(grid, patch, tries=200):
    h, w = grid.shape
    ph, pw = patch.shape
    for _ in range(tries):
        r = random.randint(0, h - ph)
        c = random.randint(0, w - pw)
        area = grid[max(0, r - 1) : min(h, r + ph + 1), max(0, c - 1) : min(w, c + pw + 1)]
        if np.any(area != 0):
            continue
        m = patch != 0
        grid[r : r + ph, c : c + pw][m] = patch[m]
        return True
    return False


def _main_1(grid):
    g = np.array(grid, dtype=int)
    out = g.copy()
    for comp in _components(g != 5):
        vals = [(int(g[r, c]), int(r), int(c)) for r, c in comp if int(g[r, c]) not in (0, 5)]
        if not vals:
            continue
        color, mr, mc = vals[0]
        for r, c in comp:
            if g[r, c] == 0 and r % 2 == mr % 2 and c % 2 == mc % 2:
                out[r, c] = color
    return out


def _generate_1():
    g = np.zeros((15, 15), dtype=int)
    rows = sorted(random.sample(range(4, 11), 2))
    cols = sorted(random.sample(range(4, 11), 2))
    g[rows, :] = 5
    g[:, cols] = 5
    r_edges = [-1] + rows + [15]
    c_edges = [-1] + cols + [15]
    cells = []
    for i in range(3):
        for j in range(3):
            r0, r1 = r_edges[i] + 1, r_edges[i + 1]
            c0, c1 = c_edges[j] + 1, c_edges[j + 1]
            if r1 - r0 >= 3 and c1 - c0 >= 3:
                cells.append((r0, r1, c0, c1))
    random.shuffle(cells)
    for r0, r1, c0, c1 in cells[: random.randint(4, min(7, len(cells)))]:
        g[random.randrange(r0, r1), random.randrange(c0, c1)] = random.choice([1, 2, 3, 4, 6, 7, 8])
    return g


def _main_2(grid):
    g = np.array(grid, dtype=int)
    h, w = g.shape
    legend_cols = [c for c in range(w) if g[h - 2, c] != 0 and g[h - 1, c] != 0]
    mapping = {int(g[h - 2, c]): int(g[h - 1, c]) for c in legend_cols}
    body = g[: h - 2, :]
    counts = {color: 0 for color in mapping}
    seen = np.zeros(body.shape, dtype=bool)
    for r in range(body.shape[0] - 1):
        for c in range(body.shape[1] - 1):
            color = int(body[r, c])
            if color not in counts or seen[r, c]:
                continue
            if np.all(body[r : r + 2, c : c + 2] == color):
                counts[color] += 1
                seen[r : r + 2, c : c + 2] = True
    order = sorted(counts, key=lambda x: (counts[x], legend_cols[list(mapping).index(x)] if x in mapping else x))
    if not order:
        return np.zeros((1, 1), dtype=int)
    size = counts[order[0]] + 2 * sum(counts[c] for c in order[1:])
    size = max(1, min(30, size))
    out = np.zeros((size, size), dtype=int)
    lo, hi = 0, size
    for color in reversed(order):
        thick = max(1, counts[color])
        out[lo:hi, lo:hi] = mapping[color]
        if hi - lo > 2 * thick:
            out[lo + thick : hi - thick, lo + thick : hi - thick] = 0
        lo += thick
        hi -= thick
        if lo >= hi:
            break
    return out


def _generate_2():
    g = np.zeros((14, 14), dtype=int)
    colors = random.sample([1, 2, 3, 6], random.randint(3, 4))
    outs = random.sample([4, 5, 7, 8, 9], len(colors))
    counts = random.sample([1, 2, 3, 4], len(colors))
    for color, cnt in zip(colors, counts):
        placed = 0
        while placed < cnt:
            p = np.full((2, 2), color, dtype=int)
            if _place_nonoverlap(g[:11, :], p):
                placed += 1
    start = random.randint(0, 14 - len(colors))
    for i, (color, outc) in enumerate(zip(colors, outs)):
        g[12, start + i] = color
        g[13, start + i] = outc
    return g


def _main_3(grid):
    g = np.array(grid, dtype=int)
    out = g.copy()
    colors = [int(v) for v in np.unique(g) if v != 0]
    for color in colors:
        comps = _components(g == color)
        if len(comps) < 2:
            continue
        comps.sort(key=lambda p: (p[:, 0].ptp() + 1) * (p[:, 1].ptp() + 1), reverse=True)
        cont, obj = comps[0], comps[-1]
        r0, c0, r1, c1 = _bbox(cont)
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        for r, c in obj:
            out[r, c] = 0
        top = np.sum(g[r0, c0 : c1 + 1] == color)
        bottom = np.sum(g[r1, c0 : c1 + 1] == color)
        left = np.sum(g[r0 : r1 + 1, c0] == color)
        right = np.sum(g[r0 : r1 + 1, c1] == color)
        side = min([("top", top), ("bottom", bottom), ("left", left), ("right", right)], key=lambda x: x[1])[0]
        d = max(1, len(obj))
        if side == "top":
            out[max(r0 + 1, r1 - d) : r1, c0 + 1 : c1] = color
        elif side == "bottom":
            out[r0 + 1 : min(r1, r0 + 1 + d), c0 + 1 : c1] = color
        elif side == "left":
            out[r0 + 1 : r1, max(c0 + 1, c1 - d) : c1] = color
        else:
            out[r0 + 1 : r1, c0 + 1 : min(c1, c0 + 1 + d)] = color
    return out


def _draw_cup(g, r0, c0, h, w, color, open_side):
    g[r0, c0 : c0 + w] = color
    g[r0 + h - 1, c0 : c0 + w] = color
    g[r0 : r0 + h, c0] = color
    g[r0 : r0 + h, c0 + w - 1] = color
    if open_side == "top":
        g[r0, c0 + 1 : c0 + w - 1] = 0
    elif open_side == "bottom":
        g[r0 + h - 1, c0 + 1 : c0 + w - 1] = 0
    elif open_side == "left":
        g[r0 + 1 : r0 + h - 1, c0] = 0
    else:
        g[r0 + 1 : r0 + h - 1, c0 + w - 1] = 0


def _generate_3():
    g = np.zeros((16, 16), dtype=int)
    specs = [(1, 1, 5, 5), (1, 10, 5, 5), (9, 4, 5, 6)]
    colors = random.sample([1, 2, 3, 4, 6, 7], 3)
    sides = random.sample(["top", "bottom", "left", "right"], 3)
    obj_spots = [(8, 1), (8, 11), (13, 12)]
    for spec, color, side, spot in zip(specs, colors, sides, obj_spots):
        _draw_cup(g, *spec, color, side)
        cnt = random.randint(2, 4)
        r, c = spot
        for i in range(cnt):
            g[r + i // 2, c + i % 2] = color
    return g


def _regions_by_gray(g):
    h, w = g.shape
    gray_rows = [-1] + [r for r in range(h) if np.all(g[r, :] == 5)] + [h]
    gray_cols = [-1] + [c for c in range(w) if np.all(g[:, c] == 5)] + [w]
    regs = []
    for i in range(len(gray_rows) - 1):
        for j in range(len(gray_cols) - 1):
            r0, r1 = gray_rows[i] + 1, gray_rows[i + 1]
            c0, c1 = gray_cols[j] + 1, gray_cols[j + 1]
            if r0 < r1 and c0 < c1:
                regs.append((r0, c0, r1, c1))
    return regs


def _main_4(grid):
    g = np.array(grid, dtype=int)
    out = g.copy()
    regs = _regions_by_gray(g)
    if not regs:
        return out
    src = regs[0]
    r0, c0, r1, c1 = src
    sub = g[r0:r1, c0:c1]
    pts = np.argwhere((sub != 0) & (sub != 5))
    if pts.size == 0:
        return out
    br0, bc0, br1, bc1 = _bbox(pts)
    sprite = sub[br0 : br1 + 1, bc0 : bc1 + 1]
    mask = sprite != 0
    for k, (rr0, cc0, rr1, cc1) in enumerate(regs[1:], start=1):
        trans = np.fliplr(sprite) if k % 2 == 1 else np.flipud(sprite)
        m = np.fliplr(mask) if k % 2 == 1 else np.flipud(mask)
        ph, pw = trans.shape
        tr = rr0 + ((rr1 - rr0) - ph) // 2
        tc = cc0 + ((cc1 - cc0) - pw) // 2
        out[tr : tr + ph, tc : tc + pw][m] = trans[m]
    return out


def _generate_4():
    cell = random.choice([4, 5])
    rows = random.choice([2, 3])
    cols = random.choice([2, 3])
    g = np.zeros((rows * cell + rows - 1, cols * cell + cols - 1), dtype=int)
    for r in range(1, rows):
        g[r * cell + r - 1, :] = 5
    for c in range(1, cols):
        g[:, c * cell + c - 1] = 5
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 3)
    sprite = np.array(
        [[colors[0], 0, colors[1]], [colors[0], colors[1], 0], [0, colors[2], colors[2]]],
        dtype=int,
    )
    k = random.randrange(4)
    sprite = np.rot90(sprite, k)
    if random.random() < 0.5:
        sprite = np.fliplr(sprite)
    if random.random() < 0.5:
        sprite = np.flipud(sprite)
    off_r = random.randint(0, cell - 3)
    off_c = random.randint(0, cell - 3)
    g[off_r : off_r + 3, off_c : off_c + 3] = sprite
    return g


def _base_color_for_row(r):
    return 4 if (r // 2) % 2 == 0 else 3


def _main_5(grid):
    g = np.array(grid, dtype=int)
    out = g.copy()
    h, w = g.shape
    gray = np.where(g[h - 1] == 5)[0]
    p = int(abs(gray[-1] - gray[0])) if len(gray) >= 2 else 3
    p = max(1, p)
    marker_colors = [v for v in np.unique(g) if v not in (3, 4, 5)]
    markers = [(r, c, int(g[r, c])) for r in range(h - 1) for c in range(w) if int(g[r, c]) in marker_colors]
    mapping = {}
    for r, c, color in markers:
        mapping[_base_color_for_row(r)] = color
    for mr, mc, color in sorted(markers):
        for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            d = p
            while True:
                r, c = mr + dr * d, mc + dc * d
                if not (0 <= r < h and 0 <= c < w):
                    break
                if g[r, c] != 5:
                    out[r, c] = mapping.get(_base_color_for_row(r), color)
                d += p
    return out


def _generate_5():
    h = w = 16
    g = np.zeros((h, w), dtype=int)
    for r in range(h):
        g[r, :] = _base_color_for_row(r)
    p = random.randint(2, 5)
    c0 = random.randint(0, w - p - 1)
    g[h - 1, c0] = 5
    g[h - 1, c0 + p] = 5
    used = set()
    for base, color in [(4, 2), (3, 1)]:
        candidates = [(r, c) for r in range(h - 2) for c in range(1, w - 1) if _base_color_for_row(r) == base]
        r, c = random.choice(candidates)
        while (r, c) in used:
            r, c = random.choice(candidates)
        used.add((r, c))
        g[r, c] = color
    return g


def _main_6(grid):
    g = np.array(grid, dtype=int)
    h, w = g.shape
    legend = {}
    body_w = w
    rows = []
    for r in range(h):
        if g[r, w - 2] != 0 and g[r, w - 1] != 0:
            legend[int(g[r, w - 2])] = int(g[r, w - 1])
            rows.append(r)
    if legend:
        body_w = w - 2
    body = g[:, :body_w]
    comps = _components(body != 0)
    crops = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        if bh % 2 == 0 or bw % 2 == 0:
            continue
        crop = body[r0 : r1 + 1, c0 : c1 + 1].copy()
        vals, counts = np.unique(crop[crop != 0], return_counts=True)
        if vals.size:
            accent = int(vals[np.argmin(counts)])
            if accent in legend:
                crop[crop == accent] = legend[accent]
        crops.append((c0, r0, crop))
    if not crops:
        return np.zeros((1, 1), dtype=int)
    crops.sort()
    out_h = max(c.shape[0] for _, _, c in crops)
    out_w = sum(c.shape[1] for _, _, c in crops) + len(crops) - 1
    out = np.zeros((out_h, out_w), dtype=int)
    x = 0
    for _, _, crop in crops:
        y = out_h - crop.shape[0]
        out[y : y + crop.shape[0], x : x + crop.shape[1]] = crop
        x += crop.shape[1] + 1
    return out


def _generate_6():
    g = np.zeros((18, 18), dtype=int)
    accents = random.sample([1, 2, 3, 4, 6, 7], 5)
    replacements = random.sample([5, 8, 9, 2, 1, 6], 5)
    specs = [(1, 1, 3, 3), (1, 7, 3, 4), (7, 1, 5, 3), (8, 8, 4, 5), (13, 3, 3, 5)]
    bodies = random.sample([2, 3, 4, 6, 7, 8], 5)
    for (r0, c0, hh, ww), body, acc in zip(specs, bodies, accents):
        g[r0 : r0 + hh, c0 : c0 + ww] = body
        for _ in range(random.randint(1, 3)):
            g[random.randrange(r0, r0 + hh), random.randrange(c0, c0 + ww)] = acc
    for r, (acc, rep) in enumerate(zip(accents, replacements)):
        g[r, 16] = acc
        g[r, 17] = rep
    return g


def _main_7(grid):
    g = np.array(grid, dtype=int)
    out = g.copy()
    h, w = g.shape
    dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    names = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for r in range(h - 1):
        for c in range(w - 1):
            block = g[r : r + 2, c : c + 2]
            vals = [int(v) for v in block.ravel() if v != 0]
            if len(vals) == 4 and 2 in vals and len(set(vals) - {2}) == 1:
                paint = list(set(vals) - {2})[0]
                out[r : r + 2, c : c + 2] = paint
                for idx, (br, bc) in enumerate(names):
                    if block[br, bc] != 2:
                        continue
                    dr, dc = dirs[idx]
                    rr, cc = r + br + dr, c + bc + dc
                    path = []
                    companion = None
                    while 0 <= rr < h and 0 <= cc < w:
                        if g[rr, cc] not in (0, 2, paint):
                            companion = int(g[rr, cc])
                            break
                        path.append((rr, cc))
                        rr += dr
                        cc += dc
                    color = companion if companion is not None else paint
                    for pr, pc in path:
                        out[pr, pc] = color
                return out
    return out


def _generate_7():
    g = np.zeros((16, 16), dtype=int)
    r, c = random.randint(6, 8), random.randint(6, 8)
    g[r : r + 2, c : c + 2] = 7
    active = random.sample([(0, 0), (0, 1), (1, 0), (1, 1)], random.randint(1, 3))
    dir_map = {(0, 0): (-1, -1), (0, 1): (-1, 1), (1, 0): (1, -1), (1, 1): (1, 1)}
    comp_colors = random.sample([1, 3, 4, 6, 8], len(active))
    for (br, bc), color in zip(active, comp_colors):
        g[r + br, c + bc] = 2
        if random.random() < 0.7:
            dr, dc = dir_map[(br, bc)]
            dist = random.randint(3, 5)
            rr, cc = r + br + dr * dist, c + bc + dc * dist
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr, cc] = color
    return g


def _window_components(g):
    return _components(g != 5)


def _main_8(grid):
    g = np.array(grid, dtype=int)
    out = g.copy()
    comps = _window_components(g)
    windows = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        windows.append((r0, c0, r1, c1, sub))
    src = None
    for item in windows:
        sub = item[4]
        if np.any((sub != 0) & (sub != 5) & (sub != 9)):
            src = item
            break
    if src is None:
        return out
    sr0, sc0, sr1, sc1, sub = src
    sprite_pts = np.argwhere((sub != 0) & (sub != 9))
    mrk = np.argwhere(sub == 9)
    if sprite_pts.size == 0 or mrk.size == 0:
        return out
    br0, bc0, br1, bc1 = _bbox(sprite_pts)
    sprite = sub[br0 : br1 + 1, bc0 : bc1 + 1].copy()
    sprite[sprite == 9] = 0
    sm_r, sm_c = mrk[0]
    off = (int(sm_r - br0), int(sm_c - bc0))
    for r0, c0, r1, c1, win in windows:
        if (r0, c0, r1, c1) == (sr0, sc0, sr1, sc1):
            continue
        marker = np.argwhere(win == 9)
        if marker.size == 0:
            continue
        mr, mc = marker[0]
        mid_r = (win.shape[0] - 1) // 2
        mid_c = (win.shape[1] - 1) // 2
        hflip = mc > mid_c
        vflip = mr > mid_r
        patch = sprite.copy()
        roff, coff = off
        if hflip:
            patch = np.fliplr(patch)
            coff = patch.shape[1] - 1 - coff
        if vflip:
            patch = np.flipud(patch)
            roff = patch.shape[0] - 1 - roff
        top = r0 + int(mr) - roff
        left = c0 + int(mc) - coff
        for rr in range(patch.shape[0]):
            for cc in range(patch.shape[1]):
                if patch[rr, cc] == 0:
                    continue
                tr, tc = top + rr, left + cc
                if r0 <= tr <= r1 and c0 <= tc <= c1:
                    out[tr, tc] = int(patch[rr, cc])
    return out


def _generate_8():
    g = np.full((20, 20), 5, dtype=int)
    wins = [(1, 1), (1, 12), (8, 1), (8, 12), (14, 7)]
    for r, c in wins:
        g[r : r + 5, c : c + 5] = 0
    sr, sc = wins[0]
    sprite = np.array([[2, 0, 3], [2, 6, 3], [0, 6, 0]], dtype=int)
    if random.random() < 0.5:
        sprite = np.rot90(sprite)
    g[sr + 1 : sr + 4, sc + 1 : sc + 4] = sprite
    g[sr + 2, sc + 2] = 9
    marks = [(1, 1), (1, 3), (3, 1), (3, 3)]
    random.shuffle(marks)
    for (wr, wc), (mr, mc) in zip(wins[1:], marks):
        g[wr + mr, wc + mc] = 9
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
        8: _main_8,
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
        8: _generate_8,
    }[task_id]()
