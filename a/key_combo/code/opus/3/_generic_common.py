import random
from collections import Counter, defaultdict, deque

import numpy as np


FOLDER_MODES = {
    11: ["flood", "gravity", "ray", "gravity", "crop", "periodic", "crop", "contour", "flood", "scale"],
    12: ["scale", "ray", "contour", "stamp", "flood", "majority", "connect", "stamp", "stamp", "rings"],
    13: ["flood", "ray", "stamp", "connect", "periodic", "overlay", "contour", "crop", "scale", "overlay"],
    14: ["contour", "rings", "connect", "crop", "contour", "overlay", "ray", "connect", "rings"],
    15: ["ray", "gravity", "rings", "ray", "stamp", "crop", "contour", "periodic"],
    16: ["gravity", "rings", "connect", "periodic", "connect", "flood", "ray", "scale", "connect"],
    17: ["gravity", "ray", "contour", "periodic", "crop", "majority", "rings", "crop", "stamp"],
    18: ["stamp", "connect", "overlay", "rings", "gravity", "flood", "ray", "crop"],
    19: ["flood", "majority", "stamp", "scale", "majority", "periodic", "crop"],
    20: ["gravity", "connect", "overlay", "rings", "ray", "gravity", "crop", "periodic", "stamp", "contour"],
    41: ["crop", "stamp", "stamp", "gravity", "periodic", "contour", "majority"],
    42: ["rings", "stamp", "connect", "rings", "connect", "crop", "ray", "overlay"],
    43: ["ray", "ray", "majority", "crop", "gravity", "scale", "flood", "scale"],
    44: ["periodic", "crop", "overlay", "connect", "contour", "scale", "stamp", "stamp"],
    45: ["overlay", "ray", "stamp", "connect", "majority", "periodic", "ray", "contour", "stamp", "crop"],
    46: ["rings", "flood", "overlay", "gravity", "periodic", "rings", "connect", "majority"],
    47: ["gravity", "rings", "ray", "scale", "ray", "overlay", "crop", "connect"],
    48: ["periodic", "overlay", "majority", "crop", "stamp", "gravity", "ray", "periodic", "stamp", "ray"],
    49: ["rings", "scale", "rings", "stamp", "ray", "periodic", "periodic", "crop"],
    50: ["gravity", "contour", "ray", "stamp", "ray", "periodic", "stamp", "scale", "gravity", "periodic"],
}


def _mode(folder, task):
    modes = FOLDER_MODES.get(folder, [])
    if 1 <= task <= len(modes):
        return modes[task - 1]
    fallback = ["flood", "gravity", "ray", "stamp", "crop", "periodic", "contour", "scale", "connect", "majority", "rings", "overlay"]
    return fallback[(folder * 17 + task * 5) % len(fallback)]


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
    points = np.asarray(points)
    rs = points[:, 0]
    cs = points[:, 1]
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _paste(out, patch, top, left, overwrite_zero=False):
    h, w = out.shape
    for r in range(patch.shape[0]):
        for c in range(patch.shape[1]):
            v = int(patch[r, c])
            if v == 0:
                continue
            rr, cc = top + r, left + c
            if 0 <= rr < h and 0 <= cc < w and (not overwrite_zero or out[rr, cc] == 0):
                out[rr, cc] = v


def _count_holes(mask):
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bg[r, c]:
                continue
            q = [(r, c)]
            seen[r, c] = True
            touches = False
            while q:
                x, y = q.pop()
                if x == 0 or y == 0 or x == h - 1 or y == w - 1:
                    touches = True
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dr, y + dc
                    if 0 <= nx < h and 0 <= ny < w and bg[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            if not touches:
                holes += 1
    return holes


def _main_flood(grid):
    g = np.array(grid, dtype=int)
    out = g.copy()
    walls = (g == 5) | (g == 1)
    for comp in _components(~walls):
        vals = [(int(g[r, c]), int(r), int(c)) for r, c in comp if g[r, c] not in (0, 5, 1)]
        touches = np.any((comp[:, 0] == 0) | (comp[:, 1] == 0) | (comp[:, 0] == g.shape[0] - 1) | (comp[:, 1] == g.shape[1] - 1))
        if vals:
            color, mr, mc = vals[0]
            for r, c in comp:
                if g[r, c] == 0:
                    out[r, c] = color if (r + c + mr + mc) % 2 == 0 else 0
        elif touches:
            for r, c in comp:
                if g[r, c] == 0:
                    out[r, c] = 8
    return out


def _main_gravity(grid):
    g = np.array(grid, dtype=int)
    out = np.zeros_like(g)
    fixed = (g == 5) | (g == 9)
    out[fixed] = g[fixed]
    for c in range(g.shape[1]):
        free_rows = [r for r in range(g.shape[0]) if not fixed[r, c]]
        vals = [int(g[r, c]) for r in free_rows if g[r, c] not in (0, 5, 9)]
        k = len(vals)
        for i, v in enumerate(vals):
            out[free_rows[-k + i], c] = v
    for r, c in np.argwhere(out == 9):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and out[rr, cc] == 0:
                out[rr, cc] = 7
    return out


def _main_ray(grid):
    g = np.array(grid, dtype=int)
    out = g.copy()
    h, w = g.shape
    markers = [(r, c, int(g[r, c])) for r in range(h) for c in range(w) if g[r, c] not in (0, 5, 1)]
    dirs = [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for idx, (r, c, color) in enumerate(sorted(markers)):
        dr, dc = dirs[(color + idx) % len(dirs)]
        rr, cc = r + dr, c + dc
        while 0 <= rr < h and 0 <= cc < w:
            if g[rr, cc] in (5, 1):
                break
            if out[rr, cc] == 0:
                out[rr, cc] = color
            rr += dr
            cc += dc
    return out


def _main_stamp(grid):
    g = np.array(grid, dtype=int)
    out = g.copy()
    comps = _components((g != 0) & (g != 5) & (g != 9), diag=True)
    if not comps:
        return out
    sprite_comp = max(comps, key=len)
    r0, c0, r1, c1 = _bbox(sprite_comp)
    sprite = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    sprite[(sprite == 5) | (sprite == 9)] = 0
    markers = np.argwhere(g == 9)
    for i, (mr, mc) in enumerate(markers):
        patch = sprite.copy()
        if (mr + mc) % 2:
            patch = np.fliplr(patch)
        if (mr // 2 + mc // 2 + i) % 2:
            patch = np.flipud(patch)
        _paste(out, patch, int(mr) - patch.shape[0] // 2, int(mc) - patch.shape[1] // 2, overwrite_zero=False)
    return out


def _main_crop(grid):
    g = np.array(grid, dtype=int)
    comps = _components(g != 0, diag=False)
    crops = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        crop = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        holes = _count_holes(crop != 0)
        area = int(np.sum(crop != 0))
        crops.append((holes, -area, r0, c0, crop))
    if not crops:
        return np.zeros((1, 1), dtype=int)
    crops.sort()
    crops = crops[: min(5, len(crops))]
    out_h = max(c.shape[0] for *_, c in crops)
    out_w = sum(c.shape[1] for *_, c in crops) + len(crops) - 1
    out = np.zeros((min(30, out_h), min(30, out_w)), dtype=int)
    x = 0
    for _, _, _, _, crop in crops:
        crop = crop[: out.shape[0], : max(0, out.shape[1] - x)]
        out[out.shape[0] - crop.shape[0] :, x : x + crop.shape[1]] = crop
        x += crop.shape[1] + 1
        if x >= out.shape[1]:
            break
    return out


def _find_period(g):
    h, w = g.shape
    best = None
    for ph in range(1, min(5, h) + 1):
        for pw in range(1, min(5, w) + 1):
            vals = {}
            ok = True
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
            if ok and len(vals) >= max(1, ph * pw // 2):
                area = ph * pw
                if best is None or area < best[0]:
                    best = (area, ph, pw, vals)
    return best


def _main_periodic(grid):
    g = np.array(grid, dtype=int)
    info = _find_period(g)
    if info is None:
        return g.copy()
    _, ph, pw, vals = info
    out = g.copy()
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            out[r, c] = vals.get((r % ph, c % pw), out[r, c])
    for r in range(out.shape[0]):
        if r % 2 == 1:
            out[r, out[r] == 3] = 4
    return out


def _main_contour(grid):
    g = np.array(grid, dtype=int)
    out = g.copy()
    for color in [int(v) for v in np.unique(g) if v not in (0, 5, 9)]:
        comps = _components(g == color, diag=False)
        for comp in comps:
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            label = 9
            inside = g[r0 + 1 : r1, c0 + 1 : c1]
            vals = [int(v) for v in np.unique(inside) if v not in (0, color)]
            if vals:
                label = vals[0]
            for r in range(r0 + 1, r1):
                for c in range(c0 + 1, c1):
                    if out[r, c] == 0:
                        out[r, c] = label
    return out


def _main_scale(grid):
    g = np.array(grid, dtype=int)
    pts = np.argwhere((g != 0) & (g != 5) & (g != 9))
    if pts.size == 0:
        return g.copy()
    r0, c0, r1, c1 = _bbox(pts)
    crop = g[r0 : r1 + 1, c0 : c1 + 1]
    markers = int(np.sum(g == 9))
    scale = max(1, min(3, markers if markers else (int(np.sum(crop != 0)) % 3) + 1))
    out = np.kron(crop, np.ones((scale, scale), dtype=int))
    return out[:30, :30]


def _main_connect(grid):
    g = np.array(grid, dtype=int)
    out = g.copy()
    colors = [int(v) for v in np.unique(g) if v not in (0, 5)]
    for color in colors:
        pts = np.argwhere(g == color)
        if len(pts) < 2:
            continue
        pts = sorted(map(tuple, pts))
        for (r0, c0), (r1, c1) in zip(pts, pts[1:]):
            r, c = int(r0), int(c0)
            while r != r1 or c != c1:
                if r < r1:
                    r += 1
                elif r > r1:
                    r -= 1
                if c < c1:
                    c += 1
                elif c > c1:
                    c -= 1
                if out[r, c] == 0:
                    out[r, c] = color
    return out


def _main_majority(grid):
    g = np.array(grid, dtype=int)
    out = g.copy()
    rows = [-1] + [r for r in range(g.shape[0]) if np.all(g[r] == 5)] + [g.shape[0]]
    cols = [-1] + [c for c in range(g.shape[1]) if np.all(g[:, c] == 5)] + [g.shape[1]]
    for i in range(len(rows) - 1):
        for j in range(len(cols) - 1):
            r0, r1 = rows[i] + 1, rows[i + 1]
            c0, c1 = cols[j] + 1, cols[j + 1]
            sub = g[r0:r1, c0:c1]
            vals = [int(v) for v in sub.ravel() if v not in (0, 5)]
            if not vals:
                continue
            majority = Counter(vals).most_common(1)[0][0]
            out[r0:r1, c0:c1][sub == 0] = majority
    return out


def _main_rings(grid):
    g = np.array(grid, dtype=int)
    out = g.copy()
    comps = _components((g != 0) & (g != 5), diag=True)
    palette = [1, 2, 3, 4, 6, 7, 8, 9]
    for idx, comp in enumerate(comps):
        r0, c0, r1, c1 = _bbox(comp)
        thick = 1 + (len(comp) % 2)
        color = palette[(idx + len(comp)) % len(palette)]
        for t in range(1, thick + 1):
            rr0, cc0 = max(0, r0 - t), max(0, c0 - t)
            rr1, cc1 = min(out.shape[0] - 1, r1 + t), min(out.shape[1] - 1, c1 + t)
            out[rr0, cc0 : cc1 + 1] = np.where(out[rr0, cc0 : cc1 + 1] == 0, color, out[rr0, cc0 : cc1 + 1])
            out[rr1, cc0 : cc1 + 1] = np.where(out[rr1, cc0 : cc1 + 1] == 0, color, out[rr1, cc0 : cc1 + 1])
            out[rr0 : rr1 + 1, cc0] = np.where(out[rr0 : rr1 + 1, cc0] == 0, color, out[rr0 : rr1 + 1, cc0])
            out[rr0 : rr1 + 1, cc1] = np.where(out[rr0 : rr1 + 1, cc1] == 0, color, out[rr0 : rr1 + 1, cc1])
    return out


def _main_overlay(grid):
    g = np.array(grid, dtype=int)
    cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == 5)]
    rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == 5)]
    if cols:
        c = cols[0]
        a, b = g[:, :c], g[:, c + 1 :]
    elif rows:
        r = rows[0]
        a, b = g[:r, :], g[r + 1 :, :]
    else:
        return g.copy()
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    a, b = a[:h, :w], b[:h, :w]
    out = np.zeros((max(1, h), max(1, w)), dtype=int)
    out[(a != 0) & (b == 0)] = a[(a != 0) & (b == 0)]
    out[(b != 0) & (a == 0)] = b[(b != 0) & (a == 0)]
    both = (a != 0) & (b != 0)
    out[both] = 3
    return out


MAIN_FUNCS = {
    "flood": _main_flood,
    "gravity": _main_gravity,
    "ray": _main_ray,
    "stamp": _main_stamp,
    "crop": _main_crop,
    "periodic": _main_periodic,
    "contour": _main_contour,
    "scale": _main_scale,
    "connect": _main_connect,
    "majority": _main_majority,
    "rings": _main_rings,
    "overlay": _main_overlay,
}


def _gen_flood(folder, task):
    h = random.randint(13, 17)
    w = random.randint(13, 17)
    g = np.zeros((h, w), dtype=int)
    for r in random.sample(range(3, h - 3), random.randint(1, 3)):
        g[r, :] = 5
    for c in random.sample(range(3, w - 3), random.randint(1, 3)):
        g[:, c] = 5
    free = np.argwhere(g == 0)
    random.shuffle(free)
    for r, c in free[: random.randint(4, 8)]:
        g[int(r), int(c)] = random.choice([2, 3, 4, 6, 7, 8, 9])
    return g


def _gen_gravity(folder, task):
    h = random.randint(14, 20)
    w = random.randint(12, 18)
    g = np.zeros((h, w), dtype=int)
    for _ in range(random.randint(18, 35)):
        g[random.randrange(h - 2), random.randrange(w)] = random.choice([1, 2, 3, 4, 6, 7, 8])
    for _ in range(random.randint(1, 3)):
        g[random.randrange(3, h - 2), random.randrange(w)] = 5
    for _ in range(random.randint(1, 4)):
        g[random.randrange(h), random.randrange(w)] = 9
    return g


def _gen_ray(folder, task):
    h = w = random.randint(14, 18)
    g = np.zeros((h, w), dtype=int)
    for _ in range(random.randint(10, 22)):
        if random.random() < 0.5:
            g[random.randrange(h), random.randrange(w)] = 5
    for color in random.sample([2, 3, 4, 6, 7, 8, 9], random.randint(3, 6)):
        r, c = random.randrange(2, h - 2), random.randrange(2, w - 2)
        g[r, c] = color
    return g


def _gen_stamp(folder, task):
    h = w = random.randint(16, 20)
    g = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8], 3)
    sprite = np.array([[colors[0], 0, colors[1]], [colors[0], colors[1], 0], [0, colors[2], colors[2]]], dtype=int)
    r0, c0 = random.randint(1, 3), random.randint(1, 3)
    _paste(g, np.rot90(sprite, random.randrange(4)), r0, c0)
    for _ in range(random.randint(4, 8)):
        g[random.randrange(3, h - 3), random.randrange(3, w - 3)] = 9
    return g


def _gen_crop(folder, task):
    h = random.randint(16, 20)
    w = random.randint(16, 20)
    g = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(4, 7))
    for color in colors:
        ph, pw = random.randint(2, 5), random.randint(2, 5)
        for _ in range(100):
            r, c = random.randint(0, h - ph), random.randint(0, w - pw)
            area = g[max(0, r - 1) : min(h, r + ph + 1), max(0, c - 1) : min(w, c + pw + 1)]
            if np.any(area != 0):
                continue
            patch = np.full((ph, pw), color, dtype=int)
            if ph > 2 and pw > 2 and random.random() < 0.5:
                patch[random.randrange(1, ph - 1), random.randrange(1, pw - 1)] = 0
            g[r : r + ph, c : c + pw] = patch
            break
    return g


def _gen_periodic(folder, task):
    ph, pw = random.choice([(2, 3), (3, 3), (3, 4), (4, 3)])
    tile = np.array(random.choices([1, 2, 3, 4, 6, 7], k=ph * pw), dtype=int).reshape(ph, pw)
    reps_r = random.randint(3, 5)
    reps_c = random.randint(3, 5)
    g = np.tile(tile, (reps_r, reps_c))[: random.randint(ph * 3, min(20, ph * reps_r)), : random.randint(pw * 3, min(20, pw * reps_c))]
    cells = [(r, c) for r in range(g.shape[0]) for c in range(g.shape[1])]
    random.shuffle(cells)
    for r, c in cells[: random.randint(6, min(25, len(cells) // 3))]:
        g[r, c] = 0
    return g


def _gen_contour(folder, task):
    h = w = random.randint(15, 20)
    g = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8], random.randint(2, 4))
    for i, color in enumerate(colors):
        ph, pw = random.randint(4, 7), random.randint(4, 7)
        r = random.randint(0, h - ph)
        c = random.randint(0, w - pw)
        g[r, c : c + pw] = color
        g[r + ph - 1, c : c + pw] = color
        g[r : r + ph, c] = color
        g[r : r + ph, c + pw - 1] = color
        g[random.randint(r + 1, r + ph - 2), random.randint(c + 1, c + pw - 2)] = random.choice([5, 8, 9])
    return g


def _gen_scale(folder, task):
    h = w = random.randint(12, 16)
    g = np.zeros((h, w), dtype=int)
    ph, pw = random.randint(3, 6), random.randint(3, 6)
    r, c = random.randint(2, h - ph - 1), random.randint(2, w - pw - 1)
    for rr in range(ph):
        for cc in range(pw):
            if random.random() < 0.65:
                g[r + rr, c + cc] = random.choice([1, 2, 3, 4, 6, 7])
    for _ in range(random.randint(1, 3)):
        g[random.randrange(h), random.randrange(w)] = 9
    return g


def _gen_connect(folder, task):
    h = w = random.randint(14, 18)
    g = np.zeros((h, w), dtype=int)
    for color in random.sample([1, 2, 3, 4, 6, 7, 8], random.randint(3, 5)):
        for _ in range(random.randint(2, 4)):
            g[random.randrange(1, h - 1), random.randrange(1, w - 1)] = color
    for _ in range(random.randint(0, 8)):
        g[random.randrange(h), random.randrange(w)] = 5
    return g


def _gen_majority(folder, task):
    cell = random.randint(3, 5)
    rows = random.randint(2, 4)
    cols = random.randint(2, 4)
    g = np.zeros((rows * cell + rows - 1, cols * cell + cols - 1), dtype=int)
    for r in range(1, rows):
        g[r * cell + r - 1, :] = 5
    for c in range(1, cols):
        g[:, c * cell + c - 1] = 5
    for rr in range(rows):
        for cc in range(cols):
            r0, c0 = rr * (cell + 1), cc * (cell + 1)
            colors = random.sample([1, 2, 3, 4, 6, 7, 8], 2)
            for _ in range(random.randint(2, cell * cell - 1)):
                g[r0 + random.randrange(cell), c0 + random.randrange(cell)] = random.choice(colors)
    return g


def _gen_rings(folder, task):
    h = w = random.randint(15, 20)
    g = np.zeros((h, w), dtype=int)
    for color in random.sample([1, 2, 3, 4, 6, 7, 8], random.randint(3, 6)):
        r, c = random.randrange(2, h - 2), random.randrange(2, w - 2)
        g[r, c] = color
        if random.random() < 0.7:
            g[r, min(w - 1, c + 1)] = color
    return g


def _gen_overlay(folder, task):
    h = random.randint(8, 14)
    w = random.randint(8, 14)
    g = np.zeros((h, 2 * w + 1), dtype=int)
    g[:, w] = 5
    for panel in (0, 1):
        off = panel * (w + 1)
        for _ in range(random.randint(12, 28)):
            g[random.randrange(h), off + random.randrange(w)] = random.choice([1, 2, 3, 4, 6, 7, 8])
    return g


GEN_FUNCS = {
    "flood": _gen_flood,
    "gravity": _gen_gravity,
    "ray": _gen_ray,
    "stamp": _gen_stamp,
    "crop": _gen_crop,
    "periodic": _gen_periodic,
    "contour": _gen_contour,
    "scale": _gen_scale,
    "connect": _gen_connect,
    "majority": _gen_majority,
    "rings": _gen_rings,
    "overlay": _gen_overlay,
}


def main_for(folder, task, grid):
    mode = _mode(folder, task)
    return MAIN_FUNCS[mode](grid)


def generate_for(folder, task):
    mode = _mode(folder, task)
    return GEN_FUNCS[mode](folder, task)
