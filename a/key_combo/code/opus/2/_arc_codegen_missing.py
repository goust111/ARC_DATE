import random
from collections import Counter, defaultdict, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def _arr(x):
    return np.array(x, dtype=int)


def _in(g, r, c):
    return 0 <= r < g.shape[0] and 0 <= c < g.shape[1]


def _components(g, colors=None, conn=4):
    if colors is not None:
        colors = set(colors)
    dirs = DIR4 if conn == 4 else DIR8
    seen = np.zeros(g.shape, dtype=bool)
    out = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if seen[r, c] or g[r, c] == 0 or (colors is not None and int(g[r, c]) not in colors):
                continue
            col = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if _in(g, nr, nc) and not seen[nr, nc] and int(g[nr, nc]) == col:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append((col, cells))
    return out


def _bbox(cells):
    rs = [p[0] for p in cells]
    cs = [p[1] for p in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _mask_from_cells(cells, size=None):
    r0, c0, r1, c1 = _bbox(cells)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if size is not None:
        h = max(h, size)
        w = max(w, size)
    m = np.zeros((h, w), dtype=int)
    for r, c in cells:
        if r - r0 < h and c - c0 < w:
            m[r - r0, c - c0] = 1
    return m, (r0, c0)


def _pad3(cells):
    m, origin = _mask_from_cells(cells, 3)
    return m[:3, :3], origin


def _transforms(m):
    mats = []
    for k in range(4):
        r = np.rot90(m, k)
        mats.append(r)
        mats.append(np.fliplr(r))
    return mats


def _rotations(m):
    return [np.rot90(m, k) for k in range(4)]


def _canon3(m):
    vals = []
    for t in _transforms(m):
        if t.shape != (3, 3):
            t = _fit_to(t, 3, 3)
        vals.append(tuple(int(x) for x in t.flatten()))
    return min(vals)


def _fit_to(m, h, w):
    out = np.zeros((h, w), dtype=int)
    out[: min(h, m.shape[0]), : min(w, m.shape[1])] = m[:h, :w]
    return out


def _place_cells(g, cells, color):
    for r, c in cells:
        if _in(g, r, c):
            g[r, c] = color


def _random_polyomino(n, h=3, w=3):
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < n:
        r, c = random.choice(tuple(cells))
        opts = [(r + dr, c + dc) for dr, dc in DIR4 if 0 <= r + dr < h and 0 <= c + dc < w]
        if opts:
            cells.add(random.choice(opts))
    return sorted(cells)


def _is_clear(g, cells, margin=0):
    h, w = g.shape
    for r, c in cells:
        for rr in range(r - margin, r + margin + 1):
            for cc in range(c - margin, c + margin + 1):
                if not (0 <= rr < h and 0 <= cc < w) or g[rr, cc] != 0:
                    return False
    return True


def main_024_2(grid):
    g = _arr(grid)
    out = g.copy()
    sources = [(c, int(g[0, c])) for c in range(g.shape[1]) if int(g[0, c]) not in (0, 5)]
    for src_c, color in sources:
        endpoints = [(r, c) for r in range(1, g.shape[0]) for c in range(g.shape[1]) if int(g[r, c]) == color]
        endpoint = endpoints[0] if endpoints else None
        active = [(0, src_c)]
        own = {(0, src_c)}
        done = False
        while active and not done:
            if endpoint:
                er, ec = endpoint
                reachers = [(abs(c - ec), c, r) for r, c in active if r + 1 == er]
                if reachers:
                    _, best_c, best_r = min(reachers)
                    step = 1 if ec >= best_c else -1
                    for cc in range(best_c, ec + step, step):
                        if out[er, cc] in (0, color):
                            out[er, cc] = color
                            own.add((er, cc))
                    break
            nxt = []
            for r, c in active:
                nr = r + 1
                if nr >= g.shape[0]:
                    continue
                if g[nr, c] == 5:
                    for dc in (-1, 1):
                        sc = c + dc
                        if 0 <= sc < g.shape[1] and g[r, sc] != 5 and (r, sc) not in own:
                            if out[r, sc] in (0, color):
                                out[r, sc] = color
                                own.add((r, sc))
                                nxt.append((r, sc))
                    continue
                if (nr, c) in own:
                    continue
                if out[nr, c] in (0, color):
                    out[nr, c] = color
                    own.add((nr, c))
                    nxt.append((nr, c))
            active = nxt
    out[g == 5] = 5
    return out


def generate_024_2():
    h, w = random.randint(15, 18), random.randint(18, 22)
    g = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 4, 6, 7], random.randint(3, 5))
    start = random.randint(3, w - len(colors) - 4)
    for i, col in enumerate(colors):
        g[0, start + i] = col
    used = set()
    for i, col in enumerate(colors):
        c = start + i
        r = random.randint(4, 8)
        if (r, c) not in used:
            g[r, c] = 5
            used.add((r, c))
        if random.random() < 0.35:
            rr = min(h - 4, r + random.randint(3, 5))
            cc = max(1, min(w - 2, c + random.choice([-2, -1, 1, 2])))
            if all(abs(rr - orr) + abs(cc - occ) > 1 for orr, occ in used):
                g[rr, cc] = 5
                used.add((rr, cc))
        if random.random() < 0.7:
            er = random.randint(max(r + 3, 9), h - 2)
            ec = max(1, min(w - 2, c + random.choice([-3, -2, -1, 1, 2, 3])))
            if g[er, ec] == 0 and all(abs(er - orr) + abs(ec - occ) > 1 for orr, occ in used):
                g[er, ec] = col
    for _ in range(random.randint(4, 8)):
        r, c = random.randrange(2, h - 1), random.randrange(1, w - 1)
        if g[r, c] == 0 and all(abs(r - orr) + abs(c - occ) > 1 for orr, occ in used):
            g[r, c] = 5
            used.add((r, c))
    return g


def _complete_blue_squares(g):
    squares = []
    for col, cells in _components((g == 1).astype(int), [1], 4):
        r0, c0, r1, c1 = _bbox(cells)
        if r1 - r0 != c1 - c0:
            continue
        s = set(cells)
        border = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                  if r in (r0, r1) or c in (c0, c1)}
        if border <= s:
            squares.append((r0, c0, r1, c1))
    return squares


def main_027_3(grid):
    g = _arr(grid)
    out = g.copy()
    squares = _complete_blue_squares(g)
    active_cols = set()
    for r0, c0, r1, c1 in squares:
        active_cols.update(range(c0, c1 + 1))
    for r in range(g.shape[0]):
        for c in range(1, g.shape[1] - 1):
            if g[r, c] == 0 and c in active_cols and g[r, c - 1] == 4 and g[r, c + 1] == 4:
                out[r, c] = 2
    for r0, c0, r1, c1 in squares:
        marks = [(r0 - 1, c0), (r0, c0 - 1), (r0 - 1, c1), (r0, c1 + 1),
                 (r1 + 1, c0), (r1, c0 - 1), (r1 + 1, c1), (r1, c1 + 1)]
        for r, c in marks:
            if _in(out, r, c):
                out[r, c] = 5
    return out


def generate_027_3():
    h, w = 16, 20
    for _ in range(200):
        g = np.zeros((h, w), dtype=int)
        squares = []
        for _ in range(random.randint(2, 3)):
            s = random.randint(2, 4)
            for __ in range(80):
                r = random.randint(2, h - s - 3)
                c = random.randint(2, w - s - 3)
                border = [(rr, cc) for rr in range(r, r + s) for cc in range(c, c + s)
                          if rr in (r, r + s - 1) or cc in (c, c + s - 1)]
                marks = [(r - 1, c), (r, c - 1), (r - 1, c + s - 1), (r, c + s),
                         (r + s, c), (r + s - 1, c - 1), (r + s, c + s - 1), (r + s - 1, c + s)]
                if _is_clear(g, border + marks, 1):
                    _place_cells(g, border, 1)
                    squares.append((r, c, r + s - 1, c + s - 1))
                    break
        for _ in range(random.randint(2, 3)):
            cells = [(random.randint(2, h - 3), random.randint(2, w - 3))]
            r, c = cells[0]
            cells += [(r + 1, c), (r + 1, c + 1)]
            if _is_clear(g, cells, 1):
                _place_cells(g, cells, 1)
        rows = random.sample(range(1, h - 1), 4)
        for r in rows:
            start = random.randint(1, w - 8)
            length = random.choice([5, 7, 9, 11])
            for c in range(start, min(w - 1, start + length), 2):
                if g[r, c] == 0:
                    g[r, c] = 4
        if len(squares) >= 2:
            return g
    return g


def _regions_not_blue(g):
    rid = -np.ones(g.shape, dtype=int)
    regions = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if g[r, c] == 1 or rid[r, c] >= 0:
                continue
            idx = len(regions)
            q = deque([(r, c)])
            rid[r, c] = idx
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if _in(g, nr, nc) and g[nr, nc] != 1 and rid[nr, nc] < 0:
                        rid[nr, nc] = idx
                        q.append((nr, nc))
            regions.append(cells)
    return rid, regions


def main_027_4(grid):
    g = _arr(grid)
    out = g.copy()
    rid, regions = _regions_not_blue(g)
    for idx, cells in enumerate(regions):
        reds = [(r, c) for r, c in cells if g[r, c] == 2]
        if not reds:
            continue
        r0, c0, r1, c1 = _bbox(reds)
        height = len({r for r, c in reds})
        greens = [(r, c) for r, c in cells if g[r, c] == 3]
        if len(greens) == height:
            split = height // 2
            for r, c in sorted(reds, reverse=True)[:split]:
                out[r, c] = 8
            for r, c in greens:
                out[r, c] = 0
    return out


def generate_027_4():
    h = w = 15
    g = np.zeros((h, w), dtype=int)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 1
    v1 = random.choice([5, 6])
    v2 = random.choice([10, 11])
    hr = random.choice([7, 8])
    g[:, v1] = 1
    g[:, v2] = 1
    g[hr, :] = 1
    rects = []
    rows = [(1, hr - 1), (hr + 1, h - 2)]
    cols = [(1, v1 - 1), (v1 + 1, v2 - 1), (v2 + 1, w - 2)]
    for rr in rows:
        for cc in cols:
            if rr[1] - rr[0] + 1 >= 4 and cc[1] - cc[0] + 1 >= 3:
                rects.append((rr[0], cc[0], rr[1], cc[1]))
    chosen = random.sample(rects, random.randint(3, min(4, len(rects))))
    match_flags = [True, False] + [random.choice([True, False]) for _ in chosen[2:]]
    random.shuffle(match_flags)
    for rect, match in zip(chosen, match_flags):
        r0, c0, r1, c1 = rect
        bh = random.randint(2, min(5, r1 - r0 + 1))
        bc = random.randint(c0 + 1, c1 - 1)
        top = random.randint(r0, r1 - bh + 1)
        g[top:top + bh, bc] = 2
        avail = [(rr, cc) for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1)
                 if g[rr, cc] == 0 and all(abs(rr - br) + abs(cc - bc) > 1 for br in range(top, top + bh))]
        if len(avail) < bh:
            avail = [(rr, cc) for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1) if g[rr, cc] == 0]
        if match:
            count = min(bh, len(avail))
        else:
            choices = [x for x in range(1, min(len(avail), 6) + 1) if x != bh]
            count = random.choice(choices) if choices else 0
        for rr, cc in random.sample(avail, count):
            g[rr, cc] = 3
    return g


def main_027_10(grid):
    g = _arr(grid)
    h, w = g.shape
    wall_rows = [r for r in range(h) if np.count_nonzero(g[r] == 5) >= w // 2]
    doors = {}
    for r in wall_rows:
        gaps = []
        c = 0
        while c < w:
            if g[r, c] != 5:
                s = c
                while c < w and g[r, c] != 5:
                    c += 1
                if c - s >= 3:
                    gaps.append((s, c - 1))
            c += 1
        doors[r] = gaps
    comps = [(col, cells) for col, cells in _components(g, [1, 2, 3], 4) if len(cells) >= 2]
    by_color = defaultdict(list)
    for col, cells in comps:
        r0, c0, r1, c1 = _bbox(cells)
        if c0 == c1 and len(cells) == r1 - r0 + 1:
            by_color[col].append((r0, c0, r1, c1))
    order = sorted(by_color, key=lambda col: min(b[0] for b in by_color[col]))
    cover = defaultdict(set)
    originals = {}
    for col in order:
        bars = sorted(by_color[col], key=lambda b: b[1])
        if len(bars) < 2:
            continue
        height = bars[0][2] - bars[0][0] + 1
        for bi, (r0, c, r1, _) in enumerate(bars[:2]):
            side = -1 if bi == 0 else 1
            for r in range(r0, r1 + 1):
                cover[(r, c)].add(col)
                originals[(r, c)] = col
            for direction in (-1, 1):
                r = r0 - 1 if direction == -1 else r1 + 1
                while 0 <= r < h:
                    dist = (r0 - r) if direction == -1 else (r - r1)
                    off = (dist - 1) // height + 1
                    cc = c + side * off
                    if not (0 <= cc < w):
                        break
                    if r in wall_rows:
                        if any(a <= cc <= b for a, b in doors.get(r, [])):
                            r += direction
                            continue
                        break
                    cover[(r, cc)].add(col)
                    r += direction
    out = np.zeros_like(g)
    out[g == 5] = 5
    for (r, c), cols in cover.items():
        if out[r, c] == 5:
            continue
        out[r, c] = 6 if len(cols) > 1 else next(iter(cols))
    for (r, c), col in originals.items():
        out[r, c] = col
    return out


def generate_027_10():
    h, w = 18, 20
    g = np.zeros((h, w), dtype=int)
    for wr in random.sample(range(7, 14), random.randint(1, 2)):
        g[wr, :] = 5
        d = random.randint(5, 13)
        g[wr, d:d + 3] = 0
    colors = random.sample([1, 2, 3], random.randint(2, 3))
    used_cols = set()
    for col in colors:
        bh = random.randint(2, 4)
        top = random.choice([2, 3, 11, 12])
        if top + bh >= h - 1:
            top = h - bh - 3
        for _ in range(60):
            c1 = random.randint(3, 9)
            c2 = random.randint(c1 + 3, min(w - 3, c1 + 8))
            if c1 not in used_cols and c2 not in used_cols:
                used_cols.update([c1, c2])
                g[top:top + bh, c1] = col
                g[top:top + bh, c2] = col
                break
    return g


def _symbol_mask(color):
    if color == 2:
        return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    if color == 1:
        return np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=int)
    return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int)


def _find_color_blocks(g):
    blocks = []
    for col, cells in _components(g, [1, 2, 3], 4):
        s = set(cells)
        for rem in cells:
            rest = list(s - {rem})
            if not rest:
                continue
            r0, c0, r1, c1 = _bbox(rest)
            rect = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
            if rect == set(rest) and 3 <= r1 - r0 + 1 <= 4 and 3 <= c1 - c0 + 1 <= 4:
                rr, cc = rem
                if rr == r0 - 1 and c0 <= cc <= c1:
                    d = "up"
                elif rr == r1 + 1 and c0 <= cc <= c1:
                    d = "down"
                elif cc == c0 - 1 and r0 <= rr <= r1:
                    d = "left"
                elif cc == c1 + 1 and r0 <= rr <= r1:
                    d = "right"
                else:
                    continue
                blocks.append({"color": col, "rect": (r0, c0, r1, c1), "indicator": rem, "dir": d, "cells": cells})
                break
    return blocks


def _find_frames(g):
    frames = []
    for col, cells in _components((g == 5).astype(int) * 5, [5], 4):
        r0, c0, r1, c1 = _bbox(cells)
        if r1 - r0 >= 2 and c1 - c0 >= 2:
            border = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                      if r in (r0, r1) or c in (c0, c1)}
            if border <= set(cells):
                frames.append((r0, c0, r1, c1))
    return frames


def main_027_11(grid):
    g = _arr(grid)
    out = g.copy()
    blocks = _find_color_blocks(g)
    frames = _find_frames(g)
    unused = set(range(len(frames)))
    for block in blocks:
        mask = _symbol_mask(block["color"])
        area = int(mask.sum())
        candidates = [i for i in unused if (frames[i][2] - frames[i][0] - 1) * (frames[i][3] - frames[i][1] - 1) == area]
        if not candidates:
            candidates = list(unused)
        br0, bc0, br1, bc1 = block["rect"]
        center = ((br0 + br1) / 2, (bc0 + bc1) / 2)
        def aligned(i):
            fr = frames[i]
            fc = ((fr[0] + fr[2]) / 2, (fr[1] + fr[3]) / 2)
            if block["dir"] == "up":
                return fc[0] < center[0]
            if block["dir"] == "down":
                return fc[0] > center[0]
            if block["dir"] == "left":
                return fc[1] < center[1]
            return fc[1] > center[1]
        aligned_candidates = [i for i in candidates if aligned(i)]
        if aligned_candidates:
            candidates = aligned_candidates
        idx = min(candidates, key=lambda i: abs(((frames[i][0] + frames[i][2]) / 2) - center[0]) + abs(((frames[i][1] + frames[i][3]) / 2) - center[1]))
        unused.discard(idx)
        for r, c in block["cells"]:
            out[r, c] = 0
        fr = frames[idx]
        cr = (fr[0] + fr[2]) // 2
        cc = (fr[1] + fr[3]) // 2
        for mr in range(3):
            for mc in range(3):
                if mask[mr, mc]:
                    rr, cc2 = cr + mr - 1, cc + mc - 1
                    if _in(out, rr, cc2):
                        out[rr, cc2] = 5
    return out


def generate_027_11():
    g = np.zeros((16, 16), dtype=int)
    colors = random.sample([1, 2, 3], random.randint(2, 3))
    pairs = [
        ((1, 1, 4, 6), (6, 2), "up"),
        ((1, 10, 5, 14), (7, 10), "up"),
        ((10, 1, 14, 5), (11, 7), "left"),
        ((10, 10, 14, 14), (6, 6), "right"),
    ]
    random.shuffle(pairs)
    pairs = pairs[:len(colors)]
    for spec, _, _ in pairs:
        r0, c0, r1, c1 = spec
        g[r0:r1 + 1, c0] = g[r0:r1 + 1, c1] = 5
        g[r0, c0:c1 + 1] = g[r1, c0:c1 + 1] = 5
    for col, (_, (r, c), d) in zip(colors, pairs):
        sz = 3
        g[r:r + sz, c:c + sz] = col
        mid = r + sz // 2
        if d == "left":
            g[mid, c - 1] = col
        elif d == "right":
            g[mid, c + sz] = col
        elif d == "up":
            g[r - 1, c + sz // 2] = col
        else:
            g[r + sz, c + sz // 2] = col
    return g


def main_028_9(grid):
    g = _arr(grid)
    out = g.copy()
    frames = []
    for col, cells in _components((g == 8).astype(int) * 8, [8], 4):
        r0, c0, r1, c1 = _bbox(cells)
        border = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                  if r in (r0, r1) or c in (c0, c1)]
        holes = [(r, c) for r, c in border if g[r, c] == 0]
        if holes:
            m, _ = _mask_from_cells(holes)
            frames.append({"bbox": (r0, c0, r1, c1), "holes": holes, "rots": [tuple(x.flatten()) for x in _rotations(m)]})
    sprites = [(col, cells) for col, cells in _components(g, [1, 2, 3, 4], 4)]
    used = set()
    for col, cells in sprites:
        sm, _ = _mask_from_cells(cells)
        keys = [tuple(x.flatten()) for x in _rotations(sm)]
        match = None
        for i, fr in enumerate(frames):
            if i in used:
                continue
            if any(k in fr["rots"] for k in keys):
                match = i
                break
        if match is None:
            continue
        used.add(match)
        for r, c in cells:
            out[r, c] = 0
        for r, c in frames[match]["holes"]:
            out[r, c] = col
    return out


def generate_028_9():
    g = np.zeros((18, 18), dtype=int)
    frame_pos = random.sample([(1, 1), (1, 10), (10, 1)], random.randint(2, 3))
    colors = random.sample([1, 2, 3, 4], len(frame_pos))
    for idx, ((r, c), col) in enumerate(zip(frame_pos, colors)):
        size = random.randint(5, 7)
        g[r:r + size, c] = g[r:r + size, c + size - 1] = 8
        g[r, c:c + size] = g[r + size - 1, c:c + size] = 8
        if idx % 3 == 0:
            holes = [(r, c + j) for j in range(1, 4)]
        elif idx % 3 == 1:
            holes = [(r + j, c) for j in range(1, 4)]
        else:
            holes = [(r + size - 1, c + j) for j in range(1, 4)]
        for rr, cc in holes:
            if _in(g, rr, cc):
                g[rr, cc] = 0
        m, _ = _mask_from_cells(holes)
        rot = random.choice(_rotations(m))
        placed = False
        for _ in range(100):
            sr = random.randint(1, 16 - rot.shape[0])
            sc = random.randint(1, 16 - rot.shape[1])
            cells = [(sr + rr, sc + cc) for rr in range(rot.shape[0]) for cc in range(rot.shape[1]) if rot[rr, cc]]
            if cells and _is_clear(g, cells, 2):
                _place_cells(g, cells, col)
                placed = True
                break
        if not placed:
            for sr in range(1, 17 - rot.shape[0]):
                for sc in range(1, 17 - rot.shape[1]):
                    cells = [(sr + rr, sc + cc) for rr in range(rot.shape[0]) for cc in range(rot.shape[1]) if rot[rr, cc]]
                    if cells and _is_clear(g, cells, 1):
                        _place_cells(g, cells, col)
                        placed = True
                        break
                if placed:
                    break
    return g


def main_029_1(grid):
    g = _arr(grid)
    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == 5)]
    cuts = [-1] + sep_cols + [g.shape[1]]
    panels = [g[:, cuts[i] + 1:cuts[i + 1]] for i in range(len(cuts) - 1) if cuts[i + 1] - cuts[i] > 1]
    ph, pw = panels[0].shape
    out = np.zeros((ph, pw), dtype=int)
    for p in panels:
        mask = (p != 0) & (p != 5)
        out[mask] = p[mask]
    by_color = defaultdict(list)
    for col, cells in _components(out, [1, 2, 3, 7], 4):
        by_color[col].append(_bbox(cells))
    for col, rects in by_color.items():
        if len(rects) < 2:
            continue
        a, b = rects[:2]
        r0a, c0a, r1a, c1a = a
        r0b, c0b, r1b, c1b = b
        if c1a < c0b or c1b < c0a:
            left, right = (a, b) if c1a < c0b else (b, a)
            ov0, ov1 = max(left[0], right[0]), min(left[2], right[2])
            if ov1 - ov0 + 1 >= 3:
                rows = range(ov0 + 1, ov1)
                for r in rows:
                    for c in range(left[3] + 1, right[1]):
                        if out[r, c] == 0:
                            out[r, c] = 8
        elif r1a < r0b or r1b < r0a:
            top, bot = (a, b) if r1a < r0b else (b, a)
            ov0, ov1 = max(top[1], bot[1]), min(top[3], bot[3])
            if ov1 - ov0 + 1 >= 3:
                cols = range(ov0 + 1, ov1)
                for r in range(top[2] + 1, bot[0]):
                    for c in cols:
                        if out[r, c] == 0:
                            out[r, c] = 8
    return out


def generate_029_1():
    h, pw = 15, 7
    g = np.zeros((h, pw * 3 + 2), dtype=int)
    g[:, pw] = 5
    g[:, pw * 2 + 1] = 5
    colors = random.sample([1, 2, 3, 7], random.randint(2, 4))
    placements = [((2, 1, 4, 4), (2, 1, 4, 4)), ((8, 1, 10, 4), (8, 1, 10, 4)),
                  ((3, 2, 6, 4), (9, 2, 12, 4)), ((5, 1, 7, 3), (5, 4, 7, 6))]
    for col, pair in zip(colors, random.sample(placements, len(colors))):
        panels = random.sample(range(3), 2)
        for panel, rect in zip(panels, pair):
            off = panel * (pw + 1)
            if panel == 2:
                off = pw * 2 + 2
            r0, c0, r1, c1 = rect
            g[r0:r1 + 1, off + c0:off + c1 + 1] = col
    return g


def main_029_2(grid):
    g = _arr(grid)
    comps = [(len(cells), cells) for col, cells in _components((g == 1).astype(int), [1], 4)]
    keep = max(comps, key=lambda x: x[0])[1]
    seed = np.zeros_like(g)
    for r, c in keep:
        seed[r, c] = 1
    out = np.zeros((g.shape[0] * g.shape[0], g.shape[1] * g.shape[1]), dtype=int)
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if seed[r, c] == 1:
                out[r * g.shape[0]:(r + 1) * g.shape[0], c * g.shape[1]:(c + 1) * g.shape[1]] = seed
    return out


def generate_029_2():
    g = np.zeros((5, 5), dtype=int)
    body = {(2, 2)}
    while len(body) < random.randint(8, 14):
        r, c = random.choice(tuple(body))
        opts = [(r + dr, c + dc) for dr, dc in DIR4 if 0 <= r + dr < 5 and 0 <= c + dc < 5]
        body.add(random.choice(opts))
    for r, c in body:
        g[r, c] = 1
    candidates = [(r, c) for r in range(5) for c in range(5)
                  if g[r, c] == 0 and all((r + dr, c + dc) not in body for dr, dc in DIR4)]
    for r, c in random.sample(candidates, min(len(candidates), random.randint(2, 4))):
        g[r, c] = 1
    return g


def main_common_3(grid):
    g = _arr(grid)
    out = g.copy()
    objs = []
    for col, cells in _components(g, None, 4):
        m, origin = _pad3(cells)
        objs.append((col, cells, m, origin, _canon3(m)))
    counts = Counter(o[4] for o in objs)
    for col, cells, m, (r0, c0), key in objs:
        for r, c in cells:
            out[r, c] = 0
        if counts[key] >= 2:
            block = np.kron(m, m)
            for rr in range(block.shape[0]):
                for cc in range(block.shape[1]):
                    if block[rr, cc] and _in(out, r0 + rr, c0 + cc):
                        out[r0 + rr, c0 + cc] = col
    return out


def generate_common_3():
    g = np.zeros((28, 28), dtype=int)
    base = [(0, 0), (0, 1), (1, 1), (2, 1)]
    masks = [base, [(c, 2 - r) for r, c in base], [(0, 0), (1, 0), (1, 1), (2, 2)], [(0, 1), (1, 0), (1, 1), (1, 2)], [(0, 0), (0, 1), (1, 0)]]
    positions = [(1, 1), (1, 12), (12, 1), (12, 12), (19, 19)]
    colors = random.sample([1, 2, 3, 4, 6, 7], 5)
    for (r, c), col, cells in zip(positions, colors, masks):
        for rr, cc in cells:
            g[r + rr, c + cc] = col
    return g


def main_common_4(grid):
    g = _arr(grid)
    bars = []
    marker_cells = []
    for col, cells in _components((g == 5).astype(int) * 5, [5], 4):
        r0, c0, r1, c1 = _bbox(cells)
        if c0 == c1:
            mid = (r0 + r1) // 2
            direction = None
            if c0 > 0 and g[mid, c0 - 1] not in (0, 5):
                direction = -1
                marker_cells.append((mid, c0 - 1))
            if c0 + 1 < g.shape[1] and g[mid, c0 + 1] not in (0, 5):
                direction = 1
                marker_cells.append((mid, c0 + 1))
            if direction:
                bars.append({"height": r1 - r0 + 1, "r0": r0, "r1": r1, "c": c0, "dir": direction})
    out = np.zeros_like(g)
    placed = []
    for b in sorted(bars, key=lambda x: -x["height"]):
        c = b["c"]
        while True:
            nc = c + b["dir"]
            if nc < 0 or nc >= g.shape[1]:
                break
            if any(nc == pc and not (b["r1"] < pr0 or b["r0"] > pr1) for pr0, pr1, pc, _ in placed):
                break
            c = nc
        placed.append((b["r0"], b["r1"], c, b))
    maxh = max(b["height"] for b in bars)
    minh = min(b["height"] for b in bars)
    for r0, r1, c, b in placed:
        color = 1 if b["height"] == maxh else 2 if b["height"] == minh else 5
        out[r0:r1 + 1, c] = color
    return out


def generate_common_4():
    h = w = 15
    g = np.zeros((h, w), dtype=int)
    n = random.randint(4, 6)
    heights = random.sample(range(3, 9), n)
    cols = sorted(random.sample([2, 4, 6, 8, 10, 12], n))
    dirs = [random.choice([-1, 1]) for _ in range(n)]
    dirs[0] = dirs[1]
    for ht, c, d in zip(heights, cols, dirs):
        r0 = h - ht
        g[r0:h, c] = 5
        mid = (r0 + h - 1) // 2
        g[mid, c + d] = 7
    return g


def main_common_5(grid):
    g = _arr(grid)
    out = g.copy()
    rects = []
    for col, cells in _components((g == 2).astype(int) * 2, [2], 4):
        r0, c0, r1, c1 = _bbox(cells)
        if r1 > r0 + 1 and c1 > c0 + 1:
            rects.append((r0, c0, r1, c1, cells))
    counts = []
    for r0, c0, r1, c1, cells in rects:
        vals = [int(g[r, c]) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if g[r, c] != 0]
        counts.append(Counter(vals))
    full = Counter()
    for cnt in counts:
        for k, v in cnt.items():
            full[k] = max(full[k], v)
    for rect, cnt in zip(rects, counts):
        r0, c0, r1, c1, cells = rect
        diff = full - cnt
        miss = next(iter(diff)) if diff else 1
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if out[r, c] == 0:
                    out[r, c] = miss
                    break
            else:
                continue
            break
        for r, c in cells:
            out[r, c] = miss
    return out


def generate_common_5():
    g = np.zeros((14, 14), dtype=int)
    poses = [(0, 0), (0, 8), (8, 0), (8, 8)]
    full = [1, 1, 3, 3, 4, 4, 6, 6, 7]
    missing = random.sample([1, 3, 4, 6], 4)
    for (r, c), miss in zip(poses, missing):
        g[r:r + 5, c] = g[r:r + 5, c + 4] = 2
        g[r, c:c + 5] = g[r + 4, c:c + 5] = 2
        vals = full.copy()
        vals.remove(miss)
        vals.append(0)
        random.shuffle(vals)
        k = 0
        for rr in range(r + 1, r + 4):
            for cc in range(c + 1, c + 4):
                g[rr, cc] = vals[k]
                k += 1
    return g


def _count_rooms(mask):
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            touches = False
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touches = True
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches:
                count += 1
    return count


def main_common_6(grid):
    g = _arr(grid)
    infos = []
    for col in sorted(set(int(x) for x in g.flatten()) - {0}):
        pts = [(r, c) for r in range(g.shape[0]) for c in range(g.shape[1]) if g[r, c] == col]
        if not pts:
            continue
        recon = np.zeros_like(g, dtype=bool)
        for r in range(g.shape[0]):
            cs = sorted(c for rr, c in pts if rr == r)
            start = 0
            while start < len(cs):
                end = start
                while end + 1 < len(cs) and cs[end + 1] == cs[end] + 1:
                    end += 1
                if end - start + 1 >= 3:
                    recon[r, cs[start]:cs[end] + 1] = True
                start = end + 1
        for c in range(g.shape[1]):
            rs = sorted(r for r, cc in pts if cc == c)
            start = 0
            while start < len(rs):
                end = start
                while end + 1 < len(rs) and rs[end + 1] == rs[end] + 1:
                    end += 1
                if end - start + 1 >= 3:
                    recon[rs[start]:rs[end] + 1, c] = True
                start = end + 1
        if not recon.any():
            continue
        rr, cc = np.where(recon)
        r0, r1, c0, c1 = rr.min(), rr.max(), cc.min(), cc.max()
        rooms = _count_rooms(recon[r0:r1 + 1, c0:c1 + 1])
        if rooms > 0:
            infos.append((min(r for r, c in pts), col, rooms))
    infos.sort()
    maxn = max(n for _, _, n in infos)
    minn = min(n for _, _, n in infos)
    out = np.zeros((maxn + 1, len(infos) * 2 - 1), dtype=int)
    for i, (_, col, n) in enumerate(infos):
        paint = 1 if n == maxn else 2 if n == minn else col
        out[-n:, i * 2] = paint
    return out


def generate_common_6():
    g = np.zeros((30, 30), dtype=int)
    n = random.randint(3, 4)
    positions = random.sample([(1, 1), (1, 16), (16, 1), (16, 16)], n)
    colors = random.sample([1, 3, 4, 6, 8], n)
    room_counts = random.sample(range(1, 6), n)
    specs = list(zip(positions, colors, room_counts))
    random.shuffle(specs)
    for (r, c), col, rooms in specs:
        width = rooms * 2 + 1
        height = random.randint(5, 7)
        g[r:r + height, c] = g[r:r + height, c + width - 1] = col
        g[r, c:c + width] = g[r + height - 1, c:c + width] = col
        for k in range(1, rooms):
            cc = c + k * 2
            g[r:r + height, cc] = col
    for _ in range(random.randint(4, 8)):
        rr, cc = random.randrange(30), random.randrange(30)
        if g[rr, cc] == 0:
            g[rr, cc] = 7
    return g


def main_common_7(grid):
    g = _arr(grid)
    markers = []
    for r in range(1, g.shape[0] - 1):
        for c in range(1, g.shape[1] - 1):
            if g[r, c] != 4:
                continue
            vals = [int(g[r - 1, c]), int(g[r + 1, c]), int(g[r, c - 1]), int(g[r, c + 1])]
            if vals[0] != 0 and vals[0] != 4 and vals.count(vals[0]) == 4:
                markers.append((vals[0], r, c))
    cnt = Counter(m[0] for m in markers)
    target = next(col for col, n in cnt.items() if n == 2)
    pts = [(r, c) for col, r, c in markers if col == target]
    r0, r1 = sorted([pts[0][0], pts[1][0]])
    c0, c1 = sorted([pts[0][1], pts[1][1]])
    out = np.zeros_like(g)
    out[r0:r1 + 1, c0:c1 + 1] = g[r0:r1 + 1, c0:c1 + 1]
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if out[r, c] == 0:
                out[r, c] = target
    return out


def generate_common_7():
    g = np.zeros((15, 15), dtype=int)
    centers = random.sample([(2, 2), (2, 7), (2, 12), (7, 2), (7, 12), (12, 7)], random.randint(4, 6))
    colors = [1, 2, 3, 6, 7]
    repeated = random.choice(colors)
    arm_colors = [repeated, repeated] + random.sample([c for c in colors if c != repeated], len(centers) - 2)
    random.shuffle(arm_colors)
    for (r, c), col in zip(centers, arm_colors):
        g[r, c] = 4
        g[r - 1, c] = g[r + 1, c] = g[r, c - 1] = g[r, c + 1] = col
    for _ in range(random.randint(4, 8)):
        rr, cc = random.randrange(15), random.randrange(15)
        if g[rr, cc] == 0 and all(not _in(g, rr + dr, cc + dc) or g[rr + dr, cc + dc] == 0 for dr, dc in DIR4):
            g[rr, cc] = random.choice([1, 2, 3, 6, 7])
    return g


def main_common_8(grid):
    g = _arr(grid)
    out = g.copy()
    sep_r = max(range(g.shape[0]), key=lambda r: np.count_nonzero(g[r] == 7))
    sep_c = max(range(g.shape[1]), key=lambda c: np.count_nonzero(g[:, c] == 7))
    quads = [(0, sep_r, 0, sep_c), (0, sep_r, sep_c + 1, g.shape[1]),
             (sep_r + 1, g.shape[0], 0, sep_c), (sep_r + 1, g.shape[0], sep_c + 1, g.shape[1])]
    current = []
    slots = []
    for r0, r1, c0, c1 in quads:
        sub = out[r0:r1, c0:c1]
        comps = _components((sub == 5).astype(int) * 5, [5], 4)
        cnt = Counter()
        for _, cells in comps:
            votes = Counter()
            first = {}
            for rr, cc in cells:
                ar, ac = r0 + rr, c0 + cc
                for dr, dc in DIR4:
                    nr, nc = ar + dr, ac + dc
                    if r0 <= nr < r1 and c0 <= nc < c1 and out[nr, nc] not in (0, 5, 7):
                        col = int(out[nr, nc])
                        votes[col] += 1
                        first.setdefault(col, (nr, nc))
            if votes:
                best = max(votes, key=lambda col: (votes[col], -first[col][0], -first[col][1]))
                for rr, cc in cells:
                    out[r0 + rr, c0 + cc] = best
                cnt[best] += 1
        current.append(cnt)
        slot = None
        for rr in range(r0, r1):
            for cc in range(c0, c1):
                if g[rr, cc] == 0:
                    slot = (rr, cc)
                    break
            if slot:
                break
        slots.append(slot)
    full = Counter()
    for cnt in current:
        for k, v in cnt.items():
            full[k] = max(full[k], v)
    for cnt, slot in zip(current, slots):
        diff = full - cnt
        if slot and diff:
            out[slot] = next(iter(diff))
    return out


def generate_common_8():
    g = np.zeros((15, 15), dtype=int)
    g[7, :] = 7
    g[:, 7] = 7
    quads = [(0, 7, 0, 7), (0, 7, 8, 15), (8, 15, 0, 7), (8, 15, 8, 15)]
    colors = [1, 2, 3, 4]
    missing = random.sample(colors, 4)
    for (r0, r1, c0, c1), miss in zip(quads, missing):
        present = [c for c in colors if c != miss]
        positions = [(r0 + 2, c0 + 2), (r0 + 4, c0 + 2), (r0 + 3, c0 + 5)]
        for (r, c), col in zip(positions, present):
            g[r, c] = g[r, c + 1] = 5
            g[r - 1, c] = col
            g[r + 1, c + 1] = col
    return g


def main_030_1(grid):
    g = _arr(grid)
    comps = [(cells, None) for col, cells in _components((g == 5).astype(int) * 5, [5], 4)]
    recolored = []
    for cells, _ in comps:
        votes = Counter()
        first = {}
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if _in(g, nr, nc) and g[nr, nc] not in (0, 5):
                    col = int(g[nr, nc])
                    votes[col] += 1
                    first.setdefault(col, (nr, nc))
        if votes:
            best = max(votes, key=lambda col: (votes[col], -first[col][0], -first[col][1]))
            recolored.append((best, cells))
    marker = np.where((g != 0) & (g != 5), g, 0)
    out = marker.copy()
    groups = defaultdict(list)
    for col, cells in recolored:
        groups[col].extend(cells)
    order = sorted(groups, key=lambda col: min(groups[col]))
    for col in order:
        cells = groups[col]
        r0, c0, r1, c1 = _bbox(cells)
        for r, c in cells:
            rr = r0 + r1 - r
            if out[rr, c] == 0:
                out[rr, c] = col
    return out


def generate_030_1():
    g = np.zeros((15, 15), dtype=int)
    positions = [(2, 2), (2, 9), (7, 3), (8, 10), (11, 6)]
    vote_cols = [1, 2, 1, 3, random.choice([1, 2, 3])]
    for (r, c), col in zip(positions, vote_cols):
        shape = random.choice([[(0, 0), (0, 1), (1, 0), (1, 1)], [(0, 0), (1, 0), (1, 1), (2, 1)]])
        for rr, cc in shape:
            g[r + rr, c + cc] = 5
        g[r - 1, c] = col
        g[r + 1, c + 2] = col
        g[r, c - 1] = random.choice([x for x in [1, 2, 3] if x != col])
    return g


def main_030_2(grid):
    g = _arr(grid)
    legend = [(r, int(g[r, g.shape[1] - 1])) for r in range(g.shape[0]) if g[r, g.shape[1] - 1] != 0]
    colors = [c for _, c in sorted(legend)]
    masks = {}
    for col in colors:
        gg = (g == col).astype(int)
        for r, _ in legend:
            if g[r, g.shape[1] - 1] == col:
                gg[r, g.shape[1] - 1] = 0
        comps = [(len(cells), cells) for _, cells in _components(gg * col, [col], 8)]
        comps = [x for x in comps if x[0] > 1]
        if not comps:
            masks[col] = np.zeros((1, 1), dtype=int)
        else:
            masks[col] = _mask_from_cells(max(comps, key=lambda x: x[0])[1])[0]
    n = len(colors)
    out = np.zeros((3 * n + 2, 3 * n + 2), dtype=int)
    for i, col in enumerate(colors):
        r = 3 + i * 3
        c = 3 + i * 3
        out[0:2, c:c + 2] = col
        out[r:r + 2, 0:2] = col
    for i, ci in enumerate(colors):
        for j, cj in enumerate(colors):
            if i == j:
                continue
            eq = any(t.shape == masks[cj].shape and np.array_equal(t, masks[cj]) for t in _transforms(masks[ci]))
            out[3 + i * 3:5 + i * 3, 3 + j * 3:5 + j * 3] = 2 if eq else 5
    return out


def generate_030_2():
    g = np.zeros((18, 18), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6], random.randint(4, 5))
    base = [(0, 1), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)]
    shapes = [base, [(c, 3 - r) for r, c in base], [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)],
              [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)], [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (3, 1)]]
    positions = [(1, 1), (1, 8), (8, 1), (8, 8), (12, 4)]
    for idx, col in enumerate(colors):
        r, c = positions[idx]
        for rr, cc in shapes[idx]:
            g[r + rr, c + cc] = col
        for _ in range(random.randint(2, 4)):
            for __ in range(50):
                nr, nc = random.randrange(17), random.randrange(17)
                if g[nr, nc] == 0 and all(not _in(g, nr + dr, nc + dc) or g[nr + dr, nc + dc] != col for dr, dc in DIR8):
                    g[nr, nc] = col
                    break
        g[1 + idx * 3, 17] = col
    return g
