import math
import random
from collections import Counter, defaultdict, deque

import numpy as np


BLACK, BLUE, RED, GREEN, YELLOW, GREY, MAGENTA, ORANGE, AZURE, MAROON = range(10)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _norm(cells):
    if not cells:
        return frozenset()
    r0, c0, _, _ = _bbox(cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _rots(cells):
    cur = set(cells)
    out = []
    for _ in range(4):
        out.append(_norm(cur))
        cur = {(c, -r) for r, c in cur}
    return out


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for sr, sc in zip(*np.where(mask & ~seen)):
        q = deque([(int(sr), int(sc))])
        seen[sr, sc] = True
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _draw_rect_border(g, r0, c0, r1, c1, color):
    g[r0, c0 : c1 + 1] = color
    g[r1, c0 : c1 + 1] = color
    g[r0 : r1 + 1, c0] = color
    g[r0 : r1 + 1, c1] = color


def _line_cells(a, b):
    r0, c0 = a
    r1, c1 = b
    dr = int(math.copysign(1, r1 - r0)) if r1 != r0 else 0
    dc = int(math.copysign(1, c1 - c0)) if c1 != c0 else 0
    n = max(abs(r1 - r0), abs(c1 - c0))
    return [(r0 + dr * i, c0 + dc * i) for i in range(n + 1)]


def _fill_poly(g, pts, color):
    rs = [p[0] for p in pts]
    cs = [p[1] for p in pts]
    for r in range(min(rs), max(rs) + 1):
        for c in range(min(cs), max(cs) + 1):
            inside = False
            j = len(pts) - 1
            y, x = r + 0.5, c + 0.5
            for i in range(len(pts)):
                yi, xi = pts[i][0], pts[i][1]
                yj, xj = pts[j][0], pts[j][1]
                if (yi > y) != (yj > y):
                    xint = (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi
                    if x < xint:
                        inside = not inside
                j = i
            if inside and g[r, c] == BLACK:
                g[r, c] = color


def _place(g, cells, top, left, color):
    for r, c in cells:
        if 0 <= top + r < g.shape[0] and 0 <= left + c < g.shape[1]:
            g[top + r, left + c] = color


def _main_from_color_band_frame(x):
    g = np.array(x, copy=True)
    bg = AZURE
    out = g.copy()
    frame = np.argwhere(g == MAROON)
    if frame.size == 0:
        return out
    fr0, fc0 = frame.min(axis=0)
    fr1, fc1 = frame.max(axis=0)
    perim = (
        [(fr0, c) for c in range(fc0, fc1 + 1)]
        + [(r, fc1) for r in range(fr0 + 1, fr1 + 1)]
        + [(fr1, c) for c in range(fc1 - 1, fc0 - 1, -1)]
        + [(r, fc0) for r in range(fr1 - 1, fr0, -1)]
    )
    blocks = []
    colors = [c for c in set(g.ravel()) if c not in (bg, RED, MAROON)]
    for color in colors:
        for comp in _components(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            if c1 < fc0 and len(comp) == (r1 - r0 + 1) * (c1 - c0 + 1):
                blocks.append((r0, c0, r1, c1, color))
    arrows = [_bbox(c) for c in _components(g == RED)]
    band = []
    for r0, c0, r1, c1, color in sorted(blocks):
        h, w = r1 - r0 + 1, c1 - c0 + 1
        best = None
        for ar0, ac0, ar1, ac1 in arrows:
            if ac1 < c0 and not (ar1 < r0 or ar0 > r1):
                best = ("left", (ar0 + ar1) // 2 - r0)
            elif ac0 > c1 and not (ar1 < r0 or ar0 > r1):
                best = ("right", (ar0 + ar1) // 2 - r0)
            elif ar1 < r0 and not (ac1 < c0 or ac0 > c1):
                best = ("top", (ac0 + ac1) // 2 - c0)
            elif ar0 > r1 and not (ac1 < c0 or ac0 > c1):
                best = ("bottom", (ac0 + ac1) // 2 - c0)
        if best is None:
            continue
        side, idx = best
        if side in ("left", "right"):
            idx = max(0, min(h - 1, idx))
            band.extend([int(v) for v in g[r0 + idx, c0 : c1 + 1]])
        else:
            idx = max(0, min(w - 1, idx))
            band.extend([int(v) for v in g[r0 : r1 + 1, c0 + idx]])
    for (r, c), color in zip(perim, band):
        out[r, c] = color
    return out


def p3_2_generate_input():
    h, w = random.randint(25, 29), random.randint(29, 33)
    g = np.full((h, w), AZURE, dtype=int)
    fr0, fc0 = 5, w - 13
    fr1, fc1 = h - 5, w - 3
    _draw_rect_border(g, fr0, fc0, fr1, fc1, MAROON)
    colors = random.sample([BLUE, GREEN, YELLOW, MAGENTA, ORANGE], 4)
    row = 2
    for i, color in enumerate(colors[: random.randint(3, 4)]):
        bh, bw = random.randint(2, 4), random.randint(2, 5)
        r0, c0 = row, random.randint(2, 5)
        g[r0 : r0 + bh, c0 : c0 + bw] = color
        side = random.choice(["left", "right", "top", "bottom"])
        if side in ("left", "right"):
            idx = random.randrange(bh)
            ar, ac = r0 + idx, c0 - 2 if side == "left" else c0 + bw
        else:
            idx = random.randrange(bw)
            ar, ac = r0 - 2 if side == "top" else r0 + bh, c0 + idx
        if 0 <= ar < h - 1 and 0 <= ac < w - 1:
            pts = [(ar, ac), (ar + 1, ac), (ar, ac + 1), (ar + 1, ac + 1)]
            for p in random.sample(pts, 3):
                g[p] = RED
        row += bh + 3
    return g


def p3_2_main(x):
    return _main_from_color_band_frame(x)


def _lock_key_main(x, marker_fill=False):
    g = np.array(x, copy=True)
    out = g.copy()
    h = g.shape[0]
    lock_colors = set()
    locks = []
    for color in set(g[: h // 2].ravel()) - {BLACK}:
        comps = _components(g == color)
        for comp in comps:
            r0, c0, r1, c1 = _bbox(comp)
            if r1 < h // 2 and (r1 - r0 >= 3 or c1 - c0 >= 3) and len(comp) >= 8:
                per = {(r0, c) for c in range(c0, c1 + 1)}
                per |= {(r1, c) for c in range(c0, c1 + 1)}
                per |= {(r, c0) for r in range(r0, r1 + 1)}
                per |= {(r, c1) for r in range(r0, r1 + 1)}
                missing = {p for p in per if g[p] == BLACK}
                if missing:
                    lock_colors.add(color)
                    locks.append({"bbox": (r0, c0, r1, c1), "color": color, "missing": missing})
    key_comps = []
    if marker_fill:
        for color in set(g[h // 2 :].ravel()) - ({BLACK} | lock_colors):
            for comp in _components(g == color):
                if min(r for r, _ in comp) >= h // 2 and 2 <= len(comp) <= 6:
                    key_comps.append((color, comp))
    else:
        for comp in _components(g == MAGENTA):
            if min(r for r, _ in comp) >= h // 2 and 2 <= len(comp) <= 6:
                key_comps.append((MAGENTA, comp))
    used = set()
    for _, key in key_comps:
        kshape = _norm(key)
        marker = None
        if marker_fill:
            kr0, kc0, kr1, kc1 = _bbox(key)
            near = []
            for color in lock_colors:
                for r, c in zip(*np.where(g == color)):
                    if r >= h // 2 and (abs(int(r) - kr0) + abs(int(c) - kc0) <= 3 or abs(int(r) - kr1) + abs(int(c) - kc1) <= 3):
                        near.append((int(r), int(c), color))
            marker = near[0][2] if near else next(iter(lock_colors), GREY)
        for i, lock in enumerate(locks):
            if i in used:
                continue
            if _norm(lock["missing"]) in _rots(kshape):
                fill = marker if marker_fill else lock["color"]
                for p in lock["missing"]:
                    out[p] = fill
                used.add(i)
                break
    for color, key in key_comps:
        for p in key:
            out[p] = BLACK
    if marker_fill:
        for color in lock_colors:
            for r, c in zip(*np.where(g == color)):
                if r >= h // 2:
                    out[int(r), int(c)] = BLACK
    return out


def p3_3_generate_input():
    g = np.zeros((28, 30), dtype=int)
    colors = [BLUE, RED, GREEN]
    key_colors = [MAGENTA, ORANGE, AZURE]
    specs = [
        ("top", {(0, 0), (0, 1)}),
        ("right", {(0, 0), (1, 0), (2, 0)}),
        ("bottom_l", {(0, 0), (0, 1), (1, 0)}),
    ]
    random.shuffle(specs)
    n = random.randint(2, 3)
    for i in range(n):
        r0, c0 = 2, 2 + i * 9
        r1, c1 = r0 + 6, c0 + 6
        color = colors[i]
        _draw_rect_border(g, r0, c0, r1, c1, color)
        kind, shape = specs[i]
        if kind == "top":
            miss = {(r0, c0 + 2), (r0, c0 + 3)}
        elif kind == "right":
            miss = {(r0 + 2, c1), (r0 + 3, c1), (r0 + 4, c1)}
        else:
            miss = {(r1, c0 + 2), (r1, c0 + 3), (r1 - 1, c0 + 2)}
        for p in miss:
            g[p] = BLACK
        kr, kc = 18, 2 + i * 8
        _place(g, _norm(miss), kr, kc, key_colors[i])
        g[kr - 2, kc] = random.choice(colors)
    return g


def p3_3_main(x):
    return _lock_key_main(x, marker_fill=True)


def p3_4_generate_input():
    h, w = 22, 24
    g = np.full((h, w), GREY, dtype=int)
    rows = random.sample(range(4, h - 4), 2)
    cols = random.sample(range(4, w - 4), 2)
    hcols = random.sample([BLUE, RED, GREEN, YELLOW], 2)
    vcols = random.sample([MAGENTA, ORANGE, AZURE, MAROON], 2)
    mixes = [c for c in range(1, 10) if c not in hcols + vcols + [GREY]]
    random.shuffle(mixes)
    corners = [(0, 0), (0, w - 2), (h - 2, 0), (h - 2, w - 2)]
    k = 0
    for a in hcols:
        for b in vcols:
            r, c = corners[k]
            g[r, c], g[r, c + 1], g[r + 1, c], g[r + 1, c + 1] = a, b, b, mixes[k % len(mixes)]
            k += 1
    for r, color in zip(rows, hcols):
        g[r, :] = color
    for c, color in zip(cols, vcols):
        g[:, c] = color
    return g


def p3_4_main(x):
    g = np.array(x, copy=True)
    out = g.copy()
    h, w = g.shape
    mix = {}
    for r, c in [(0, 0), (0, w - 2), (h - 2, 0), (h - 2, w - 2)]:
        a, b, b2, cc = int(g[r, c]), int(g[r, c + 1]), int(g[r + 1, c]), int(g[r + 1, c + 1])
        if b == b2:
            mix[frozenset((a, b))] = cc
    hpaths = []
    for r in range(h):
        vals = [int(v) for v in g[r] if v != GREY]
        if vals:
            color, count = Counter(vals).most_common(1)[0]
            if count > w // 2:
                hpaths.append((r, color))
    vpaths = []
    for c in range(w):
        vals = [int(v) for v in g[:, c] if v != GREY]
        if vals:
            color, count = Counter(vals).most_common(1)[0]
            if count > h // 2:
                vpaths.append((c, color))
    hrows = {r for r, _ in hpaths}
    vcols = {c for c, _ in vpaths}
    for r, hc in hpaths:
        for c, vc in vpaths:
            mc = mix.get(frozenset((hc, vc)))
            if mc is None:
                continue
            out[r, c] = mc
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and nr not in hrows and nc not in vcols:
                    out[nr, nc] = mc
    return out


_TEMPLATES = [
    {(0, 0)},
    {(0, 0), (0, 1)},
    {(0, 0), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (1, 0), (1, 1)},
]


def p3_5_generate_input():
    g = np.zeros((24, 30), dtype=int)
    rooms = [(2, 2, 6, 6), (2, 10, 6, 17), (9, 2, 15, 8)]
    colors = random.sample([RED, GREEN, YELLOW, MAGENTA, ORANGE], 3)
    for box, color in zip(random.sample(rooms, 3), colors):
        r0, c0, r1, c1 = box
        _draw_rect_border(g, r0, c0, r1, c1, BLUE)
        g[(r0 + r1) // 2, (c0 + c1) // 2] = color
    col = 2
    for shape in _TEMPLATES:
        _place(g, shape, 21, col, GREEN)
        col += max(c for _, c in shape) + 3
    return g


def p3_5_main(x):
    g = np.array(x, copy=True)
    out = g.copy()
    # Parse bottom template components before clearing the library.
    templates = []
    for comp in _components(g == GREEN):
        if min(r for r, _ in comp) >= g.shape[0] - 4:
            templates.append(_norm(comp))
    templates.sort(key=lambda s: (len(s), min((c for _, c in s), default=0)))
    if not templates:
        templates = [frozenset(t) for t in _TEMPLATES]
    out[g == GREEN] = BLACK
    passable = (g != BLUE) & (np.indices(g.shape)[0] < g.shape[0] - 4)
    regions = _components(passable)
    areas = []
    data = []
    for reg in regions:
        seeds = [(r, c, int(g[r, c])) for r, c in reg if int(g[r, c]) not in (BLACK, BLUE)]
        if seeds:
            area = len(reg)
            areas.append(area)
            data.append((reg, seeds[0], area))
    if not areas:
        return out
    base = max(1, min(areas))
    for _, (sr, sc, color), area in data:
        k = max(1, min(len(templates), round(area / base)))
        shape = templates[k - 1]
        br0, bc0, br1, bc1 = _bbox(shape)
        top = sr - (br1 - br0) // 2
        left = sc - (bc1 - bc0) // 2
        _place(out, shape, top, left, color)
        out[sr, sc] = color
    return out


def p3_6_generate_input():
    g = np.zeros((26, 30), dtype=int)
    # Legend cells: red=horizontal, blue=vertical, green=both.
    g[21:25, 20:29] = GREY
    g[22, 21], g[22, 24], g[22, 27] = RED, BLUE, GREEN
    shapes = [
        ([(0, 0), (2, 2), (4, 0)], RED),
        ([(0, 0), (0, 4), (2, 2)], BLUE),
        ([(0, 2), (2, 4), (4, 2), (2, 0)], GREEN),
    ]
    bases = random.sample([(2, 2), (2, 13), (12, 5), (11, 17)], 3)
    for (pts, _), (br, bc) in zip(shapes, bases):
        for r, c in pts:
            g[br + r, bc + c] = ORANGE
    return g


def p3_6_main(x):
    g = np.array(x, copy=True)
    out = g.copy()
    for comp in _components(g == ORANGE):
        if len(comp) < 3:
            continue
        cr = sum(r for r, _ in comp) / len(comp)
        cc = sum(c for _, c in comp) / len(comp)
        pts = sorted(comp, key=lambda p: math.atan2(p[1] - cc, cr - p[0]))
        rel = _norm(comp)
        rmax = max(r for r, _ in rel)
        cmax = max(c for _, c in rel)
        hs = rel == frozenset((rmax - r, c) for r, c in rel)
        vs = rel == frozenset((r, cmax - c) for r, c in rel)
        color = GREEN if hs and vs else RED if hs else BLUE
        for a, b in zip(pts, pts[1:] + pts[:1]):
            for p in _line_cells(a, b):
                out[p] = ORANGE
        _fill_poly(out, pts, color)
    return out


def _transform(cells, op):
    cells = set(cells)
    if op == "vflip":
        r0, c0, r1, c1 = _bbox(cells)
        cells = {(r1 - (r - r0), c) for r, c in cells}
    elif op == "hflip":
        r0, c0, r1, c1 = _bbox(cells)
        cells = {(r, c1 - (c - c0)) for r, c in cells}
    elif op == "cw":
        cells = {(c, -r) for r, c in cells}
    elif op == "ccw":
        cells = {(-c, r) for r, c in cells}
    elif op == "rot180":
        cells = {(-r, -c) for r, c in cells}
    return _norm(cells)


def p3_7_generate_input():
    g = np.full((24, 30), AZURE, dtype=int)
    anchor = (15, 16)
    g[anchor] = YELLOW
    colors = random.sample([BLUE, RED, GREEN, MAGENTA, ORANGE], 4)
    ops = ["id", "vflip", "hflip", "cw", "ccw", "rot180"]
    shapes = [{(0, 0), (1, 0), (1, 1)}, {(0, 1), (1, 0), (1, 1), (2, 1)}, {(0, 0), (0, 1), (1, 1)}]
    for i, color in enumerate(colors[:3]):
        _place(g, random.choice(shapes), 3 + i * 5, 2, color)
        br, bc = 2 + i * 6, 20
        _draw_rect_border(g, br, bc, br + 4, bc + 4, MAROON)
        op = random.choice(ops)
        rels = {"id": [(1, 1)], "vflip": [(1, 1), (0, 1)], "hflip": [(1, 1), (1, 2)], "cw": [(1, 1), (0, 2)], "ccw": [(1, 1), (0, 0)], "rot180": [(0, 0)]}[op]
        for rr, cc in rels:
            g[br + 1 + rr, bc + 1 + cc] = color
    return g


def p3_7_main(x):
    g = np.array(x, copy=True)
    out = np.full_like(g, AZURE)
    anchors = np.argwhere(g == YELLOW)
    if anchors.size == 0:
        return np.array(x, copy=True)
    ar, ac = map(int, anchors[0])
    box_bboxes = []
    for comp in _components(g == MAROON):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 == 4 and c1 - c0 == 4:
            box_bboxes.append((r0, c0, r1, c1))
    instruction_cells = set()
    instructions = []
    for box in sorted(box_bboxes):
        r0, c0, _, _ = box
        pts = [(r - (r0 + 1), c - (c0 + 1), int(g[r, c])) for r in range(r0 + 1, r0 + 4) for c in range(c0 + 1, c0 + 4) if g[r, c] not in (AZURE,)]
        for rr, cc, _ in pts:
            instruction_cells.add((r0 + 1 + rr, c0 + 1 + cc))
        colors = [p[2] for p in pts if p[2] != MAROON]
        if colors:
            color = Counter(colors).most_common(1)[0][0]
            rel = {(rr, cc) for rr, cc, col in pts if col == color}
            instructions.append((box, color, rel))
    sprite_comps = []
    for color in set(g.ravel()) - {AZURE, MAROON, YELLOW}:
        mask = (g == color)
        for comp in _components(mask):
            if not any(p in instruction_cells for p in comp):
                sprite_comps.append((color, comp))
    for idx, (_, color, rel) in enumerate(instructions):
        sprites = [c for ccol, c in sprite_comps if ccol == color]
        if not sprites:
            continue
        cells = _norm(sprites[0])
        if rel == {(1, 1), (0, 1)}:
            op = "vflip"
        elif rel == {(1, 1), (1, 2)}:
            op = "hflip"
        elif rel == {(1, 1), (0, 2)}:
            op = "cw"
        elif rel == {(1, 1), (0, 0)}:
            op = "ccw"
        elif (1, 1) not in rel:
            op = "rot180"
        else:
            op = "id"
        _place(out, _transform(cells, op), ar, ac, color)
    return out


def p4_1_generate_input():
    g = np.zeros((25, 25), dtype=int)
    r0, c0, r1, c1 = 4, 4, 20, 20
    _draw_rect_border(g, r0, c0, r1, c1, GREY)
    colors = random.sample([BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE], 4)
    bands = [("top", 6, 2), ("right", 8, 3), ("bottom", 10, 4), ("left", 7, 2)]
    for color, (side, start, length) in zip(colors, bands):
        if side == "top":
            g[r0, c0 + start : c0 + start + length] = color
        elif side == "bottom":
            g[r1, c0 + start : c0 + start + length] = color
        elif side == "left":
            g[r0 + start : r0 + start + length, c0] = color
        else:
            g[r0 + start : r0 + start + length, c1] = color
        for _ in range(random.randint(1, 4)):
            rr, cc = random.randint(r0 + 2, r1 - 2), random.randint(c0 + 2, c1 - 2)
            g[rr, cc] = color
    return g


def p4_1_main(x):
    g = np.array(x, copy=True)
    out = g.copy()
    bg = Counter(g.ravel()).most_common(1)[0][0]
    grey = GREY
    cells = np.argwhere(g == grey)
    if cells.size == 0:
        return out
    r0, c0 = cells.min(axis=0)
    r1, c1 = cells.max(axis=0)
    bands = {}
    for side, coords in [
        ("top", [(r0, c) for c in range(c0, c1 + 1)]),
        ("bottom", [(r1, c) for c in range(c0, c1 + 1)]),
        ("left", [(r, c0) for r in range(r0, r1 + 1)]),
        ("right", [(r, c1) for r in range(r0, r1 + 1)]),
    ]:
        for color, pts in defaultdict(list).items():
            pass
        cur = []
        last = None
        for p in coords:
            val = int(g[p])
            if val not in (bg, grey):
                cur.append(p)
                last = val
            elif cur:
                bands[last] = (side, cur[:])
                cur = []
        if cur:
            bands[last] = (side, cur[:])
    frags = []
    for color, (side, pts) in bands.items():
        for comp in _components((g == color)):
            if all(r0 < r < r1 and c0 < c < c1 for r, c in comp):
                frags.append((len(comp), color, side, pts))
                for p in comp:
                    out[p] = bg
    for length, color, side, pts in sorted(frags):
        if side in ("top", "bottom"):
            cols = [c for _, c in pts]
            rows = range(r0 + 1, min(r1, r0 + 1 + length)) if side == "top" else range(max(r0 + 1, r1 - length), r1)
            for rr in rows:
                for cc in cols:
                    out[rr, cc] = color
        else:
            rows = [r for r, _ in pts]
            cols = range(c0 + 1, min(c1, c0 + 1 + length)) if side == "left" else range(max(c0 + 1, c1 - length), c1)
            for rr in rows:
                for cc in cols:
                    out[rr, cc] = color
    return out


def p4_2_generate_input():
    g = np.zeros((28, 30), dtype=int)
    colors = random.sample([BLUE, RED, GREEN, YELLOW, ORANGE, AZURE], 3)
    for i, color in enumerate(colors):
        r0, c0 = random.randint(1, 3), 2 + i * 9
        side = random.randint(5, 7)
        r1, c1 = r0 + side, c0 + side
        _draw_rect_border(g, r0, c0, r1, c1, color)
        if i == 0:
            start = random.randint(2, side - 2)
            miss = {(r0, c0 + start), (r0, c0 + start + 1)}
        elif i == 1:
            start = random.randint(2, side - 3)
            miss = {(r0 + start, c1), (r0 + start + 1, c1), (r0 + start + 2, c1)}
        else:
            start = random.randint(2, side - 2)
            miss = {(r1, c0 + start), (r1, c0 + start + 1), (r1 - 1, c0 + start)}
        for p in miss:
            g[p] = BLACK
        _place(g, _norm(miss), random.randint(17, 21), 2 + i * 8, MAGENTA)
    return g


def p4_2_main(x):
    return _lock_key_main(x, marker_fill=False)


def _rect_rooms(seed_colors_per_room):
    g = np.zeros((26, 30), dtype=int)
    boxes = [(2, 2, 11, 10), (2, 15, 12, 26), (15, 4, 23, 14)]
    for box, cols in zip(boxes, seed_colors_per_room):
        r0, c0, r1, c1 = box
        _draw_rect_border(g, r0, c0, r1, c1, GREY)
        for color in cols:
            rr = random.randint(r0 + 2, r1 - 2)
            cc = random.randint(c0 + 2, c1 - 2)
            g[rr, cc] = color
    return g


def p4_3_generate_input():
    colors = random.sample([BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, AZURE, MAROON], 7)
    return _rect_rooms([colors[:2], colors[2:5], colors[5:7]])


def p4_3_main(x):
    g = np.array(x, copy=True)
    out = g.copy()
    passable = g != GREY
    for reg in _components(passable):
        seeds = [(r, c, int(g[r, c])) for r, c in reg if int(g[r, c]) not in (BLACK, GREY)]
        if not seeds:
            continue
        dist = {}
        q = deque()
        regset = set(reg)
        for r, c in reg:
            if any((r + dr, c + dc) not in regset for dr, dc in DIR4):
                dist[(r, c)] = 1
                q.append((r, c))
        while q:
            r, c = q.popleft()
            for dr, dc in DIR4:
                nb = (r + dr, c + dc)
                if nb in regset and nb not in dist:
                    dist[nb] = dist[(r, c)] + 1
                    q.append(nb)
        palette = [col for _, _, col in sorted(seeds, key=lambda s: (dist.get((s[0], s[1]), 99), s[0], s[1]))]
        for r, c in reg:
            d = dist.get((r, c), 99)
            if d <= len(palette):
                out[r, c] = palette[d - 1]
    return out


_ARROW = {
    "right": {(1, 0), (1, 1), (1, 2), (0, 2), (2, 2)},
    "left": {(1, 0), (0, 0), (2, 0), (1, 1), (1, 2)},
    "up": {(0, 1), (0, 0), (0, 2), (1, 1), (2, 1)},
    "down": {(2, 1), (2, 0), (2, 2), (1, 1), (0, 1)},
}


def p4_4_generate_input():
    g = np.zeros((28, 30), dtype=int)
    groups = [(BLUE, (6, 7), RED), (GREEN, (17, 18), YELLOW)]
    for color, center, target_color in groups:
        cr, cc = center
        g[cr, cc] = target_color
        g[cr, cc + 1] = target_color
        dirs = random.choices(["right", "left", "up", "down"], k=5)
        dirs[:3] = [random.choice(["right", "down"])] * 3
        locs = [(cr - 5, cc - 2), (cr + 3, cc - 2), (cr - 1, cc - 6), (cr - 1, cc + 5), (cr + 3, cc + 5)]
        for d, (r, c) in zip(dirs, locs):
            _place(g, _ARROW[d], r, c, color)
    return g


def _arrow_dir(cells):
    n = _norm(cells)
    for d, shape in _ARROW.items():
        if n == shape:
            return d
    return "right"


def p4_4_main(x):
    g = np.array(x, copy=True)
    out = g.copy()
    bg = Counter(g.ravel()).most_common(1)[0][0]
    arrow_colors = {c for c in set(g.ravel()) if c not in (bg,) and any(len(comp) == 5 for comp in _components(g == c))}
    for color in arrow_colors:
        comps = [comp for comp in _components(g == color) if len(comp) == 5]
        if not comps:
            continue
        direction = Counter(_arrow_dir(c) for c in comps).most_common(1)[0][0]
        cells = [p for comp in comps for p in comp]
        r0, c0, r1, c1 = _bbox(cells)
        r0, c0, r1, c1 = max(0, r0), max(0, c0), min(g.shape[0] - 1, r1), min(g.shape[1] - 1, c1)
        target = [(r, c, int(g[r, c])) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if g[r, c] not in (bg, color) and int(g[r, c]) not in arrow_colors]
        for p in cells:
            out[p] = bg
        if not target:
            continue
        tr0, tc0, tr1, tc1 = _bbox([(r, c) for r, c, _ in target])
        dr = (tr1 - tr0 + 1) if direction == "down" else -(tr1 - tr0 + 1) if direction == "up" else 0
        dc = (tc1 - tc0 + 1) if direction == "right" else -(tc1 - tc0 + 1) if direction == "left" else 0
        for r, c, val in target:
            out[r + dr, c + dc] = val
    return out


def p4_5_generate_input():
    g = np.zeros((27, 30), dtype=int)
    colors = random.sample([BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, AZURE], 3)
    specs = [
        ((random.randint(1, 3), 2, random.randint(7, 9), random.randint(8, 10)), colors[0], random.choice(["h", "v"]), random.randint(2, 4)),
        ((random.randint(1, 3), 15, random.randint(8, 10), random.randint(23, 26)), colors[1], random.choice(["h", "v"]), random.randint(2, 4)),
        ((random.randint(13, 15), 5, random.randint(21, 24), random.randint(12, 15)), colors[2], random.choice(["h", "v"]), random.randint(2, 4)),
    ]
    for box, color, orient, n in specs:
        r0, c0, r1, c1 = box
        if r1 - r0 < 5:
            r1 = r0 + 5
        if c1 - c0 < 5:
            c1 = c0 + 5
        _draw_rect_border(g, r0, c0, r1, c1, color)
        g[r0, random.randint(c0 + 1, c1 - 1)] = BLACK
        if orient == "h":
            rr, cc = min(g.shape[0] - 1, r1 + random.randint(2, 3)), c0
            for i in range(n):
                if cc + i < g.shape[1]:
                    g[rr, cc + i] = color
        else:
            rr, cc = min(g.shape[0] - n - 1, r1 + 2), min(g.shape[1] - 1, c1 + 3)
            for i in range(n):
                g[rr + i, cc] = color
    return g


def p4_5_main(x):
    g = np.array(x, copy=True)
    out = g.copy()
    bg = Counter(g.ravel()).most_common(1)[0][0]
    for color in set(g.ravel()) - {bg}:
        comps = _components(g == color)
        big = [c for c in comps if len(c) >= 10]
        small = [p for c in comps if len(c) < 10 for p in c]
        if not big or not small:
            continue
        r0, c0, r1, c1 = _bbox(big[0])
        per = {(r0, c) for c in range(c0, c1 + 1)} | {(r1, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0, r1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)}
        for p in per:
            if g[p] == bg:
                out[p] = color
        rows = {r for r, _ in small}
        cols = {c for _, c in small}
        n = len(small)
        if len(rows) == 1:
            width = c1 - c0 - 1
            start = c0 + 1
            for i in range(n):
                end = c0 + 1 + round(width * (i + 1) / n)
                if i % 2 == 0:
                    out[r0 + 1 : r1, start:end] = color
                start = end
        else:
            height = r1 - r0 - 1
            start = r0 + 1
            for i in range(n):
                end = r0 + 1 + round(height * (i + 1) / n)
                if i % 2 == 0:
                    out[start:end, c0 + 1 : c1] = color
                start = end
        for p in small:
            out[p] = bg
    return out


def p4_6_generate_input():
    g = np.zeros((28, 32), dtype=int)
    rooms = [((5, 4, 12, 12), "top"), ((8, 19, 16, 28), "right")]
    for box, side in rooms:
        r0, c0, r1, c1 = box
        _draw_rect_border(g, r0, c0, r1, c1, GREY)
        colors = random.sample([BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE], 3)
        pr, pc = r0 + 2, c0 + 2
        pat = [(0, 0, colors[0]), (0, 1, colors[1]), (1, 0, colors[0]), (1, 1, colors[2]), (2, 1, colors[0])]
        for rr, cc, col in pat:
            g[pr + rr, pc + cc] = col
        if side == "top":
            g[r0, pc] = colors[0]
        elif side == "right":
            g[pr, c1] = colors[0]
    return g


def _find_window_rooms(g):
    rooms = []
    for comp in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue
        per = [(r0, c) for c in range(c0, c1 + 1)] + [(r1, c) for c in range(c0, c1 + 1)] + [(r, c0) for r in range(r0, r1 + 1)] + [(r, c1) for r in range(r0, r1 + 1)]
        windows = [(r, c) for r, c in per if g[r, c] not in (BLACK, GREY)]
        if len(windows) != 1:
            continue
        wr, wc = windows[0]
        if np.any((g[r0 + 1 : r1, c0 + 1 : c1] != BLACK) & (g[r0 + 1 : r1, c0 + 1 : c1] != GREY)):
            rooms.append((r0, c0, r1, c1, wr, wc))
    return rooms


def p4_6_main(x):
    g = np.array(x, copy=True)
    out = g.copy()
    for r0, c0, r1, c1, wr, wc in _find_window_rooms(g):
        color = int(g[wr, wc])
        pattern = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if g[r, c] == color]
        if not pattern:
            continue
        pr0, pc0, pr1, pc1 = _bbox(pattern)
        stamp = [(r - pr0, c - pc0) for r, c in pattern]
        sh, sw = pr1 - pr0 + 1, pc1 - pc0 + 1
        out[wr, wc] = GREY
        if wr == r0:
            row = r0 - sh
            while row >= 0:
                for rr, cc in stamp:
                    if out[row + rr, pc0 + cc] != GREY:
                        out[row + rr, pc0 + cc] = color
                row -= sh
        elif wr == r1:
            row = r1 + 1
            while row + sh <= g.shape[0]:
                for rr, cc in stamp:
                    if out[row + rr, pc0 + cc] != GREY:
                        out[row + rr, pc0 + cc] = color
                row += sh
        elif wc == c0:
            col = c0 - sw
            while col >= 0:
                for rr, cc in stamp:
                    if out[pr0 + rr, col + cc] != GREY:
                        out[pr0 + rr, col + cc] = color
                col -= sw
        else:
            col = c1 + 1
            while col + sw <= g.shape[1]:
                for rr, cc in stamp:
                    if out[pr0 + rr, col + cc] != GREY:
                        out[pr0 + rr, col + cc] = color
                col += sw
    return out


def p5_1_generate_input():
    g = np.zeros((24, 30), dtype=int)
    boxes = [(2, 2, 10, 12, "v"), (3, 17, 13, 27, "h"), (14, 4, 22, 14, "v")]
    for r0, c0, r1, c1, orient in boxes:
        _draw_rect_border(g, r0, c0, r1, c1, GREY)
        color1, color2 = random.sample([BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE], 2)
        if orient == "v":
            mid = (c0 + c1) // 2
            g[r0:r1 + 1, mid] = GREY
            pts = [(r0 + 2, c0 + 2), (r0 + 3, c0 + 3), (r0 + 5, c0 + 2)]
            for r, c in pts[: random.randint(2, 3)]:
                g[r, c] = color1
            for r, c in random.sample(pts, random.randint(1, 3)):
                mc = mid + (mid - c)
                g[r, mc] = color2
        else:
            mid = (r0 + r1) // 2
            g[mid, c0:c1 + 1] = GREY
            pts = [(r0 + 2, c0 + 2), (r0 + 3, c0 + 4), (r0 + 2, c0 + 6)]
            for r, c in pts[: random.randint(2, 3)]:
                g[r, c] = color1
            for r, c in random.sample(pts, random.randint(1, 3)):
                mr = mid + (mid - r)
                g[mr, c] = color2
    return g


def p5_1_main(x):
    g = np.array(x, copy=True)
    out = g.copy()
    for comp in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue
        grey_rows = [r for r in range(r0 + 1, r1) if np.all(g[r, c0:c1 + 1] == GREY)]
        grey_cols = [c for c in range(c0 + 1, c1) if np.all(g[r0:r1 + 1, c] == GREY)]
        if grey_cols:
            mid = grey_cols[0]
            left_color = next((int(g[r, c]) for r in range(r0 + 1, r1) for c in range(c0 + 1, mid) if g[r, c] not in (BLACK, GREY)), BLUE)
            right_color = next((int(g[r, c]) for r in range(r0 + 1, r1) for c in range(mid + 1, c1) if g[r, c] not in (BLACK, GREY)), RED)
            for r in range(r0 + 1, r1):
                for c in range(c0 + 1, mid):
                    mc = mid + (mid - c)
                    if g[r, c] != BLACK and out[r, mc] == BLACK:
                        out[r, mc] = right_color
                    if g[r, mc] != BLACK and out[r, c] == BLACK:
                        out[r, c] = left_color
        if grey_rows:
            mid = grey_rows[0]
            top_color = next((int(g[r, c]) for r in range(r0 + 1, mid) for c in range(c0 + 1, c1) if g[r, c] not in (BLACK, GREY)), BLUE)
            bottom_color = next((int(g[r, c]) for r in range(mid + 1, r1) for c in range(c0 + 1, c1) if g[r, c] not in (BLACK, GREY)), RED)
            for r in range(r0 + 1, mid):
                for c in range(c0 + 1, c1):
                    mr = mid + (mid - r)
                    if g[r, c] != BLACK and out[mr, c] == BLACK:
                        out[mr, c] = bottom_color
                    if g[mr, c] != BLACK and out[r, c] == BLACK:
                        out[r, c] = top_color
    return out


def p5_2_generate_input():
    g = np.zeros((26, 30), dtype=int)
    g[3:18, 5:24] = GREY
    shapes = [{(0, 0), (0, 1), (1, 0)}, {(0, 0), (1, 0), (1, 1), (2, 1)}, {(0, 1), (1, 0), (1, 1), (1, 2)}]
    marks = random.sample([BLUE, RED, GREEN, YELLOW], 3)
    for i, (shape, mark) in enumerate(zip(shapes, marks)):
        top, left = 5 + i * 4, 8 + (i % 2) * 7
        for r, c in shape:
            g[top + r, left + c] = BLACK
        ar, ac = next(iter(shape))
        g[top + ar, left + ac] = mark
        _place(g, shape, 20, 3 + i * 8, MAROON)
    return g


def p5_2_main(x):
    g = np.array(x, copy=True)
    out = g.copy()
    grey_comps = _components(g == GREY)
    if not grey_comps:
        return out
    frame = max(grey_comps, key=len)
    r0, c0, r1, c1 = _bbox(frame)
    inside = np.zeros_like(g, dtype=bool)
    inside[r0 + 1 : r1, c0 + 1 : c1] = True
    slot_comps = _components(inside & (g != GREY))
    slots = []
    for comp in slot_comps:
        marker = [(r, c, int(g[r, c])) for r, c in comp if g[r, c] not in (BLACK, GREY)]
        if marker:
            slots.append((comp, marker[0]))
    pieces = [comp for comp in _components(g == MAROON) if min(r for r, _ in comp) > r1]
    for piece in pieces:
        pshape = _norm(piece)
        for slot, (mr, mc, mark) in slots:
            if _norm(slot) in _rots(pshape):
                for p in piece:
                    out[p] = BLACK
                for r, c in slot:
                    out[r, c] = mark if r == mr or c == mc else MAROON
                break
    return out


def p5_3_generate_input():
    h, w = random.randint(22, 27), random.randint(26, 32)
    g = np.zeros((h, w), dtype=int)
    center = random.randint(11, w - 12)
    junction = random.randint(6, 10)
    start = (h - 4, center)
    g[start] = YELLOW
    g[junction:start[0], center] = GREY
    widths = random.choice([(1, 2), (2, 1), (2, 2)])
    left_end = random.randint(3, center - 5)
    right_end = random.randint(center + 5, w - 4)
    # Left branch.
    for off in range(widths[0]):
        g[junction + off, left_end : center + 1] = GREY
    g[junction, left_end - 1] = BLUE
    # Right branch.
    for off in range(widths[1]):
        g[junction + off, center:right_end + 1] = GREY
    g[junction, right_end + 1] = BLUE
    return g


def _terminal_width(g, term):
    r, c = term
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and g[nr, nc] == GREY:
            if dr == 0:
                return 1 + int(nr + 1 < g.shape[0] and g[nr + 1, nc] == GREY) + int(nr - 1 >= 0 and g[nr - 1, nc] == GREY)
            return 1 + int(nc + 1 < g.shape[1] and g[nr, nc + 1] == GREY) + int(nc - 1 >= 0 and g[nr, nc - 1] == GREY)
    return 1


def _shortest_path(g, start, target):
    q = deque([start])
    prev = {start: None}
    while q:
        p = q.popleft()
        if p == target:
            break
        r, c = p
        for dr, dc in DIR4:
            nb = (r + dr, c + dc)
            if 0 <= nb[0] < g.shape[0] and 0 <= nb[1] < g.shape[1] and nb not in prev and g[nb] in (GREY, YELLOW, BLUE):
                prev[nb] = p
                q.append(nb)
    if target not in prev:
        return []
    path = []
    p = target
    while p is not None:
        path.append(p)
        p = prev[p]
    return path


def p5_3_main(x):
    g = np.array(x, copy=True)
    out = g.copy()
    starts = np.argwhere(g == YELLOW)
    if starts.size == 0:
        return out
    start = tuple(map(int, starts[0]))
    terms = [tuple(map(int, p)) for p in np.argwhere(g == BLUE)]
    if not terms:
        return out
    widths = {t: _terminal_width(g, t) for t in terms}
    best = max(widths.values())
    for term, width in widths.items():
        if width == best:
            for p in _shortest_path(g, start, term):
                if out[p] == GREY:
                    out[p] = ORANGE
            out[term] = RED
    return out


def p5_4_generate_input():
    g = np.zeros((24, 30), dtype=int)
    boxes = [(2, 2, 10, 10), (2, 15, 12, 26), (14, 5, 22, 15)]
    colors = random.sample([BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE], 3)
    for box, color in zip(boxes, colors):
        r0, c0, r1, c1 = box
        _draw_rect_border(g, r0, c0, r1, c1, GREY)
        for _ in range(random.randint(1, 4)):
            g[random.randint(r0 + 1, r1 - 1), random.randint(c0 + 1, c1 - 1)] = color
    return g


def p5_4_main(x):
    g = np.array(x, copy=True)
    out = g.copy()
    for comp in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        inside = g[r0 + 1 : r1, c0 + 1 : c1]
        vals = [int(v) for v in inside.ravel() if v not in (BLACK, GREY)]
        if not vals:
            continue
        color = Counter(vals).most_common(1)[0][0]
        n = len(vals)
        for k in range(n):
            rr0, cc0 = r0 + 1 + k, c0 + 1 + k
            rr1, cc1 = r1 - 1 - k, c1 - 1 - k
            if rr0 > rr1 or cc0 > cc1:
                break
            _draw_rect_border(out, rr0, cc0, rr1, cc1, color)
    return out


def p5_5_generate_input():
    g = np.zeros((24, 32), dtype=int)
    div = random.randint(14, 17)
    g[:, div] = GREY
    rows = random.sample(range(4, 19), 3)
    for row in rows:
        length = random.randint(2, 5)
        colors = random.sample([BLUE, GREEN, YELLOW, MAGENTA, ORANGE, MAROON], 3)
        tip = div - 2
        g[row, tip] = RED
        g[row - 1 : row + 2, tip - 1] = RED
        for i, col in enumerate(colors):
            g[row - 1 + i, tip - 1 - length : tip - 1] = col
    return g


def p5_5_main(x):
    g = np.array(x, copy=True)
    out = g.copy()
    div_cols = [c for c in range(g.shape[1]) if np.count_nonzero(g[:, c] == GREY) > g.shape[0] // 2]
    if not div_cols:
        return out
    div = div_cols[0]
    arrows = []
    for r in range(1, g.shape[0] - 1):
        for c in range(1, div):
            if g[r, c] == RED and np.all(g[r - 1 : r + 2, c - 1] == RED):
                length = 0
                cc = c - 2
                while cc >= 0 and all(g[r - 1 + i, cc] not in (BLACK, GREY, RED) for i in range(3)):
                    length += 1
                    cc -= 1
                if length:
                    cols = [int(g[r - 1 + i, c - 2]) for i in range(3)]
                    arrows.append((length, r, cols))
    for length, r, cols in sorted(arrows):
        for i, color in enumerate(cols):
            out[r - 1 + i, div + 1 :] = color
    return out


def p5_6_generate_input():
    h, w = random.randint(24, 29), random.randint(28, 34)
    g = np.zeros((h, w), dtype=int)
    colors = random.sample([BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, AZURE], 4)
    r0, c0 = random.randint(1, 3), random.randint(1, 3)
    r1, c1 = h - random.randint(3, 4), w - random.randint(3, 4)
    levels = random.randint(3, 4)
    for color in colors[:levels]:
        if r1 - r0 < 4 or c1 - c0 < 4:
            break
        _draw_rect_border(g, r0, c0, r1, c1, color)
        r0 += random.randint(2, 3)
        c0 += random.randint(2, 4)
        r1 -= random.randint(2, 3)
        c1 -= random.randint(2, 4)
    return g


def p5_6_main(x):
    g = np.array(x, copy=True)
    out = g.copy()
    rects = []
    for color in set(g.ravel()) - {BLACK}:
        for comp in _components(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 >= 3 and c1 - c0 >= 3:
                rects.append((r0, c0, r1, c1, int(color)))
    rects.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    for rect in rects:
        r0, c0, r1, c1, color = rect
        depth = 1 + sum(or0 < r0 and oc0 < c0 and or1 > r1 and oc1 > c1 for or0, oc0, or1, oc1, _ in rects)
        inner_rects = [(a, b, c, d) for a, b, c, d, _ in rects if r0 < a and c < r1 and c0 < b and d < c1]
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if any(a <= r <= c_ and b <= c <= d for a, b, c_, d in inner_rects):
                    continue
                if (depth % 2 == 1 and (r - r0 - 1) % 2 == 0) or (depth % 2 == 0 and (c - c0 - 1) % 2 == 0):
                    out[r, c] = color
    return out


def p5_7_generate_input():
    g = np.zeros((24, 30), dtype=int)
    colors = random.sample([BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE], 4)
    locs = [(2, 2), (3, 16), (13, 4), (14, 18)]
    for color, (r, c) in zip(colors, locs):
        hh, ww = random.randint(3, 5), random.randint(3, 6)
        g[r : r + hh, c : c + ww] = color
        g[r + 1, c + ww] = GREEN
        if ww > 3:
            g[r + hh - 1, c + 1] = GREY
    return g


def p5_7_main(x):
    g = np.array(x, copy=True)
    pieces = []
    for color in set(g.ravel()) - {BLACK, GREEN, GREY}:
        for comp in _components(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            mask = [(r - r0, c - c0) for r, c in comp]
            pieces.append((len(comp), r0, c0, int(color), mask, r1 - r0 + 1, c1 - c0 + 1))
    if not pieces:
        return g.copy()
    pieces.sort(key=lambda x: (x[1], x[2]))
    height = max(p[5] for p in pieces)
    width = sum(p[6] for p in pieces)
    out = np.zeros((max(height, 1), max(width, 1)), dtype=int)
    col = 0
    placed = []
    for area, _, _, color, mask, hh, ww in pieces:
        for r, c in mask:
            out[r, col + c] = color
        placed.append((area, color))
        col += ww
    comps = []
    for color in set(out.ravel()) - {BLACK}:
        for comp in _components(out == color):
            comps.append((len(comp), min(comp), int(color), comp))
    comps.sort(key=lambda x: (-x[0], x[1]))
    palette = [None, RED, BLUE, YELLOW, GREEN, ORANGE]
    for i, (_, _, color, comp) in enumerate(comps):
        new = color if i == 0 else palette[min(i, len(palette) - 1)]
        for p in comp:
            out[p] = new
    return out
