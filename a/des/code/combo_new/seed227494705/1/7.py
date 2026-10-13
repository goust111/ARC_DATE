
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
    ans = []
    for col in sorted(set(int(x) for x in np.unique(grid)) - set(exclude)):
        for cells in _components(grid == col):
            ans.append({"color": col, "cells": cells, "bbox": _bbox(cells)})
    return ans


def _draw_frame(g, r0, c0, h, w, color):
    g[r0, c0:c0 + w] = color
    g[r0 + h - 1, c0:c0 + w] = color
    g[r0:r0 + h, c0] = color
    g[r0:r0 + h, c0 + w - 1] = color


def _is_frame(cells):
    r0, c0, r1, c1 = _bbox(cells)
    if r1 - r0 < 3 or c1 - c0 < 3:
        return False
    border = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    s = set(cells)
    return len(s - border) == 0 and len(border - s) <= 2 and len(s) >= len(border) - 2


def _put(g, mask, r0, c0, color):
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if mask[r, c]:
                g[r0 + r, c0 + c] = color


def _place_free(g, h, w, pad=1, tries=300):
    rows, cols = g.shape
    for _ in range(tries):
        r = random.randint(pad, rows - h - pad)
        c = random.randint(pad, cols - w - pad)
        if np.all(g[max(0, r-pad):min(rows, r+h+pad), max(0, c-pad):min(cols, c+w+pad)] == BG):
            return r, c
    return None


def main_1(grid):
    g = np.array(grid, copy=True)
    bg = _bg(g)
    comps = _components_by_color(g, exclude={bg})
    frames = [c for c in comps if _is_frame(c["cells"])]
    frame_cells = {p for fr in frames for p in fr["cells"]}
    for fr in frames:
        color = fr["color"]
        r0, c0, r1, c1 = fr["bbox"]
        inside = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        indicators = [(r, c) for r, c in inside if g[r, c] not in (bg, color)]
        if not indicators:
            continue
        ar = sum(r for r, _ in indicators) / len(indicators)
        ac = sum(c for _, c in indicators) / len(indicators)
        dists = {"up": ar - (r0 + 1), "down": (r1 - 1) - ar, "left": ac - (c0 + 1), "right": (c1 - 1) - ac}
        direction = min(dists, key=dists.get)
        for r, c in indicators:
            g[r, c] = bg
        sprites = [c for c in comps if c["color"] == color and not any(p in frame_cells for p in c["cells"])]
        if not sprites:
            continue
        sp = max(sprites, key=lambda x: len(x["cells"]))
        for r, c in sp["cells"]:
            g[r, c] = bg
        m = _mask(sp["cells"])
        k = {"up": 0, "right": 3, "down": 2, "left": 1}[direction]
        rm = np.rot90(m, k)
        top = r0 + 1 + max(0, ((r1 - r0 - 1) - rm.shape[0]) // 2)
        left = c0 + 1 + max(0, ((c1 - c0 - 1) - rm.shape[1]) // 2)
        _put(g, rm, top, left, color)
    return g


def generate_1():
    g = np.zeros((24, 24), dtype=int)
    colors = random.sample([1, 2, 3, 4, 7, 9], random.randint(2, 3))
    frame_slots = [(2, 2), (2, 13), (12, 3)]
    sprite_slots = [(18, 2), (18, 10), (18, 17)]
    masks = [
        np.array([[1, 1], [1, 0]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.ones((2, 2), dtype=bool),
        np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    ]
    for col, (fr, fc), (sr, sc) in zip(colors, frame_slots, sprite_slots):
        h = w = 6
        _draw_frame(g, fr, fc, h, w, col)
        direction = random.choice(["up", "down", "left", "right"])
        gap_side = {"up": "down", "down": "up", "left": "right", "right": "left"}[direction]
        if gap_side == "up":
            g[fr, fc + w // 2] = BG
        elif gap_side == "down":
            g[fr + h - 1, fc + w // 2] = BG
        elif gap_side == "left":
            g[fr + h // 2, fc] = BG
        else:
            g[fr + h // 2, fc + w - 1] = BG
        ind_col = AZURE if col != AZURE else MAGENTA
        if direction == "up":
            g[fr + 1, fc + 2:fc + 4] = ind_col
        elif direction == "down":
            g[fr + h - 2, fc + 2:fc + 4] = ind_col
        elif direction == "left":
            g[fr + 2:fr + 4, fc + 1] = ind_col
        else:
            g[fr + 2:fr + 4, fc + w - 2] = ind_col
        _put(g, random.choice(masks), sr, sc, col)
    return g


def _rect_components(g, bg):
    rects = []
    for comp in _components_by_color(g, exclude={bg, GREY, ORANGE}):
        r0, c0, r1, c1 = comp["bbox"]
        if len(comp["cells"]) == (r1 - r0 + 1) * (c1 - c0 + 1):
            rects.append(comp)
    return rects


def main_2(grid):
    g = np.array(grid, copy=True)
    bg = _bg(g)
    rects = _rect_components(g, bg)
    label = {}
    for i, rect in enumerate(rects):
        for p in rect["cells"]:
            label[p] = i
    assignments = []
    for line in _components((g == GREY) | (g == ORANGE)):
        touches = set()
        for r, c in line:
            for dr, dc in DIR4:
                p = (r + dr, c + dc)
                if p in label:
                    touches.add(label[p])
        if len(touches) != 2:
            continue
        a, b = list(touches)
        k = sum(1 for r, c in line if g[r, c] == ORANGE)
        if k == 0:
            continue
        ra, rb = rects[a], rects[b]
        if len(ra["cells"]) == len(rb["cells"]):
            continue
        small, large = (ra, rb) if len(ra["cells"]) < len(rb["cells"]) else (rb, ra)
        if k == 1:
            assignments.append((1, small, large["color"]))
        elif k >= 2:
            assignments.append((2, large, small["color"]))
    for _, target, color in sorted(assignments, key=lambda x: x[0]):
        for r, c in target["cells"]:
            g[r, c] = color
    return g


def _draw_line_with_markers(g, a, b, k):
    cells = []
    if a[0] == b[0]:
        r = a[0]
        for c in range(min(a[1], b[1]), max(a[1], b[1]) + 1):
            cells.append((r, c))
    else:
        c = a[1]
        for r in range(min(a[0], b[0]), max(a[0], b[0]) + 1):
            cells.append((r, c))
    for r, c in cells:
        g[r, c] = GREY
    if k:
        picks = cells[len(cells)//3:len(cells)//3 + k]
        for r, c in picks:
            g[r, c] = ORANGE


def generate_2():
    g = np.full((22, 22), AZURE, dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 9], 3)
    rect_specs = [(3, 2, random.randint(3, 5), random.randint(3, 4)), (3, 13, random.randint(2, 4), random.randint(2, 3)), (13, 13, random.randint(3, 5), random.randint(2, 4))]
    rects = []
    for col, (r, c, h, w) in zip(colors, rect_specs):
        g[r:r+h, c:c+w] = col
        rects.append((r, c, h, w, col))
    a = (rects[0][0] + rects[0][2] // 2, rects[0][1] + rects[0][3])
    b = (rects[1][0] + rects[1][2] // 2, rects[1][1] - 1)
    _draw_line_with_markers(g, a, b, random.choice([0, 1, 2]))
    a = (rects[1][0] + rects[1][2], rects[1][1] + rects[1][3] // 2)
    b = (rects[2][0] - 1, rects[2][1] + rects[2][3] // 2)
    _draw_line_with_markers(g, a, b, random.choice([0, 1, 2]))
    return g


def main_3(grid):
    g = np.array(grid, copy=True)
    bg = _bg(g)
    frames = [c for c in _components_by_color(g, exclude={bg}) if _is_frame(c["cells"])]
    frame_cells = {p for fr in frames for p in fr["cells"]}
    for fr in frames:
        color = fr["color"]
        r0, c0, r1, c1 = fr["bbox"]
        center = ((r0 + r1) / 2.0, (c0 + c1) / 2.0)
        for comp in _components_by_color(g, exclude={bg}):
            if comp["color"] != color or any(p in frame_cells for p in comp["cells"]):
                continue
            for r, c in comp["cells"]:
                qr = 1 if r > center[0] else 0
                qc = 1 if c > center[1] else 0
                rr = r0 + 1 if qr == 0 else r1 - 1
                cc = c0 + 1 if qc == 0 else c1 - 1
                g[rr, cc] = color
                g[r, c] = bg
    return g


def generate_3():
    g = np.zeros((24, 24), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 4))
    slots = [(2, 2), (2, 14), (13, 2), (13, 14)]
    for col, (r, c) in zip(colors, slots):
        _draw_frame(g, r, c, 5, 5, col)
        quads = random.sample([(0, 0), (0, 1), (1, 0), (1, 1)], random.randint(1, 4))
        for qr, qc in quads:
            rr = random.randint(0, r - 1) if qr == 0 and r > 1 else random.randint(r + 6, 23)
            cc = random.randint(0, c - 1) if qc == 0 and c > 1 else random.randint(c + 6, 23)
            rr = max(0, min(23, rr)); cc = max(0, min(23, cc))
            if g[rr, cc] == BG:
                g[rr, cc] = col
    return g


def main_4(grid):
    g = np.array(grid, copy=True)
    bg = _bg(g)
    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    if not sep_cols:
        return g
    sep = sep_cols[0]
    left = g[:, :sep].copy()
    seed_comps = _components_by_color(left, exclude={bg})
    if not seed_comps:
        return left
    seed = max(seed_comps, key=lambda x: len(x["cells"]))
    mask = _mask(seed["cells"])
    sr0, sc0, _, _ = seed["bbox"]
    right = g[:, sep + 1:]
    oranges = int(np.sum(right == ORANGE))
    y = list(zip(*np.where(right == YELLOW)))
    m = list(zip(*np.where(right == MAROON))) + y
    if not y or not m:
        return left
    mr = sum(r for r, _ in m) / len(m); mc = sum(c for _, c in m) / len(m)
    yr, yc = y[0]
    if abs(yr - mr) >= abs(yc - mc):
        direction = "down" if yr > mr else "up"
    else:
        direction = "right" if yc > mc else "left"
    copies = []
    step_r = mask.shape[0] if direction in ("up", "down") else 0
    step_c = mask.shape[1] if direction in ("left", "right") else 0
    for i in range(oranges + 1):
        dr = i * step_r * (1 if direction == "down" else -1 if direction == "up" else 0)
        dc = i * step_c * (1 if direction == "right" else -1 if direction == "left" else 0)
        copies.append((sr0 + dr, sc0 + dc))
    minr = min(r for r, _ in copies)
    minc = min(c for _, c in copies)
    off_r = max(0, -minr)
    off_c = max(0, -minc)
    out_h = max(left.shape[0] + off_r, max(r + off_r + mask.shape[0] for r, _ in copies))
    out_w = max(left.shape[1] + off_c, max(c + off_c + mask.shape[1] for _, c in copies))
    out = np.full((out_h, out_w), bg, dtype=int)
    out[off_r:off_r + left.shape[0], off_c:off_c + left.shape[1]] = left
    for r, c in seed["cells"]:
        out[r + off_r, c + off_c] = bg
    for r, c in copies:
        _put(out, mask, r + off_r, c + off_c, seed["color"])
    return out


def generate_4():
    left_h, left_w = 9, 9
    right_w = 12
    g = np.zeros((left_h, left_w + 1 + right_w), dtype=int)
    g[:, left_w] = GREY
    seed_col = random.choice([1, 2, 3, 4, 6, 8, 9])
    mask = random.choice([
        np.array([[1, 1], [1, 0]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.ones((2, 2), dtype=bool),
    ])
    _put(g, mask, 3, 3, seed_col)
    cr, cc = 4, left_w + 6
    for dr, dc in DIR4:
        g[cr + dr, cc + dc] = MAROON
    g[cr, cc] = MAROON
    direction = random.choice(["up", "down", "left", "right"])
    tip = {"up": (cr - 1, cc), "down": (cr + 1, cc), "left": (cr, cc - 1), "right": (cr, cc + 1)}[direction]
    g[tip] = YELLOW
    for i in range(random.randint(1, 4)):
        r = random.randint(1, left_h - 2)
        c = random.randint(left_w + 1, left_w + right_w - 1)
        if g[r, c] == BG:
            g[r, c] = ORANGE
    return g


def main_5(grid):
    g = np.array(grid, copy=True)
    bg = _bg(g)
    maroon = MAROON
    interiors = [c for c in _components(g == maroon)]
    for cells in interiors:
        r0, c0, r1, c1 = _bbox(cells)
        fr = (r0 - 1, c0 - 1, r1 + 1, c1 + 1)
        lock = []
        for r in range(fr[0], fr[2] + 1):
            for c in range(fr[1], fr[3] + 1):
                if 0 <= r < g.shape[0] and 0 <= c < g.shape[1] and (r in (fr[0], fr[2]) or c in (fr[1], fr[3])) and g[r, c] not in (bg, GREY, maroon):
                    lock.append((r, c))
        if not lock:
            continue
        lcol = int(g[lock[0]])
        lmask = _norm(lock)
        _, lc0, _, lc1 = _bbox(lock)
        mirror = frozenset((r - min(x for x, _ in lock), (lc1 - c)) for r, c in lock)
        for comp in _components_by_color(g, exclude={bg, GREY, maroon}):
            if comp["color"] != lcol or any(fr[0] <= r <= fr[2] and fr[1] <= c <= fr[3] for r, c in comp["cells"]):
                continue
            if _norm(comp["cells"]) != mirror:
                continue
            hint = None
            for r, c in comp["cells"]:
                if r + 1 < g.shape[0] and g[r + 1, c] not in (bg, GREY, maroon, lcol):
                    hint = (r + 1, c, int(g[r + 1, c]))
                    break
            if hint is None:
                continue
            for r, c in cells:
                g[r, c] = hint[2]
            for r, c in comp["cells"]:
                g[r, c] = bg
            g[hint[0], hint[1]] = bg
            break
    return g


def generate_5():
    g = np.zeros((25, 25), dtype=int)
    locks = [np.array([[1, 0], [1, 1]], dtype=bool), np.ones((1, 3), dtype=bool), np.array([[1, 1], [0, 1]], dtype=bool)]
    colors = random.sample([1, 2, 3, 4, 6, 7, 8], random.randint(2, 3))
    slots = [(2, 2), (2, 14), (13, 4)]
    key_slots = [(18, 2), (18, 10), (18, 17)]
    hints = [9, 4, 8, 6, 3]
    for col, (r, c), (kr, kc) in zip(colors, slots, key_slots):
        h, w = 6, 7
        _draw_frame(g, r, c, h, w, GREY)
        g[r + 1:r + h - 1, c + 1:c + w - 1] = MAROON
        m = random.choice(locks)
        _put(g, m, r, c + 2, col)
        km = np.fliplr(m)
        _put(g, km, kr, kc, col)
        g[kr + km.shape[0], kc] = random.choice([x for x in hints if x != col])
    return g


def _find_blocks(g, color):
    out = []
    for r in range(g.shape[0] - 1):
        for c in range(g.shape[1] - 1):
            if np.all(g[r:r+2, c:c+2] == color):
                out.append((r, c))
    return out


def main_6(grid):
    g = np.array(grid, copy=True)
    bg = _bg(g)
    legend = []
    for r in range(g.shape[0]):
        if g[r, 0] != bg:
            legend.append(int(g[r, 0]))
        elif legend:
            break
    if not legend:
        return g
    paths = {}
    for col in legend:
        blocks = _find_blocks(g, col)
        if len(blocks) < 2:
            continue
        a, b = blocks[0], blocks[-1]
        if abs(a[0] - b[0]) <= 1:
            row = a[0]
            paths[col] = [(row, c) for c in range(min(a[1], b[1]), max(a[1], b[1]) + 2)]
        elif abs(a[1] - b[1]) <= 1:
            cc = a[1]
            paths[col] = [(r, cc) for r in range(min(a[0], b[0]), max(a[0], b[0]) + 2)]
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            hits = [col for col, cells in paths.items() if (r, c) in cells]
            if len(hits) > 1:
                hits.sort(key=lambda x: legend.index(x))
                g[r, c] = hits[0]
            elif len(hits) == 1 and g[r, c] == bg:
                g[r, c] = hits[0]
    return g


def generate_6():
    g = np.zeros((22, 22), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 4))
    for i, col in enumerate(colors):
        g[i, 0] = col
    rows = random.sample([4, 8, 12, 16], len(colors))
    cols = random.sample([5, 9, 13, 17], len(colors))
    for i, col in enumerate(colors):
        if i % 2 == 0:
            r = rows[i]
            g[r:r+2, 2:4] = col
            g[r:r+2, 18:20] = col
            g[r, 4:18] = col
        else:
            c = cols[i]
            g[2:4, c:c+2] = col
            g[18:20, c:c+2] = col
            g[4:18, c] = col
    return g


def main_7(grid):
    g = np.array(grid, copy=True)
    bg = _bg(g)
    barrier = g == GREY
    labels = -np.ones(g.shape, dtype=int)
    regions = []
    for i, cells in enumerate(_components(~barrier)):
        for r, c in cells:
            labels[r, c] = i
        votes = [int(g[r, c]) for r, c in cells if g[r, c] not in (bg, GREY)]
        maj = Counter(votes).most_common(1)[0][0] if votes else bg
        regions.append({"cells": cells, "maj": maj})
    out = np.full_like(g, bg)
    out[barrier] = GREY
    for r, c in zip(*np.where(g == GREY)):
        neigh = set()
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and labels[rr, cc] >= 0:
                neigh.add(labels[rr, cc])
        if len(neigh) >= 2:
            majs = [regions[i]["maj"] for i in neigh]
            for col, cnt in Counter(majs).items():
                if col != bg and cnt >= 2:
                    out[r, c] = col
                    break
    return out


def generate_7():
    rows = random.choice([2, 3])
    cols = random.choice([2, 3])
    cell = random.choice([5, 6])
    h = rows * cell + rows + 1
    w = cols * cell + cols + 1
    g = np.zeros((h, w), dtype=int)
    for r in range(0, h, cell + 1):
        g[r, :] = GREY
    for c in range(0, w, cell + 1):
        g[:, c] = GREY
    colors = [RED, BLUE, GREEN, YELLOW, ORANGE, MAGENTA]
    for i in range(rows):
        for j in range(cols):
            maj = random.choice(colors[:3])
            r0 = i * (cell + 1) + 1
            c0 = j * (cell + 1) + 1
            pts = random.sample([(r, c) for r in range(r0, r0 + cell) for c in range(c0, c0 + cell)], 5)
            for p in pts[:3]:
                g[p] = maj
            for p in pts[3:]:
                g[p] = random.choice(colors)
    return g

def main(grid):
    'Dissolve grey boundaries between regions with matching vote majorities.\n\n    Input structure: Connected grey walls partition black regions that contain temporary colored vote pixels with a unique majority in each region.\n    Rule: Clear votes, keep outer walls, and recolor any grey boundary separating two regions with the same majority to that majority color.\n    Key insight: Vote pixels program boundary behavior and are erased after their majority colors are read.\n    '
    return main_7(grid)


def generate_input():
    return generate_7()
