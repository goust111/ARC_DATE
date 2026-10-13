import random
from collections import Counter, deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
AZURE = 8
MAROON = 9
COLORS = [BLUE, RED, GREEN, YELLOW, ORANGE, AZURE, MAROON]



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
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def _norm(cells):
    if not cells:
        return frozenset()
    r0, _, c0, _ = _bbox(cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _shape_size(shape):
    if not shape:
        return 0, 0
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _paint_cells(grid, cells, color):
    h, w = grid.shape
    for r, c in cells:
        if 0 <= r < h and 0 <= c < w:
            grid[r, c] = color


def _draw_rect(grid, r0, r1, c0, c1, color):
    h, w = grid.shape
    for r in range(max(0, r0), min(h, r1 + 1)):
        for c in range(max(0, c0), min(w, c1 + 1)):
            grid[r, c] = color


def _perimeter_cells(r0, r1, c0, c1):
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _random_shape(max_h=3, max_w=3):
    shapes = [
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 1)],
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
    ]
    candidates = [s for s in shapes if max(r for r, _ in s) < max_h and max(c for _, c in s) < max_w]
    return random.choice(candidates)


def _find_rect_holes(grid, r0, r1, c0, c1):
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    holes = []
    for comp in _components(sub == BG):
        if any(r == 0 or c == 0 or r == sub.shape[0] - 1 or c == sub.shape[1] - 1 for r, c in comp):
            continue
        holes.append([(r0 + r, c0 + c) for r, c in comp])
    return holes


def main_1(grid):
    out = np.array(grid, copy=True)
    for comp in _components(grid == GREY):
        r0, r1, c0, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        seeds = [
            (r, c, int(grid[r, c]))
            for r in range(r0 + 1, r1)
            for c in range(c0 + 1, c1)
            if grid[r, c] not in (BG, GREY)
        ]
        if len(seeds) != 1:
            continue
        sr, sc, color = seeds[0]
        for c in range(c0 + 1, c1):
            if grid[r0, c] != GREY:
                _paint_cells(out, [(r0 - k, c) for k in range(1, sr - r0 + 1)], color)
            if grid[r1, c] != GREY:
                _paint_cells(out, [(r1 + k, c) for k in range(1, r1 - sr + 1)], color)
        for r in range(r0 + 1, r1):
            if grid[r, c0] != GREY:
                _paint_cells(out, [(r, c0 - k) for k in range(1, sc - c0 + 1)], color)
            if grid[r, c1] != GREY:
                _paint_cells(out, [(r, c1 + k) for k in range(1, c1 - sc + 1)], color)
    return out


def gen_1():
    h, w = random.randint(18, 30), random.randint(18, 30)
    grid = np.zeros((h, w), dtype=int)
    for _ in range(random.randint(1, 3)):
        for _attempt in range(200):
            fh, fw = random.randint(4, 7), random.randint(4, 7)
            r0 = random.randint(2, h - fh - 3)
            c0 = random.randint(2, w - fw - 3)
            r1, c1 = r0 + fh - 1, c0 + fw - 1
            sr = random.randint(r0 + 1, r1 - 1)
            sc = random.randint(c0 + 1, c1 - 1)
            sides = random.sample(["top", "bottom", "left", "right"], random.randint(1, 3))
            gaps = set()
            for side in sides:
                if side == "top":
                    gaps.add((r0, random.randint(c0 + 1, c1 - 1)))
                elif side == "bottom":
                    gaps.add((r1, random.randint(c0 + 1, c1 - 1)))
                elif side == "left":
                    gaps.add((random.randint(r0 + 1, r1 - 1), c0))
                else:
                    gaps.add((random.randint(r0 + 1, r1 - 1), c1))
            frame = _perimeter_cells(r0, r1, c0, c1) - gaps
            rays = []
            for gr, gc in gaps:
                if gr == r0:
                    rays += [(r0 - k, gc) for k in range(1, sr - r0 + 1)]
                elif gr == r1:
                    rays += [(r1 + k, gc) for k in range(1, r1 - sr + 1)]
                elif gc == c0:
                    rays += [(gr, c0 - k) for k in range(1, sc - c0 + 1)]
                else:
                    rays += [(gr, c1 + k) for k in range(1, c1 - sc + 1)]
            all_cells = list(frame) + rays + [(sr, sc)]
            if all(0 <= r < h and 0 <= c < w and grid[r, c] == BG for r, c in all_cells):
                _paint_cells(grid, frame, GREY)
                grid[sr, sc] = random.choice([BLUE, RED, GREEN, YELLOW, ORANGE])
                break
    return grid


def main_2(grid):
    grey_rows = np.where(grid == GREY)[0]
    if len(grey_rows) == 0:
        return np.array(grid, copy=True)
    top_h = int(grey_rows.max()) + 1
    upper = np.array(grid[:top_h], copy=True)
    lower = grid[top_h:]
    fragments = []
    for comp in _components((lower != BG) & (lower != GREY)):
        color = int(lower[comp[0][0], comp[0][1]])
        abs_comp = [(top_h + r, c) for r, c in comp]
        fragments.append((_norm(abs_comp), color))
    for comp in _components(upper == GREY):
        r0, r1, c0, c1 = _bbox(comp)
        holes = _find_rect_holes(upper, r0, r1, c0, c1)
        for hole in holes:
            shape = _norm(hole)
            match = next(((s, col) for s, col in fragments if s == shape), None)
            if match is None:
                continue
            color = match[1]
            for r, c in comp:
                upper[r, c] = color
            _paint_cells(upper, hole, color)
    return upper


def gen_2():
    n = random.randint(2, 3)
    top_h = random.randint(8, 12)
    w = random.randint(22, 30)
    lower_h = random.randint(7, 11)
    grid = np.zeros((top_h + lower_h, w), dtype=int)
    shapes = random.sample([
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 1)],
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ], n)
    colors = random.sample([BLUE, RED, GREEN, YELLOW, ORANGE, AZURE, MAROON], n)
    slot_w = w // n
    used_lower = np.zeros_like(grid, dtype=bool)
    for i, shape in enumerate(shapes):
        rh, rw = random.randint(5, 6), random.randint(5, 7)
        r0 = random.randint(1, top_h - rh - 1)
        c0 = i * slot_w + random.randint(1, max(1, slot_w - rw - 1))
        _draw_rect(grid, r0, r0 + rh - 1, c0, c0 + rw - 1, GREY)
        sr = r0 + 2
        sc = c0 + 2
        for dr, dc in shape:
            grid[sr + dr, sc + dc] = BG
        for _attempt in range(100):
            fr = random.randint(top_h + 1, top_h + lower_h - 4)
            fc = random.randint(1, w - 4)
            cells = [(fr + dr, fc + dc) for dr, dc in shape]
            if all(grid[r, c] == BG and not used_lower[r, c] for r, c in cells):
                for r, c in cells:
                    grid[r, c] = colors[i]
                    used_lower[r, c] = True
                break
    return grid


def main_3(grid):
    instr = [int(x) for x in grid[0] if x != BG]
    area = grid[2:]
    if area.size == 0:
        return area.copy()
    first = area[0]
    runs = {}
    c = 0
    while c < len(first):
        color = int(first[c])
        start = c
        while c + 1 < len(first) and first[c + 1] == color:
            c += 1
        if color != BG:
            runs[color] = (start, c)
        c += 1
    h, w = area.shape
    band_h = 3 if h % 3 == 0 and h % 2 != 0 else (2 if h % 2 == 0 else 1)
    out = np.zeros_like(area)
    for br in range(0, h, band_h):
        band_index = br // band_h
        draw_order = list(reversed(instr)) if band_index % 2 == 0 else list(instr)
        for color in draw_order:
            if color not in runs:
                continue
            c0, c1 = runs[color]
            for rr in range(br, min(h, br + band_h)):
                for cc in range(max(0, c0 - 1), min(w, c1 + 2)):
                    out[rr, cc] = color
    return out


def gen_3():
    n = random.randint(3, 5)
    colors = random.sample([BLUE, RED, GREEN, YELLOW, ORANGE, AZURE, MAROON], n)
    widths = [random.randint(2, 4) for _ in range(n)]
    w = sum(widths)
    band_h = random.choice([2, 3])
    band_count = random.choice([4, 5]) if band_h == 2 else random.choice([3, 5])
    h = 2 + band_h * band_count
    grid = np.zeros((h, w), dtype=int)
    instr = colors[:]
    random.shuffle(instr)
    grid[0, :n] = instr
    stripe_order = colors[:]
    random.shuffle(stripe_order)
    c = 0
    for color, width in zip(stripe_order, widths):
        grid[2:, c : c + width] = color
        c += width
    return grid


def main_4(grid):
    out = np.array(grid, copy=True)
    rects = []
    for comp in _components(grid == GREY):
        r0, r1, c0, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        inside = grid[r0 + 1 : r1, c0 + 1 : c1]
        votes = [int(x) for x in inside.ravel() if x not in (BG, GREY)]
        if not votes:
            continue
        win = Counter(votes).most_common(1)[0][0]
        out[r0 + 1 : r1, c0 + 1 : c1] = win
        rects.append(((r1 - r0 - 1) * (c1 - c0 - 1), comp))
    border_colors = [RED, BLUE, GREEN, YELLOW, ORANGE]
    for rank, (_, comp) in enumerate(sorted(rects, key=lambda x: x[0])):
        for r, c in comp:
            out[r, c] = border_colors[rank % len(border_colors)]
    return out


def gen_4():
    h, w = random.randint(18, 30), random.randint(18, 30)
    grid = np.zeros((h, w), dtype=int)
    areas = set()
    for _ in range(random.randint(2, 5)):
        for _attempt in range(200):
            rh, rw = random.randint(4, 8), random.randint(4, 8)
            area = (rh - 2) * (rw - 2)
            if area in areas:
                continue
            r0 = random.randint(1, h - rh - 1)
            c0 = random.randint(1, w - rw - 1)
            frame = _perimeter_cells(r0, r0 + rh - 1, c0, c0 + rw - 1)
            pad = [(r, c) for r in range(r0 - 1, r0 + rh + 1) for c in range(c0 - 1, c0 + rw + 1) if 0 <= r < h and 0 <= c < w]
            if all(grid[r, c] == BG for r, c in pad):
                _paint_cells(grid, frame, GREY)
                vote_colors = random.sample([BLUE, RED, GREEN, YELLOW, ORANGE, AZURE, MAROON], random.randint(2, 3))
                winner = vote_colors[0]
                cells = [(r, c) for r in range(r0 + 1, r0 + rh - 1) for c in range(c0 + 1, c0 + rw - 1)]
                random.shuffle(cells)
                count = random.randint(3, min(7, len(cells)))
                vals = [winner] * (count // 2 + 1) + [random.choice(vote_colors[1:]) for _ in range(count - (count // 2 + 1))]
                random.shuffle(vals)
                for (r, c), color in zip(cells, vals):
                    grid[r, c] = color
                areas.add(area)
                break
    return grid


def main_5(grid):
    out = np.array(grid, copy=True)
    h_lines = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == AZURE)]
    v_lines = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == AZURE)]
    r_edges = [-1] + h_lines + [grid.shape[0]]
    c_edges = [-1] + v_lines + [grid.shape[1]]
    cells = []
    for i in range(len(r_edges) - 1):
        for j in range(len(c_edges) - 1):
            r0, r1 = r_edges[i] + 1, r_edges[i + 1] - 1
            c0, c1 = c_edges[j] + 1, c_edges[j + 1] - 1
            if r0 <= r1 and c0 <= c1:
                cells.append((r0, r1, c0, c1))
    if not cells:
        return out
    ch = cells[0][1] - cells[0][0] + 1
    cw = cells[0][3] - cells[0][2] + 1
    master = np.zeros((ch, cw), dtype=int)
    for r0, r1, c0, c1 in cells:
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        mask = (sub != BG) & (sub != AZURE)
        master[mask] = sub[mask]
    for r0, r1, c0, c1 in cells:
        out[r0 : r1 + 1, c0 : c1 + 1] = master
    return out


def gen_5():
    rows, cols = random.choice([(2, 2), (2, 3), (3, 2), (3, 3)])
    ch, cw = random.randint(4, 7), random.randint(4, 7)
    h = rows * ch + rows - 1
    w = cols * cw + cols - 1
    grid = np.zeros((h, w), dtype=int)
    for r in range(ch, h, ch + 1):
        grid[r, :] = AZURE
    for c in range(cw, w, cw + 1):
        grid[:, c] = AZURE
    coords = random.sample([(r, c) for r in range(ch) for c in range(cw)], random.randint(5, min(12, ch * cw)))
    colors = [BLUE, RED, GREEN, YELLOW, ORANGE, MAROON]
    cell_indices = [(i, j) for i in range(rows) for j in range(cols)]
    random.shuffle(cell_indices)
    for k, (rr, cc) in enumerate(coords):
        i, j = cell_indices[k % len(cell_indices)]
        r0 = i * (ch + 1)
        c0 = j * (cw + 1)
        grid[r0 + rr, c0 + cc] = random.choice(colors)
    return grid


def main_6(grid):
    out = np.array(grid, copy=True)
    for comp in _components(grid != BG):
        vals = [int(grid[r, c]) for r, c in comp]
        body_vals = [v for v in vals if v != MAROON]
        if not body_vals or MAROON not in vals:
            continue
        color = Counter(body_vals).most_common(1)[0][0]
        r0, r1, c0, c1 = _bbox(comp)
        height, width = r1 - r0 + 1, c1 - c0 + 1
        if height == width:
            continue
        k = vals.count(MAROON)
        short = min(height, width)
        if width > height:
            new_h, new_w = short, k * short
        else:
            new_h, new_w = k * short, short
        top = (r0 + r1 - new_h + 1) // 2
        left = (c0 + c1 - new_w + 1) // 2
        for r, c in comp:
            out[r, c] = BG
        _draw_rect(out, top, top + new_h - 1, left, left + new_w - 1, color)
    return out


def gen_6():
    h, w = random.randint(18, 30), random.randint(18, 30)
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample([BLUE, RED, GREEN, YELLOW, ORANGE, AZURE], random.randint(2, 4))
    for color in colors:
        for _attempt in range(200):
            if random.random() < 0.5:
                rh, rw = random.randint(2, 3), random.randint(4, 6)
            else:
                rh, rw = random.randint(4, 6), random.randint(2, 3)
            r0 = random.randint(2, h - rh - 3)
            c0 = random.randint(2, w - rw - 3)
            pad = [(r, c) for r in range(r0 - 1, r0 + rh + 1) for c in range(c0 - 1, c0 + rw + 1)]
            if all(0 <= r < h and 0 <= c < w and grid[r, c] == BG for r, c in pad):
                grid[r0 : r0 + rh, c0 : c0 + rw] = color
                inner = [(r, c) for r in range(r0, r0 + rh) for c in range(c0, c0 + rw)]
                for r, c in random.sample(inner, random.randint(1, min(5, len(inner)))):
                    grid[r, c] = MAROON
                break
    return grid


def main_7(grid):
    out = np.array(grid, copy=True)
    h, w = grid.shape
    mailbox = {}
    for c in range(w):
        if grid[0, c] not in (BG, GREY):
            mailbox[int(grid[0, c])] = ("top", 0, c)
        if grid[h - 1, c] not in (BG, GREY):
            mailbox[int(grid[h - 1, c])] = ("bottom", h - 1, c)
    for r in range(h):
        if grid[r, 0] not in (BG, GREY):
            mailbox[int(grid[r, 0])] = ("left", r, 0)
        if grid[r, w - 1] not in (BG, GREY):
            mailbox[int(grid[r, w - 1])] = ("right", r, w - 1)
    edge_cells = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w) if (r in (0, h - 1) or c in (0, w - 1)) and grid[r, c] != BG]
    for comp in _components(grid == GREY):
        r0, r1, c0, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        side_counts = {
            "top": sum(grid[r0, c] == GREY for c in range(c0, c1 + 1)),
            "bottom": sum(grid[r1, c] == GREY for c in range(c0, c1 + 1)),
            "left": sum(grid[r, c0] == GREY for r in range(r0, r1 + 1)),
            "right": sum(grid[r, c1] == GREY for r in range(r0, r1 + 1)),
        }
        open_side = min(side_counts, key=side_counts.get)
        inside = [(r, c, int(grid[r, c])) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if grid[r, c] not in (BG, GREY)]
        addr = [x for x in inside if x[2] in mailbox and mailbox[x[2]][0] == open_side]
        if len(addr) != 1:
            continue
        ar, ac, addr_color = addr[0]
        content = [(r, c, col) for r, c, col in inside if (r, c) != (ar, ac)]
        if not content:
            continue
        cr0, cr1, cc0, cc1 = _bbox([(r, c) for r, c, _ in content])
        mh_side, mr, mc = mailbox[addr_color]
        ch, cw = cr1 - cr0 + 1, cc1 - cc0 + 1
        if mh_side == "top":
            tr, tc = 0, mc - cw // 2
        elif mh_side == "bottom":
            tr, tc = h - ch, mc - cw // 2
        elif mh_side == "left":
            tr, tc = mr - ch // 2, 0
        else:
            tr, tc = mr - ch // 2, w - cw
        _draw_rect(out, r0, r1, c0, c1, BG)
        for r, c, col in content:
            nr, nc = tr + (r - cr0), tc + (c - cc0)
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = col
    for r, c, col in edge_cells:
        out[r, c] = col
    return out


def gen_7():
    h = w = 30
    grid = np.zeros((h, w), dtype=int)
    sides = ["top", "bottom", "left", "right"]
    mail_colors = random.sample([BLUE, RED, GREEN, YELLOW, ORANGE, AZURE, MAROON], random.randint(2, 4))
    mail = {}
    for color in mail_colors:
        side = random.choice(sides)
        if side == "top":
            pos = (0, random.randint(3, w - 4))
        elif side == "bottom":
            pos = (h - 1, random.randint(3, w - 4))
        elif side == "left":
            pos = (random.randint(3, h - 4), 0)
        else:
            pos = (random.randint(3, h - 4), w - 1)
        grid[pos] = color
        mail[color] = (side, pos)
    used = grid != BG
    for color, (side, _) in list(mail.items())[: random.randint(1, len(mail))]:
        for _attempt in range(200):
            rh, rw = random.randint(5, 7), random.randint(5, 7)
            r0 = random.randint(4, h - rh - 4)
            c0 = random.randint(4, w - rw - 4)
            box = [(r, c) for r in range(r0 - 1, r0 + rh + 1) for c in range(c0 - 1, c0 + rw + 1)]
            if not all(0 <= r < h and 0 <= c < w and not used[r, c] for r, c in box):
                continue
            r1, c1 = r0 + rh - 1, c0 + rw - 1
            frame = _perimeter_cells(r0, r1, c0, c1)
            if side == "top":
                frame -= {(r0, c) for c in range(c0 + 1, c1)}
            elif side == "bottom":
                frame -= {(r1, c) for c in range(c0 + 1, c1)}
            elif side == "left":
                frame -= {(r, c0) for r in range(r0 + 1, r1)}
            else:
                frame -= {(r, c1) for r in range(r0 + 1, r1)}
            _paint_cells(grid, frame, GREY)
            ar, ac = random.randint(r0 + 1, r1 - 1), random.randint(c0 + 1, c1 - 1)
            grid[ar, ac] = color
            content_color = random.choice([x for x in COLORS if x not in (color,)])
            shape = _random_shape(3, 3)
            sr, sc = r0 + 1, c0 + 1
            for dr, dc in shape:
                if (sr + dr, sc + dc) != (ar, ac):
                    grid[sr + dr, sc + dc] = content_color
            for r, c in box:
                used[r, c] = True
            break
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    "Unroll each marked colored object into a centered rectangular strip.\n\nInput structure:\nBlack background contains colored connected objects with non-square bounding\nboxes. Each object includes one or more maroon marker pixels embedded in its\nbody.\nRule:\nThe object's longer bounding-box axis gives the strip direction. The maroon\nmarker count K makes the long dimension K times the short dimension, while\nthe short dimension and body color are preserved. The original object is\nreplaced by the clipped centered strip.\nKey insight:\nShape orientation gives direction, and marker count gives scale."
    return main_6(grid)


def generate_input() -> np.ndarray:
    return gen_6()
