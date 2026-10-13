from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
PINK = 6
ORANGE = 7
CYAN = 8
MAROON = 9


def n4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def comps(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in n4(rr, cc, h, w):
                    if not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def split_by_full_color_lines(g: np.ndarray, color: int = GRAY) -> list[tuple[slice, slice]]:
    row_lines = [r for r in range(g.shape[0]) if np.all(g[r, :] == color)]
    col_lines = [c for c in range(g.shape[1]) if np.all(g[:, c] == color)]
    row_starts = [0] + [r + 1 for r in row_lines]
    row_ends = row_lines + [g.shape[0]]
    col_starts = [0] + [c + 1 for c in col_lines]
    col_ends = col_lines + [g.shape[1]]
    cells = []
    for rs, re in zip(row_starts, row_ends):
        if re <= rs:
            continue
        for cs, ce in zip(col_starts, col_ends):
            if ce > cs:
                cells.append((slice(rs, re), slice(cs, ce)))
    return cells


# Puzzle 1 -----------------------------------------------------------------


def main1(grid: np.ndarray) -> np.ndarray:
    """
    From each gray head adjacent to a colored sprite, shoot a ray away from the
    sprite centroid until a green boundary, then erase the original sprite.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    body_mask = (g != BG) & (g != GREEN) & (g != GRAY)
    for body in comps(body_mask):
        color = int(g[body[0]])
        head = None
        for r, c in body:
            for nr, nc in n4(r, c, *g.shape):
                if g[nr, nc] == GRAY:
                    head = (nr, nc)
                    break
            if head is not None:
                break
        if head is None:
            continue
        cr = sum(r for r, _ in body) / len(body)
        cc = sum(c for _, c in body) / len(body)
        hr, hc = head
        dr_abs, dc_abs = abs(hr - cr), abs(hc - cc)
        if dr_abs >= dc_abs:
            dr, dc = (-1, 0) if hr < cr else (1, 0)
        else:
            dr, dc = (0, -1) if hc < cc else (0, 1)
        rr, cc2 = hr + dr, hc + dc
        while 0 <= rr < g.shape[0] and 0 <= cc2 < g.shape[1] and g[rr, cc2] != GREEN:
            if out[rr, cc2] == BG:
                out[rr, cc2] = color
            rr += dr
            cc2 += dc
        for r, c in body:
            out[r, c] = BG
        out[hr, hc] = BG
    return out


def gen1() -> np.ndarray:
    g = np.zeros((12, 12), dtype=int)
    boxes = [(0, 0, 5, 5), (0, 6, 5, 11), (6, 0, 11, 5), (6, 6, 11, 11)]
    colors = random.sample([BLUE, RED, YELLOW, PINK], 4)
    dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    random.shuffle(dirs)
    for (r0, c0, r1, c1), color, (dr, dc) in zip(boxes, colors, dirs):
        g[r0, c0 : c1 + 1] = GREEN
        g[r1, c0 : c1 + 1] = GREEN
        g[r0 : r1 + 1, c0] = GREEN
        g[r0 : r1 + 1, c1] = GREEN
        hr = (r0 + r1) // 2 + dr
        hc = (c0 + c1) // 2 + dc
        body = [(hr - dr, hc - dc), (hr - dr - dc, hc - dc - dr), (hr - dr + dc, hc - dc + dr)]
        for r, c in body:
            if r0 < r < r1 and c0 < c < c1:
                g[r, c] = color
        g[hr, hc] = GRAY
    return g


# Puzzle 2 -----------------------------------------------------------------


EDGE_DIR = {"bottom": (1, 0), "top": (-1, 0), "right": (0, 1), "left": (0, -1)}


def _edge_direction(g: np.ndarray) -> tuple[int, int]:
    if np.all(g[-1, :] == ORANGE):
        return EDGE_DIR["bottom"]
    if np.all(g[0, :] == ORANGE):
        return EDGE_DIR["top"]
    if np.all(g[:, -1] == ORANGE):
        return EDGE_DIR["right"]
    if np.all(g[:, 0] == ORANGE):
        return EDGE_DIR["left"]
    raise AssertionError("missing orange edge line")


def _dist_to_edge_for_comp(comp: list[tuple[int, int]], shape: tuple[int, int], dr: int, dc: int) -> int:
    h, w = shape
    if dr == 1:
        return h - 1 - max(r for r, _ in comp)
    if dr == -1:
        return min(r for r, _ in comp)
    if dc == 1:
        return w - 1 - max(c for _, c in comp)
    return min(c for _, c in comp)


def main2(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    dr, dc = _edge_direction(g)
    out = g.copy()
    fixed = g == ORANGE
    objects = []
    for comp in comps((g != BG) & (g != ORANGE)):
        color = int(g[comp[0]])
        objects.append((comp, color, _dist_to_edge_for_comp(comp, g.shape, dr, dc)))
    for comp, color, _dist in sorted(objects, key=lambda x: x[2]):
        for r, c in comp:
            out[r, c] = BG
            fixed[r, c] = False
        offset = 0
        while True:
            trial = [(r + (offset + 1) * dr, c + (offset + 1) * dc) for r, c in comp]
            if any(r < 0 or c < 0 or r >= g.shape[0] or c >= g.shape[1] or fixed[r, c] for r, c in trial):
                break
            offset += 1
        placed = [(r + offset * dr, c + offset * dc) for r, c in comp]
        for r, c in placed:
            out[r, c] = color
            fixed[r, c] = True
    return out


def gen2() -> np.ndarray:
    g = np.zeros((15, 15), dtype=int)
    side = random.choice(["bottom", "top", "left", "right"])
    if side == "bottom":
        g[-1, :] = ORANGE
        starts = [(2, 2), (4, 6), (1, 10), (7, 12)]
    elif side == "top":
        g[0, :] = ORANGE
        starts = [(11, 2), (8, 6), (12, 10), (6, 12)]
    elif side == "left":
        g[:, 0] = ORANGE
        starts = [(2, 11), (5, 8), (9, 12), (12, 6)]
    else:
        g[:, -1] = ORANGE
        starts = [(2, 2), (5, 5), (9, 3), (12, 8)]
    shapes = [[(0, 0), (1, 0), (1, 1)], [(0, 0), (0, 1), (1, 1)], [(0, 0), (1, 0), (2, 0)], [(0, 0), (0, 1), (0, 2), (1, 1)]]
    for color, (r0, c0), shape in zip(random.sample([BLUE, RED, GREEN, YELLOW, PINK], 4), starts, shapes):
        for r, c in shape:
            g[r0 + r, c0 + c] = color
    return g


# Puzzle 3 -----------------------------------------------------------------


def _slots(g: np.ndarray) -> list[tuple[slice, slice]]:
    cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == GRAY)]
    starts = [0] + [c + 1 for c in cols]
    ends = cols + [g.shape[1]]
    return [(slice(0, g.shape[0]), slice(s, e)) for s, e in zip(starts, ends) if e > s]


def main3(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    slots = _slots(g)
    src_r, src_c = slots[0]
    template = g[src_r, src_c] == MAROON
    coords = np.argwhere(template)
    center = np.array([(template.shape[0] - 1) / 2, (template.shape[1] - 1) / 2])
    offsets = coords - center
    for sr, sc in slots[1:]:
        patch = out[sr, sc]
        markers = [(r, c, int(patch[r, c])) for r, c in zip(*np.where((patch != BG) & (patch != GRAY)))]
        if not markers:
            continue
        mr, mc, color = markers[0]
        h, w = patch.shape
        top = mr < h / 2
        left = mc < w / 2
        k = 0 if top and left else 1 if top and not left else 2 if (not top and not left) else 3
        patch[patch == color] = BG
        rot = np.rot90(template.astype(int), -k).astype(bool)
        rr, cc = np.where(rot)
        for r, c in zip(rr, cc):
            patch[r, c] = color
    return out


def gen3() -> np.ndarray:
    h, sw, n = 5, 5, random.randint(4, 5)
    g = np.zeros((h, n * sw + n - 1), dtype=int)
    for i in range(1, n):
        g[:, i * sw + i - 1] = GRAY
    tmpl = np.array([[0, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=bool)
    g[1:4, 1:4][tmpl] = MAROON
    quads = [(0, 0), (0, 4), (4, 4), (4, 0)]
    for i, color in enumerate(random.sample([BLUE, RED, GREEN, YELLOW, PINK, ORANGE], n - 1)):
        c0 = (i + 1) * (sw + 1)
        r, c = random.choice(quads)
        g[r, c0 + c] = color
    return g


# Puzzle 4 -----------------------------------------------------------------


def main4(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    cells = split_by_full_color_lines(g, GRAY)
    n = int(len(cells) ** 0.5)
    patches = [g[rs, cs] for rs, cs in cells]
    colors = sorted({int(v) for p in patches for v in np.unique(p) if int(v) not in (BG, GRAY)})
    known = [[None for _ in range(n)] for _ in range(n)]
    for idx, p in enumerate(patches):
        vals = [int(v) for v in np.unique(p) if int(v) not in (BG, GRAY)]
        if vals:
            known[idx // n][idx % n] = vals[0]
    out = g.copy()
    changed = True
    while changed:
        changed = False
        for r in range(n):
            for c in range(n):
                if known[r][c] is not None:
                    continue
                missing_r = set(colors) - {x for x in known[r] if x is not None}
                missing_c = set(colors) - {known[rr][c] for rr in range(n) if known[rr][c] is not None}
                cand = missing_r & missing_c
                if len(cand) == 1:
                    known[r][c] = next(iter(cand))
                    rs, cs = cells[r * n + c]
                    out[rs, cs] = known[r][c]
                    changed = True
    return out


def gen4() -> np.ndarray:
    n = random.choice([3, 4])
    cell = 3
    size = n * cell + n - 1
    g = np.zeros((size, size), dtype=int)
    for k in range(1, n):
        g[k * cell + k - 1, :] = GRAY
        g[:, k * cell + k - 1] = GRAY
    colors = random.sample([BLUE, RED, GREEN, YELLOW, PINK, CYAN], n)
    latin = [[colors[(r + c) % n] for c in range(n)] for r in range(n)]
    perm = list(range(n))
    random.shuffle(perm)
    blanks = {(r, perm[r]) for r in range(n)}
    for r in range(n):
        for c in range(n):
            if (r, c) in blanks:
                continue
            r0, c0 = r * (cell + 1), c * (cell + 1)
            g[r0 : r0 + cell, c0 : c0 + cell] = latin[r][c]
    return g


# Puzzle 5 -----------------------------------------------------------------


def main5(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    sep = g.shape[1] // 2
    out[:, sep] = GRAY
    right_colors = [v for v in np.unique(g[:, sep + 1 :]) if v not in (BG, GRAY)]
    for comp in comps((g[:, :sep] != BG) & (g[:, :sep] != GRAY)):
        abs_comp = [(r, c) for r, c in comp]
        mirrored = [(r, g.shape[1] - 1 - c) for r, c in abs_comp]
        landing = [int(g[r, c]) for r, c in mirrored if int(g[r, c]) in right_colors]
        if not landing:
            continue
        color = Counter(landing).most_common(1)[0][0]
        for r, c in mirrored:
            out[r, c] = color
    return out


def gen5() -> np.ndarray:
    g = np.zeros((15, 14), dtype=int)
    sep = 7
    g[:, sep] = GRAY
    panels = [(2, 8, 5, 11), (7, 9, 10, 12), (10, 8, 13, 10)]
    for color, (r0, c0, r1, c1) in zip(random.sample([BLUE, RED, GREEN, YELLOW], 3), panels):
        g[r0 : r1 + 1, c0 : c1 + 1] = color
        shape = [(0, 0), (1, 0), (1, 1)]
        mr0, mc0 = r0 + 1, g.shape[1] - 1 - (c0 + 2)
        for r, c in shape:
            g[mr0 + r, mc0 + c] = random.choice([PINK, ORANGE, MAROON])
    return g


# Puzzle 6 -----------------------------------------------------------------


def _is_frame(g: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    return (
        r1 - r0 >= 2
        and c1 - c0 >= 2
        and np.all(g[r0, c0 : c1 + 1] == GRAY)
        and np.all(g[r1, c0 : c1 + 1] == GRAY)
        and np.all(g[r0 : r1 + 1, c0] == GRAY)
        and np.all(g[r0 : r1 + 1, c1] == GRAY)
    )


def main6(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    frames = []
    for comp in comps(g == GRAY):
        r0, c0, r1, c1 = bbox(comp)
        perimeter = 2 * (r1 - r0 + 1) + 2 * (c1 - c0 + 1) - 4
        if len(comp) == perimeter and _is_frame(g, r0, c0, r1, c1):
            frames.append((r0, c0, r1, c1))
    keep = sorted(frames, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    colors = [YELLOW, BLUE, RED, GREEN, PINK]
    out = g.copy()
    for i, f in enumerate(keep):
        r0, c0, r1, c1 = f
        inner = keep[i + 1] if i + 1 < len(keep) else None
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if inner and inner[0] <= r <= inner[2] and inner[1] <= c <= inner[3]:
                    continue
                if out[r, c] == BG:
                    out[r, c] = colors[i]
    if keep:
        r0, c0, r1, c1 = keep[-1]
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if out[r, c] == BG:
                    out[r, c] = colors[len(keep)]
    return out


def gen6() -> np.ndarray:
    g = np.zeros((15, 15), dtype=int)
    margin = random.choice([0, 1, 2])
    layers = random.choice([2, 3, 4])
    frames = []
    for i in range(layers):
        r0 = margin + 2 * i
        c0 = margin + 2 * i
        r1 = 14 - margin - 2 * i
        c1 = 14 - margin - 2 * i
        if r1 - r0 >= 2 and c1 - c0 >= 2:
            frames.append((r0, c0, r1, c1))
    for r0, c0, r1, c1 in frames:
        g[r0, c0 : c1 + 1] = GRAY
        g[r1, c0 : c1 + 1] = GRAY
        g[r0 : r1 + 1, c0] = GRAY
        g[r0 : r1 + 1, c1] = GRAY
    if random.random() < 0.7:
        # Incomplete decoy outside the nested structure.
        g[0, 0:4] = GRAY
        g[0:4, 0] = GRAY
    for _ in range(random.randint(1, 3)):
        r = random.randrange(15)
        c = random.randrange(15)
        if g[r, c] == BG:
            g[r, c] = GRAY
    return g


# Puzzle 7 -----------------------------------------------------------------


def _split_cells_1d(g: np.ndarray) -> list[np.ndarray]:
    cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == GRAY)]
    starts = [0] + [c + 1 for c in cols]
    ends = cols + [g.shape[1]]
    return [g[:, s:e] for s, e in zip(starts, ends) if e > s]


def main7(grid: np.ndarray) -> np.ndarray:
    cells = _split_cells_1d(np.asarray(grid, dtype=int))
    colors = {int(v) for cell in cells for v in np.unique(cell) if int(v) not in (BG, GRAY)}
    k = len(colors)
    chosen = cells[::k]
    return np.hstack(chosen)


def gen7() -> np.ndarray:
    n = 7
    cellw = 3
    g = np.zeros((3, n * cellw + n - 1), dtype=int)
    palette = random.sample([BLUE, RED, GREEN, YELLOW, PINK], random.randint(2, 4))
    for i in range(1, n):
        g[:, i * cellw + i - 1] = GRAY
    for i in range(n):
        c0 = i * (cellw + 1)
        color = random.choice(palette)
        g[random.randrange(3), c0 + random.randrange(3)] = color
    for i, color in enumerate(palette):
        g[i % 3, (i % n) * (cellw + 1)] = color
    return g


# Puzzle 8 -----------------------------------------------------------------


def main8(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    marks = [tuple(x) for x in np.argwhere(g == ORANGE)]
    assert len(marks) == 2
    (r1, c1), (r2, c2) = marks
    if r1 == r2:
        mid = g.shape[1] // 2
        crop = g[:, :mid].copy() if c1 < mid and c2 < mid else g[:, mid:].copy()
        second = not (c1 < mid and c2 < mid)
    else:
        mid = g.shape[0] // 2
        crop = g[:mid, :].copy() if r1 < mid and r2 < mid else g[mid:, :].copy()
        second = not (r1 < mid and r2 < mid)
    crop[crop == ORANGE] = BG
    return np.rot90(crop, -1) if second else crop


def gen8() -> np.ndarray:
    g = np.zeros((12, 12), dtype=int)
    for _ in range(18):
        g[random.randrange(12), random.randrange(12)] = random.choice([BLUE, RED, GREEN, YELLOW, PINK, CYAN])
    mode = random.choice(["left", "right", "top", "bottom"])
    if mode in ("left", "right"):
        cbase = random.randrange(0, 5) if mode == "left" else random.randrange(6, 11)
        r = random.randrange(1, 11)
        g[r, cbase] = ORANGE
        g[r, min(11, cbase + 1)] = ORANGE
    else:
        rbase = random.randrange(0, 5) if mode == "top" else random.randrange(6, 11)
        c = random.randrange(1, 11)
        g[rbase, c] = ORANGE
        g[min(11, rbase + 1), c] = ORANGE
    return g


# Puzzle 9 -----------------------------------------------------------------


def _resize_nearest(patch: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = patch.shape
    oh, ow = shape
    out = np.zeros(shape, dtype=int)
    for r in range(oh):
        for c in range(ow):
            out[r, c] = patch[int(r * h / oh), int(c * w / ow)]
    return out


def main9(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    coords = np.argwhere(g == GRAY)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    inside = g[r0 + 1 : r1, c0 + 1 : c1]
    fg = np.argwhere(inside != BG)
    rr0, cc0 = fg.min(axis=0)
    rr1, cc1 = fg.max(axis=0)
    norm = _resize_nearest(inside[rr0 : rr1 + 1, cc0 : cc1 + 1], (3, 3))
    k = len({int(v) for v in np.unique(norm) if int(v) != BG})
    return np.tile(norm, (k, k))


def gen9() -> np.ndarray:
    size = random.choice([7, 9, 11])
    g = np.zeros((size, size), dtype=int)
    g[0, 1:-1] = GRAY
    g[-1, 1:-1] = GRAY
    g[1:-1, 0] = GRAY
    g[1:-1, -1] = GRAY
    colors = random.sample([BLUE, RED, GREEN, YELLOW], random.randint(2, 4))
    for i, color in enumerate(colors):
        r = 1 + (i % 3)
        c = 1 + ((i * 2) % max(1, size - 2))
        g[r, min(size - 2, c)] = color
    return g


# Puzzle 10 ----------------------------------------------------------------


RANK_COLORS = [BLUE, GREEN, YELLOW, PINK, ORANGE, MAROON]


def main10(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    regions = comps(g == BG)
    regions = sorted(regions, key=len)
    out = g.copy()
    for i, region in enumerate(regions):
        color = RANK_COLORS[min(i, len(RANK_COLORS) - 1)]
        for r, c in region:
            out[r, c] = color
    return out


def gen10() -> np.ndarray:
    g = np.zeros((12, 12), dtype=int)
    rows = sorted(random.sample(range(2, 10), 2))
    cols = sorted(random.sample(range(2, 10), 2))
    g[rows[0], :] = RED
    g[rows[1], :] = RED
    g[:, cols[0]] = RED
    g[:, cols[1]] = RED
    return g


# Puzzle 11 ----------------------------------------------------------------


def _template_patch(g: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    comps9 = comps(g == MAROON)
    comp = max(comps9, key=len)
    r0, c0, r1, c1 = bbox(comp)
    return g[r0 : r1 + 1, c0 : c1 + 1] == MAROON, (r0, c0, r1, c1)


def main11(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    templ, _box = _template_patch(g)
    th, tw = templ.shape
    markers = [(r, c, int(g[r, c])) for r in range(g.shape[0]) for c in range(g.shape[1]) if (r in (0, g.shape[0]-1) or c in (0, g.shape[1]-1)) and int(g[r, c]) not in (BG, MAROON)]
    for r, c, color in markers:
        if c == 0:
            start = c + tw
            while start + tw <= g.shape[1]:
                top = max(0, min(g.shape[0] - th, r - th // 2))
                patch = out[top : top + th, start : start + tw]
                patch[templ] = color
                start += tw
    return out


def gen11() -> np.ndarray:
    g = np.zeros((15, 15), dtype=int)
    templ = np.array([[1, 1, 0], [0, 1, 1]], dtype=bool)
    g[0:2, 12:15][templ] = MAROON
    rows = random.sample(range(3, 13), 3)
    for row, color in zip(rows, random.sample([BLUE, RED, YELLOW], 3)):
        g[row, 0] = color
    return g


# Puzzle 12 ----------------------------------------------------------------


def main12(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.shape[0] == 2
    n = g.shape[1] // 2
    blocks = [g[:, 2 * i : 2 * i + 2] for i in range(n)]
    order = sorted(range(n), key=lambda i: len(set(map(int, blocks[i].ravel()))))
    out = np.zeros((2 * n, 2 * n), dtype=int)
    for layer, idx in enumerate(order):
        block = blocks[idx]
        lo, hi = layer, 2 * n - 1 - layer
        if lo > hi:
            break
        for r in range(lo, hi + 1):
            for c in range(lo, hi + 1):
                if r in (lo, hi) or c in (lo, hi):
                    qr = 0 if r < n else 1
                    qc = 0 if c < n else 1
                    out[r, c] = int(block[qr, qc])
    return out


def gen12() -> np.ndarray:
    n = random.choice([3, 4, 5])
    counts = list(range(1, n + 1))
    random.shuffle(counts)
    g = np.zeros((2, 2 * n), dtype=int)
    palette = [BLUE, RED, GREEN, YELLOW, PINK, ORANGE, CYAN, MAROON]
    for i, count in enumerate(counts):
        colors = random.sample(palette, min(count, 4))
        vals = (colors * 4)[:4]
        random.shuffle(vals)
        g[:, 2 * i : 2 * i + 2] = np.array(vals, dtype=int).reshape(2, 2)
    return g
