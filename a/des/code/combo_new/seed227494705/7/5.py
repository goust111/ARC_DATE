import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
YELLOW = 4


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            yield pts


def _reflect(direction, slash):
    dr, dc = direction
    if slash:
        return (-dc, -dr)
    return (dc, dr)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace each room's colored light ray through one diagonal mirror to the wall.

    Input structure:
    - Black background with grey rectangular rooms.
    - Each room has one colored source pixel replacing a wall cell.
    - Each room interior has one two-pixel yellow diagonal mirror, either slash or backslash.

    Rule:
    - The source wall side gives the initial inward ray direction.
    - Paint black cells along the ray with the source color; keep yellow mirror cells yellow.
    - Reflect once or more at yellow mirror cells according to slash/backslash geometry, then color the wall pixel hit last.

    Key insight:
    - The source position gives the starting direction and the two yellow pixels give the reflection rule.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for pts in _components(g == GREY):
        rows = [p[0] for p in pts]
        cols = [p[1] for p in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue
        border = [(r0, c) for c in range(c0, c1 + 1)] + [(r1, c) for c in range(c0, c1 + 1)]
        border += [(r, c0) for r in range(r0 + 1, r1)] + [(r, c1) for r in range(r0 + 1, r1)]
        sources = [p for p in border if int(g[p]) not in (BG, GREY, YELLOW)]
        if len(sources) != 1:
            continue
        sr, sc = sources[0]
        color = int(g[sr, sc])
        if sr == r0:
            direction = (1, 0)
        elif sr == r1:
            direction = (-1, 0)
        elif sc == c0:
            direction = (0, 1)
        else:
            direction = (0, -1)
        mirrors = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if int(g[r, c]) == YELLOW]
        if len(mirrors) != 2:
            continue
        mirrors_sorted = sorted(mirrors)
        slash = mirrors_sorted[0][1] > mirrors_sorted[1][1]
        mirror_set = set(mirrors)
        r, c = sr + direction[0], sc + direction[1]
        for _step in range(g.size):
            if not (0 <= r < g.shape[0] and 0 <= c < g.shape[1]):
                break
            if (r, c) in mirror_set:
                direction = _reflect(direction, slash)
            elif int(g[r, c]) == GREY:
                out[r, c] = color
                break
            elif int(g[r, c]) == BG:
                out[r, c] = color
            r += direction[0]
            c += direction[1]
    return out


def _border_cells(r0, c0, r1, c1):
    for c in range(c0, c1 + 1):
        yield r0, c
        yield r1, c
    for r in range(r0 + 1, r1):
        yield r, c0
        yield r, c1


def generate_input() -> np.ndarray:
    rng = random.Random()
    colors = [1, 2, 3, 6, 7, 8, 9]
    for _ in range(500):
        h = rng.randint(16, 30)
        w = rng.randint(16, 30)
        g = np.zeros((h, w), dtype=int)
        n = rng.randint(1, 3)
        occupied = set()
        ok = True
        for color in rng.sample(colors, n):
            placed = False
            for _try in range(100):
                rh = rng.randint(7, 10)
                rw = rng.randint(7, 10)
                r0 = rng.randint(1, h - rh - 1)
                c0 = rng.randint(1, w - rw - 1)
                r1, c1 = r0 + rh - 1, c0 + rw - 1
                pad = {(r, c) for r in range(r0 - 1, r1 + 2) for c in range(c0 - 1, c1 + 2)}
                if pad & occupied:
                    continue
                occupied |= pad
                side = rng.choice(["top", "bottom", "left", "right"])
                if side == "top":
                    source = (r0, rng.randint(c0 + 2, c1 - 2))
                    direction = (1, 0)
                elif side == "bottom":
                    source = (r1, rng.randint(c0 + 2, c1 - 2))
                    direction = (-1, 0)
                elif side == "left":
                    source = (rng.randint(r0 + 2, r1 - 2), c0)
                    direction = (0, 1)
                else:
                    source = (rng.randint(r0 + 2, r1 - 2), c1)
                    direction = (0, -1)
                slash = rng.choice([False, True])
                new_dir = _reflect(direction, slash)
                candidates = []
                rr, cc = source[0] + direction[0], source[1] + direction[1]
                while r0 < rr < r1 and c0 < cc < c1:
                    if r0 < rr + new_dir[0] < r1 and c0 < cc + new_dir[1] < c1:
                        candidates.append((rr, cc))
                    rr += direction[0]
                    cc += direction[1]
                rng.shuffle(candidates)
                for hit in candidates:
                    poss = []
                    for off in ((1, 1), (-1, -1)) if not slash else ((1, -1), (-1, 1)):
                        other = (hit[0] + off[0], hit[1] + off[1])
                        if r0 < other[0] < r1 and c0 < other[1] < c1:
                            poss.append(other)
                    if not poss:
                        continue
                    other = rng.choice(poss)
                    for r, c in _border_cells(r0, c0, r1, c1):
                        g[r, c] = GREY
                    g[source] = color
                    g[hit] = YELLOW
                    g[other] = YELLOW
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                ok = False
                break
        if ok:
            return g
    raise RuntimeError("generate_input failed")
