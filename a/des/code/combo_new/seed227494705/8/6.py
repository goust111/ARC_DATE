import random
from collections import deque

import numpy as np


BG = 0
RED = 2
GREY = 5


ARROWS = {
    (0, 1): {(1, 0), (1, 1), (1, 2), (0, 2), (2, 2)},
    (0, -1): {(1, 2), (1, 1), (1, 0), (0, 0), (2, 0)},
    (1, 0): {(0, 1), (1, 1), (2, 1), (2, 0), (2, 2)},
    (-1, 0): {(2, 1), (1, 1), (0, 1), (0, 0), (0, 2)},
}


def _components(mask):
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


def _arrow_dir(pts):
    rows = [p[0] for p in pts]
    cols = [p[1] for p in pts]
    norm = {(r - min(rows), c - min(cols)) for r, c in pts}
    for direction, shape in ARROWS.items():
        if norm == shape:
            return direction
    return None


def _bbox(pts):
    rows = [p[0] for p in pts]
    cols = [p[1] for p in pts]
    return min(rows), min(cols), max(rows), max(cols)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project arrow-marked sprites through matching frame gaps into the frame crop.

    Input structure:
    - Black background with one grey rectangular frame that has one gap on each side.
    - Multicolor sprites sit outside the frame, each paired with a red arrow.
    - Arrow direction points along the sprite's travel direction toward the frame.

    Rule:
    - Match right/down/left/up arrows to the left/top/right/bottom frame gap respectively.
    - Crop the output to the frame's outer bounding box and keep grey frame pixels and gaps.
    - Place each sprite inside the frame with its center aligned to the matching gap and its far side tight to the opposite inner wall.

    Key insight:
    - The arrow gives the entry side, while the gap coordinate gives the placement row or column.
    """
    g = np.asarray(grid, dtype=int)
    grey = np.argwhere(g == GREY)
    r0, c0 = grey.min(axis=0)
    r1, c1 = grey.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    out = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    out[(out != GREY) & (out != BG)] = BG

    gaps = {
        (0, 1): ("left", [r for r in range(r0 + 1, r1) if int(g[r, c0]) != GREY][0]),
        (0, -1): ("right", [r for r in range(r0 + 1, r1) if int(g[r, c1]) != GREY][0]),
        (1, 0): ("top", [c for c in range(c0 + 1, c1) if int(g[r0, c]) != GREY][0]),
        (-1, 0): ("bottom", [c for c in range(c0 + 1, c1) if int(g[r1, c]) != GREY][0]),
    }

    sprite_components = []
    sprite_mask = (g != BG) & (g != GREY) & (g != RED)
    for pts in _components(sprite_mask):
        sprite_components.append(pts)
    used = set()
    for arrow in _components(g == RED):
        direction = _arrow_dir(arrow)
        if direction is None or direction not in gaps:
            continue
        ar0, ac0, ar1, ac1 = _bbox(arrow)
        acenter = ((ar0 + ar1) / 2, (ac0 + ac1) / 2)
        best = None
        for idx, pts in enumerate(sprite_components):
            if idx in used:
                continue
            sr0, sc0, sr1, sc1 = _bbox(pts)
            scenter = ((sr0 + sr1) / 2, (sc0 + sc1) / 2)
            dist = abs(acenter[0] - scenter[0]) + abs(acenter[1] - scenter[1])
            if best is None or dist < best[0]:
                best = (dist, idx, (sr0, sc0, sr1, sc1))
        if best is None:
            continue
        used.add(best[1])
        sr0, sc0, sr1, sc1 = best[2]
        patch = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
        ph, pw = patch.shape
        side, pos = gaps[direction]
        if side == "left":
            tr = pos - ph // 2
            tc = c1 - pw
        elif side == "right":
            tr = pos - ph // 2
            tc = c0 + 1
        elif side == "top":
            tr = r1 - ph
            tc = pos - pw // 2
        else:
            tr = r0 + 1
            tc = pos - pw // 2
        for rr, cc in np.argwhere((patch != BG) & (patch != RED) & (patch != GREY)):
            orow = tr + int(rr) - r0
            ocol = tc + int(cc) - c0
            if 0 <= orow < out.shape[0] and 0 <= ocol < out.shape[1]:
                out[orow, ocol] = int(patch[rr, cc])
    return out


def _draw_arrow(g, top, left, direction):
    for r, c in ARROWS[direction]:
        g[top + r, left + c] = RED


def _random_sprite(rng, colors):
    h = rng.randint(3, 4)
    w = rng.randint(3, 4)
    patch = np.zeros((h, w), dtype=int)
    chosen = rng.sample(colors, rng.randint(2, min(3, len(colors))))
    for r in range(h):
        for c in range(w):
            if rng.random() < 0.65 or (r, c) == (h // 2, w // 2):
                patch[r, c] = rng.choice(chosen)
    patch[h // 2, w // 2] = chosen[0]
    return patch


def generate_input() -> np.ndarray:
    rng = random.Random()
    colors = [1, 3, 4, 6, 7, 8, 9]
    for _ in range(300):
        h = w = 30
        g = np.zeros((h, w), dtype=int)
        r0, c0 = 9, 9
        fh = rng.randint(12, 15)
        fw = rng.randint(12, 15)
        r1, c1 = r0 + fh - 1, c0 + fw - 1
        g[r0, c0 : c1 + 1] = GREY
        g[r1, c0 : c1 + 1] = GREY
        g[r0 : r1 + 1, c0] = GREY
        g[r0 : r1 + 1, c1] = GREY
        gaps = {
            "left": rng.randint(r0 + 3, r1 - 3),
            "right": rng.randint(r0 + 3, r1 - 3),
            "top": rng.randint(c0 + 3, c1 - 3),
            "bottom": rng.randint(c0 + 3, c1 - 3),
        }
        g[gaps["left"], c0] = BG
        g[gaps["right"], c1] = BG
        g[r0, gaps["top"]] = BG
        g[r1, gaps["bottom"]] = BG
        directions = rng.sample(list(ARROWS), rng.randint(2, 4))
        ok = True
        occupied = set(zip(*np.where(g != BG)))
        for direction in directions:
            patch = _random_sprite(rng, colors)
            ph, pw = patch.shape
            if direction == (0, 1):
                sr, sc = gaps["left"] - ph // 2, c0 - pw - 5
                ar, ac = sr, sc + pw + 1
            elif direction == (0, -1):
                sr, sc = gaps["right"] - ph // 2, c1 + 5
                ar, ac = sr, sc - 4
            elif direction == (1, 0):
                sr, sc = r0 - ph - 5, gaps["top"] - pw // 2
                ar, ac = sr + ph + 1, sc
            else:
                sr, sc = r1 + 5, gaps["bottom"] - pw // 2
                ar, ac = sr - 4, sc
            footprint = {(sr + r, sc + c) for r, c in np.argwhere(patch != BG)}
            footprint |= {(ar + r, ac + c) for r, c in ARROWS[direction]}
            if any(not (0 <= r < h and 0 <= c < w) for r, c in footprint) or footprint & occupied:
                ok = False
                break
            occupied |= footprint
            for rr, cc in np.argwhere(patch != BG):
                g[sr + int(rr), sc + int(cc)] = int(patch[rr, cc])
            _draw_arrow(g, ar, ac, direction)
        if ok:
            return g
    raise RuntimeError("generate_input failed")
