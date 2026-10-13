import random
from collections import Counter, deque

import numpy as np


BG = 0


def _components(mask: np.ndarray):
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
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble same-color fragments into rectangles and stack the rectangles by descending area.

    Input structure:
    - A black background contains scattered monochrome fragments in several colors.
    - All fragments of one color translate to a solid rectangle without rotation.
    - Rectangle areas are all distinct across colors.

    Rule:
    - Reconstruct each color's rectangle dimensions from its fragment set.
    - Create an output canvas with the largest rectangle size.
    - Paint rectangles from largest area to smallest, always center-aligning the next rectangle and letting it cover lower layers.

    Key insight:
    - The fragments first need to be mentally merged into full rectangles, and only then can the rectangles be ordered and nested.
    """
    g = np.array(grid, dtype=int)
    rects = []
    for color in [v for v in np.unique(g) if v != BG]:
        comps = _components(g == int(color))
        boxes = [_bbox(cells) for cells in comps]
        hs = [r1 - r0 + 1 for r0, _, r1, _ in boxes]
        ws = [c1 - c0 + 1 for _, c0, _, c1 in boxes]
        area = int(sum(len(c) for c in comps))
        if len(set(hs)) == 1:
            h, w = hs[0], sum(ws)
        else:
            h, w = sum(hs), ws[0]
        rects.append((area, h, w, int(color)))
    rects.sort(reverse=True)
    _, h0, w0, _ = rects[0]
    out = np.zeros((h0, w0), dtype=int)
    for _, h, w, color in rects:
        r0 = (h0 - h) // 2
        c0 = (w0 - w) // 2
        out[r0 : r0 + h, c0 : c0 + w] = color
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 28), random.randint(18, 28)
    g = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(3, 4))
    used_areas = set()
    for color in colors:
        for _ in range(200):
            rh, rw = random.randint(2, 6), random.randint(2, 6)
            area = rh * rw
            if area in used_areas:
                continue
            used_areas.add(area)
            vertical = random.choice([True, False])
            cuts = sorted(random.sample(range(1, (rw if vertical else rh)), random.randint(0, min(2, (rw if vertical else rh) - 1))))
            spans = [0] + cuts + [rw if vertical else rh]
            pieces = []
            for a, b in zip(spans, spans[1:]):
                if vertical:
                    pieces.append(np.ones((rh, b - a), dtype=int))
                else:
                    pieces.append(np.ones((b - a, rw), dtype=int))
            ok = True
            for piece in pieces:
                ph, pw = piece.shape
                placed = False
                for _ in range(100):
                    r0 = random.randint(0, h - ph)
                    c0 = random.randint(0, w - pw)
                    if np.any(g[max(0, r0 - 1) : min(h, r0 + ph + 1), max(0, c0 - 1) : min(w, c0 + pw + 1)]):
                        continue
                    g[r0 : r0 + ph, c0 : c0 + pw] = color
                    placed = True
                    break
                ok &= placed
            if ok:
                break
    return g
